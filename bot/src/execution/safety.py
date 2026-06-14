"""Milestone-0 safety rails for the vGen OKX volume farmer.

Pure, unit-testable helpers plus a file-backed kill switch. NONE of this places
or sizes an order — it only validates config, checks leverage/liquidation
geometry, and gates entries. It is deliberately separate from the trading code
so the guards can be tested in isolation and reviewed at a glance.

Design rule (real money): a guard must BLOCK the genuinely-unsafe case while
NEVER spuriously blocking the current, known-good live config — a false block at
startup halts the live campaign, the worst outcome. Every bound here is chosen to
pass the shipped ``config_volume_farmer_okx_v3.yaml`` with headroom; see the
measured ATR margins in the tests.
"""
from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

# OKX isolated-margin maintenance buffer used to approximate the liquidation
# line at  liq_distance ≈ (1/leverage − mmr).  Matches the backtest harness.
DEFAULT_MMR = 0.005

# ---------------------------------------------------------------------------
# timeframe / fee math (mirrors volume_farmer + the runner so the validator
# reasons about the SAME numbers the live bot will use)
# ---------------------------------------------------------------------------


def tf_seconds(tf: str) -> int:
    """Seconds per bar for an OKX timeframe like '5m'. Raises on garbage so a
    bad timeframe can never silently become a 0-second cooldown multiply."""
    digits = "".join(ch for ch in tf if ch.isdigit())
    unit = "".join(ch for ch in tf if ch.isalpha()).lower()
    if not digits or unit not in {"s", "m", "h", "d"}:
        raise ValueError(f"unparseable timeframe {tf!r}")
    secs = int(digits) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    if secs <= 0:
        raise ValueError(f"non-positive timeframe seconds from {tf!r}")
    return secs


def round_trip_fee_bps(cfg: Dict[str, Any]) -> float:
    """Net round-trip fee (bps) under the configured rebate, mirroring
    VolumeFarmerSession._round_trip_fee_bps. Open is always maker (post_only);
    close is maker when limit_tp / resting_tp is on, else taker."""
    fees = cfg.get("fees", {}) or {}
    farmer = cfg.get("farmer", {}) or {}
    maker = float(fees.get("maker", 0.0))
    taker = float(fees.get("taker", 0.0))
    rebate = float(fees.get("rebate_pct", 0.0))
    rebate_all = bool(fees.get("rebate_all_legs", False))
    maker_eff = maker * (1.0 - rebate)
    taker_eff = taker * (1.0 - rebate) if rebate_all else taker
    close_maker = bool(farmer.get("limit_tp", False)) or bool(
        (farmer.get("resting_tp", {}) or {}).get("enabled", False))
    close_leg = maker_eff if close_maker else taker_eff
    return (maker_eff + close_leg) * 10_000.0


def fee_floor_bps(cfg: Dict[str, Any]) -> float:
    """The fee-aware TP floor: tp_fee_cover_mult*round_trip + tp_profit_buffer."""
    atr = (cfg.get("farmer", {}) or {}).get("atr", {}) or {}
    return (float(atr.get("tp_fee_cover_mult", 0.0)) * round_trip_fee_bps(cfg)
            + float(atr.get("tp_profit_buffer_bps", 0.0)))


def liq_distance_bps(max_leverage: float, mmr: float = DEFAULT_MMR) -> float:
    """Approximate isolated-margin liquidation distance from entry, in bps."""
    if max_leverage <= 0:
        return 0.0
    return max(1.0 / max_leverage - mmr, 1e-6) * 10_000.0


def atr_bps_median(history, period: int = 14) -> float:
    """Median Wilder-ATR over a candle DataFrame, expressed in bps of close.

    Robust to single volatility spikes (uses the median, not the max) so the
    liquidation check reflects TYPICAL conditions, not a one-bar outlier.
    Returns 0.0 if there isn't enough data to compute it.
    """
    try:
        import numpy as np  # local import — keep this module dependency-light
        h = history["high"].astype(float).to_numpy()
        l = history["low"].astype(float).to_numpy()
        c = history["close"].astype(float).to_numpy()
    except Exception:  # noqa: BLE001
        return 0.0
    n = len(c)
    if n < period + 2:
        return 0.0
    prev_c = c[:-1]
    tr = np.maximum.reduce([
        (h[1:] - l[1:]),
        np.abs(h[1:] - prev_c),
        np.abs(l[1:] - prev_c),
    ])
    # Wilder smoothing via EMA(alpha=1/period); take median of the bps series.
    atr = np.empty_like(tr)
    a = 1.0 / period
    acc = tr[:period].mean()
    for i, x in enumerate(tr):
        acc = acc + a * (x - acc)
        atr[i] = acc
    bps = atr / c[1:] * 10_000.0
    bps = bps[np.isfinite(bps)]
    if bps.size == 0:
        return 0.0
    return float(np.median(bps))


# ---------------------------------------------------------------------------
# config schema validation
# ---------------------------------------------------------------------------

def _num(d: Dict[str, Any], key: str, errs: List[str], where: str,
         lo: Optional[float] = None, hi: Optional[float] = None,
         required: bool = True) -> Optional[float]:
    """Fetch a numeric config value with presence/type/bounds checking. A missing
    REQUIRED key is itself the typo guard: e.g. 'margin_fration_per_trade'
    leaves 'margin_fraction_per_trade' absent → a hard error, not a silent
    default."""
    if key not in d:
        if required:
            errs.append(f"{where}.{key} is MISSING (typo? required for safety)")
        return None
    v = d[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        errs.append(f"{where}.{key} must be a number, got {type(v).__name__} {v!r}")
        return None
    v = float(v)
    if lo is not None and v < lo:
        errs.append(f"{where}.{key}={v} below safe minimum {lo}")
    if hi is not None and v > hi:
        errs.append(f"{where}.{key}={v} above safe maximum {hi}")
    return v


_ALLOWED_SIZING = {"dynamic_leverage", "risk_per_trade_pct", "max_leverage", "min_leverage"}
_ALLOWED_PACE = {"enabled", "campaign_days", "min_margin_fraction",
                 "max_margin_fraction", "warmup_trips", "factor_ceiling_day1",
                 "factor_ceiling_day2", "factor_ceiling"}


def validate_config(cfg: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable errors; empty list == config is safe.

    Blocks on: missing required sections/keys, wrong types, out-of-bounds safety
    params, a TP cap below the fee floor (a silently-unprofitable TP), an
    unparseable timeframe, and unknown keys in the two small high-risk dicts
    (sizing, pace) where a typo would change real-money sizing.
    """
    errs: List[str] = []
    for section in ("app", "exchange", "fees", "farmer", "risk", "target"):
        if section not in cfg:
            errs.append(f"top-level section '{section}' is MISSING")
    if errs:
        return errs  # nothing else is meaningful without the sections

    ex = cfg["exchange"]
    if "symbol" not in ex:
        errs.append("exchange.symbol is MISSING")
    tf = ex.get("timeframe")
    if not tf:
        errs.append("exchange.timeframe is MISSING")
    else:
        try:
            tf_seconds(str(tf))
        except ValueError as e:
            errs.append(f"exchange.timeframe invalid: {e}")

    fees = cfg["fees"]
    _num(fees, "maker", errs, "fees", lo=0.0, hi=0.01)
    _num(fees, "taker", errs, "fees", lo=0.0, hi=0.01)
    _num(fees, "rebate_pct", errs, "fees", lo=0.0, hi=1.0)

    farmer = cfg["farmer"]
    _num(farmer, "margin_fraction_per_trade", errs, "farmer", lo=1e-4, hi=0.5)
    _num(farmer, "working_capital_usd", errs, "farmer", lo=1.0, hi=1e9)

    sizing = farmer.get("sizing", {}) or {}
    if not sizing:
        errs.append("farmer.sizing is MISSING")
    else:
        for k in sizing:
            if k not in _ALLOWED_SIZING:
                errs.append(f"farmer.sizing has unknown key '{k}' (typo? "
                            f"allowed: {sorted(_ALLOWED_SIZING)})")
        maxlev = _num(sizing, "max_leverage", errs, "farmer.sizing", lo=1.0, hi=30.0)
        minlev = _num(sizing, "min_leverage", errs, "farmer.sizing", lo=1.0, hi=30.0)
        _num(sizing, "risk_per_trade_pct", errs, "farmer.sizing", lo=1e-4, hi=1.0)
        if maxlev is not None and minlev is not None and minlev > maxlev:
            errs.append(f"farmer.sizing.min_leverage={minlev} > max_leverage={maxlev}")

    pace = farmer.get("pace", {}) or {}
    if pace:
        for k in pace:
            if k not in _ALLOWED_PACE:
                errs.append(f"farmer.pace has unknown key '{k}' (typo? "
                            f"allowed: {sorted(_ALLOWED_PACE)})")
        _num(pace, "campaign_days", errs, "farmer.pace", lo=1.0, hi=365.0)
        pmin = _num(pace, "min_margin_fraction", errs, "farmer.pace", lo=1e-4, hi=0.5)
        pmax = _num(pace, "max_margin_fraction", errs, "farmer.pace", lo=1e-4, hi=0.5)
        _num(pace, "warmup_trips", errs, "farmer.pace", lo=0.0, hi=1e5, required=False)
        if pmin is not None and pmax is not None and pmin > pmax:
            errs.append(f"farmer.pace.min_margin_fraction={pmin} > max_margin_fraction={pmax}")

    atr = farmer.get("atr", {}) or {}
    if atr:
        _num(atr, "tp_mult", errs, "farmer.atr", lo=1e-6, hi=100.0, required=False)
        _num(atr, "sl_mult", errs, "farmer.atr", lo=1e-6, hi=50.0, required=False)
        tp_cap = _num(atr, "tp_bps_max", errs, "farmer.atr", lo=0.0, hi=1e4, required=False)
        if tp_cap is not None and tp_cap > 0:
            floor = fee_floor_bps(cfg)
            if tp_cap < floor:
                errs.append(
                    f"farmer.atr.tp_bps_max={tp_cap} is BELOW the fee floor "
                    f"{floor:.2f}bps — every TP would be capped to a loss. "
                    f"Raise tp_bps_max to >= {floor:.2f}.")

    risk = cfg["risk"]
    _num(risk, "live_breaker_daily_loss_pct", errs, "risk", lo=0.01, hi=0.50, required=False)
    _num(risk, "live_breaker_max_drawdown_pct", errs, "risk", lo=0.05, hi=0.99, required=False)
    _num(risk, "consecutive_losses_cooldown_bars", errs, "risk", lo=0.0, hi=1e4, required=False)

    _num(cfg["target"], "volume_usd", errs, "target", lo=1.0, hi=1e12)
    return errs


# ---------------------------------------------------------------------------
# liquidation / leverage safety
# ---------------------------------------------------------------------------

def check_liquidation_safety(
    symbol: str, atr_bps_ref: float, *, max_leverage: float, sl_mult: float,
    safety_factor: float = 2.0, mmr: float = DEFAULT_MMR,
) -> Tuple[List[str], List[str]]:
    """Verify the SL the strategy will place sits comfortably INSIDE the
    liquidation line under TYPICAL volatility.

    Returns (errors, warnings). ``atr_bps_ref`` should be the MEDIAN ATR in bps
    over recent candles (robust to spikes). A genuine over-leverage misconfig
    (liq line nearer than safety_factor * SL distance at typical vol) is an
    ERROR (blocks startup). The 9xATR stop is a designed disaster floor that, in
    extreme bars, can reach the liq line — that is reported only as a WARNING.
    """
    errs: List[str] = []
    warns: List[str] = []
    if atr_bps_ref <= 0 or max_leverage <= 0 or sl_mult <= 0:
        warns.append(f"{symbol}: liq-safety check skipped (insufficient ATR/lev data)")
        return errs, warns
    liq = liq_distance_bps(max_leverage, mmr)
    sl = sl_mult * atr_bps_ref
    margin = liq / sl if sl > 0 else float("inf")
    if margin < safety_factor:
        errs.append(
            f"{symbol}: liquidation UNSAFE — at typical ATR ({atr_bps_ref:.1f}bps) "
            f"the {sl_mult:g}xATR stop is {sl:.0f}bps vs liq line {liq:.0f}bps "
            f"({margin:.2f}x < required {safety_factor:g}x). Lower max_leverage "
            f"or sl_mult, or raise working capital.")
    elif margin < safety_factor * 1.25:
        warns.append(
            f"{symbol}: liq margin {margin:.2f}x at typical ATR — thin; a vol "
            f"spike can push the disaster stop onto the liq line.")
    return errs, warns


def deadman_idle_exceeded(now: float, last_fill: float, last_blocked_seen: float,
                          start_wall: float, idle_s: float) -> bool:
    """True if the bot has been ABLE to trade yet produced no fill for idle_s.

    The baseline is the most recent of: a confirmed fill, a moment the gate was
    legitimately blocked (breaker/daily-halt/cooldown — 'no trades' is expected
    then, not a wedge), and process start. So a daily-loss halt that lasts past
    UTC midnight never trips the dead-man; only a gate-OPEN-but-silent stretch
    does. Pure function so the firing rule is unit-tested in isolation.
    """
    baseline = max(last_fill, last_blocked_seen, start_wall)
    return (now - baseline) >= idle_s


# ---------------------------------------------------------------------------
# file-backed kill switch (process-wide, persists across restarts)
# ---------------------------------------------------------------------------

class KillSwitch:
    """A persistent halt flag. While engaged, every live entry is blocked and a
    restart refuses to trade until it is cleared — so a /kill survives a crash or
    a systemd restart. The flag is a small JSON file; existence == engaged."""

    def __init__(self, flag_path) -> None:
        self.flag_path = pathlib.Path(flag_path)

    def is_engaged(self) -> bool:
        return self.flag_path.exists()

    def reason(self) -> str:
        try:
            return str(json.loads(self.flag_path.read_text()).get("reason", ""))
        except Exception:  # noqa: BLE001
            return ""

    def engage(self, reason: str) -> None:
        """Idempotent: keeps the FIRST engagement's reason/timestamp so a later
        re-trigger doesn't overwrite why the bot was originally halted."""
        if self.flag_path.exists():
            return
        try:
            self.flag_path.parent.mkdir(parents=True, exist_ok=True)
            self.flag_path.write_text(json.dumps(
                {"reason": reason, "ts": int(time.time() * 1000)}))
        except Exception:  # noqa: BLE001
            pass

    def clear(self) -> bool:
        """Remove the flag. Returns True if a flag was actually cleared."""
        try:
            self.flag_path.unlink()
            return True
        except FileNotFoundError:
            return False
        except Exception:  # noqa: BLE001
            return False
