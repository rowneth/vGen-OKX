"""LIVE-config bar-stop (time-stop) sweep on historical 5m BTC.

Mirrors the CURRENT live config (config_volume_farmer_okx.yaml):
  - micro_momentum entry (close>open => long else short), bar-range gate
  - FIRM 12bps TP (force_tp_bps) — NOT ATR-relative; SL = 1.5 x ATR_bps (floor 8bps)
  - skip entry when ATR < $120 (min_usd); dynamic leverage, max_leverage = 68
  - exits: intrabar TP / SL (SL-first tie-break), else TIME-STOP at max_hold bars
  - fees: open = maker (post-only), TP close = maker (limit_tp), SL close = taker,
    TIME-STOP close = TAKER (maker_exit is OFF in live → market close) + slippage
  - rebate = 40% of gross fees (separate pool, like the live state JSON)

Sweeps max_hold ∈ {1..24, no-stop} with everything else fixed at the live values,
to answer: does a bar time-stop save the account from bleeding, or is no-stop best?

Usage:
    python scripts/backtest_live_barstop_sweep.py                 # realistic 3bps exit slip
    python scripts/backtest_live_barstop_sweep.py --sl-slippage-bps 0   # frictionless
"""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data/historical/BTC_USDT_5m.parquet"

# ── LIVE params (config_volume_farmer_okx.yaml) ──────────────────────────────
MAKER, TAKER, REBATE = 0.0002, 0.0005, 0.40
CAPITAL = 500.0
MARGIN_FRAC, RISK_PCT = 0.03, 0.025
MAX_LEV, MIN_LEV = 68.0, 5.0           # live SL-safety cap = 68
MIN_RANGE_BPS, MAX_RANGE_BPS = 3.0, 40.0
FORCE_TP_BPS = 12.0                    # firm effective TP (force_tp_bps)
SL_MULT, SL_BPS_MIN = 1.5, 8.0         # ATR-relative SL
ATR_MIN_USD = 120.0
LIMIT_TP = True                        # TP fills maker
MAKER_EXIT = False                     # live: maker_exit OFF → time_stop = taker market close
NO_STOP = 10**9


@dataclass
class Cfg:
    max_hold: int
    tag: str = ""


def _atr14(df: pd.DataFrame) -> np.ndarray:
    hi, lo, cl = (df[c].to_numpy(float) for c in ("high", "low", "close"))
    pcl = np.concatenate([[cl[0]], cl[:-1]])
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - pcl), np.abs(lo - pcl)))
    atr = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr[13] = tr[:14].mean()
        for k in range(14, len(tr)):
            atr[k] = atr[k - 1] * (13 / 14) + tr[k] * (1 / 14)
    return atr


def simulate(df: pd.DataFrame, atr: np.ndarray, cfg: Cfg, sl_slip_bps: float) -> dict:
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    n = len(df)
    rng_bps = np.abs(c - o) / o * 1e4

    # Fixed-capital sizing so per-trade economics are comparable across max_hold
    # values (matches the live working_capital_usd=500 cap; no compounding).
    nets: list[float] = []
    bps_list: list[float] = []
    reason_net: dict[str, list] = {}
    gross_sum = maker_fee = taker_fee = volume = 0.0

    i = 14
    while i < n - 1:
        a = atr[i]
        if np.isnan(a) or a < ATR_MIN_USD:
            i += 1; continue
        if not (MIN_RANGE_BPS <= rng_bps[i] <= MAX_RANGE_BPS):
            i += 1; continue

        side = 1 if c[i] > o[i] else -1          # micro_momentum (alternate=false)
        atr_bps = a / o[i] * 1e4
        tp_bps = FORCE_TP_BPS                     # firm 12bps (force_tp_bps)
        sl_bps = max(SL_MULT * atr_bps, SL_BPS_MIN)

        entry = c[i]
        margin = CAPITAL * MARGIN_FRAC
        sl_frac = sl_bps / 1e4
        lev = max(MIN_LEV, min((CAPITAL * RISK_PCT) / (margin * sl_frac), MAX_LEV)) if sl_frac else MAX_LEV
        notional = margin * lev
        open_fee = notional * MAKER
        maker_fee += open_fee; volume += notional

        tp = entry * (1 + tp_bps / 1e4) if side == 1 else entry * (1 - tp_bps / 1e4)
        sl = entry * (1 - sl_bps / 1e4) if side == 1 else entry * (1 + sl_bps / 1e4)

        reason = None; px = 0.0; exit_bar = i
        for j in range(i + 1, n):
            bh = j - i
            tp_hit = (h[j] >= tp) if side == 1 else (lo[j] <= tp)
            sl_hit = (lo[j] <= sl) if side == 1 else (h[j] >= sl)
            if tp_hit and sl_hit:
                reason, px, exit_bar = "sl_ambiguous", sl, j; break
            if tp_hit:
                reason, px, exit_bar = "tp", tp, j; break
            if sl_hit:
                reason, px, exit_bar = "sl", sl, j; break
            if bh >= cfg.max_hold:
                reason, px, exit_bar = "time_stop", c[j], j; break
        if reason is None:
            i += 1; continue

        gross = ((px - entry) if side == 1 else (entry - px)) / entry * notional
        if reason in ("sl", "sl_ambiguous"):
            gross -= sl_slip_bps * 1e-4 * notional          # taker SL slips against us
            close_fee = notional * TAKER; taker_fee += close_fee
        elif reason == "tp":
            close_fee = notional * (MAKER if LIMIT_TP else TAKER); maker_fee += close_fee
        else:  # time_stop — live maker_exit OFF → taker market close + slippage
            gross -= sl_slip_bps * 1e-4 * notional
            close_fee = notional * (MAKER if MAKER_EXIT else TAKER)
            (maker_fee, taker_fee) = (maker_fee + close_fee, taker_fee) if MAKER_EXIT else (maker_fee, taker_fee + close_fee)

        net = gross - open_fee - close_fee
        gross_sum += gross; volume += notional
        nets.append(net)
        bps_list.append(net / notional * 1e4)
        reason_net.setdefault(reason, []).append(net)
        i = exit_bar + 1

    nets_a = np.array(nets)
    wins = nets_a[nets_a > 0]; losses = nets_a[nets_a <= 0]
    total_fee = maker_fee + taker_fee
    rebate = REBATE * total_fee
    true_net = nets_a.sum() + rebate
    curve = CAPITAL + np.cumsum(nets_a)
    peak = np.maximum.accumulate(curve) if len(curve) else np.array([CAPITAL])
    maxdd = float((curve - peak).min()) if len(curve) else 0.0
    return dict(
        tag=cfg.tag, max_hold=cfg.max_hold, trades=len(nets),
        wr=len(wins) / len(nets) * 100 if nets else 0,
        avg_win=wins.mean() if len(wins) else 0, avg_loss=losses.mean() if len(losses) else 0,
        pf=(wins.sum() / -losses.sum()) if len(losses) and losses.sum() else float("inf"),
        exp_bps=float(np.mean(bps_list)) if bps_list else 0.0,
        ledger_net=float(nets_a.sum()), rebate=float(rebate), true_net=float(true_net),
        total_fee=float(total_fee), volume=float(volume),
        maxdd=maxdd, maxdd_pct=maxdd / CAPITAL * 100,
        reasons={k: (len(v), float(np.sum(v))) for k, v in reason_net.items()},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sl-slippage-bps", type=float, default=3.0,
                    help="adverse slippage on SL/time-stop market exits (live realism ~3bps)")
    args = ap.parse_args()

    df = pd.read_parquet(DATA).reset_index(drop=True)
    atr = _atr14(df)
    days = len(df) * 5 / 60 / 24
    print(f"data: {len(df)} 5m bars (~{days:.0f}d)  TP=firm {FORCE_TP_BPS:.0f}bps  "
          f"SL=1.5xATR(floor8)  lev<=68  sl/ts slip={args.sl_slippage_bps:.0f}bps  "
          f"time_stop=TAKER(maker_exit off)\n")

    holds = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, NO_STOP]
    rows = []
    for mh in holds:
        tag = "NO-STOP" if mh == NO_STOP else f"{mh:>2}bar"
        rows.append(simulate(df, atr, Cfg(max_hold=mh, tag=tag), args.sl_slippage_bps))

    hdr = (f"{'hold':>8} {'trades':>7} {'WR%':>6} {'exp(bps)':>9} {'ledgerNet':>10} "
           f"{'rebate':>8} {'TRUE-NET':>10} {'maxDD%':>7} {'vol($M)':>8} "
           f"{'tp':>6} {'sl':>6} {'tstop':>6} {'ts$':>9}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        rs = r["reasons"]
        tp_n = rs.get("tp", (0, 0))[0]
        sl_n = rs.get("sl", (0, 0))[0] + rs.get("sl_ambiguous", (0, 0))[0]
        ts_n, ts_d = rs.get("time_stop", (0, 0.0))
        print(f"{r['tag']:>8} {r['trades']:>7} {r['wr']:>6.1f} {r['exp_bps']:>9.3f} "
              f"{r['ledger_net']:>10.2f} {r['rebate']:>8.2f} {r['true_net']:>10.2f} "
              f"{r['maxdd_pct']:>7.1f} {r['volume']/1e6:>8.2f} "
              f"{tp_n:>6} {sl_n:>6} {ts_n:>6} {ts_d:>9.2f}")

    # Rank
    print()
    best_net = max(rows, key=lambda r: r["true_net"])
    best_dd = max(rows, key=lambda r: r["maxdd_pct"])   # least negative
    nostop = next(r for r in rows if r["max_hold"] == NO_STOP)
    print(f"best TRUE-NET : {best_net['tag']}  (${best_net['true_net']:.2f}, maxDD {best_net['maxdd_pct']:.1f}%)")
    print(f"best maxDD    : {best_dd['tag']}  (maxDD {best_dd['maxdd_pct']:.1f}%, net ${best_dd['true_net']:.2f})")
    print(f"NO-STOP       : net ${nostop['true_net']:.2f}, maxDD {nostop['maxdd_pct']:.1f}%, "
          f"WR {nostop['wr']:.1f}%, vol ${nostop['volume']/1e6:.2f}M")


if __name__ == "__main__":
    main()
