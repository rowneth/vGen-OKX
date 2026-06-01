"""
OKX Fix-Stack Backtest — incremental impact of each structural fix.

Tests each fix applied cumulatively on top of the MEXC/OKX baseline:
  Baseline  : alternation ON, fixed 8/50 bps, all sessions, MEXC 70% rebate
  OKX base  : alternation ON, fixed 8/50 bps, all sessions, OKX  40% rebate
  Fix 1     : remove alternation  (candle-body direction only)
  Fix 2     : + ATR-relative TP/SL  (TP=0.5×ATR, SL=1.5×ATR per entry bar)
  Fix 3     : + skip low-vol bars   (ATR < p33 → no entry)
  Fix 4     : + high-vol-only       (ATR < p67 → no entry)
  Fix 5     : + OKX limit-TP        (TP exits fill as MAKER on limit order)

For each config, reports:
  • Win rate vs geometric break-even WR
  • Net P&L under OKX 40% rebate (or MEXC 70%)
  • Volume throughput (trades/yr, $/yr, days to $100k target)
  • Regime breakdown (low / mid / high ATR)
  • Breakeven rebate required

Usage:
    cd /Users/rowneth/vGen/bot
    python scripts/backtest_okx_fixes.py
    python scripts/backtest_okx_fixes.py --capital 300
"""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data/historical/BTC_USDT_5m.parquet"

# ── OKX fees ──────────────────────────────────────────────────────────────────
OKX_MAKER   = 0.0002   # 0.02%
OKX_TAKER   = 0.0005   # 0.05%
OKX_REBATE  = 0.40

MEXC_MAKER  = 0.0001   # 0.01%
MEXC_TAKER  = 0.0005   # 0.05%
MEXC_REBATE = 0.70

# ── position sizing ───────────────────────────────────────────────────────────
MARGIN_FRAC = 0.05
RISK_PCT    = 0.025
MAX_LEV     = 125.0
MIN_LEV     = 5.0

# ── entry gates (common to all configs) ──────────────────────────────────────
MIN_RANGE_BPS = 4.0
MAX_RANGE_BPS = 40.0


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    label:         str
    alternate:     bool          # force L/S/L/S alternation?
    fixed_tp:      Optional[float]  # bps; None → ATR-relative
    fixed_sl:      Optional[float]  # bps; None → ATR-relative
    atr_tp_mult:   float = 0.5   # TP = atr_tp_mult × ATR_bps
    atr_sl_mult:   float = 1.5   # SL = atr_sl_mult × ATR_bps
    min_atr_pct:   float = 0.0   # skip entry if ATR < this percentile (0=none)
    limit_tp:      bool  = False # True → TP close uses maker fee
    maker:         float = OKX_MAKER
    taker:         float = OKX_TAKER
    rebate:        float = OKX_REBATE
    # Trend-break: disabled for ATR-relative (SL already tight)
    tb_enabled:    bool  = True
    tb_min_bars:   int   = 3
    tb_adverse:    float = 20.0  # bps; set huge to disable


@dataclass
class Trade:
    side:         str
    entry_bar:    int
    exit_bar:     int
    bars_held:    int
    entry_price:  float
    exit_price:   float
    notional:     float
    tp_bps_used:  float       # actual TP bps for this trade
    sl_bps_used:  float       # actual SL bps for this trade
    reason:       str         # tp | sl | sl_ambiguous | trend_break
    maker_fill:   bool        # True if TP exit can fill as maker
    gross_pnl:    float
    equity_after: float
    atr_regime:   str         # low | mid | high


# ─────────────────────────────────────────────────────────────────────────────
def _atr14(df: pd.DataFrame) -> np.ndarray:
    hi  = df["high"].values.astype(float)
    lo  = df["low"].values.astype(float)
    cl  = df["close"].values.astype(float)
    pcl = np.concatenate([[cl[0]], cl[:-1]])
    tr  = np.maximum(hi - lo, np.maximum(np.abs(hi - pcl), np.abs(lo - pcl)))
    atr = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr[13] = tr[:14].mean()
        for k in range(14, len(tr)):
            atr[k] = atr[k-1] * (13/14) + tr[k] * (1/14)
    return atr


def _leverage(equity: float, sl_bps: float) -> float:
    margin = equity * MARGIN_FRAC
    risk   = equity * RISK_PCT
    sl_f   = sl_bps / 10_000
    lev    = risk / (margin * sl_f) if sl_f > 0 else MAX_LEV
    return max(MIN_LEV, min(lev, MAX_LEV))


# ─────────────────────────────────────────────────────────────────────────────
def simulate(
    df:          pd.DataFrame,
    cfg:         Config,
    atr_vals:    np.ndarray,
    atr_p33:     float,
    atr_p67:     float,
    capital:     float,
) -> List[Trade]:
    """Bar-by-bar simulation for one Config. Returns list of Trade objects."""

    opens  = df["open"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    n      = len(df)

    # ATR threshold for session filter
    if cfg.min_atr_pct >= 67:
        atr_threshold = atr_p67
    elif cfg.min_atr_pct >= 33:
        atr_threshold = atr_p33
    else:
        atr_threshold = 0.0

    equity    = capital
    last_side = ""
    trades: List[Trade] = []

    i = 14  # need ATR warm-up
    while i < n - 1:
        o = opens[i]
        c = closes[i]

        if o <= 0:
            i += 1
            continue

        # ── ATR session filter ────────────────────────────────────────────────
        atr_now = atr_vals[i]
        if np.isnan(atr_now) or atr_now < atr_threshold:
            i += 1
            continue

        # ── Gate: bar range ───────────────────────────────────────────────────
        rng_bps = abs(c - o) / o * 10_000
        if rng_bps < MIN_RANGE_BPS or rng_bps > MAX_RANGE_BPS:
            i += 1
            continue

        # ── Direction signal (candle body) ────────────────────────────────────
        bias = "long" if c > o else "short"

        # ── Alternation gate ─────────────────────────────────────────────────
        if cfg.alternate and last_side == bias:
            bias = "short" if bias == "long" else "long"

        # ── TP / SL sizing ────────────────────────────────────────────────────
        if cfg.fixed_tp is not None:
            tp_bps = cfg.fixed_tp
            sl_bps = cfg.fixed_sl
        else:
            # ATR-relative: convert ATR $ → bps at entry price
            atr_bps = atr_now / o * 10_000
            tp_bps  = cfg.atr_tp_mult * atr_bps
            sl_bps  = cfg.atr_sl_mult * atr_bps
            # Guard: TP must be above round-trip maker fee (4 bps) to be viable
            tp_bps  = max(tp_bps, 5.0)
            sl_bps  = max(sl_bps, 8.0)

        # ── Entry ─────────────────────────────────────────────────────────────
        entry_price = c
        entry_bar   = i

        lev      = _leverage(equity, sl_bps)
        margin   = equity * MARGIN_FRAC
        notional = margin * lev
        open_fee = notional * cfg.maker
        equity  -= open_fee

        tp = entry_price * (1 + tp_bps / 10_000) if bias == "long" \
             else entry_price * (1 - tp_bps / 10_000)
        sl = entry_price * (1 - sl_bps / 10_000) if bias == "long" \
             else entry_price * (1 + sl_bps / 10_000)

        # ── Exit loop ─────────────────────────────────────────────────────────
        reason:     Optional[str] = None
        exit_price: float         = 0.0
        maker_fill: bool          = False
        exit_bar:   int           = i

        for j in range(i + 1, n):
            h    = highs[j]
            l    = lows[j]
            ob   = opens[j]
            cb   = closes[j]
            bh   = j - entry_bar

            tp_hit = (bias == "long"  and h >= tp) or (bias == "short" and l <= tp)
            sl_hit = (bias == "long"  and l <= sl) or (bias == "short" and h >= sl)

            if tp_hit and sl_hit:
                reason = "sl_ambiguous"; exit_price = sl; exit_bar = j; break

            if tp_hit:
                reason = "tp"; exit_price = tp; exit_bar = j
                # Maker-fill: bar opened on the "safe" side of TP
                maker_fill = (ob < tp) if bias == "long" else (ob > tp)
                break

            if sl_hit:
                reason = "sl"; exit_price = sl; exit_bar = j; break

            # Trend-break
            if cfg.tb_enabled and bh >= cfg.tb_min_bars:
                adverse = ((entry_price - cb) / entry_price * 10_000) if bias == "long" \
                          else ((cb - entry_price) / entry_price * 10_000)
                if adverse >= cfg.tb_adverse:
                    reason = "trend_break"; exit_price = cb; exit_bar = j; break

        if reason is None:
            i += 1
            continue

        # ── P&L (use taker for equity tracking conservatively) ────────────────
        pnl_pct   = (exit_price - entry_price) / entry_price if bias == "long" \
                    else (entry_price - exit_price) / entry_price
        gross_pnl = pnl_pct * notional
        equity   += gross_pnl - notional * cfg.taker
        equity    = max(equity, 0.01)

        # ATR regime
        a = atr_vals[entry_bar]
        regime = "mid"
        if not np.isnan(a):
            regime = "low" if a <= atr_p33 else ("mid" if a <= atr_p67 else "high")

        trades.append(Trade(
            side         = bias,
            entry_bar    = entry_bar,
            exit_bar     = exit_bar,
            bars_held    = exit_bar - entry_bar,
            entry_price  = entry_price,
            exit_price   = exit_price,
            notional     = notional,
            tp_bps_used  = tp_bps,
            sl_bps_used  = sl_bps,
            reason       = reason,
            maker_fill   = maker_fill,
            gross_pnl    = gross_pnl,
            equity_after = equity,
            atr_regime   = regime,
        ))

        last_side = bias
        i = exit_bar + 1

    return trades


# ─────────────────────────────────────────────────────────────────────────────
def analyze(trades: List[Trade], cfg: Config, capital: float, n_days: float) -> dict:
    if not trades:
        return {"_empty": True, "_label": cfg.label}

    total_gross = 0.0
    total_fees  = 0.0
    wins = losses = 0
    by_regime   = {"low": [0,0], "mid": [0,0], "high": [0,0]}
    tp_maker = tp_taker = 0
    avg_tp_bps_list = []
    avg_sl_bps_list = []

    for t in trades:
        # Fee under this scenario
        if t.reason == "tp" and cfg.limit_tp and t.maker_fill:
            close_fee = t.notional * cfg.maker   # maker close
            tp_maker  += 1
        else:
            close_fee = t.notional * cfg.taker   # taker close
            if t.reason == "tp":
                tp_taker += 1

        open_fee  = t.notional * cfg.maker
        fee       = open_fee + close_fee
        net_pnl   = t.gross_pnl - fee

        total_gross += t.gross_pnl
        total_fees  += fee

        reg = t.atr_regime
        if net_pnl > 0:
            wins += 1
            by_regime[reg][0] += 1
        else:
            losses += 1
            by_regime[reg][1] += 1

        avg_tp_bps_list.append(t.tp_bps_used)
        avg_sl_bps_list.append(t.sl_bps_used)

    n = wins + losses
    wr  = wins / n if n else 0.0

    # Geometric break-even WR  (weighted avg TP/SL bps)
    avg_tp = float(np.mean(avg_tp_bps_list))
    avg_sl = float(np.mean(avg_sl_bps_list))
    be_wr  = avg_sl / (avg_tp + avg_sl) if (avg_tp + avg_sl) > 0 else 0.5

    rebate   = total_fees * cfg.rebate
    true_net = total_gross - total_fees + rebate

    be_rebate = max(0.0, (total_fees - total_gross) / total_fees) if total_fees > 0 else 0.0

    total_vol   = sum(t.notional * 2 for t in trades)
    vol_per_day = total_vol / n_days if n_days > 0 else 0.0
    days_100k   = 100_000 / vol_per_day if vol_per_day > 0 else 9999
    days_1M     = 1_000_000 / vol_per_day if vol_per_day > 0 else 9999

    net_per_M = true_net / total_vol * 1_000_000 if total_vol > 0 else 0.0

    tp_hits   = sum(1 for t in trades if t.reason == "tp")
    maker_pct = tp_maker / tp_hits * 100 if tp_hits > 0 else 0.0

    return {
        "_label":       cfg.label,
        "_limit_tp":    cfg.limit_tp,
        "_alternate":   cfg.alternate,
        "trades":       n,
        "wins":         wins,
        "losses":       losses,
        "wr_pct":       wr * 100,
        "be_wr_pct":    be_wr * 100,
        "wr_gap":       (wr - be_wr) * 100,
        "avg_tp_bps":   avg_tp,
        "avg_sl_bps":   avg_sl,
        "gross_pnl":    total_gross,
        "total_fees":   total_fees,
        "rebate":       rebate,
        "true_net":     true_net,
        "net_per_M":    net_per_M,
        "be_rebate_pct": be_rebate * 100,
        "total_vol":    total_vol,
        "vol_per_day":  vol_per_day,
        "days_100k":    days_100k,
        "days_1M":      days_1M,
        "tp_hits":      tp_hits,
        "maker_fill_pct": maker_pct,
        "by_regime":    by_regime,
        "rebate_pct_used": cfg.rebate * 100,
    }


# ─────────────────────────────────────────────────────────────────────────────
def print_main_table(rows: List[dict]) -> None:
    W = 105
    print()
    print("═" * W)
    print("  FIX-STACK BACKTEST  |  Each row = cumulative fixes applied on top of baseline")
    print("  OKX fees: maker=0.02%  taker=0.05%  rebate=40%   |  MEXC: maker=0.01%  rebate=70%")
    print("═" * W)
    hdr = (f"  {'Config':<26} {'Trades':>7} {'WR%':>7} {'BE-WR%':>7} {'WRgap':>7} "
           f"{'AvgTP':>6} {'AvgSL':>6} {'Gross$':>8} {'Fees$':>7} {'Rebate$':>8} "
           f"{'Net$':>8} {'$/1M':>8} {'BEreb%':>7} {'OK?':>5}")
    print(hdr)
    print("  " + "─" * (W-2))

    prev_group = None
    for r in rows:
        if r.get("_empty"):
            continue
        grp = r["_label"].split(":")[0].strip()
        if prev_group and grp != prev_group:
            print("  " + "─" * (W-2))
        prev_group = grp

        ok   = r["true_net"] > 0
        star = "✅" if ok else "❌"
        row  = (
            f"  {r['_label']:<26}"
            f" {r['trades']:>7,}"
            f" {r['wr_pct']:>6.1f}%"
            f" {r['be_wr_pct']:>6.1f}%"
            f" {r['wr_gap']:>+6.1f}%"
            f" {r['avg_tp_bps']:>5.1f}b"
            f" {r['avg_sl_bps']:>5.1f}b"
            f" {r['gross_pnl']:>+8.2f}"
            f" {r['total_fees']:>7.2f}"
            f" {r['rebate']:>+8.2f}"
            f" {r['true_net']:>+8.2f}"
            f" {r['net_per_M']:>+8.2f}"
            f" {r['be_rebate_pct']:>6.1f}%"
            f"  {star}"
        )
        print(row)
    print("  " + "═" * (W-2))


def print_volume_table(rows: List[dict], capital: float) -> None:
    print()
    print("═" * 90)
    print(f"  VOLUME THROUGHPUT  (capital=${capital:.0f}, 1-year dataset ~362 days)")
    print("═" * 90)
    print(f"  {'Config':<26} {'Trades/yr':>10} {'Vol$/yr':>12} {'Vol$/mo':>10} "
          f"{'Days→$100k':>12} {'Days→$1M':>10}")
    print("  " + "─" * 82)

    for r in rows:
        if r.get("_empty"):
            continue
        print(
            f"  {r['_label']:<26}"
            f" {r['trades']:>10,}"
            f" {r['total_vol']:>12,.0f}"
            f" {r['total_vol']/12:>10,.0f}"
            f" {r['days_100k']:>12.1f}"
            f" {r['days_1M']:>10.1f}"
        )
    print("  " + "═" * 82)
    print("  NOTE: 'days to target' = target / (vol per day from backtest period).")
    print("        Higher capital → proportionally faster (notional scales with equity).")


def print_wr_impact(rows: List[dict]) -> None:
    """Show WR improvement from each fix as a delta."""
    print()
    print("═" * 70)
    print("  WIN RATE IMPROVEMENT  — cumulative effect of each fix")
    print("═" * 70)
    print(f"  {'Config':<26} {'WR%':>7} {'Δ vs prev':>10} {'BE-WR%':>8} {'Gap':>8}")
    print("  " + "─" * 62)

    prev_wr = None
    for r in rows:
        if r.get("_empty"):
            continue
        wr  = r["wr_pct"]
        delta = f"{wr - prev_wr:+.2f}%" if prev_wr is not None else "—"
        bar_ch = ""
        if prev_wr is not None:
            diff = wr - prev_wr
            bar_ch = ("▲" * min(int(abs(diff)*2), 8)) if diff > 0 \
                     else ("▼" * min(int(abs(diff)*2), 8))
            bar_ch = f" {bar_ch}"
        print(
            f"  {r['_label']:<26}"
            f" {wr:>6.2f}%"
            f" {delta:>10}"
            f" {r['be_wr_pct']:>7.2f}%"
            f" {r['wr_gap']:>+7.2f}%"
            f"{bar_ch}"
        )
        prev_wr = wr
    print("  " + "═" * 62)


def print_regime_breakdown(rows: List[dict]) -> None:
    print()
    print("═" * 72)
    print("  REGIME BREAKDOWN  — WR% per volatility bucket (each config)")
    print("  ATR terciles: Low < p33  |  p33 < Mid < p67  |  High > p67")
    print("═" * 72)
    print(f"  {'Config':<26} {'Low WR%':>9} {'Med WR%':>9} {'High WR%':>10} "
          f"{'Low N':>7} {'Med N':>7} {'High N':>7}")
    print("  " + "─" * 64)

    for r in rows:
        if r.get("_empty"):
            continue
        br  = r["by_regime"]
        def wr_of(k):
            w, l = br[k]
            return w / (w+l) * 100 if (w+l) > 0 else 0.0
        def n_of(k):
            w, l = br[k]
            return w + l

        print(
            f"  {r['_label']:<26}"
            f" {wr_of('low'):>8.1f}%"
            f" {wr_of('mid'):>8.1f}%"
            f" {wr_of('high'):>9.1f}%"
            f" {n_of('low'):>7,}"
            f" {n_of('mid'):>7,}"
            f" {n_of('high'):>7,}"
        )
    print("  " + "═" * 64)


def print_breakeven_table(rows: List[dict]) -> None:
    print()
    print("═" * 72)
    print("  BREAKEVEN REBATE — minimum rebate% for TrueNet = 0")
    print("  OKX current: 40%  |  MEXC: 70%")
    print("═" * 72)
    print(f"  {'Config':<26} {'Rebate used':>12} {'BE Rebate%':>12} {'Gap':>8} {'Status':>8}")
    print("  " + "─" * 64)

    for r in rows:
        if r.get("_empty"):
            continue
        cur = r["rebate_pct_used"]
        be  = r["be_rebate_pct"]
        gap = cur - be
        ok  = "✅ ok" if gap >= 0 else "❌ need more"
        print(
            f"  {r['_label']:<26}"
            f" {cur:>11.0f}%"
            f" {be:>11.1f}%"
            f" {gap:>+7.1f}%"
            f"  {ok}"
        )
    print("  " + "═" * 64)


def print_per_trade_economics(rows: List[dict], capital: float) -> None:
    """Show per-trade avg economics for key configs."""
    print()
    print("═" * 80)
    print("  PER-TRADE ECONOMICS  (avg notional & fee $ at starting equity)")
    print("═" * 80)
    print(f"  {'Config':<26} {'Avg Notional':>13} {'OpenFee$':>9} {'WinFee$':>9} "
          f"{'LossFee$':>9} {'NetWin$':>9} {'NetLoss$':>10}")
    print("  " + "─" * 72)

    for r in rows:
        if r.get("_empty"):
            continue
        tp  = r["avg_tp_bps"]
        sl  = r["avg_sl_bps"]
        # Rough avg notional: equity * MARGIN_FRAC * leverage
        # leverage ≈ RISK_PCT / (MARGIN_FRAC * sl/10000) clamped to [5,125]
        lev = min(MAX_LEV, max(MIN_LEV, RISK_PCT / (MARGIN_FRAC * sl / 10_000)))
        notional = capital * MARGIN_FRAC * lev

        maker = OKX_MAKER if "MEXC" not in r["_label"] else MEXC_MAKER
        taker = OKX_TAKER

        open_fee  = notional * maker
        win_fee   = open_fee + notional * (maker if r["_limit_tp"] else taker)
        loss_fee  = open_fee + notional * taker
        net_win   = tp / 10_000 * notional - win_fee
        net_loss  = -sl / 10_000 * notional - loss_fee

        print(
            f"  {r['_label']:<26}"
            f" {notional:>13.2f}"
            f" {open_fee:>9.4f}"
            f" {win_fee:>9.4f}"
            f" {loss_fee:>9.4f}"
            f" {net_win:>+9.4f}"
            f" {net_loss:>+10.4f}"
        )
    print("  " + "═" * 72)


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=30.0)
    parser.add_argument("--rows", type=int, default=None)
    args = parser.parse_args()

    print(f"\nLoading {DATA} …")
    df = pd.read_parquet(DATA)
    df.columns = [c.lower() for c in df.columns]
    if args.rows:
        df = df.iloc[-args.rows:].reset_index(drop=True)
    print(f"Loaded {len(df):,} bars  "
          f"({df['open_time'].iloc[0]}  →  {df['open_time'].iloc[-1]})\n")

    # Days in dataset
    t0 = pd.to_datetime(df["open_time"].iloc[0],  utc=True)
    t1 = pd.to_datetime(df["open_time"].iloc[-1], utc=True)
    n_days = (t1 - t0).total_seconds() / 86_400
    print(f"Dataset span: {n_days:.1f} days\n")

    # ATR
    atr_vals = _atr14(df)
    valid    = atr_vals[~np.isnan(atr_vals)]
    atr_p33  = float(np.percentile(valid, 33))
    atr_p67  = float(np.percentile(valid, 67))
    print(f"ATR-14:  p33=${atr_p33:.1f}  p67=${atr_p67:.1f}\n")

    # ── Config definitions ───────────────────────────────────────────────────
    CONFIGS = [
        Config(
            label     = "MEXC baseline",
            alternate = True, fixed_tp=8.0, fixed_sl=50.0,
            maker=MEXC_MAKER, taker=MEXC_TAKER, rebate=MEXC_REBATE,
            tb_enabled=True, tb_adverse=20.0,
        ),
        Config(
            label     = "OKX base (alt+fixed)",
            alternate = True, fixed_tp=8.0, fixed_sl=50.0,
            maker=OKX_MAKER, taker=OKX_TAKER, rebate=OKX_REBATE,
            tb_enabled=True, tb_adverse=20.0,
        ),
        Config(
            label     = "Fix1: no-alt",
            alternate = False, fixed_tp=8.0, fixed_sl=50.0,
            maker=OKX_MAKER, taker=OKX_TAKER, rebate=OKX_REBATE,
            tb_enabled=True, tb_adverse=20.0,
        ),
        Config(
            label     = "Fix2: +ATR-TP/SL",
            alternate = False, fixed_tp=None, fixed_sl=None,
            atr_tp_mult=0.5, atr_sl_mult=1.5,
            maker=OKX_MAKER, taker=OKX_TAKER, rebate=OKX_REBATE,
            tb_enabled=False, tb_adverse=999.0,  # SL already adaptive
        ),
        Config(
            label     = "Fix3: +skip-lowvol",
            alternate = False, fixed_tp=None, fixed_sl=None,
            atr_tp_mult=0.5, atr_sl_mult=1.5, min_atr_pct=33.0,
            maker=OKX_MAKER, taker=OKX_TAKER, rebate=OKX_REBATE,
            tb_enabled=False, tb_adverse=999.0,
        ),
        Config(
            label     = "Fix4: +highvol-only",
            alternate = False, fixed_tp=None, fixed_sl=None,
            atr_tp_mult=0.5, atr_sl_mult=1.5, min_atr_pct=67.0,
            maker=OKX_MAKER, taker=OKX_TAKER, rebate=OKX_REBATE,
            tb_enabled=False, tb_adverse=999.0,
        ),
        Config(
            label     = "Fix5: +limit-TP",
            alternate = False, fixed_tp=None, fixed_sl=None,
            atr_tp_mult=0.5, atr_sl_mult=1.5, min_atr_pct=67.0,
            maker=OKX_MAKER, taker=OKX_TAKER, rebate=OKX_REBATE,
            tb_enabled=False, tb_adverse=999.0,
            limit_tp  = True,
        ),
    ]

    # ── Run ─────────────────────────────────────────────────────────────────
    rows = []
    print("Running simulations …")
    # Cache trades by (alternate, fixed_tp, fixed_sl, atr_tp_mult, atr_sl_mult, min_atr_pct)
    # because MEXC / OKX base share the same trades, only fees differ.
    trade_cache: dict = {}

    for cfg in CONFIGS:
        cache_key = (
            cfg.alternate, cfg.fixed_tp, cfg.fixed_sl,
            cfg.atr_tp_mult, cfg.atr_sl_mult, cfg.min_atr_pct,
            cfg.tb_enabled, cfg.tb_adverse,
        )
        if cache_key in trade_cache:
            trades = trade_cache[cache_key]
            print(f"  {cfg.label:<26}  (cached {len(trades):,} trades)", end="")
        else:
            trades = simulate(df, cfg, atr_vals, atr_p33, atr_p67, args.capital)
            trade_cache[cache_key] = trades
            print(f"  {cfg.label:<26}  {len(trades):,} trades", end="")

        r = analyze(trades, cfg, args.capital, n_days)
        rows.append(r)
        print(f"   WR={r['wr_pct']:.1f}%  net={r['true_net']:+.2f}  BE-reb={r['be_rebate_pct']:.1f}%")

    # ── Output ───────────────────────────────────────────────────────────────
    print_main_table(rows)
    print_wr_impact(rows)
    print_volume_table(rows, args.capital)
    print_regime_breakdown(rows)
    print_breakeven_table(rows)
    print_per_trade_economics(rows, args.capital)

    # ── Final verdict ────────────────────────────────────────────────────────
    print()
    print("═" * 72)
    print("  VERDICT")
    print("═" * 72)
    profitable = [r for r in rows if r["true_net"] > 0 and not r.get("_empty")]
    if profitable:
        best = max(profitable, key=lambda r: r["net_per_M"])
        print(f"  Best config:  {best['_label']}")
        print(f"  Net/trade:    ${best['true_net']/best['trades']:+.4f}")
        print(f"  Net per $1M:  ${best['net_per_M']:+.2f}")
        print(f"  WR:           {best['wr_pct']:.2f}%  (be={best['be_wr_pct']:.2f}%,  gap={best['wr_gap']:+.2f}%)")
        print(f"  Days→$100k:   {best['days_100k']:.1f}")
    else:
        print("  ❌ No configuration is profitable at current rebate levels.")
        print()
        # Show what's closest
        closest = min(rows, key=lambda r: r.get("be_rebate_pct", 999) if not r.get("_empty") else 999)
        shortfall = closest["be_rebate_pct"] - closest["rebate_pct_used"]
        print(f"  Closest:      {closest['_label']}")
        print(f"  BE rebate:    {closest['be_rebate_pct']:.1f}%  "
              f"(need {shortfall:.1f}pp more than current {closest['rebate_pct_used']:.0f}%)")
        print(f"  WR gap:       {closest['wr_gap']:+.1f}%  "
              f"(actual {closest['wr_pct']:.1f}% vs BE {closest['be_wr_pct']:.1f}%)")
        print()
        print("  NEXT ACTIONS:")
        print("  → Check if OKX actual rebate negotiation can reach the BE level")
        print("  → Consider a directional bias filter (EMA or HTF trend) to lift WR")
        print("  → Consider 1m timeframe (more granular ATR-relative sizing, less holding time)")
    print()


if __name__ == "__main__":
    main()
