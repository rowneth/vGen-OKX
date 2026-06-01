"""
OKX Advanced Backtest — TP sensitivity, maker-fill simulation, regime analysis.

Tests TP ∈ {9, 10, 11, 15, 18} bps with:
  • OKX fee structure  maker=0.02%  taker=0.05%  rebate=40%
  • Maker-fill heuristic: limit-TP fills as MAKER when bar walks to TP,
    TAKER when bar OPENS at/above TP (gap-through → delayed limit order)

Outputs:
  A. Win-rate vs TP (actual from 1-year data)
  B. Maker fill rate per TP level
  C. Net P&L under two scenarios:
       S1 – All exits taker   (current MEXC behaviour)
       S2 – Limit TP          (OKX: TP fills as maker when possible)
  D. Breakeven rebate required per TP / scenario
  E. MEXC baseline row for direct comparison
  F. Regime breakdown (low / medium / high ATR)
  G. Bars-to-TP distribution (quartiles)

Usage:
    cd /Users/rowneth/vGen/bot
    source /Users/rowneth/vGen/.venv/bin/activate
    python scripts/backtest_okx_advanced.py
    python scripts/backtest_okx_advanced.py --tp-list 8 9 10 11 15 18 20
    python scripts/backtest_okx_advanced.py --sl 30
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data/historical/BTC_USDT_5m.parquet"

# ── fee scenarios ─────────────────────────────────────────────────────────────
SCENARIOS = {
    "mexc_baseline": dict(maker=0.0001, taker=0.0005, rebate=0.70, label="MEXC  (baseline)"),
    "okx_all_taker": dict(maker=0.0002, taker=0.0005, rebate=0.40, label="OKX  S1 all-taker"),
    "okx_limit_tp":  dict(maker=0.0002, taker=0.0005, rebate=0.40, label="OKX  S2 limit-TP"),
}

# ── position sizing (matches live config) ─────────────────────────────────────
MARGIN_FRAC   = 0.05    # 5% of equity per trade
RISK_PCT      = 0.025   # 2.5% max risk per trade
MAX_LEV       = 125.0
MIN_LEV       = 5.0

# ── entry gates ───────────────────────────────────────────────────────────────
MIN_RANGE_BPS = 4.0
MAX_RANGE_BPS = 40.0

# ── trend-break (same as live config) ────────────────────────────────────────
TB_ENABLED    = True
TB_MIN_BARS   = 3
TB_ADVERSE    = 20.0    # bps


# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Trade:
    entry_bar:   int
    exit_bar:    int
    bars_held:   int
    side:        str
    entry_price: float
    exit_price:  float
    notional:    float
    reason:      str            # tp | sl | sl_ambiguous | trend_break
    maker_fill:  bool           # True → TP closed as maker on limit order
    gross_pnl:   float
    equity_after: float
    atr_regime:  str            # low | mid | high


# ═══════════════════════════════════════════════════════════════════════════════
def _atr14(df: pd.DataFrame) -> np.ndarray:
    hi = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    cl = df["close"].values.astype(float)
    prev_cl = np.concatenate([[cl[0]], cl[:-1]])
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_cl), np.abs(lo - prev_cl)))
    atr = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr[13] = tr[:14].mean()
        alpha = 1.0 / 14.0
        for k in range(14, len(tr)):
            atr[k] = atr[k - 1] * (1 - alpha) + tr[k] * alpha
    return atr


def _leverage(equity: float, sl_bps: float) -> float:
    margin = equity * MARGIN_FRAC
    risk   = equity * RISK_PCT
    sl_f   = sl_bps / 10_000.0
    lev    = risk / (margin * sl_f) if sl_f > 0 else MAX_LEV
    return max(MIN_LEV, min(lev, MAX_LEV))


# ═══════════════════════════════════════════════════════════════════════════════
def simulate(
    df: pd.DataFrame,
    tp_bps: float,
    sl_bps: float,
    capital: float,
    maker_rate: float,
    taker_rate: float,
    atr_vals: np.ndarray,
    atr_p33: float,
    atr_p67: float,
) -> List[Trade]:
    """Bar-by-bar micro_momentum simulation.

    Implements the SAME gates as VolumeFarmerSession:
      Gate 1 bar-range filter, Gate 2 candle-body direction, Gate 3 alternation.

    Returns a list of Trade objects with per-exit detail.
    Maker-fill heuristic (for limit-TP simulation):
      LONG TP  → maker if exit_bar.open < tp_price  (price walked up to our resting sell)
      SHORT TP → maker if exit_bar.open > tp_price  (price walked down to our resting buy)
    """
    opens  = df["open"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    n      = len(df)

    equity     = capital
    last_side  = ""
    trades: List[Trade] = []

    i = 1
    while i < n:
        o = opens[i]
        c = closes[i]

        # ── check open position first (exit) ─────────────────────────────────
        if trades and trades[-1].exit_bar == -1:
            # should not happen — placeholder logic not used; skip
            pass

        # ── Gate 1: range filter ─────────────────────────────────────────────
        if o <= 0:
            i += 1
            continue
        rng = abs(c - o) / o * 10_000
        if rng < MIN_RANGE_BPS or rng > MAX_RANGE_BPS:
            i += 1
            continue

        # ── Gate 2: direction ─────────────────────────────────────────────────
        bias = "long" if c > o else "short"

        # ── Gate 3: forced alternation ────────────────────────────────────────
        if last_side == bias:
            bias = "short" if bias == "long" else "long"

        # ── Entry ─────────────────────────────────────────────────────────────
        entry_price = c
        entry_bar   = i

        lev      = _leverage(equity, sl_bps)
        margin   = equity * MARGIN_FRAC
        notional = margin * lev

        open_fee = notional * maker_rate
        equity  -= open_fee

        tp = entry_price * (1 + tp_bps / 10_000) if bias == "long" \
             else entry_price * (1 - tp_bps / 10_000)
        sl = entry_price * (1 - sl_bps / 10_000) if bias == "long" \
             else entry_price * (1 + sl_bps / 10_000)

        # ── Exit loop ─────────────────────────────────────────────────────────
        reason     : Optional[str]  = None
        exit_price : float          = 0.0
        maker_fill : bool           = False
        exit_bar   : int            = i

        for j in range(i + 1, n):
            h = highs[j]
            lo_bar = lows[j]
            ob = opens[j]
            cb = closes[j]
            bars_held = j - entry_bar

            tp_hit = (bias == "long"  and h  >= tp) or (bias == "short" and lo_bar <= tp)
            sl_hit = (bias == "long"  and lo_bar <= sl) or (bias == "short" and h  >= sl)

            if tp_hit and sl_hit:
                reason     = "sl_ambiguous"
                exit_price = sl
                maker_fill = False
                exit_bar   = j
                break

            if tp_hit:
                reason     = "tp"
                exit_price = tp
                # Maker-fill heuristic:  did price walk to TP or gap through?
                if bias == "long":
                    maker_fill = ob < tp   # bar opened below TP → walked up → MAKER
                else:
                    maker_fill = ob > tp   # bar opened above TP → walked down → MAKER
                exit_bar = j
                break

            if sl_hit:
                reason     = "sl"
                exit_price = sl
                maker_fill = False
                exit_bar   = j
                break

            # Trend-break early exit
            if TB_ENABLED and bars_held >= TB_MIN_BARS:
                adverse = ((entry_price - cb) / entry_price * 10_000) if bias == "long" \
                          else ((cb - entry_price) / entry_price * 10_000)
                if adverse >= TB_ADVERSE:
                    reason     = "trend_break"
                    exit_price = cb
                    maker_fill = False
                    exit_bar   = j
                    break

        if reason is None:
            # Hit end of dataset without resolution — skip this trade
            i += 1
            continue

        # ── P&L (close fee applied by caller per scenario) ───────────────────
        pnl_pct  = ((exit_price - entry_price) / entry_price) if bias == "long" \
                   else ((entry_price - exit_price) / entry_price)
        gross_pnl = pnl_pct * notional

        # For state tracking use TAKER close (conservative equity tracking)
        close_fee = notional * taker_rate
        equity   += gross_pnl - close_fee
        equity    = max(equity, 0.01)

        # ATR regime at entry bar
        a = atr_vals[entry_bar]
        if np.isnan(a):
            regime = "mid"
        elif a <= atr_p33:
            regime = "low"
        elif a <= atr_p67:
            regime = "mid"
        else:
            regime = "high"

        trades.append(Trade(
            entry_bar   = entry_bar,
            exit_bar    = exit_bar,
            bars_held   = exit_bar - entry_bar,
            side        = bias,
            entry_price = entry_price,
            exit_price  = exit_price,
            notional    = notional,
            reason      = reason,
            maker_fill  = maker_fill,
            gross_pnl   = gross_pnl,
            equity_after= equity,
            atr_regime  = regime,
        ))

        last_side = bias
        i = exit_bar + 1   # next entry only after this trade resolves

    return trades


# ═══════════════════════════════════════════════════════════════════════════════
def analyze(
    trades:      List[Trade],
    capital:     float,
    tp_bps:      float,
    sl_bps:      float,
    maker_rate:  float,
    taker_rate:  float,
    rebate_pct:  float,
    limit_tp:    bool,         # if True, TP exits use maker_fill flag for fee
) -> dict:
    """Compute summary stats for one TP value under one fee scenario."""
    if not trades:
        return {}

    total_gross   = 0.0
    total_fees    = 0.0
    wins = losses = maker_wins = taker_wins = 0
    bars_to_tp: List[int] = []
    by_regime: dict = {"low": {"w": 0, "l": 0}, "mid": {"w": 0, "l": 0}, "high": {"w": 0, "l": 0}}

    notional_0 = trades[0].notional  # first trade notional (for fee-per-$M scaling)

    for t in trades:
        # Close fee under this scenario
        if t.reason == "tp" and limit_tp and t.maker_fill:
            close_fee = t.notional * maker_rate
            maker_wins += 1
        else:
            close_fee = t.notional * taker_rate
            if t.reason == "tp":
                taker_wins += 1

        open_fee   = t.notional * maker_rate
        total_fee  = open_fee + close_fee
        net_pnl    = t.gross_pnl - total_fee

        total_gross += t.gross_pnl
        total_fees  += total_fee

        if net_pnl > 0:
            wins += 1
        else:
            losses += 1

        if t.reason == "tp":
            bars_to_tp.append(t.bars_held)

        reg = t.atr_regime
        if net_pnl > 0:
            by_regime[reg]["w"] += 1
        else:
            by_regime[reg]["l"] += 1

    n_trades = wins + losses
    wr       = wins / n_trades if n_trades else 0.0
    total_vol = sum(t.notional * 2 for t in trades)   # open + close

    rebate   = total_fees * rebate_pct
    true_net = total_gross - total_fees + rebate

    # Breakeven rebate: how much rebate_pct to make true_net = 0?
    # gross - fees + fees × breakeven_rebate = 0
    #   breakeven_rebate = (fees - gross) / fees   if fees > 0
    if total_fees > 0:
        be_rebate = max(0.0, (total_fees - total_gross) / total_fees)
    else:
        be_rebate = 0.0

    tp_hits   = sum(1 for t in trades if t.reason == "tp")
    maker_frac = maker_wins / tp_hits if tp_hits else 0.0

    btp_arr = np.array(bars_to_tp) if bars_to_tp else np.array([0])
    btp_med = float(np.median(btp_arr))
    btp_p75 = float(np.percentile(btp_arr, 75))
    btp_p90 = float(np.percentile(btp_arr, 90))
    btp_p1  = float(np.sum(btp_arr == 1)) / len(btp_arr) if len(btp_arr) else 0.0

    # Net per $1M volume (normalised)
    net_per_M = (true_net / total_vol * 1_000_000) if total_vol > 0 else 0.0

    return {
        "tp_bps":          tp_bps,
        "sl_bps":          sl_bps,
        "trades":          n_trades,
        "wins":            wins,
        "losses":          losses,
        "wr_pct":          wr * 100,
        "be_wr_pct":       sl_bps / (tp_bps + sl_bps) * 100,
        "wr_gap":          (wr - sl_bps / (tp_bps + sl_bps)) * 100,
        "tp_hits":         tp_hits,
        "maker_wins":      maker_wins,
        "taker_wins":      taker_wins,
        "maker_fill_pct":  maker_frac * 100,
        "gross_pnl":       total_gross,
        "total_fees":      total_fees,
        "rebate":          rebate,
        "true_net":        true_net,
        "net_per_M":       net_per_M,
        "total_volume":    total_vol,
        "be_rebate_pct":   be_rebate * 100,
        "by_regime":       by_regime,
        "bars_to_tp_med":  btp_med,
        "bars_to_tp_p75":  btp_p75,
        "bars_to_tp_p90":  btp_p90,
        "bars_hit_bar1_pct": btp_p1 * 100,
    }


# ═══════════════════════════════════════════════════════════════════════════════
def _pct_bar(v: float, w: int = 18) -> str:
    """ASCII bar chart cell for a percentage -100..+100."""
    clamped = max(-100.0, min(100.0, v))
    ratio   = abs(clamped) / 100.0
    filled  = int(ratio * w)
    ch      = "█" if v >= 0 else "░"
    return ch * filled + " " * (w - filled)


def _yn(v: bool) -> str:
    return "✅" if v else "❌"


def _sign(v: float) -> str:
    return f"+{v:.2f}" if v >= 0 else f"{v:.2f}"


# ═══════════════════════════════════════════════════════════════════════════════
def print_main_table(rows: list[dict], capital: float) -> None:
    HD = (
        f"{'TP':>6} {'SL':>5} {'Scenario':<22} {'Trades':>7} "
        f"{'WR%':>7} {'BE-WR%':>7} {'WRgap':>7} "
        f"{'Gross $':>9} {'Fees $':>8} {'Rebate':>8} "
        f"{'TrueNet$':>10} {'$/1M':>9} "
        f"{'BE-Reb%':>8} {'PROFIT?':>8}"
    )
    SEP = "─" * len(HD)
    print()
    print("═" * len(HD))
    print(f"  OKX VOLUME FARMER — TP SENSITIVITY  "
          f"(SL=50bps, capital=${capital:.0f}, trend-break enabled)")
    print(f"  Data: BTC/USDT 5m · 1 year (~May 2025 – Apr 2026)")
    print("═" * len(HD))
    print(HD)
    print(SEP)

    prev_tp = None
    for r in rows:
        if r["tp_bps"] != prev_tp and prev_tp is not None:
            print(SEP)
        prev_tp = r["tp_bps"]

        ok = r["true_net"] > 0
        row = (
            f"{r['tp_bps']:>5.0f}b"
            f" {r['sl_bps']:>4.0f}b"
            f" {r['_label']:<22}"
            f" {r['trades']:>7,}"
            f" {r['wr_pct']:>6.2f}%"
            f" {r['be_wr_pct']:>6.2f}%"
            f" {r['wr_gap']:>+6.2f}%"
            f" {r['gross_pnl']:>+9.2f}"
            f" {r['total_fees']:>8.2f}"
            f" {r['rebate']:>+8.2f}"
            f" {r['true_net']:>+10.2f}"
            f" {r['net_per_M']:>+9.2f}"
            f" {r['be_rebate_pct']:>7.1f}%"
            f"  {'✅ YES' if ok else '❌ NO':>8}"
        )
        print(row)

    print(SEP)


def print_maker_fill_table(rows: list[dict]) -> None:
    print()
    print("═" * 80)
    print("  MAKER-FILL ANALYSIS  (OKX limit-TP scenario)")
    print("  Maker fill = TP exit bar opened BELOW tp_price (price walked to limit order)")
    print("═" * 80)
    print(f"  {'TP':>6}  {'TP hits':>8}  {'Maker%':>8}  "
          f"{'Bar1%':>8}  {'Med bars':>9}  {'P75 bars':>9}  {'P90 bars':>9}  "
          f"{'FeeWin(mk)':>12}  {'FeeWin(tk)':>12}")
    print("  " + "─" * 76)

    seen = set()
    for r in rows:
        tp = r["tp_bps"]
        if tp in seen:
            continue
        seen.add(tp)
        # only use limit_tp scenario row (has maker_wins populated)
        fee_maker_win = r.get("_notional_sample", 150) * 0.0002 * 2   # maker open + maker close
        fee_taker_win = r.get("_notional_sample", 150) * 0.0002 + \
                        r.get("_notional_sample", 150) * 0.0005         # maker open + taker close
        print(
            f"  {tp:>5.0f}b"
            f"  {r['tp_hits']:>8,}"
            f"  {r['maker_fill_pct']:>7.1f}%"
            f"  {r['bars_hit_bar1_pct']:>7.1f}%"
            f"  {r['bars_to_tp_med']:>9.1f}"
            f"  {r['bars_to_tp_p75']:>9.1f}"
            f"  {r['bars_to_tp_p90']:>9.1f}"
        )


def print_regime_table(tp_regime_rows: dict) -> None:
    print()
    print("═" * 70)
    print("  WIN RATE BY VOLATILITY REGIME  (ATR-14 terciles)")
    print("═" * 70)
    print(f"  {'TP':>6}  {'Regime':>8}  {'Wins':>7}  {'Losses':>8}  {'WR%':>8}  "
          f"{'Trades':>8}")
    print("  " + "─" * 58)

    for tp_bps, regime_data in sorted(tp_regime_rows.items()):
        print(f"  {'─'*60}")
        for regime in ("low", "mid", "high"):
            w = regime_data[regime]["w"]
            l = regime_data[regime]["l"]
            t = w + l
            wr = w / t * 100 if t else 0
            label = {"low": "Low vol", "mid": "Med vol", "high": "High vol"}[regime]
            print(f"  {tp_bps:>5.0f}b  {label:>8}  {w:>7,}  {l:>8,}  {wr:>7.1f}%  {t:>8,}")


def print_breakeven_table(rows: list[dict]) -> None:
    """Show what rebate % is needed to break even at each TP × scenario."""
    print()
    print("═" * 65)
    print("  BREAKEVEN REBATE REQUIRED  (to make TrueNet = 0)")
    print("  → If your OKX rebate > this, the strategy is profitable")
    print("═" * 65)
    print(f"  {'TP':>6}  {'Scenario':<22}  {'BE Rebate%':>11}  {'Current':>9}  {'Gap':>8}")
    print("  " + "─" * 54)
    current = 40.0  # OKX current rebate
    seen: dict = {}
    for r in rows:
        key = (r["tp_bps"], r["_label"])
        if key in seen:
            continue
        seen[key] = True
        be  = r["be_rebate_pct"]
        gap = current - be
        ok  = "✅" if gap >= 0 else "❌"
        print(
            f"  {r['tp_bps']:>5.0f}b  {r['_label']:<22}  {be:>10.1f}%  "
            f"{current:>8.1f}%  {gap:>+7.1f}%  {ok}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="OKX advanced volume-farmer backtest")
    parser.add_argument("--tp-list", type=float, nargs="+",
                        default=[8, 9, 10, 11, 15, 18])
    parser.add_argument("--sl", type=float, default=50.0)
    parser.add_argument("--capital", type=float, default=30.0)
    parser.add_argument("--rows", type=int, default=None,
                        help="Use last N bars only (default: full dataset)")
    args = parser.parse_args()

    print(f"\nLoading {DATA} …")
    df = pd.read_parquet(DATA)
    df.columns = [c.lower() for c in df.columns]
    if args.rows:
        df = df.iloc[-args.rows:].reset_index(drop=True)
    print(f"Loaded {len(df):,} bars  "
          f"({df['open_time'].min()}  →  {df['open_time'].max()})\n")

    # ── ATR regime ─────────────────────────────────────────────────────────────
    atr_vals = _atr14(df)
    valid_atr = atr_vals[~np.isnan(atr_vals)]
    atr_p33   = float(np.percentile(valid_atr, 33))
    atr_p67   = float(np.percentile(valid_atr, 67))
    print(f"ATR-14 percentiles:  p33=${atr_p33:.1f}  p67=${atr_p67:.1f}\n")

    capital   = args.capital
    sl_bps    = args.sl
    tp_list   = args.tp_list

    all_rows: list[dict]    = []
    regime_rows: dict       = {}
    maker_fill_rows: list   = []

    # ── Run simulations ────────────────────────────────────────────────────────
    for tp_bps in tp_list:
        print(f"  Running TP={tp_bps:.0f}bps / SL={sl_bps:.0f}bps …", end="", flush=True)

        # We run the simulation once (using OKX maker rate for open-fee equity tracking)
        trades = simulate(
            df         = df,
            tp_bps     = tp_bps,
            sl_bps     = sl_bps,
            capital    = capital,
            maker_rate = 0.0002,   # OKX
            taker_rate = 0.0005,
            atr_vals   = atr_vals,
            atr_p33    = atr_p33,
            atr_p67    = atr_p67,
        )

        notional_sample = trades[0].notional if trades else capital * 5

        for sc_key, sc in SCENARIOS.items():
            limit_tp = (sc_key == "okx_limit_tp")
            r = analyze(
                trades      = trades,
                capital     = capital,
                tp_bps      = tp_bps,
                sl_bps      = sl_bps,
                maker_rate  = sc["maker"],
                taker_rate  = sc["taker"],
                rebate_pct  = sc["rebate"],
                limit_tp    = limit_tp,
            )
            r["_label"]           = sc["label"]
            r["_scenario"]        = sc_key
            r["_notional_sample"] = notional_sample
            all_rows.append(r)

            if sc_key == "okx_limit_tp":
                maker_fill_rows.append(r)

        # Regime breakdown (from okx_limit_tp)
        limit_row = next(r for r in all_rows if r["tp_bps"] == tp_bps
                         and r["_scenario"] == "okx_limit_tp")
        regime_rows[tp_bps] = limit_row["by_regime"]

        wr = limit_row["wr_pct"]
        net = limit_row["true_net"]
        print(f"  {len(trades):,} trades  WR={wr:.1f}%  net={_sign(net)}")

    # ── Tables ─────────────────────────────────────────────────────────────────
    print_main_table(all_rows, capital)
    print_maker_fill_table(maker_fill_rows)
    print_regime_table(regime_rows)
    print_breakeven_table(all_rows)

    # ── Optimal TP summary ─────────────────────────────────────────────────────
    print()
    print("═" * 65)
    print("  OPTIMAL TP SUMMARY  (OKX limit-TP scenario, 40% rebate)")
    print("═" * 65)
    ltp_rows = [r for r in all_rows if r["_scenario"] == "okx_limit_tp"]
    best     = max(ltp_rows, key=lambda r: r["true_net"])
    berow    = min(ltp_rows, key=lambda r: r["be_rebate_pct"])
    print(f"  Best net P&L:      TP={best['tp_bps']:.0f}bps  "
          f"→ TrueNet={_sign(best['true_net'])}  WR={best['wr_pct']:.1f}%")
    print(f"  Lowest BE rebate:  TP={berow['tp_bps']:.0f}bps  "
          f"→ need {berow['be_rebate_pct']:.1f}% rebate  "
          f"(gap {40.0 - berow['be_rebate_pct']:+.1f}%)")
    print()

    print("  MEXC vs OKX per-$1M cost (limit-TP path):")
    mexc_rows = [r for r in all_rows if r["_scenario"] == "mexc_baseline"]
    for tp_bps in tp_list:
        m = next((r for r in mexc_rows if r["tp_bps"] == tp_bps), None)
        o = next((r for r in ltp_rows  if r["tp_bps"] == tp_bps), None)
        if m and o:
            diff = o["net_per_M"] - m["net_per_M"]
            print(f"    TP={tp_bps:.0f}bps  MEXC ${m['net_per_M']:+.2f}/M  "
                  f"OKX ${o['net_per_M']:+.2f}/M  "
                  f"delta ${diff:+.2f}/M")

    print()
    print("  NOTE: 'limit-TP' assumes OKX TP-limit order fills as MAKER when")
    print("        exit bar opens below tp_price (valid for ~90%+ of 5m TP hits).")
    print("        Verify maker_fill_pct above — if >80%, scenario S2 is reliable.")
    print()


if __name__ == "__main__":
    main()
