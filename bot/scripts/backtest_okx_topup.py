"""
OKX Fix5 Backtest — $500 initial capital with $250 top-up reserve.

Fix5 config:
  • No alternation (pure momentum signal)
  • ATR-relative TP/SL  (TP = 0.5×ATR_bps, SL = 1.5×ATR_bps)
  • High-vol-only session filter  (ATR ≥ p67)
  • Limit-TP: TP exits fill as MAKER (99.9% fill rate confirmed)
  • OKX fees: maker=0.020%, taker=0.050%, rebate=40%

Capital mechanics:
  • Start: $500 equity, $250 reserve pool
  • Top-up rule: after any trade that brings equity below TOPUP_FLOOR ($500),
    inject min(TOPUP_FLOOR - equity, remaining_reserve) from reserve.
  • Once reserve is exhausted, no further injections → equity can keep falling.
  • Max total at risk: $750 ($500 + $250 reserve).

Reports:
  1. Trade-by-trade equity curve (first 30 + last 10 rows)
  2. Top-up event log
  3. Summary stats: WR, net P&L, volume, days to $100k
  4. Comparison: same Fix5 without top-up at $500

Usage:
    cd /Users/rowneth/vGen/bot
    python scripts/backtest_okx_topup.py
    python scripts/backtest_okx_topup.py --topup-floor 400 --reserve 250
"""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data/historical/BTC_USDT_5m.parquet"

# ── OKX fees ──────────────────────────────────────────────────────────────────
OKX_MAKER  = 0.0002   # 0.020%
OKX_TAKER  = 0.0005   # 0.050%
OKX_REBATE = 0.40

# ── Sizing ────────────────────────────────────────────────────────────────────
MARGIN_FRAC = 0.05
RISK_PCT    = 0.025
MAX_LEV     = 125.0
MIN_LEV     = 5.0

# ── Entry gates ───────────────────────────────────────────────────────────────
MIN_RANGE_BPS = 3.0
MAX_RANGE_BPS = 40.0

# ── ATR-relative multipliers ──────────────────────────────────────────────────
ATR_TP_MULT   = 0.5
ATR_SL_MULT   = 1.5
ATR_TP_MIN    = 5.0   # bps floor
ATR_SL_MIN    = 8.0   # bps floor


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TradeRecord:
    idx:          int
    side:         str
    entry_bar:    int
    exit_bar:     int
    bars_held:    int
    entry_price:  float
    exit_price:   float
    tp_bps:       float
    sl_bps:       float
    notional:     float
    open_fee:     float
    close_fee:    float
    gross_pnl:    float
    net_pnl:      float
    reason:       str
    maker_fill:   bool
    equity_before: float
    equity_after:  float
    injected:      float   # top-up added after this trade
    reserve_after: float


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
    df:           pd.DataFrame,
    atr_vals:     np.ndarray,
    atr_p67:      float,
    initial_cap:  float,
    reserve:      float,
    topup_floor:  float,
    limit_tp:     bool = True,
) -> List[TradeRecord]:
    """
    Bar-by-bar Fix5 simulation with optional top-up reserve.

    Top-up rule: after any trade that leaves equity < topup_floor AND reserve > 0,
    inject min(topup_floor - equity, reserve) from the reserve pool.
    """
    opens  = df["open"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    n      = len(df)

    equity        = initial_cap
    reserve_left  = reserve
    last_side     = ""
    records: List[TradeRecord] = []
    trade_idx = 0

    i = 14  # ATR warm-up
    while i < n - 1:
        o = opens[i]
        c = closes[i]

        if o <= 0 or equity <= 0:
            i += 1
            continue

        # ATR session filter (high-vol-only: ATR ≥ p67)
        atr_now = atr_vals[i]
        if np.isnan(atr_now) or atr_now < atr_p67:
            i += 1
            continue

        # Bar range gate
        rng_bps = abs(c - o) / o * 10_000
        if rng_bps < MIN_RANGE_BPS or rng_bps > MAX_RANGE_BPS:
            i += 1
            continue

        # ATR-relative TP/SL
        atr_bps = atr_now / o * 10_000
        tp_bps  = max(ATR_TP_MIN, ATR_TP_MULT * atr_bps)
        sl_bps  = max(ATR_SL_MIN, ATR_SL_MULT * atr_bps)

        # Direction: pure momentum (no alternation)
        bias = "long" if c > o else "short"

        # Entry at close of signal bar
        entry_price = c
        entry_bar   = i

        lev      = _leverage(equity, sl_bps)
        margin   = equity * MARGIN_FRAC
        notional = margin * lev
        open_fee = notional * OKX_MAKER
        equity_before = equity
        equity  -= open_fee

        tp = entry_price * (1 + tp_bps / 10_000) if bias == "long" \
             else entry_price * (1 - tp_bps / 10_000)
        sl = entry_price * (1 - sl_bps / 10_000) if bias == "long" \
             else entry_price * (1 + sl_bps / 10_000)

        # Exit scan
        reason:     Optional[str] = None
        exit_price: float         = 0.0
        maker_fill: bool          = False
        exit_bar:   int           = i

        for j in range(i + 1, n):
            h  = highs[j]
            l  = lows[j]
            ob = opens[j]

            tp_hit = (bias == "long"  and h >= tp) or (bias == "short" and l <= tp)
            sl_hit = (bias == "long"  and l <= sl) or (bias == "short" and h >= sl)

            if tp_hit and sl_hit:
                reason = "sl_ambiguous"; exit_price = sl; exit_bar = j; break
            if tp_hit:
                reason = "tp"; exit_price = tp; exit_bar = j
                maker_fill = (ob < tp) if bias == "long" else (ob > tp)
                break
            if sl_hit:
                reason = "sl"; exit_price = sl; exit_bar = j; break

        if reason is None:
            i += 1
            continue

        # P&L
        pnl_pct   = (exit_price - entry_price) / entry_price if bias == "long" \
                    else (entry_price - exit_price) / entry_price
        gross_pnl = pnl_pct * notional

        if limit_tp and reason == "tp" and maker_fill:
            close_fee = notional * OKX_MAKER
        else:
            close_fee = notional * OKX_TAKER

        net_pnl = gross_pnl - close_fee
        equity += gross_pnl - close_fee
        equity  = max(equity, 0.0)

        # Top-up from reserve if equity fell below floor
        injected = 0.0
        if equity < topup_floor and reserve_left > 0:
            inject       = min(topup_floor - equity, reserve_left)
            equity      += inject
            reserve_left -= inject
            injected     = inject

        records.append(TradeRecord(
            idx           = trade_idx,
            side          = bias,
            entry_bar     = entry_bar,
            exit_bar      = exit_bar,
            bars_held     = exit_bar - entry_bar,
            entry_price   = entry_price,
            exit_price    = exit_price,
            tp_bps        = tp_bps,
            sl_bps        = sl_bps,
            notional      = notional,
            open_fee      = open_fee,
            close_fee     = close_fee,
            gross_pnl     = gross_pnl,
            net_pnl       = net_pnl,
            reason        = reason,
            maker_fill    = maker_fill,
            equity_before = equity_before,
            equity_after  = equity,
            injected      = injected,
            reserve_after = reserve_left,
        ))

        trade_idx += 1
        last_side  = bias
        i          = exit_bar + 1

    return records


# ─────────────────────────────────────────────────────────────────────────────
def analyze(records: List[TradeRecord], initial_cap: float, reserve: float,
            n_days: float, label: str) -> dict:
    if not records:
        return {"_label": label, "_empty": True}

    wins = losses = 0
    total_gross = total_fees = total_vol = 0.0
    total_injected = 0.0
    injection_events = 0
    tp_maker = tp_taker = 0

    for r in records:
        fee = r.open_fee + r.close_fee
        total_gross += r.gross_pnl
        total_fees  += fee
        total_vol   += r.notional * 2
        if r.net_pnl > 0:
            wins += 1
        else:
            losses += 1
        if r.reason == "tp":
            if r.maker_fill:
                tp_maker += 1
            else:
                tp_taker += 1
        if r.injected > 0:
            total_injected   += r.injected
            injection_events += 1

    n = wins + losses
    wr = wins / n if n else 0.0

    avg_tp = float(np.mean([r.tp_bps for r in records]))
    avg_sl = float(np.mean([r.sl_bps for r in records]))
    be_wr  = avg_sl / (avg_tp + avg_sl) if (avg_tp + avg_sl) > 0 else 0.5

    rebate   = total_fees * OKX_REBATE
    true_net = total_gross - total_fees + rebate

    be_rebate = max(0.0, (total_fees - total_gross) / total_fees) if total_fees > 0 else 0.0

    vol_per_day = total_vol / n_days if n_days > 0 else 0.0
    days_100k   = 100_000 / vol_per_day if vol_per_day > 0 else 9999

    final_equity  = records[-1].equity_after
    peak_equity   = max(r.equity_after for r in records)
    trough_equity = min(r.equity_after for r in records)
    max_dd        = (peak_equity - trough_equity) / peak_equity * 100 if peak_equity > 0 else 0.0
    total_deployed = initial_cap + total_injected

    tp_hits     = sum(1 for r in records if r.reason == "tp")
    maker_pct   = tp_maker / tp_hits * 100 if tp_hits > 0 else 0.0

    return {
        "_label":           label,
        "trades":           n,
        "wins":             wins,
        "losses":           losses,
        "wr_pct":           wr * 100,
        "be_wr_pct":        be_wr * 100,
        "wr_gap":           (wr - be_wr) * 100,
        "avg_tp_bps":       avg_tp,
        "avg_sl_bps":       avg_sl,
        "gross_pnl":        total_gross,
        "total_fees":       total_fees,
        "rebate":           rebate,
        "true_net":         true_net,
        "be_rebate_pct":    be_rebate * 100,
        "total_vol":        total_vol,
        "vol_per_day":      vol_per_day,
        "days_100k":        days_100k,
        "final_equity":     final_equity,
        "peak_equity":      peak_equity,
        "trough_equity":    trough_equity,
        "max_dd_pct":       max_dd,
        "initial_cap":      initial_cap,
        "reserve_budgeted": reserve,
        "total_injected":   total_injected,
        "injection_events": injection_events,
        "total_deployed":   total_deployed,
        "pnl_on_deployed":  true_net / total_deployed * 100 if total_deployed > 0 else 0.0,
        "tp_hits":          tp_hits,
        "maker_fill_pct":   maker_pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
def print_equity_curve(records: List[TradeRecord], head: int = 30, tail: int = 10) -> None:
    print()
    print("═" * 100)
    print("  EQUITY CURVE  (first {:d} trades + last {:d} trades)".format(head, tail))
    print("═" * 100)
    hdr = (f"  {'#':>5}  {'Side':<6}  {'Reason':<13}  {'Notional':>10}  "
           f"{'GrossPnL':>10}  {'NetPnL':>10}  {'Equity':>9}  "
           f"{'Inject':>7}  {'Reserve':>8}  {'TP':>5}  {'SL':>5}")
    print(hdr)
    print("  " + "─" * 93)

    def _row(r: TradeRecord) -> str:
        inj_str = f"+{r.injected:.2f}" if r.injected > 0 else "—"
        return (
            f"  {r.idx:>5}  {r.side:<6}  {r.reason:<13}  "
            f"{r.notional:>10,.2f}  "
            f"{r.gross_pnl:>+10.4f}  "
            f"{r.net_pnl:>+10.4f}  "
            f"{r.equity_after:>9.4f}  "
            f"{inj_str:>7}  "
            f"{r.reserve_after:>8.4f}  "
            f"{r.tp_bps:>5.1f}  "
            f"{r.sl_bps:>5.1f}"
        )

    n = len(records)
    shown_head = min(head, n)
    for r in records[:shown_head]:
        print(_row(r))

    if n > head + tail:
        print(f"  {'...':>5}  ({n - head - tail:,} trades omitted)")

    for r in records[max(shown_head, n - tail):]:
        print(_row(r))

    print("  " + "═" * 93)


def print_injection_log(records: List[TradeRecord]) -> None:
    inj_records = [r for r in records if r.injected > 0]
    print()
    print("═" * 72)
    print(f"  TOP-UP INJECTION LOG  ({len(inj_records)} events)")
    print("═" * 72)
    if not inj_records:
        print("  No injections made — reserve untouched.")
    else:
        print(f"  {'Trade#':>7}  {'Side':<6}  {'Reason':<13}  "
              f"{'Equity@close':>13}  {'Injected':>9}  {'Reserve rem.':>13}")
        print("  " + "─" * 64)
        for r in inj_records:
            equity_pre_inj = r.equity_after - r.injected
            print(
                f"  {r.idx:>7}  {r.side:<6}  {r.reason:<13}  "
                f"{equity_pre_inj:>13.4f}  "
                f"+{r.injected:>8.4f}  "
                f"{r.reserve_after:>13.4f}"
            )
    print("  " + "═" * 64)


def print_summary(stats: List[dict]) -> None:
    print()
    print("═" * 110)
    print("  FIX5 SUMMARY  |  OKX maker=0.020% taker=0.050% rebate=40%  |  ATR-relative TP/SL + high-vol-only + limit-TP")
    print("═" * 110)
    hdr = (
        f"  {'Scenario':<30}  {'Trades':>7}  {'WR%':>7}  {'BE-WR%':>7}  "
        f"{'Gross$':>8}  {'Fees$':>7}  {'Rebate$':>8}  {'Net$':>8}  "
        f"{'Deployed$':>10}  {'NetOnDep%':>10}  {'Days→100k':>10}  {'FinalEq$':>10}"
    )
    print(hdr)
    print("  " + "─" * 103)

    for s in stats:
        if s.get("_empty"):
            continue
        print(
            f"  {s['_label']:<30}  "
            f"{s['trades']:>7,}  "
            f"{s['wr_pct']:>6.1f}%  "
            f"{s['be_wr_pct']:>6.1f}%  "
            f"{s['gross_pnl']:>+8.2f}  "
            f"{s['total_fees']:>7.2f}  "
            f"+{s['rebate']:>7.2f}  "
            f"{s['true_net']:>+8.2f}  "
            f"{s['total_deployed']:>10.2f}  "
            f"{s['pnl_on_deployed']:>+9.2f}%  "
            f"{s['days_100k']:>10.1f}  "
            f"{s['final_equity']:>10.4f}"
        )

    print("  " + "═" * 103)


def print_drawdown_table(stats: List[dict]) -> None:
    print()
    print("═" * 80)
    print("  DRAWDOWN & CAPITAL RISK")
    print("═" * 80)
    print(f"  {'Scenario':<30}  {'InitCap$':>9}  {'Reserve$':>9}  "
          f"{'Injected$':>10}  {'PeakEq$':>9}  {'TroughEq$':>10}  {'MaxDD%':>7}")
    print("  " + "─" * 72)
    for s in stats:
        if s.get("_empty"):
            continue
        print(
            f"  {s['_label']:<30}  "
            f"{s['initial_cap']:>9.2f}  "
            f"{s['reserve_budgeted']:>9.2f}  "
            f"{s['total_injected']:>10.4f}  "
            f"{s['peak_equity']:>9.4f}  "
            f"{s['trough_equity']:>10.4f}  "
            f"{s['max_dd_pct']:>6.1f}%"
        )
    print("  " + "═" * 72)


def print_volume_table(stats: List[dict]) -> None:
    print()
    print("═" * 75)
    print("  VOLUME THROUGHPUT")
    print("═" * 75)
    print(f"  {'Scenario':<30}  {'Trades/yr':>10}  {'Vol$/yr':>12}  "
          f"{'Vol$/mo':>10}  {'Days→$100k':>11}")
    print("  " + "─" * 67)
    for s in stats:
        if s.get("_empty"):
            continue
        print(
            f"  {s['_label']:<30}  "
            f"{s['trades']:>10,}  "
            f"{s['total_vol']:>12,.0f}  "
            f"{s['total_vol']/12:>10,.0f}  "
            f"{s['days_100k']:>11.1f}"
        )
    print("  " + "═" * 67)


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital",      type=float, default=500.0,  help="Initial capital USD")
    parser.add_argument("--reserve",      type=float, default=250.0,  help="Top-up reserve USD")
    parser.add_argument("--topup-floor",  type=float, default=None,   help="Top-up when equity < X (default = initial capital)")
    parser.add_argument("--rows",         type=int,   default=None,   help="Use last N bars only")
    parser.add_argument("--no-curve",     action="store_true",        help="Skip equity curve print")
    args = parser.parse_args()

    topup_floor = args.topup_floor if args.topup_floor is not None else args.capital

    print(f"\nLoading {DATA} …")
    df = pd.read_parquet(DATA)
    df.columns = [c.lower() for c in df.columns]
    if args.rows:
        df = df.iloc[-args.rows:].reset_index(drop=True)
    t0 = pd.to_datetime(df["open_time"].iloc[0],  utc=True)
    t1 = pd.to_datetime(df["open_time"].iloc[-1], utc=True)
    n_days = (t1 - t0).total_seconds() / 86_400
    print(f"Loaded {len(df):,} bars  ({t0}  →  {t1})")
    print(f"Dataset span: {n_days:.1f} days\n")

    # ATR
    atr_vals = _atr14(df)
    valid    = atr_vals[~np.isnan(atr_vals)]
    atr_p33  = float(np.percentile(valid, 33))
    atr_p67  = float(np.percentile(valid, 67))
    print(f"ATR-14:  p33=${atr_p33:.1f}  p67=${atr_p67:.1f}\n")

    print(f"Capital config:")
    print(f"  Initial capital : ${args.capital:.2f}")
    print(f"  Reserve pool    : ${args.reserve:.2f}  (top-up when equity < ${topup_floor:.2f})")
    print(f"  Max at risk     : ${args.capital + args.reserve:.2f}\n")

    # ── Scenario A: with top-up reserve ──────────────────────────────────────
    print("Running Fix5 WITH top-up reserve …")
    rec_topup = simulate(
        df, atr_vals, atr_p67,
        initial_cap=args.capital,
        reserve=args.reserve,
        topup_floor=topup_floor,
        limit_tp=True,
    )
    label_a = f"Fix5 ${args.capital:.0f}+${args.reserve:.0f} reserve"
    stats_a = analyze(rec_topup, args.capital, args.reserve, n_days, label_a)
    print(f"  {len(rec_topup):,} trades  WR={stats_a['wr_pct']:.1f}%  "
          f"net={stats_a['true_net']:+.2f}  injected=${stats_a['total_injected']:.4f}")

    # ── Scenario B: no top-up, same $500 ─────────────────────────────────────
    print("Running Fix5 NO top-up (pure $500) …")
    rec_notu = simulate(
        df, atr_vals, atr_p67,
        initial_cap=args.capital,
        reserve=0.0,
        topup_floor=args.capital,
        limit_tp=True,
    )
    label_b = f"Fix5 ${args.capital:.0f} no reserve"
    stats_b = analyze(rec_notu, args.capital, 0.0, n_days, label_b)
    print(f"  {len(rec_notu):,} trades  WR={stats_b['wr_pct']:.1f}%  "
          f"net={stats_b['true_net']:+.2f}  injected=$0")

    # ── Scenario C: $30 original config (reference) ───────────────────────────
    print("Running Fix5 $30 reference (original capital) …")
    rec_30   = simulate(
        df, atr_vals, atr_p67,
        initial_cap=30.0,
        reserve=0.0,
        topup_floor=30.0,
        limit_tp=True,
    )
    stats_c  = analyze(rec_30, 30.0, 0.0, n_days, "Fix5 $30 (reference)")
    print(f"  {len(rec_30):,} trades  WR={stats_c['wr_pct']:.1f}%  "
          f"net={stats_c['true_net']:+.2f}")

    all_stats = [stats_a, stats_b, stats_c]

    # ── Output ────────────────────────────────────────────────────────────────
    print_summary(all_stats)
    print_drawdown_table(all_stats)
    print_volume_table(all_stats)

    if not args.no_curve:
        print_equity_curve(rec_topup, head=40, tail=15)

    print_injection_log(rec_topup)

    # ── Capital efficiency note ───────────────────────────────────────────────
    print()
    print("═" * 72)
    print("  CAPITAL SCALING NOTE")
    print("═" * 72)
    s_ref = stats_c
    s_500 = stats_a
    vol_ratio = s_500["total_vol"] / s_ref["total_vol"] if s_ref["total_vol"] > 0 else 0
    print(f"  $30 → ${args.capital:.0f} capital multiplier   : {args.capital/30:.1f}×")
    print(f"  Volume ratio ($500 / $30)      : {vol_ratio:.1f}×  (notional scales with equity)")
    print(f"  Days to $100k at $30           : {s_ref['days_100k']:.1f}")
    print(f"  Days to $100k at ${args.capital:.0f}           : {s_500['days_100k']:.1f}")
    print(f"  Volume boost (exact, with DD)  : {vol_ratio:.2f}×")
    print()
    print("  NOTE: Volume doesn't scale linearly with capital because:")
    print("    1. Equity-weighted sizing: notional = equity × 5% × leverage")
    print("    2. Losses reduce equity mid-run, compressing notional in later trades")
    print("    3. Top-up injections restore equity → partially recovers notional")
    print(f"  Breakeven rebate required: {s_500['be_rebate_pct']:.1f}%  (OKX offers 40%)")
    print("  " + "═" * 68)


if __name__ == "__main__":
    main()
