"""Which UTC hours lose? — proven by 360d backtest + out-of-sample validation.

Uses the chosen live config (TP 0.5xATR / SL 1.5xATR, max_hold=1 bar). Computes
per-trade net bps by entry hour, then VALIDATES: rank hours on the first half of
the data, block the worst, and measure the lift on the *second* half (out of
sample) — so we're not just cherry-picking the worst hours in hindsight.

Usage: python scripts/analyze_bad_hours.py [--max-hold 1] [--sl-slippage-bps 5] [--block N]
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data/historical/BTC_USDT_5m.parquet"

MAKER_BPS, TAKER_BPS, REBATE = 2.0, 5.0, 0.40
MIN_RANGE_BPS, MAX_RANGE_BPS = 3.0, 40.0
TP_MULT, SL_MULT = 0.5, 1.5
TP_BPS_MIN, SL_BPS_MIN = 5.0, 8.0
ATR_MIN_USD = 120.0


def _atr14(df):
    hi, lo, cl = (df[c].to_numpy(float) for c in ("high", "low", "close"))
    pcl = np.concatenate([[cl[0]], cl[:-1]])
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - pcl), np.abs(lo - pcl)))
    atr = np.full(len(tr), np.nan); atr[13] = tr[:14].mean()
    for k in range(14, len(tr)):
        atr[k] = atr[k - 1] * (13 / 14) + tr[k] * (1 / 14)
    return atr


def run(df, max_hold, slip):
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    hour = pd.DatetimeIndex(df["open_time"]).hour.to_numpy()
    atr = _atr14(df); n = len(df); rng = np.abs(c - o) / o * 1e4
    nets, hrs = [], []
    fee_reb = lambda close_bps: REBATE * (MAKER_BPS + close_bps)
    i = 14
    while i < n - 1:
        a = atr[i]
        if np.isnan(a) or a < ATR_MIN_USD or not (MIN_RANGE_BPS <= rng[i] <= MAX_RANGE_BPS):
            i += 1; continue
        side = 1 if c[i] > o[i] else -1
        ab = a / o[i] * 1e4
        tpb = max(TP_MULT * ab, TP_BPS_MIN); slb = max(SL_MULT * ab, SL_BPS_MIN)
        entry = c[i]
        tp = entry * (1 + tpb / 1e4) if side == 1 else entry * (1 - tpb / 1e4)
        sl = entry * (1 - slb / 1e4) if side == 1 else entry * (1 + slb / 1e4)
        reason, net, exitb = None, None, i
        for j in range(i + 1, min(i + max_hold, n - 1) + 1):
            tp_hit = (h[j] >= tp) if side == 1 else (lo[j] <= tp)
            sl_hit = (lo[j] <= sl) if side == 1 else (h[j] >= sl)
            if sl_hit and tp_hit:
                net = -slb - slip - MAKER_BPS - TAKER_BPS + fee_reb(TAKER_BPS); exitb = j; break
            if tp_hit:
                net = tpb - MAKER_BPS - MAKER_BPS + fee_reb(MAKER_BPS); exitb = j; break
            if sl_hit:
                net = -slb - slip - MAKER_BPS - TAKER_BPS + fee_reb(TAKER_BPS); exitb = j; break
            if j - i >= max_hold:
                fav = ((c[j] - entry) if side == 1 else (entry - c[j])) / entry * 1e4
                net = fav - MAKER_BPS - MAKER_BPS + fee_reb(MAKER_BPS); exitb = j; break
        if net is None:
            i += 1; continue
        nets.append(net); hrs.append(int(hour[i])); i = exitb + 1
    return np.array(nets), np.array(hrs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hold", type=int, default=1)
    ap.add_argument("--sl-slippage-bps", type=float, default=5.0)
    ap.add_argument("--block", type=int, default=8, help="how many worst hours to block in the OOS test")
    args = ap.parse_args()

    df = pd.read_parquet(DATA); df.columns = [c.lower() for c in df.columns]
    nets, hrs = run(df, args.max_hold, args.sl_slippage_bps)
    t = pd.to_datetime(df["open_time"]); span = (t.max() - t.min()).days
    print(f"Data 5m {t.min().date()}→{t.max().date()} ({span}d) | max_hold={args.max_hold} | SL-slip={args.sl_slippage_bps}bps")
    print(f"Trades: {len(nets):,}   overall net: {nets.mean():+.2f} bps/trade\n")

    print("="*70)
    print("  BY UTC HOUR  (full sample)")
    print("="*70)
    print(f"  {'hr':>3} {'trades':>7} {'net bps/t':>10} {'win%':>6} {'total bleed bps':>16}")
    print("  " + "-"*64)
    order = []
    for hh in range(24):
        m = hrs == hh; nn = m.sum()
        if not nn: continue
        nb = nets[m].mean(); wr = (nets[m] > 0).mean() * 100; tot = nets[m].sum()
        order.append((hh, nb, nn, tot))
        flag = "  ◀ losing" if nb < nets.mean() else ""
        print(f"  {hh:>3} {nn:>7,} {nb:>+10.2f} {wr:>5.1f}% {tot:>+16,.0f}{flag}")
    print("  " + "-"*64)
    worst = sorted(order, key=lambda x: x[1])[:args.block]
    print(f"  Worst {args.block} hours (full-sample): {sorted(h for h,_,_,_ in worst)}")

    # ── OUT-OF-SAMPLE VALIDATION ────────────────────────────────────────────
    print("\n" + "="*70)
    print("  OUT-OF-SAMPLE TEST  — rank hours on 1st half, block worst, score 2nd half")
    print("="*70)
    mid = len(df) // 2
    nets_tr, hrs_tr = run(df.iloc[:mid].reset_index(drop=True), args.max_hold, args.sl_slippage_bps)
    nets_te, hrs_te = run(df.iloc[mid:].reset_index(drop=True), args.max_hold, args.sl_slippage_bps)
    # rank on train
    train_by_hr = {hh: nets_tr[hrs_tr == hh].mean() for hh in range(24) if (hrs_tr == hh).any()}
    bad = sorted(train_by_hr, key=lambda hh: train_by_hr[hh])[:args.block]
    bad = sorted(bad)
    keep_te = ~np.isin(hrs_te, bad)
    print(f"  Bad hours (chosen on 1st-half only):  {bad}")
    print(f"  2nd-half net, ALL hours:              {nets_te.mean():+.2f} bps/t   ({len(nets_te):,} trades)")
    print(f"  2nd-half net, EXCLUDING bad hours:    {nets_te[keep_te].mean():+.2f} bps/t   "
          f"({keep_te.sum():,} trades, dropped {(~keep_te).sum():,})")
    lift = nets_te[keep_te].mean() - nets_te.mean()
    print(f"  Out-of-sample lift from the filter:   {lift:+.2f} bps/t")
    verdict = ("REAL — filter helps out-of-sample" if lift > 0.05 else
               "NOISE — no out-of-sample benefit (don't filter)" if lift < 0.05 else "marginal")
    print(f"  Verdict: {verdict}")
    if nets_te[keep_te].mean() > 0:
        print("  NOTE: even the kept hours are net POSITIVE out-of-sample — worth a closer look.")
    else:
        print("  NOTE: kept hours are still net NEGATIVE — filter reduces the bleed but doesn't create edge.")


if __name__ == "__main__":
    main()
