"""When does holding stop paying? — data-proven max_hold for the live demo-v2 entry.

Single forward-walk per entry on 5m BTC (live config: micro_momentum entry,
TP=0.5xATR, SL=1.5xATR, same gates). For each trade we record the exact bar TP
or SL is touched, and the time-stop close P&L at every bar 1..HMAX. From that one
pass we can evaluate EVERY max_hold without re-simulating, plus a per-bar hazard:
how many positions resolve as TP vs SL at each held bar, and the marginal edge of
holding one more bar.

Economics are reported in bps of notional (size-free, so a negative edge can't
spiral and corrupt the average). Fees: open=maker 2bps; TP/time_stop close=maker
2bps; SL close=taker 5bps + slippage. Rebate = 40% of fees, added per trade.

Usage: python scripts/analyze_hold_horizon.py [--sl-slippage-bps 5]
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
HMAX = 24                      # record time-stop P&L out to 24 bars (2h on 5m)


def _atr14(df):
    hi, lo, cl = (df[c].to_numpy(float) for c in ("high", "low", "close"))
    pcl = np.concatenate([[cl[0]], cl[:-1]])
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - pcl), np.abs(lo - pcl)))
    atr = np.full(len(tr), np.nan)
    atr[13] = tr[:14].mean()
    for k in range(14, len(tr)):
        atr[k] = atr[k - 1] * (13 / 14) + tr[k] * (1 / 14)
    return atr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sl-slippage-bps", type=float, default=5.0)
    args = ap.parse_args()
    slip = args.sl_slippage_bps

    df = pd.read_parquet(DATA); df.columns = [c.lower() for c in df.columns]
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    atr = _atr14(df); n = len(df)
    rng = np.abs(c - o) / o * 1e4
    t0, t1 = pd.to_datetime(df["open_time"]).min(), pd.to_datetime(df["open_time"]).max()
    print(f"Data: {n:,} bars  {t0.date()} → {t1.date()}  ({(t1-t0).days}d, 5m)  SL-slip={slip}bps")
    print(f"Entry: micro_momentum  TP=0.5xATR  SL=1.5xATR  (live demo-v2)\n")

    # per-trade records
    tp_bar = []      # bar index (1-based, relative) of TP touch, or 9999
    sl_bar = []      # bar index of SL touch, or 9999
    tp_bps_l = []    # this trade's TP band in bps
    sl_bps_l = []
    ts_favor = []    # favor-bps at time-stop close for bars 1..HMAX (np array per trade)

    i = 14
    while i < n - 1:
        a = atr[i]
        if np.isnan(a) or a < ATR_MIN_USD or not (MIN_RANGE_BPS <= rng[i] <= MAX_RANGE_BPS):
            i += 1; continue
        side = 1 if c[i] > o[i] else -1
        atr_bps = a / o[i] * 1e4
        tpb = max(TP_MULT * atr_bps, TP_BPS_MIN)
        slb = max(SL_MULT * atr_bps, SL_BPS_MIN)
        entry = c[i]
        tp = entry * (1 + tpb / 1e4) if side == 1 else entry * (1 - tpb / 1e4)
        sl = entry * (1 - slb / 1e4) if side == 1 else entry * (1 + slb / 1e4)

        t_tp = t_sl = 9999
        favor = np.full(HMAX + 1, np.nan)
        last = min(i + HMAX, n - 1)
        for j in range(i + 1, last + 1):
            k = j - i
            tp_hit = (h[j] >= tp) if side == 1 else (lo[j] <= tp)
            sl_hit = (lo[j] <= sl) if side == 1 else (h[j] >= sl)
            favor[k] = ((c[j] - entry) if side == 1 else (entry - c[j])) / entry * 1e4
            if sl_hit and t_sl == 9999:
                t_sl = k
            if tp_hit and t_tp == 9999:
                t_tp = k
            if t_tp != 9999 or t_sl != 9999:
                break
        # extend resolution search beyond HMAX if unresolved (for the no-stop row)
        if t_tp == 9999 and t_sl == 9999:
            for j in range(last + 1, n):
                k = j - i
                tp_hit = (h[j] >= tp) if side == 1 else (lo[j] <= tp)
                sl_hit = (lo[j] <= sl) if side == 1 else (h[j] >= sl)
                if sl_hit: t_sl = k
                if tp_hit: t_tp = k
                if t_tp != 9999 or t_sl != 9999:
                    break

        tp_bar.append(t_tp); sl_bar.append(t_sl)
        tp_bps_l.append(tpb); sl_bps_l.append(slb)
        ts_favor.append(favor)
        i = (i + (min(t_tp, t_sl) if min(t_tp, t_sl) != 9999 else 1)) + 1 if min(t_tp, t_sl) != 9999 else i + 1

    tp_bar = np.array(tp_bar); sl_bar = np.array(sl_bar)
    tp_bps = np.array(tp_bps_l); sl_bps = np.array(sl_bps_l)
    FAV = np.vstack(ts_favor)            # (trades, HMAX+1)
    N = len(tp_bar)
    print(f"Entries simulated: {N:,}\n")

    open_fee = MAKER_BPS
    def net_bps_for_maxhold(k):
        """Per-trade net bps (incl 40% rebate) if we time-stop at bar k."""
        tpk = tp_bar <= k
        slk = sl_bar <= k
        first_tp = tp_bar <= sl_bar         # TP strictly first (ties -> SL via <=, see below)
        is_tp = tpk & (tp_bar < sl_bar)
        is_sl = slk & (sl_bar <= tp_bar)
        resolved = is_tp | is_sl
        is_ts = ~resolved                    # time-stop at bar k
        net = np.empty(N)
        # TP: gross +tp_bps, close maker
        net[is_tp] = tp_bps[is_tp] - open_fee - MAKER_BPS + REBATE * (open_fee + MAKER_BPS)
        # SL: gross -sl_bps - slip, close taker
        net[is_sl] = -sl_bps[is_sl] - slip - open_fee - TAKER_BPS + REBATE * (open_fee + TAKER_BPS)
        # time-stop: gross = favor at bar k, close maker
        fav_k = FAV[is_ts, np.minimum(k, HMAX)]
        net[is_ts] = fav_k - open_fee - MAKER_BPS + REBATE * (open_fee + MAKER_BPS)
        valid = ~np.isnan(net)           # drop end-of-data trades w/ truncated window
        net = net[valid]
        return net, int(is_tp.sum()), int(is_sl.sum()), int(is_ts.sum())

    print("="*104)
    print("  MAX_HOLD SWEEP  (TP 0.5×ATR / SL 1.5×ATR fixed)  — net is bps of notional, incl 40% rebate")
    print("="*104)
    print(f"  {'maxHold':>8} {'minutes':>8} {'net bps/t':>10} {'WR%':>6} {'avgWin bps':>11} "
          f"{'avgLoss bps':>12} {'RR':>5} {'tp%':>5} {'ts%':>5} {'sl%':>5}")
    print("  " + "-"*98)
    grid = [1,2,3,4,5,6,7,8,10,12,16,24]
    best_k, best_net = None, -1e9
    for k in grid:
        net, ntp, nsl, nts = net_bps_for_maxhold(k)
        wins = net[net > 0]; losses = net[net <= 0]
        wr = len(wins)/len(net)*100
        aw = wins.mean() if len(wins) else 0; al = losses.mean() if len(losses) else 0
        rr = aw/-al if al else float('inf')
        mb = net.mean()
        if mb > best_net: best_net, best_k = mb, k
        mark = ""
        if k == 2: mark = "  ← CURRENT live"
        if k == 3: mark = "  ← proposed"
        print(f"  {k:>8} {k*5:>7}m {mb:>+10.2f} {wr:>5.1f}% {aw:>+11.2f} {al:>+12.2f} {rr:>5.2f} "
              f"{ntp/N*100:>4.0f}% {nts/N*100:>4.0f}% {nsl/N*100:>4.0f}%{mark}")
    # no time-stop (resolve to actual TP/SL only)
    is_tp = tp_bar < sl_bar; is_sl = ~is_tp
    net = np.empty(N)
    net[is_tp] = tp_bps[is_tp] - open_fee - MAKER_BPS + REBATE*(open_fee+MAKER_BPS)
    net[is_sl] = -sl_bps[is_sl] - slip - open_fee - TAKER_BPS + REBATE*(open_fee+TAKER_BPS)
    wins = net[net>0]
    print(f"  {'none':>8} {'   —':>8} {net.mean():>+10.2f} {len(wins)/N*100:>5.1f}% "
          f"{wins.mean():>+11.2f} {net[net<=0].mean():>+12.2f} "
          f"{wins.mean()/-net[net<=0].mean():>5.2f} {is_tp.sum()/N*100:>4.0f}%    0% {is_sl.sum()/N*100:>4.0f}%  ← no time-stop")
    print("  " + "-"*98)
    print(f"  DATA-PROVEN OPTIMUM: max_hold = {best_k} bars ({best_k*5} min)  →  net {best_net:+.2f} bps/trade")

    # ── hazard: marginal value of holding bar k ──────────────────────────────
    print("\n" + "="*84)
    print("  HAZARD — of positions still open entering bar k, what happens ON bar k?")
    print("="*84)
    print(f"  {'bar k':>6} {'min':>5} {'still-open@start':>16} {'TP@k':>7} {'SL@k':>7} "
          f"{'P(TP|open)':>11} {'P(SL|open)':>11} {'mean favor bps':>15}")
    print("  " + "-"*78)
    for k in range(1, 9):
        open_start = ((tp_bar >= k) & (sl_bar >= k)).sum()    # neither resolved before bar k
        tp_at = ((tp_bar == k) & (sl_bar >= k)).sum()
        sl_at = ((sl_bar == k) & (tp_bar >= k)).sum()
        pno = open_start if open_start else 1
        still = (tp_bar > k) & (sl_bar > k)
        mfav = np.nanmean(FAV[still, min(k, HMAX)]) if still.sum() else 0.0
        print(f"  {k:>6} {k*5:>4}m {open_start:>16,} {tp_at:>7,} {sl_at:>7,} "
              f"{tp_at/pno*100:>10.1f}% {sl_at/pno*100:>10.1f}% {mfav:>+14.2f}")
    print("  " + "-"*78)
    print("  Read: once P(TP|open) on the next bar no longer beats the adverse drift + P(SL),")
    print("  holding longer only adds fee-and-loss exposure. That crossover is the stop.")


if __name__ == "__main__":
    main()
