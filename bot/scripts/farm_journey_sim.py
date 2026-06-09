"""Volume-farm JOURNEY simulator — can a $X account reach a $5M volume target?

Walks the real BTC 5m path with the LIVE config (micro_momentum, firm 12bps TP,
1.5xATR SL, lev<=68, maker open/TP + taker SL, 3bps SL slippage), but with
EQUITY-SCALED sizing (margin = equity x 3%, so positions shrink as the account
bleeds) and DAILY rebate recycling (40% of fees added back to the wallet each day).

Stops when cumulative volume hits the target (success → collect reward) OR the
account decays below a floor (blown). Answers: from $500/$750/$1000, do we ever
reach $5M, what's the equity path, and what's the net after a $1,500 reward?

Usage: python scripts/farm_journey_sim.py
"""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data/historical/BTC_USDT_5m.parquet"

MAKER, TAKER, REBATE = 0.0002, 0.0005, 0.40
MARGIN_FRAC, RISK_PCT = 0.03, 0.025
MAX_LEV, MIN_LEV = 68.0, 5.0
MIN_RANGE_BPS, MAX_RANGE_BPS = 3.0, 40.0
FORCE_TP_BPS = 12.0
SL_MULT, SL_BPS_MIN = 1.5, 8.0
ATR_MIN_USD = 120.0
SL_SLIP_BPS = 3.0
VOL_TARGET = 5_000_000.0
REWARD = 1500.0
FLOOR = 5.0          # account considered blown below this
REBATE_ALL_LEGS = True   # current code behaviour; set False for OKX maker-only (worse)


def _atr14(df):
    hi, lo, cl = (df[c].to_numpy(float) for c in ("high", "low", "close"))
    pcl = np.concatenate([[cl[0]], cl[:-1]])
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - pcl), np.abs(lo - pcl)))
    atr = np.full(len(tr), np.nan)
    atr[13] = tr[:14].mean()
    for k in range(14, len(tr)):
        atr[k] = atr[k - 1] * (13 / 14) + tr[k] * (1 / 14)
    return atr


def journey(df, atr, E0, rebate_all_legs=True):
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    day = df["open_time"].dt.floor("D").to_numpy()
    rng_bps = np.abs(c - o) / o * 1e4
    n = len(df)

    equity = E0; volume = 0.0; rebate_pool = 0.0; cur_day = None
    min_eq = E0; trades = 0; wins = 0; reached = False; reach_bar = None
    i = 14
    while i < n - 1:
        if equity < FLOOR:
            break
        if cur_day is None or day[i] != cur_day:        # daily rebate recycle
            equity += rebate_pool; rebate_pool = 0.0; cur_day = day[i]
        a = atr[i]
        if np.isnan(a) or a < ATR_MIN_USD or not (MIN_RANGE_BPS <= rng_bps[i] <= MAX_RANGE_BPS):
            i += 1; continue
        side = 1 if c[i] > o[i] else -1
        atr_bps = a / o[i] * 1e4
        tp_bps = FORCE_TP_BPS
        sl_bps = max(SL_MULT * atr_bps, SL_BPS_MIN)
        entry = c[i]
        margin = equity * MARGIN_FRAC
        sl_frac = sl_bps / 1e4
        lev = max(MIN_LEV, min((equity * RISK_PCT) / (margin * sl_frac), MAX_LEV))
        notional = margin * lev
        tp = entry * (1 + tp_bps / 1e4) if side == 1 else entry * (1 - tp_bps / 1e4)
        sl = entry * (1 - sl_bps / 1e4) if side == 1 else entry * (1 + sl_bps / 1e4)
        open_fee = notional * MAKER
        rebate_pool += REBATE * open_fee
        volume += notional

        reason = None; px = 0.0; ebar = i
        for j in range(i + 1, n):
            tp_hit = (h[j] >= tp) if side == 1 else (lo[j] <= tp)
            sl_hit = (lo[j] <= sl) if side == 1 else (h[j] >= sl)
            if tp_hit and sl_hit: reason, px, ebar = "sl", sl, j; break
            if tp_hit: reason, px, ebar = "tp", tp, j; break
            if sl_hit: reason, px, ebar = "sl", sl, j; break
        if reason is None:
            break
        gross = ((px - entry) if side == 1 else (entry - px)) / entry * notional
        if reason == "sl":
            gross -= SL_SLIP_BPS * 1e-4 * notional
            close_fee = notional * TAKER
            if rebate_all_legs: rebate_pool += REBATE * close_fee
        else:
            close_fee = notional * MAKER
            rebate_pool += REBATE * close_fee
        equity += gross - open_fee - close_fee
        volume += notional
        trades += 1; wins += 1 if reason == "tp" else 0
        min_eq = min(min_eq, equity)
        if volume >= VOL_TARGET and not reached:
            reached = True; reach_bar = ebar; break
        i = ebar + 1

    equity += rebate_pool  # flush remaining rebate
    bars_used = (reach_bar or i) - 14
    days = bars_used * 5 / 60 / 24
    net = (REWARD if reached else 0.0) + equity - E0
    return dict(E0=E0, reached=reached, volume=volume, trades=trades,
                wr=wins / trades * 100 if trades else 0, final_eq=equity,
                min_eq=min_eq, days=days, net=net)


def main():
    df = pd.read_parquet(DATA).reset_index(drop=True)
    if not pd.api.types.is_datetime64_any_dtype(df["open_time"]):
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
    atr = _atr14(df)
    print(f"target ${VOL_TARGET/1e6:.0f}M volume, reward ${REWARD:.0f}, rebate {REBATE*100:.0f}% recycled daily\n")
    for rebate_all in (True, False):
        label = "rebate ALL legs (current code)" if rebate_all else "rebate MAKER-only (conservative; if OKX excludes taker)"
        print(f"### {label}")
        hdr = f"{'init$':>7} {'reach5M?':>9} {'vol($M)':>8} {'trades':>7} {'WR%':>6} {'minEq$':>8} {'finalEq$':>9} {'~days':>7} {'NET($)':>9}"
        print(hdr); print("-" * len(hdr))
        for E0 in (500, 750, 1000, 1125, 1250, 1500, 2000, 3000):
            r = journey(df, atr, float(E0), rebate_all_legs=rebate_all)
            print(f"{E0:>7} {('YES' if r['reached'] else 'no'):>9} {r['volume']/1e6:>8.2f} "
                  f"{r['trades']:>7} {r['wr']:>6.1f} {r['min_eq']:>8.2f} {r['final_eq']:>9.2f} "
                  f"{r['days']:>7.0f} {r['net']:>+9.0f}")
        print()
    print("NET = (reward if $5M reached) + final equity − initial.  'no' = bled out / plateaued before $5M.")


if __name__ == "__main__":
    main()
