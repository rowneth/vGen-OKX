"""Martingale (3%→4%→5% after losses) vs flat 3% — does loss-recovery sizing help?

Runs the live-config volume farm from many different start points on the real
357d BTC 5m path, with $1,500 funding, comparing:
  - FLAT:       margin_frac = 3% always
  - MARTINGALE: margin_frac = 3% + 1% per consecutive SL (cap 5%), reset on a win

For each start it walks forward until $5M volume (success), the account bleeds
below the floor (BLOWUP), or data ends (incomplete). Reports success/blowup
rates, median net, worst drawdown, and bleed-per-$M for each scheme.

Usage: python scripts/martingale_compare.py
"""
from __future__ import annotations
import pathlib, statistics, numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data/historical/BTC_USDT_5m.parquet"
MAKER, TAKER, REBATE = 0.0002, 0.0005, 0.40
MARGIN_FRAC, RISK_PCT = 0.03, 0.025
MAX_LEV, MIN_LEV = 68.0, 5.0
MIN_RANGE_BPS, MAX_RANGE_BPS = 3.0, 40.0
FORCE_TP_BPS, SL_MULT, SL_BPS_MIN = 12.0, 1.5, 8.0
ATR_MIN_USD, SL_SLIP_BPS = 120.0, 3.0
VOL_TARGET, REWARD, FLOOR = 5_000_000.0, 1500.0, 5.0


def _atr14(df):
    hi, lo, cl = (df[c].to_numpy(float) for c in ("high", "low", "close"))
    pcl = np.concatenate([[cl[0]], cl[:-1]])
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - pcl), np.abs(lo - pcl)))
    atr = np.full(len(tr), np.nan); atr[13] = tr[:14].mean()
    for k in range(14, len(tr)):
        atr[k] = atr[k - 1] * (13 / 14) + tr[k] * (1 / 14)
    return atr


def run(o, h, lo, c, rng_bps, atr, day, E0, start, mart):
    equity = E0; volume = 0.0; rebate_pool = 0.0; cur_day = None
    min_eq = E0; loss_streak = 0; reached = blew = False
    n = len(o); i = start
    while i < n - 1:
        if equity < FLOOR:
            blew = True; break
        if cur_day is None or day[i] != cur_day:
            equity += rebate_pool; rebate_pool = 0.0; cur_day = day[i]
        a = atr[i]
        if np.isnan(a) or a < ATR_MIN_USD or not (MIN_RANGE_BPS <= rng_bps[i] <= MAX_RANGE_BPS):
            i += 1; continue
        side = 1 if c[i] > o[i] else -1
        atr_bps = a / o[i] * 1e4
        sl_bps = max(SL_MULT * atr_bps, SL_BPS_MIN)
        mf = min(MARGIN_FRAC + 0.01 * loss_streak, 0.05) if mart else MARGIN_FRAC
        entry = c[i]; margin = equity * mf
        sl_frac = sl_bps / 1e4
        lev = max(MIN_LEV, min((equity * RISK_PCT) / (margin * sl_frac), MAX_LEV))
        notional = margin * lev
        tp = entry * (1 + FORCE_TP_BPS / 1e4) if side == 1 else entry * (1 - FORCE_TP_BPS / 1e4)
        sl = entry * (1 - sl_bps / 1e4) if side == 1 else entry * (1 + sl_bps / 1e4)
        open_fee = notional * MAKER; rebate_pool += REBATE * open_fee; volume += notional
        reason = None; px = 0.0; ebar = i
        for j in range(i + 1, n):
            tp_hit = (h[j] >= tp) if side == 1 else (lo[j] <= tp)
            sl_hit = (lo[j] <= sl) if side == 1 else (h[j] >= sl)
            if sl_hit and tp_hit: reason, px, ebar = "sl", sl, j; break
            if tp_hit: reason, px, ebar = "tp", tp, j; break
            if sl_hit: reason, px, ebar = "sl", sl, j; break
        if reason is None: break
        gross = ((px - entry) if side == 1 else (entry - px)) / entry * notional
        if reason == "sl":
            gross -= SL_SLIP_BPS * 1e-4 * notional; close_fee = notional * TAKER
            rebate_pool += REBATE * close_fee  # all-legs (generous; reality maybe maker-only)
        else:
            close_fee = notional * MAKER; rebate_pool += REBATE * close_fee
        equity += gross - open_fee - close_fee; volume += notional
        min_eq = min(min_eq, equity)
        loss_streak = 0 if reason == "tp" else loss_streak + 1
        if volume >= VOL_TARGET: reached = True; break
        i = ebar + 1
    equity += rebate_pool
    net = (REWARD if reached else 0.0) + equity - E0
    bleed_per_m = (E0 - equity) / (volume / 1e6) if volume > 0 else 0.0
    return dict(reached=reached, blew=blew, volume=volume, final_eq=equity,
                min_eq=min_eq, net=net, bleed_per_m=bleed_per_m)


def summarize(rows, label):
    done = [r for r in rows if r["reached"] or r["blew"]]
    succ = [r for r in rows if r["reached"]]
    blow = [r for r in rows if r["blew"]]
    med = lambda xs: statistics.median(xs) if xs else 0.0
    print(f"  {label:>11}: starts={len(rows)}  reached$5M={len(succ)}  BLEW-UP={len(blow)}  "
          f"incomplete={len(rows)-len(done)}")
    print(f"  {'':>11}  success-rate={len(succ)/max(len(done),1)*100:>4.0f}%  "
          f"blowup-rate={len(blow)/max(len(done),1)*100:>4.0f}%  "
          f"median-net=${med([r['net'] for r in rows]):>+7.0f}  "
          f"median-minEq=${med([r['min_eq'] for r in rows]):>6.0f}  "
          f"median-bleed/$M=${med([r['bleed_per_m'] for r in done]):>5.0f}")


def main():
    df = pd.read_parquet(DATA).reset_index(drop=True)
    if not pd.api.types.is_datetime64_any_dtype(df["open_time"]):
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
    atr = _atr14(df)
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    rng_bps = np.abs(c - o) / o * 1e4
    day = df["open_time"].dt.floor("D").to_numpy()
    # start offsets across the first ~9 months (room to finish a ~1-3mo farm)
    starts = list(range(14, 75_000, 6_000))
    E0 = 1500.0
    print(f"$1,500 funding, $5M target, {len(starts)} start points across the year\n")
    flat = [run(o, h, lo, c, rng_bps, atr, day, E0, s, mart=False) for s in starts]
    mart = [run(o, h, lo, c, rng_bps, atr, day, E0, s, mart=True) for s in starts]
    summarize(flat, "FLAT 3%")
    print()
    summarize(mart, "MART 3→4→5%")


if __name__ == "__main__":
    main()
