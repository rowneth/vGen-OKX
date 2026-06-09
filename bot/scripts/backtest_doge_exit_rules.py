"""When to close? — DOGE exit-rule backtest (sequential, non-overlapping).

Answers the question "should we stop force-closing at 1 bar, and is there a
smarter exit?" by holding ONE position at a time (realistic for the live bot)
and walking each trade forward under different exit rules, then comparing:

  * gross_bps  -> is there real post-entry EDGE to capture (the only thing a
                 smarter exit can monetise)? If ~0, no exit rule helps directionally.
  * net_bps / cost_per_1M -> the volume-farming cost.
  * trades/month -> VOLUME THROUGHPUT. Longer holds = fewer round trips = less
                 volume, which fights a 5M/month target.
  * maker% -> fee efficiency (maker closes are ~3 bps cheaper than taker).

Fees: maker 2bps (40% rebate), taker 5bps. TP fills MAKER (limit), SL fills
TAKER (stop crosses), time-stop/trailing/reversal fill TAKER unless maker_close.
"""
from __future__ import annotations
import numpy as np, pandas as pd

MAKER, TAKER, REBATE = 0.0002, 0.0005, 0.40


def load_5m(path="data/doge_5m.csv"):
    return pd.read_csv(path).sort_values("ts").reset_index(drop=True)


def atr_wilder(df, period=14):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values


def backtest(df, *, rule, max_hold, tp_mult=0.5, sl_mult=1.5, tp_floor=6.0,
             tp_cap=25.0, sl_floor=8.0, trail_mult=1.0, maker_close=False,
             min_range=4.0, max_range=80.0):
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    atr = atr_wilder(df)
    n = len(df)
    fee_days = (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 1000 / 86400
    trades = []
    i = 14
    while i < n - 1:
        side = 1 if c[i] > o[i] else (-1 if c[i] < o[i] else 0)
        rng = (h[i] - l[i]) / o[i] * 1e4
        atr_bps = atr[i] / c[i] * 1e4
        if side == 0 or not np.isfinite(atr_bps) or rng < min_range or rng > max_range:
            i += 1; continue
        entry = c[i]
        tp_bps = min(max(tp_mult * atr_bps, tp_floor), tp_cap)
        sl_bps = max(sl_mult * atr_bps, sl_floor)
        tp_px = entry * (1 + side * tp_bps / 1e4)
        sl_px = entry * (1 - side * sl_bps / 1e4)
        trail = trail_mult * atr[i]
        best = entry
        exit_px = None; reason = None; held = 0
        for hh in range(1, max_hold + 1):
            b = i + hh
            if b >= n: break
            held = hh
            hi, lo, cl, op = h[b], l[b], c[b], o[b]
            # SL checked first (worst-case if both in one bar)
            if (side > 0 and lo <= sl_px) or (side < 0 and hi >= sl_px):
                exit_px, reason = sl_px, "sl"; break
            if (side > 0 and hi >= tp_px) or (side < 0 and lo <= tp_px):
                exit_px, reason = tp_px, "tp"; break
            # trailing stop
            if rule == "trail":
                best = max(best, hi) if side > 0 else min(best, lo)
                stop = best - trail if side > 0 else best + trail
                if (side > 0 and lo <= stop) or (side < 0 and hi >= stop):
                    exit_px, reason = stop, "trail"; break
            # momentum-reversal: bar closed against us
            if rule == "reversal" and ((side > 0 and cl < op) or (side < 0 and cl > op)):
                exit_px, reason = cl, "reversal"; break
        if exit_px is None:                       # hit max_hold -> time-stop
            b = min(i + max_hold, n - 1)
            exit_px, reason, held = c[b], "time_stop", b - i
        # fees
        close_maker = (reason == "tp") or (maker_close and reason in ("time_stop", "reversal"))
        gross = side * (exit_px - entry) / entry * 1e4
        open_fee, close_fee = MAKER * 1e4, (MAKER if close_maker else TAKER) * 1e4
        rebate = REBATE * (open_fee + (MAKER * 1e4 if close_maker else 0))
        net = gross - open_fee - close_fee + rebate
        trades.append((reason, held, gross, net, close_maker))
        i = i + held + 1                          # non-overlapping: next entry after exit

    if not trades:
        return None
    arr = trades
    nt = len(arr)
    gross = np.mean([t[2] for t in arr])
    net = np.mean([t[3] for t in arr])
    held = np.mean([t[1] for t in arr])
    tp_rate = np.mean([t[0] == "tp" for t in arr]) * 100
    maker_rate = np.mean([t[4] for t in arr]) * 100
    tpm = nt / fee_days * 30                       # trades per 30d
    return dict(n=nt, tpm=tpm, held=held, gross=gross, net=net,
                cost_1m=-net * 50, tp_rate=tp_rate, maker_rate=maker_rate,
                vol_month_per_dollar=tpm)  # volume/mo ∝ trades/mo at fixed sizing


def main():
    df = load_5m()
    print(f"DOGE exit-rule backtest | {len(df)} bars (~{len(df)*5/1440:.0f}d) | "
          f"one position at a time, sequential\n")
    print(f"{'rule':<22}{'maxH':>5}{'trades/mo':>10}{'avgHeld':>8}{'TP%':>6}"
          f"{'maker%':>8}{'gross_bps':>10}{'net_bps':>9}{'cost/$1M':>10}")
    print("-" * 88)

    configs = [
        ("time-stop (TAKER)", dict(rule="hold", max_hold=1)),
        ("time-stop (TAKER)", dict(rule="hold", max_hold=2)),
        ("time-stop (TAKER)", dict(rule="hold", max_hold=3)),
        ("time-stop (TAKER)", dict(rule="hold", max_hold=6)),
        ("time-stop (TAKER)", dict(rule="hold", max_hold=12)),
        ("time-stop (MAKER)", dict(rule="hold", max_hold=1, maker_close=True)),
        ("time-stop (MAKER)", dict(rule="hold", max_hold=3, maker_close=True)),
        ("hold-to-TP/SL", dict(rule="hold", max_hold=24)),
        ("hold-to-TP/SL", dict(rule="hold", max_hold=48)),
        ("trailing 1xATR", dict(rule="trail", max_hold=24, trail_mult=1.0)),
        ("trailing 2xATR", dict(rule="trail", max_hold=24, trail_mult=2.0)),
        ("reversal exit", dict(rule="reversal", max_hold=24)),
        ("reversal (MAKER)", dict(rule="reversal", max_hold=24, maker_close=True)),
    ]
    for label, kw in configs:
        r = backtest(df, **kw)
        if r is None:
            continue
        print(f"{label:<22}{kw['max_hold']:>5}{r['tpm']:>10.0f}{r['held']:>8.2f}"
              f"{r['tp_rate']:>6.0f}{r['maker_rate']:>8.0f}{r['gross']:>10.3f}"
              f"{r['net']:>9.3f}{r['cost_1m']:>10.0f}")

    print("\nKey: gross_bps ~ post-entry EDGE (a smarter exit can only help if this")
    print("is positive). net_bps/cost are after fees+rebate. trades/mo ~ volume you")
    print("can farm — longer holds slash it, fighting the 5M/month goal.")


if __name__ == "__main__":
    main()
