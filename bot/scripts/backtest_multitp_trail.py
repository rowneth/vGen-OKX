"""Multi-TP scale-out (50/30/20) + trailing-stop remainder — does it beat the
single-TP 1-bar farmer on cost/$1M? Tested on real BTC 5m.

Each trade: 100% on at entry (maker). Scale out 50% at TP1, 30% at TP2, 20% at
TP3 (maker limits at tp1/tp2/tp3 x ATR). The still-open portion trails by
trail x ATR from its best excursion (taker when hit). Catastrophic wide SL.
Time-stop closes any remainder at max_hold (maker). Volume = 2x notional per
full round trip either way, so cost/$1M = -net_bps * 50.
"""
from __future__ import annotations
import numpy as np, pandas as pd

MAKER, TAKER, REBATE = 0.0002, 0.0005, 0.40


def load(path):
    return pd.read_csv(path).sort_values("ts").reset_index(drop=True)


def atr_w(df, p=14):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1/p, adjust=False).mean().values


def run(df, *, multi_tp, trail, max_hold, tp_mults=(1.0, 2.0, 3.0),
        portions=(0.5, 0.3, 0.2), trail_mult=1.5, sl_mult=6.0,
        single_tp_mult=1.0, min_range=4.0, max_range=80.0, maker_timestop=True):
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    atr = atr_w(df); n = len(df)
    days = (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 1000 / 86400
    rows = []
    i = 15
    while i < n - 1:
        side = 1 if c[i] > o[i] else (-1 if c[i] < o[i] else 0)
        rng = (h[i] - l[i]) / o[i] * 1e4
        ab = atr[i] / c[i] * 1e4
        if side == 0 or not np.isfinite(ab) or rng < min_range or rng > max_range:
            i += 1; continue
        entry = c[i]; a = atr[i]
        if multi_tp:
            tps = [(p, entry + side * m * a) for p, m in zip(portions, tp_mults)]
        else:
            tps = [(1.0, entry + side * single_tp_mult * a)]
        sl_px = entry - side * sl_mult * a
        remaining = 1.0
        best = entry
        legs = []  # (portion, px, maker?)
        held = 0
        for hb in range(1, max_hold + 1):
            b = i + hb
            if b >= n:
                break
            held = hb
            best = max(best, h[b]) if side > 0 else min(best, l[b])
            # catastrophic SL on remainder (taker)
            if remaining > 0 and ((side > 0 and l[b] <= sl_px) or (side < 0 and h[b] >= sl_px)):
                legs.append((remaining, sl_px, False)); remaining = 0.0; break
            # scale-out TPs (maker) — fill in order while price reaches them
            for k, (por, tpx) in enumerate(tps):
                if por <= 0:
                    continue
                if (side > 0 and h[b] >= tpx) or (side < 0 and l[b] <= tpx):
                    fill = min(por, remaining)
                    if fill > 0:
                        legs.append((fill, tpx, True)); remaining -= fill
                    tps[k] = (0.0, tpx)
            if remaining <= 1e-9:
                break
            # trailing stop on the remainder (taker)
            if trail and remaining > 0:
                stop = best - trail_mult * a if side > 0 else best + trail_mult * a
                if (side > 0 and l[b] <= stop) or (side < 0 and h[b] >= stop):
                    legs.append((remaining, stop, False)); remaining = 0.0; break
        if remaining > 1e-9:  # time-stop the rest at close
            b = min(i + max_hold, n - 1)
            legs.append((remaining, c[b], maker_timestop)); remaining = 0.0
            held = b - i
        # economics (bps of FULL notional)
        gross = sum(por * side * (px - entry) / entry * 1e4 for por, px, _ in legs)
        open_fee = MAKER * 1e4
        close_fee = sum(por * (MAKER if mk else TAKER) * 1e4 for por, _, mk in legs)
        rebate = REBATE * (open_fee + sum(por * (MAKER * 1e4 if mk else 0) for por, _, mk in legs))
        net = gross - open_fee - close_fee + rebate
        maker_notional = 1.0 + sum(por for por, _, mk in legs if mk)  # open + maker closes
        rows.append((gross, net, held, len(legs), maker_notional / 2.0))
        i = i + held + 1
    if not rows:
        return None
    g = np.array([r[0] for r in rows]); nt = len(rows)
    return dict(n=nt, tpm=nt/days*30, gross=g.mean(),
                net=np.mean([r[1] for r in rows]), cost_1m=-np.mean([r[1] for r in rows])*50,
                held=np.mean([r[2] for r in rows]), legs=np.mean([r[3] for r in rows]),
                maker=np.mean([r[4] for r in rows])*100)


def main():
    df = load("data/btc_5m.csv")
    print(f"BTC 5m | multi-TP 50/30/20 @ 1/2/3xATR + trailing 1.5xATR | {len(df)} bars\n")
    print(f"{'config':<34}{'maxH':>5}{'avgLegs':>8}{'maker%':>8}{'gross_bps':>10}{'cost/$1M':>10}{'trades/mo':>10}")
    print("-" * 85)
    configs = [
        ("BASELINE single-TP, 1-bar stop", dict(multi_tp=False, trail=False, max_hold=1)),
        ("multi-TP, 1-bar stop",           dict(multi_tp=True,  trail=False, max_hold=1)),
        ("multi-TP, 3-bar stop",           dict(multi_tp=True,  trail=False, max_hold=3)),
        ("multi-TP, 6-bar stop",           dict(multi_tp=True,  trail=False, max_hold=6)),
        ("multi-TP + TRAIL, 6-bar",        dict(multi_tp=True,  trail=True,  max_hold=6)),
        ("multi-TP + TRAIL, 12-bar",       dict(multi_tp=True,  trail=True,  max_hold=12)),
    ]
    base = None
    for label, kw in configs:
        r = run(df, **kw)
        if r is None:
            continue
        if base is None:
            base = r["cost_1m"]
        delta = f"  ({r['cost_1m']-base:+.0f} vs base)" if label[0] != "B" else ""
        print(f"{label:<34}{kw['max_hold']:>5}{r['legs']:>8.2f}{r['maker']:>7.0f}%"
              f"{r['gross']:>10.2f}{r['cost_1m']:>10.0f}{r['tpm']:>10.0f}{delta}")
    print("\ncost/$1M lower = cheaper. The single-TP 1-bar farmer is the cost floor;")
    print("anything that holds longer or trails pays more (negative edge + taker legs).")


if __name__ == "__main__":
    main()
