"""Trend-reversal exit backtest — no time-stop, hold until reversal confirms.

Tests whether 'let the move run, exit on confirmed trend reversal' extracts real
directional EDGE (positive gross_bps) on BTC/DOGE at 5m & 15m. No bar/hold limit;
only a wide catastrophic ATR stop bounds the loss. Reversal definitions tested:

  ema   : exit when fast EMA crosses back through slow EMA (trend flip)
  chand : Chandelier / ATR-trailing — exit when price retraces k*ATR from the
          best excursion since entry (classic trend-follow trailing stop)
  nopp  : exit after N consecutive bars close against the position

Entry is tested two ways: the existing micro-momentum trigger, and a proper
EMA-cross trend entry (so entry and exit agree). gross_bps is the verdict — a
smarter exit can only mint money if there is post-entry drift to ride.
"""
from __future__ import annotations
import numpy as np, pandas as pd

MAKER, TAKER, REBATE = 0.0002, 0.0005, 0.40
MAXBARS = 1000  # effectively "no time limit"


def load(path):
    return pd.read_csv(path).sort_values("ts").reset_index(drop=True)


def ema(a, span):
    return pd.Series(a).ewm(span=span, adjust=False).mean().values


def atr_w(df, p=14):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1/p, adjust=False).mean().values


def run(df, *, entry, exit_rule, ema_fast=9, ema_slow=21, chand_k=3.0, n_opp=2,
        sl_mult=3.0, min_range=4.0, max_range=80.0, maker_close=False):
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    atr = atr_w(df)
    ef, es = ema(c, ema_fast), ema(c, ema_slow)
    n = len(df)
    days = (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 1000 / 86400
    trades = []
    i = max(ema_slow, 14) + 1
    while i < n - 1:
        if entry == "momentum":
            side = 1 if c[i] > o[i] else (-1 if c[i] < o[i] else 0)
            rng = (h[i] - l[i]) / o[i] * 1e4
            if side == 0 or rng < min_range or rng > max_range:
                i += 1; continue
        else:  # ema-cross entry
            up = ef[i-1] <= es[i-1] and ef[i] > es[i]
            dn = ef[i-1] >= es[i-1] and ef[i] < es[i]
            side = 1 if up else (-1 if dn else 0)
            if side == 0:
                i += 1; continue
        entry_px = c[i]
        sl_px = entry_px * (1 - side * sl_mult * atr[i] / entry_px)
        ext = entry_px  # best excursion (chandelier anchor)
        opp = 0
        exit_px = reason = None; held = 0
        for hb in range(1, MAXBARS + 1):
            b = i + hb
            if b >= n:
                break
            held = hb
            if (side > 0 and l[b] <= sl_px) or (side < 0 and h[b] >= sl_px):
                exit_px, reason = sl_px, "sl"; break
            if exit_rule == "ema":
                if (side > 0 and ef[b] < es[b]) or (side < 0 and ef[b] > es[b]):
                    exit_px, reason = c[b], "rev"; break
            elif exit_rule == "chand":
                ext = max(ext, h[b]) if side > 0 else min(ext, l[b])
                stop = ext - chand_k*atr[b] if side > 0 else ext + chand_k*atr[b]
                if (side > 0 and l[b] <= stop) or (side < 0 and h[b] >= stop):
                    exit_px, reason = stop, "rev"; break
            elif exit_rule == "nopp":
                if (side > 0 and c[b] < o[b]) or (side < 0 and c[b] > o[b]):
                    opp += 1
                else:
                    opp = 0
                if opp >= n_opp:
                    exit_px, reason = c[b], "rev"; break
        if exit_px is None:
            b = min(i + MAXBARS, n - 1)
            exit_px, reason, held = c[b], "maxbars", b - i
        gross = side * (exit_px - entry_px) / entry_px * 1e4
        cm = maker_close and reason == "rev"
        open_fee, close_fee = MAKER*1e4, (MAKER if cm else TAKER)*1e4
        reb = REBATE * (open_fee + (MAKER*1e4 if cm else 0))
        net = gross - open_fee - close_fee + reb
        trades.append((gross, net, held, reason))
        i = i + held + 1
    if not trades:
        return None
    g = np.array([t[0] for t in trades]); nt = len(trades)
    held = np.mean([t[2] for t in trades])
    wins = g > 0
    return dict(n=nt, tpm=nt/days*30, held=held, gross=g.mean(),
                net=np.mean([t[1] for t in trades]), cost_1m=-np.mean([t[1] for t in trades])*50,
                win=wins.mean()*100, avg_win=g[wins].mean() if wins.any() else 0,
                avg_loss=g[~wins].mean() if (~wins).any() else 0)


def main():
    files = [("BTC", "5m", "data/btc_5m.csv"), ("BTC", "15m", "data/btc_15m.csv"),
             ("DOGE", "5m", "data/doge_5m.csv"), ("DOGE", "15m", "data/doge_15m.csv")]
    combos = [("momentum", "ema"), ("momentum", "chand"), ("momentum", "nopp"),
              ("ema", "ema"), ("ema", "chand")]
    print("TREND-REVERSAL EXIT (no time-stop; hold to reversal or catastrophic SL)")
    print("gross_bps>0 == real edge to ride. avgHold in bars. trades/mo == volume capacity.\n")
    print(f"{'asset':<5}{'TF':<5}{'entry':<9}{'exit':<7}{'trades/mo':>10}{'avgHold':>8}"
          f"{'win%':>6}{'avgWin':>8}{'avgLoss':>8}{'gross_bps':>10}{'net_bps':>9}")
    print("-" * 85)
    for asset, tf, path in files:
        df = load(path)
        for entry, exit_rule in combos:
            r = run(df, entry=entry, exit_rule=exit_rule)
            if r is None:
                continue
            flag = "  <-- +edge" if r["gross"] > 0 else ""
            print(f"{asset:<5}{tf:<5}{entry:<9}{exit_rule:<7}{r['tpm']:>10.0f}{r['held']:>8.1f}"
                  f"{r['win']:>6.0f}{r['avg_win']:>8.1f}{r['avg_loss']:>8.1f}"
                  f"{r['gross']:>10.2f}{r['net']:>9.2f}{flag}")
        print()


if __name__ == "__main__":
    main()
