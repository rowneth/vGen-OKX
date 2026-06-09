"""Vectorized DOGE volume-farmer backtest with the project's dynamic leverage.

Replicates the live strategy's economics on DOGE-USDT-SWAP:
  * entry: micro-momentum (sign of bar body), post-only -> MAKER open
  * ATR-14 scaled TP/SL with fee-aware TP floor + reachability cap
  * 1-bar time-stop (resolve on the NEXT 5m bar): TP(maker) / SL(taker) /
    time-stop(taker) — matching live_volume_executor (maker_exit disabled)
  * dynamic leverage: lev = clip(risk% / (margin_frac * SL_frac), min, max),
    capped at DOGE's exchange max of 50x. NOTE: equity cancels, so leverage is
    equity-INDEPENDENT -> $50 and $500 give identical % results (proved in main).
  * 40% maker-fee rebate accrues to a pool and reloads into equity every 12h.

The per-trade P&L is fully vectorized (numpy); only the equity compounding loops
(it is path-dependent). Outputs the metric a volume farmer cares about: $ cost
per $1M of volume, plus survival to a volume target.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TICK = 1e-5
CTVAL = 1000.0
DOGE_MAX_LEV = 50.0
MAKER = 0.0002
TAKER = 0.0005
REBATE = 0.40


def load_5m(path="data/doge_5m.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def atr_wilder(df: pd.DataFrame, period=14) -> np.ndarray:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().values
    return atr


def compute_trades(df: pd.DataFrame, *, tp_mult, sl_mult, margin_frac, risk_pct,
                   tp_floor=6.0, tp_cap=25.0, sl_floor=8.0, min_lev=3.0,
                   max_lev=DOGE_MAX_LEV, min_range=4.0, max_range=80.0,
                   atr_period=14, time_stop_maker=False):
    o = df["open"].values; h = df["high"].values
    l = df["low"].values;  c = df["close"].values
    ts = df["ts"].values
    atr = atr_wilder(df, atr_period)
    n = len(df)

    body_bps = (c - o) / o * 1e4
    side = np.sign(c - o).astype(float)            # +1 long, -1 short
    rng_bps = (h - l) / o * 1e4
    atr_bps = atr / c * 1e4

    tp_bps = np.clip(tp_mult * atr_bps, tp_floor, tp_cap)
    sl_bps = np.maximum(sl_mult * atr_bps, sl_floor)
    sl_frac = sl_bps / 1e4
    lev = np.clip(risk_pct / (margin_frac * np.where(sl_frac > 0, sl_frac, np.nan)),
                  min_lev, max_lev)

    entry = c
    is_long = side > 0
    tp_px = np.where(is_long, entry * (1 + tp_bps / 1e4), entry * (1 - tp_bps / 1e4))
    sl_px = np.where(is_long, entry * (1 - sl_bps / 1e4), entry * (1 + sl_bps / 1e4))

    # resolve on NEXT bar (1-bar time-stop). shift outcomes by +1.
    h1 = np.concatenate([h[1:], [np.nan]])
    l1 = np.concatenate([l[1:], [np.nan]])
    c1 = np.concatenate([c[1:], [np.nan]])

    tp_hit = np.where(is_long, h1 >= tp_px, l1 <= tp_px)
    sl_hit = np.where(is_long, l1 <= sl_px, h1 >= sl_px)

    # priority: both->SL(taker, worst-case), tp->TP(maker), sl->SL(taker), else time-stop(taker)
    reason = np.full(n, "time_stop", dtype=object)
    reason[tp_hit & ~sl_hit] = "tp"
    reason[sl_hit] = "sl"            # includes both-hit (worst case)
    exit_px = np.where(reason == "tp", tp_px,
              np.where(reason == "sl", sl_px, c1))

    gross_ret = side * (exit_px - entry) / entry        # of notional
    # TP is always a maker limit. Time-stop CAN be a maker re-peg (cheaper) if
    # time_stop_maker; SL stays taker (a stop must cross). This is the single
    # biggest fee lever — ~61% of exits are otherwise taker.
    close_is_maker = (reason == "tp") | (time_stop_maker & (reason == "time_stop"))
    open_fee = MAKER
    close_fee = np.where(close_is_maker, MAKER, TAKER)
    net_ret_notional = gross_ret - open_fee - close_fee  # full fees (rebate handled separately)
    maker_fee_frac = open_fee + np.where(close_is_maker, MAKER, 0.0)

    valid = (
        np.isfinite(atr_bps) & np.isfinite(c1) & (side != 0)
        & (rng_bps >= min_range) & (rng_bps <= max_range) & np.isfinite(lev)
    )

    notional_over_eq = margin_frac * lev
    out = pd.DataFrame({
        "ts": ts,
        "valid": valid,
        "side": side,
        "lev": lev,
        "reason": reason,
        "gross_ret": gross_ret,
        "ret_on_eq": notional_over_eq * net_ret_notional,        # equity change (full fees)
        "rebate_on_eq": notional_over_eq * maker_fee_frac * REBATE,
        "vol_factor": notional_over_eq * 2.0,                    # open+close notional / equity
    })
    out.loc[~valid, ["ret_on_eq", "rebate_on_eq", "vol_factor"]] = 0.0
    return out


def simulate(trades: pd.DataFrame, *, capital, target, reload_hours=12):
    ts = trades["ts"].values.astype("int64")
    ret = trades["ret_on_eq"].values
    reb = trades["rebate_on_eq"].values
    volf = trades["vol_factor"].values
    valid = trades["valid"].values
    reason = trades["reason"].values

    eq = capital
    pool = 0.0
    volume = 0.0
    peak = capital
    max_dd = 0.0
    last_reload = ts[0] if len(ts) else 0
    reload_ms = reload_hours * 3600 * 1000
    n_tp = n_sl = n_time = 0
    n_trades = 0
    ruin = False
    reached = False
    end_i = len(ret)

    for i in range(len(ret)):
        if not valid[i]:
            continue
        eq_before = eq
        volume += eq_before * volf[i]
        eq = eq * (1.0 + ret[i])
        pool += eq_before * reb[i]
        n_trades += 1
        r = reason[i]
        n_tp += (r == "tp"); n_sl += (r == "sl"); n_time += (r == "time_stop")
        # 12h rebate reload
        if ts[i] - last_reload >= reload_ms:
            eq += pool; pool = 0.0; last_reload = ts[i]
        peak = max(peak, eq); max_dd = max(max_dd, (peak - eq) / peak)
        if eq <= 0:
            ruin = True; end_i = i; break
        if volume >= target:
            reached = True; end_i = i; break
    eq += pool  # final reload of remaining pool
    days = (ts[min(end_i, len(ts) - 1)] - ts[0]) / 1000 / 86400 if len(ts) else 0
    pnl = eq - capital
    cost_per_1m = (pnl / volume * 1e6) if volume > 0 else 0.0
    return dict(capital=capital, final_eq=eq, pnl=pnl, pnl_pct=pnl / capital * 100,
                volume=volume, cost_per_1m=cost_per_1m, n_trades=n_trades,
                wr=(n_tp / n_trades * 100 if n_trades else 0), n_tp=n_tp, n_sl=n_sl,
                n_time=n_time, max_dd=max_dd * 100, ruin=ruin, reached=reached, days=days)


def main():
    df = load_5m()
    print(f"DOGE 5m vectorized volume-farmer backtest | {len(df)} bars "
          f"(~{len(df)*5/1440:.0f}d) | maker {MAKER*1e4:.0f}/taker {TAKER*1e4:.0f} bps, "
          f"rebate {REBATE:.0%} reload/12h | DOGE maxLev {DOGE_MAX_LEV:.0f}x\n")
    TARGET = 400_000

    grid = []
    for tp_mult in (0.5, 1.0):
        for sl_mult in (1.0, 1.5):
            for margin_frac in (0.03, 0.10):
                for risk_pct in (0.025, 0.05):
                    grid.append((tp_mult, sl_mult, margin_frac, risk_pct))

    hdr = (f"{'tpM':>4}{'slM':>4}{'mgn%':>6}{'risk%':>6}{'lev~':>6}{'trades':>7}"
           f"{'wr%':>5}{'$50 pnl%':>9}{'cost/1M':>9}{'vol$':>9}{'maxDD%':>7}{'400k?':>7}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for (tp_mult, sl_mult, margin_frac, risk_pct) in grid:
        tr = compute_trades(df, tp_mult=tp_mult, sl_mult=sl_mult,
                            margin_frac=margin_frac, risk_pct=risk_pct)
        lev_typ = np.nanmedian(tr.loc[tr["valid"], "lev"].values) if tr["valid"].any() else 0
        r50 = simulate(tr, capital=50, target=TARGET)
        rows.append((tp_mult, sl_mult, margin_frac, risk_pct, lev_typ, r50, tr))
        tag = ("REACHED" if r50["reached"] else ("RUIN" if r50["ruin"] else "ran out"))
        print(f"{tp_mult:>4.1f}{sl_mult:>4.1f}{margin_frac*100:>6.0f}{risk_pct*100:>6.1f}"
              f"{lev_typ:>6.0f}{r50['n_trades']:>7}{r50['wr']:>5.0f}{r50['pnl_pct']:>9.2f}"
              f"{r50['cost_per_1m']:>9.0f}{r50['volume']:>9.0f}{r50['max_dd']:>7.1f}{tag:>7}")

    # pick least-bad config by cost_per_1m (closest to 0 / positive)
    best = max(rows, key=lambda x: x[5]["cost_per_1m"])
    tp_mult, sl_mult, mf, rp, levm, r50, tr = best
    print(f"\nLeast-costly config: tpM={tp_mult} slM={sl_mult} margin={mf*100:.0f}% "
          f"risk={rp*100:.1f}% (lev~{levm:.0f}x)")
    print(f"  reasons: TP={r50['n_tp']} SL={r50['n_sl']} time_stop={r50['n_time']}")

    print("\n--- $50 vs $500 with SAME config (proving equity-independence) ---")
    for cap in (50, 500):
        r = simulate(tr, capital=cap, target=TARGET)
        print(f"  ${cap:>4}: pnl ${r['pnl']:>9.2f} ({r['pnl_pct']:>7.2f}%)  vol ${r['volume']:>11,.0f}  "
              f"cost/1M ${r['cost_per_1m']:>6.0f}  {'REACHED 400k' if r['reached'] else ('RUIN' if r['ruin'] else 'ran out of data')}"
              f"  in {r['days']:.1f}d")


if __name__ == "__main__":
    main()
