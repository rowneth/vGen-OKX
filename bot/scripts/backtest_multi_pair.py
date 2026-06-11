"""Multi-pair backtest: v3 strategy on BTC/ETH/SOL/XRP/DOGE/LTC with
tick-size-aware slippage and a per-pair TP/SL volatility grid."""
import asyncio, sys, pathlib
import importlib.util
import numpy as np
sys.path.insert(0, str(pathlib.Path("src").resolve()))
spec = importlib.util.spec_from_file_location("bt", "scripts/backtest_v1_vs_v3.py")
bt = importlib.util.module_from_spec(spec); spec.loader.exec_module(bt)
from exchange.okx_client import OKXClient

PAIRS = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "XRP_USDT", "DOGE_USDT", "LTC_USDT"]

async def main():
    # instrument specs -> tick size in bps of price (the honest alt handicap)
    ticks = {}
    async with OKXClient() as c:
        for p in PAIRS:
            inst = await c.get_instrument(p)
            tk = await c.get_ticker(p)
            px = float(tk["last"]); tick = float(inst["tickSz"])
            ticks[p] = tick / px * 1e4
            await asyncio.sleep(0.15)

    base = dict(alternate=False, time_stop_bars=1, tp_mode="atr_floor",
                sl_atr_mult=9.0, min_range_bps=1.0, atr_min_usd=0.0,
                same_bar_reentry=True, lev=15.0)
    results = {}
    for p in PAIRS:
        df = await bt.fetch_bars(60, symbol=p)
        atr = bt.wilder_atr(df)
        atr_bps = np.nanmedian(atr / df["close"].values * 1e4)
        tick_bps = ticks[p]
        # tick-aware slippage: every slip is at least ~half a tick (maker exits
        # re-peg across the spread; stops gap through wider books on alts)
        slip = dict(scratch_slip_bps=max(0.5, tick_bps*0.6),
                    sl_slip_bps=max(3.0, tick_bps*2.0),
                    entry_slip_bps=max(0.4, tick_bps*0.6))
        # default v3 settings
        d = bt.simulate(df, name="", **base, **slip)
        # volatility-adjusted grid per pair
        best = None
        for tp_m in (0.5, 1.0, 1.5):
            for cap in (18.0, 25.0, 35.0):
                for mr in (40.0, 80.0):
                    r = bt.simulate(df, name="", **base, **slip,
                                    tp_atr_mult=tp_m, tp_cap_bps=cap, max_range_bps=mr)
                    key = (r["cost_per_1M"], -r["vol_per_day"])
                    if best is None or key < best[0]:
                        best = (key, dict(tp_mult=tp_m, cap=cap, max_range=mr), r)
        results[p] = (atr_bps, tick_bps, d, best)

    print(f"\n{'pair':10} {'ATRbps':>6} {'tickbps':>7} | {'DEFAULT $/1M':>12} {'vol/day':>9} {'net':>6} | "
          f"{'TUNED $/1M':>10} {'vol/day':>9} {'net':>6}  tuned-params")
    for p, (a, t, d, best) in results.items():
        _, bp, b = best
        print(f"{p:10} {a:>6.1f} {t:>7.3f} | {d['cost_per_1M']:>12} {d['vol_per_day']:>9,.0f} {d['campaign_net']:>6} | "
              f"{b['cost_per_1M']:>10} {b['vol_per_day']:>9,.0f} {b['campaign_net']:>6}  "
              f"tp={bp['tp_mult']}x cap={bp['cap']:.0f} maxrng={bp['max_range']:.0f}")
    # halves robustness for the best non-BTC candidate
    nb = min((p for p in PAIRS if p != "BTC_USDT"),
             key=lambda p: results[p][3][2]["cost_per_1M"])
    _, bp, _ = results[nb][3]
    df = await bt.fetch_bars(60, symbol=nb)
    half = len(df)//2
    for lbl, part in (("H1", df.iloc[:half]), ("H2", df.iloc[half:])):
        slip = dict(scratch_slip_bps=max(0.5, results[nb][1]*0.6),
                    sl_slip_bps=max(3.0, results[nb][1]*2.0),
                    entry_slip_bps=max(0.4, results[nb][1]*0.6))
        r = bt.simulate(part.reset_index(drop=True), name="", **base, **slip,
                        tp_atr_mult=bp['tp_mult'], tp_cap_bps=bp['cap'], max_range_bps=bp['max_range'])
        print(f"robustness {nb} {lbl}: ${r['cost_per_1M']}/1M  vol/day ${r['vol_per_day']:,.0f}  net ${r['campaign_net']}")

asyncio.run(main())
