"""Fresh head-to-head backtest: v1 (alternation, no time-stop) vs v3 (momentum,
1-bar time-stop) on real OKX BTC-USDT-SWAP 5m candles.

Built from scratch on 2026-06-10 — deliberately does NOT reuse the older
backtest scripts or their cached conclusions. Downloads its own data from the
OKX public API (cached to CSV), replays every variant through IDENTICAL fill
and fee mechanics, and prints a campaign-level comparison.

Mechanics (same for every variant — only strategy logic differs):
  * Entry at the signal bar's close, as maker if the NEXT bar touches the
    entry price (post-only would have been hit), else taker fallback with
    0.4bp adverse slip (mirrors entry_repeg.taker_fallback).
  * TP is a resting maker limit: fills when the bar range touches it.
  * SL is a taker market stop: 3bp adverse slip through the trigger.
  * Both-touched bar resolves WORST CASE (SL first).
  * Time-stop (where enabled) closes at bar close, maker fee, 0.5bp adverse
    slip (the live maker-exit ladder measured ~0 adverse today; 0.5bp is a
    safety margin so v3 isn't flattered).
  * Fees: maker 2bps, taker 5bps, 40% rebate on ALL legs (campaign terms).
  * Volume counts both legs (open notional + close notional at exit price).
  * Fixed $1,800 notional per leg for every variant, so the comparison
    isolates strategy, not sizing.

Run:  .venv/bin/python scripts/backtest_v1_vs_v3.py [--days 60]
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from exchange.okx_client import OKXClient  # noqa: E402

MAKER = 0.0002
TAKER = 0.0005
REBATE = 0.20   # current live deal (conservative reading)
MAKER_NET = MAKER * (1 - REBATE)
TAKER_NET = TAKER * (1 - REBATE)
NOTIONAL = 1_800.0                        # $/leg, fixed for all variants
SL_SLIP_BPS = 3.0                         # taker stop slips through trigger
TAKER_ENTRY_SLIP_BPS = 0.4                # entry taker-fallback slip
SCRATCH_SLIP_BPS = 0.5                    # time-stop maker-exit adverse
VOL_TARGET = 5_000_000.0
REWARD = 1_500.0
CAMPAIGN_DAYS = 30.0

CACHE = pathlib.Path(__file__).resolve().parents[1] / "data" / "btc_5m_backtest_cache.csv"


async def fetch_bars(days: int) -> pd.DataFrame:
    """Download `days` of closed 5m candles, oldest→newest, with CSV cache."""
    need = days * 288
    if CACHE.exists():
        df = pd.read_csv(CACHE)
        if len(df) >= need:
            print(f"cache hit: {len(df)} bars from {CACHE.name}")
            return df.tail(need).reset_index(drop=True)
    rows: list = []
    async with OKXClient() as client:
        # Walk backwards with /market/history-candles (100 bars per call).
        after = None
        while len(rows) < need + 200:
            raw = await client.get_candles(
                "BTC_USDT", "5m", limit=100, history=True,
                after=after,
            )
            if not raw:
                break
            for r in raw:
                if r[8] == "1":
                    rows.append({
                        "ts": int(r[0]),
                        "open": float(r[1]), "high": float(r[2]),
                        "low": float(r[3]), "close": float(r[4]),
                    })
            after = int(raw[-1][0])      # oldest ts of this page → go older
            if len(rows) % 2000 < 100:
                print(f"  fetched {len(rows)} bars…")
            await asyncio.sleep(0.12)    # stay under 10 req/2s public limit
    df = pd.DataFrame(rows).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    print(f"downloaded {len(df)} bars → cached")
    return df.tail(need).reset_index(drop=True)


def wilder_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    hi, lo, cl = df["high"].values, df["low"].values, df["close"].values
    prev = np.concatenate([[cl[0]], cl[:-1]])
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev), np.abs(lo - prev)))
    atr = np.full(len(tr), np.nan)
    if len(tr) <= period:
        return atr
    atr[period] = tr[:period].mean()
    a = 1.0 / period
    for i in range(period + 1, len(tr)):
        atr[i] = atr[i - 1] * (1 - a) + tr[i] * a
    return atr


def simulate(df: pd.DataFrame, *, name: str,
             alternate: bool,            # strict long→short→long
             time_stop_bars: int,        # 0 = disabled (hold to TP/SL)
             tp_mode: str,               # "fixed12" (v1) | "atr_floor" (v3)
             sl_atr_mult: float | None,  # None = NO price stop (time-stop+liq only)
             min_range_bps: float,
             atr_min_usd: float,
             same_bar_reentry: bool,
             tp_atr_mult: float = 1.0,   # TP = clamp(max(6, mult*ATRbps), floor, 18)
             be_seek_bars: int = 0,      # bars AFTER bar1 hunting breakeven+fees
             be_exit_bps: float = 0.0,   # the collapsed target (maker close)
             lev: float = 0.0,           # >0 enables intrabar LIQUIDATION modeling
             mmr_buffer: float = 0.005,  # maintenance margin + fee buffer
             liq_penalty_bps: float = 2.0) -> dict:
    o = df["open"].values; h = df["high"].values
    l = df["low"].values; c = df["close"].values
    ts = df["ts"].values
    atr = wilder_atr(df)
    n = len(df)

    pos = None          # dict(side, entry, tp, sl, bars_held, open_fee_type)
    last_side = "short"  # so an alternating book starts long
    trades: list = []
    pending = None      # entry decided at bar i-1's close, fee resolved on bar i

    def decide_entry(i: int) -> dict | None:
        nonlocal last_side
        if o[i] <= 0:
            return None
        rng_bps = abs(c[i] - o[i]) / o[i] * 1e4
        if rng_bps < min_range_bps or rng_bps > 40.0:
            return None
        if atr_min_usd > 0 and (np.isnan(atr[i]) or atr[i] < atr_min_usd):
            return None
        if np.isnan(atr[i]):
            return None
        atr_bps = atr[i] / c[i] * 1e4
        if tp_mode == "fixed12":
            tp_bps = 12.0
        else:  # v3: ATR-relative, fee floor = 2*round_trip+2, cap 18
            rt_bps = MAKER_NET * 2 * 1e4
            tp_bps = min(max(max(6.0, tp_atr_mult * atr_bps), 2*rt_bps + 2.0), 18.0)
        if alternate:
            side = "short" if last_side == "long" else "long"
        else:
            side = "long" if c[i] > o[i] else "short"
        last_side = side
        e = c[i]
        sig = {
            "side": side, "entry": e,
            "tp": e * (1 + tp_bps / 1e4) if side == "long" else e * (1 - tp_bps / 1e4),
            "sl": None,
        }
        if sl_atr_mult is not None:
            sl_bps = max(8.0, sl_atr_mult * atr_bps)
            sig["sl"] = e * (1 - sl_bps / 1e4) if side == "long" else e * (1 + sl_bps / 1e4)
        return sig

    def close_trade(p: dict, exit_px: float, reason: str, fee_type: str) -> None:
        e = p["entry"]
        gross = ((exit_px - e) / e if p["side"] == "long" else (e - exit_px) / e) * NOTIONAL
        open_fee = NOTIONAL * (MAKER_NET if p["open_fee_type"] == "maker" else TAKER_NET)
        close_rate = MAKER_NET if fee_type == "maker" else TAKER_NET
        close_leg = NOTIONAL * (exit_px / e)
        close_fee = close_leg * close_rate
        trades.append({
            "reason": reason,
            "net": gross - open_fee - close_fee,
            "gross": gross,
            "fees": open_fee + close_fee,
            "volume": NOTIONAL + close_leg,
            "bars": p["bars_held"],
        })

    for i in range(1, n):
        # resolve a pending entry's fee type on this bar (post-only touched?)
        if pending is not None:
            touched = (l[i] <= pending["entry"]) if pending["side"] == "long" \
                else (h[i] >= pending["entry"])
            pending["open_fee_type"] = "maker" if touched else "taker"
            if not touched:  # taker fallback slips slightly against us
                adj = pending["entry"] * (TAKER_ENTRY_SLIP_BPS / 1e4)
                pending["entry"] += adj if pending["side"] == "long" else -adj
            pending["bars_held"] = 0
            pos = pending
            pending = None
            # NOTE: exits for the fill bar are evaluated this same bar below —
            # identical treatment for every variant.

        closed_this_bar = False
        if pos is not None:
            pos["bars_held"] += 1
            long = pos["side"] == "long"
            # Two-phase favorable exit (user proposal): after bar 1 the resting
            # TP limit is amended DOWN to breakeven+fees; first touch closes.
            if be_seek_bars > 0 and pos["bars_held"] >= 2:
                fav_px = (pos["entry"] * (1 + be_exit_bps / 1e4) if long
                          else pos["entry"] * (1 - be_exit_bps / 1e4))
                fav_reason = "be_exit"
            else:
                fav_px = pos["tp"]
                fav_reason = "tp"
            tp_hit = (h[i] >= fav_px) if long else (l[i] <= fav_px)
            # Adverse exits: the binding level is whichever sits CLOSER to the
            # entry — the price reaches it first on the way down/up. With high
            # leverage the liquidation line can sit INSIDE the stop, in which
            # case the exchange liquidates before the stop ever fires.
            adverse: list = []
            if pos["sl"] is not None:
                adverse.append(("sl", pos["sl"]))
            if lev > 0:
                liq_dist = max(1.0 / lev - mmr_buffer, 0.001)
                liq_px = pos["entry"] * (1 - liq_dist) if long else pos["entry"] * (1 + liq_dist)
                adverse.append(("liq", liq_px))
            adverse_reason = adverse_px = None
            if adverse:
                adverse_reason, adverse_px = (max(adverse, key=lambda x: x[1]) if long
                                              else min(adverse, key=lambda x: x[1]))
            adverse_hit = adverse_px is not None and (
                (l[i] <= adverse_px) if long else (h[i] >= adverse_px))
            if adverse_hit:  # worst case: adverse exit first whenever touched
                if adverse_reason == "liq":
                    # whole isolated margin is gone: price at liq line plus
                    # penalty, taker-fee'd — strictly worse than any stop.
                    slip = adverse_px * (liq_penalty_bps / 1e4)
                else:
                    slip = adverse_px * (SL_SLIP_BPS / 1e4)
                px = adverse_px - slip if long else adverse_px + slip
                close_trade(pos, px, adverse_reason, "taker")
                pos = None; closed_this_bar = True
            elif tp_hit:
                close_trade(pos, fav_px, fav_reason, "maker")
                pos = None; closed_this_bar = True
            elif (be_seek_bars > 0 and pos["bars_held"] >= 1 + be_seek_bars) or \
                 (be_seek_bars == 0 and time_stop_bars > 0 and pos["bars_held"] >= time_stop_bars):
                slip = c[i] * (SCRATCH_SLIP_BPS / 1e4)
                px = c[i] - slip if long else c[i] + slip
                close_trade(pos, px, "time_stop", "maker")
                pos = None; closed_this_bar = True

        if pos is None and pending is None:
            if closed_this_bar and not same_bar_reentry:
                continue
            sig = decide_entry(i)
            if sig is not None:
                pending = sig

    tr = pd.DataFrame(trades)
    days = (ts[-1] - ts[0]) / 86_400_000
    if tr.empty:
        return {"name": name, "trades": 0}
    vol = tr["volume"].sum()
    net = tr["net"].sum()
    cost_per_1m = -net / vol * 1e6
    vol_per_day = vol / days
    days_to_5m = VOL_TARGET / vol_per_day if vol_per_day > 0 else float("inf")
    cost_at_5m = cost_per_1m * 5
    reach = days_to_5m <= CAMPAIGN_DAYS
    return {
        "name": name,
        "days": round(days, 1),
        "trades": len(tr),
        "trades_per_day": round(len(tr) / days, 1),
        "win_rate_pct": round((tr["net"] > 0).mean() * 100, 1),
        "exits": tr["reason"].value_counts().to_dict(),
        "net_bps_per_trade": round(tr["net"].mean() / NOTIONAL * 1e4, 2),
        "total_net_usd": round(net, 2),
        "volume_usd": round(vol, 0),
        "vol_per_day": round(vol_per_day, 0),
        "cost_per_1M": round(cost_per_1m, 1),
        "days_to_5M": round(days_to_5m, 1),
        "cost_at_5M": round(cost_at_5m, 0),
        "reaches_5M_in_30d": reach,
        "campaign_net": round(REWARD - cost_at_5m, 0) if reach else
        round(-cost_per_1m * (vol_per_day * CAMPAIGN_DAYS) / 1e6, 0),
        "worst_trade": round(tr["net"].min(), 2),
    }


VARIANTS = [
    dict(name="V1  alternation, NO time-stop (stopped bot)",
         alternate=True, time_stop_bars=0, tp_mode="fixed12", sl_atr_mult=1.5,
         min_range_bps=3.0, atr_min_usd=120.0, same_bar_reentry=False),
    dict(name="V3  momentum + 1-bar time-stop (running bot)",
         alternate=False, time_stop_bars=1, tp_mode="atr_floor", sl_atr_mult=6.0,
         min_range_bps=1.0, atr_min_usd=0.0, same_bar_reentry=True),
    dict(name="HYB-A  alternation + 1-bar time-stop",
         alternate=True, time_stop_bars=1, tp_mode="atr_floor", sl_atr_mult=6.0,
         min_range_bps=1.0, atr_min_usd=0.0, same_bar_reentry=True),
    dict(name="HYB-B  momentum, NO time-stop",
         alternate=False, time_stop_bars=0, tp_mode="fixed12", sl_atr_mult=1.5,
         min_range_bps=3.0, atr_min_usd=120.0, same_bar_reentry=False),
]


def report(rows: list, title: str) -> None:
    print(f"\n=== {title} ===")
    for r in rows:
        if r.get("trades", 0) == 0:
            print(f"{r['name']}: no trades"); continue
        flag = "✅ reaches 5M" if r["reaches_5M_in_30d"] else "❌ MISSES 5M in 30d"
        print(f"\n{r['name']}")
        print(f"  trades {r['trades']} ({r['trades_per_day']}/day)  WR {r['win_rate_pct']}%  exits {r['exits']}")
        print(f"  net {r['net_bps_per_trade']} bps/trade  total ${r['total_net_usd']}  worst ${r['worst_trade']}")
        print(f"  volume ${r['vol_per_day']:,.0f}/day → {r['days_to_5M']}d to 5M  {flag}")
        print(f"  bleed ${r['cost_per_1M']}/1M → ${r['cost_at_5M']} for 5M  → campaign net ${r['campaign_net']}")


V3_BASE = dict(alternate=False, time_stop_bars=1, tp_mode="atr_floor",
               min_range_bps=1.0, atr_min_usd=0.0, same_bar_reentry=True)


def sweep_brackets(df: pd.DataFrame) -> dict:
    """Phase 1: TP×ATR vs SL×ATR grid at 15× leverage. Decide by cost/1M."""
    print("\n=== PHASE 1 — bracket sweep (lev 15×, liq modeled) ===")
    print(f"{'tp×ATR':>7} {'sl×ATR':>7} {'cost/1M':>9} {'net bps/t':>10} "
          f"{'t/day':>6} {'vol/day':>10} {'tp':>5} {'scratch':>8} {'sl':>4} {'liq':>4} {'worst':>8}")
    best = None
    for tp_m in (0.5, 1.0, 1.5, 2.0):
        for sl_m in (1.5, 3.0, 6.0, 9.0, None):
            r = simulate(df, name="", **V3_BASE, sl_atr_mult=sl_m,
                         tp_atr_mult=tp_m, lev=15.0)
            ex = r["exits"]
            sl_lbl = f"{sl_m:g}" if sl_m is not None else "none"
            print(f"{tp_m:>7g} {sl_lbl:>7} {r['cost_per_1M']:>9} {r['net_bps_per_trade']:>10} "
                  f"{r['trades_per_day']:>6} {r['vol_per_day']:>10,.0f} {ex.get('tp', 0):>5} "
                  f"{ex.get('time_stop', 0):>8} {ex.get('sl', 0):>4} {ex.get('liq', 0):>4} "
                  f"{r['worst_trade']:>8}")
            key = (r["cost_per_1M"], -(r["vol_per_day"]))
            if best is None or key < best[0]:
                best = (key, tp_m, sl_m, r)
    _, tp_m, sl_m, r = best
    sl_lbl = f"{sl_m:g}×ATR" if sl_m is not None else "NO price stop"
    print(f"\nPhase-1 winner: TP {tp_m:g}×ATR, SL {sl_lbl} → ${r['cost_per_1M']}/1M")
    return {"tp_atr_mult": tp_m, "sl_atr_mult": sl_m}


def sweep_leverage(df: pd.DataFrame, bracket: dict) -> None:
    """Phase 2: leverage sweep with the winning bracket. Volume/fees are
    leverage-independent at fixed notional — what changes is the liquidation
    line, the margin locked, and (at high lev) liq events replacing the SL."""
    print("\n=== PHASE 2 — leverage sweep (winning bracket, liq modeled) ===")
    print(f"{'lev':>5} {'liq dist':>9} {'margin/trade':>13} {'cost/1M':>9} "
          f"{'sl':>4} {'liq':>4} {'worst':>9} {'campaign net':>13}")
    for lv in (10.0, 15.0, 20.0, 30.0, 50.0, 100.0):
        r = simulate(df, name="", **V3_BASE, **bracket, lev=lv)
        ex = r["exits"]
        liq_dist = max(1.0 / lv - 0.005, 0.001)
        print(f"{lv:>5g} {liq_dist*1e4:>7.0f}bp {NOTIONAL/lv:>12,.0f} {r['cost_per_1M']:>9} "
              f"{ex.get('sl', 0):>4} {ex.get('liq', 0):>4} {r['worst_trade']:>9} "
              f"{r['campaign_net']:>13}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--sweep", action="store_true",
                    help="run the bracket + leverage sweeps instead of v1-vs-v3")
    args = ap.parse_args()
    df = await fetch_bars(args.days)
    print(f"bars: {len(df)}  span: {pd.Timestamp(df['ts'].iloc[0], unit='ms')} → "
          f"{pd.Timestamp(df['ts'].iloc[-1], unit='ms')}")

    if args.sweep:
        bracket = sweep_brackets(df)
        sweep_leverage(df, bracket)
        # robustness: re-run the winning bracket on each half
        half = len(df) // 2
        for label, part in (("FIRST HALF", df.iloc[:half]), ("SECOND HALF", df.iloc[half:])):
            r = simulate(part.reset_index(drop=True), name="", **V3_BASE, **bracket, lev=15.0)
            print(f"{label}: cost ${r['cost_per_1M']}/1M  liq {r['exits'].get('liq', 0)}  "
                  f"sl {r['exits'].get('sl', 0)}  vol/day ${r['vol_per_day']:,.0f}")
        return 0

    full = [simulate(df, **v) for v in VARIANTS]
    report(full, f"FULL SAMPLE ({args.days} days)")

    half = len(df) // 2
    a = [simulate(df.iloc[:half].reset_index(drop=True), **v) for v in VARIANTS]
    b = [simulate(df.iloc[half:].reset_index(drop=True), **v) for v in VARIANTS]
    report(a, "FIRST HALF (robustness)")
    report(b, "SECOND HALF (robustness)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
