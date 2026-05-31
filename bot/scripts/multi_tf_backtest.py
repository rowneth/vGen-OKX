"""Multi-TF parallel backtest: 1m + 3m + 5m sharing one capital pool.

Strategy per session:
  - Entry mode:     micro_momentum  (same as live)
  - TP:             10 bps
  - SL:             50 bps
  - Trend-break:    OFF
  - Session filter: skip UTC hours [0,1,6,12,17,19,21,22]  (Option-A filter)
  - Margin/trade:   2.5% of current shared equity
  - Alternate dir:  ON

Capital sharing:
  All 3 sessions draw from one $30 pool.  Before feeding a flat session a bar,
  session.equity is synchronised to shared equity so sizing is always 2.5% of
  the current combined balance.  Open-fees and exit P&L flow through shared
  equity via event callbacks.

Concurrency:
  Each TF can hold at most one position at a time → natural max = 3 concurrent
  open trades (one per TF).  Observed peak is reported.

Usage:
    python scripts/multi_tf_backtest.py
"""

from __future__ import annotations

import pathlib
import sys
from typing import List, Tuple

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from execution.volume_farmer import FarmerEvent, VolumeFarmerSession  # noqa: E402
from backtest_volume_farmer import run_one                              # noqa: E402

DATA_FILE   = PROJECT_ROOT / "data/historical/BTC_USDT_1m.parquet"
CAPITAL_USD = 30.0
SKIP_HOURS  = [0, 1, 6, 12, 17, 19, 21, 22]
MAKER       = 0.0001
TAKER       = 0.0005
REBATE_PCT  = 0.70

# ─── Config factory ──────────────────────────────────────────────────────────

def _cfg(tf: str, min_range: float, session_filter: bool = True,
         capital: float = CAPITAL_USD) -> dict:
    entry: dict = {
        "mode": "micro_momentum",
        "min_bar_range_bps": min_range,
        "max_bar_range_bps": 40.0,
    }
    if session_filter:
        entry["skip_hours"] = SKIP_HOURS
    return {
        "app": {"timezone": "UTC"},
        "exchange": {"symbol": "BTC_USDT", "timeframe": tf},
        "fees": {"maker": MAKER, "taker": TAKER, "rebate_pct": REBATE_PCT},
        "farmer": {
            "capital_usd": capital,
            "leverage": 0,
            "margin_fraction_per_trade": 0.025,   # 2.5% of current equity
            "sizing": {
                "dynamic_leverage": True,
                "risk_per_trade_pct": 0.025,
                "max_leverage": 125,
                "min_leverage": 5,
            },
            "tp_bps": 10.0,
            "sl_bps": 50.0,
            "max_hold_bars": 999,
            "entry": entry,
            "alternate_direction": True,
            "trend_break": {"enabled": False},
        },
        "risk": {
            "daily_loss_limit_pct": 0.99,
            "max_drawdown_pct": 0.99,
            "consecutive_losses_limit": 9999,
            "consecutive_losses_cooldown_bars": 0,
            "stop_on_volume_target": False,
        },
        "target": {"volume_usd": 999_999_999},
    }


# ─── Resample helper ─────────────────────────────────────────────────────────

def _resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample tz-aware 1m DataFrame; drop the trailing forming bar."""
    out = (
        df.set_index("open_time")
        .resample(freq, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
        .reset_index()
    )
    if len(out) > 1:
        out = out.iloc[:-1]   # drop last (possibly forming) bar
    return out.reset_index(drop=True)


# ─── Parallel coordinator ────────────────────────────────────────────────────

def run_multi_tf(df_1m: pd.DataFrame) -> dict:
    """Run 1m + 3m + 5m sessions in parallel sharing one capital pool."""
    df_1m = df_1m.copy()
    # Ensure open_time is tz-aware datetime (UTC)
    if not pd.api.types.is_datetime64_any_dtype(df_1m["open_time"]):
        df_1m["open_time"] = pd.to_datetime(df_1m["open_time"], unit="ms", utc=True)
    elif getattr(df_1m["open_time"].dt, "tz", None) is None:
        df_1m["open_time"] = df_1m["open_time"].dt.tz_localize("UTC")

    df_3m = _resample(df_1m, "3min")
    df_5m = _resample(df_1m, "5min")
    print(f"    Bars — 1m: {len(df_1m):,}  |  3m: {len(df_3m):,}  |  5m: {len(df_5m):,}")

    # Create sessions  (each starts knowing the full capital for reference;
    # actual equity is synced from shared pool before every flat-session candle)
    s1 = VolumeFarmerSession(config=_cfg("1m", 3.0))
    s3 = VolumeFarmerSession(config=_cfg("3m", 4.0))
    s5 = VolumeFarmerSession(config=_cfg("5m", 4.0))

    # ── Shared state ─────────────────────────────────────────────────────────
    # eq[0]   = running shared equity (fees/pnl flow through callbacks)
    # peak[0] = peak shared equity (for max-drawdown tracking)
    eq   = [float(CAPITAL_USD)]
    peak = [float(CAPITAL_USD)]

    acc = {
        "wins": 0, "losses": 0,
        "gross_pnl": 0.0,
        "real_fees": 0.0,
        "volume": 0.0,         # round-trip (entry notional + exit notional)
        "max_dd": 0.0,
        "by_tf": {"1m": 0, "3m": 0, "5m": 0},
        "reasons": {},
        "win_bps": [],
        "loss_bps": [],
        "open_now": 0,
        "peak_concurrent": 0,
    }

    def _handler(tf: str):
        def cb(evt: FarmerEvent):
            p = evt.payload
            if evt.kind == "entry":
                notional = float(p["notional"])
                open_fee = float(p["open_fee"])
                eq[0] -= open_fee
                acc["real_fees"] += open_fee
                acc["volume"] += notional          # entry leg
                acc["open_now"] += 1
                acc["peak_concurrent"] = max(acc["peak_concurrent"], acc["open_now"])

            elif evt.kind == "exit":
                notional  = float(p["notional"])
                gross_pnl = float(p["gross_pnl"])
                close_fee = notional * TAKER
                open_fee  = notional * MAKER       # for net-per-trade accounting
                reason    = p.get("reason", "?")

                eq[0] += gross_pnl - close_fee
                acc["real_fees"] += close_fee
                acc["volume"] += notional          # exit leg
                acc["gross_pnl"] += gross_pnl

                peak[0] = max(peak[0], eq[0])
                if peak[0] > 0:
                    dd = (peak[0] - eq[0]) / peak[0] * 100
                    acc["max_dd"] = max(acc["max_dd"], dd)

                # True net = gross - both legs of fees
                true_net  = gross_pnl - (open_fee + close_fee)
                bps       = gross_pnl / notional * 10_000 if notional > 0 else 0.0
                if true_net > 0:
                    acc["wins"] += 1
                    acc["win_bps"].append(bps)
                else:
                    acc["losses"] += 1
                    acc["loss_bps"].append(-bps)

                acc["by_tf"][tf] = acc["by_tf"].get(tf, 0) + 1
                acc["reasons"][reason] = acc["reasons"].get(reason, 0) + 1
                acc["open_now"] = max(0, acc["open_now"] - 1)
        return cb

    s1.event_callback = _handler("1m")
    s3.event_callback = _handler("3m")
    s5.event_callback = _handler("5m")

    sess_map = {"1m": s1, "3m": s3, "5m": s5}
    df_map   = {"1m": df_1m, "3m": df_3m, "5m": df_5m}

    # TF durations used to compute bar close-time for correct temporal ordering
    dur = {
        "1m": pd.Timedelta("1min"),
        "3m": pd.Timedelta("3min"),
        "5m": pd.Timedelta("5min"),
    }
    # Secondary sort key so 1m bars are processed before longer-TF bars that
    # close at the same timestamp (e.g. 1m bar T+2min and 3m bar T both close
    # at T+3min — process 1m first).
    tf_order = {"1m": 0, "3m": 1, "5m": 2}

    # Build merged event list: (close_time, tf_order, tf_name, bar_index)
    events: List[Tuple] = []
    for tf, df in df_map.items():
        for i in range(2, len(df)):
            close_ts = df["open_time"].iloc[i] + dur[tf]
            events.append((close_ts, tf_order[tf], tf, i))
    events.sort(key=lambda x: (x[0], x[1]))

    # Feed bars in chronological order
    for close_ts, _ord, tf, idx in events:
        sess = sess_map[tf]
        df   = df_map[tf]
        # Sync shared equity to session when flat (so sizing = 2.5% of shared pool)
        if sess.position is None and eq[0] > 0:
            sess.equity      = eq[0]
            sess.peak_equity = max(sess.peak_equity, eq[0])
        sess.on_new_candle(df.iloc[: idx + 1])

    # Gather per-session diagnostic counts
    acc["session_blocks"]      = {tf: sess_map[tf]._session_blocks      for tf in sess_map}
    acc["entries_considered"]  = {tf: sess_map[tf]._entries_considered  for tf in sess_map}
    acc["end_equity"]          = eq[0]
    acc["start_equity"]        = CAPITAL_USD
    return acc


# ─── Main / report ───────────────────────────────────────────────────────────

def main() -> None:
    print(f"\nLoading {DATA_FILE} ...")
    df = pd.read_parquet(DATA_FILE)
    df.columns = [c.lower() for c in df.columns]
    first = pd.Timestamp(df["open_time"].iloc[0])
    last  = pd.Timestamp(df["open_time"].iloc[-1])
    days  = max((last - first).total_seconds() / 86400, 1)
    print(f"Loaded {len(df):,} bars  ({first.date()} → {last.date()})  [{days:.1f} days]\n")

    # ── Multi-TF run ─────────────────────────────────────────────────────────
    print("Running multi-TF parallel (1m + 3m + 5m, shared $30, Option-A filter)...")
    r = run_multi_tf(df)

    trades    = r["wins"] + r["losses"]
    wr        = r["wins"] / trades * 100 if trades else 0.0
    rebate    = r["real_fees"] * REBATE_PCT
    true_net  = r["gross_pnl"] - r["real_fees"] + rebate
    end_w_reb = r["end_equity"] + rebate          # rebate paid separately by MEXC
    trades_d  = trades / days
    vol_30d   = r["volume"] / days * 30.0
    bonus_mo  = vol_30d / 1e6 * 80.0
    net_mo    = true_net / days * 30.0
    adj_mo    = net_mo + bonus_mo
    cap_1k    = CAPITAL_USD * 1000.0 / adj_mo if adj_mo > 0 else float("inf")
    be_wr     = 50.0 / (10.0 + 50.0) * 100       # break-even WR at TP=10/SL=50
    avg_win   = sum(r["win_bps"])  / len(r["win_bps"])  if r["win_bps"]  else 0.0
    avg_loss  = sum(r["loss_bps"]) / len(r["loss_bps"]) if r["loss_bps"] else 0.0

    W = 74
    print("\n" + "=" * W)
    print("  MULTI-TF PARALLEL: 1m + 3m + 5m  (shared $30 capital)")
    print(f"  TP=10 bps  SL=50 bps  Trend-break=OFF  Session filter: skip UTC {SKIP_HOURS}")
    print(f"  Each trade: 2.5% margin of current shared equity  |  Max 3 concurrent")
    print("=" * W)
    print(f"\n  Combined ({days:.1f} days):  {trades:,} trades  ({trades_d:.1f}/day)")
    print(f"  Win rate:            {wr:.2f}%   (break-even: {be_wr:.1f}%)")
    print(f"  Avg win bps:         {avg_win:.2f}   avg loss bps: {avg_loss:.2f}")
    print(f"  Max drawdown:        {r['max_dd']:.1f}%")
    print(f"  Peak concurrent:     {r['peak_concurrent']} positions open simultaneously")

    print(f"\n  By timeframe:         trades   /day   session_blocks(filter %)")
    for tf in ["1m", "3m", "5m"]:
        n  = r["by_tf"][tf]
        bl = r["session_blocks"][tf]
        ec = r["entries_considered"][tf]
        total_opp = bl + ec
        pct = bl / total_opp * 100 if total_opp > 0 else 0.0
        print(f"    {tf}:                {n:>5,}  {n/days:>5.1f}   {bl:,} bars blocked ({pct:.0f}%)")

    print(f"\n  Exit reasons:         {r['reasons']}")
    print(f"\n  Equity:")
    print(f"    Start:              ${r['start_equity']:.2f}")
    print(f"    End (no rebate):    ${r['end_equity']:.2f}")
    print(f"    Rebate (70%):       +${rebate:.2f}")
    print(f"    End (incl rebate):  ${end_w_reb:.2f}")
    print(f"    True net income:    ${true_net:+.4f}")
    print(f"\n  30-day extrapolation:")
    print(f"    Volume / 30d:       ${vol_30d:,.0f}")
    print(f"    MEXC $80/M bonus:   ${bonus_mo:.2f} / month")
    print(f"    Trading net / mo:   ${net_mo:.2f} / month")
    print(f"    Total adj / mo:     ${adj_mo:.2f} / month")
    if adj_mo > 0:
        print(f"    Capital for $1k/mo: ${cap_1k:,.0f}")
    else:
        print(f"    Capital for $1k/mo: n/a  (net negative)")

    # ── Comparison baselines ─────────────────────────────────────────────────
    print("\n" + "─" * W)
    print("  Running comparison baselines (single-session, same 1m data)...")

    baselines = [
        ("1m, no filter, TP=10 SL=50 no TB",  _cfg("1m", 3.0, session_filter=False)),
        ("1m, filter ON, TP=10 SL=50 no TB",  _cfg("1m", 3.0, session_filter=True)),
        ("1m live  (TP=8 SL=50 TB=ON no filt)", None),   # built below
        ("3m, no filter, TP=10 SL=50 no TB",  _cfg("3m", 4.0, session_filter=False)),
        ("5m, no filter, TP=10 SL=50 no TB",  _cfg("5m", 4.0, session_filter=False)),
    ]
    # Build live config
    live_cfg = _cfg("1m", 3.0, session_filter=False)
    live_cfg["farmer"]["tp_bps"] = 8.0
    live_cfg["farmer"]["trend_break"] = {"enabled": True, "min_bars_held": 3, "adverse_bps": 20.0}
    baselines[2] = ("1m live  (TP=8 SL=50 TB=ON no filt)", live_cfg)

    # Build 3m / 5m DataFrames for single-session baselines
    df_3m_b = _resample(df.copy().assign(
        open_time=lambda d: pd.to_datetime(d["open_time"], utc=True)
        if not pd.api.types.is_datetime64_any_dtype(d["open_time"])
        else d["open_time"]
    ), "3min")
    df_5m_b = _resample(df.copy().assign(
        open_time=lambda d: pd.to_datetime(d["open_time"], utc=True)
        if not pd.api.types.is_datetime64_any_dtype(d["open_time"])
        else d["open_time"]
    ), "5min")

    data_map = {
        "1m, no filter, TP=10 SL=50 no TB":    df,
        "1m, filter ON, TP=10 SL=50 no TB":    df,
        "1m live  (TP=8 SL=50 TB=ON no filt)": df,
        "3m, no filter, TP=10 SL=50 no TB":    df_3m_b,
        "5m, no filter, TP=10 SL=50 no TB":    df_5m_b,
    }

    print(f"\n  {'Config':<40} {'Trd/d':>5} {'WR%':>6} {'End+Reb':>8} {'AdjNet/mo':>10} {'Vol/30d':>10}")
    print("  " + "─" * 72)

    # Multi-TF result row first
    cap_s = f"${cap_1k:,.0f}" if adj_mo > 0 else "n/a"
    print(f"  {'Multi-TF 1m+3m+5m (filter ON)':<40} {trades_d:>5.1f} "
          f"{wr:>6.1f} ${end_w_reb:>7.2f} ${adj_mo:>9.2f} ${vol_30d:>9,.0f}  ← {cap_s}")

    for label, cfg in baselines:
        data = data_map[label]
        br = run_one(data, 0, 0, override_cfg=cfg)
        b_wr    = br["win_rate_pct"]
        b_end   = br["end_equity_with_rebate"]
        b_true  = br["true_net_income"]
        b_vol   = br["total_volume_usd"] / days * 30
        b_bonus = b_vol / 1e6 * 80
        b_adj   = b_true / days * 30 + b_bonus
        b_tpd   = br["trades"] / days
        b_cap   = CAPITAL_USD * 1000 / b_adj if b_adj > 0 else float("inf")
        cap_b   = f"${b_cap:,.0f}" if b_adj > 0 else "n/a"
        print(f"  {label:<40} {b_tpd:>5.1f} {b_wr:>6.1f} ${b_end:>7.2f} ${b_adj:>9.2f} ${b_vol:>9,.0f}  ← {cap_b}")

    print("  " + "─" * 72)
    print(f"\n  Notes:")
    print(f"    Break-even WR @ TP=10/SL=50 (after real fees) = {be_wr:.1f}%")
    print(f"    AdjNet/mo = (trading net + $80/M volume bonus) × 30d projection")
    print(f"    Multi-TF volume is from 3 sources: each TF contributes independently")
    print(f"    Capital-for-$1k uses AdjNet/mo from $30 scaled linearly")
    print(f"    Peak concurrent positions observed: {r['peak_concurrent']} / 3 max")
    print()


if __name__ == "__main__":
    main()
