"""Backtest 1m BTC_USDT data with both the current 5m config (direct port)
and a 1m-native scaled config (TP/SL proportional to bar size).

Runs:
  A. Current optimal (TP=8bps, SL=50bps) -- same as live, on 1m bars
  B. Scaled A by 1/5  (TP=1.6, SL=10bps) -- proportional to 1m bar size
  C. Grid of 1m-native TP/SL combos chosen around the 1m bar range stats

Key question: does 1m produce more volume/rebate, or does the extra noise
eat the edge?
"""
from __future__ import annotations

import copy
import pathlib
import sys

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from backtest_volume_farmer import run_one  # noqa: E402

DATA_1M = PROJECT_ROOT / "data/historical/BTC_USDT_1m.parquet"
DATA_5M = PROJECT_ROOT / "data/historical/BTC_USDT_5m.parquet"

BASE = {
    "app": {"timezone": "Asia/Colombo"},
    "exchange": {"symbol": "BTC_USDT", "timeframe": "1m"},
    "fees": {"maker": 0.0001, "taker": 0.0005, "rebate_pct": 0.70},
    "farmer": {
        "capital_usd": 30.0,
        "leverage": 0,
        "margin_fraction_per_trade": 0.05,
        "sizing": {"dynamic_leverage": True, "risk_per_trade_pct": 0.025,
                   "max_leverage": 125, "min_leverage": 5},
        "tp_bps": 8.0,
        "sl_bps": 50.0,
        "max_hold_bars": 999,
        "entry": {"mode": "micro_momentum",
                  "min_bar_range_bps": 3.0, "max_bar_range_bps": 40.0},
        "alternate_direction": True,
        "trend_break": {"enabled": False},
    },
    "risk": {
        "daily_loss_limit_pct": 0.50,
        "max_drawdown_pct": 0.95,
        "consecutive_losses_limit": 9999,
        "consecutive_losses_cooldown_bars": 0,
        "stop_on_volume_target": False,
    },
    "target": {"volume_usd": 999_999_999},
}


def cfg(tp, sl, min_range=1.0, max_range=20.0, max_hold=999):
    c = copy.deepcopy(BASE)
    c["farmer"]["tp_bps"] = tp
    c["farmer"]["sl_bps"] = sl
    c["farmer"]["entry"]["min_bar_range_bps"] = min_range
    c["farmer"]["entry"]["max_bar_range_bps"] = max_range
    c["farmer"]["max_hold_bars"] = max_hold
    return c


def main():
    print(f"Loading 1m data: {DATA_1M}")
    df1m = pd.read_parquet(DATA_1M)
    df1m.columns = [c.lower() for c in df1m.columns]
    t = pd.to_datetime(df1m["open_time"], unit="ms", utc=True)
    span_days = (t.max() - t.min()).total_seconds() / 86400
    range_bps = (abs(df1m.close.astype(float) - df1m.open.astype(float))
                 / df1m.open.astype(float) * 10_000)
    print(f"Rows: {len(df1m):,}  span: {span_days:.1f} days")
    print(f"Bar range bps — median: {range_bps.median():.2f}  mean: {range_bps.mean():.2f}"
          f"  p90: {range_bps.quantile(0.9):.2f}  p99: {range_bps.quantile(0.99):.2f}")
    pct_ge3 = (range_bps >= 3).mean() * 100
    pct_ge1 = (range_bps >= 1).mean() * 100
    print(f"Bars >=3bps: {pct_ge3:.1f}%   >=1bps: {pct_ge1:.1f}%")
    print()

    print("Loading 5m baseline for comparison:")
    df5m = pd.read_parquet(DATA_5M)
    df5m.columns = [c.lower() for c in df5m.columns]
    print(f"Rows: {len(df5m):,}")
    print()

    runs = [
        # label, df, cfg
        ("5m-baseline  TP=8  SL=50 (live config)", df5m,
         cfg(8, 50, min_range=3.0, max_range=40.0)),
        # Direct port — same TP/SL on 1m bars (expected to be garbage)
        ("1m-DIRECT    TP=8  SL=50 (no rescale)", df1m,
         cfg(8, 50, min_range=3.0, max_range=40.0)),
        # Scaled to ~1/5 of 5m bar size
        ("1m-scaled/5  TP=2  SL=10", df1m,
         cfg(2, 10, min_range=1.0, max_range=8.0)),
        # Grid of 1m-native candidates
        ("1m-grid TP=3  SL=8 ", df1m, cfg(3, 8,  min_range=1.0, max_range=8.0)),
        ("1m-grid TP=3  SL=6 ", df1m, cfg(3, 6,  min_range=1.0, max_range=6.0)),
        ("1m-grid TP=4  SL=10", df1m, cfg(4, 10, min_range=1.5, max_range=10.0)),
        ("1m-grid TP=2  SL=6 ", df1m, cfg(2, 6,  min_range=1.0, max_range=6.0)),
        ("1m-grid TP=5  SL=15", df1m, cfg(5, 15, min_range=2.0, max_range=15.0)),
        ("1m-grid TP=2  SL=8  hold<=5", df1m,
         cfg(2, 8, min_range=1.0, max_range=8.0, max_hold=5)),
        ("1m-grid TP=3  SL=9  hold<=5", df1m,
         cfg(3, 9, min_range=1.5, max_range=9.0, max_hold=5)),
    ]

    fmt = "{:<35} {:>6} {:>7} {:>7} {:>8} {:>8} {:>8} {:>11}"
    header = fmt.format("label", "trades", "WR%", "gap%", "gross$",
                        "rebate$", "end+reb", "volume$")
    print(header)
    print("─" * len(header))

    for label, df, c in runs:
        r = run_one(df, 0, 0, override_cfg=c)
        print(fmt.format(
            label,
            r["trades"],
            f"{r['win_rate_pct']:.1f}",
            f"{r['wr_gap_pct']:+.1f}",
            f"{r['gross_pnl']:+.2f}",
            f"{r['total_rebate']:+.2f}",
            f"{r['end_equity_with_rebate']:.2f}",
            f"{r['total_volume_usd']:>10,.0f}",
        ))

    print()
    print("NOTE: 1m data from MEXC is capped — check 'span' above to see")
    print("how many actual days are in the 1m dataset vs 5m (12 months).")


if __name__ == "__main__":
    main()
