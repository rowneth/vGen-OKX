"""1m live strategy grid — $30 capital, 5% margin.

Shows daily volume, rebate income, and key metrics for the current live
config (TP=8, SL=50, trend_break=on) and nearby TP/SL candidates.
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
        "trend_break": {"enabled": True, "min_bars_held": 3, "adverse_bps": 20.0},
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


def cfg(tp, sl, min_range=3.0, max_range=40.0, tb_adv=20.0, max_hold=999):
    c = copy.deepcopy(BASE)
    c["farmer"]["tp_bps"] = tp
    c["farmer"]["sl_bps"] = sl
    c["farmer"]["entry"]["min_bar_range_bps"] = min_range
    c["farmer"]["entry"]["max_bar_range_bps"] = max_range
    c["farmer"]["trend_break"]["adverse_bps"] = tb_adv
    c["farmer"]["max_hold_bars"] = max_hold
    return c


def main():
    df = pd.read_parquet(DATA_1M)
    df.columns = [c.lower() for c in df.columns]
    t = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    span_days = (t.max() - t.min()).total_seconds() / 86400
    print(f"1m data: {len(df):,} rows  span: {span_days:.1f} days\n")

    # Notional per trade at $30 cap, 5% margin, dynamic leverage ~100x:
    # margin = 30 * 0.05 = $1.50  → notional ~$150 at 100x
    print("Notional per trade: ~$150 (5% × $30 × 100x leverage)")
    print("Maker fee per trade: $0.015  Rebate (70%): $0.0105 per trade")
    print()

    runs = [
        # label, config
        ("LIVE NOW  TP=8  SL=50  TB=20",  cfg(8,  50,  tb_adv=20.0)),  # ← current live
        ("          TP=8  SL=50  TB=off", cfg(8,  50,  tb_adv=9999)),
        ("          TP=8  SL=30  TB=18",  cfg(8,  30,  tb_adv=18.0)),
        ("          TP=8  SL=30  TB=off", cfg(8,  30,  tb_adv=9999)),
        ("          TP=10 SL=50  TB=20",  cfg(10, 50,  tb_adv=20.0)),
        ("          TP=10 SL=30  TB=18",  cfg(10, 30,  tb_adv=18.0)),
        ("          TP=12 SL=50  TB=20",  cfg(12, 50,  tb_adv=20.0)),
        ("          TP=12 SL=30  TB=18",  cfg(12, 30,  tb_adv=18.0)),
        ("          TP=8  SL=50  range=2",cfg(8,  50,  min_range=2.0, tb_adv=20.0)),
        ("          TP=8  SL=50  range=4",cfg(8,  50,  min_range=4.0, tb_adv=20.0)),
        ("          TP=8  SL=50  hold=5", cfg(8,  50,  max_hold=5,   tb_adv=20.0)),
    ]

    fmt = "{:<32} {:>6} {:>6} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8}"
    hdr = fmt.format("config", "trades", "WR%", "gap%",
                     "vol$/day", "reb$/day", "gross$", "end+reb$", "avg_L")
    print(hdr)
    print("─" * len(hdr))

    for label, c in runs:
        r = run_one(df, 0, 0, override_cfg=c)
        td = r["trades"]
        wr = r["win_rate_pct"]
        gap = r["wr_gap_pct"]
        vol_day = r["total_volume_usd"] / span_days
        reb_day = r["total_rebate"] / span_days
        gross = r["gross_pnl"]
        end_r = r["end_equity_with_rebate"]
        avg_l = r.get("avg_loss_bps", 0.0)
        star = " ◄ LIVE" if "LIVE NOW" in label else ""
        print(fmt.format(
            label,
            td,
            f"{wr:.1f}",
            f"{gap:+.1f}",
            f"{vol_day:,.0f}",
            f"{reb_day:.3f}",
            f"{gross:+.2f}",
            f"{end_r:.2f}",
            f"{avg_l:.1f}",
        ) + star)

    print()
    print("vol$/day  = notional volume traded per day")
    print("reb$/day  = rebate income per day (70% of maker fees)")
    print("end+reb$  = final equity + cumulative rebate over full 25-day window")
    print("avg_L     = avg loss size in bps (lower = tighter stop cuts losses sooner)")
    print(f"\nData window: {span_days:.1f} days")


if __name__ == "__main__":
    main()
