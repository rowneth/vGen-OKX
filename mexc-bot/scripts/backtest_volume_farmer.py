"""Quick backtest of VolumeFarmerSession over historical BTC_USDT 5m data.

Runs multiple TP configurations and prints a comparison table.

Usage:
    python scripts/backtest_volume_farmer.py
    python scripts/backtest_volume_farmer.py --tp-list 5 8 10 15 20
    python scripts/backtest_volume_farmer.py --config config/config_volume_farmer_optimal.yaml
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from execution.volume_farmer import FarmerEvent, VolumeFarmerSession  # noqa: E402


DATA_FILE = PROJECT_ROOT / "data/historical/BTC_USDT_5m.parquet"

# Base config matching config_volume_farmer_optimal.yaml
BASE_CONFIG = {
    "app": {"timezone": "Asia/Colombo"},
    "exchange": {"symbol": "BTC_USDT", "timeframe": "5m"},
    "fees": {
        "maker": 0.0001,   # 0.01% — confirmed live
        "taker": 0.0005,   # 0.05% — confirmed live
        "rebate_pct": 0.70,
    },
    "farmer": {
        "capital_usd": 30.0,
        "leverage": 0,
        "margin_fraction_per_trade": 0.05,
        "sizing": {
            "dynamic_leverage": True,
            "risk_per_trade_pct": 0.025,
            "max_leverage": 125,
            "min_leverage": 5,
        },
        "tp_bps": 10.0,    # overridden per run
        "sl_bps": 50.0,
        "max_hold_bars": 999,
        "entry": {
            "mode": "micro_momentum",
            "min_bar_range_bps": 4.0,
            "max_bar_range_bps": 40.0,
        },
        "alternate_direction": True,
    },
    "risk": {
        "daily_loss_limit_pct": 0.50,
        "max_drawdown_pct": 0.95,
        "consecutive_losses_limit": 999,
        "consecutive_losses_cooldown_bars": 0,
        "stop_on_volume_target": False,  # run full dataset
    },
    "target": {"volume_usd": 999_999_999},
}


def run_one(df: pd.DataFrame, tp_bps: float, sl_bps: float, capital_usd: float = 30.0) -> dict:
    """Run the session over full df, return summary stats."""
    import copy
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["farmer"]["tp_bps"] = tp_bps
    cfg["farmer"]["sl_bps"] = sl_bps
    cfg["farmer"]["capital_usd"] = capital_usd

    session = VolumeFarmerSession(config=cfg)

    wins = 0
    losses = 0
    total_pnl = 0.0          # gross price-move P&L
    total_real_fees = 0.0    # CORRECTED: open=maker, close=ALWAYS taker
    total_volume = 0.0

    MAKER_RATE = 0.0001      # 0.01%
    TAKER_RATE = 0.0005      # 0.05%

    def on_event(evt: FarmerEvent) -> None:
        nonlocal wins, losses, total_pnl, total_real_fees, total_volume
        if evt.kind == "exit":
            p = evt.payload
            notional = float(p.get("notional", 0))
            gross_pnl = float(p.get("gross_pnl", 0))

            # Real fees: open=maker, close=taker (TP triggers MARKET order on MEXC)
            real_open_fee = notional * MAKER_RATE
            real_close_fee = notional * TAKER_RATE
            real_total_fee = real_open_fee + real_close_fee
            real_net = gross_pnl - real_total_fee

            if real_net > 0:
                wins += 1
            else:
                losses += 1
            total_pnl += gross_pnl
            total_real_fees += real_total_fee
            total_volume += notional * 2  # entry + exit

    session.event_callback = on_event

    # Feed bars one at a time with growing history window
    for i in range(2, len(df)):
        history = df.iloc[:i]
        session.on_new_candle(history)
        if session.halted:
            break

    trades = wins + losses
    wr = wins / trades if trades else 0
    break_even_wr = sl_bps / (tp_bps + sl_bps)

    total_rebate = total_real_fees * 0.70
    # True wallet end = capital + gross_pnl - real_fees   (rebate paid separately)
    end_no_rebate = capital_usd + total_pnl - total_real_fees
    end_with_rebate = end_no_rebate + total_rebate
    true_net = total_pnl - total_real_fees + total_rebate

    return {
        "tp_bps": tp_bps,
        "sl_bps": sl_bps,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": wr * 100,
        "break_even_wr_pct": break_even_wr * 100,
        "wr_gap_pct": (wr - break_even_wr) * 100,
        "gross_pnl": total_pnl,
        "total_gross_fees": total_real_fees,
        "total_rebate": total_rebate,
        "net_fees_paid": total_real_fees * 0.30,
        "true_net_income": true_net,
        "end_equity_no_rebate": end_no_rebate,
        "end_equity_with_rebate": end_with_rebate,
        "net_per_trade": (true_net / trades) if trades else 0,
        "total_volume_usd": total_volume,
        "halted": session.halted,
        "halt_reason": session.halt_reason,
    }


def _bar(value: float, max_val: float, width: int = 20, positive_char: str = "█", negative_char: str = "▓") -> str:
    if max_val == 0:
        return ""
    ratio = min(abs(value) / max_val, 1.0)
    filled = int(ratio * width)
    char = positive_char if value >= 0 else negative_char
    return char * filled


def main() -> None:
    parser = argparse.ArgumentParser(description="Volume farmer TP/SL backtest comparison")
    parser.add_argument("--tp-list", type=float, nargs="+", default=[5, 7, 8, 10, 12, 15, 20],
                        help="TP values in bps to test")
    parser.add_argument("--sl", type=float, default=50.0, help="SL in bps (fixed)")
    parser.add_argument("--rows", type=int, default=None, help="Limit to last N rows")
    parser.add_argument("--capitals", type=float, nargs="+", default=None,
                        help="Compare different starting capitals at fixed TP (--tp must be single)")
    parser.add_argument("--tp", type=float, default=15.0,
                        help="TP in bps used when --capitals is set")
    args = parser.parse_args()

    print(f"\nLoading {DATA_FILE} ...")
    df = pd.read_parquet(DATA_FILE)
    # Normalise column names
    df.columns = [c.lower() for c in df.columns]
    if args.rows:
        df = df.iloc[-args.rows:]
    print(f"Loaded {len(df):,} bars  ({df.index[0]} → {df.index[-1]})\n")

    # ─── capital-comparison mode ─────────────────────────────────────
    if args.capitals:
        print(f"  Fixed TP={args.tp}bps / SL={args.sl}bps — varying starting capital\n")
        cap_results = []
        for cap in args.capitals:
            print(f"  Running capital=${cap} ...", end="", flush=True)
            r = run_one(df, tp_bps=args.tp, sl_bps=args.sl, capital_usd=cap)
            r["start_capital"] = cap
            cap_results.append(r)
            roi = ((r["end_equity_with_rebate"] - cap) / cap) * 100
            print(f"  end=${r['end_equity_with_rebate']:.2f}  ROI={roi:+.2f}%  trades={r['trades']}")

        W = 13
        cols = ["Start $", "Trades", "WR%", "Gross P&L", "Fees Paid",
                "Rebate 70%", "Wallet End", "End+Rebate", "P&L ($)", "ROI %", "Volume $"]
        sep = "─" * (W * len(cols) + len(cols))
        print()
        print("=" * len(sep))
        print(f"  VOLUME FARMER — CAPITAL SCALING (TP={args.tp}bps, SL={args.sl}bps, real fees)")
        print("=" * len(sep))
        print("  ".join(c.rjust(W) for c in cols))
        print(sep)
        for r in cap_results:
            cap = r["start_capital"]
            end = r["end_equity_with_rebate"]
            pnl = end - cap
            roi = pnl / cap * 100
            marker = "✅" if pnl >= 0 else "❌"
            row = "  ".join([
                f"${cap:,.0f} {marker}".rjust(W),
                f"{r['trades']:,}".rjust(W),
                f"{r['win_rate_pct']:.1f}%".rjust(W),
                f"${r['gross_pnl']:+.2f}".rjust(W),
                f"${r['total_gross_fees']:.2f}".rjust(W),
                f"${r['total_rebate']:.2f}".rjust(W),
                f"${r['end_equity_no_rebate']:.2f}".rjust(W),
                f"${end:.2f}".rjust(W),
                f"${pnl:+.2f}".rjust(W),
                f"{roi:+.2f}%".rjust(W),
                f"${r['total_volume_usd']:,.0f}".rjust(W),
            ])
            print(row)
        print(sep)
        print("\n  NOTE: position size scales linearly with capital (margin_fraction=5%,")
        print("        dynamic leverage capped at 125×), so volume & P&L both scale ~linearly.\n")
        return

    results = []
    for tp in args.tp_list:
        print(f"  Running TP={tp}bps / SL={args.sl}bps ...", end="", flush=True)
        r = run_one(df, tp_bps=tp, sl_bps=args.sl)
        results.append(r)
        marker = "✅" if r["net_pnl_after_rebate"] >= 0 else "❌"
        print(f"  {marker}  {r['trades']} trades, WR={r['win_rate_pct']:.1f}%, net={r['net_pnl_after_rebate']:+.2f}")

    # ─── pretty table ────────────────────────────────────────────────
    W = 12
    cols = ["TP", "Trades", "WR%", "BE-WR%", "WR Gap", "Gross P&L", "Fees Paid",
            "Rebate (70%)", "True Net", "End Balance", "End+Rebate", "Volume $"]
    sep = "─" * (W * len(cols) + len(cols))
    header = "  ".join(c.rjust(W) for c in cols)
    print()
    print("=" * len(sep))
    print("  VOLUME FARMER BACKTEST — TP SENSITIVITY (SL=50bps, real fees: maker=0.01% / taker=0.05%)")
    print("  Starting capital: $30.00")
    print("=" * len(sep))
    print(header)
    print(sep)
    for r in results:
        marker = " ✅" if r["end_equity_with_rebate"] >= 30.0 else " ❌"
        row = "  ".join([
            f"{r['tp_bps']:.0f}bps{marker}".rjust(W),
            f"{r['trades']:,}".rjust(W),
            f"{r['win_rate_pct']:.1f}%".rjust(W),
            f"{r['break_even_wr_pct']:.1f}%".rjust(W),
            f"{r['wr_gap_pct']:+.1f}%".rjust(W),
            f"${r['gross_pnl']:+.2f}".rjust(W),
            f"${r['total_gross_fees']:.2f}".rjust(W),
            f"${r['total_rebate']:+.2f}".rjust(W),
            f"${r['true_net_income']:+.2f}".rjust(W),
            f"${r['end_equity_no_rebate']:.2f}".rjust(W),
            f"${r['end_equity_with_rebate']:.2f}".rjust(W),
            f"${r['total_volume_usd']:,.0f}".rjust(W),
        ])
        print(row)
    print(sep)

    # ─── key insight ─────────────────────────────────────────────────
    print()
    print("  KEY:")
    print("  WR%         = actual win rate from backtest")
    print("  BE-WR%      = SL/(TP+SL) — break-even WR on P&L alone")
    print("  WR Gap      = WR% - BE-WR%  (positive = P&L profitable without rebate)")
    print("  Gross P&L   = pure price-movement gains/losses")
    print("  Fees Paid   = maker+taker fees (before rebate)")
    print("  Rebate 70%  = 70% of fees returned by rebate program")
    print("  True Net    = Gross P&L - Fees + Rebate  (actual income)")
    print("  End Balance = session.equity (wallet without rebate; starts $30)")
    print("  End+Rebate  = wallet + rebate income (what you actually have)")
    print()

    best = max(results, key=lambda r: r["end_equity_with_rebate"])
    print(f"  Best TP: {best['tp_bps']:.0f}bps"
          f"  →  end balance ${best['end_equity_with_rebate']:.2f}"
          f"  (${best['end_equity_no_rebate']:.2f} in wallet + ${best['total_rebate']:.2f} rebate)"
          f"  over {best['trades']:,} trades\n")


if __name__ == "__main__":
    main()
