"""Quick backtest of VolumeFarmerSession over historical BTC_USDT 5m data.

Runs multiple TP configurations and prints a comparison table.

Usage:
    python scripts/backtest_volume_farmer.py
    python scripts/backtest_volume_farmer.py --tp-list 5 8 10 15 20
    python scripts/backtest_volume_farmer.py --config config/config_volume_farmer_optimal.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys

import pandas as pd
import yaml

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


def run_one(df: pd.DataFrame, tp_bps: float, sl_bps: float, capital_usd: float = 30.0,
            tb_enabled: bool = False, tb_min_bars: int = 2, tb_adverse_bps: float = 20.0,
            override_cfg: dict | None = None) -> dict:
    """Run the session over full df, return summary stats.

    If ``override_cfg`` is given, it is used verbatim (after deepcopy) instead
    of the baked-in ``BASE_CONFIG``.  TP/SL/capital/trend-break args are then
    ignored — the YAML controls them.
    """
    if override_cfg is not None:
        cfg = copy.deepcopy(override_cfg)
        # Pull effective tp/sl/capital from the override so the summary renders
        tp_bps = float(cfg["farmer"]["tp_bps"])
        sl_bps = float(cfg["farmer"]["sl_bps"])
        capital_usd = float(cfg["farmer"]["capital_usd"])
    else:
        cfg = copy.deepcopy(BASE_CONFIG)
        cfg["farmer"]["tp_bps"] = tp_bps
        cfg["farmer"]["sl_bps"] = sl_bps
        cfg["farmer"]["capital_usd"] = capital_usd
        cfg["farmer"]["trend_break"] = {
            "enabled": tb_enabled,
            "min_bars_held": tb_min_bars,
            "adverse_bps": tb_adverse_bps,
        }

    session = VolumeFarmerSession(config=cfg)

    wins = 0
    losses = 0
    trend_breaks = 0
    total_pnl = 0.0          # gross price-move P&L
    total_real_fees = 0.0    # CORRECTED: open=maker, close=ALWAYS taker
    total_volume = 0.0
    exit_reasons: dict = {}  # {"tp": N, "sl": N, ...}
    win_bps_list: list = []
    loss_bps_list: list = []
    peak_eq = float(capital_usd)
    max_dd_pct = 0.0

    MAKER_RATE = 0.0001      # 0.01%
    TAKER_RATE = 0.0005      # 0.05%

    def on_event(evt: FarmerEvent) -> None:
        nonlocal wins, losses, total_pnl, total_real_fees, total_volume, trend_breaks
        nonlocal peak_eq, max_dd_pct
        if evt.kind == "exit":
            p = evt.payload
            notional = float(p.get("notional", 0))
            gross_pnl = float(p.get("gross_pnl", 0))
            reason = p.get("reason", "unknown")
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            if reason == "trend_break":
                trend_breaks += 1

            # Real fees: open=maker, close=taker (TP triggers MARKET order on MEXC)
            real_open_fee = notional * MAKER_RATE
            real_close_fee = notional * TAKER_RATE
            real_total_fee = real_open_fee + real_close_fee
            real_net = gross_pnl - real_total_fee

            bps = (gross_pnl / notional * 10_000) if notional > 0 else 0.0
            if real_net > 0:
                wins += 1
                win_bps_list.append(bps)
            else:
                losses += 1
                loss_bps_list.append(-bps)  # store as positive loss magnitude
            total_pnl += gross_pnl
            total_real_fees += real_total_fee
            total_volume += notional * 2  # entry + exit

            # Track running max drawdown
            eq = float(p.get("equity", peak_eq))
            nonlocal_peak = max(peak_eq, eq)
            if nonlocal_peak > 0:
                dd = (nonlocal_peak - eq) / nonlocal_peak * 100
                if dd > max_dd_pct:
                    max_dd_pct = dd
            peak_eq = max(peak_eq, eq)

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

    avg_win_bps = sum(win_bps_list) / len(win_bps_list) if win_bps_list else 0.0
    avg_loss_bps = sum(loss_bps_list) / len(loss_bps_list) if loss_bps_list else 0.0

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
        "trend_breaks": trend_breaks,
        "by_exit_reason": exit_reasons,
        "max_drawdown_pct": max_dd_pct,
        "avg_win_bps": avg_win_bps,
        "avg_loss_bps": avg_loss_bps,
        "entries_attempted": session._entries_considered,
        "entries_skipped_by_mtf": session._mtf_skips,
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
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config (e.g. config/config_volume_farmer_mtf.yaml). "
                             "When set, runs a SINGLE backtest using that config verbatim "
                             "and ignores --tp-list / --capitals.")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to a parquet data file to use instead of default BTC_USDT_5m. "
                             "Only used in single-config mode (--config).")
    parser.add_argument("--output", type=str, default=None,
                        help="Save backtest result as JSON to this path. "
                             "Only used in single-config mode (--config).")
    args = parser.parse_args()

    print(f"\nLoading {DATA_FILE} ...")
    data_file = pathlib.Path(args.data) if args.data else DATA_FILE
    if not data_file.is_absolute():
        data_file = PROJECT_ROOT / data_file
    df = pd.read_parquet(data_file)
    # Normalise column names
    df.columns = [c.lower() for c in df.columns]
    if args.rows:
        df = df.iloc[-args.rows:]
    print(f"Loaded {len(df):,} bars  ({df.index[0]} → {df.index[-1]})\n")

    # ─── single-config mode ──────────────────────────────────────────
    if args.config:
        cfg_path = pathlib.Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = PROJECT_ROOT / cfg_path
        with cfg_path.open("r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh)
        # Force full-history run (don't halt on volume target during backtest)
        user_cfg.setdefault("risk", {})["stop_on_volume_target"] = False
        user_cfg.setdefault("target", {})["volume_usd"] = 999_999_999
        # Lift consecutive-loss halt for backtest realism
        user_cfg["risk"]["consecutive_losses_limit"] = 9999
        user_cfg["risk"]["consecutive_losses_cooldown_bars"] = 0
        print(f"  Running single backtest with {cfg_path.name}\n")
        r = run_one(df, tp_bps=0, sl_bps=0, override_cfg=user_cfg)
        cap = float(user_cfg["farmer"]["capital_usd"])
        roi_no_rebate = ((r["end_equity_no_rebate"] - cap) / cap) * 100
        roi_with_rebate = ((r["end_equity_with_rebate"] - cap) / cap) * 100
        print("=" * 80)
        print(f"  CONFIG: {cfg_path.name}")
        print(f"  ENTRY MODE: {user_cfg['farmer']['entry'].get('mode')}")
        print(f"  TP={r['tp_bps']}bps  SL={r['sl_bps']}bps  CAP=${cap:.2f}")
        print("=" * 80)
        print(f"  Trades:                {r['trades']:,}")
        print(f"  Wins / Losses:         {r['wins']} / {r['losses']}")
        print(f"  Win rate:              {r['win_rate_pct']:.2f}%")
        print(f"  Break-even WR:         {r['break_even_wr_pct']:.2f}%   "
              f"(gap {r['wr_gap_pct']:+.2f}%)")
        print(f"  Trend-break exits:     {r['trend_breaks']}")
        print(f"  Gross P&L:             ${r['gross_pnl']:+,.2f}")
        print(f"  Total fees paid:       ${r['total_gross_fees']:,.2f}")
        print(f"  Rebate (70%):          ${r['total_rebate']:+,.2f}")
        print(f"  True net income:       ${r['true_net_income']:+,.2f}")
        print(f"  Wallet end (no reb):   ${r['end_equity_no_rebate']:,.2f}   "
              f"(ROI {roi_no_rebate:+.2f}%)")
        print(f"  Wallet end + rebate:   ${r['end_equity_with_rebate']:,.2f}   "
              f"(ROI {roi_with_rebate:+.2f}%)")
        print(f"  Net per trade:         ${r['net_per_trade']:+,.4f}")
        print(f"  Avg win bps:           {r['avg_win_bps']:.2f}")
        print(f"  Avg loss bps:          {r['avg_loss_bps']:.2f}")
        print(f"  Exit reasons:          {r['by_exit_reason']}")
        print(f"  Max drawdown:          {r['max_drawdown_pct']:.1f}%")
        print(f"  Entries attempted:     {r['entries_attempted']:,}")
        if r["entries_skipped_by_mtf"]:
            print(f"  MTF skips:             {r['entries_skipped_by_mtf']:,}")
        print(f"  Volume traded:         ${r['total_volume_usd']:,.0f}")
        if r["halted"]:
            print(f"  HALTED:                {r['halt_reason']}")
        print("=" * 80)
        # Verdict
        verdict = "✅ PROFITABLE" if r["true_net_income"] > 0 else "❌ UNPROFITABLE"
        wr_verdict = (
            "✅ above break-even" if r["wr_gap_pct"] > 0
            else "⚠️ below break-even (relies on rebate)"
        )
        print(f"  P&L:  {verdict}")
        print(f"  WR:   {wr_verdict}")
        print()

        # ─── JSON output ────────────────────────────────────────────
        if args.output:
            import time
            t_col = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            period_days = (t_col.max() - t_col.min()).total_seconds() / 86400
            result_doc = {
                "config_name": cfg_path.stem,
                "timeframe": user_cfg.get("exchange", {}).get("timeframe", "?"),
                "period_days": round(period_days, 1),
                "total_trades": r["trades"],
                "wr_pct": round(r["win_rate_pct"], 2),
                "break_even_wr_pct": round(r["break_even_wr_pct"], 2),
                "wr_gap_pct": round(r["wr_gap_pct"], 2),
                "avg_win_bps": round(r["avg_win_bps"], 2),
                "avg_loss_bps_weighted": round(r["avg_loss_bps"], 2),
                "by_exit_reason": r["by_exit_reason"],
                "gross_pnl": round(r["gross_pnl"], 4),
                "fees_paid": round(r["total_gross_fees"], 4),
                "rebate_earned": round(r["total_rebate"], 4),
                "end_equity": round(r["end_equity_no_rebate"], 4),
                "end_plus_rebate": round(r["end_equity_with_rebate"], 4),
                "net_per_trade": round(r["net_per_trade"], 6),
                "max_drawdown_pct": round(r["max_drawdown_pct"], 2),
                "volume_usd": round(r["total_volume_usd"], 0),
                "halted_early": r["halted"],
                "halt_reason": r["halt_reason"] or None,
                "entries_attempted": r["entries_attempted"],
                "entries_skipped_by_mtf": r["entries_skipped_by_mtf"],
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            out_path = pathlib.Path(args.output)
            if not out_path.is_absolute():
                out_path = PROJECT_ROOT / out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as fh:
                json.dump(result_doc, fh, indent=2)
            print(f"  Saved results → {out_path}")
        return

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
        marker = "✅" if r["true_net_income"] >= 0 else " ❌"
        print(f"  {marker}  {r['trades']} trades, WR={r['win_rate_pct']:.1f}%, net={r['true_net_income']:+.2f}")

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
