"""Vectorized backtest of VolumeFarmerSession over BTC_USDT 5m (~360d).

Re-implements the volume-farmer entry/exit logic in numpy-friendly form:
pre-computes per-bar arrays once, then runs a single O(n) scalar pass for
position state (which is unavoidably sequential because of alternation
and TP/SL chaining).

Matches the semantics of execution.volume_farmer.VolumeFarmerSession for:
  - micro_momentum entry (close > open => long, else short)
  - alternate_direction post-filter
  - bar-range gate (min/max bar range bps)
  - intrabar TP/SL with SL-first tie-break
  - trend_break early exit (close vs entry adverse_bps after min_bars_held)
  - consecutive-loss cooldown
  - dynamic leverage sizing
  - real fee model: open=maker, close=taker, 70% rebate

Does NOT yet vectorize: rsi_wt / bollinger_fade / multi_timeframe entries,
1h-veto post-filter, skip_hours session filter. Those configs will raise.

Usage:
    python scripts/backtest_vectorized.py
    python scripts/backtest_vectorized.py --config config/config_volume_farmer_optimal.yaml
    python scripts/backtest_vectorized.py --data data/historical/BTC_USDT_5m.parquet
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = "config/config_volume_farmer_optimal.yaml"
DEFAULT_DATA = "data/historical/BTC_USDT_5m.parquet"

DEFAULT_MAKER = 0.0001   # 0.01% — historical
DEFAULT_TAKER = 0.0005   # 0.05% — historical
DEFAULT_REBATE = 0.70     # 70% — historical


def _check_supported(cfg: dict) -> None:
    f = cfg["farmer"]
    entry_mode = str(f.get("entry", {}).get("mode", "micro_momentum"))
    if entry_mode != "micro_momentum":
        raise NotImplementedError(
            f"Vectorized backtest currently supports entry mode 'micro_momentum' only "
            f"(got {entry_mode!r}). Use scripts/backtest_volume_farmer.py for other modes."
        )
    if f.get("h1_veto", {}).get("enabled", False):
        raise NotImplementedError("1h-veto post-filter not yet vectorized.")
    if f.get("entry", {}).get("skip_hours"):
        raise NotImplementedError("skip_hours session filter not yet vectorized.")


def run_vectorized(df: pd.DataFrame, cfg: dict,
                   tp_slip_bps: float = 0.0,
                   sl_slip_bps: float = 0.0) -> dict[str, Any]:
    _check_supported(cfg)

    fees_cfg = cfg.get("fees", {})
    maker_rate = float(fees_cfg.get("maker", DEFAULT_MAKER))
    taker_rate = float(fees_cfg.get("taker", DEFAULT_TAKER))
    rebate_pct = float(fees_cfg.get("rebate_pct", DEFAULT_REBATE))

    f = cfg["farmer"]
    capital_usd = float(f["capital_usd"])
    margin_frac = float(f.get("margin_fraction_per_trade", 0.05))
    sizing = f.get("sizing", {})
    dynamic_lev = bool(sizing.get("dynamic_leverage", False))
    risk_pct = float(sizing.get("risk_per_trade_pct", 0.025))
    max_lev = float(sizing.get("max_leverage", 125))
    min_lev = float(sizing.get("min_leverage", 5))
    fixed_lev = float(f.get("leverage", 0))

    tp_bps = float(f["tp_bps"])
    sl_bps = float(f["sl_bps"])
    max_hold = int(f["max_hold_bars"])
    entry_cfg = f.get("entry", {})
    min_range_bps = float(entry_cfg.get("min_bar_range_bps", 0.0))
    max_range_bps = float(entry_cfg.get("max_bar_range_bps", 1e6))
    alternate = bool(f.get("alternate_direction", True))

    tb_cfg = f.get("trend_break", {})
    tb_enabled = bool(tb_cfg.get("enabled", False))
    tb_min_bars = int(tb_cfg.get("min_bars_held", 2))
    tb_adverse_bps = float(tb_cfg.get("adverse_bps", 20.0))

    risk_cfg = cfg.get("risk", {})
    daily_loss_limit = float(risk_cfg.get("daily_loss_limit_pct", 0.05))
    max_dd = float(risk_cfg.get("max_drawdown_pct", 0.25))
    consec_limit = int(risk_cfg.get("consecutive_losses_limit", 3))
    cooldown_bars = int(risk_cfg.get("consecutive_losses_cooldown_bars", 24))
    stop_on_target = bool(risk_cfg.get("stop_on_volume_target", True))
    volume_target = float(cfg.get("target", {}).get("volume_usd", 1e9))

    # --- precompute arrays --------------------------------------------------
    open_arr = df["open"].to_numpy(dtype=np.float64)
    high_arr = df["high"].to_numpy(dtype=np.float64)
    low_arr = df["low"].to_numpy(dtype=np.float64)
    close_arr = df["close"].to_numpy(dtype=np.float64)
    time_arr = pd.DatetimeIndex(df["open_time"])

    with np.errstate(divide="ignore", invalid="ignore"):
        bar_range_bps = np.abs(close_arr - open_arr) / open_arr * 10_000.0
    range_valid = (bar_range_bps >= min_range_bps) & (bar_range_bps <= max_range_bps)
    raw_bias = np.where(close_arr > open_arr, 1, -1)  # 1=long, -1=short
    # Per micro_momentum: equal close==open => short (close>open is the only long trigger)

    day_arr = time_arr.strftime("%Y-%m-%d").to_numpy()

    # --- state --------------------------------------------------------------
    equity = capital_usd
    peak_equity = equity
    start_equity = equity
    total_volume = 0.0
    total_fees_gross = 0.0
    total_pnl = 0.0
    wins = 0
    losses = 0
    round_trips = 0
    consec_losses = 0
    cooldown_left = 0
    last_side = 0  # 0=none, 1=long, -1=short
    daily_pnl = 0.0
    daily_date = ""
    halted = False
    halt_reason = ""

    in_pos = False
    pos_side = 0
    pos_entry_price = 0.0
    pos_notional = 0.0
    pos_tp = 0.0
    pos_sl = 0.0
    pos_bars_held = 0

    exit_reasons: dict[str, int] = {}
    win_bps_list: list[float] = []
    loss_bps_list: list[float] = []
    max_dd_pct = 0.0
    entries_considered = 0

    n = len(df)
    # First two bars are skipped to mirror the original loop's range(2, n).
    for i in range(2, n):
        # --- daily PnL reset ------------------------------------------------
        d = day_arr[i]
        if d != daily_date:
            daily_date = d
            daily_pnl = 0.0

        # --- halt checks ----------------------------------------------------
        if not halted:
            dd = (peak_equity - equity) / max(peak_equity, 1e-9)
            if dd >= max_dd:
                halted = True
                halt_reason = f"max_drawdown {dd*100:.2f}%"
            elif -daily_pnl / max(start_equity, 1e-9) >= daily_loss_limit:
                halted = True
                halt_reason = f"daily_loss {(-daily_pnl/max(start_equity,1e-9))*100:.2f}%"
            elif stop_on_target and total_volume >= volume_target:
                halted = True
                halt_reason = f"volume_target_reached {total_volume:,.0f}"
        if halted:
            break

        hi = high_arr[i]
        lo = low_arr[i]
        cl = close_arr[i]

        # --- in position: check TP/SL/trend_break/time_stop -----------------
        if in_pos:
            pos_bars_held += 1
            if pos_side == 1:
                tp_hit = hi >= pos_tp
                sl_hit = lo <= pos_sl
            else:
                tp_hit = lo <= pos_tp
                sl_hit = hi >= pos_sl

            exit_price = None
            reason = None
            if sl_hit and tp_hit:
                exit_price = pos_sl
                reason = "sl_ambiguous"
            elif tp_hit:
                exit_price = pos_tp
                reason = "tp"
            elif sl_hit:
                exit_price = pos_sl
                reason = "sl"
            elif tb_enabled and pos_bars_held >= tb_min_bars:
                if pos_side == 1:
                    adverse = (pos_entry_price - cl) / pos_entry_price * 1e4
                else:
                    adverse = (cl - pos_entry_price) / pos_entry_price * 1e4
                if adverse >= tb_adverse_bps:
                    exit_price = cl
                    reason = "trend_break"
            if exit_price is None and pos_bars_held >= max_hold:
                exit_price = cl
                reason = "time_stop"

            if exit_price is not None:
                if pos_side == 1:
                    gross_pnl = (exit_price - pos_entry_price) / pos_entry_price * pos_notional
                else:
                    gross_pnl = (pos_entry_price - exit_price) / pos_entry_price * pos_notional
                # Slippage on close: realized fill is N bps worse than trigger.
                # Applied as bps of notional, always against us.
                if reason == "tp":
                    gross_pnl -= tp_slip_bps * 1e-4 * pos_notional
                elif reason in ("sl", "sl_ambiguous", "trend_break", "time_stop"):
                    gross_pnl -= sl_slip_bps * 1e-4 * pos_notional
                close_fee = pos_notional * taker_rate
                net_pnl = gross_pnl - close_fee
                total_fees_gross += close_fee
                total_volume += pos_notional
                total_pnl += net_pnl
                equity += gross_pnl - close_fee
                peak_equity = max(peak_equity, equity)
                daily_pnl += net_pnl
                round_trips += 1
                last_side = pos_side
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                bps = gross_pnl / pos_notional * 1e4
                if net_pnl > 0:
                    wins += 1
                    consec_losses = 0
                    win_bps_list.append(bps)
                else:
                    losses += 1
                    consec_losses += 1
                    loss_bps_list.append(-bps)
                    if consec_losses >= consec_limit:
                        cooldown_left = cooldown_bars
                        consec_losses = 0
                # drawdown tracking
                if peak_equity > 0:
                    cur_dd = (peak_equity - equity) / peak_equity * 100
                    if cur_dd > max_dd_pct:
                        max_dd_pct = cur_dd
                in_pos = False
                pos_side = 0
                pos_bars_held = 0
            continue  # don't open a new trade on a bar we just closed on

        # --- flat: cooldown / entry decision --------------------------------
        if cooldown_left > 0:
            cooldown_left -= 1
            continue

        if not range_valid[i]:
            continue
        entries_considered += 1
        side = int(raw_bias[i])
        if alternate and last_side == side:
            side = -side

        # Open position at close of this bar (mirrors live session's
        # _open_position(close) on the just-closed candle).
        margin = equity * margin_frac
        if dynamic_lev and margin > 0:
            sl_frac = sl_bps / 10_000.0
            if sl_frac > 0:
                lev = (equity * risk_pct) / (margin * sl_frac)
                lev = max(min_lev, min(lev, max_lev))
            else:
                lev = max_lev
        else:
            lev = fixed_lev if fixed_lev > 0 else 20.0
        notional = margin * lev
        open_fee = notional * maker_rate
        total_fees_gross += open_fee
        total_volume += notional
        equity -= open_fee

        in_pos = True
        pos_side = side
        pos_entry_price = cl
        pos_notional = notional
        if side == 1:
            pos_tp = cl * (1 + tp_bps / 10_000.0)
            pos_sl = cl * (1 - sl_bps / 10_000.0)
        else:
            pos_tp = cl * (1 - tp_bps / 10_000.0)
            pos_sl = cl * (1 + sl_bps / 10_000.0)
        pos_bars_held = 0

    # --- finalize -----------------------------------------------------------
    trades = wins + losses
    wr = wins / trades if trades else 0.0
    be_wr = sl_bps / (tp_bps + sl_bps)
    rebate = total_fees_gross * rebate_pct
    gross_pnl_total = total_pnl + total_fees_gross  # back out fees to recover gross
    # Above: total_pnl already nets close_fees; opens were taken straight from equity
    # so gross_pnl_total = total_pnl + close_fees only. We need gross including open
    # fees too:
    #   true_gross_price_pnl = total_pnl + (close_fees_only)
    # but we tracked total_fees_gross = opens + closes. Recompute differently:
    #   end_equity_no_rebate = capital + price_pnl - fees_total
    #   true_net_income      = price_pnl - fees_total + rebate
    # We have equity already (capital + price_pnl - fees_total), so:
    end_no_rebate = equity
    end_with_rebate = end_no_rebate + rebate
    true_net = end_with_rebate - capital_usd
    gross_price_pnl = (end_no_rebate - capital_usd) + total_fees_gross
    # Total trading volume (open + close legs):
    total_volume_full = total_volume  # we already counted open + close legs

    return {
        "tp_bps": tp_bps,
        "sl_bps": sl_bps,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": wr * 100,
        "break_even_wr_pct": be_wr * 100,
        "wr_gap_pct": (wr - be_wr) * 100,
        "gross_pnl": gross_price_pnl,
        "total_gross_fees": total_fees_gross,
        "total_rebate": rebate,
        "net_fees_paid": total_fees_gross * (1 - rebate_pct),
        "true_net_income": true_net,
        "end_equity_no_rebate": end_no_rebate,
        "end_equity_with_rebate": end_with_rebate,
        "net_per_trade": (true_net / trades) if trades else 0.0,
        "total_volume_usd": total_volume_full,
        "trend_breaks": exit_reasons.get("trend_break", 0),
        "by_exit_reason": exit_reasons,
        "max_drawdown_pct": max_dd_pct,
        "avg_win_bps": float(np.mean(win_bps_list)) if win_bps_list else 0.0,
        "avg_loss_bps": float(np.mean(loss_bps_list)) if loss_bps_list else 0.0,
        "entries_attempted": entries_considered,
        "entries_skipped_by_mtf": 0,
        "halted": halted,
        "halt_reason": halt_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorized 360d backtest")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help=f"YAML config (default: {DEFAULT_CONFIG})")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA,
                        help=f"Parquet candles (default: {DEFAULT_DATA})")
    parser.add_argument("--rows", type=int, default=None,
                        help="Limit to last N bars")
    parser.add_argument("--no-stop-on-target", action="store_true",
                        help="Force-disable volume-target halt for full-history run")
    parser.add_argument("--lift-loss-halt", action="store_true",
                        help="Lift consecutive-loss halts for backtest realism")
    parser.add_argument("--maker", type=float, default=None,
                        help="Override maker fee rate (e.g. 0.0002 for 0.02 pct)")
    parser.add_argument("--taker", type=float, default=None,
                        help="Override taker fee rate (e.g. 0.0006 for 0.06 pct)")
    parser.add_argument("--rebate", type=float, default=None,
                        help="Override rebate pct (e.g. 0.40 for 40 pct)")
    parser.add_argument("--trend-break", action="store_true",
                        help="Force-enable trend_break exit (3 bars / 20 bps)")
    parser.add_argument("--max-dd", type=float, default=None,
                        help="Override max_drawdown_pct (e.g. 0.999 to disable)")
    parser.add_argument("--tp-slippage-bps", type=float, default=0.0,
                        help="Slippage in bps applied to TP fills (taker MEXC ~5-10, maker OKX ~0)")
    parser.add_argument("--sl-slippage-bps", type=float, default=0.0,
                        help="Slippage in bps applied to SL / trend_break / time_stop fills")
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with cfg_path.open() as fh:
        cfg = yaml.safe_load(fh)

    if args.no_stop_on_target:
        cfg.setdefault("risk", {})["stop_on_volume_target"] = False
        cfg.setdefault("target", {})["volume_usd"] = 999_999_999
    if args.lift_loss_halt:
        cfg["risk"]["consecutive_losses_limit"] = 9999
        cfg["risk"]["consecutive_losses_cooldown_bars"] = 0
    if args.maker is not None or args.taker is not None or args.rebate is not None:
        cfg.setdefault("fees", {})
        if args.maker is not None:
            cfg["fees"]["maker"] = args.maker
        if args.taker is not None:
            cfg["fees"]["taker"] = args.taker
        if args.rebate is not None:
            cfg["fees"]["rebate_pct"] = args.rebate
    if args.trend_break:
        cfg["farmer"].setdefault("trend_break", {})
        cfg["farmer"]["trend_break"]["enabled"] = True
        cfg["farmer"]["trend_break"].setdefault("min_bars_held", 3)
        cfg["farmer"]["trend_break"].setdefault("adverse_bps", 20.0)
    if args.max_dd is not None:
        cfg.setdefault("risk", {})["max_drawdown_pct"] = args.max_dd

    data_path = pathlib.Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    print(f"Loading {data_path} ...")
    df = pd.read_parquet(data_path)
    df.columns = [c.lower() for c in df.columns]
    if args.rows:
        df = df.iloc[-args.rows:].reset_index(drop=True)
    t0_d = pd.to_datetime(df["open_time"]).min()
    t1_d = pd.to_datetime(df["open_time"]).max()
    span_d = (t1_d - t0_d).total_seconds() / 86400
    print(f"Loaded {len(df):,} bars  ({t0_d} → {t1_d})  span={span_d:.1f}d\n")

    fees_show = cfg.get("fees", {})
    print(f"  Config: {cfg_path.name}")
    print(f"  Entry: {cfg['farmer']['entry'].get('mode')}  alternate={cfg['farmer'].get('alternate_direction')}")
    print(f"  TP={cfg['farmer']['tp_bps']}bps  SL={cfg['farmer']['sl_bps']}bps  "
          f"capital=${cfg['farmer']['capital_usd']}")
    print(f"  Fees: maker={float(fees_show.get('maker', DEFAULT_MAKER))*1e4:.1f}bps  "
          f"taker={float(fees_show.get('taker', DEFAULT_TAKER))*1e4:.1f}bps  "
          f"rebate={float(fees_show.get('rebate_pct', DEFAULT_REBATE))*100:.0f}%")
    tb_show = cfg['farmer'].get('trend_break', {})
    if tb_show.get('enabled'):
        print(f"  trend_break: enabled (min_bars={tb_show.get('min_bars_held',2)}, "
              f"adverse={tb_show.get('adverse_bps',20)}bps)")
    print()

    if args.tp_slippage_bps or args.sl_slippage_bps:
        print(f"  Slippage:  TP={args.tp_slippage_bps:.1f}bps  "
              f"SL/TB/time={args.sl_slippage_bps:.1f}bps\n")

    t0 = time.perf_counter()
    r = run_vectorized(df, cfg,
                       tp_slip_bps=args.tp_slippage_bps,
                       sl_slip_bps=args.sl_slippage_bps)
    elapsed = time.perf_counter() - t0

    cap = float(cfg["farmer"]["capital_usd"])
    roi_no_reb = (r["end_equity_no_rebate"] - cap) / cap * 100
    roi_with_reb = (r["end_equity_with_rebate"] - cap) / cap * 100

    bar = "=" * 80
    print(bar)
    print(f"  VECTORIZED BACKTEST — {cfg_path.name}")
    print(f"  TP={r['tp_bps']}bps  SL={r['sl_bps']}bps  CAP=${cap:.2f}  "
          f"span={span_d:.1f}d  ran in {elapsed*1000:.0f}ms")
    print(bar)
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
          f"(ROI {roi_no_reb:+.2f}%)")
    print(f"  Wallet end + rebate:   ${r['end_equity_with_rebate']:,.2f}   "
          f"(ROI {roi_with_reb:+.2f}%)")
    print(f"  Net per trade:         ${r['net_per_trade']:+,.4f}")
    print(f"  Avg win bps:           {r['avg_win_bps']:.2f}")
    print(f"  Avg loss bps:          {r['avg_loss_bps']:.2f}")
    print(f"  Exit reasons:          {r['by_exit_reason']}")
    print(f"  Max drawdown:          {r['max_drawdown_pct']:.1f}%")
    print(f"  Entries considered:    {r['entries_attempted']:,}")
    print(f"  Volume traded:         ${r['total_volume_usd']:,.0f}")
    if r["halted"]:
        print(f"  HALTED:                {r['halt_reason']}")
    print(bar)
    verdict = "PROFITABLE" if r["true_net_income"] > 0 else "UNPROFITABLE"
    wr_verdict = ("above break-even" if r["wr_gap_pct"] > 0
                  else "below break-even (relies on rebate)")
    print(f"  P&L:  {verdict}")
    print(f"  WR:   {wr_verdict}")
    print()


if __name__ == "__main__":
    main()
