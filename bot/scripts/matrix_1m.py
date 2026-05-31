"""1m backtest matrix: Option A (session filter + SL30 + MTF) combinations.

Runs 8 configurations on BTC_USDT_1m.parquet and prints a comparison table
with volume/30d and capital-needed estimates.

Usage:
    python scripts/matrix_1m.py
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

from backtest_volume_farmer import run_one  # noqa: E402

DATA_FILE = PROJECT_ROOT / "data/historical/BTC_USDT_1m.parquet"

# ─── Skip hours that historically produce low-quality signals ───────────────
# Based on UTC low-liquidity windows + known high-volatility open hours
SKIP_HOURS = [0, 1, 6, 12, 17, 19, 21, 22]

# ─── Base config for 1m micro_momentum (matches live config_volume_farmer_1m.yaml)
BASE_1M = {
    "app": {"timezone": "Asia/Colombo"},
    "exchange": {"symbol": "BTC_USDT", "timeframe": "1m"},
    "fees": {
        "maker": 0.0001,
        "taker": 0.0005,
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
        "tp_bps": 8.0,
        "sl_bps": 50.0,
        "max_hold_bars": 999,
        "entry": {
            "mode": "micro_momentum",
            "min_bar_range_bps": 3.0,
            "max_bar_range_bps": 40.0,
        },
        "alternate_direction": True,
        "trend_break": {
            "enabled": True,
            "min_bars_held": 3,
            "adverse_bps": 20.0,
        },
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


def _make_cfg(
    sl_bps: float = 50.0,
    tb_adverse_bps: float = 20.0,
    alternate: bool = True,
    mtf: bool = False,
    session_filter: bool = False,
) -> dict:
    cfg = copy.deepcopy(BASE_1M)
    cfg["farmer"]["sl_bps"] = sl_bps
    cfg["farmer"]["trend_break"]["adverse_bps"] = tb_adverse_bps
    cfg["farmer"]["alternate_direction"] = alternate

    if mtf:
        cfg["farmer"]["entry"]["mode"] = "multi_timeframe"
        cfg["farmer"]["entry"]["primary_3m_lookback_bars"] = 5
        cfg["farmer"]["entry"]["macro_15m_lookback_bars"] = 5
        cfg["farmer"]["entry"]["allow_neutral_micro"] = True
        cfg["farmer"]["entry"]["skip_neutral_macro"] = True

    if session_filter:
        cfg["farmer"]["entry"]["skip_hours"] = SKIP_HOURS

    return cfg


# ─── Matrix definitions ─────────────────────────────────────────────────────
MATRIX = [
    ("Baseline",         _make_cfg(sl_bps=50, tb_adverse_bps=20, alternate=True,  mtf=False, session_filter=False)),
    ("SL30",             _make_cfg(sl_bps=30, tb_adverse_bps=18, alternate=True,  mtf=False, session_filter=False)),
    ("MTF",              _make_cfg(sl_bps=50, tb_adverse_bps=20, alternate=False, mtf=True,  session_filter=False)),
    ("Session",          _make_cfg(sl_bps=50, tb_adverse_bps=20, alternate=True,  mtf=False, session_filter=True)),
    ("SL30+MTF",         _make_cfg(sl_bps=30, tb_adverse_bps=18, alternate=False, mtf=True,  session_filter=False)),
    ("SL30+Session",     _make_cfg(sl_bps=30, tb_adverse_bps=18, alternate=True,  mtf=False, session_filter=True)),
    ("MTF+Session",      _make_cfg(sl_bps=50, tb_adverse_bps=20, alternate=False, mtf=True,  session_filter=True)),
    ("OptionA (full)",   _make_cfg(sl_bps=30, tb_adverse_bps=18, alternate=False, mtf=True,  session_filter=True)),
]

# ─── Decision-rule thresholds (pre-set, not to change after seeing results) ──
RULE_NET_BPS    = 0.30   # net bps per $1 volume must be ≥ this
RULE_WR         = 78.0   # win rate %
RULE_TRADES_DAY = 30.0   # trades/day
RULE_MAX_DD_PP  = 5.0    # max drawdown ≤ baseline + this many pp


def _pass_fail(value: float, threshold: float, higher_better: bool = True) -> str:
    ok = value >= threshold if higher_better else value <= threshold
    return "✅" if ok else "❌"


def main() -> None:
    print(f"\nLoading {DATA_FILE} ...")
    df = pd.read_parquet(DATA_FILE)
    df.columns = [c.lower() for c in df.columns]

    # open_time may be datetime64[ns, UTC] — ensure it's numeric ms for the session
    if pd.api.types.is_datetime64_any_dtype(df["open_time"]):
        # Keep as-is; VolumeFarmerSession calls pd.Timestamp(last["open_time"])
        # which handles tz-aware timestamps correctly.
        pass

    rows = len(df)
    first_ts = pd.Timestamp(df["open_time"].iloc[0])
    last_ts  = pd.Timestamp(df["open_time"].iloc[-1])
    days_covered = max((last_ts - first_ts).total_seconds() / 86400, 1)

    print(f"Loaded {rows:,} bars  ({first_ts.date()} → {last_ts.date()})  "
          f"[{days_covered:.1f} days]\n")

    results = []
    baseline_dd = None

    for name, cfg in MATRIX:
        print(f"  Running {name:<18} ...", end=" ", flush=True)
        r = run_one(df, tp_bps=0, sl_bps=0, override_cfg=cfg)
        trades     = r["trades"]
        wr         = r["win_rate_pct"]
        end_eq     = r["end_equity_with_rebate"]
        volume     = r["total_volume_usd"]
        max_dd     = r["max_drawdown_pct"]
        capital    = float(cfg["farmer"]["capital_usd"])
        net_income = r["true_net_income"]
        tb_exits   = r["trend_breaks"]
        session_bl = r.get("session_blocks_pct", 0.0)  # computed below

        # Session block rate: how many flat-bars were skipped by hour filter
        # We can derive it from entries_attempted vs entries_considered
        entries_considered = r.get("entries_attempted", 0)
        session_blocks     = r.get("entries_skipped_by_mtf", 0)  # fallback

        trades_per_day = trades / days_covered
        volume_30d     = volume / days_covered * 30.0
        # Net bps per dollar of volume (one-sided notional)
        # volume in run_one is round-trip (entry+exit), so one-side = volume/2
        net_bps_per_vol = (net_income / (volume / 2) * 10_000) if volume > 0 else 0.0
        # Capital needed to earn $1000/month net:
        # net_income is over days_covered days → monthly_net = net_income/days * 30
        monthly_net = net_income / days_covered * 30.0 if days_covered > 0 else 0.0
        if monthly_net > 0:
            capital_for_1k = capital * 1000.0 / monthly_net
        else:
            capital_for_1k = float("inf")

        if baseline_dd is None:
            baseline_dd = max_dd  # store for comparison

        results.append({
            "name":            name,
            "trades":          trades,
            "trades_day":      trades_per_day,
            "wr":              wr,
            "end_eq":          end_eq,
            "net_income":      net_income,
            "monthly_net":     monthly_net,
            "volume_30d":      volume_30d,
            "net_bps":         net_bps_per_vol,
            "max_dd":          max_dd,
            "capital_for_1k":  capital_for_1k,
            "tb_exits":        tb_exits,
            "cfg":             cfg,
        })
        symbol = "✅" if (wr >= RULE_WR and trades_per_day >= RULE_TRADES_DAY and net_bps_per_vol >= RULE_NET_BPS) else "❌"
        print(f"{symbol}  {trades:,} trades, WR={wr:.1f}%, end+reb=${end_eq:.2f}, DD={max_dd:.1f}%")

    # ─── Add volume bonus ($80/M, separate MEXC program, not in backtest P&L) ─
    BONUS_PER_MILLION = 80.0
    for r in results:
        bonus_mo = r["volume_30d"] / 1_000_000.0 * BONUS_PER_MILLION
        r["bonus_mo"]     = bonus_mo
        r["adj_monthly"]  = r["monthly_net"] + bonus_mo
        # Adjusted net bps per one-side volume, including bonus
        vol_25d = r["volume_30d"] / 30.0 * days_covered
        adj_net_25d = r["net_income"] + (vol_25d / 1_000_000.0 * BONUS_PER_MILLION)
        r["adj_net_bps"]  = (adj_net_25d / (vol_25d / 2) * 10_000) if vol_25d > 0 else 0.0
        r["cap4_1k_adj"]  = (float(r["cfg"]["farmer"]["capital_usd"]) * 1000.0 / r["adj_monthly"]
                              if r["adj_monthly"] > 0 else float("inf"))

    # ─── Print table 1: trading-only (no bonus) ──────────────────────────────
    print("\n─── TABLE 1: Trading P&L only (fees + 70% rebate, NO volume bonus) ───")
    print("=" * 110)
    print(f"{'Config':<18} {'Trd/d':>5} {'WR%':>6} {'End+Reb':>8} {'Net/mo':>7} {'Vol/30d':>10} {'NetBps':>7} {'MaxDD%':>7} {'Cap4$1k':>8}  Rules")
    print("-" * 110)

    for r in results:
        rule_net  = _pass_fail(r["net_bps"],    RULE_NET_BPS)
        rule_wr   = _pass_fail(r["wr"],         RULE_WR)
        rule_td   = _pass_fail(r["trades_day"], RULE_TRADES_DAY)
        rule_dd   = _pass_fail(r["max_dd"],     baseline_dd + RULE_MAX_DD_PP, higher_better=False)
        cap_str   = f"${r['capital_for_1k']:,.0f}" if r["capital_for_1k"] < 1e7 else "n/a"
        rules_str = f"{rule_wr} WR  {rule_td} T/d  {rule_net} bps  {rule_dd} DD"
        print(
            f"{r['name']:<18} "
            f"{r['trades_day']:>5.1f} "
            f"{r['wr']:>6.1f} "
            f"${r['end_eq']:>7.2f} "
            f"${r['monthly_net']:>6.2f} "
            f"${r['volume_30d']:>9,.0f} "
            f"{r['net_bps']:>7.3f} "
            f"{r['max_dd']:>7.1f} "
            f"{cap_str:>8}  "
            f"{rules_str}"
        )
    print("=" * 110)

    # ─── Print table 2: including $80/M volume bonus ─────────────────────────
    print(f"\n─── TABLE 2: With $80/M MEXC volume bonus (total economic view) ───")
    print("=" * 110)
    print(f"{'Config':<18} {'Trd/d':>5} {'Vol/30d':>10} {'Bonus/mo':>9} {'AdjNet/mo':>10} {'AdjBps':>7} {'Cap4$1k':>8}  Pass?")
    print("-" * 110)

    adj_pass_count = 0
    for r in results:
        cap_str = f"${r['cap4_1k_adj']:,.0f}" if r["cap4_1k_adj"] < 1e7 else "n/a"
        pf = "✅" if (r["adj_net_bps"] >= RULE_NET_BPS and r["wr"] >= RULE_WR and r["trades_day"] >= RULE_TRADES_DAY) else "❌"
        if r["adj_net_bps"] >= RULE_NET_BPS:
            adj_pass_count += 1
        print(
            f"{r['name']:<18} "
            f"{r['trades_day']:>5.1f} "
            f"${r['volume_30d']:>9,.0f} "
            f"${r['bonus_mo']:>8.2f} "
            f"${r['adj_monthly']:>9.2f} "
            f"{r['adj_net_bps']:>7.3f} "
            f"{cap_str:>8}  "
            f"{pf}"
        )
    print("=" * 110)

    print(f"\nDecision rules: WR≥{RULE_WR}% | Trades/day≥{RULE_TRADES_DAY} | AdjBps≥{RULE_NET_BPS} | MaxDD≤baseline+{RULE_MAX_DD_PP}pp")
    print(f"Data: {days_covered:.1f} days  |  Skip hours (UTC): {SKIP_HOURS}")
    print(f"Note: 'AdjBps' = (trading net + $80/M bonus) per one-side volume × 10,000")
    print(f"Note: Break-even WR at TP=8/SL=50 with fees = {50*1.3/(8*1.0+50*1.3)*100:.1f}%  |  at TP=8/SL=30 = {30*1.3/(8*1.0+30*1.3)*100:.1f}%")

    # ─── Best candidate summary ──────────────────────────────────────────────
    # Check with adjusted bps (including volume bonus)
    passing = [r for r in results if (
        r["adj_net_bps"] >= RULE_NET_BPS
        and r["wr"] >= RULE_WR
        and r["trades_day"] >= RULE_TRADES_DAY
        and r["max_dd"] <= (baseline_dd or 999) + RULE_MAX_DD_PP
    )]
    if passing:
        best = max(passing, key=lambda x: x["adj_net_bps"])
        print(f"\n✅  Best passing config (with bonus): {best['name']}")
        print(f"   Trades/day: {best['trades_day']:.1f}  |  WR: {best['wr']:.1f}%  |  Adj bps: {best['adj_net_bps']:.3f}")
        print(f"   Monthly: ${best['monthly_net']:.2f} trading  +  ${best['bonus_mo']:.2f} bonus  =  ${best['adj_monthly']:.2f} total")
        print(f"   Volume/30d: ${best['volume_30d']:,.0f}  |  Capital for $1,000/mo: ${best['cap4_1k_adj']:,.0f}")
    else:
        best_partial = max(results, key=lambda x: x["adj_net_bps"])
        print(f"\n⚠️  No config passed all decision rules (even with volume bonus).")
        print(f"   Best adj bps: {best_partial['name']}  adj_bps={best_partial['adj_net_bps']:.3f}")
        print(f"   Monthly: ${best_partial['monthly_net']:.2f} trading  +  ${best_partial['bonus_mo']:.2f} bonus  =  ${best_partial['adj_monthly']:.2f}")
        print(f"   Volume/30d: ${best_partial['volume_30d']:,.0f}  |  Capital for $1,000/mo: "
              + (f"${best_partial['cap4_1k_adj']:,.0f}" if best_partial['cap4_1k_adj'] < 1e7 else "n/a (net negative)"))


if __name__ == "__main__":
    main()
