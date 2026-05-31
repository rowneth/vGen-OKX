"""Session filter comparison: Baseline vs Filter-2h [1,6] vs Filter-4h [1,6,12,22].

Data:    BTC/USDT 5m, 360 days (2025-05-01 → 2026-04-27)
Capital: $30 (matches production)
Fixed:   TP=8 bps, SL=50 bps, micro_momentum, alternate=true, TB=on (20 bps)

Volume bonus ($80/M MEXC program) is included in all economic metrics.

Decision rules are pre-set and applied without modification.
"""

from __future__ import annotations

import copy
import pathlib
import sys

import pandas as pd
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backtest_volume_farmer import run_one  # noqa: E402

DATA_FILE   = PROJECT_ROOT / "data/historical/BTC_USDT_5m.parquet"
BONUS_PER_M = 80.0   # MEXC $80 per $1M volume (separate from fee rebate)

CONFIGS = [
    ("Baseline",          "config/config_volume_farmer_optimal.yaml"),
    ("Filter-2h [1,6]",   "config/config_volume_farmer_filter2h.yaml"),
    ("Filter-4h [1,6,12,22]", "config/config_volume_farmer_filter4h.yaml"),
]

# ─── Decision rules (pre-set; do not modify after seeing results) ────────────

def check_filter2h(f2h: dict, baseline: dict) -> tuple[bool, list]:
    """Filter-2h passes if ALL hold."""
    checks = [
        ("End+Rebate ≥ baseline × 1.02",
         f2h["end_plus_rebate"] >= baseline["end_plus_rebate"] * 1.02),
        ("WR within 2pp of baseline",
         abs(f2h["wr"] - baseline["wr"]) <= 2.0),
        ("Trades ≥ 85% of baseline",
         f2h["trades"] >= baseline["trades"] * 0.85),
        ("Net bps (adj) > baseline net bps (adj)",
         f2h["adj_net_bps"] > baseline["adj_net_bps"]),
    ]
    return all(c[1] for c in checks), checks


def check_filter4h(f4h: dict, baseline: dict, f2h: dict) -> tuple[bool, list]:
    """Filter-4h passes if ALL hold."""
    checks = [
        ("End+Rebate ≥ baseline × 1.03",
         f4h["end_plus_rebate"] >= baseline["end_plus_rebate"] * 1.03),
        ("WR within 2pp of baseline",
         abs(f4h["wr"] - baseline["wr"]) <= 2.0),
        ("Trades ≥ 80% of baseline",
         f4h["trades"] >= baseline["trades"] * 0.80),
        ("Net bps (adj) > baseline net bps (adj)",
         f4h["adj_net_bps"] > baseline["adj_net_bps"]),
        ("Net bps (adj) > Filter-2h net bps (adj)",
         f4h["adj_net_bps"] > f2h["adj_net_bps"]),
    ]
    return all(c[1] for c in checks), checks


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\nLoading {DATA_FILE} ...")
    df = pd.read_parquet(DATA_FILE)
    df.columns = [c.lower() for c in df.columns]
    first = pd.Timestamp(df["open_time"].iloc[0])
    last  = pd.Timestamp(df["open_time"].iloc[-1])
    days  = max((last - first).total_seconds() / 86400, 1)
    print(f"Loaded {len(df):,} bars  ({first.date()} → {last.date()})  [{days:.1f} days]\n")

    results = []

    for name, cfg_rel in CONFIGS:
        cfg_path = PROJECT_ROOT / cfg_rel
        with cfg_path.open() as fh:
            cfg = yaml.safe_load(fh)
        # Force full-dataset run
        cfg.setdefault("risk", {})
        cfg["risk"]["stop_on_volume_target"]         = False
        cfg["risk"]["consecutive_losses_limit"]      = 9999
        cfg["risk"]["consecutive_losses_cooldown_bars"] = 0
        cfg.setdefault("target", {})["volume_usd"]  = 999_999_999

        print(f"  Running {name:<28} ...", end=" ", flush=True)
        r = run_one(df, tp_bps=0, sl_bps=0, override_cfg=cfg)

        capital   = float(cfg["farmer"]["capital_usd"])
        trades    = r["trades"]
        wr        = r["win_rate_pct"]
        end_reb   = r["end_equity_with_rebate"]
        true_net  = r["true_net_income"]
        volume    = r["total_volume_usd"]   # round-trip
        max_dd    = r["max_drawdown_pct"]

        vol_30d   = volume / days * 30.0
        bonus_mo  = vol_30d / 1e6 * BONUS_PER_M
        net_mo    = true_net / days * 30.0
        adj_mo    = net_mo + bonus_mo
        # Adjusted net bps per one-side volume (entry+exit / 2)
        adj_net_total = true_net + (volume / 2 / 1e6 * BONUS_PER_M * days / 30.0 * (days / 30.0))
        # Simpler: net_bps over the actual period, bonus credited at per-volume rate
        adj_net_total_actual = true_net + (volume / 2 / 1e6 * BONUS_PER_M)
        adj_net_bps = (adj_net_total_actual / (volume / 2) * 10_000) if volume > 0 else 0.0
        block_rate = r.get("entries_skipped_by_mtf", 0)  # reuse field; session_blocks via summary
        # session_blocks is exposed through summary but not in run_one dict directly;
        # approximate: (baseline_trades - this_trades) / baseline if we have baseline
        # We'll compute it post-hoc in the report.

        results.append({
            "name":          name,
            "cfg_path":      str(cfg_rel),
            "trades":        trades,
            "wr":            wr,
            "end_plus_rebate": end_reb,
            "true_net":      true_net,
            "vol_30d":       vol_30d,
            "bonus_mo":      bonus_mo,
            "net_mo":        net_mo,
            "adj_mo":        adj_mo,
            "adj_net_bps":   adj_net_bps,
            "max_dd":        max_dd,
            "capital":       capital,
            "days":          days,
            "raw":           r,
        })

        cap_1k = capital * 1000 / adj_mo if adj_mo > 0 else float("inf")
        cap_s  = f"${cap_1k:,.0f}" if adj_mo > 0 else "n/a"
        print(f"trades={trades:,}  WR={wr:.1f}%  end+reb=${end_reb:.2f}  "
              f"adj_mo=${adj_mo:.2f}  cap4$1k={cap_s}")

    baseline = results[0]
    f2h      = results[1]
    f4h      = results[2]

    # Block-rate approximation
    for r in results[1:]:
        r["block_pct"] = (1 - r["trades"] / max(baseline["trades"], 1)) * 100

    # ─── Decision rules ──────────────────────────────────────────────────────
    pass_2h, checks_2h = check_filter2h(f2h, baseline)
    pass_4h, checks_4h = check_filter4h(f4h, baseline, f2h)

    # Selection
    if pass_2h and pass_4h:
        winner = "Filter-4h" if f4h["adj_net_bps"] > f2h["adj_net_bps"] else "Filter-2h"
        reason = "Both pass; picked higher adj net bps"
    elif pass_2h:
        winner = "Filter-2h"
        reason = "Only Filter-2h passed"
    elif pass_4h:
        winner = "Filter-4h"
        reason = "Only Filter-4h passed"
    else:
        winner = None
        reason = "Neither passed — keep current production unchanged"

    # ─── Print report ────────────────────────────────────────────────────────
    W = 110
    print("\n\n" + "=" * W)
    print("  SESSION FILTER COMPARISON: Baseline vs Filter-2h vs Filter-4h")
    print(f"  Data: BTC/USDT 5m  |  {days:.0f} days  |  Capital $30  |  "
          f"TP=8bps SL=50bps TB=ON(20bps) micro_momentum")
    print(f"  Volume bonus: ${BONUS_PER_M}/M MEXC program included in AdjBps and AdjNet/mo")
    print("=" * W)

    hdr = f"{'Config':<28} {'Trades':>7} {'WR%':>6} {'End+Reb':>8} {'Trd/d':>5} " \
          f"{'Vol/30d':>10} {'Bonus/mo':>9} {'AdjNet/mo':>10} {'AdjBps':>7} {'MaxDD%':>7} {'BlkRate':>8}"
    print(hdr)
    print("─" * W)

    for r in results:
        blk = f"{r.get('block_pct',0):+.1f}%"
        cap = r["capital"] * 1000 / r["adj_mo"] if r["adj_mo"] > 0 else float("inf")
        tpd = r["trades"] / r["days"]
        print(
            f"  {r['name']:<26} {r['trades']:>7,} {r['wr']:>6.1f} "
            f"${r['end_plus_rebate']:>7.2f} {tpd:>5.1f} "
            f"${r['vol_30d']:>9,.0f} ${r['bonus_mo']:>8.2f} "
            f"${r['adj_mo']:>9.2f} {r['adj_net_bps']:>7.3f} "
            f"{r['max_dd']:>7.1f} {blk:>8}"
        )
    print("─" * W)

    # ─── Decision rule verdicts ───────────────────────────────────────────────
    def pf(passed: bool) -> str:
        return "✅" if passed else "❌"

    print(f"\n{'─'*W}")
    print("  DECISION RULE EVALUATION")
    print(f"{'─'*W}")

    print("\n  Filter-2h [1,6]:")
    for name, passed in checks_2h:
        print(f"    {pf(passed)} {name}")
    print(f"  Verdict: {'PASS ✅' if pass_2h else 'FAIL ❌'}")

    print("\n  Filter-4h [1,6,12,22]:")
    for name, passed in checks_4h:
        print(f"    {pf(passed)} {name}")
    print(f"  Verdict: {'PASS ✅' if pass_4h else 'FAIL ❌'}")

    print(f"\n{'─'*W}")
    print(f"  WINNER: {winner if winner else 'None — keep production unchanged'}")
    print(f"  Reason: {reason}")

    if winner:
        wr = next(r for r in results if winner in r["name"])
        impr = (wr["end_plus_rebate"] / baseline["end_plus_rebate"] - 1) * 100
        adj_impr = (wr["adj_mo"] - baseline["adj_mo"])
        print(f"\n  Expected improvement:")
        print(f"    End+Rebate vs baseline:  {impr:+.2f}%")
        print(f"    AdjNet/mo vs baseline:   ${adj_impr:+.2f}/mo")
        print(f"    Capital for $1k/mo:      ${wr['capital'] * 1000 / wr['adj_mo']:,.0f}")
    else:
        print(f"\n  Action: No config changes. Current production is the local optimum.")
        print(f"  Future directions: different symbols, different bracket geometry,")
        print(f"  spot/perp arb, or alternative exchange rebate programs.")

    print(f"\n{'─'*W}")
    print("  BLOCK RATE ANALYSIS")
    print(f"{'─'*W}")
    print(f"  Baseline: 0 bars blocked")
    for r in results[1:]:
        projected = r.get("block_pct", 0)
        skip = r["raw"].get("by_exit_reason", {})   # proxy for session blocks
        print(f"  {r['name']}: {projected:.1f}% trade reduction "
              f"({int(r.get('block_pct', 0) / 100 * baseline['trades'])} fewer trades)")
    hours_2h = len([1, 6])
    hours_4h = len([1, 6, 12, 22])
    projected_2h = hours_2h / 24 * 100
    projected_4h = hours_4h / 24 * 100
    print(f"\n  Naïve projection (uniform distribution):")
    print(f"    Filter-2h expected block rate: {projected_2h:.1f}%  |  actual: {results[1].get('block_pct',0):.1f}%")
    print(f"    Filter-4h expected block rate: {projected_4h:.1f}%  |  actual: {results[2].get('block_pct',0):.1f}%")
    if results[2].get("block_pct", 0) > 25:
        print(f"\n  ⚠️  WARNING: Filter-4h block rate > 25% (actual: "
              f"{results[2].get('block_pct',0):.1f}%) — "
              f"bar density may have shifted regime weights.")

    print(f"\n{'═'*W}\n")

    # ─── Save report to results/ ──────────────────────────────────────────────
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    winner_name = winner if winner else "None (keep production)"
    report_lines = [
        "# Session Filter Comparison Results\n",
        f"Date: 2026-04-29",
        f"Data: BTC/USDT 5m, {days:.0f} days ({first.date()} → {last.date()})",
        "Capital: $30 (matches production)\n",
        "## Config Comparison\n",
        f"| {'Config':<28} | {'Trades':>7} | {'WR%':>5} | {'AdjBps':>7} | {'End+Reb':>8} | {'AdjNet/mo':>10} | {'Vol/30d':>10} | Block |",
        f"|{'-'*30}|{'-'*9}|{'-'*7}|{'-'*9}|{'-'*10}|{'-'*12}|{'-'*12}|{'-'*7}|",
    ]
    for r in results:
        blk = f"{r.get('block_pct',0):.1f}%"
        report_lines.append(
            f"| {r['name']:<28} | {r['trades']:>7,} | {r['wr']:>5.1f} | "
            f"{r['adj_net_bps']:>7.3f} | ${r['end_plus_rebate']:>7.2f} | "
            f"${r['adj_mo']:>9.2f} | ${r['vol_30d']:>9,.0f} | {blk:>5} |"
        )

    report_lines += [
        "\n## Decision Rule Evaluation\n",
        "### Filter-2h [1, 6]",
    ]
    for name, passed in checks_2h:
        report_lines.append(f"- [{'✓' if passed else '✗'}] {name}")
    report_lines.append(f"\nVerdict: **{'PASS' if pass_2h else 'FAIL'}**\n")

    report_lines.append("### Filter-4h [1, 6, 12, 22]")
    for name, passed in checks_4h:
        report_lines.append(f"- [{'✓' if passed else '✗'}] {name}")
    report_lines.append(f"\nVerdict: **{'PASS' if pass_4h else 'FAIL'}**\n")

    report_lines += [
        "## Recommendation\n",
        f"**{winner_name}**\n",
        f"Reason: {reason}",
    ]
    if winner:
        wr_r = next(r for r in results if winner in r["name"])
        impr = (wr_r["end_plus_rebate"] / baseline["end_plus_rebate"] - 1) * 100
        report_lines.append(f"\nExpected improvement over baseline: {impr:+.2f}%")
        report_lines.append(f"\n## Next Steps\n")
        report_lines.append(f"1. Chosen config: `config/config_volume_farmer_{'filter2h' if '2h' in winner else 'filter4h'}.yaml`")
        report_lines.append("2. Deploy to live trading at $30 capital")
        report_lines.append("3. Run 200 trades minimum before evaluating")
        report_lines.append("4. If live WR diverges by >25% from backtest WR, pause and investigate")
    else:
        report_lines.append("\n## Next Steps\n")
        report_lines.append("This is the third filter rejection (h1_veto, Option A 8h, now 2h/4h).")
        report_lines.append("Current production config is the local optimum for this strategy class.")
        report_lines.append("Future directions: different symbols, different bracket geometries,")
        report_lines.append("spot/perp arb, or alternative exchange rebate programs.")

    out_file = out_dir / "session_filter_comparison.md"
    out_file.write_text("\n".join(report_lines))
    print(f"Report saved to {out_file.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
