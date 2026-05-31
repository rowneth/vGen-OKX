"""Compare backtest JSON results and apply deployment decision rules.

Usage:
    python scripts/compare_backtests.py results/baseline_5m.json results/sl30_5m.json ...
    python scripts/compare_backtests.py results/*.json --output results/comparison.md
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Decision rules (set in advance — do not change after seeing results)
# SL30+MTF passes deployment if ALL hold:
PASS_RULES = {
    "wr_pct >= 80":          lambda r: r["wr_pct"] >= 80.0,
    "end+rebate >= base+10%": None,   # evaluated dynamically vs baseline
    "dd <= base+5pp":         None,   # evaluated dynamically
    "trades >= 50% baseline": None,   # evaluated dynamically
    "avg_loss_bps <= 25":     lambda r: r["avg_loss_bps_weighted"] <= 25.0,
}
# Fail if ANY:
FAIL_RULES = {
    "wr < 78%":               lambda r: r["wr_pct"] < 78.0,
    "end+rebate < baseline":  None,   # dynamic
    "dd > base+10pp":         None,   # dynamic
    "trades < 30% baseline":  None,   # dynamic
}

# Known config ordering for table display
CONFIG_ORDER = ["baseline", "sl30", "mtf", "sl30_mtf"]
CONFIG_LABELS = {
    "baseline": "Baseline (TP=8/SL=50, alternate=on)",
    "sl30":     "SL30 only (TP=8/SL=30, alternate=on)",
    "mtf":      "MTF only  (TP=8/SL=50, MTF=on)",
    "sl30_mtf": "SL30+MTF  (TP=8/SL=30, MTF=on)",
}


def short_name(stem: str) -> str:
    """Derive a config short name from a file stem like 'config_volume_farmer_sl30_5m'."""
    s = stem.replace("config_volume_farmer_", "").replace("_optimal", "baseline")
    # Strip trailing _5m or _1m if present (comes from --output path naming convention)
    for suffix in ("_5m", "_1m"):
        if s.endswith(suffix):
            s = s[:-len(suffix)]
            break
    return s


def load_results(paths: list[str]) -> dict[tuple, dict]:
    """Load JSON results keyed by (config_short_name, timeframe)."""
    out = {}
    for p in paths:
        path = pathlib.Path(p)
        if not path.exists():
            print(f"  WARNING: {p} not found, skipping")
            continue
        with path.open() as fh:
            doc = json.load(fh)
        cfg = doc.get("config_name", "")
        tf = doc.get("timeframe", "?")
        # Try to extract short name from config_name or from filename
        short = short_name(cfg) if cfg else short_name(path.stem)
        # Also try filename-based override: results/sl30_5m.json → ("sl30", "5m")
        stem = path.stem  # e.g. "sl30_5m"
        for suffix in ("_5m", "_1m"):
            if stem.endswith(suffix):
                tf_override = suffix[1:]  # "5m" or "1m"
                cfg_override = stem[:-len(suffix)]
                short = cfg_override
                tf = tf_override
                break
        out[(short, tf)] = doc
    return out


def fmt_pass(ok: bool) -> str:
    return "✓" if ok else "✗"


def apply_rules(r: dict, base: dict | None, timeframe: str) -> dict:
    """Return a dict of rule_name → bool for a single result."""
    evals = {}

    evals["wr >= 80%"]        = r["wr_pct"] >= 80.0
    evals["avg_loss <= 25bps"] = r["avg_loss_bps_weighted"] <= 25.0
    evals["wr >= 78% (fail)"] = r["wr_pct"] >= 78.0   # inverted: True = NOT failing

    if base:
        base_end = base["end_plus_rebate"]
        base_trades = base["total_trades"]
        base_dd = base["max_drawdown_pct"]

        evals["end+reb >= base+10%"] = r["end_plus_rebate"] >= base_end * 1.10
        evals["end+reb >= baseline"] = r["end_plus_rebate"] >= base_end
        evals["dd <= base+5pp"]      = r["max_drawdown_pct"] <= base_dd + 5.0
        evals["dd <= base+10pp (fail)"] = r["max_drawdown_pct"] <= base_dd + 10.0
        evals["trades >= 50% base"]  = r["total_trades"] >= base_trades * 0.50
        evals["trades >= 30% base (fail)"] = r["total_trades"] >= base_trades * 0.30
    else:
        for k in ["end+reb >= base+10%", "end+reb >= baseline", "dd <= base+5pp",
                  "dd <= base+10pp (fail)", "trades >= 50% base", "trades >= 30% base (fail)"]:
            evals[k] = None  # can't evaluate without baseline

    return evals


def build_report(results: dict) -> list[str]:
    lines = []
    lines.append("=" * 72)
    lines.append("  BACKTEST COMPARISON REPORT")
    lines.append("=" * 72)

    # Data range info
    for tf in ("5m", "1m"):
        for cfg in CONFIG_ORDER:
            if (cfg, tf) in results:
                r = results[(cfg, tf)]
                lines.append(
                    f"  Date range {tf}: ~{r['period_days']:.0f} days  "
                    f"({r['total_trades']:,} trades)"
                )
                break

    lines.append("")

    # Per-config table
    for tf in ("5m", "1m"):
        lines.append(f"  ── Timeframe: {tf} ──────────────────────────────────────────")
        header = f"  {'Config':<35} {'Trades':>7} {'WR%':>6} {'WR-gap':>7} {'end+reb':>8} {'DD%':>6} {'avgL':>5} {'MTFskip':>8}"
        lines.append(header)
        lines.append("  " + "─" * 70)
        for cfg in CONFIG_ORDER:
            key = (cfg, tf)
            if key not in results:
                lines.append(f"  {CONFIG_LABELS.get(cfg, cfg):<35}  (no result)")
                continue
            r = results[key]
            mtf_skip = r.get("entries_skipped_by_mtf", 0)
            mtf_str = f"{mtf_skip:,}" if mtf_skip else "-"
            halt = " HALTED" if r.get("halted_early") else ""
            lines.append(
                f"  {CONFIG_LABELS.get(cfg, cfg):<35}"
                f" {r['total_trades']:>7,}"
                f" {r['wr_pct']:>5.1f}%"
                f" {r['wr_gap_pct']:>+6.1f}%"
                f" ${r['end_plus_rebate']:>7.2f}"
                f" {r['max_drawdown_pct']:>5.1f}%"
                f" {r['avg_loss_bps_weighted']:>4.1f}"
                f" {mtf_str:>8}"
                f"{halt}"
            )
        lines.append("")

    # Decision rule evaluation
    lines.append("  ── DECISION RULE EVALUATION (5m 12-month data) ─────────────")
    lines.append("  Rules set in advance — results applied without modification.")
    lines.append("")

    base_5m = results.get(("baseline", "5m"))
    base_1m = results.get(("baseline", "1m"))

    for cfg, label in [
        ("sl30",     "SL30 only"),
        ("mtf",      "MTF only"),
        ("sl30_mtf", "SL30+MTF (proposed live)"),
    ]:
        lines.append(f"  CONFIG: {label}")
        r5 = results.get((cfg, "5m"))
        r1 = results.get((cfg, "1m"))

        if r5 is None:
            lines.append("    (no 5m result)")
            lines.append("")
            continue

        evals_5m = apply_rules(r5, base_5m, "5m")
        evals_1m = apply_rules(r1, base_1m, "1m") if r1 else {}

        # PASS rules (all must hold)
        pass_rules = [
            ("wr >= 80%",         evals_5m.get("wr >= 80%")),
            ("end+reb >= base+10%", evals_5m.get("end+reb >= base+10%")),
            ("dd <= base+5pp",    evals_5m.get("dd <= base+5pp")),
            ("trades >= 50% base", evals_5m.get("trades >= 50% base")),
            ("avg_loss <= 25bps", evals_5m.get("avg_loss <= 25bps")),
        ]
        # FAIL rules (any triggers rejection)
        fail_rules = [
            ("wr >= 78% (NOT fail)", evals_5m.get("wr >= 78% (fail)")),
            ("end+reb >= baseline",  evals_5m.get("end+reb >= baseline")),
            ("dd <= base+10pp",      evals_5m.get("dd <= base+10pp (fail)")),
            ("trades >= 30% base",   evals_5m.get("trades >= 30% base (fail)")),
        ]
        if evals_1m:
            wr_sensitivity = abs(r5["wr_pct"] - r1["wr_pct"]) <= 5.0
            fail_rules.append(("1m/5m WR within 5pts", wr_sensitivity))

        lines.append("    PASS conditions (all required):")
        all_pass = True
        for name, val in pass_rules:
            mark = fmt_pass(val) if val is not None else "?"
            lines.append(f"      [{mark}] {name}")
            if val is False:
                all_pass = False

        lines.append("    FAIL conditions (none must trigger):")
        any_fail = False
        for name, val in fail_rules:
            mark = fmt_pass(val) if val is not None else "?"
            lines.append(f"      [{mark}] {name}")
            if val is False:
                any_fail = True

        passed = all_pass and not any_fail
        if r5 and base_5m:
            delta = r5["end_plus_rebate"] - base_5m["end_plus_rebate"]
            delta_pct = delta / base_5m["end_plus_rebate"] * 100
            lines.append(f"    vs baseline: end+rebate {delta:+.2f} ({delta_pct:+.1f}%)")

        if passed:
            lines.append(f"    ➜ RECOMMENDATION: ✓ DEPLOY {label.upper()}")
        elif not any_fail and not all_pass:
            lines.append(f"    ➜ RECOMMENDATION: ⚠ PARTIAL — pass rules not all met")
        else:
            lines.append(f"    ➜ RECOMMENDATION: ✗ REJECT — fail condition triggered")
        lines.append("")

    lines.append("=" * 72)
    lines.append("  NOTE: 1m results use ~25-day window (MEXC API cap).")
    lines.append("  5m results use ~12 months. Deploy decisions based on 5m only.")
    lines.append("=" * 72)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare backtest JSON results")
    parser.add_argument("results", nargs="+", help="JSON result files to compare")
    parser.add_argument("--output", type=str, default=None,
                        help="Save comparison report as Markdown (default: stdout only)")
    args = parser.parse_args()

    results = load_results(args.results)
    if not results:
        print("No results loaded. Check file paths.")
        sys.exit(1)

    print(f"  Loaded {len(results)} result(s): {list(results.keys())}\n")
    report_lines = build_report(results)
    report_text = "\n".join(report_lines)
    print(report_text)

    if args.output:
        out_path = pathlib.Path(args.output)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write("# Backtest Comparison Report\n\n```\n")
            fh.write(report_text)
            fh.write("\n```\n")
        print(f"\n  Saved report → {out_path}")


if __name__ == "__main__":
    main()
