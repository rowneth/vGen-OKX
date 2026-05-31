"""Quick A/B grid: min_bar_range_bps × alternate_direction sweep.

Goal: find a config that increases trade count without crushing WR-gap or
ROI-with-rebate. Builds on top of the existing ``run_one`` helper.
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

# Reuse the existing harness (loads BASE_CONFIG and runs the session).
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from backtest_volume_farmer import run_one, DATA_FILE  # noqa: E402

BASE_PATH = PROJECT_ROOT / "config" / "config_volume_farmer_optimal.yaml"


def main() -> None:
    print(f"Loading {DATA_FILE} ...")
    df = pd.read_parquet(DATA_FILE)
    df.columns = [c.lower() for c in df.columns]
    print(f"Loaded {len(df):,} bars\n")

    base_cfg = yaml.safe_load(BASE_PATH.read_text())
    # Force full-history run.
    base_cfg.setdefault("risk", {})["stop_on_volume_target"] = False
    base_cfg["risk"]["consecutive_losses_limit"] = 9999
    base_cfg["risk"]["consecutive_losses_cooldown_bars"] = 0
    base_cfg.setdefault("target", {})["volume_usd"] = 999_999_999

    range_grid = [4.0, 3.0, 2.0, 1.0]
    alt_grid = [True, False]

    rows: list[dict] = []
    for r in range_grid:
        for alt in alt_grid:
            cfg = copy.deepcopy(base_cfg)
            cfg["farmer"]["entry"]["min_bar_range_bps"] = r
            cfg["farmer"]["alternate_direction"] = alt
            tag = f"range>={r}bps alt={'on' if alt else 'off'}"
            print(f"  Running {tag} ...", end="", flush=True)
            res = run_one(df, tp_bps=0, sl_bps=0, override_cfg=cfg)
            res["range"] = r
            res["alt"] = alt
            rows.append(res)
            print(
                f"  trades={res['trades']:>4}  WR={res['win_rate_pct']:5.2f}%  "
                f"gap={res['wr_gap_pct']:+5.2f}%  end+reb=${res['end_equity_with_rebate']:6.2f}"
                f"  vol=${res['total_volume_usd']:>10,.0f}"
            )

    print()
    hdr = ["range", "alt", "trades", "WR%", "gap%", "gross", "fees", "rebate",
           "end+reb", "vol $"]
    fmt = "{:>5}  {:>4}  {:>6}  {:>6}  {:>6}  {:>7}  {:>7}  {:>7}  {:>7}  {:>11}"
    print(fmt.format(*hdr))
    print("─" * 90)
    for r in rows:
        print(fmt.format(
            f"{r['range']:.0f}",
            "on" if r["alt"] else "off",
            r["trades"],
            f"{r['win_rate_pct']:.1f}",
            f"{r['wr_gap_pct']:+.1f}",
            f"{r['gross_pnl']:+.2f}",
            f"{r['total_gross_fees']:.2f}",
            f"{r['total_rebate']:+.2f}",
            f"{r['end_equity_with_rebate']:.2f}",
            f"{r['total_volume_usd']:,.0f}",
        ))
    print()

    # Pick winner: highest end+rebate
    winner = max(rows, key=lambda x: x["end_equity_with_rebate"])
    print(f"Best end+rebate : range={winner['range']:.0f} alt={'on' if winner['alt'] else 'off'} "
          f"-> ${winner['end_equity_with_rebate']:.2f} ({winner['trades']} trades, "
          f"WR {winner['win_rate_pct']:.1f}%, gap {winner['wr_gap_pct']:+.1f}%)")
    most_trades = max(rows, key=lambda x: x["trades"])
    print(f"Most trades     : range={most_trades['range']:.0f} alt={'on' if most_trades['alt'] else 'off'} "
          f"-> {most_trades['trades']} trades (WR {most_trades['win_rate_pct']:.1f}%, "
          f"end+reb ${most_trades['end_equity_with_rebate']:.2f})")
    # baseline = current live (range=4, alt=on)
    baseline = next(x for x in rows if x["range"] == 4.0 and x["alt"])
    print(f"Baseline        : range=4 alt=on -> {baseline['trades']} trades, "
          f"WR {baseline['win_rate_pct']:.1f}%, end+reb ${baseline['end_equity_with_rebate']:.2f}")


if __name__ == "__main__":
    main()
