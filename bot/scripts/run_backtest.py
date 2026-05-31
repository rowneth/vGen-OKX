"""Run strategy backtests and generate reports."""

from __future__ import annotations

import asyncio
import argparse
import pathlib
import sys
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from backtest.engine import BacktestEngine
from backtest.metrics import calculate_metrics
from backtest.report import generate_html_report, generate_markdown_report
from data.storage import initialize_audit_db, load_parquet, persist_backtest_audit
from strategy.base import Strategy
from strategy.bollinger import BollingerMeanReversionStrategy
from strategy.cipher_confluence import CipherConfluenceStrategy
from strategy.ema_cipher_divergence import EmaCipherDivergenceStrategy
from strategy.rsi_wt import RsiWtStrategy


def build_strategy(config: dict) -> Strategy:
	"""Instantiate the strategy selected in config.

	Args:
		config: Parsed config dict.

	Returns:
		Strategy implementation.

	Raises:
		ValueError: If strategy name is unknown.
	"""
	name = str(config["strategy"].get("name", "cipher_confluence")).lower()
	if name == "cipher_confluence":
		return CipherConfluenceStrategy(config)
	if name in {"bollinger_mean_reversion", "bollinger"}:
		return BollingerMeanReversionStrategy(config)
	if name in {"rsi_wt", "rsi_wavetrend", "wtf"}:
		return RsiWtStrategy(config)
	if name in {"ema_cipher_divergence", "ema_cipher", "ema_vmc"}:
		return EmaCipherDivergenceStrategy(config)
	raise ValueError(f"Unknown strategy: {name}")

console = Console()


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments.

	Returns:
		Parsed CLI namespace.
	"""
	parser = argparse.ArgumentParser(description="Run Bollinger strategy backtest.")
	parser.add_argument(
		"--config",
		type=str,
		default="config/config.yaml",
		help="Path to config YAML (relative to project root or absolute).",
	)
	parser.add_argument(
		"--data",
		type=str,
		default=None,
		help="Path to parquet candles. Defaults to data/historical/<symbol>_<timeframe>.parquet",
	)
	parser.add_argument("--rows", type=int, default=None, help="Optional number of most recent rows to use.")
	parser.add_argument(
		"--report-prefix",
		type=str,
		default="backtest_report",
		help="Prefix for output report files.",
	)
	return parser.parse_args()


def load_config(path: pathlib.Path | None = None) -> dict:
	"""Load config YAML.

	Args:
		path: Optional absolute or project-relative path. Defaults to config/config.yaml.

	Returns:
		Parsed config dict.
	"""
	if path is None:
		path = PROJECT_ROOT / "config" / "config.yaml"
	with path.open("r", encoding="utf-8") as handle:
		return yaml.safe_load(handle)


def main() -> None:
	"""Execute backtest and report generation."""
	args = parse_args()
	cfg_path = pathlib.Path(args.config)
	if not cfg_path.is_absolute():
		cfg_path = PROJECT_ROOT / cfg_path
	config = load_config(cfg_path)

	symbol = config["exchange"]["symbol"]
	timeframe = config["exchange"]["timeframe"]
	default_data = PROJECT_ROOT / "data" / "historical" / f"{symbol}_{timeframe}.parquet"
	data_path = pathlib.Path(args.data) if args.data else default_data

	if not data_path.exists():
		raise FileNotFoundError(
			f"Historical data file not found at {data_path}. Run scripts/download_data.py first."
		)

	candles = load_parquet(data_path)
	candles["open_time"] = pd.to_datetime(candles["open_time"], utc=True)
	candles["close_time"] = pd.to_datetime(candles["close_time"], utc=True)
	candles = candles.sort_values("open_time").reset_index(drop=True)

	if args.rows is not None:
		candles = candles.tail(args.rows).reset_index(drop=True)

	strategy = build_strategy(config)
	engine = BacktestEngine(config)
	result = engine.run(candles, strategy)
	metrics = calculate_metrics(
		result=result,
		win_rate_drop_pp=float(config["backtest"]["stress_test"]["win_rate_drop_percentage_points"]),
	)

	out_dir = PROJECT_ROOT / config["reporting"]["output_dir"]
	md_path = out_dir / f"{args.report_prefix}.md"
	html_path = out_dir / f"{args.report_prefix}.html"
	generate_markdown_report(result, metrics, md_path)
	generate_html_report(result, metrics, html_path)

	run_id = f"backtest-{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
	audit_db_path = PROJECT_ROOT / config["storage"]["sqlite_path"]
	asyncio.run(initialize_audit_db(audit_db_path))
	asyncio.run(persist_backtest_audit(result=result, sqlite_path=audit_db_path, run_id=run_id))

	table = Table(title="Backtest Summary")
	table.add_column("Metric")
	table.add_column("Value", justify="right")
	table.add_row("Initial Equity", f"{metrics['initial_equity']:.2f}")
	table.add_row("Final Equity", f"{metrics['final_equity']:.2f}")
	table.add_row("Total Return %", f"{metrics['total_return_pct']:.2f}")
	table.add_row("Sharpe", f"{metrics['sharpe_ratio']:.4f}")
	table.add_row("Max Drawdown %", f"{metrics['max_drawdown_pct'] * 100.0:.2f}")
	table.add_row("Win Rate %", f"{metrics['win_rate'] * 100.0:.2f}")
	table.add_row("Profit Factor", f"{metrics['profit_factor']:.4f}")
	table.add_row("Total Trades", f"{metrics['total_trades']}")
	table.add_row("Fees Paid", f"{metrics['total_fees_paid']:.2f}")
	console.print(table)
	console.print(f"Markdown report: {md_path}")
	console.print(f"HTML report: {html_path}")
	console.print(f"Audit DB: {audit_db_path}")
	console.print(f"Run ID: {run_id}")


if __name__ == "__main__":
	main()
