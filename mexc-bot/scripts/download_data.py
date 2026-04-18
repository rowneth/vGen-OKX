"""Download MEXC historical futures data and save it as Parquet."""

from __future__ import annotations

import argparse
import asyncio
import logging
import pathlib
import sys
from datetime import datetime, timedelta, timezone

import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from data.historical import DownloadRequest, HistoricalDataDownloader
from data.storage import save_parquet
from exchange.mexc_client import MEXCClient


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments.

	Returns:
		Parsed argument namespace.
	"""
	parser = argparse.ArgumentParser(description="Download MEXC historical klines.")
	parser.add_argument("--months", type=int, default=12, help="Number of months to download.")
	parser.add_argument("--symbol", type=str, default=None, help="Override symbol from config.")
	parser.add_argument("--timeframe", type=str, default=None, help="Override timeframe from config.")
	parser.add_argument(
		"--output",
		type=str,
		default=None,
		help="Output parquet file path. Defaults to data/historical/<symbol>_<timeframe>.parquet",
	)
	return parser.parse_args()


def load_config(config_path: pathlib.Path) -> dict:
	"""Load YAML configuration.

	Args:
		config_path: Path to yaml config.

	Returns:
		Parsed config dictionary.
	"""
	with config_path.open("r", encoding="utf-8") as handle:
		return yaml.safe_load(handle)


def timeframe_to_mexc_interval(timeframe: str) -> str:
	"""Convert generic timeframe string to MEXC interval string.

	Args:
		timeframe: Timeframe like ``15m``.

	Returns:
		MEXC interval string.

	Raises:
		ValueError: If timeframe is unsupported.
	"""
	mapping = {
		"1m": "Min1",
		"5m": "Min5",
		"15m": "Min15",
		"30m": "Min30",
		"1h": "Min60",
		"4h": "Hour4",
		"1d": "Day1",
	}
	if timeframe not in mapping:
		raise ValueError(f"unsupported timeframe: {timeframe}")
	return mapping[timeframe]


async def run() -> None:
	"""Run historical data download workflow."""
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(name)s - %(message)s",
	)
	args = parse_args()

	config = load_config(PROJECT_ROOT / "config" / "config.yaml")
	symbol = args.symbol or config["exchange"]["symbol"]
	timeframe = args.timeframe or config["exchange"]["timeframe"]
	interval = timeframe_to_mexc_interval(timeframe)

	now = datetime.now(tz=timezone.utc)
	start = now - timedelta(days=30 * args.months)

	output_path = pathlib.Path(args.output) if args.output else (
		PROJECT_ROOT / "data" / "historical" / f"{symbol}_{timeframe}.parquet"
	)

	logging.info(
		"Starting download symbol=%s timeframe=%s months=%d",
		symbol,
		timeframe,
		args.months,
	)

	async with MEXCClient(
		api_key="",
		api_secret="",
		base_url=config["exchange"]["base_url_rest"],
		timeout_seconds=int(config["exchange"]["request_timeout_seconds"]),
		requests_per_second=float(config["exchange"]["rate_limits"]["requests_per_second"]),
		burst_capacity=int(config["exchange"]["rate_limits"]["burst_capacity"]),
	) as client:
		downloader = HistoricalDataDownloader(client=client)
		frame = await downloader.download_klines(
			DownloadRequest(
				symbol=symbol,
				interval=interval,
				start_time=start,
				end_time=now,
				chunk_days=7,
			)
		)

	if frame.empty:
		logging.warning("No data downloaded. Check symbol/timeframe and API response.")
		return

	compression = config["storage"].get("parquet_compression", "snappy")
	save_parquet(frame, output_path=output_path, compression=compression)
	logging.info("Saved %d rows to %s", len(frame), output_path)


if __name__ == "__main__":
	asyncio.run(run())
