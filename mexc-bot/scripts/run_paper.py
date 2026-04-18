"""Entrypoint for paper trading phase."""

from __future__ import annotations

import asyncio
import argparse
from datetime import datetime, timezone
import os
import pathlib
import sys

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from exchange.mexc_client import MEXCClient
from exchange.security import verify_startup_permissions
from data.live_feed import LiveFeedMessage, MEXCLiveFeed
from execution.order_manager import OrderManager
from execution.paper_broker import PaperBroker
from monitoring.alerts import AlertPublisher
from monitoring.dashboard import DashboardServer
from monitoring.logger import configure_logging


def _load_config() -> dict:
	with (PROJECT_ROOT / "config" / "config.yaml").open("r", encoding="utf-8") as handle:
		return yaml.safe_load(handle)


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run paper trading with live MEXC feed.")
	parser.add_argument("--max-messages", type=int, default=100, help="Stop after N parsed feed messages.")
	parser.add_argument("--dashboard-host", type=str, default="127.0.0.1", help="Dashboard bind host.")
	parser.add_argument("--dashboard-port", type=int, default=8080, help="Dashboard bind port.")
	return parser.parse_args()


def _require_env(name: str) -> str:
	value = os.getenv(name, "").strip()
	if not value:
		raise RuntimeError(f"Required environment variable {name} is missing.")
	return value


async def _startup_checks() -> None:
	config = _load_config()
	load_dotenv(PROJECT_ROOT / ".env", override=False)

	api_key = _require_env("MEXC_API_KEY")
	api_secret = _require_env("MEXC_API_SECRET")

	startup_cfg = config["risk"]["startup_checks"]
	require_futures = bool(startup_cfg["require_futures_trade_permission"])
	require_withdraw_disabled = bool(startup_cfg["require_withdrawal_disabled"])

	async with MEXCClient(
		api_key=api_key,
		api_secret=api_secret,
		base_url=str(config["exchange"]["base_url_rest"]),
		timeout_seconds=int(config["exchange"]["request_timeout_seconds"]),
		requests_per_second=float(config["exchange"]["rate_limits"]["requests_per_second"]),
		burst_capacity=int(config["exchange"]["rate_limits"]["burst_capacity"]),
	) as client:
		status = await verify_startup_permissions(
			client=client,
			require_futures_trade_permission=require_futures,
			require_withdrawal_disabled=require_withdraw_disabled,
		)

	print(
		"Startup checks passed. "
		f"futures_trade_enabled={status.futures_trade_enabled} "
		f"withdrawal_enabled={status.withdrawal_enabled} source={status.source}"
	)


def _extract_ohlc_from_message(message: LiveFeedMessage) -> tuple[float, float, float] | None:
	payload = message.payload
	if message.event_type != "kline":
		return None

	possible_nodes = [payload]
	if "kline" in payload and isinstance(payload["kline"], dict):
		possible_nodes.append(payload["kline"])

	for node in possible_nodes:
		try:
			high = float(node.get("high") or node.get("h"))
			low = float(node.get("low") or node.get("l"))
			close = float(node.get("close") or node.get("c"))
			return low, high, close
		except (TypeError, ValueError):
			continue
	return None


async def _run_paper_loop(args: argparse.Namespace) -> None:
	config = _load_config()
	log_file = PROJECT_ROOT / "data" / "logs" / "paper.log"
	configure_logging(log_level=str(config["app"].get("log_level", "INFO")), log_file_path=log_file)

	await _startup_checks()

	dashboard = DashboardServer(host=args.dashboard_host, port=args.dashboard_port)
	await dashboard.start()

	alerts = AlertPublisher()
	broker = PaperBroker()
	order_manager = OrderManager(broker=broker, max_entry_age_candles=1)

	candle_index = 0

	async def on_message(message: LiveFeedMessage) -> None:
		nonlocal candle_index
		dashboard.update(last_message_at=message.timestamp.isoformat(), open_orders=len(order_manager.active_orders))

		ohlc = _extract_ohlc_from_message(message)
		if ohlc is None:
			return

		low, high, close = ohlc
		fills = order_manager.process_candle(
			candle_index=candle_index,
			candle_low=low,
			candle_high=high,
			now=message.timestamp,
		)
		dashboard.update(
			open_orders=len(order_manager.active_orders),
			equity=dashboard.state.equity,
			extra={"last_close": close},
		)

		for fill in fills:
			await alerts.publish(
				level="INFO",
				title="Paper fill",
				body=(
					f"order_id={fill.order_id} symbol={fill.symbol} side={fill.side} "
					f"qty={fill.qty:.6f} price={fill.price:.2f}"
				),
			)

		candle_index += 1

	feed = MEXCLiveFeed(
		ws_url=str(config["exchange"]["base_url_ws"]),
		symbol=str(config["exchange"]["symbol"]),
		interval="Min15",
		heartbeat_timeout_seconds=int(config["exchange"]["heartbeat_timeout_seconds"]),
	)

	try:
		await feed.run(callback=on_message, max_messages=args.max_messages)
	except Exception as exc:  # noqa: BLE001
		dashboard.update(last_error=str(exc))
		await alerts.publish(level="ERROR", title="Paper trading halted", body=str(exc))
		order_manager.cancel_all(reason="paper_loop_halt", candle_index=candle_index)
		raise
	finally:
		await dashboard.stop()

	print(
		"Paper loop completed. "
		f"processed_candles={candle_index} "
		f"timestamp={datetime.now(tz=timezone.utc).isoformat()}"
	)


def main() -> None:
	"""Run paper trading loop placeholder.

	This command intentionally keeps behavior minimal until phase 2 is built.
	"""
	args = _parse_args()
	asyncio.run(_run_paper_loop(args))


if __name__ == "__main__":
	main()
