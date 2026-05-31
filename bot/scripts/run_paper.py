"""Live paper-trading runner against the MEXC futures feed.

Polls MEXC futures 15m klines, feeds every newly closed bar through
:class:`PaperSession`, and notifies a Telegram chat on each lifecycle event
(signal, entry, TP1, TP2, exit) plus a daily digest.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pathlib
import signal
import sys
from datetime import datetime, time as dt_time, timedelta, timezone
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from data.historical import _normalize_kline_payload  # noqa: E402
from exchange.mexc_client import MEXCClient  # noqa: E402
from execution.paper_session import PaperEvent, PaperSession  # noqa: E402
from monitoring.logger import configure_logging  # noqa: E402
from monitoring.telegram_notifier import TelegramNotifier, TradeSnapshot  # noqa: E402
from strategy.bollinger import BollingerMeanReversionStrategy  # noqa: E402

LOGGER = logging.getLogger("paper_runner")

POLL_SECONDS = 30
SEED_CANDLES = 300
STATE_FILENAME = "paper_session_state.json"

_TIMEFRAME_TO_MEXC = {
	"1m": "Min1",
	"5m": "Min5",
	"15m": "Min15",
	"30m": "Min30",
	"1h": "Min60",
	"4h": "Hour4",
	"1d": "Day1",
}


def _load_config(path: pathlib.Path) -> Dict[str, Any]:
	with path.open("r", encoding="utf-8") as handle:
		return yaml.safe_load(handle)


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Paper trade against MEXC live data.")
	parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path (relative to project root or absolute).")
	parser.add_argument("--state-file", type=str, default=None, help="State JSON filename under data/ (default derived from config).")
	parser.add_argument("--log-file", type=str, default=None, help="Log file path (default data/logs/paper.log).")
	parser.add_argument("--label", type=str, default=None, help="Short label prefixed to every Telegram message (e.g. '5m-BTC').")
	parser.add_argument("--duration-days", type=float, default=7.0, help="Stop after N days.")
	parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS, help="Kline poll interval.")
	parser.add_argument("--daily-report-hour", type=int, default=None, help="Override daily digest hour (local time).")
	parser.add_argument("--resume", action="store_true", help="Reload previously saved session state.")
	return parser.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

async def _fetch_candles(
	client: MEXCClient,
	symbol: str,
	interval: str,
	*,
	start: Optional[int] = None,
	end: Optional[int] = None,
) -> pd.DataFrame:
	payload = await client.get_klines(symbol=symbol, interval=interval, start=start, end=end)
	rows = _normalize_kline_payload(payload)
	if not rows:
		return pd.DataFrame()
	df = pd.DataFrame(rows)
	return df.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)


async def _seed_history(
	client: MEXCClient, symbol: str, interval: str, n: int
) -> pd.DataFrame:
	end_dt = datetime.now(tz=timezone.utc)
	start_dt = end_dt - timedelta(minutes=15 * (n + 5))
	df = await _fetch_candles(
		client, symbol, interval,
		start=int(start_dt.timestamp()),
		end=int(end_dt.timestamp()),
	)
	if not df.empty:
		df = df[df["close_time"].apply(lambda t: pd.Timestamp(t) <= pd.Timestamp(end_dt))]
	return df.tail(n + 50).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Event → Telegram bridge
# ---------------------------------------------------------------------------

def _build_event_handler(notifier: TelegramNotifier, symbol: str, send: Dict[str, bool]):
	async def handler(event: PaperEvent) -> None:
		p = event.payload
		try:
			if event.kind == "signal" and send["signals"]:
				await notifier.notify_signal(_snapshot(p, symbol))
			elif event.kind == "entry" and send["entries"]:
				await notifier.notify_entry_filled(_snapshot(p, symbol), fill_price=p["fill_price"])
			elif event.kind == "tp1" and send["partials"]:
				await notifier.notify_tp1_hit(
					symbol=p["symbol"], side=p["side"], price=p["price"],
					partial_qty=p["partial_qty"], partial_pnl=p["partial_pnl"],
					new_stop=p["new_stop"],
				)
			elif event.kind == "tp2" and send["partials"]:
				await notifier.notify_tp2_hit(
					symbol=p["symbol"], side=p["side"], price=p["price"],
					partial_qty=p["partial_qty"], partial_pnl=p["partial_pnl"],
					new_stop=p["new_stop"],
				)
			elif event.kind == "exit" and send["exits"]:
				await notifier.notify_exit(
					symbol=p["symbol"], side=p["side"], reason=p["reason"],
					exit_price=p["exit_price"], net_pnl=p["net_pnl"],
					total_pnl_pct=p["total_pnl_pct"], equity_after=p["equity_after"],
					bars_held=p["bars_held"],
				)
		except Exception as exc:  # noqa: BLE001
			LOGGER.exception("Event handler failed: %s", exc)

	return handler


def _snapshot(p: Dict[str, Any], symbol: str) -> TradeSnapshot:
	return TradeSnapshot(
		side=p["side"],
		symbol=p.get("symbol", symbol),
		entry_price=p["entry_price"],
		stop_price=p["stop_price"],
		tp1_price=p.get("tp1_price"),
		tp2_price=p.get("tp2_price"),
		final_tp_price=p["take_profit_price"],
		qty=p["qty"],
		equity=p["equity"],
	)


# ---------------------------------------------------------------------------
# Daily digest scheduler
# ---------------------------------------------------------------------------

async def _daily_report_loop(
	session: PaperSession,
	notifier: TelegramNotifier,
	hour_local: int,
	stop_event: asyncio.Event,
	local_tz: ZoneInfo,
) -> None:
	while not stop_event.is_set():
		now = datetime.now(tz=local_tz)
		next_run = datetime.combine(now.date(), dt_time(hour=hour_local, tzinfo=local_tz))
		if next_run <= now:
			next_run += timedelta(days=1)
		try:
			await asyncio.wait_for(stop_event.wait(), timeout=(next_run - now).total_seconds())
			return
		except asyncio.TimeoutError:
			pass

		report_date = (datetime.now(tz=local_tz) - timedelta(days=1)).strftime("%Y-%m-%d")
		stats = session.daily_summary(report_date)
		try:
			await notifier.notify_daily_report(date_utc=report_date, **stats)
		except Exception as exc:  # noqa: BLE001
			LOGGER.exception("Daily report send failed: %s", exc)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def _run(args: argparse.Namespace) -> None:
	config_path = pathlib.Path(args.config)
	if not config_path.is_absolute():
		config_path = PROJECT_ROOT / config_path
	config = _load_config(config_path)

	if args.log_file:
		log_file = pathlib.Path(args.log_file)
		if not log_file.is_absolute():
			log_file = PROJECT_ROOT / log_file
	else:
		log_file = PROJECT_ROOT / "data" / "logs" / "paper.log"
	configure_logging(log_level=str(config["app"].get("log_level", "INFO")), log_file_path=log_file)

	load_dotenv(PROJECT_ROOT / ".env", override=False)
	if args.label:
		os.environ["BOT_LABEL"] = args.label

	notif_cfg = config.get("notifications", {}).get("telegram", {}) or {}
	send = {
		"signals": bool(notif_cfg.get("send_signals", True)),
		"entries": bool(notif_cfg.get("send_entries", True)),
		"partials": bool(notif_cfg.get("send_partials", True)),
		"exits": bool(notif_cfg.get("send_exits", True)),
		"errors": bool(notif_cfg.get("send_errors", True)),
	}
	report_hour = (
		args.daily_report_hour
		if args.daily_report_hour is not None
		else int(notif_cfg.get("daily_report_hour", 0))
	)
	tz_name = str(config["app"].get("timezone", "Asia/Colombo"))
	try:
		local_tz = ZoneInfo(tz_name)
	except Exception:
		local_tz = ZoneInfo("UTC")

	symbol = str(config["exchange"]["symbol"])
	timeframe = str(config["exchange"].get("timeframe", "15m")).lower()
	interval = _TIMEFRAME_TO_MEXC.get(timeframe, "Min15")
	if timeframe not in _TIMEFRAME_TO_MEXC:
		LOGGER.warning("Unknown timeframe %r in config, defaulting to Min15", timeframe)

	strategy = BollingerMeanReversionStrategy(config)
	session = PaperSession(config=config, strategy=strategy)

	state_filename = args.state_file or STATE_FILENAME
	state_path = PROJECT_ROOT / "data" / state_filename
	if args.resume:
		session.load_state(state_path)
		LOGGER.info("Resumed: equity=%.2f trades=%d", session.equity, len(session.ledger))

	notifier = TelegramNotifier()
	await notifier.start()
	session._event_cb = _build_event_handler(notifier, symbol, send)  # type: ignore[attr-defined]

	stop_event = asyncio.Event()

	def _handle_stop(*_a: Any) -> None:
		LOGGER.info("Stop signal received.")
		stop_event.set()

	loop = asyncio.get_running_loop()
	for sig in (signal.SIGINT, signal.SIGTERM):
		try:
			loop.add_signal_handler(sig, _handle_stop)
		except NotImplementedError:  # pragma: no cover
			pass

	async with MEXCClient(
		api_key=os.getenv("MEXC_API_KEY", ""),
		api_secret=os.getenv("MEXC_API_SECRET", ""),
		base_url=str(config["exchange"]["base_url_rest"]),
		timeout_seconds=int(config["exchange"]["request_timeout_seconds"]),
		requests_per_second=float(config["exchange"]["rate_limits"]["requests_per_second"]),
		burst_capacity=int(config["exchange"]["rate_limits"]["burst_capacity"]),
	) as client:
		LOGGER.info("Seeding %d candles for %s %s ...", SEED_CANDLES, symbol, interval)
		history = await _seed_history(client, symbol, interval, SEED_CANDLES)
		if history.empty:
			raise RuntimeError("Failed to seed historical candles.")
		last_time = pd.Timestamp(history["open_time"].iloc[-1])
		LOGGER.info("Seeded %d candles; last=%s", len(history), last_time.isoformat())

		await notifier.notify_startup(
			mode="paper",
			symbol=symbol,
			timeframe=str(config["exchange"]["timeframe"]),
			equity=session.equity,
			strategy=str(config["strategy"]["name"]),
		)

		report_task = asyncio.create_task(
			_daily_report_loop(session, notifier, report_hour, stop_event, local_tz),
			name="daily-report",
		)

		end_at = datetime.now(tz=timezone.utc) + timedelta(days=args.duration_days)
		shutdown_reason = "duration_elapsed"
		try:
			while not stop_event.is_set():
				if datetime.now(tz=timezone.utc) >= end_at:
					break
				try:
					fresh = await _fetch_candles(
						client, symbol, interval,
						start=int((last_time - timedelta(minutes=60)).timestamp()),
						end=int(datetime.now(tz=timezone.utc).timestamp()),
					)
				except Exception as exc:  # noqa: BLE001
					LOGGER.exception("Kline poll failed: %s", exc)
					if send["errors"]:
						await notifier.notify_error(title="Kline poll failed", detail=str(exc))
					await _wait(args.poll_seconds, stop_event)
					continue

				if fresh.empty:
					await _wait(args.poll_seconds, stop_event)
					continue

				new_bars = fresh[fresh["open_time"] > last_time]
				now_ts = datetime.now(tz=timezone.utc)
				closed = new_bars[
					new_bars["close_time"].apply(lambda t: pd.Timestamp(t) <= pd.Timestamp(now_ts))
				]
				for _, row in closed.iterrows():
					history = pd.concat([history, row.to_frame().T], ignore_index=True)
					history = history.tail(SEED_CANDLES + 100).reset_index(drop=True)
					last_time = pd.Timestamp(row["open_time"])
					try:
						await session.on_new_candle(history)
					except Exception as exc:  # noqa: BLE001
						LOGGER.exception("Session error: %s", exc)
						if send["errors"]:
							await notifier.notify_error(title="Session error", detail=str(exc))

				session.save_state(state_path)
				await _wait(args.poll_seconds, stop_event)
		except Exception as exc:  # noqa: BLE001
			shutdown_reason = f"exception: {exc!r}"
			LOGGER.exception("Fatal runner error: %s", exc)
			if send["errors"]:
				await notifier.notify_error(title="Paper runner halted", detail=str(exc))
		finally:
			report_task.cancel()
			try:
				await report_task
			except (asyncio.CancelledError, Exception):  # noqa: BLE001
				pass
			session.save_state(state_path)
			await notifier.notify_shutdown(reason=shutdown_reason)
			await notifier.stop()


async def _wait(seconds: float, stop_event: asyncio.Event) -> None:
	try:
		await asyncio.wait_for(stop_event.wait(), timeout=seconds)
	except asyncio.TimeoutError:
		pass


def main() -> None:
	args = _parse_args()
	asyncio.run(_run(args))


if __name__ == "__main__":
	main()
