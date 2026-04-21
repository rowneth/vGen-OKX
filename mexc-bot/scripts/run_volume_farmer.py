"""Live paper runner for the volume-farmer bot.

Polls BTC_USDT 5m klines from MEXC (same feed as run_paper.py but fully isolated
state / log / config) and drives :class:`VolumeFarmerSession`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pathlib
import signal
import sys
from datetime import datetime, time as dt_time, timedelta
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
from execution.live_volume_executor import LiveVolumeExecutor  # noqa: E402
from execution.volume_farmer import FarmerEvent, VolumeFarmerSession  # noqa: E402
from monitoring.logger import configure_logging  # noqa: E402
from monitoring.telegram_notifier import TelegramNotifier  # noqa: E402

LOGGER = logging.getLogger("volume_farmer_runner")

POLL_SECONDS = 30
SEED_CANDLES = 50
STATE_FILENAME = "volume_farmer_state.json"


_TIMEFRAME_TO_MEXC = {
	"1m": "Min1", "5m": "Min5", "15m": "Min15",
	"30m": "Min30", "1h": "Min60", "4h": "Hour4", "1d": "Day1",
}


def _load_config(path: pathlib.Path) -> Dict[str, Any]:
	with path.open("r", encoding="utf-8") as handle:
		return yaml.safe_load(handle)


def _parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Run the volume farmer paper bot.")
	p.add_argument("--config", type=str, default="config/config_volume_farmer.yaml")
	p.add_argument("--state-file", type=str, default=None)
	p.add_argument("--log-file", type=str, default="data/logs/volume_farmer.log")
	p.add_argument("--label", type=str, default="VOL-FARM")
	p.add_argument("--duration-days", type=float, default=7.0)
	p.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
	p.add_argument("--resume", action="store_true")
	p.add_argument(
		"--live", action="store_true",
		help="Place REAL orders on MEXC in addition to paper simulation. "
			 "Requires LIVE_FARMER_ACK=I_UNDERSTAND in env.",
	)
	p.add_argument(
		"--live-dry-run", action="store_true",
		help="With --live: log order intents but do not submit to MEXC.",
	)
	p.add_argument(
		"--max-live-trades", type=int, default=5,
		help="Hard cap on number of real orders placed in --live mode.",
	)
	p.add_argument(
		"--max-live-notional", type=float, default=200.0,
		help="Per-trade notional cap (USD) in --live mode.",
	)
	p.add_argument(
		"--live-leverage", type=int, default=20,
		help="Leverage for real orders in --live mode.",
	)
	return p.parse_args()


async def _fetch_candles(
	client: MEXCClient, symbol: str, interval: str,
	*, start: Optional[int] = None, end: Optional[int] = None,
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
	df = await _fetch_candles(client, symbol, interval)
	if len(df) < n:
		return df
	return df.tail(n).reset_index(drop=True)


def _fmt(v: float, digits: int = 2) -> str:
	return f"{v:,.{digits}f}"


def _build_event_handler(
	notifier: TelegramNotifier,
	symbol: str,
	send_milestones: bool,
	live_executor: Optional[LiveVolumeExecutor] = None,
):
	loop = asyncio.get_event_loop()

	def _escape(text: str) -> str:
		for ch in "_*[]()~`>#+-=|{}.!":
			text = text.replace(ch, f"\\{ch}")
		return text

	def _n(v: float, d: int = 2) -> str:
		"""Escape a plain number."""
		return _escape(f"{v:,.{d}f}")

	def _money(v: float, d: int = 2) -> str:
		"""Format and escape a dollar amount."""
		return _escape(f"${v:,.{d}f}")

	def _signed_money(v: float, d: int = 4) -> str:
		"""Format with explicit sign prefix and escape."""
		s = f"+${v:,.{d}f}" if v >= 0 else f"-${abs(v):,.{d}f}"
		return _escape(s)

	def _fee_label(fee_type: str) -> str:
		"""Return escaped fee-type badge: maker ✅ or taker ⚠️."""
		if fee_type == "maker":
			return "\\[maker 0\\.01% ✅\\]"
		return "\\[taker 0\\.05% ⚠️\\]"

	DIV = "━━━━━━━━━━━━━━━━━━━━━━"

	def handler(evt: FarmerEvent) -> None:
		LOGGER.info("farmer evt %s %s", evt.kind, evt.payload)
		p = evt.payload

		# Dispatch to live executor FIRST so real order is in-flight while the
		# Telegram message is formatted/sent.
		if live_executor is not None:
			if evt.kind == "entry":
				asyncio.run_coroutine_threadsafe(
					live_executor.handle_entry(p), loop,
				)
			elif evt.kind == "exit":
				asyncio.run_coroutine_threadsafe(
					live_executor.handle_exit(p), loop,
				)

		if evt.kind == "entry":
			side = p["side"]
			side_emoji = "🟢" if side == "long" else "🔴"
			trade_num = p["round_trips"] + 1
			vol_pct = p["volume"] / max(p.get("volume_target", 1.0), 1.0) * 100
			wr = (p["wins"] / p["round_trips"] * 100) if p["round_trips"] > 0 else 0.0
			msg = (
				f"{side_emoji} *{side.upper()}* · \\#{trade_num} `{symbol}`\n"
				f"`{int(round(p['leverage']))}x` leverage\n"
				f"{DIV}\n"
				f"Entry   `{_n(p['price'], 1)}`\n"
				f"TP  →   `{_n(p['tp'], 1)}`   \\(\\+{_escape(str(p['tp_bps']))} bps\\)\n"
				f"SL  →   `{_n(p['sl'], 1)}`   \\(\\-{_escape(str(p['sl_bps']))} bps\\)\n"
				f"{DIV}\n"
				f"Size     {_money(p['notional'])}   Margin {_money(p['margin'], 2)}\n"
				f"Fee      {_money(p['open_fee'], 4)}   {_fee_label(p['fee_type'])}\n"
				f"{DIV}\n"
				f"Capital   {_money(p['capital'], 2)}\n"
				f"Volume    {_money(p['volume'], 0)} / {_money(p.get('volume_target', 0.0), 0)}   \\({_n(vol_pct, 1)}%\\)\n"
				f"Record    `{p['wins']}W` `{p['losses']}L`   `{_n(wr, 1)}%`"
			)
			asyncio.run_coroutine_threadsafe(notifier.send_raw(msg), loop)

		elif evt.kind == "exit":
			reason = p["reason"]
			if reason == "tp":
				result_emoji, result_label = "✅", "TP"
			elif reason in ("sl", "sl_ambiguous"):
				result_emoji, result_label = "❌", "SL"
			else:
				result_emoji, result_label = "⏹", "TIME"
			trade_num = p["round_trips"]
			vol_pct = p["volume"] / max(p.get("volume_target", 1.0), 1.0) * 100
			streak_line = (
				f"Streak loss   `{p['consec_losses']}` / `10`\n"
				if result_label == "SL" and p.get("consec_losses", 0) > 0
				else ""
			)
			msg = (
				f"{result_emoji} *{p['side'].upper()} {result_label}* · \\#{trade_num} `{symbol}`\n"
				f"`{_n(p['entry_price'], 1)}` → `{_n(p['exit_price'], 1)}`   \\({p.get('bars_held', '?')} bars\\)\n"
				f"{DIV}\n"
				f"Gross PnL    {_signed_money(p['gross_pnl'])}\n"
				f"Open fee     {_signed_money(-p['open_fee'])}   {_fee_label('maker')}\n"
				f"Close fee    {_signed_money(-p['close_fee'])}   {_fee_label(p['close_fee_type'])}\n"
				f"Net PnL      {_signed_money(p['net_pnl'])}\n"
				f"{streak_line}"
				f"{DIV}\n"
				f"Capital   {_money(p['capital'], 2)}   Δ {_signed_money(p['net_pnl'])}\n"
				f"Volume    {_money(p['volume'], 0)} / {_money(p.get('volume_target', 0.0), 0)}\n"
				f"Record    `{p['wins']}W` `{p['losses']}L`   `{_n(p.get('win_rate_pct', 0.0), 1)}%`"
			)
			asyncio.run_coroutine_threadsafe(notifier.send_raw(msg), loop)

		elif evt.kind == "milestone" and send_milestones:
			pct = int(p["pct"] * 100)
			msg = (
				f"🎯 *Milestone {pct}% reached\\!*\n"
				f"{DIV}\n"
				f"Volume   {_money(p['volume'], 0)} USD\n"
				f"Equity   {_money(p['equity'], 4)}\n"
				f"Fees     {_money(p['fees_gross'], 2)}   PnL {_signed_money(p['pnl'], 2)}"
			)
			asyncio.run_coroutine_threadsafe(notifier.send_raw(msg), loop)

		elif evt.kind == "halt":
			msg = (
				f"🛑 *HALTED*\n"
				f"{DIV}\n"
				f"Reason   `{_escape(p['reason'])}`\n"
				f"Equity   {_money(p['equity'], 4)}\n"
				f"Volume   {_money(p['volume'], 0)} USD"
			)
			asyncio.run_coroutine_threadsafe(notifier.send_raw(msg), loop)

	return handler


async def _daily_report_loop(
	session: VolumeFarmerSession,
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

		s = session.summary()
		def esc(t: str) -> str:
			for ch in "_*[]()~`>#+-=|{}.!":
				t = t.replace(ch, f"\\{ch}")
			return t
		msg = (
			f"📊 *Daily digest — volume farmer*\n"
			f"Volume: `{esc(_fmt(s['volume_usd']))}` USD  "
			f"\\({esc(str(s['volume_target_pct']))}% of target\\)\n"
			f"Round\\-trips: `{s['round_trips']}`  "
			f"Win rate: `{esc(str(s['win_rate_pct']))}`%\n"
			f"Equity: `{esc(_fmt(s['equity'], 4))}`  "
			f"Δ: `{esc(_fmt(s['equity_delta'], 4))}`\n"
			f"Fees gross: `{esc(_fmt(s['fees_gross'], 2))}`  "
			f"Rebate est: `{esc(_fmt(s['rebate_estimate'], 2))}`  "
			f"Net: `{esc(_fmt(s['fees_net'], 2))}`\n"
			f"PnL: `{esc(_fmt(s['total_pnl'], 2))}`  "
			f"Fee cover: `{esc(str(s['fee_cover_pct']))}`%"
		)
		try:
			await notifier.send_raw(msg)
		except Exception as exc:  # noqa: BLE001
			LOGGER.exception("daily digest send failed: %s", exc)


async def _run(args: argparse.Namespace) -> None:
	config_path = pathlib.Path(args.config)
	if not config_path.is_absolute():
		config_path = PROJECT_ROOT / config_path
	config = _load_config(config_path)

	log_file = pathlib.Path(args.log_file)
	if not log_file.is_absolute():
		log_file = PROJECT_ROOT / log_file
	configure_logging(log_level=str(config["app"].get("log_level", "INFO")), log_file_path=log_file)

	load_dotenv(PROJECT_ROOT / ".env", override=False)
	if args.label:
		os.environ["BOT_LABEL"] = args.label

	tz_name = str(config["app"].get("timezone", "Asia/Colombo"))
	try:
		local_tz = ZoneInfo(tz_name)
	except Exception:
		local_tz = ZoneInfo("UTC")

	symbol = str(config["exchange"]["symbol"])
	timeframe = str(config["exchange"].get("timeframe", "5m")).lower()
	interval = _TIMEFRAME_TO_MEXC.get(timeframe, "Min5")

	session = VolumeFarmerSession(config=config)

	state_filename = args.state_file or STATE_FILENAME
	state_path = PROJECT_ROOT / "data" / state_filename
	if args.resume:
		session.load_state(state_path)
		LOGGER.info(
			"Resumed: equity=%.2f vol=%.2f trips=%d",
			session.equity, session.total_volume_usd, session.round_trips,
		)

	notif_cfg = config.get("notifications", {}).get("telegram", {}) or {}
	notifier = TelegramNotifier()
	await notifier.start()

	# --- LIVE MODE GUARD ------------------------------------------------
	live_executor: Optional[LiveVolumeExecutor] = None
	if args.live:
		ack = os.getenv("LIVE_FARMER_ACK", "").strip()
		if ack != "I_UNDERSTAND":
			raise RuntimeError(
				"--live requires env LIVE_FARMER_ACK=I_UNDERSTAND to be set."
			)
		LOGGER.warning(
			"LIVE MODE ENABLED — real orders will be placed on MEXC "
			"(max_trades=%d, max_notional=$%.2f, leverage=%dx, dry_run=%s)",
			args.max_live_trades, args.max_live_notional,
			args.live_leverage, args.live_dry_run,
		)
		await notifier.send_raw(
			f"⚠️ *LIVE MODE* engaged\n"
			f"Max trades: `{args.max_live_trades}`\n"
			f"Max notional: `\\${args.max_live_notional:.2f}`\n"
			f"Leverage: `{args.live_leverage}x`\n"
			f"Dry\\-run: `{str(args.live_dry_run).lower()}`"
		)

	session.event_callback = _build_event_handler(
		notifier,
		symbol,
		bool(notif_cfg.get("send_milestones", True)),
		live_executor=None,  # will be rebound after client opens if --live
	)

	stop_event = asyncio.Event()

	def _handle_stop(*_a: Any) -> None:
		LOGGER.info("Stop signal received.")
		stop_event.set()

	loop = asyncio.get_running_loop()
	for sig in (signal.SIGINT, signal.SIGTERM):
		try:
			loop.add_signal_handler(sig, _handle_stop)
		except NotImplementedError:
			signal.signal(sig, _handle_stop)

	report_hour = int(notif_cfg.get("daily_report_hour", 0))
	report_task = asyncio.create_task(
		_daily_report_loop(session, notifier, report_hour, stop_event, local_tz),
		name="farmer-daily-report",
	)

	api_key = os.getenv("MEXC_API_KEY", "").strip()
	api_secret = os.getenv("MEXC_API_SECRET", "").strip()
	client = MEXCClient(
		api_key=api_key or "paper",
		api_secret=api_secret or "paper",
		base_url="https://contract.mexc.com",
	)
	try:
		async with client:
			LOGGER.info("Seeding %d candles for %s %s ...", SEED_CANDLES, symbol, interval)
			history = await _seed_history(client, symbol, interval, SEED_CANDLES)
			if history.empty:
				LOGGER.error("No candles returned; aborting.")
				return
			LOGGER.info("Seeded %d candles; last=%s", len(history), history.iloc[-1]["open_time"])

			# Build LiveExecutor once the client is inside the async context so
			# it can issue authenticated calls.
			if args.live:
				live_executor = LiveVolumeExecutor(
					client=client,
					symbol=symbol,
					leverage=args.live_leverage,
					max_live_trades=args.max_live_trades,
					max_notional_usd=args.max_live_notional,
					dry_run=args.live_dry_run,
					notify_callback=notifier.send_raw,
				)
				await live_executor.startup()
				# Re-bind handler now that executor exists.
				session.event_callback = _build_event_handler(
					notifier,
					symbol,
					bool(notif_cfg.get("send_milestones", True)),
					live_executor=live_executor,
				)
				LOGGER.info("Live executor ready.")

			# Drop forming bar if any — keep only closed
			now_ms = int(datetime.now(tz=timezone_utc()).timestamp() * 1000)
			history = _drop_forming(history, interval, now_ms)

			deadline = datetime.now() + timedelta(days=args.duration_days)
			last_ts = history.iloc[-1]["open_time"] if not history.empty else None

			await notifier.send_raw(
				"🚜 *Volume farmer started*\n"
				f"Symbol: `BTC\\_USDT`  TF: `{timeframe}`\n"
				f"Capital: `{_fmt(session.equity, 2)}` USD  "
				f"Target: `{_fmt(session._volume_target, 0)}` USD"
			)

			while not stop_event.is_set():
				if datetime.now() >= deadline:
					LOGGER.info("Duration elapsed; stopping.")
					break
				if session.halted:
					LOGGER.info("Session halted (%s); stopping poll loop.", session.halt_reason)
					break
				try:
					fresh = await _fetch_candles(client, symbol, interval)
					fresh = _drop_forming(fresh, interval, int(datetime.now(tz=timezone_utc()).timestamp() * 1000))
					if not fresh.empty and last_ts is not None:
						new_rows = fresh[fresh["open_time"] > last_ts]
						if not new_rows.empty:
							for _, row in new_rows.iterrows():
								history = pd.concat([history, row.to_frame().T], ignore_index=True)
								history = history.tail(200).reset_index(drop=True)
								session.on_new_candle(history)
								last_ts = row["open_time"]
								session.save_state(state_path)
				except Exception as exc:  # noqa: BLE001
					LOGGER.exception("poll iteration failed: %s", exc)
				try:
					await asyncio.wait_for(stop_event.wait(), timeout=args.poll_seconds)
					break
				except asyncio.TimeoutError:
					continue
	finally:
		s = session.summary()
		def esc(t: str) -> str:
			for ch in "_*[]()~`>#+-=|{}.!":
				t = t.replace(ch, f"\\{ch}")
			return t
		try:
			await notifier.send_raw(
				"🛑 *Volume farmer stopped*\n"
				f"Volume: `{esc(_fmt(s['volume_usd']))}` USD  "
				f"\\({esc(str(s['volume_target_pct']))}%\\)\n"
				f"Round\\-trips: `{s['round_trips']}`  WR: `{esc(str(s['win_rate_pct']))}`%\n"
				f"Equity: `{esc(_fmt(s['equity'], 4))}`  "
				f"Fees gross: `{esc(_fmt(s['fees_gross'], 2))}`  "
				f"Net: `{esc(_fmt(s['fees_net'], 2))}`\n"
				f"PnL: `{esc(_fmt(s['total_pnl'], 2))}`  "
				f"Fee cover: `{esc(str(s['fee_cover_pct']))}`%"
			)
		except Exception:  # noqa: BLE001
			LOGGER.exception("final notify failed")
		stop_event.set()
		report_task.cancel()
		try:
			await report_task
		except (asyncio.CancelledError, Exception):
			pass
		await notifier.stop()
		session.save_state(state_path)
		LOGGER.info("Final summary: %s", s)


def timezone_utc():
	from datetime import timezone as _tz
	return _tz.utc


def _drop_forming(df: pd.DataFrame, interval: str, now_ms: int) -> pd.DataFrame:
	if df.empty:
		return df
	bar_minutes = {
		"Min1": 1, "Min5": 5, "Min15": 15, "Min30": 30,
		"Min60": 60, "Hour4": 240, "Day1": 1440,
	}.get(interval, 5)
	last = df.iloc[-1]
	last_open = pd.Timestamp(last["open_time"]).value // 1_000_000
	close_ms = last_open + bar_minutes * 60 * 1000
	if close_ms > now_ms:
		return df.iloc[:-1].reset_index(drop=True)
	return df


if __name__ == "__main__":
	asyncio.run(_run(_parse_args()))
