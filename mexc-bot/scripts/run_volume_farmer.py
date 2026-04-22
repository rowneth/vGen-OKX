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
from monitoring.chart_generator import build_entry_chart  # noqa: E402

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
	p.add_argument("--label", type=str, default="")
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
		"--max-live-notional", type=float, default=5000.0,
		help="Per-trade notional cap (USD) in --live mode. "
			 "The session's dynamic sizing is used; this is just a safety ceiling.",
	)
	p.add_argument(
		"--live-leverage", type=int, default=125,
		help="Max leverage cap for real orders in --live mode. "
			 "Actual leverage comes from the session's dynamic calculator "
			 "(ebirth.net formula) and is capped at this value.",
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
	session: "VolumeFarmerSession",
	live_executor: Optional[LiveVolumeExecutor] = None,
	live_mode: bool = False,
):
	loop = asyncio.get_event_loop()
	# Mutable container so the inner handler can update it across calls
	_state: dict = {"entry_msg_id": None, "real_exit_sent": False}

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
			# In LIVE mode the paper session's entry is a simulation only -- the
			# real order may be rejected, repriced many times, or filled at a
			# very different price. All Telegram must reflect confirmed MEXC
			# state. The real-time fill callback (real_entry_callback) is the
			# only code path that may announce an opened trade.
			if live_mode:
				return
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
			async def _send_entry(m: str) -> None:
				_state["entry_msg_id"] = await notifier.send_and_get_id(m)
				_state["real_exit_sent"] = False
			asyncio.run_coroutine_threadsafe(_send_entry(msg), loop)

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
			reply_id = _state.get("entry_msg_id")
			# In LIVE mode the bar-driven exit is unreliable — the paper session's
			# synthetic entry price drifts from the real MEXC fill, so the 5m bar
			# can "hit TP" while the exchange position is still open. Only the
			# real-time position watcher (real_close_callback) may send exit
			# Telegrams when live. Keep entry_msg_id so the real callback can
			# still reply to the entry message when MEXC actually closes.
			if live_mode:
				return
			_state["entry_msg_id"] = None
			# If the real-time position watcher already sent an exit message,
			# skip the bar-driven one to avoid duplicates.
			if _state.get("real_exit_sent"):
				_state["real_exit_sent"] = False
				return
			async def _send_exit(m: str, rid: Optional[int]) -> None:
				await notifier.send_and_get_id(m, reply_to_message_id=rid)
			asyncio.run_coroutine_threadsafe(_send_exit(msg, reply_id), loop)

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

	async def real_entry_callback(info: dict) -> None:
		"""Called by LiveVolumeExecutor the moment MEXC confirms a fill.

		This is the ONLY place that may announce a new live trade to
		Telegram. It uses confirmed MEXC values (real fill price, real
		maker fee, real TP/SL attached to the order) plus current session
		stats (capital/volume/record) for the context lines.
		"""
		side = str(info.get("side", "long"))
		side_emoji = "🟢" if side == "long" else "🔴"
		entry = _as_float_safe(info.get("entry_price"))
		tp_price = _as_float_safe(info.get("tp_price"))
		sl_price = _as_float_safe(info.get("sl_price"))
		tp_bps = _as_float_safe(info.get("tp_bps"))
		sl_bps = _as_float_safe(info.get("sl_bps"))
		leverage = int(round(_as_float_safe(info.get("leverage"))))
		notional = _as_float_safe(info.get("notional"))
		margin = _as_float_safe(info.get("margin"))
		open_fee = _as_float_safe(info.get("open_fee"))
		reprice_attempts = int(info.get("reprice_attempts") or 0)
		# Session-derived context
		trade_num = session.wins + session.losses + 1
		reprice_tag = f" \\(repriced {reprice_attempts}x\\)" if reprice_attempts > 0 else ""
		msg = (
			f"{side_emoji} *{side.upper()} \\#{trade_num}* · {leverage}x{reprice_tag}\n"
			f"`{symbol.replace('_', '/')}`\n"
			f"{DIV}\n"
			f"Entry   `{_n(entry, 1)}`\n"
			f"TP  →   `{_n(tp_price, 1)}`  \+{_escape(str(tp_bps))}bps\n"
			f"SL  →   `{_n(sl_price, 1)}`  \-{_escape(str(sl_bps))}bps\n"
			f"{DIV}\n"
			f"Size    {_money(notional)}\n"
			f"Margin  {_money(margin, 2)}\n"
			f"Fee     {_money(open_fee, 4)}\n"
			f"{DIV}\n"
			f"Balance {_money(session.equity, 2)}"
		)
		try:
			msg_id = await notifier.send_and_get_id(msg)
			_state["entry_msg_id"] = msg_id
			_state["real_exit_sent"] = False
		except Exception as exc:  # noqa: BLE001
			LOGGER.exception("real entry send failed: %s", exc)

		# Fire chart asynchronously — send as a reply to the entry message
		async def _send_chart() -> None:
			if live_executor is None or live_executor.client is None:
				return
			try:
				png = await build_entry_chart(
					client=live_executor.client,
					symbol=symbol,
					side=side,
					entry_price=entry,
					tp_price=tp_price,
					sl_price=sl_price,
				)
				if png:
					await notifier.send_photo(
						png,
						caption=None,
						reply_to_message_id=_state.get("entry_msg_id"),
					)
			except Exception as exc:  # noqa: BLE001
				LOGGER.warning("chart send failed (non-fatal): %s", exc)

		asyncio.ensure_future(_send_chart())

	async def real_close_callback(info: dict) -> None:
		"""Called by LiveVolumeExecutor the moment MEXC closes our position.

		Sends an immediate exit Telegram as a reply to the entry message
		and marks ``real_exit_sent`` so the later bar-driven exit event is
		suppressed (no duplicate message).
		"""
		reason = info.get("reason", "unknown")
		if reason == "tp":
			result_emoji, result_label = "✅", "TP"
		elif reason == "sl":
			result_emoji, result_label = "❌", "SL"
		elif reason == "trend_break":
			result_emoji, result_label = "✂️", "Exit"
		else:
			result_emoji, result_label = "⏹", reason.replace("_", " ").title()
		side = str(info.get("side", "")).upper()
		trade_num_close = session.wins + session.losses + 1
		entry = _as_float_safe(info.get("entry_price"))
		exit_ = _as_float_safe(info.get("exit_price"))
		gross = _as_float_safe(info.get("gross_pnl"))
		open_fee = _as_float_safe(info.get("open_fee"))
		close_fee = _as_float_safe(info.get("close_fee"))
		net = _as_float_safe(info.get("net_pnl"))
		# Current session stats — add this trade's outcome (paper session
		# increments wins/losses only on next bar, so we apply it here for display).
		is_win = net > 0
		wins_now = int(session.wins) + (1 if is_win else 0)
		losses_now = int(session.losses) + (0 if is_win else 1)
		total_now = wins_now + losses_now
		wr_now = (wins_now / max(total_now, 1)) * 100
		volume_now = float(session.total_volume_usd)
		vol_target_now = float(session._volume_target)  # noqa: SLF001
		vol_pct_now = volume_now / max(vol_target_now, 1.0) * 100
		# Health badge — only meaningful after 10 trades
		if total_now < 10:
			health = ""
		elif wr_now >= 87.0:
			health = "🟢 On Track"
		elif wr_now >= 78.0:
			health = "🟡 Watch"
		else:
			health = "🔴 ⚠ Warning"
		health_line = f"\n{health}" if health else ""
		msg = (
			f"{result_emoji} *{result_label} · {side} \\#{trade_num_close}*\n"
			f"`{symbol.replace('_', '/')}`\n"
			f"`{_n(entry, 1)}` → `{_n(exit_, 1)}`\n"
			f"{DIV}\n"
			f"Gross   {_signed_money(gross)}\n"
			f"Fees    {_signed_money(-(open_fee + close_fee))}\n"
			f"*Net     {_signed_money(net)}*\n"
			f"{DIV}\n"
			f"{wins_now}W · {losses_now}L · *{_n(wr_now, 1)}% WR*{health_line}\n"
			f"{DIV}\n"
			f"Balance {_money(session.equity, 2)}\n"
			f"Vol {_money(volume_now, 0)} / {_money(vol_target_now, 0)}\n"
			f"{_n(vol_pct_now, 1)}% of 30d goal"
		)
		reply_id = _state.get("entry_msg_id")
		_state["real_exit_sent"] = True
		_state["entry_msg_id"] = None
		try:
			await notifier.send_and_get_id(msg, reply_to_message_id=reply_id)
		except Exception as exc:  # noqa: BLE001
			LOGGER.exception("real exit send failed: %s", exc)

	return handler, real_close_callback, real_entry_callback


def _as_float_safe(x: Any, default: float = 0.0) -> float:
	try:
		return float(x)
	except (TypeError, ValueError):
		return default


def _extract_usdt_equity(payload: Any) -> Optional[float]:
	"""Return USDT equity from MEXC /account/assets payload, or None."""

	def _walk(node: Any):
		if isinstance(node, dict):
			cur = str(node.get("currency") or node.get("coin") or "").upper()
			if cur == "USDT":
				yield node
			for v in node.values():
				yield from _walk(v)
		elif isinstance(node, list):
			for v in node:
				yield from _walk(v)

	for row in _walk(payload):
		eq = _as_float_safe(row.get("equity", row.get("balance", 0)))
		if eq > 0:
			return eq
	return None


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
		session=session,
		live_executor=None,  # will be rebound after client opens if --live
	)[0]

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

			# Fetch real USDT equity from MEXC and override session capital
			# so Telegram / sizing reflect the actual wallet (not the YAML default).
			# Skip the override on --resume to preserve session P&L tracking.
			if api_key and api_secret and not args.resume:
				try:
					assets_payload = await client.get_account_info()
					real_equity = _extract_usdt_equity(assets_payload)
					if real_equity is not None and real_equity > 0:
						LOGGER.info(
							"Overriding session capital from MEXC wallet: %.4f USDT (was %.4f)",
							real_equity, session.equity,
						)
						session.equity = real_equity
						session.start_equity = real_equity
						session.peak_equity = real_equity
					else:
						LOGGER.warning("Could not parse USDT equity from /account/assets; using config capital=%.2f", session.equity)
				except Exception as exc:  # noqa: BLE001
					LOGGER.warning("Failed to fetch MEXC account balance (%s); using config capital=%.2f", exc, session.equity)

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
				# Re-bind handler now that executor exists, and wire both
				# real-time callbacks so Telegram only fires on confirmed
				# MEXC state (entry fill / position close), never on paper
				# simulation.
				handler, real_close_cb, real_entry_cb = _build_event_handler(
					notifier,
					symbol,
					bool(notif_cfg.get("send_milestones", True)),
					session=session,
					live_executor=live_executor,
					live_mode=True,
				)
				session.event_callback = handler
				live_executor.real_entry_callback = real_entry_cb
				live_executor.real_close_callback = real_close_cb
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
