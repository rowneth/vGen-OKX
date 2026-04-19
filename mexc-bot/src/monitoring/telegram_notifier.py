"""Structured Telegram notifier for paper & live trading events.

Provides a single ``TelegramNotifier`` with one method per lifecycle event
(signal / entry / TP1 / TP2 / stop / exit / daily / error / startup) so the
rest of the codebase never builds message strings by hand.

Messages use Telegram ``MarkdownV2``; the module escapes user-supplied values
defensively. If no bot token or chat id is configured, all methods become
no-ops (so tests and dev runs can call them freely).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

import aiohttp
from zoneinfo import ZoneInfo

LOGGER = logging.getLogger(__name__)

_UTC = ZoneInfo("UTC")


_MD_V2_ESCAPE = re.compile(r"([_\*\[\]\(\)~`>#\+\-=\|{}\.!\\])")


def _md(value: Any) -> str:
	"""Escape a value for Telegram MarkdownV2 safely."""

	text = str(value) if value is not None else "-"
	return _MD_V2_ESCAPE.sub(r"\\\1", text)


def _fmt_usd(amount: float) -> str:
	sign = "-" if amount < 0 else ""
	return f"{sign}${abs(amount):,.2f}"


def _fmt_signed(amount: float) -> str:
	return f"{amount:+,.2f}"


def _fmt_pct(value: float) -> str:
	return f"{value * 100:+.2f}%"


@dataclass(frozen=True)
class TradeSnapshot:
	"""Lightweight view of a live/paper trade for notifications."""

	side: str
	symbol: str
	entry_price: float
	stop_price: float
	tp1_price: Optional[float]
	tp2_price: Optional[float]
	final_tp_price: float
	qty: float
	equity: float
	reason: Optional[str] = None


class TelegramNotifier:
	"""Async Telegram Bot API client scoped to a single chat.

	The notifier does not block the caller: every message is enqueued onto
	an internal ``asyncio.Queue`` and dispatched by a background worker
	started via :meth:`start`. Call :meth:`stop` during shutdown.
	"""

	def __init__(
		self,
		bot_token: Optional[str] = None,
		chat_id: Optional[str] = None,
		*,
		enabled: Optional[bool] = None,
		session: Optional[aiohttp.ClientSession] = None,
		request_timeout_seconds: float = 10.0,
	) -> None:
		"""Initialise notifier.

		Args:
			bot_token: Telegram bot token. Falls back to ``TELEGRAM_BOT_TOKEN`` env var.
			chat_id: Target chat id. Falls back to ``TELEGRAM_CHAT_ID`` env var.
			enabled: Force enable/disable. If None, enabled iff both token and chat id are present.
			session: Optional injected aiohttp session (for testing).
			request_timeout_seconds: Per-request timeout.
		"""
		self._bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or None
		self._chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip() or None
		if enabled is None:
			self._enabled = bool(self._bot_token and self._chat_id)
		else:
			self._enabled = bool(enabled)
		tz_name = os.getenv("BOT_TIMEZONE", "Asia/Colombo")
		try:
			self._tz = ZoneInfo(tz_name)
		except Exception:
			self._tz = _UTC
		self._session = session
		self._owns_session = session is None
		self._timeout = aiohttp.ClientTimeout(total=request_timeout_seconds)
		self._queue: Optional["asyncio.Queue[str]"] = None
		self._worker: Optional[asyncio.Task[None]] = None
		self._stopped = False

	# ------------------------------------------------------------------
	# Lifecycle
	# ------------------------------------------------------------------
	async def start(self) -> None:
		"""Start the background dispatcher worker."""

		if not self._enabled or self._worker is not None:
			return
		if self._queue is None:
			self._queue = asyncio.Queue()
		if self._session is None:
			self._session = aiohttp.ClientSession(timeout=self._timeout)
			self._owns_session = True
		self._worker = asyncio.create_task(self._run_worker(), name="telegram-notifier")

	async def stop(self) -> None:
		"""Flush the queue and stop the worker."""

		self._stopped = True
		if self._worker is not None and self._queue is not None:
			# Push a sentinel so the worker exits its get().
			await self._queue.put("")
			try:
				await asyncio.wait_for(self._worker, timeout=5.0)
			except (asyncio.TimeoutError, asyncio.CancelledError):
				self._worker.cancel()
			self._worker = None
		if self._owns_session and self._session is not None:
			await self._session.close()
			self._session = None

	@property
	def enabled(self) -> bool:
		"""Whether messages will actually be sent."""

		return self._enabled

	# ------------------------------------------------------------------
	# Low-level send
	# ------------------------------------------------------------------
	async def send_raw(self, markdown_v2_text: str) -> None:
		"""Queue a pre-formatted MarkdownV2 message."""

		if not self._enabled or self._stopped:
			LOGGER.info("[telegram-disabled]\n%s", markdown_v2_text)
			return
		if self._worker is None:
			# Lazy-start: let ``start()`` be optional for simple scripts.
			await self.start()
		assert self._queue is not None
		await self._queue.put(markdown_v2_text)

	async def _run_worker(self) -> None:
		assert self._session is not None and self._queue is not None
		url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
		while True:
			text = await self._queue.get()
			if text == "" and self._stopped:
				return
			if not text:
				continue
			payload = {
				"chat_id": self._chat_id,
				"text": text,
				"parse_mode": "MarkdownV2",
				"disable_web_page_preview": True,
			}
			try:
				async with self._session.post(url, json=payload) as response:
					if response.status != 200:
						body = await response.text()
						LOGGER.error(
							"Telegram send failed status=%s body=%s",
							response.status,
							body[:500],
						)
			except Exception as exc:  # noqa: BLE001 - network is noisy; never crash caller
				LOGGER.error("Telegram send exception: %s", exc)

	# ------------------------------------------------------------------
	# High-level lifecycle events
	# ------------------------------------------------------------------
	async def notify_startup(
		self,
		*,
		mode: str,
		symbol: str,
		timeframe: str,
		equity: float,
		strategy: str,
	) -> None:
		"""Send the 'bot started' banner."""

		lines = [
			"🚀 *Bot started*",
			f"Mode: `{_md(mode)}`",
			f"Symbol: `{_md(symbol)}`  Timeframe: `{_md(timeframe)}`",
			f"Strategy: `{_md(strategy)}`",
			f"Equity: `{_md(_fmt_usd(equity))}`",
			f"Time: `{_md(_now_iso(self._tz))}`",
		]
		await self.send_raw("\n".join(lines))

	async def notify_shutdown(self, *, reason: str) -> None:
		"""Send the 'bot stopping' banner."""

		await self.send_raw(
			"🛑 *Bot stopping*\n"
			f"Reason: `{_md(reason)}`\n"
			f"Time: `{_md(_now_iso(self._tz))}`"
		)

	async def notify_signal(self, trade: TradeSnapshot) -> None:
		"""Announce a fresh signal (before entry fill)."""

		await self.send_raw(self._format_trade(trade, header="🎯 *Signal*"))

	async def notify_entry_filled(self, trade: TradeSnapshot, *, fill_price: float) -> None:
		"""Announce a successful entry fill."""

		body = self._format_trade(trade, header="✅ *Entry filled*")
		body += f"\nFill: `{_md(f'{fill_price:,.2f}')}`"
		await self.send_raw(body)

	async def notify_tp1_hit(
		self,
		*,
		symbol: str,
		side: str,
		price: float,
		partial_qty: float,
		partial_pnl: float,
		new_stop: float,
	) -> None:
		"""Announce TP1 partial scale-out."""

		lines = [
			"💰 *TP1 hit \\(partial\\)*",
			f"`{_md(symbol)}` `{_md(side.upper())}`",
			f"Price: `{_md(f'{price:,.2f}')}`  Qty: `{_md(f'{partial_qty:.6f}')}`",
			f"Partial PnL: `{_md(_fmt_signed(partial_pnl))}`",
			f"Stop moved to BE: `{_md(f'{new_stop:,.2f}')}`",
		]
		await self.send_raw("\n".join(lines))

	async def notify_tp2_hit(
		self,
		*,
		symbol: str,
		side: str,
		price: float,
		partial_qty: float,
		partial_pnl: float,
		new_stop: float,
	) -> None:
		"""Announce TP2 partial scale-out."""

		lines = [
			"💎 *TP2 hit \\(partial\\)*",
			f"`{_md(symbol)}` `{_md(side.upper())}`",
			f"Price: `{_md(f'{price:,.2f}')}`  Qty: `{_md(f'{partial_qty:.6f}')}`",
			f"Partial PnL: `{_md(_fmt_signed(partial_pnl))}`",
			f"Stop trailed to: `{_md(f'{new_stop:,.2f}')}`",
		]
		await self.send_raw("\n".join(lines))

	async def notify_exit(
		self,
		*,
		symbol: str,
		side: str,
		reason: str,
		exit_price: float,
		net_pnl: float,
		total_pnl_pct: float,
		equity_after: float,
		bars_held: int,
	) -> None:
		"""Announce a final exit (stop, final TP, or time-stop)."""

		emoji = {
			"take_profit": "🏁",
			"stop_loss": "🛑",
			"time_stop": "⏱️",
		}.get(reason, "📤")
		lines = [
			f"{emoji} *Position closed*",
			f"`{_md(symbol)}` `{_md(side.upper())}`  Reason: `{_md(reason)}`",
			f"Exit: `{_md(f'{exit_price:,.2f}')}`  Bars: `{_md(bars_held)}`",
			f"Net PnL: `{_md(_fmt_signed(net_pnl))}`  \\({_md(_fmt_pct(total_pnl_pct))}\\)",
			f"Equity: `{_md(_fmt_usd(equity_after))}`",
		]
		await self.send_raw("\n".join(lines))

	async def notify_signal_rejected(self, *, reason: str, details: str = "") -> None:
		"""Announce why a candidate setup was skipped (optional diagnostic)."""

		body = f"🚫 *Signal rejected*\nReason: `{_md(reason)}`"
		if details:
			body += f"\n{_md(details)}"
		await self.send_raw(body)

	async def notify_error(self, *, title: str, detail: str) -> None:
		"""Send an error banner."""

		await self.send_raw(
			f"❗ *{_md(title)}*\n```\n{_md(detail)}\n```"
		)

	async def notify_daily_report(
		self,
		*,
		date_utc: str,
		trades: int,
		wins: int,
		losses: int,
		win_rate: float,
		gross_pnl: float,
		fees: float,
		net_pnl: float,
		equity_start: float,
		equity_end: float,
		volume: float,
		extra_lines: Optional[Iterable[str]] = None,
	) -> None:
		"""Send the daily digest."""

		day_return = ((equity_end / equity_start) - 1.0) if equity_start > 0 else 0.0
		lines = [
			f"📊 *Daily report* `{_md(date_utc)}`",
			"─────────────",
			f"Trades: `{_md(trades)}`  Wins: `{_md(wins)}`  Losses: `{_md(losses)}`",
			f"Win rate: `{_md(f'{win_rate * 100:.1f}%')}`",
			f"Volume: `{_md(_fmt_usd(volume))}`",
			f"Gross PnL: `{_md(_fmt_signed(gross_pnl))}`",
			f"Fees: `{_md(_fmt_usd(fees))}`",
			f"*Net PnL*: `{_md(_fmt_signed(net_pnl))}`",
			f"Equity: `{_md(_fmt_usd(equity_start))}` → `{_md(_fmt_usd(equity_end))}` "
			f"\\({_md(_fmt_pct(day_return))}\\)",
		]
		if extra_lines:
			lines.append("")
			lines.extend(_md(line) for line in extra_lines)
		await self.send_raw("\n".join(lines))

	# ------------------------------------------------------------------
	# Helpers
	# ------------------------------------------------------------------
	def _format_trade(self, trade: TradeSnapshot, *, header: str) -> str:
		tp1 = f"{trade.tp1_price:,.2f}" if trade.tp1_price is not None else "-"
		tp2 = f"{trade.tp2_price:,.2f}" if trade.tp2_price is not None else "-"
		reason = trade.reason or "signal"
		lines = [
			header,
			f"`{_md(trade.symbol)}` `{_md(trade.side.upper())}`  "
			f"`{_md(reason)}`",
			f"Entry: `{_md(f'{trade.entry_price:,.2f}')}`",
			f"Stop: `{_md(f'{trade.stop_price:,.2f}')}`",
			f"TP1: `{_md(tp1)}`  TP2: `{_md(tp2)}`  "
			f"Final: `{_md(f'{trade.final_tp_price:,.2f}')}`",
			f"Qty: `{_md(f'{trade.qty:.6f}')}`  "
			f"Notional: `{_md(_fmt_usd(trade.qty * trade.entry_price))}`",
			f"Equity: `{_md(_fmt_usd(trade.equity))}`",
		]
		return "\n".join(lines)


def _now_iso(tz: Optional[Any] = None) -> str:
	use_tz = tz or ZoneInfo("Asia/Colombo")
	return datetime.now(tz=use_tz).strftime("%Y-%m-%d %H:%M:%S %Z")


def build_notifier_from_env() -> TelegramNotifier:
	"""Construct a notifier from environment variables (common entry point)."""

	return TelegramNotifier()


# Re-export helpers for tests.
__all__ = [
	"TelegramNotifier",
	"TradeSnapshot",
	"build_notifier_from_env",
	"_md",
]
