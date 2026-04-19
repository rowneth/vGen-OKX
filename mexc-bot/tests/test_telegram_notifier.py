"""Tests for the Telegram notifier: formatting and dispatch."""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
from typing import Any, Dict, List

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from monitoring.telegram_notifier import TelegramNotifier, TradeSnapshot, _md


class _FakeResponse:
	def __init__(self, status: int = 200, body: str = "{}") -> None:
		self.status = status
		self._body = body

	async def __aenter__(self) -> "_FakeResponse":
		return self

	async def __aexit__(self, *_: Any) -> None:
		return None

	async def text(self) -> str:
		return self._body


class _FakeSession:
	def __init__(self) -> None:
		self.requests: List[Dict[str, Any]] = []

	def post(self, url: str, json: Dict[str, Any]) -> _FakeResponse:  # noqa: A002
		self.requests.append({"url": url, "json": json})
		return _FakeResponse()

	async def close(self) -> None:
		return None


def _run(coro_factory) -> Any:
	async def runner() -> Any:
		return await coro_factory()
	return asyncio.run(runner())


def test_md_escapes_special_characters() -> None:
	assert _md("a.b-c") == "a\\.b\\-c"
	assert _md("price=$100") == "price\\=$100"


def test_disabled_notifier_is_noop() -> None:
	n = TelegramNotifier(bot_token=None, chat_id=None)
	assert not n.enabled

	async def go() -> None:
		await n.send_raw("hello")
		await n.stop()

	_run(lambda: go())


def test_notifier_sends_startup_message() -> None:
	captured: Dict[str, Any] = {}

	async def go() -> None:
		fake = _FakeSession()
		n = TelegramNotifier(
			bot_token="T",
			chat_id="C",
			enabled=True,
			session=fake,  # type: ignore[arg-type]
		)
		await n.start()
		await n.notify_startup(
			mode="paper", symbol="BTC_USDT", timeframe="15m",
			equity=10_000.0, strategy="bollinger",
		)
		await n.stop()
		captured["requests"] = fake.requests

	_run(lambda: go())

	requests = captured["requests"]
	assert len(requests) == 1
	body = requests[0]["json"]
	assert body["chat_id"] == "C"
	assert body["parse_mode"] == "MarkdownV2"
	assert "Bot started" in body["text"]
	assert "BTC\\_USDT" in body["text"]


def test_notifier_sends_lifecycle_events() -> None:
	captured: Dict[str, Any] = {}

	async def go() -> None:
		fake = _FakeSession()
		n = TelegramNotifier(
			bot_token="T", chat_id="C", enabled=True,
			session=fake,  # type: ignore[arg-type]
		)
		await n.start()

		trade = TradeSnapshot(
			side="long", symbol="BTC_USDT",
			entry_price=50_000.0, stop_price=49_500.0,
			tp1_price=50_200.0, tp2_price=50_400.0,
			final_tp_price=50_800.0, qty=0.1, equity=10_000.0,
		)
		await n.notify_signal(trade)
		await n.notify_entry_filled(trade, fill_price=50_000.0)
		await n.notify_tp1_hit(
			symbol="BTC_USDT", side="long", price=50_200.0,
			partial_qty=0.04, partial_pnl=8.0, new_stop=50_000.0,
		)
		await n.notify_tp2_hit(
			symbol="BTC_USDT", side="long", price=50_400.0,
			partial_qty=0.03, partial_pnl=12.0, new_stop=50_200.0,
		)
		await n.notify_exit(
			symbol="BTC_USDT", side="long", reason="take_profit",
			exit_price=50_800.0, net_pnl=35.0, total_pnl_pct=0.0035,
			equity_after=10_035.0, bars_held=6,
		)
		await n.notify_daily_report(
			date_utc="2024-01-01", trades=3, wins=2, losses=1, win_rate=0.6667,
			gross_pnl=50.0, fees=1.2, net_pnl=48.8,
			equity_start=10_000.0, equity_end=10_048.8, volume=15_000.0,
		)
		await n.stop()
		captured["requests"] = fake.requests

	_run(lambda: go())

	texts = [r["json"]["text"] for r in captured["requests"]]
	assert any("Signal" in t for t in texts)
	assert any("Entry filled" in t for t in texts)
	assert any("TP1" in t for t in texts)
	assert any("TP2" in t for t in texts)
	assert any("Position closed" in t for t in texts)
	assert any("Daily report" in t for t in texts)
