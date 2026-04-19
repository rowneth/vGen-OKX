"""Tests for PaperSession lifecycle (entry → TP1 → TP2 → exit)."""

from __future__ import annotations

import asyncio
import pathlib
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from execution.paper_session import PaperEvent, PaperSession
from strategy.base import Strategy, TradeSignal


class _StubStrategy(Strategy):
	"""Emits exactly one long signal on bar index 1, then never again."""

	def __init__(self) -> None:
		self._emitted = False

	def prepare(self, candles: pd.DataFrame) -> pd.DataFrame:
		return candles.copy()

	def generate_signal(self, candles: pd.DataFrame, index: int) -> Optional[TradeSignal]:
		if self._emitted or index < 1:
			return None
		self._emitted = True
		entry = float(candles.iloc[index]["close"])
		return TradeSignal(
			side="long",
			reference_index=index - 1,
			entry_index=index,
			entry_price=entry,
			stop_price=entry - 100.0,
			take_profit_price=entry + 300.0,
			tp1_price=entry + 100.0,
			tp1_size_fraction=0.4,
			tp2_price=entry + 200.0,
			tp2_size_fraction=0.3,
			move_stop_to_breakeven_after_tp1=True,
		)


def _bar(ts: datetime, open_: float, high: float, low: float, close: float) -> dict:
	return {
		"open_time": ts,
		"close_time": ts + timedelta(minutes=15),
		"open": open_, "high": high, "low": low, "close": close,
		"volume": 100.0, "turnover": 0.0,
	}


def _config() -> dict:
	return {
		"exchange": {"symbol": "BTC_USDT", "tick_size": 0.1},
		"fees": {"maker": 0.0001, "taker": 0.0005},
		"risk": {"risk_per_trade_pct": 0.02, "max_leverage": 5},
		"strategy": {"exits": {"time_stop_candles": 50}},
		"backtest": {"initial_equity": 10_000.0},
	}


def _run(coro_factory):
	return asyncio.run(coro_factory())


def test_full_tp_lifecycle_emits_all_events() -> None:
	"""Strategy fires at bar 1, TP1 and TP2 hit on later bars, final TP closes."""

	events: List[PaperEvent] = []

	async def handler(ev: PaperEvent) -> None:
		events.append(ev)

	session = PaperSession(_config(), _StubStrategy(), event_callback=handler)

	base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
	bars = [
		_bar(base, 50_000, 50_050, 49_950, 50_000),
		_bar(base + timedelta(minutes=15), 50_000, 50_020, 49_980, 50_000),  # signal bar
		_bar(base + timedelta(minutes=30), 50_000, 50_150, 50_000, 50_120),  # TP1 hit
		_bar(base + timedelta(minutes=45), 50_120, 50_250, 50_100, 50_220),  # TP2 hit
		_bar(base + timedelta(minutes=60), 50_220, 50_350, 50_200, 50_320),  # Final TP
	]

	async def run() -> None:
		for i in range(2, len(bars) + 1):
			hist = pd.DataFrame(bars[:i])
			await session.on_new_candle(hist)

	_run(lambda: run())

	kinds = [e.kind for e in events]
	assert "signal" in kinds
	assert "entry" in kinds
	assert "tp1" in kinds
	assert "tp2" in kinds
	assert "exit" in kinds

	exit_event = [e for e in events if e.kind == "exit"][0]
	assert exit_event.payload["reason"] == "take_profit"
	assert exit_event.payload["net_pnl"] > 0
	assert len(session.ledger) == 1
	assert session.ledger[0].tp1_hit and session.ledger[0].tp2_hit


def test_stop_loss_closes_trade_with_loss() -> None:
	events: List[PaperEvent] = []

	async def handler(ev: PaperEvent) -> None:
		events.append(ev)

	session = PaperSession(_config(), _StubStrategy(), event_callback=handler)
	base = datetime(2024, 2, 1, tzinfo=timezone.utc)
	bars = [
		_bar(base, 50_000, 50_050, 49_950, 50_000),
		_bar(base + timedelta(minutes=15), 50_000, 50_010, 49_990, 50_000),  # signal
		_bar(base + timedelta(minutes=30), 50_000, 50_010, 49_800, 49_850),  # stop
	]

	async def run() -> None:
		for i in range(2, len(bars) + 1):
			await session.on_new_candle(pd.DataFrame(bars[:i]))

	_run(lambda: run())

	exits = [e for e in events if e.kind == "exit"]
	assert len(exits) == 1
	assert exits[0].payload["reason"] == "stop_loss"
	assert session.equity < 10_000.0


def test_daily_summary_aggregates_ledger() -> None:
	session = PaperSession(_config(), _StubStrategy())
	# Manufacture ledger entries directly.
	from execution.paper_session import PaperTradeLedgerEntry

	session.ledger.append(PaperTradeLedgerEntry(
		entry_time="2024-03-01T00:00:00+00:00",
		exit_time="2024-03-01T01:00:00+00:00",
		symbol="BTC_USDT", side="long",
		qty=0.1, initial_qty=0.1,
		entry_price=50_000.0, exit_price=50_300.0,
		reason="take_profit",
		gross_pnl=30.0, partial_pnl=0.0,
		entry_fee=0.5, exit_fee=0.5, net_pnl=29.0,
		equity_after=10_029.0, bars_held=4,
		tp1_hit=False, tp2_hit=False,
		notional=10_030.0,
	))
	session.ledger.append(PaperTradeLedgerEntry(
		entry_time="2024-03-01T02:00:00+00:00",
		exit_time="2024-03-01T03:00:00+00:00",
		symbol="BTC_USDT", side="short",
		qty=0.1, initial_qty=0.1,
		entry_price=50_300.0, exit_price=50_400.0,
		reason="stop_loss",
		gross_pnl=-10.0, partial_pnl=0.0,
		entry_fee=0.5, exit_fee=0.5, net_pnl=-11.0,
		equity_after=10_018.0, bars_held=4,
		tp1_hit=False, tp2_hit=False,
		notional=10_070.0,
	))
	session.equity = 10_018.0

	stats = session.daily_summary("2024-03-01")
	assert stats["trades"] == 2
	assert stats["wins"] == 1
	assert stats["losses"] == 1
	assert round(stats["win_rate"], 2) == 0.5
	assert round(stats["net_pnl"], 2) == 18.0
