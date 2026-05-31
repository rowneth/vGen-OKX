"""Tests for paper broker fill simulation."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from execution.paper_broker import PaperBroker, PaperOrder


def test_paper_broker_fills_when_price_touched() -> None:
	broker = PaperBroker()
	order = PaperOrder(side="buy", price=100.0, qty=1.0)
	fill = broker.try_fill(order, candle_low=99.5, candle_high=101.0)
	assert fill.filled
	assert fill.fill_qty == 1.0


def test_paper_broker_no_fill_outside_range() -> None:
	broker = PaperBroker()
	order = PaperOrder(side="buy", price=100.0, qty=1.0)
	fill = broker.try_fill(order, candle_low=101.0, candle_high=102.0)
	assert not fill.filled
	assert fill.fill_qty == 0.0
