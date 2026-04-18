"""Typed exchange-side models for normalized MEXC data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class Candle:
	"""Represents one OHLCV candle."""

	open_time: datetime
	close_time: datetime
	open: float
	high: float
	low: float
	close: float
	volume: float


@dataclass(frozen=True)
class Position:
	"""Represents a normalized futures position snapshot."""

	symbol: str
	side: str
	size: float
	entry_price: float
	mark_price: float
	unrealized_pnl: float
	leverage: int


@dataclass(frozen=True)
class Order:
	"""Represents an exchange order view."""

	order_id: str
	symbol: str
	side: str
	order_type: str
	quantity: float
	price: float
	status: str
	filled_quantity: float
	average_fill_price: Optional[float] = None


@dataclass(frozen=True)
class AccountInfo:
	"""Represents a simplified account balance and margin state."""

	equity: float
	available_balance: float
	margin_used: float
