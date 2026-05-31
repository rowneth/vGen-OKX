"""Base strategy abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class TradeSignal:
	"""Represents an entry signal emitted by a strategy.

	Attributes:
		side: ``long`` or ``short``.
		reference_index: Trigger candle index.
		entry_index: Confirmation candle index (order placed after close).
		entry_price: Target entry (post-only limit).
		stop_price: Hard protective stop price.
		take_profit_price: Final (full-exit) take-profit price.
		tp1_price: Optional partial take-profit price (scale-out).
		tp1_size_fraction: Fraction of qty to exit at TP1 (0..1).
		tp2_price: Optional second partial take-profit price.
		tp2_size_fraction: Fraction of qty to exit at TP2 (0..1).
		move_stop_to_breakeven_after_tp1: If True, runner stop becomes entry.
	"""

	side: str
	reference_index: int
	entry_index: int
	entry_price: float
	stop_price: float
	take_profit_price: float
	tp1_price: Optional[float] = None
	tp1_size_fraction: float = 0.0
	tp2_price: Optional[float] = None
	tp2_size_fraction: float = 0.0
	move_stop_to_breakeven_after_tp1: bool = False


class Strategy(ABC):
	"""Abstract strategy interface."""

	@abstractmethod
	def prepare(self, candles: pd.DataFrame) -> pd.DataFrame:
		"""Prepare candle data with required derived features.

		Args:
			candles: Input OHLCV dataframe.

		Returns:
			Dataframe with additional strategy columns.
		"""

	@abstractmethod
	def generate_signal(self, candles: pd.DataFrame, index: int) -> Optional[TradeSignal]:
		"""Return trade signal for the specified index.

		Args:
			candles: Feature-enriched dataframe.
			index: Current candle index in the dataframe.

		Returns:
			TradeSignal if entry conditions are met, else None.
		"""
