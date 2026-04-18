"""Base strategy abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class TradeSignal:
	"""Represents an entry signal emitted by a strategy."""

	side: str
	reference_index: int
	entry_index: int
	entry_price: float
	stop_price: float
	take_profit_price: float


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
