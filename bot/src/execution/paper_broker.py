"""Paper broker for simulated order execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperOrder:
	"""Represents a paper order request."""

	side: str
	price: float
	qty: float
	post_only: bool = True


@dataclass(frozen=True)
class PaperFill:
	"""Represents a simulated fill result."""

	filled: bool
	fill_price: float
	fill_qty: float


class PaperBroker:
	"""Simple paper broker fill simulator based on candle bounds."""

	def try_fill(self, order: PaperOrder, candle_low: float, candle_high: float) -> PaperFill:
		"""Attempt filling an order against candle range.

		Args:
			order: Order request.
			candle_low: Candle low.
			candle_high: Candle high.

		Returns:
			PaperFill describing simulated outcome.
		"""
		touched = candle_low <= order.price <= candle_high
		if not touched:
			return PaperFill(filled=False, fill_price=order.price, fill_qty=0.0)
		return PaperFill(filled=True, fill_price=order.price, fill_qty=order.qty)
