"""Order lifecycle manager for paper execution and future broker adapters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from execution.paper_broker import PaperBroker, PaperFill, PaperOrder

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderRequest:
	"""Represents one entry request submitted by strategy."""

	symbol: str
	side: str
	price: float
	qty: float
	submitted_at: datetime
	candle_index: int


@dataclass
class ManagedOrder:
	"""Represents mutable order state tracked by the order manager."""

	order_id: str
	request: OrderRequest
	status: str
	filled_qty: float = 0.0
	fill_price: float = 0.0
	closed_at_candle_index: Optional[int] = None
	reason: str = ""


@dataclass(frozen=True)
class FillEvent:
	"""Represents one fill emitted by the manager."""

	order_id: str
	symbol: str
	side: str
	qty: float
	price: float
	filled_at: datetime


class OrderManager:
	"""Manage order submission, fill checks, and stale-order cancellation."""

	def __init__(self, broker: PaperBroker, max_entry_age_candles: int = 1) -> None:
		"""Initialize order manager.

		Args:
			broker: Broker adapter used to simulate/execute order fills.
			max_entry_age_candles: Cancel unfilled entries after this candle age.
		"""
		self._broker = broker
		self._max_entry_age_candles = max_entry_age_candles
		self._active_orders: List[ManagedOrder] = []
		self._archived_orders: List[ManagedOrder] = []
		self._sequence = 0

	def submit_post_only_entry(self, request: OrderRequest) -> ManagedOrder:
		"""Submit a post-only entry order.

		Args:
			request: Order request payload.

		Returns:
			Tracked ManagedOrder.
		"""
		self._sequence += 1
		order = ManagedOrder(
			order_id=f"ORD-{self._sequence:06d}",
			request=request,
			status="open",
		)
		self._active_orders.append(order)
		LOGGER.info(
			"Submitted post-only entry order id=%s side=%s qty=%.6f price=%.2f",
			order.order_id,
			request.side,
			request.qty,
			request.price,
		)
		return order

	def process_candle(
		self,
		candle_index: int,
		candle_low: float,
		candle_high: float,
		now: datetime,
	) -> List[FillEvent]:
		"""Process active orders against a new candle.

		Args:
			candle_index: Current candle index.
			candle_low: Candle low price.
			candle_high: Candle high price.
			now: Current timestamp.

		Returns:
			List of fill events generated on this candle.
		"""
		fills: List[FillEvent] = []
		remaining: List[ManagedOrder] = []

		for order in self._active_orders:
			if self._is_stale(order=order, candle_index=candle_index):
				order.status = "canceled"
				order.closed_at_candle_index = candle_index
				order.reason = "stale_unfilled"
				self._archived_orders.append(order)
				LOGGER.info("Canceled stale entry order id=%s", order.order_id)
				continue

			fill = self._simulate_fill(order=order, candle_low=candle_low, candle_high=candle_high)
			if fill.filled:
				order.status = "filled"
				order.filled_qty = fill.fill_qty
				order.fill_price = fill.fill_price
				order.closed_at_candle_index = candle_index
				order.reason = "filled"
				self._archived_orders.append(order)
				fills.append(
					FillEvent(
						order_id=order.order_id,
						symbol=order.request.symbol,
						side=order.request.side,
						qty=fill.fill_qty,
						price=fill.fill_price,
						filled_at=now,
					)
				)
				continue

			remaining.append(order)

		self._active_orders = remaining
		return fills

	def cancel_all(self, reason: str, candle_index: int) -> None:
		"""Cancel all active orders.

		Args:
			reason: Cancellation reason.
			candle_index: Current candle index for audit.
		"""
		for order in self._active_orders:
			order.status = "canceled"
			order.reason = reason
			order.closed_at_candle_index = candle_index
			self._archived_orders.append(order)
		self._active_orders = []

	@property
	def active_orders(self) -> List[ManagedOrder]:
		"""Return active order list copy."""
		return list(self._active_orders)

	@property
	def archived_orders(self) -> List[ManagedOrder]:
		"""Return archived order list copy."""
		return list(self._archived_orders)

	def _is_stale(self, order: ManagedOrder, candle_index: int) -> bool:
		age = candle_index - order.request.candle_index
		return age > self._max_entry_age_candles

	def _simulate_fill(self, order: ManagedOrder, candle_low: float, candle_high: float) -> PaperFill:
		paper_order = PaperOrder(
			side=order.request.side,
			price=order.request.price,
			qty=order.request.qty,
			post_only=True,
		)
		return self._broker.try_fill(
			order=paper_order,
			candle_low=candle_low,
			candle_high=candle_high,
		)
