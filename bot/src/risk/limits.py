"""Runtime risk limits and circuit breakers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone


@dataclass
class RiskState:
	"""Tracks rolling state for risk checks."""

	day: date
	day_start_equity: float
	consecutive_losses: int = 0
	pause_until: datetime | None = None


class RiskLimits:
	"""Evaluates hard risk limits and pause conditions."""

	def __init__(
		self,
		daily_drawdown_limit_pct: float,
		consecutive_losses_limit: int,
		consecutive_losses_pause_hours: int,
	) -> None:
		"""Initialize risk limits engine.

		Args:
			daily_drawdown_limit_pct: Daily loss threshold to halt trading.
			consecutive_losses_limit: Number of losses before pause.
			consecutive_losses_pause_hours: Pause duration after threshold is hit.
		"""
		self._daily_drawdown_limit_pct = daily_drawdown_limit_pct
		self._consecutive_losses_limit = consecutive_losses_limit
		self._pause_hours = consecutive_losses_pause_hours

	def reset_for_day(self, now: datetime, equity: float) -> RiskState:
		"""Initialize a fresh day-level risk state.

		Args:
			now: Current timestamp.
			equity: Current account equity.

		Returns:
			Fresh RiskState object.
		"""
		day = now.astimezone(timezone.utc).date()
		return RiskState(day=day, day_start_equity=equity)

	def is_paused(self, state: RiskState, now: datetime) -> bool:
		"""Check whether trading is currently paused.

		Args:
			state: Mutable risk state.
			now: Current timestamp.

		Returns:
			True if pause is active.
		"""
		if state.pause_until is None:
			return False
		return now < state.pause_until

	def check_daily_drawdown(self, state: RiskState, equity: float) -> bool:
		"""Check if daily drawdown limit has been breached.

		Args:
			state: Mutable risk state.
			equity: Current equity.

		Returns:
			True if trading should halt for the day.
		"""
		if state.day_start_equity <= 0:
			return True
		drawdown = (state.day_start_equity - equity) / state.day_start_equity
		return drawdown >= self._daily_drawdown_limit_pct

	def register_trade_result(self, state: RiskState, now: datetime, pnl: float) -> None:
		"""Update loss streak and pause state from one closed trade.

		Args:
			state: Mutable risk state.
			now: Current timestamp.
			pnl: Realized PnL for the trade.
		"""
		if pnl < 0:
			state.consecutive_losses += 1
			if state.consecutive_losses >= self._consecutive_losses_limit:
				state.pause_until = now + timedelta(hours=self._pause_hours)
		else:
			state.consecutive_losses = 0
