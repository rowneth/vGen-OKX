"""Reusable strategy filters."""

from __future__ import annotations


def ema_slope_ok_for_long(slope: float, min_slope: float = 0.0) -> bool:
	"""Check trend filter for long entries.

	Args:
		slope: EMA slope estimate.
		min_slope: Minimum acceptable slope.

	Returns:
		True when trend is flat/upward enough for long entries.
	"""
	return slope >= min_slope


def ema_slope_ok_for_short(slope: float, max_slope: float = 0.0) -> bool:
	"""Check trend filter for short entries.

	Args:
		slope: EMA slope estimate.
		max_slope: Maximum acceptable slope.

	Returns:
		True when trend is flat/downward enough for short entries.
	"""
	return slope <= max_slope


def bollinger_width_ok(width_pct: float, min_width_pct: float) -> bool:
	"""Check minimum Bollinger width volatility filter.

	Args:
		width_pct: Current Bollinger width as percentage of price.
		min_width_pct: Minimum required width percentage.

	Returns:
		True when volatility threshold is met.
	"""
	return width_pct >= min_width_pct
