"""Position sizing logic based on account risk constraints."""

from __future__ import annotations


def risk_based_position_size(
	equity: float,
	risk_per_trade_pct: float,
	entry_price: float,
	stop_price: float,
	max_leverage: float,
) -> float:
	"""Compute position quantity from fixed fractional risk model.

	Args:
		equity: Current account equity.
		risk_per_trade_pct: Fraction of equity to risk (0.02 => 2%).
		entry_price: Intended entry price.
		stop_price: Protective stop price.
		max_leverage: Maximum allowed leverage cap.

	Returns:
		Quantity in base asset units.
	"""
	if equity <= 0:
		return 0.0

	stop_distance = abs(entry_price - stop_price)
	if stop_distance <= 0:
		return 0.0

	max_risk_amount = equity * risk_per_trade_pct
	qty_by_risk = max_risk_amount / stop_distance

	max_notional = equity * max_leverage
	qty_by_leverage = max_notional / entry_price

	return max(0.0, min(qty_by_risk, qty_by_leverage))
