"""Reusable strategy filters."""

from __future__ import annotations


def ema_slope_ok_for_long(slope: float, min_slope: float = 0.0) -> bool:
	"""Check trend filter for long entries.

	Args:
		slope: EMA slope estimate. Caller may pre-normalize by price.
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


def bollinger_width_ok(
	width_pct: float,
	min_width_pct: float,
	max_width_pct: float = float("inf"),
) -> bool:
	"""Check Bollinger width volatility regime filter.

	Args:
		width_pct: Current Bollinger width as percentage of price.
		min_width_pct: Minimum required width percentage.
		max_width_pct: Maximum allowed width percentage.

	Returns:
		True when volatility is inside acceptable band.
	"""
	return min_width_pct <= width_pct <= max_width_pct


def volume_ok(current_volume: float, avg_volume: float, min_ratio: float) -> bool:
	"""Check minimum volume ratio filter.

	Args:
		current_volume: Volume on the trigger candle.
		avg_volume: SMA volume reference.
		min_ratio: Minimum acceptable ratio current/avg.

	Returns:
		True when liquidity threshold is met.
	"""
	if avg_volume <= 0.0:
		return False
	return (current_volume / avg_volume) >= min_ratio


def body_fraction_ok(
	open_price: float,
	close_price: float,
	high: float,
	low: float,
	min_fraction: float,
) -> bool:
	"""Confirmation candle must have a meaningful directional body.

	Args:
		open_price: Candle open.
		close_price: Candle close.
		high: Candle high.
		low: Candle low.
		min_fraction: Minimum |close-open| / (high-low) ratio.

	Returns:
		True if body is at least ``min_fraction`` of the candle range.
	"""
	rng = max(0.0, high - low)
	if rng <= 0.0:
		return False
	body = abs(close_price - open_price)
	return (body / rng) >= min_fraction


def rsi_crossback_long(rsi_prev: float, rsi_now: float, oversold: float, min_delta: float) -> bool:
	"""RSI momentum-turn confirmation for longs.

	Trigger must be oversold; confirmation must tick RSI back up.

	Args:
		rsi_prev: RSI at trigger candle.
		rsi_now: RSI at confirmation candle.
		oversold: Oversold threshold.
		min_delta: Minimum positive change required.

	Returns:
		True if RSI was oversold and is turning up.
	"""
	return rsi_prev < oversold and (rsi_now - rsi_prev) >= min_delta


def rsi_crossback_short(rsi_prev: float, rsi_now: float, overbought: float, min_delta: float) -> bool:
	"""RSI momentum-turn confirmation for shorts.

	Args:
		rsi_prev: RSI at trigger candle.
		rsi_now: RSI at confirmation candle.
		overbought: Overbought threshold.
		min_delta: Minimum absolute negative change required.

	Returns:
		True if RSI was overbought and is turning down.
	"""
	return rsi_prev > overbought and (rsi_prev - rsi_now) >= min_delta


def rr_ratio_ok(entry: float, stop: float, take_profit: float, min_ratio: float) -> bool:
	"""Reward-to-risk geometry gate.

	Args:
		entry: Entry price.
		stop: Stop price.
		take_profit: Final take-profit price.
		min_ratio: Minimum acceptable |TP-entry|/|entry-stop|.

	Returns:
		True if the trade's RR meets the minimum.
	"""
	risk = abs(entry - stop)
	reward = abs(take_profit - entry)
	if risk <= 0.0:
		return False
	return (reward / risk) >= min_ratio


def fee_edge_ok(
	entry: float,
	take_profit: float,
	round_trip_fee_rate: float,
	slippage_rate: float,
	min_multiple: float,
) -> bool:
	"""Ensure TP edge covers fees + slippage by a safety multiple.

	Args:
		entry: Entry price.
		take_profit: TP price.
		round_trip_fee_rate: Sum of entry + exit fee rates.
		slippage_rate: Estimated round-trip slippage rate.
		min_multiple: Edge must be >= this * (fees + slippage).

	Returns:
		True if TP distance clears friction cost with margin.
	"""
	if entry <= 0.0:
		return False
	edge_pct = abs(take_profit - entry) / entry
	friction = round_trip_fee_rate + slippage_rate
	if friction <= 0.0:
		return True
	return edge_pct >= min_multiple * friction


def wavetrend_cross_up(wt1_prev: float, wt2_prev: float, wt1_now: float, wt2_now: float) -> bool:
	"""Detect WaveTrend bullish cross (WT1 crosses above WT2).

	Args:
		wt1_prev: Prior bar WT1.
		wt2_prev: Prior bar WT2.
		wt1_now: Current WT1.
		wt2_now: Current WT2.

	Returns:
		True if WT1 crossed above WT2 between bars.
	"""
	return wt1_prev <= wt2_prev and wt1_now > wt2_now


def wavetrend_cross_down(wt1_prev: float, wt2_prev: float, wt1_now: float, wt2_now: float) -> bool:
	"""Detect WaveTrend bearish cross (WT1 crosses below WT2).

	Args:
		wt1_prev: Prior bar WT1.
		wt2_prev: Prior bar WT2.
		wt1_now: Current WT1.
		wt2_now: Current WT2.

	Returns:
		True if WT1 crossed below WT2 between bars.
	"""
	return wt1_prev >= wt2_prev and wt1_now < wt2_now


def mfi_bullish(mfi: float, threshold: float = 50.0) -> bool:
	"""True if money flow indicates buying pressure."""
	return mfi >= threshold


def mfi_bearish(mfi: float, threshold: float = 50.0) -> bool:
	"""True if money flow indicates selling pressure."""
	return mfi <= threshold
