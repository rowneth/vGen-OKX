"""Pure technical indicator functions used by strategy and backtests."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def sma(values: Iterable[float], period: int) -> np.ndarray:
	"""Compute simple moving average.

	Args:
		values: Input numeric sequence.
		period: Window size for the moving average.

	Returns:
		Array of SMA values with ``np.nan`` for unavailable warmup points.

	Raises:
		ValueError: If ``period`` is less than 1.
	"""
	if period < 1:
		raise ValueError("period must be >= 1")

	arr = _to_numpy(values)
	out = np.full(arr.shape, np.nan, dtype=float)
	if arr.size < period:
		return out

	csum = np.cumsum(arr, dtype=float)
	csum[period:] = csum[period:] - csum[:-period]
	out[period - 1 :] = csum[period - 1 :] / period
	return out


def ema(values: Iterable[float], period: int) -> np.ndarray:
	"""Compute exponential moving average using smoothing factor 2 / (period + 1).

	Args:
		values: Input numeric sequence.
		period: EMA period.

	Returns:
		Array of EMA values with ``np.nan`` for unavailable warmup points.

	Raises:
		ValueError: If ``period`` is less than 1.
	"""
	if period < 1:
		raise ValueError("period must be >= 1")

	arr = _to_numpy(values)
	out = np.full(arr.shape, np.nan, dtype=float)
	if arr.size < period:
		return out

	alpha = 2.0 / (period + 1.0)
	seed = float(np.mean(arr[:period]))
	out[period - 1] = seed

	prev = seed
	for i in range(period, arr.size):
		prev = alpha * arr[i] + (1.0 - alpha) * prev
		out[i] = prev
	return out


def rsi(values: Iterable[float], period: int = 14) -> np.ndarray:
	"""Compute RSI using Wilder's smoothing.

	Args:
		values: Close price sequence.
		period: RSI period.

	Returns:
		Array of RSI values in range [0, 100], with warmup as ``np.nan``.

	Raises:
		ValueError: If ``period`` is less than 1.
	"""
	if period < 1:
		raise ValueError("period must be >= 1")

	arr = _to_numpy(values)
	out = np.full(arr.shape, np.nan, dtype=float)
	if arr.size <= period:
		return out

	deltas = np.diff(arr)
	gains = np.where(deltas > 0.0, deltas, 0.0)
	losses = np.where(deltas < 0.0, -deltas, 0.0)

	avg_gain = float(np.mean(gains[:period]))
	avg_loss = float(np.mean(losses[:period]))
	out[period] = _rsi_from_averages(avg_gain, avg_loss)

	for i in range(period + 1, arr.size):
		gain = gains[i - 1]
		loss = losses[i - 1]
		avg_gain = ((period - 1) * avg_gain + gain) / period
		avg_loss = ((period - 1) * avg_loss + loss) / period
		out[i] = _rsi_from_averages(avg_gain, avg_loss)

	return out


def atr(
	high: Iterable[float],
	low: Iterable[float],
	close: Iterable[float],
	period: int = 14,
) -> np.ndarray:
	"""Compute Average True Range (ATR) using Wilder's smoothing.

	Args:
		high: High price sequence.
		low: Low price sequence.
		close: Close price sequence.
		period: ATR period.

	Returns:
		Array of ATR values with warmup as ``np.nan``.

	Raises:
		ValueError: If ``period`` is less than 1.
		ValueError: If input arrays do not have the same length.
	"""
	if period < 1:
		raise ValueError("period must be >= 1")

	h = _to_numpy(high)
	l = _to_numpy(low)
	c = _to_numpy(close)
	if not (h.size == l.size == c.size):
		raise ValueError("high, low, and close must have same length")

	out = np.full(h.shape, np.nan, dtype=float)
	if h.size <= period:
		return out

	tr = np.empty_like(h, dtype=float)
	tr[0] = h[0] - l[0]
	for i in range(1, h.size):
		hl = h[i] - l[i]
		hc = abs(h[i] - c[i - 1])
		lc = abs(l[i] - c[i - 1])
		tr[i] = max(hl, hc, lc)

	seed = float(np.mean(tr[1 : period + 1]))
	out[period] = seed
	prev = seed
	for i in range(period + 1, h.size):
		prev = ((period - 1) * prev + tr[i]) / period
		out[i] = prev
	return out


def bollinger_bands(
	values: Iterable[float],
	period: int = 20,
	std_dev: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Compute Bollinger Bands.

	Args:
		values: Input numeric sequence.
		period: Lookback period for rolling mean/std.
		std_dev: Standard deviation multiplier.

	Returns:
		Tuple of ``(middle, upper, lower)`` arrays.

	Raises:
		ValueError: If ``period`` is less than 1.
	"""
	if period < 1:
		raise ValueError("period must be >= 1")

	arr = _to_numpy(values)
	mid = sma(arr, period)
	std = rolling_std(arr, period)
	upper = mid + std_dev * std
	lower = mid - std_dev * std
	return mid, upper, lower


def rolling_std(values: Iterable[float], period: int) -> np.ndarray:
	"""Compute rolling standard deviation with population variance.

	Args:
		values: Input numeric sequence.
		period: Rolling window period.

	Returns:
		Array of rolling standard deviation values.
	"""
	if period < 1:
		raise ValueError("period must be >= 1")

	arr = _to_numpy(values)
	out = np.full(arr.shape, np.nan, dtype=float)
	if arr.size < period:
		return out

	for i in range(period - 1, arr.size):
		window = arr[i - period + 1 : i + 1]
		out[i] = float(np.std(window, ddof=0))
	return out


def _to_numpy(values: Iterable[float]) -> np.ndarray:
	arr = np.asarray(list(values), dtype=float)
	if arr.ndim != 1:
		raise ValueError("input must be 1-dimensional")
	return arr


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
	if avg_loss == 0.0 and avg_gain == 0.0:
		return 50.0
	if avg_loss == 0.0:
		return 100.0
	rs = avg_gain / avg_loss
	return 100.0 - (100.0 / (1.0 + rs))
