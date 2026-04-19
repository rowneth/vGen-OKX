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


def smoothed_sma(values: Iterable[float], period: int) -> np.ndarray:
	"""Smoothed SMA (SSMA): SMA of the SMA. Used as Fibonacci-band baseline.

	Args:
		values: Input numeric sequence.
		period: SMA period (applied twice).

	Returns:
		Array with SSMA values, NaN during warmup.
	"""
	if period < 1:
		raise ValueError("period must be >= 1")
	arr = _to_numpy(values)
	first = sma(arr, period)
	out = np.full(arr.shape, np.nan, dtype=float)
	# Manual rolling mean over ``first`` (which has NaN warmup) to avoid
	# cumsum NaN propagation.
	start = period - 1
	for i in range(start + period - 1, arr.size):
		window = first[i - period + 1 : i + 1]
		if np.isnan(window).any():
			continue
		out[i] = float(np.mean(window))
	return out


def fib_bands(
	values: Iterable[float],
	period: int = 20,
	mult: float = 2.618,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Fibonacci-scaled volatility bands around SSMA (FGI COM indicator).

	Args:
		values: Close price sequence.
		period: SSMA/std window.
		mult: Fibonacci multiplier (2.618 = extreme zone default).

	Returns:
		Tuple of ``(middle, upper, lower)`` arrays.
	"""
	arr = _to_numpy(values)
	mid = smoothed_sma(arr, period)
	std = rolling_std(arr, period)
	upper = mid + mult * std
	lower = mid - mult * std
	return mid, upper, lower


def wavetrend(
	high: Iterable[float],
	low: Iterable[float],
	close: Iterable[float],
	channel_len: int = 9,
	average_len: int = 12,
	signal_len: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
	"""VuManchu Cipher B WaveTrend oscillator (WT1, WT2).

	Formula (LazyBear): on hlc3 source, esa = EMA(src, chlen);
	d = EMA(|src - esa|, chlen); ci = (src - esa) / (0.015 * d);
	wt1 = EMA(ci, average_len); wt2 = SMA(wt1, signal_len).

	Args:
		high: High price sequence.
		low: Low price sequence.
		close: Close price sequence.
		channel_len: Channel length (default 9).
		average_len: Averaging length (default 12).
		signal_len: Signal SMA smoothing length (default 3).

	Returns:
		Tuple ``(wt1, wt2)`` aligned to input; NaN during warmup.
	"""
	h = _to_numpy(high)
	l = _to_numpy(low)
	c = _to_numpy(close)
	if not (h.size == l.size == c.size):
		raise ValueError("high, low, close must match in length")
	src = (h + l + c) / 3.0
	esa_full = ema(src, channel_len)
	# Work on the valid tail so NaN warmup doesn't poison EMAs/SMAs.
	valid_start = channel_len - 1
	if src.size <= valid_start + 1:
		nan = np.full(src.shape, np.nan, dtype=float)
		return nan, nan.copy()

	src_v = src[valid_start:]
	esa_v = esa_full[valid_start:]
	dev_v = ema(np.abs(src_v - esa_v), channel_len)
	with np.errstate(divide="ignore", invalid="ignore"):
		ci_v = np.where(dev_v > 0.0, (src_v - esa_v) / (0.015 * dev_v), 0.0)
	ci_v = np.where(np.isfinite(ci_v), ci_v, 0.0)
	wt1_v = ema(ci_v, average_len)
	# NaN-safe rolling mean for wt2 (cumsum-based SMA propagates NaN).
	wt2_v = np.full(wt1_v.shape, np.nan, dtype=float)
	for i in range(signal_len - 1, wt1_v.size):
		window = wt1_v[i - signal_len + 1 : i + 1]
		if np.isnan(window).any():
			continue
		wt2_v[i] = float(np.mean(window))

	wt1 = np.full(src.shape, np.nan, dtype=float)
	wt2 = np.full(src.shape, np.nan, dtype=float)
	wt1[valid_start:] = wt1_v
	wt2[valid_start:] = wt2_v
	return wt1, wt2


def money_flow_index(
	high: Iterable[float],
	low: Iterable[float],
	close: Iterable[float],
	volume: Iterable[float],
	period: int = 14,
) -> np.ndarray:
	"""Classic Money Flow Index (0..100). Values >50 = bullish flow.

	Args:
		high: High prices.
		low: Low prices.
		close: Close prices.
		volume: Volume per bar.
		period: MFI window.

	Returns:
		MFI array with NaN during warmup.
	"""
	if period < 1:
		raise ValueError("period must be >= 1")
	h = _to_numpy(high)
	l = _to_numpy(low)
	c = _to_numpy(close)
	v = _to_numpy(volume)
	if not (h.size == l.size == c.size == v.size):
		raise ValueError("high, low, close, volume must match in length")

	tp = (h + l + c) / 3.0
	rmf = tp * v
	out = np.full(h.shape, np.nan, dtype=float)
	if h.size <= period:
		return out
	pos = np.zeros_like(tp)
	neg = np.zeros_like(tp)
	pos[1:] = np.where(tp[1:] > tp[:-1], rmf[1:], 0.0)
	neg[1:] = np.where(tp[1:] < tp[:-1], rmf[1:], 0.0)
	for i in range(period, h.size):
		p = float(np.sum(pos[i - period + 1 : i + 1]))
		n = float(np.sum(neg[i - period + 1 : i + 1]))
		if n == 0.0 and p == 0.0:
			out[i] = 50.0
		elif n == 0.0:
			out[i] = 100.0
		else:
			mr = p / n
			out[i] = 100.0 - (100.0 / (1.0 + mr))
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
