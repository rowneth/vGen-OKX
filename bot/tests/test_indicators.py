"""Unit tests for technical indicators."""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from strategy.indicators import (
	atr,
	bollinger_bands,
	ema,
	fib_bands,
	money_flow_index,
	rolling_std,
	rsi,
	sma,
	smoothed_sma,
	wavetrend,
)


def test_sma_computes_expected_values() -> None:
	values = [1, 2, 3, 4, 5]
	result = sma(values, period=3)
	expected = np.array([np.nan, np.nan, 2.0, 3.0, 4.0])
	assert np.allclose(result[2:], expected[2:])
	assert np.isnan(result[0])
	assert np.isnan(result[1])


def test_ema_seed_and_progression() -> None:
	values = [10, 11, 12, 13, 14]
	result = ema(values, period=3)

	assert np.isnan(result[0])
	assert np.isnan(result[1])
	assert result[2] == pytest.approx(11.0)
	assert result[3] == pytest.approx(12.0)
	assert result[4] == pytest.approx(13.0)


def test_rsi_constant_series_equals_50_after_warmup() -> None:
	values = [100.0] * 30
	result = rsi(values, period=14)
	assert np.isnan(result[:14]).all()
	assert result[14:] == pytest.approx(np.array([50.0] * (len(values) - 14)))


def test_rsi_uptrend_goes_high() -> None:
	values = list(np.linspace(100, 140, 60))
	result = rsi(values, period=14)
	assert result[-1] > 70.0


def test_atr_returns_positive_values_after_warmup() -> None:
	high = [10, 11, 12, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
	low = [9, 10, 10.5, 10, 11, 12.5, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
	close = [9.5, 10.5, 11, 10.5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

	result = atr(high, low, close, period=14)
	assert np.isnan(result[:14]).all()
	assert not math.isnan(result[14])
	assert result[14] > 0.0
	assert result[15] > 0.0


def test_bollinger_bands_shape_and_order() -> None:
	values = [100 + (i % 3) for i in range(40)]
	middle, upper, lower = bollinger_bands(values, period=20, std_dev=2.0)
	assert middle.shape == upper.shape == lower.shape
	valid = ~np.isnan(middle)
	assert np.all(upper[valid] >= middle[valid])
	assert np.all(lower[valid] <= middle[valid])


def test_rolling_std_zero_for_flat_window() -> None:
	values = [5.0] * 10
	result = rolling_std(values, period=5)
	assert np.isnan(result[:4]).all()
	assert np.allclose(result[4:], 0.0)


def test_indicator_period_validation() -> None:
	with pytest.raises(ValueError):
		sma([1, 2, 3], period=0)
	with pytest.raises(ValueError):
		ema([1, 2, 3], period=0)
	with pytest.raises(ValueError):
		rsi([1, 2, 3], period=0)
	with pytest.raises(ValueError):
		rolling_std([1, 2, 3], period=0)


def test_smoothed_sma_matches_double_sma() -> None:
	values = np.linspace(1.0, 10.0, 50)
	direct = sma(sma(values, 5), 5)
	ssma = smoothed_sma(values, 5)
	mask = ~np.isnan(direct)
	assert np.allclose(ssma[mask], direct[mask])


def test_fib_bands_order_and_mult_scales() -> None:
	values = np.concatenate([np.linspace(100, 110, 30), np.linspace(110, 95, 30)])
	mid, up, lo = fib_bands(values, period=20, mult=2.618)
	mask = ~np.isnan(up)
	assert (up[mask] >= mid[mask]).all()
	assert (lo[mask] <= mid[mask]).all()
	# Wider than Bollinger at same period with 2 std-dev multiplier.
	_, bbu, _ = bollinger_bands(values, period=20, std_dev=2.0)
	assert np.nanmean(up - mid) > np.nanmean(bbu - mid)


def test_wavetrend_emits_bounded_oscillator() -> None:
	rng = np.random.default_rng(7)
	close = 100 + np.cumsum(rng.normal(0, 0.5, 300))
	high = close + rng.uniform(0.1, 1.0, 300)
	low = close - rng.uniform(0.1, 1.0, 300)
	wt1, wt2 = wavetrend(high, low, close)
	mask = ~np.isnan(wt2)
	assert mask.sum() > 100
	# WT typically oscillates within ~[-120, 120]; assert finite / bounded.
	assert np.all(np.abs(wt2[mask]) < 500.0)


def test_money_flow_index_bounds() -> None:
	rng = np.random.default_rng(3)
	close = 100 + np.cumsum(rng.normal(0, 0.4, 200))
	high = close + 0.5
	low = close - 0.5
	volume = rng.uniform(500, 2000, 200)
	out = money_flow_index(high, low, close, volume, period=14)
	mask = ~np.isnan(out)
	assert mask.sum() > 100
	assert np.all(out[mask] >= 0.0)
	assert np.all(out[mask] <= 100.0)
