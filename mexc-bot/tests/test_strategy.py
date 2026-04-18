"""Strategy behavior tests."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from strategy.bollinger import BollingerMeanReversionStrategy


def _config() -> dict:
	return {
		"strategy": {
			"bollinger": {"period": 20, "std_dev": 2.0},
			"rsi": {"period": 14, "oversold": 30, "overbought": 70},
			"ema": {"period": 10, "slope_lookback_candles": 3, "slope_min": 0.0},
			"atr": {"period": 14, "stop_multiple": 1.5},
			"volatility_filter": {"min_bandwidth_pct": 0.001},
			"confirmation": {"require_next_candle_close_confirmation": True},
			"exits": {"time_stop_candles": 16},
		}
	}


def test_strategy_returns_no_signal_with_short_history() -> None:
	strategy = BollingerMeanReversionStrategy(_config())
	candles = pd.DataFrame(
		{
			"open_time": pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC"),
			"close_time": pd.date_range("2024-01-01 00:15", periods=10, freq="15min", tz="UTC"),
			"open": [100.0] * 10,
			"high": [101.0] * 10,
			"low": [99.0] * 10,
			"close": [100.0] * 10,
			"volume": [1000.0] * 10,
		}
	)
	prepared = strategy.prepare(candles)
	assert strategy.generate_signal(prepared, index=5) is None


def test_strategy_emits_long_signal_on_constructed_setup() -> None:
	strategy = BollingerMeanReversionStrategy(_config())
	candles = pd.DataFrame(
		{
			"open_time": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
			"close_time": pd.date_range("2024-01-01 00:15", periods=3, freq="15min", tz="UTC"),
			"open": [100.0, 99.0, 100.0],
			"high": [101.0, 100.0, 101.0],
			"low": [99.0, 95.0, 99.0],
			"close": [100.0, 98.0, 101.0],
			"volume": [1000.0, 1200.0, 1400.0],
			"bb_lower": [np.nan, 96.0, np.nan],
			"bb_upper": [np.nan, 104.0, np.nan],
			"bb_middle": [np.nan, 100.0, np.nan],
			"bb_width_pct": [np.nan, 0.01, np.nan],
			"rsi": [np.nan, 25.0, np.nan],
			"ema_slope": [np.nan, 0.2, np.nan],
			"atr": [np.nan, 2.0, np.nan],
		}
	)
	signal = strategy.generate_signal(candles, index=2)
	assert signal is not None
	assert signal.side == "long"


def test_strategy_emits_short_signal_on_constructed_setup() -> None:
	strategy = BollingerMeanReversionStrategy(_config())
	candles = pd.DataFrame(
		{
			"open_time": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
			"close_time": pd.date_range("2024-01-01 00:15", periods=3, freq="15min", tz="UTC"),
			"open": [100.0, 101.0, 100.0],
			"high": [101.0, 106.0, 101.0],
			"low": [99.0, 100.0, 99.0],
			"close": [100.0, 104.0, 99.0],
			"volume": [1000.0, 1200.0, 1400.0],
			"bb_lower": [np.nan, 96.0, np.nan],
			"bb_upper": [np.nan, 103.0, np.nan],
			"bb_middle": [np.nan, 100.0, np.nan],
			"bb_width_pct": [np.nan, 0.01, np.nan],
			"rsi": [np.nan, 75.0, np.nan],
			"ema_slope": [np.nan, -0.3, np.nan],
			"atr": [np.nan, 2.0, np.nan],
		}
	)
	signal = strategy.generate_signal(candles, index=2)
	assert signal is not None
	assert signal.side == "short"
