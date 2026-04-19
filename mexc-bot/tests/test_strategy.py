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
		"fees": {"maker": 0.0001, "taker": 0.0005},
		"exchange": {"tick_size": 0.1},
		"execution": {"slippage": {"base_ticks_per_fill": 1}},
		"strategy": {
			"bollinger": {"period": 20, "std_dev": 2.0},
			"rsi": {
				"period": 14,
				"oversold": 35,
				"overbought": 65,
				"require_crossback": True,
				"crossback_min_delta": 0.5,
			},
			"ema": {
				"period": 10,
				"slope_lookback_candles": 3,
				"slope_min": 0.0,
				"side_bias_max_atr_distance": 6.0,
			},
			"atr": {"period": 14, "stop_multiple": 1.2},
			"volatility_filter": {"min_bandwidth_pct": 0.001, "max_bandwidth_pct": 0.5},
			"volume_filter": {"enabled": False, "sma_period": 20, "min_ratio": 0.0},
			"confirmation": {
				"require_next_candle_close_confirmation": True,
				"min_body_fraction": 0.0,
			},
			"exits": {
				"time_stop_candles": 12,
				"take_profit_at_middle_band": True,
				"partial_tp": {
					"enabled": True,
					"tp1_fraction_to_mid": 0.55,
					"tp1_size_fraction": 0.6,
					"move_stop_to_breakeven": True,
				},
			},
			"signal_quality": {
				"min_rr_ratio": 1.1,
				"min_fee_cover_multiple": 4.0,
			},
		},
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


def _constructed_frame(*, long_setup: bool) -> pd.DataFrame:
	if long_setup:
		return pd.DataFrame(
			{
				"open_time": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
				"close_time": pd.date_range("2024-01-01 00:15", periods=3, freq="15min", tz="UTC"),
				"open": [100.0, 99.0, 98.5],
				"high": [101.0, 100.0, 101.5],
				"low": [99.0, 95.0, 98.5],
				"close": [100.0, 98.0, 101.0],
				"volume": [1000.0, 1200.0, 1400.0],
				"bb_lower": [np.nan, 96.0, np.nan],
				"bb_upper": [np.nan, 104.0, np.nan],
				"bb_middle": [np.nan, 105.0, np.nan],
				"bb_width_pct": [np.nan, 0.08, np.nan],
				"rsi": [np.nan, 25.0, 31.0],
				"ema": [np.nan, 100.0, np.nan],
				"ema_slope": [np.nan, 0.0001, np.nan],
				"atr": [np.nan, 2.0, np.nan],
				"vol_sma": [np.nan, 1100.0, np.nan],
			}
		)
	return pd.DataFrame(
		{
			"open_time": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
			"close_time": pd.date_range("2024-01-01 00:15", periods=3, freq="15min", tz="UTC"),
			"open": [100.0, 101.0, 102.5],
			"high": [101.0, 106.0, 102.5],
			"low": [99.0, 100.0, 98.5],
			"close": [100.0, 104.0, 99.0],
			"volume": [1000.0, 1200.0, 1400.0],
			"bb_lower": [np.nan, 95.0, np.nan],
			"bb_upper": [np.nan, 103.0, np.nan],
			"bb_middle": [np.nan, 94.0, np.nan],
			"bb_width_pct": [np.nan, 0.08, np.nan],
			"rsi": [np.nan, 75.0, 70.0],
			"ema": [np.nan, 100.0, np.nan],
			"ema_slope": [np.nan, -0.0001, np.nan],
			"atr": [np.nan, 2.0, np.nan],
			"vol_sma": [np.nan, 1100.0, np.nan],
		}
	)


def test_strategy_emits_long_signal_on_constructed_setup() -> None:
	strategy = BollingerMeanReversionStrategy(_config())
	frame = _constructed_frame(long_setup=True)
	signal = strategy.generate_signal(frame, index=2)
	assert signal is not None
	assert signal.side == "long"
	assert signal.tp1_price is not None
	assert 0.0 < signal.tp1_size_fraction <= 1.0
	risk = abs(signal.entry_price - signal.stop_price)
	reward = abs(signal.take_profit_price - signal.entry_price)
	assert reward / risk >= 1.1


def test_strategy_emits_short_signal_on_constructed_setup() -> None:
	strategy = BollingerMeanReversionStrategy(_config())
	frame = _constructed_frame(long_setup=False)
	signal = strategy.generate_signal(frame, index=2)
	assert signal is not None
	assert signal.side == "short"


def test_rsi_crossback_required_blocks_flat_rsi() -> None:
	strategy = BollingerMeanReversionStrategy(_config())
	frame = _constructed_frame(long_setup=True)
	frame.loc[2, "rsi"] = 25.0
	assert strategy.generate_signal(frame, index=2) is None
