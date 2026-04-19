"""Integration-level test for CipherConfluenceStrategy."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from strategy.cipher_confluence import CipherConfluenceStrategy


def _default_config() -> dict:
	return {
		"fees": {"maker": 0.0001, "taker": 0.0005},
		"exchange": {"tick_size": 0.1},
		"execution": {"slippage": {"base_ticks_per_fill": 1}},
		"strategy": {
			"bollinger": {"period": 20, "std_dev": 2.0},
			"fib": {"mult": 2.618},
			"wavetrend": {
				"channel_len": 9,
				"average_len": 12,
				"signal_len": 3,
				"oversold": -53.0,
				"oversold_deep": -75.0,
				"overbought": 53.0,
				"overbought_deep": 75.0,
			},
			"mfi": {"period": 14, "bull_threshold": 45.0, "bear_threshold": 55.0},
			"rsi": {"period": 14, "oversold": 35, "overbought": 65, "require_crossback": True, "crossback_min_delta": 0.0},
			"ema": {"period": 50, "slope_lookback_candles": 10, "slope_min": 0.0, "side_bias_max_atr_distance": 6.0},
			"atr": {"period": 14, "stop_multiple": 1.2},
			"volatility_filter": {"min_bandwidth_pct": 0.0005, "max_bandwidth_pct": 0.5},
			"volume_filter": {"enabled": False, "sma_period": 20, "min_ratio": 0.0},
			"confirmation": {"require_next_candle_close_confirmation": True, "min_body_fraction": 0.0},
			"exits": {
				"time_stop_candles": 12,
				"take_profit_at_middle_band": True,
				"partial_tp": {"enabled": True, "tp1_fraction_to_mid": 0.5, "tp1_size_fraction": 0.6, "move_stop_to_breakeven": True},
			},
			"signal_quality": {"min_rr_ratio": 0.5, "min_fee_cover_multiple": 1.0},
		},
	}


def _make_candles(rng_seed: int = 11, n: int = 400) -> pd.DataFrame:
	rng = np.random.default_rng(rng_seed)
	close = 30000 + np.cumsum(rng.normal(0, 25, n))
	high = close + rng.uniform(5, 25, n)
	low = close - rng.uniform(5, 25, n)
	open_ = close + rng.normal(0, 5, n)
	volume = rng.uniform(500, 3000, n)
	start = pd.Timestamp("2025-01-01", tz="UTC")
	open_time = pd.date_range(start, periods=n, freq="15min")
	close_time = open_time + pd.Timedelta(minutes=15)
	return pd.DataFrame(
		{
			"open_time": open_time,
			"close_time": close_time,
			"open": open_,
			"high": high,
			"low": low,
			"close": close,
			"volume": volume,
		}
	)


def test_prepare_attaches_all_expected_columns() -> None:
	strategy = CipherConfluenceStrategy(_default_config())
	frame = strategy.prepare(_make_candles())
	for col in ["bb_middle", "bb_upper", "bb_lower", "fib_upper", "fib_lower", "rsi", "atr", "wt1", "wt2", "mfi"]:
		assert col in frame.columns


def test_generate_signal_runs_without_errors_on_random_walk() -> None:
	strategy = CipherConfluenceStrategy(_default_config())
	frame = strategy.prepare(_make_candles())
	# Just ensure that calling across the whole frame never raises.
	fired = 0
	for i in range(50, len(frame)):
		sig = strategy.generate_signal(frame, i)
		if sig is not None:
			fired += 1
			assert sig.side in {"long", "short"}
			assert sig.entry_price > 0
			assert sig.stop_price > 0
			assert sig.take_profit_price > 0
	# Random walk should fire at least a few signals.
	assert fired >= 1
