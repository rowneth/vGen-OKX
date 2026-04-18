"""Bollinger band mean reversion strategy implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from strategy.base import Strategy, TradeSignal
from strategy.filters import (
	bollinger_width_ok,
	ema_slope_ok_for_long,
	ema_slope_ok_for_short,
)
from strategy.indicators import atr, bollinger_bands, ema, rsi


@dataclass(frozen=True)
class BollingerStrategyConfig:
	"""Config container for Bollinger strategy parameters."""

	bb_period: int
	bb_std_dev: float
	rsi_period: int
	rsi_oversold: float
	rsi_overbought: float
	ema_period: int
	ema_slope_lookback: int
	ema_slope_min: float
	atr_period: int
	atr_stop_multiple: float
	min_bandwidth_pct: float


class BollingerMeanReversionStrategy(Strategy):
	"""Bollinger mean-reversion strategy for 15m BTC perpetual candles."""

	def __init__(self, config: Dict[str, object]) -> None:
		"""Initialize strategy from parsed config.

		Args:
			config: Global configuration dictionary.
		"""
		strategy_cfg = config["strategy"]
		self.cfg = BollingerStrategyConfig(
			bb_period=int(strategy_cfg["bollinger"]["period"]),
			bb_std_dev=float(strategy_cfg["bollinger"]["std_dev"]),
			rsi_period=int(strategy_cfg["rsi"]["period"]),
			rsi_oversold=float(strategy_cfg["rsi"]["oversold"]),
			rsi_overbought=float(strategy_cfg["rsi"]["overbought"]),
			ema_period=int(strategy_cfg["ema"]["period"]),
			ema_slope_lookback=int(strategy_cfg["ema"]["slope_lookback_candles"]),
			ema_slope_min=float(strategy_cfg["ema"]["slope_min"]),
			atr_period=int(strategy_cfg["atr"]["period"]),
			atr_stop_multiple=float(strategy_cfg["atr"]["stop_multiple"]),
			min_bandwidth_pct=float(strategy_cfg["volatility_filter"]["min_bandwidth_pct"]),
		)

	def prepare(self, candles: pd.DataFrame) -> pd.DataFrame:
		"""Add indicator columns required by the strategy.

		Args:
			candles: Input OHLCV dataframe.

		Returns:
			Dataframe with derived indicator columns.
		"""
		frame = candles.copy()
		close = frame["close"].to_numpy(dtype=float)
		high = frame["high"].to_numpy(dtype=float)
		low = frame["low"].to_numpy(dtype=float)

		middle, upper, lower = bollinger_bands(
			close,
			period=self.cfg.bb_period,
			std_dev=self.cfg.bb_std_dev,
		)
		frame["bb_middle"] = middle
		frame["bb_upper"] = upper
		frame["bb_lower"] = lower
		frame["bb_width_pct"] = (upper - lower) / close

		frame["rsi"] = rsi(close, period=self.cfg.rsi_period)
		frame["ema"] = ema(close, period=self.cfg.ema_period)
		frame["atr"] = atr(high, low, close, period=self.cfg.atr_period)

		frame["ema_slope"] = np.nan
		lb = self.cfg.ema_slope_lookback
		for i in range(lb, len(frame)):
			prev = frame.iloc[i - lb]["ema"]
			cur = frame.iloc[i]["ema"]
			if np.isnan(prev) or np.isnan(cur):
				continue
			frame.at[i, "ema_slope"] = (cur - prev) / lb

		return frame

	def generate_signal(self, candles: pd.DataFrame, index: int) -> Optional[TradeSignal]:
		"""Generate signal where current candle acts as confirmation candle.

		Args:
			candles: Prepared candle dataframe with indicator columns.
			index: Confirmation candle index.

		Returns:
			TradeSignal if all entry conditions pass, else None.
		"""
		if index <= 0 or index >= len(candles):
			return None

		trigger = candles.iloc[index - 1]
		confirm = candles.iloc[index]

		required = [
			trigger["bb_lower"],
			trigger["bb_upper"],
			trigger["bb_middle"],
			trigger["rsi"],
			trigger["ema_slope"],
			trigger["bb_width_pct"],
			trigger["atr"],
		]
		if any(np.isnan(value) for value in required):
			return None

		if bollinger_width_ok(
			width_pct=float(trigger["bb_width_pct"]),
			min_width_pct=self.cfg.min_bandwidth_pct,
		):
			long_signal = self._long_signal(trigger, confirm)
			if long_signal:
				return long_signal

			short_signal = self._short_signal(trigger, confirm)
			if short_signal:
				return short_signal

		return None

	def _long_signal(self, trigger: pd.Series, confirm: pd.Series) -> Optional[TradeSignal]:
		touched = float(trigger["low"]) <= float(trigger["bb_lower"])
		rsi_ok = float(trigger["rsi"]) < self.cfg.rsi_oversold
		trend_ok = ema_slope_ok_for_long(
			slope=float(trigger["ema_slope"]),
			min_slope=self.cfg.ema_slope_min,
		)
		confirm_ok = float(confirm["close"]) > float(trigger["close"])

		if not (touched and rsi_ok and trend_ok and confirm_ok):
			return None

		entry = float(confirm["close"])
		stop = entry - self.cfg.atr_stop_multiple * float(trigger["atr"])
		take_profit = float(trigger["bb_middle"])
		return TradeSignal(
			side="long",
			reference_index=int(trigger.name),
			entry_index=int(confirm.name),
			entry_price=entry,
			stop_price=stop,
			take_profit_price=take_profit,
		)

	def _short_signal(self, trigger: pd.Series, confirm: pd.Series) -> Optional[TradeSignal]:
		touched = float(trigger["high"]) >= float(trigger["bb_upper"])
		rsi_ok = float(trigger["rsi"]) > self.cfg.rsi_overbought
		trend_ok = ema_slope_ok_for_short(
			slope=float(trigger["ema_slope"]),
			max_slope=-self.cfg.ema_slope_min,
		)
		confirm_ok = float(confirm["close"]) < float(trigger["close"])

		if not (touched and rsi_ok and trend_ok and confirm_ok):
			return None

		entry = float(confirm["close"])
		stop = entry + self.cfg.atr_stop_multiple * float(trigger["atr"])
		take_profit = float(trigger["bb_middle"])
		return TradeSignal(
			side="short",
			reference_index=int(trigger.name),
			entry_index=int(confirm.name),
			entry_price=entry,
			stop_price=stop,
			take_profit_price=take_profit,
		)
