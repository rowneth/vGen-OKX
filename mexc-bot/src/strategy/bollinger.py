"""Bollinger band mean reversion strategy implementation (v2, fee-aware)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from strategy.base import Strategy, TradeSignal
from strategy.filters import (
	body_fraction_ok,
	bollinger_width_ok,
	ema_slope_ok_for_long,
	ema_slope_ok_for_short,
	fee_edge_ok,
	rr_ratio_ok,
	rsi_crossback_long,
	rsi_crossback_short,
	volume_ok,
)
from strategy.indicators import atr, bollinger_bands, ema, rsi, sma


@dataclass(frozen=True)
class BollingerStrategyConfig:
	"""Config container for Bollinger strategy parameters."""

	bb_period: int
	bb_std_dev: float
	rsi_period: int
	rsi_oversold: float
	rsi_overbought: float
	rsi_require_crossback: bool
	rsi_crossback_min_delta: float
	ema_period: int
	ema_slope_lookback: int
	ema_slope_min: float
	ema_side_bias_max_atr_distance: float
	atr_period: int
	atr_stop_multiple: float
	min_bandwidth_pct: float
	max_bandwidth_pct: float
	volume_filter_enabled: bool
	volume_sma_period: int
	volume_min_ratio: float
	require_confirmation: bool
	min_body_fraction: float
	partial_tp_enabled: bool
	tp1_fraction_to_mid: float
	tp1_size_fraction: float
	move_stop_to_breakeven: bool
	min_rr_ratio: float
	min_fee_cover_multiple: float
	maker_fee: float
	taker_fee: float
	estimated_slippage_rate: float


class BollingerMeanReversionStrategy(Strategy):
	"""Bollinger mean-reversion strategy for 15m BTC perpetual candles.

	This v2 implementation adds:
	- RSI momentum-crossback confirmation (avoids catching falling knives).
	- Volume-ratio liquidity filter.
	- Candle-body strength filter on confirmation.
	- Price-normalized EMA trend slope + side-bias vs EMA(200).
	- Reward-to-risk and fee-edge gates so every trade covers frictions.
	- Partial TP (TP1) metadata for scale-out + runner exits.
	"""

	def __init__(self, config: Dict[str, object]) -> None:
		"""Initialize strategy from parsed config.

		Args:
			config: Global configuration dictionary.
		"""
		strategy_cfg = config["strategy"]
		fees_cfg = config.get("fees", {"maker": 0.0001, "taker": 0.0005})
		exec_cfg = config.get("execution", {})
		slip_cfg = exec_cfg.get("slippage", {})

		# Approximate round-trip slippage: 2 base ticks on a BTC ~70k price ~ 0.2/70000.
		tick_size = float(config.get("exchange", {}).get("tick_size", 0.1))
		approx_slip_rate = 2.0 * tick_size / 70000.0
		try:
			ticks = int(slip_cfg.get("base_ticks_per_fill", 1))
			approx_slip_rate = 2.0 * ticks * tick_size / 70000.0
		except Exception:  # pragma: no cover - defensive
			pass

		partial_cfg = strategy_cfg["exits"].get("partial_tp", {})
		signal_quality = strategy_cfg.get("signal_quality", {})
		volume_filter = strategy_cfg.get("volume_filter", {})
		confirmation = strategy_cfg.get("confirmation", {})

		self.cfg = BollingerStrategyConfig(
			bb_period=int(strategy_cfg["bollinger"]["period"]),
			bb_std_dev=float(strategy_cfg["bollinger"]["std_dev"]),
			rsi_period=int(strategy_cfg["rsi"]["period"]),
			rsi_oversold=float(strategy_cfg["rsi"]["oversold"]),
			rsi_overbought=float(strategy_cfg["rsi"]["overbought"]),
			rsi_require_crossback=bool(strategy_cfg["rsi"].get("require_crossback", True)),
			rsi_crossback_min_delta=float(strategy_cfg["rsi"].get("crossback_min_delta", 0.0)),
			ema_period=int(strategy_cfg["ema"]["period"]),
			ema_slope_lookback=int(strategy_cfg["ema"]["slope_lookback_candles"]),
			ema_slope_min=float(strategy_cfg["ema"]["slope_min"]),
			ema_side_bias_max_atr_distance=float(
				strategy_cfg["ema"].get("side_bias_max_atr_distance", 1e9)
			),
			atr_period=int(strategy_cfg["atr"]["period"]),
			atr_stop_multiple=float(strategy_cfg["atr"]["stop_multiple"]),
			min_bandwidth_pct=float(strategy_cfg["volatility_filter"]["min_bandwidth_pct"]),
			max_bandwidth_pct=float(
				strategy_cfg["volatility_filter"].get("max_bandwidth_pct", 1.0)
			),
			volume_filter_enabled=bool(volume_filter.get("enabled", False)),
			volume_sma_period=int(volume_filter.get("sma_period", 20)),
			volume_min_ratio=float(volume_filter.get("min_ratio", 0.0)),
			require_confirmation=bool(
				confirmation.get("require_next_candle_close_confirmation", True)
			),
			min_body_fraction=float(confirmation.get("min_body_fraction", 0.0)),
			partial_tp_enabled=bool(partial_cfg.get("enabled", False)),
			tp1_fraction_to_mid=float(partial_cfg.get("tp1_fraction_to_mid", 0.5)),
			tp1_size_fraction=float(partial_cfg.get("tp1_size_fraction", 0.5)),
			move_stop_to_breakeven=bool(partial_cfg.get("move_stop_to_breakeven", True)),
			min_rr_ratio=float(signal_quality.get("min_rr_ratio", 1.0)),
			min_fee_cover_multiple=float(signal_quality.get("min_fee_cover_multiple", 0.0)),
			maker_fee=float(fees_cfg.get("maker", 0.0001)),
			taker_fee=float(fees_cfg.get("taker", 0.0005)),
			estimated_slippage_rate=approx_slip_rate,
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

		# Vectorized price-normalized EMA slope (slope per bar / price).
		ema_vals = frame["ema"].to_numpy(dtype=float)
		lb = self.cfg.ema_slope_lookback
		slope = np.full_like(ema_vals, np.nan, dtype=float)
		if lb > 0 and ema_vals.size > lb:
			diff = ema_vals[lb:] - ema_vals[:-lb]
			prices = close[lb:]
			with np.errstate(divide="ignore", invalid="ignore"):
				norm = (diff / lb) / prices
			slope[lb:] = norm
		frame["ema_slope"] = slope

		# Volume SMA for liquidity filter.
		if "volume" in frame.columns:
			vol = frame["volume"].to_numpy(dtype=float)
			frame["vol_sma"] = sma(vol, period=self.cfg.volume_sma_period)
		else:
			frame["vol_sma"] = np.nan

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
			trigger["ema"],
			trigger["ema_slope"],
			trigger["bb_width_pct"],
			trigger["atr"],
			confirm["rsi"],
		]
		if any(pd.isna(v) for v in required):
			return None

		if not bollinger_width_ok(
			width_pct=float(trigger["bb_width_pct"]),
			min_width_pct=self.cfg.min_bandwidth_pct,
			max_width_pct=self.cfg.max_bandwidth_pct,
		):
			return None

		if self.cfg.volume_filter_enabled:
			avg_vol = float(trigger.get("vol_sma", np.nan))
			cur_vol = float(trigger.get("volume", 0.0))
			if np.isnan(avg_vol) or not volume_ok(cur_vol, avg_vol, self.cfg.volume_min_ratio):
				return None

		long_signal = self._long_signal(trigger, confirm)
		if long_signal is not None:
			return long_signal
		return self._short_signal(trigger, confirm)

	def _long_signal(self, trigger: pd.Series, confirm: pd.Series) -> Optional[TradeSignal]:
		touched = float(trigger["low"]) <= float(trigger["bb_lower"])
		if not touched:
			return None

		rsi_prev = float(trigger["rsi"])
		rsi_now = float(confirm["rsi"])
		if self.cfg.rsi_require_crossback:
			if not rsi_crossback_long(
				rsi_prev=rsi_prev,
				rsi_now=rsi_now,
				oversold=self.cfg.rsi_oversold,
				min_delta=self.cfg.rsi_crossback_min_delta,
			):
				return None
		else:
			if rsi_prev >= self.cfg.rsi_oversold:
				return None

		if not ema_slope_ok_for_long(
			slope=float(trigger["ema_slope"]),
			min_slope=self.cfg.ema_slope_min,
		):
			return None

		# Side-bias: don't long if price is very far BELOW EMA(200) (strong downtrend).
		ema_val = float(trigger["ema"])
		atr_val = float(trigger["atr"])
		price = float(trigger["close"])
		if atr_val > 0 and (ema_val - price) / atr_val > self.cfg.ema_side_bias_max_atr_distance:
			return None

		if self.cfg.require_confirmation:
			if float(confirm["close"]) <= float(trigger["close"]):
				return None
			if self.cfg.min_body_fraction > 0.0:
				if not body_fraction_ok(
					open_price=float(confirm["open"]),
					close_price=float(confirm["close"]),
					high=float(confirm["high"]),
					low=float(confirm["low"]),
					min_fraction=self.cfg.min_body_fraction,
				):
					return None

		entry = float(confirm["close"])
		stop = entry - self.cfg.atr_stop_multiple * atr_val
		take_profit = float(trigger["bb_middle"])
		return self._finalize_signal(
			side="long",
			trigger=trigger,
			confirm=confirm,
			entry=entry,
			stop=stop,
			take_profit=take_profit,
		)

	def _short_signal(self, trigger: pd.Series, confirm: pd.Series) -> Optional[TradeSignal]:
		touched = float(trigger["high"]) >= float(trigger["bb_upper"])
		if not touched:
			return None

		rsi_prev = float(trigger["rsi"])
		rsi_now = float(confirm["rsi"])
		if self.cfg.rsi_require_crossback:
			if not rsi_crossback_short(
				rsi_prev=rsi_prev,
				rsi_now=rsi_now,
				overbought=self.cfg.rsi_overbought,
				min_delta=self.cfg.rsi_crossback_min_delta,
			):
				return None
		else:
			if rsi_prev <= self.cfg.rsi_overbought:
				return None

		if not ema_slope_ok_for_short(
			slope=float(trigger["ema_slope"]),
			max_slope=-self.cfg.ema_slope_min,
		):
			return None

		ema_val = float(trigger["ema"])
		atr_val = float(trigger["atr"])
		price = float(trigger["close"])
		if atr_val > 0 and (price - ema_val) / atr_val > self.cfg.ema_side_bias_max_atr_distance:
			return None

		if self.cfg.require_confirmation:
			if float(confirm["close"]) >= float(trigger["close"]):
				return None
			if self.cfg.min_body_fraction > 0.0:
				if not body_fraction_ok(
					open_price=float(confirm["open"]),
					close_price=float(confirm["close"]),
					high=float(confirm["high"]),
					low=float(confirm["low"]),
					min_fraction=self.cfg.min_body_fraction,
				):
					return None

		entry = float(confirm["close"])
		stop = entry + self.cfg.atr_stop_multiple * atr_val
		take_profit = float(trigger["bb_middle"])
		return self._finalize_signal(
			side="short",
			trigger=trigger,
			confirm=confirm,
			entry=entry,
			stop=stop,
			take_profit=take_profit,
		)

	def _finalize_signal(
		self,
		side: str,
		trigger: pd.Series,
		confirm: pd.Series,
		entry: float,
		stop: float,
		take_profit: float,
	) -> Optional[TradeSignal]:
		"""Apply RR / fee-edge gates and compute partial-TP levels."""

		if not rr_ratio_ok(entry=entry, stop=stop, take_profit=take_profit, min_ratio=self.cfg.min_rr_ratio):
			return None

		round_trip_fee = self.cfg.maker_fee * 2.0  # both sides as maker ideal case
		if not fee_edge_ok(
			entry=entry,
			take_profit=take_profit,
			round_trip_fee_rate=round_trip_fee,
			slippage_rate=self.cfg.estimated_slippage_rate,
			min_multiple=self.cfg.min_fee_cover_multiple,
		):
			return None

		tp1_price: Optional[float] = None
		tp1_size = 0.0
		if self.cfg.partial_tp_enabled:
			frac = max(0.0, min(1.0, self.cfg.tp1_fraction_to_mid))
			tp1_price = entry + frac * (take_profit - entry)
			tp1_size = max(0.0, min(1.0, self.cfg.tp1_size_fraction))

		return TradeSignal(
			side=side,
			reference_index=int(trigger.name),
			entry_index=int(confirm.name),
			entry_price=entry,
			stop_price=stop,
			take_profit_price=take_profit,
			tp1_price=tp1_price,
			tp1_size_fraction=tp1_size,
			move_stop_to_breakeven_after_tp1=self.cfg.move_stop_to_breakeven,
		)
