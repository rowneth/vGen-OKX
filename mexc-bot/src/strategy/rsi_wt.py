"""RSI + WaveTrend (Market Cipher B) cross strategy.

Ports two Pine Scripts to Python:

1. **WTF Indicator** (ChrisMoody, 2014) — simple RSI with:
   - Buy = RSI[1] < lowLine AND RSI > lowLine  (crossback up from oversold)
   - Sell = RSI[1] > upLine AND RSI < upLine  (crossback down from overbought)

2. **WTF Money Flow** (Market Cipher B clone, LazyBear WaveTrend):
   - Green dot = WT cross AND WT cross-up AND WT oversold (wt2 <= -53)
   - Red dot = WT cross AND WT cross-down AND WT overbought (wt2 >= 53)
   - Gold dot = green dot + bullish divergence + RSI < 30

The strategy fires entries on RSI crossback signals (primary), optionally gated
by a WaveTrend "green/red dot" confluence within a lookback window. Exits use
ATR-based stop + fixed RR target + time-stop. Opposite signals trigger an exit
on the current position before a new one opens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from strategy.base import Strategy, TradeSignal
from strategy.indicators import atr, rsi, wavetrend


@dataclass(frozen=True)
class RsiWtConfig:
	"""Parameters for the RSI + WaveTrend strategy."""

	# RSI
	rsi_period: int
	rsi_upper: float
	rsi_lower: float

	# WaveTrend (Market Cipher B)
	wt_channel_len: int
	wt_average_len: int
	wt_signal_len: int
	wt_oversold: float
	wt_overbought: float
	wt_oversold_deep: float  # for gold

	# Confluence
	require_wt_confirmation: bool  # if True, need WT dot within lookback window
	wt_confluence_lookback: int     # bars to look back for WT dot

	# ATR for stops
	atr_period: int
	atr_stop_multiple: float

	# Exits
	min_rr_ratio: float
	time_stop_candles: int

	# Fees
	maker_fee: float
	taker_fee: float
	min_fee_cover_multiple: float

	# Confirmation bar behavior
	require_confirmation: bool


class RsiWtStrategy(Strategy):
	"""RSI crossback strategy with optional WaveTrend confluence."""

	def __init__(self, config: Dict[str, object]) -> None:
		s = config["strategy"]
		fees_cfg = config.get("fees", {"maker": 0.0001, "taker": 0.0005})

		rsi_cfg = s.get("rsi", {})
		wt_cfg = s.get("wavetrend", {})
		exit_cfg = s.get("exits", {})

		self.cfg = RsiWtConfig(
			rsi_period=int(rsi_cfg.get("period", 14)),
			rsi_upper=float(rsi_cfg.get("upper", 70.0)),
			rsi_lower=float(rsi_cfg.get("lower", 30.0)),
			wt_channel_len=int(wt_cfg.get("channel_len", 9)),
			wt_average_len=int(wt_cfg.get("average_len", 12)),
			wt_signal_len=int(wt_cfg.get("signal_len", 3)),
			wt_oversold=float(wt_cfg.get("oversold", -53.0)),
			wt_overbought=float(wt_cfg.get("overbought", 53.0)),
			wt_oversold_deep=float(wt_cfg.get("oversold_deep", -75.0)),
			require_wt_confirmation=bool(s.get("require_wt_confirmation", False)),
			wt_confluence_lookback=int(s.get("wt_confluence_lookback", 3)),
			atr_period=int(s.get("atr_period", 14)),
			atr_stop_multiple=float(s.get("atr_stop_multiple", 1.5)),
			min_rr_ratio=float(exit_cfg.get("min_rr_ratio", 2.0)),
			time_stop_candles=int(exit_cfg.get("time_stop_candles", 24)),
			maker_fee=float(fees_cfg.get("maker", 0.0001)),
			taker_fee=float(fees_cfg.get("taker", 0.0005)),
			min_fee_cover_multiple=float(s.get("min_fee_cover_multiple", 3.0)),
			require_confirmation=bool(s.get("require_confirmation", False)),
		)

	# ------------------------------------------------------------------
	def prepare(self, candles: pd.DataFrame) -> pd.DataFrame:
		"""Compute RSI, WaveTrend, and ATR columns."""
		df = candles.copy()
		close = df["close"].astype(float).values

		df["rsi"] = rsi(close, self.cfg.rsi_period)

		wt1, wt2 = wavetrend(
			high=df["high"].astype(float).values,
			low=df["low"].astype(float).values,
			close=close,
			channel_len=self.cfg.wt_channel_len,
			average_len=self.cfg.wt_average_len,
			signal_len=self.cfg.wt_signal_len,
		)
		df["wt1"] = wt1
		df["wt2"] = wt2

		df["atr"] = atr(
			high=df["high"].astype(float).values,
			low=df["low"].astype(float).values,
			close=close,
			period=self.cfg.atr_period,
		)

		# Derived boolean signals (vectorised for speed / diagnostics)
		rsi_series = df["rsi"]
		rsi_prev = rsi_series.shift(1)
		df["rsi_buy"] = (rsi_prev < self.cfg.rsi_lower) & (rsi_series >= self.cfg.rsi_lower)
		df["rsi_sell"] = (rsi_prev > self.cfg.rsi_upper) & (rsi_series <= self.cfg.rsi_upper)

		# WaveTrend cross (wt1 crosses wt2)
		wt1_s = df["wt1"]
		wt2_s = df["wt2"]
		wt1_prev = wt1_s.shift(1)
		wt2_prev = wt2_s.shift(1)
		cross_up = (wt1_prev <= wt2_prev) & (wt1_s > wt2_s)
		cross_dn = (wt1_prev >= wt2_prev) & (wt1_s < wt2_s)
		df["wt_green_dot"] = cross_up & (wt2_s <= self.cfg.wt_oversold)
		df["wt_red_dot"] = cross_dn & (wt2_s >= self.cfg.wt_overbought)
		df["wt_gold_dot"] = df["wt_green_dot"] & (wt2_s <= self.cfg.wt_oversold_deep) & (rsi_series < 30.0)

		return df

	# ------------------------------------------------------------------
	def _wt_confluence_recent(self, candles: pd.DataFrame, index: int, col: str) -> bool:
		"""Was a WT dot of given type present within lookback window up to index?"""
		if not self.cfg.require_wt_confirmation:
			return True
		start = max(0, index - self.cfg.wt_confluence_lookback)
		return bool(candles[col].iloc[start:index + 1].any())

	# ------------------------------------------------------------------
	def generate_signal(self, candles: pd.DataFrame, index: int) -> Optional[TradeSignal]:
		"""Produce a TradeSignal on RSI crossback; optional WT confirmation."""

		if index < max(self.cfg.rsi_period, self.cfg.atr_period, self.cfg.wt_channel_len) + 2:
			return None

		# The trigger bar is the bar that produced the signal; the entry bar
		# (confirmation) is the next bar. If require_confirmation is False,
		# we treat the same bar as both.
		if self.cfg.require_confirmation:
			if index + 1 >= len(candles):
				return None
			trig = candles.iloc[index]
			conf = candles.iloc[index + 1]
			conf_idx = index + 1
		else:
			trig = candles.iloc[index]
			conf = trig
			conf_idx = index

		atr_val = float(trig["atr"]) if not pd.isna(trig["atr"]) else 0.0
		if atr_val <= 0.0:
			return None

		rsi_buy = bool(trig["rsi_buy"])
		rsi_sell = bool(trig["rsi_sell"])
		if not (rsi_buy or rsi_sell):
			return None

		if rsi_buy:
			if not self._wt_confluence_recent(candles, index, "wt_green_dot"):
				return None
			return self._long_signal(trig, conf, conf_idx, atr_val, index)

		# rsi_sell
		if not self._wt_confluence_recent(candles, index, "wt_red_dot"):
			return None
		return self._short_signal(trig, conf, conf_idx, atr_val, index)

	# ------------------------------------------------------------------
	def _long_signal(
		self, trig: pd.Series, conf: pd.Series, conf_idx: int,
		atr_val: float, trig_idx: int,
	) -> Optional[TradeSignal]:
		entry = float(conf["close"])
		stop = entry - self.cfg.atr_stop_multiple * atr_val
		if stop >= entry:
			return None
		risk = entry - stop
		rr_tp = entry + self.cfg.min_rr_ratio * risk
		# Fee-edge sanity: TP must cover round-trip + margin
		round_trip = self.cfg.maker_fee + self.cfg.taker_fee
		fee_tp = entry * (1.0 + self.cfg.min_fee_cover_multiple * round_trip)
		take_profit = max(rr_tp, fee_tp)
		if take_profit <= entry:
			return None
		return TradeSignal(
			side="long",
			reference_index=int(trig_idx),
			entry_index=int(conf_idx),
			entry_price=entry,
			stop_price=stop,
			take_profit_price=take_profit,
		)

	# ------------------------------------------------------------------
	def _short_signal(
		self, trig: pd.Series, conf: pd.Series, conf_idx: int,
		atr_val: float, trig_idx: int,
	) -> Optional[TradeSignal]:
		entry = float(conf["close"])
		stop = entry + self.cfg.atr_stop_multiple * atr_val
		if stop <= entry:
			return None
		risk = stop - entry
		rr_tp = entry - self.cfg.min_rr_ratio * risk
		round_trip = self.cfg.maker_fee + self.cfg.taker_fee
		fee_tp = entry * (1.0 - self.cfg.min_fee_cover_multiple * round_trip)
		take_profit = min(rr_tp, fee_tp)
		if take_profit >= entry:
			return None
		return TradeSignal(
			side="short",
			reference_index=int(trig_idx),
			entry_index=int(conf_idx),
			entry_price=entry,
			stop_price=stop,
			take_profit_price=take_profit,
		)
