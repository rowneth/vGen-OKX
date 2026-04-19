"""Cipher Confluence strategy: WaveTrend + MFI + RSI crossback + Fib bands.

This strategy combines three proven mean-reversion tools:

1. VuManchu Cipher B **WaveTrend** (WT1/WT2) — primary turn signal.
2. **Money Flow Index** — capital-flow confirmation (no "dead-cat" reversals).
3. **RSI crossback** from oversold/overbought — momentum confirmation.
4. **Fibonacci bands** (SSMA ± 2.618σ) — extreme-zone filter: entries deep
   inside Bollinger while price is near Fib extremes have historically the
   highest reversion probability.

Entry logic (long; short is mirrored):

- WaveTrend bullish cross (WT1 crosses above WT2) on confirmation bar,
- with WT2 in oversold zone (``wt_oversold``) or deeper "gold" zone.
- Either RSI < oversold (strong setup) OR price ≤ lower Fib band
  (extreme-zone fallback so we trade when WT alone fires).
- MFI ≥ ``mfi_bull_threshold`` (capital flow bullish or neutral).
- Bollinger / Fib width filter (skip dead-volatility regimes).
- Volume ≥ ``volume_min_ratio`` × SMA(volume).
- Confirmation candle must close above trigger close.
- Reward-to-risk ≥ ``min_rr_ratio``.
- TP edge must cover ``min_fee_cover_multiple`` × (2·maker + slippage).

Exits:

- TP1 (scale-out): fraction of distance to mid-band; stop moves to
  break-even on TP1 hit to lock in maker-fee coverage and turn many
  drifting trades into zero-loss trades.
- TP2 (final): Bollinger mid-band.
- Stop: ``atr_stop_multiple`` × ATR(14).
- Time stop: ``time_stop_candles`` bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from strategy.base import Strategy, TradeSignal
from strategy.filters import (
	bollinger_width_ok,
	fee_edge_ok,
	mfi_bearish,
	mfi_bullish,
	rr_ratio_ok,
	rsi_crossback_long,
	rsi_crossback_short,
	volume_ok,
	wavetrend_cross_down,
	wavetrend_cross_up,
)
from strategy.indicators import (
	atr,
	bollinger_bands,
	ema,
	fib_bands,
	money_flow_index,
	rsi,
	sma,
	wavetrend,
)


@dataclass(frozen=True)
class CipherConfig:
	"""Typed container for Cipher Confluence parameters."""

	bb_period: int
	bb_std_dev: float
	fib_mult: float
	rsi_period: int
	rsi_oversold: float
	rsi_overbought: float
	rsi_crossback_min_delta: float
	ema_period: int
	ema_side_bias_max_atr_distance: float
	atr_period: int
	atr_stop_multiple: float
	min_bandwidth_pct: float
	max_bandwidth_pct: float
	volume_sma_period: int
	volume_min_ratio: float
	wt_channel_len: int
	wt_average_len: int
	wt_signal_len: int
	wt_oversold: float
	wt_oversold_deep: float
	wt_overbought: float
	wt_overbought_deep: float
	mfi_period: int
	mfi_bull_threshold: float
	mfi_bear_threshold: float
	require_confirmation: bool
	require_extreme_zone: bool
	partial_tp_enabled: bool
	tp1_fraction_to_mid: float
	tp1_size_fraction: float
	tp2_fraction_to_mid: float
	tp2_size_fraction: float
	move_stop_to_breakeven: bool
	min_rr_ratio: float
	min_fee_cover_multiple: float
	maker_fee: float
	taker_fee: float
	estimated_slippage_rate: float


class CipherConfluenceStrategy(Strategy):
	"""Confluence mean-reversion strategy combining WT + MFI + RSI + Fib."""

	def __init__(self, config: Dict[str, object]) -> None:
		"""Initialize from the global config dict.

		Args:
			config: Full application configuration.
		"""
		s = config["strategy"]
		fees_cfg = config.get("fees", {"maker": 0.0001, "taker": 0.0005})
		exec_cfg = config.get("execution", {})
		slip_cfg = exec_cfg.get("slippage", {})
		tick_size = float(config.get("exchange", {}).get("tick_size", 0.1))
		ticks = int(slip_cfg.get("base_ticks_per_fill", 1))
		approx_slip = 2.0 * ticks * tick_size / 70000.0

		partial = s["exits"].get("partial_tp", {})
		sig_q = s.get("signal_quality", {})
		vol_f = s.get("volume_filter", {})
		conf = s.get("confirmation", {})
		wt = s.get("wavetrend", {})
		mfi_cfg = s.get("mfi", {})
		fib = s.get("fib", {})

		self.cfg = CipherConfig(
			bb_period=int(s["bollinger"]["period"]),
			bb_std_dev=float(s["bollinger"]["std_dev"]),
			fib_mult=float(fib.get("mult", 2.618)),
			rsi_period=int(s["rsi"]["period"]),
			rsi_oversold=float(s["rsi"]["oversold"]),
			rsi_overbought=float(s["rsi"]["overbought"]),
			rsi_crossback_min_delta=float(s["rsi"].get("crossback_min_delta", 0.0)),
			ema_period=int(s["ema"]["period"]),
			ema_side_bias_max_atr_distance=float(s["ema"].get("side_bias_max_atr_distance", 1e9)),
			atr_period=int(s["atr"]["period"]),
			atr_stop_multiple=float(s["atr"]["stop_multiple"]),
			min_bandwidth_pct=float(s["volatility_filter"]["min_bandwidth_pct"]),
			max_bandwidth_pct=float(s["volatility_filter"].get("max_bandwidth_pct", 1.0)),
			volume_sma_period=int(vol_f.get("sma_period", 20)),
			volume_min_ratio=float(vol_f.get("min_ratio", 0.0)),
			wt_channel_len=int(wt.get("channel_len", 9)),
			wt_average_len=int(wt.get("average_len", 12)),
			wt_signal_len=int(wt.get("signal_len", 3)),
			wt_oversold=float(wt.get("oversold", -53.0)),
			wt_oversold_deep=float(wt.get("oversold_deep", -75.0)),
			wt_overbought=float(wt.get("overbought", 53.0)),
			wt_overbought_deep=float(wt.get("overbought_deep", 75.0)),
			mfi_period=int(mfi_cfg.get("period", 14)),
			mfi_bull_threshold=float(mfi_cfg.get("bull_threshold", 50.0)),
			mfi_bear_threshold=float(mfi_cfg.get("bear_threshold", 50.0)),
			require_confirmation=bool(conf.get("require_next_candle_close_confirmation", True)),
			require_extreme_zone=bool(s.get("require_extreme_zone", True)),
			partial_tp_enabled=bool(partial.get("enabled", True)),
			tp1_fraction_to_mid=float(partial.get("tp1_fraction_to_mid", 0.5)),
			tp1_size_fraction=float(partial.get("tp1_size_fraction", 0.6)),
			tp2_fraction_to_mid=float(partial.get("tp2_fraction_to_mid", 0.7)),
			tp2_size_fraction=float(partial.get("tp2_size_fraction", 0.0)),
			move_stop_to_breakeven=bool(partial.get("move_stop_to_breakeven", True)),
			min_rr_ratio=float(sig_q.get("min_rr_ratio", 1.0)),
			min_fee_cover_multiple=float(sig_q.get("min_fee_cover_multiple", 0.0)),
			maker_fee=float(fees_cfg.get("maker", 0.0001)),
			taker_fee=float(fees_cfg.get("taker", 0.0005)),
			estimated_slippage_rate=approx_slip,
		)

	def prepare(self, candles: pd.DataFrame) -> pd.DataFrame:
		"""Compute and attach all indicator columns.

		Args:
			candles: OHLCV dataframe.

		Returns:
			Dataframe with indicator columns.
		"""
		f = candles.copy()
		close = f["close"].to_numpy(dtype=float)
		high = f["high"].to_numpy(dtype=float)
		low = f["low"].to_numpy(dtype=float)
		vol = (
			f["volume"].to_numpy(dtype=float)
			if "volume" in f.columns
			else np.zeros_like(close)
		)

		mid, up, lo = bollinger_bands(close, period=self.cfg.bb_period, std_dev=self.cfg.bb_std_dev)
		f["bb_middle"] = mid
		f["bb_upper"] = up
		f["bb_lower"] = lo
		f["bb_width_pct"] = (up - lo) / close

		fib_m, fib_u, fib_l = fib_bands(close, period=self.cfg.bb_period, mult=self.cfg.fib_mult)
		f["fib_middle"] = fib_m
		f["fib_upper"] = fib_u
		f["fib_lower"] = fib_l

		f["rsi"] = rsi(close, period=self.cfg.rsi_period)
		f["ema"] = ema(close, period=self.cfg.ema_period)
		f["atr"] = atr(high, low, close, period=self.cfg.atr_period)

		wt1, wt2 = wavetrend(
			high,
			low,
			close,
			channel_len=self.cfg.wt_channel_len,
			average_len=self.cfg.wt_average_len,
			signal_len=self.cfg.wt_signal_len,
		)
		f["wt1"] = wt1
		f["wt2"] = wt2

		f["mfi"] = money_flow_index(high, low, close, vol, period=self.cfg.mfi_period)
		f["vol_sma"] = sma(vol, period=self.cfg.volume_sma_period)
		return f

	def generate_signal(self, candles: pd.DataFrame, index: int) -> Optional[TradeSignal]:
		"""Generate confluence signal where ``index`` is confirmation bar.

		Args:
			candles: Prepared dataframe (output of ``prepare``).
			index: Confirmation candle index.

		Returns:
			TradeSignal if all gates pass, else None.
		"""
		if index <= 1 or index >= len(candles):
			return None

		trigger = candles.iloc[index - 1]
		confirm = candles.iloc[index]

		required_keys = [
			"bb_lower",
			"bb_upper",
			"bb_middle",
			"bb_width_pct",
			"fib_lower",
			"fib_upper",
			"rsi",
			"atr",
			"wt1",
			"wt2",
			"mfi",
		]
		for key in required_keys:
			if pd.isna(trigger[key]) or pd.isna(confirm[key]):
				return None

		if not bollinger_width_ok(
			width_pct=float(trigger["bb_width_pct"]),
			min_width_pct=self.cfg.min_bandwidth_pct,
			max_width_pct=self.cfg.max_bandwidth_pct,
		):
			return None

		if self.cfg.volume_min_ratio > 0.0:
			avg_v = float(trigger.get("vol_sma", np.nan))
			cur_v = float(trigger.get("volume", 0.0))
			if np.isnan(avg_v) or not volume_ok(cur_v, avg_v, self.cfg.volume_min_ratio):
				return None

		long_sig = self._long_signal(trigger, confirm)
		if long_sig is not None:
			return long_sig
		return self._short_signal(trigger, confirm)

	def _long_signal(self, trig: pd.Series, conf: pd.Series) -> Optional[TradeSignal]:
		wt_cross = wavetrend_cross_up(
			wt1_prev=float(trig["wt1"]),
			wt2_prev=float(trig["wt2"]),
			wt1_now=float(conf["wt1"]),
			wt2_now=float(conf["wt2"]),
		)
		if not wt_cross:
			return None

		wt2_trig = float(trig["wt2"])
		if wt2_trig > self.cfg.wt_oversold:
			return None  # WT must be in oversold zone at cross

		atr_val = float(trig["atr"])
		if atr_val <= 0.0:
			return None

		# Trend side-bias: don't chase longs deep below EMA(200).
		ema_val = float(trig["ema"]) if not pd.isna(trig.get("ema")) else None
		price = float(trig["close"])
		if ema_val is not None and ema_val > 0 and atr_val > 0:
			if (ema_val - price) / atr_val > self.cfg.ema_side_bias_max_atr_distance:
				return None

		# Extreme-zone requirement: price at/under lower BB or lower Fib band.
		touched_lower = (
			float(trig["low"]) <= float(trig["bb_lower"])
			or float(trig["low"]) <= float(trig["fib_lower"])
		)
		gold = wt2_trig <= self.cfg.wt_oversold_deep
		rsi_prev = float(trig["rsi"])
		rsi_now = float(conf["rsi"])
		rsi_ok = rsi_crossback_long(
			rsi_prev=rsi_prev,
			rsi_now=rsi_now,
			oversold=self.cfg.rsi_oversold,
			min_delta=self.cfg.rsi_crossback_min_delta,
		)
		mfi_val = float(conf["mfi"])
		mfi_prev = float(trig["mfi"]) if not pd.isna(trig.get("mfi")) else mfi_val
		mfi_rising = mfi_val > mfi_prev
		mfi_deep_oversold = mfi_val <= 45.0
		if self.cfg.require_extreme_zone:
			# Need at least 2 of {extreme-zone touch, RSI crossback, gold WT,
			# MFI deeply oversold + rising}. This 4-way confluence keeps
			# conviction high while allowing more distinct setups.
			mfi_signal = mfi_deep_oversold and mfi_rising
			strength = int(touched_lower) + int(rsi_ok) + int(gold) + int(mfi_signal)
			if strength < 2:
				return None
		else:
			if not (touched_lower or rsi_ok or gold):
				return None

		# Baseline MFI support (unless gold setup relaxes it).
		if not gold:
			if not mfi_bullish(mfi_val, self.cfg.mfi_bull_threshold):
				return None
		else:
			if mfi_val < 35.0:
				return None

		if self.cfg.require_confirmation and float(conf["close"]) <= float(trig["close"]):
			return None

		entry = float(conf["close"])
		# Tight structural stop: below trigger low (swing) OR 1.3*ATR, whichever closer.
		swing_stop = float(trig["low"]) - 0.25 * atr_val
		atr_stop = entry - self.cfg.atr_stop_multiple * atr_val
		stop = max(swing_stop, atr_stop)  # for long, higher stop = tighter
		if stop >= entry:
			stop = entry - max(0.5 * atr_val, 1e-8)
		# Dynamic TP: at least mid-band, extended to guarantee RR/fee-edge.
		mid_tp = float(trig["bb_middle"])
		risk = entry - stop
		rr_tp = entry + self.cfg.min_rr_ratio * risk
		round_trip = self.cfg.maker_fee * 2.0 + self.cfg.estimated_slippage_rate
		fee_tp = entry * (1.0 + self.cfg.min_fee_cover_multiple * round_trip)
		take_profit = max(mid_tp, rr_tp, fee_tp)
		# But cap TP so it's achievable (<= fib_upper).
		fib_upper = float(trig.get("fib_upper", take_profit))
		if not np.isnan(fib_upper) and take_profit > fib_upper:
			take_profit = fib_upper
		if take_profit <= entry:
			return None
		return self._finalize(
			side="long",
			trig=trig,
			conf=conf,
			entry=entry,
			stop=stop,
			take_profit=take_profit,
		)

	def _short_signal(self, trig: pd.Series, conf: pd.Series) -> Optional[TradeSignal]:
		wt_cross = wavetrend_cross_down(
			wt1_prev=float(trig["wt1"]),
			wt2_prev=float(trig["wt2"]),
			wt1_now=float(conf["wt1"]),
			wt2_now=float(conf["wt2"]),
		)
		if not wt_cross:
			return None

		wt2_trig = float(trig["wt2"])
		if wt2_trig < self.cfg.wt_overbought:
			return None

		atr_val = float(trig["atr"])
		if atr_val <= 0.0:
			return None

		ema_val = float(trig["ema"]) if not pd.isna(trig.get("ema")) else None
		price = float(trig["close"])
		if ema_val is not None and ema_val > 0 and atr_val > 0:
			if (price - ema_val) / atr_val > self.cfg.ema_side_bias_max_atr_distance:
				return None

		touched_upper = (
			float(trig["high"]) >= float(trig["bb_upper"])
			or float(trig["high"]) >= float(trig["fib_upper"])
		)
		deep = wt2_trig >= self.cfg.wt_overbought_deep
		rsi_prev = float(trig["rsi"])
		rsi_now = float(conf["rsi"])
		rsi_ok = rsi_crossback_short(
			rsi_prev=rsi_prev,
			rsi_now=rsi_now,
			overbought=self.cfg.rsi_overbought,
			min_delta=self.cfg.rsi_crossback_min_delta,
		)
		mfi_val = float(conf["mfi"])
		mfi_prev = float(trig["mfi"]) if not pd.isna(trig.get("mfi")) else mfi_val
		mfi_falling = mfi_val < mfi_prev
		mfi_deep_overbought = mfi_val >= 55.0
		if self.cfg.require_extreme_zone:
			mfi_signal = mfi_deep_overbought and mfi_falling
			strength = int(touched_upper) + int(rsi_ok) + int(deep) + int(mfi_signal)
			if strength < 2:
				return None
		else:
			if not (touched_upper or rsi_ok or deep):
				return None

		if not deep:
			if not mfi_bearish(mfi_val, self.cfg.mfi_bear_threshold):
				return None
		else:
			if mfi_val > 65.0:
				return None

		if self.cfg.require_confirmation and float(conf["close"]) >= float(trig["close"]):
			return None

		entry = float(conf["close"])
		swing_stop = float(trig["high"]) + 0.25 * atr_val
		atr_stop = entry + self.cfg.atr_stop_multiple * atr_val
		stop = min(swing_stop, atr_stop)
		if stop <= entry:
			stop = entry + max(0.5 * atr_val, 1e-8)
		mid_tp = float(trig["bb_middle"])
		risk = stop - entry
		rr_tp = entry - self.cfg.min_rr_ratio * risk
		round_trip = self.cfg.maker_fee * 2.0 + self.cfg.estimated_slippage_rate
		fee_tp = entry * (1.0 - self.cfg.min_fee_cover_multiple * round_trip)
		take_profit = min(mid_tp, rr_tp, fee_tp)
		fib_lower = float(trig.get("fib_lower", take_profit))
		if not np.isnan(fib_lower) and take_profit < fib_lower:
			take_profit = fib_lower
		if take_profit >= entry:
			return None
		return self._finalize(
			side="short",
			trig=trig,
			conf=conf,
			entry=entry,
			stop=stop,
			take_profit=take_profit,
		)

	def _finalize(
		self,
		side: str,
		trig: pd.Series,
		conf: pd.Series,
		entry: float,
		stop: float,
		take_profit: float,
	) -> Optional[TradeSignal]:
		"""Apply RR / fee-edge gates, build TradeSignal with optional TP1."""
		if not rr_ratio_ok(entry=entry, stop=stop, take_profit=take_profit, min_ratio=self.cfg.min_rr_ratio):
			return None
		round_trip_fee = self.cfg.maker_fee * 2.0
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
		tp2_price: Optional[float] = None
		tp2_size = 0.0
		if self.cfg.partial_tp_enabled:
			frac = max(0.0, min(1.0, self.cfg.tp1_fraction_to_mid))
			tp1_price = entry + frac * (take_profit - entry)
			tp1_size = max(0.0, min(1.0, self.cfg.tp1_size_fraction))
			frac2 = max(frac, min(1.0, self.cfg.tp2_fraction_to_mid))
			if self.cfg.tp2_size_fraction > 0.0 and frac2 > frac:
				tp2_price = entry + frac2 * (take_profit - entry)
				tp2_size = max(0.0, min(1.0 - tp1_size, self.cfg.tp2_size_fraction))

		return TradeSignal(
			side=side,
			reference_index=int(trig.name),
			entry_index=int(conf.name),
			entry_price=entry,
			stop_price=stop,
			take_profit_price=take_profit,
			tp1_price=tp1_price,
			tp1_size_fraction=tp1_size,
			tp2_price=tp2_price,
			tp2_size_fraction=tp2_size,
			move_stop_to_breakeven_after_tp1=self.cfg.move_stop_to_breakeven,
		)
