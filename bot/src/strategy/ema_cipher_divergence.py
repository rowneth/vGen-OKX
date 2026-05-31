"""EMA 200/50/55 + VuManchu Cipher B + WT-Money-Flow trend-pullback strategy.

Implements the 5m scalping spec:

LONG sequence (short mirrored):

1. Bullish EMA cross: EMA50 crossed above EMA200 within the last
   ``cross_lookback_bars``. Don't enter the cross bar itself — wait for
   retracement.
2. Retracement: price touched the EMA50/55 zone within the last
   ``retrace_lookback_bars`` (with an ATR-scaled tolerance), and the
   trigger bar has reclaimed price above the zone (or wicked below and
   recovered).
3. Money Flow cloud GREEN at the confirmation bar — WT-style MFI built
   from ``sma((close-open)/(high-low) * mult, period)`` is positive.
4. WT bullish cross (small green dot): WT1 crosses above WT2 between
   the trigger and confirmation bars while WT2 is below the
   ``wt_cross_zero_threshold`` (bullish crosses happen below zero).
5. Optional bullish divergence on WT2 within the last
   ``divergence_lookback_bars`` (price lower low, WT2 higher low).
   Required when ``require_divergence`` is True; otherwise it just adds
   conviction.
6. Confirmation candle closes bullish: close > open and close > trigger
   close.
7. Stop = swing low over the last ``swing_lookback_bars`` minus a
   buffer (``stop_buffer_atr`` × ATR).
8. Take-profit = entry + ``tp_rr_ratio`` × risk. RR must clear
   ``min_rr_ratio`` and the fee/slippage edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from strategy.base import Strategy, TradeSignal
from strategy.filters import (
	fee_edge_ok,
	rr_ratio_ok,
	wavetrend_cross_down,
	wavetrend_cross_up,
)
from strategy.indicators import atr, ema, sma, wavetrend


@dataclass(frozen=True)
class EmaCipherConfig:
	"""Typed parameters for the EMA + Cipher B trend-pullback strategy."""

	ema_fast: int
	ema_mid: int
	ema_slow: int
	wt_channel_len: int
	wt_average_len: int
	wt_signal_len: int
	wt_cross_zero_threshold: float
	wt_long_zone_max: float
	wt_short_zone_min: float
	mfi_period: int
	mfi_multiplier: float
	mfi_pos_y: float
	atr_period: int
	stop_buffer_atr: float
	swing_lookback_bars: int
	cross_lookback_bars: int
	retrace_lookback_bars: int
	retrace_tolerance_atr: float
	divergence_lookback_bars: int
	divergence_fractal_left: int
	divergence_fractal_right: int
	require_divergence: bool
	require_bullish_body: bool
	tp_rr_ratio: float
	min_rr_ratio: float
	min_fee_cover_multiple: float
	maker_fee: float
	taker_fee: float
	estimated_slippage_rate: float
	partial_tp_enabled: bool
	tp1_size_fraction: float
	tp1_rr_fraction: float
	move_stop_to_breakeven: bool
	allow_long: bool
	allow_short: bool


def wt_money_flow(
	open_: np.ndarray,
	high: np.ndarray,
	low: np.ndarray,
	close: np.ndarray,
	period: int,
	multiplier: float,
	pos_y: float,
) -> np.ndarray:
	"""WT-style money-flow oscillator (sign = cloud color).

	Replicates VuManchu's ``f_rsimfi``:
	``sma((close-open)/(high-low) * multiplier, period) - pos_y``.

	Returns:
		Array aligned to inputs; positive = green cloud, negative = red.
	"""
	rng = high - low
	with np.errstate(divide="ignore", invalid="ignore"):
		raw = np.where(rng > 0.0, (close - open_) / rng, 0.0) * multiplier
	raw = np.where(np.isfinite(raw), raw, 0.0)
	smoothed = sma(raw, period)
	return smoothed - pos_y


def _find_bullish_divergence(
	low_prices: np.ndarray,
	wt2: np.ndarray,
	index: int,
	lookback: int,
	left: int,
	right: int,
) -> bool:
	"""Detect bullish WT divergence ending at or before ``index``.

	A bullish divergence is two consecutive bottom fractals where:
		- The newer bar has a strictly lower low than the older bar.
		- The newer bar has a strictly higher WT2 than the older bar.

	A bottom fractal at bar ``b`` requires ``left`` bars to its left and
	``right`` bars to its right to all have higher WT2 values.
	"""
	span_end = index - right
	span_start = max(left, span_end - lookback)
	if span_end <= span_start:
		return False
	fractals = []
	for b in range(span_start, span_end + 1):
		center = wt2[b]
		if not np.isfinite(center):
			continue
		ok = True
		for k in range(1, left + 1):
			if not np.isfinite(wt2[b - k]) or wt2[b - k] <= center:
				ok = False
				break
		if not ok:
			continue
		for k in range(1, right + 1):
			if not np.isfinite(wt2[b + k]) or wt2[b + k] <= center:
				ok = False
				break
		if ok:
			fractals.append(b)
	if len(fractals) < 2:
		return False
	prev_b, curr_b = fractals[-2], fractals[-1]
	return low_prices[curr_b] < low_prices[prev_b] and wt2[curr_b] > wt2[prev_b]


def _find_bearish_divergence(
	high_prices: np.ndarray,
	wt2: np.ndarray,
	index: int,
	lookback: int,
	left: int,
	right: int,
) -> bool:
	"""Detect bearish WT divergence ending at or before ``index``.

	Symmetric to ``_find_bullish_divergence``: two top fractals where
	price made a higher high but WT2 made a lower high.
	"""
	span_end = index - right
	span_start = max(left, span_end - lookback)
	if span_end <= span_start:
		return False
	fractals = []
	for b in range(span_start, span_end + 1):
		center = wt2[b]
		if not np.isfinite(center):
			continue
		ok = True
		for k in range(1, left + 1):
			if not np.isfinite(wt2[b - k]) or wt2[b - k] >= center:
				ok = False
				break
		if not ok:
			continue
		for k in range(1, right + 1):
			if not np.isfinite(wt2[b + k]) or wt2[b + k] >= center:
				ok = False
				break
		if ok:
			fractals.append(b)
	if len(fractals) < 2:
		return False
	prev_b, curr_b = fractals[-2], fractals[-1]
	return high_prices[curr_b] > high_prices[prev_b] and wt2[curr_b] < wt2[prev_b]


class EmaCipherDivergenceStrategy(Strategy):
	"""EMA 200/50/55 trend filter + WaveTrend + WT-MFI confluence."""

	def __init__(self, config: Dict[str, object]) -> None:
		s = config["strategy"]
		fees_cfg = config.get("fees", {"maker": 0.0001, "taker": 0.0005})
		exec_cfg = config.get("execution", {})
		slip_cfg = exec_cfg.get("slippage", {})
		tick_size = float(config.get("exchange", {}).get("tick_size", 0.1))
		ticks = int(slip_cfg.get("base_ticks_per_fill", 1))
		# Approximate slippage rate at BTC ~ $70k; only used for the fee
		# edge gate, so exactness isn't critical.
		approx_slip = 2.0 * ticks * tick_size / 70000.0

		emas = s.get("emas", {})
		wt = s.get("wavetrend", {})
		wt_zone = s.get("wavetrend_zone", {})
		mfi = s.get("money_flow", {})
		div = s.get("divergence", {})
		retrace = s.get("retracement", {})
		stop_cfg = s.get("stop", {})
		tp_cfg = s.get("take_profit", {})
		sig_q = s.get("signal_quality", {})
		partial = s.get("exits", {}).get("partial_tp", {})

		self.cfg = EmaCipherConfig(
			ema_fast=int(emas.get("fast", 50)),
			ema_mid=int(emas.get("mid", 55)),
			ema_slow=int(emas.get("slow", 200)),
			wt_channel_len=int(wt.get("channel_len", 9)),
			wt_average_len=int(wt.get("average_len", 12)),
			wt_signal_len=int(wt.get("signal_len", 3)),
			wt_cross_zero_threshold=float(wt.get("cross_zero_threshold", 0.0)),
			wt_long_zone_max=float(wt_zone.get("long_wt2_max", 0.0)),
			wt_short_zone_min=float(wt_zone.get("short_wt2_min", 0.0)),
			mfi_period=int(mfi.get("period", 60)),
			mfi_multiplier=float(mfi.get("multiplier", 150.0)),
			mfi_pos_y=float(mfi.get("pos_y", 2.5)),
			atr_period=int(s.get("atr", {}).get("period", 14)),
			stop_buffer_atr=float(stop_cfg.get("buffer_atr", 0.25)),
			swing_lookback_bars=int(stop_cfg.get("swing_lookback_bars", 10)),
			cross_lookback_bars=int(emas.get("cross_lookback_bars", 50)),
			retrace_lookback_bars=int(retrace.get("lookback_bars", 20)),
			retrace_tolerance_atr=float(retrace.get("tolerance_atr", 0.25)),
			divergence_lookback_bars=int(div.get("lookback_bars", 30)),
			divergence_fractal_left=int(div.get("fractal_left", 2)),
			divergence_fractal_right=int(div.get("fractal_right", 2)),
			require_divergence=bool(div.get("required", False)),
			require_bullish_body=bool(s.get("confirmation", {}).get("require_directional_body", True)),
			tp_rr_ratio=float(tp_cfg.get("rr_ratio", 2.0)),
			min_rr_ratio=float(sig_q.get("min_rr_ratio", 2.0)),
			min_fee_cover_multiple=float(sig_q.get("min_fee_cover_multiple", 0.0)),
			maker_fee=float(fees_cfg.get("maker", 0.0001)),
			taker_fee=float(fees_cfg.get("taker", 0.0005)),
			estimated_slippage_rate=approx_slip,
			partial_tp_enabled=bool(partial.get("enabled", False)),
			tp1_size_fraction=float(partial.get("tp1_size_fraction", 0.5)),
			tp1_rr_fraction=float(partial.get("tp1_rr_fraction", 0.5)),
			move_stop_to_breakeven=bool(partial.get("move_stop_to_breakeven", True)),
			allow_long=bool(s.get("sides", {}).get("long", True)),
			allow_short=bool(s.get("sides", {}).get("short", True)),
		)

	def prepare(self, candles: pd.DataFrame) -> pd.DataFrame:
		f = candles.copy()
		close = f["close"].to_numpy(dtype=float)
		open_ = f["open"].to_numpy(dtype=float)
		high = f["high"].to_numpy(dtype=float)
		low = f["low"].to_numpy(dtype=float)

		f["ema_fast"] = ema(close, period=self.cfg.ema_fast)
		f["ema_mid"] = ema(close, period=self.cfg.ema_mid)
		f["ema_slow"] = ema(close, period=self.cfg.ema_slow)
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

		f["wt_mfi"] = wt_money_flow(
			open_=open_,
			high=high,
			low=low,
			close=close,
			period=self.cfg.mfi_period,
			multiplier=self.cfg.mfi_multiplier,
			pos_y=self.cfg.mfi_pos_y,
		)
		return f

	def generate_signal(self, candles: pd.DataFrame, index: int) -> Optional[TradeSignal]:
		# Skip warmup. Use 220 (EMA200 + small buffer) plus divergence
		# fractal headroom. ``cross_lookback_bars`` is intentionally not
		# part of this — it's a soft "how far back to search" and may be
		# set to a very large sentinel (e.g. 9999 = persistent regime).
		warmup_needed = max(220, self.cfg.divergence_lookback_bars + 5)
		if index <= warmup_needed:
			return None
		if index >= len(candles) - 1:
			return None

		trigger = candles.iloc[index - 1]
		confirm = candles.iloc[index]

		required = ("ema_fast", "ema_mid", "ema_slow", "atr", "wt1", "wt2", "wt_mfi")
		for key in required:
			if pd.isna(trigger[key]) or pd.isna(confirm[key]):
				return None

		if self.cfg.allow_long:
			long_sig = self._long_signal(candles, index, trigger, confirm)
			if long_sig is not None:
				return long_sig
		if self.cfg.allow_short:
			return self._short_signal(candles, index, trigger, confirm)
		return None

	def _long_signal(
		self,
		candles: pd.DataFrame,
		index: int,
		trig: pd.Series,
		conf: pd.Series,
	) -> Optional[TradeSignal]:
		ema_fast_conf = float(conf["ema_fast"])
		ema_slow_conf = float(conf["ema_slow"])
		if ema_fast_conf <= ema_slow_conf:
			return None

		# Fresh bullish cross within lookback: EMA50 was at/below EMA200,
		# then crossed above. We don't insist on cross at bar i — only
		# that the regime transitioned recently.
		ema_fast_series = candles["ema_fast"].to_numpy(dtype=float)
		ema_slow_series = candles["ema_slow"].to_numpy(dtype=float)
		start = max(1, index - self.cfg.cross_lookback_bars)
		had_cross = False
		for b in range(start, index + 1):
			prev = b - 1
			if (
				np.isfinite(ema_fast_series[prev])
				and np.isfinite(ema_slow_series[prev])
				and np.isfinite(ema_fast_series[b])
				and np.isfinite(ema_slow_series[b])
				and ema_fast_series[prev] <= ema_slow_series[prev]
				and ema_fast_series[b] > ema_slow_series[b]
			):
				had_cross = True
				break
		if not had_cross:
			return None

		atr_val = float(trig["atr"])
		if atr_val <= 0.0:
			return None

		# Retracement: a recent bar wicked into the EMA50/55 zone (with
		# ATR-scaled tolerance), and the trigger bar's body sits at or
		# above that zone (price has bounced back up).
		ema_mid_series = candles["ema_mid"].to_numpy(dtype=float)
		lows = candles["low"].to_numpy(dtype=float)
		closes = candles["close"].to_numpy(dtype=float)
		retrace_start = max(0, index - self.cfg.retrace_lookback_bars)
		touched = False
		for b in range(retrace_start, index + 1):
			lo = lows[b]
			zone_top = max(ema_fast_series[b], ema_mid_series[b]) + self.cfg.retrace_tolerance_atr * atr_val
			zone_bot = min(ema_fast_series[b], ema_mid_series[b]) - self.cfg.retrace_tolerance_atr * atr_val
			if lo <= zone_top and lo >= zone_bot - 5.0 * atr_val:
				touched = True
				break
		if not touched:
			return None
		# Trigger bar must have reclaimed the zone — close above EMA55.
		if closes[index - 1] < float(trig["ema_mid"]) - self.cfg.retrace_tolerance_atr * atr_val:
			return None

		# Money Flow cloud must be green (positive) on confirmation bar.
		if float(conf["wt_mfi"]) <= 0.0:
			return None

		# WT bullish cross between trigger and confirm, below the
		# zero-line threshold (the "green dot" condition).
		if not wavetrend_cross_up(
			wt1_prev=float(trig["wt1"]),
			wt2_prev=float(trig["wt2"]),
			wt1_now=float(conf["wt1"]),
			wt2_now=float(conf["wt2"]),
		):
			return None
		if float(trig["wt2"]) > self.cfg.wt_cross_zero_threshold:
			return None
		if float(trig["wt2"]) > self.cfg.wt_long_zone_max:
			return None

		# Optional bullish divergence gate.
		if self.cfg.require_divergence:
			wt2_arr = candles["wt2"].to_numpy(dtype=float)
			low_arr = candles["low"].to_numpy(dtype=float)
			if not _find_bullish_divergence(
				low_prices=low_arr,
				wt2=wt2_arr,
				index=index,
				lookback=self.cfg.divergence_lookback_bars,
				left=self.cfg.divergence_fractal_left,
				right=self.cfg.divergence_fractal_right,
			):
				return None

		# Confirmation candle: bullish body, closing above trigger close.
		if self.cfg.require_bullish_body and float(conf["close"]) <= float(conf["open"]):
			return None
		if float(conf["close"]) <= float(trig["close"]):
			return None

		entry = float(conf["close"])
		highs = candles["high"].to_numpy(dtype=float)  # noqa: F841 — kept for symmetry
		swing_start = max(0, index - self.cfg.swing_lookback_bars)
		swing_low = float(np.min(lows[swing_start : index + 1]))
		stop = swing_low - self.cfg.stop_buffer_atr * atr_val
		if stop >= entry:
			stop = entry - max(0.5 * atr_val, 1e-8)
		risk = entry - stop
		take_profit = entry + self.cfg.tp_rr_ratio * risk
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

	def _short_signal(
		self,
		candles: pd.DataFrame,
		index: int,
		trig: pd.Series,
		conf: pd.Series,
	) -> Optional[TradeSignal]:
		ema_fast_conf = float(conf["ema_fast"])
		ema_slow_conf = float(conf["ema_slow"])
		if ema_fast_conf >= ema_slow_conf:
			return None

		ema_fast_series = candles["ema_fast"].to_numpy(dtype=float)
		ema_slow_series = candles["ema_slow"].to_numpy(dtype=float)
		start = max(1, index - self.cfg.cross_lookback_bars)
		had_cross = False
		for b in range(start, index + 1):
			prev = b - 1
			if (
				np.isfinite(ema_fast_series[prev])
				and np.isfinite(ema_slow_series[prev])
				and np.isfinite(ema_fast_series[b])
				and np.isfinite(ema_slow_series[b])
				and ema_fast_series[prev] >= ema_slow_series[prev]
				and ema_fast_series[b] < ema_slow_series[b]
			):
				had_cross = True
				break
		if not had_cross:
			return None

		atr_val = float(trig["atr"])
		if atr_val <= 0.0:
			return None

		ema_mid_series = candles["ema_mid"].to_numpy(dtype=float)
		highs = candles["high"].to_numpy(dtype=float)
		closes = candles["close"].to_numpy(dtype=float)
		retrace_start = max(0, index - self.cfg.retrace_lookback_bars)
		touched = False
		for b in range(retrace_start, index + 1):
			hi = highs[b]
			zone_top = max(ema_fast_series[b], ema_mid_series[b]) + self.cfg.retrace_tolerance_atr * atr_val
			zone_bot = min(ema_fast_series[b], ema_mid_series[b]) - self.cfg.retrace_tolerance_atr * atr_val
			if hi >= zone_bot and hi <= zone_top + 5.0 * atr_val:
				touched = True
				break
		if not touched:
			return None
		if closes[index - 1] > float(trig["ema_mid"]) + self.cfg.retrace_tolerance_atr * atr_val:
			return None

		# Money Flow cloud must be red (negative).
		if float(conf["wt_mfi"]) >= 0.0:
			return None

		if not wavetrend_cross_down(
			wt1_prev=float(trig["wt1"]),
			wt2_prev=float(trig["wt2"]),
			wt1_now=float(conf["wt1"]),
			wt2_now=float(conf["wt2"]),
		):
			return None
		if float(trig["wt2"]) < -self.cfg.wt_cross_zero_threshold:
			return None
		if float(trig["wt2"]) < self.cfg.wt_short_zone_min:
			return None

		if self.cfg.require_divergence:
			wt2_arr = candles["wt2"].to_numpy(dtype=float)
			high_arr = candles["high"].to_numpy(dtype=float)
			if not _find_bearish_divergence(
				high_prices=high_arr,
				wt2=wt2_arr,
				index=index,
				lookback=self.cfg.divergence_lookback_bars,
				left=self.cfg.divergence_fractal_left,
				right=self.cfg.divergence_fractal_right,
			):
				return None

		if self.cfg.require_bullish_body and float(conf["close"]) >= float(conf["open"]):
			return None
		if float(conf["close"]) >= float(trig["close"]):
			return None

		entry = float(conf["close"])
		swing_start = max(0, index - self.cfg.swing_lookback_bars)
		swing_high = float(np.max(highs[swing_start : index + 1]))
		stop = swing_high + self.cfg.stop_buffer_atr * atr_val
		if stop <= entry:
			stop = entry + max(0.5 * atr_val, 1e-8)
		risk = stop - entry
		take_profit = entry - self.cfg.tp_rr_ratio * risk
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
		if not rr_ratio_ok(
			entry=entry,
			stop=stop,
			take_profit=take_profit,
			min_ratio=self.cfg.min_rr_ratio,
		):
			return None
		round_trip_fee = self.cfg.maker_fee + self.cfg.taker_fee
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
		if self.cfg.partial_tp_enabled and 0.0 < self.cfg.tp1_rr_fraction < 1.0:
			tp1_size = max(0.0, min(1.0, self.cfg.tp1_size_fraction))
			tp1_price = entry + self.cfg.tp1_rr_fraction * (take_profit - entry)

		return TradeSignal(
			side=side,
			reference_index=int(trig.name),
			entry_index=int(conf.name),
			entry_price=entry,
			stop_price=stop,
			take_profit_price=take_profit,
			tp1_price=tp1_price,
			tp1_size_fraction=tp1_size,
			move_stop_to_breakeven_after_tp1=self.cfg.move_stop_to_breakeven,
		)
