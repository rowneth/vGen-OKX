"""Multi-timeframe trend detector for the volume-farmer bot.

Pure-function module — no I/O, no state.  Intended to be imported by
``execution.volume_farmer`` to add a `multi_timeframe` entry mode without
touching any of the existing entry paths.

Algorithm (matches the spec the operator pasted):

* Resample the supplied 5m kline history to 15m bars.
* Compute SMA(5) on the 15m closes — the *short trend* on the higher TF.
* Compute SMA(5) on the 5m closes — the *short trend* on the entry TF.
* Up if last close > SMA, down if last close < SMA, else neutral.
* If *both* TFs agree, return that direction; otherwise return None
  (no trade — wait for alignment).

Why SMA(5) on a 5-bar window: it is the cheapest filter that still rejects
sideways chop without lagging too far on real moves.  Operator can swap
``lookback`` per config.

New API (v2): 3m micro + 15m macro alignment using structural trend detection.
See ``get_aligned_direction`` and helper functions below.  The old
``detect_direction`` function is preserved for backward-compatibility.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


def _resample_5m_to_15m(df_5m: pd.DataFrame) -> pd.DataFrame:
	"""Aggregate 5m OHLC bars to closed 15m bars.

	The most recent partial 15m bucket is dropped to avoid biasing on a
	still-forming bar.
	"""
	if df_5m.empty or "open_time" not in df_5m.columns:
		return df_5m.iloc[0:0]
	t = pd.to_datetime(df_5m["open_time"], unit="ms", utc=True)
	g = (
		df_5m.assign(_ts=t)
		.set_index("_ts")
		.resample("15min", label="left", closed="left")
		.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
		.dropna()
	)
	# Drop the trailing (possibly forming) 15m bucket.
	if len(g) > 0:
		g = g.iloc[:-1]
	g = g.reset_index().rename(columns={"_ts": "open_time"})
	g["open_time"] = (g["open_time"].astype("int64") // 1_000_000)
	return g


def _trend_direction(closes: pd.Series, lookback: int) -> str:
	"""Return 'up' / 'down' / 'neutral' for the latest close vs SMA."""
	if len(closes) < lookback + 1:
		return "neutral"
	sma = closes.rolling(lookback).mean().iloc[-1]
	last = float(closes.iloc[-1])
	if pd.isna(sma):
		return "neutral"
	if last > float(sma):
		return "up"
	if last < float(sma):
		return "down"
	return "neutral"


def detect_direction(
	history_5m: pd.DataFrame,
	lookback: int = 5,
	min_5m_bars: int = 20,
) -> Optional[str]:
	"""Return 'long' / 'short' / None based on aligned 5m + 15m trend.

	Args:
		history_5m: 5-minute OHLC dataframe with at least ``min_5m_bars`` rows.
		lookback: SMA window (default 5 bars on each TF).
		min_5m_bars: Refuse to signal until we have this much history.
	"""
	if history_5m is None or len(history_5m) < min_5m_bars:
		return None

	df_15m = _resample_5m_to_15m(history_5m)
	if len(df_15m) < lookback + 1:
		return None

	d_5m = _trend_direction(history_5m["close"].astype(float), lookback)
	d_15m = _trend_direction(df_15m["close"].astype(float), lookback)

	if d_5m == "up" and d_15m == "up":
		return "long"
	if d_5m == "down" and d_15m == "down":
		return "short"
	return None


# ──────────────────────────────────────────────────────────────────────────────
# New API (v2): 3m micro + 15m macro structural trend detection
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Bar:
	"""Minimal OHLCV bar for MTF computations."""
	open: float
	high: float
	low: float
	close: float
	volume: float
	timestamp_ms: int = 0


def detect_micro_trend(bars: Sequence[Bar], lookback: int = 5) -> str:
	"""Return 'up' / 'down' / 'neutral' from 3m bar micro-momentum.

	Combines last-bar direction with a short-vs-long SMA slope so a single
	outlier bar doesn't override a sideways market.
	"""
	if len(bars) < lookback:
		return "neutral"

	recent = list(bars[-lookback:])
	last_bar = recent[-1]
	if last_bar.close > last_bar.open:
		last_dir = "up"
	elif last_bar.close < last_bar.open:
		last_dir = "down"
	else:
		last_dir = "neutral"

	closes = [b.close for b in recent]
	sma_short = sum(closes[-3:]) / 3
	sma_long = sum(closes) / lookback
	ema_rising = sma_short > sma_long * 1.00005   # 0.5 bps buffer
	ema_falling = sma_short < sma_long * 0.99995

	if last_dir == "up" and ema_rising:
		return "up"
	if last_dir == "down" and ema_falling:
		return "down"
	return "neutral"


def detect_macro_trend(bars: Sequence[Bar], lookback: int = 5) -> str:
	"""Return 'up' / 'down' / 'neutral' from 15m bar structure.

	Uses higher-highs/higher-lows logic plus close position within range to
	give a 2-of-3 majority vote per direction.
	"""
	if len(bars) < lookback + 1:
		return "neutral"

	recent = list(bars[-(lookback + 1):])
	highs = [b.high for b in recent]
	lows = [b.low for b in recent]
	closes = [b.close for b in recent]

	last_high = highs[-1]
	last_low = lows[-1]
	prior_high = max(highs[:-1])
	prior_low = min(lows[:-1])

	higher_high = last_high > prior_high
	higher_low = last_low > prior_low
	lower_high = last_high < prior_high
	lower_low = last_low < prior_low

	range_high = max(highs)
	range_low = min(lows)
	if range_high > range_low:
		position_in_range = (closes[-1] - range_low) / (range_high - range_low)
	else:
		position_in_range = 0.5

	bullish = sum([higher_high, higher_low, position_in_range > 0.6])
	bearish = sum([lower_high, lower_low, position_in_range < 0.4])

	if bullish >= 2 and bullish > bearish:
		return "up"
	if bearish >= 2 and bearish > bullish:
		return "down"
	return "neutral"


def get_aligned_direction(
	bars_3m: Sequence[Bar],
	bars_15m: Sequence[Bar],
	allow_neutral_micro: bool = True,
	skip_neutral_macro: bool = True,
) -> Optional[str]:
	"""Return 'long' / 'short' / None.

	None means skip this entry — timeframes don't align or macro is choppy.
	"""
	micro = detect_micro_trend(bars_3m)
	macro = detect_macro_trend(bars_15m)

	if skip_neutral_macro and macro == "neutral":
		return None

	# Strong alignment: both agree
	if micro == "up" and macro == "up":
		return "long"
	if micro == "down" and macro == "down":
		return "short"

	# Acceptable: macro is trending, micro is neutral (not fighting macro)
	if allow_neutral_micro and micro == "neutral":
		if macro == "up":
			return "long"
		if macro == "down":
			return "short"

	# Conflict: micro fights macro — skip
	return None


# ──────────────────────────────────────────────────────────────────────────────
# DataFrame helpers — used internally by VolumeFarmerSession
# ──────────────────────────────────────────────────────────────────────────────

def _df_to_bars(df: pd.DataFrame) -> List[Bar]:
	"""Convert an OHLCV DataFrame to a list of Bar objects (row-order preserved)."""
	result: List[Bar] = []
	for _, row in df.iterrows():
		ts_raw = row.get("open_time", 0)
		# open_time may be int (ms), int64, or pd.Timestamp depending on parquet schema
		try:
			ts_ms = int(ts_raw)
		except (TypeError, ValueError):
			try:
				ts_ms = int(pd.Timestamp(ts_raw).timestamp() * 1000)
			except Exception:
				ts_ms = 0
		result.append(Bar(
			open=float(row.get("open", 0)),
			high=float(row.get("high", 0)),
			low=float(row.get("low", 0)),
			close=float(row.get("close", 0)),
			volume=float(row.get("volume", 0)),
			timestamp_ms=ts_ms,
		))
	return result


def _resample_df_to_nmin(df: pd.DataFrame, n_minutes: int) -> List[Bar]:
	"""Aggregate a DataFrame with integer-ms ``open_time`` to n-minute bars.

	The trailing (potentially forming) bucket is dropped so callers always
	see closed bars only.
	"""
	if df.empty or "open_time" not in df.columns:
		return []

	if pd.api.types.is_datetime64_any_dtype(df["open_time"]):
		t = df["open_time"]
		if getattr(t.dt, "tz", None) is None:
			t = t.dt.tz_localize("UTC")
	else:
		t = pd.to_datetime(df["open_time"], unit="ms", utc=True)
	agg_dict = {}
	for col, func in (
		("open", "first"), ("high", "max"), ("low", "min"),
		("close", "last"), ("volume", "sum"),
	):
		if col in df.columns:
			agg_dict[col] = func

	if not agg_dict:
		return []

	g = (
		df.assign(_ts=t)
		.set_index("_ts")
		.resample(f"{n_minutes}min", label="left", closed="left")
		.agg(agg_dict)
		.dropna(subset=["close"])
	)
	# Drop the last bucket (may be forming)
	if len(g) > 0:
		g = g.iloc[:-1]
	if g.empty:
		return []

	result: List[Bar] = []
	for ts, row in g.iterrows():
		result.append(Bar(
			open=float(row.get("open", 0)),
			high=float(row.get("high", 0)),
			low=float(row.get("low", 0)),
			close=float(row.get("close", 0)),
			volume=float(row.get("volume", 0)),
			timestamp_ms=int(ts.timestamp() * 1000),
		))
	return result


def get_micro_bars(
	history: pd.DataFrame,
	primary_tf: str,
	lookback: int,
) -> List[Bar]:
	"""Derive micro-trend bars from primary-TF history.

	For 1m data: aggregates the recent tail to 3m bars.
	For 5m (or any other TF): uses raw primary bars as the micro proxy
	(3m isn't derivable from 5m, so 5m bars serve as the micro signal).

	Only the most recent ``lookback + 2`` bars (post-aggregation) are returned
	to keep resample cost O(1) regardless of history length.
	"""
	n_out = lookback + 2
	if primary_tf == "1m":
		tail = history.iloc[-(n_out * 3):]
		return _resample_df_to_nmin(tail, 3)
	else:
		tail = history.iloc[-n_out:]
		return _df_to_bars(tail)


def get_macro_bars(
	history: pd.DataFrame,
	primary_tf: str,
	lookback: int,
) -> List[Bar]:
	"""Derive macro-trend (15m) bars from primary-TF history.

	For 1m data: aggregates from 1m to 15m.
	For 5m data: aggregates from 5m to 15m (3 × 5m = 15m).

	Only the most recent ``lookback + 2`` completed 15m bars are returned.
	"""
	n_out = lookback + 2
	if primary_tf == "1m":
		tail = history.iloc[-(n_out * 15):]
	else:
		# 3 × 5m bars per 15m bucket
		tail = history.iloc[-(n_out * 3):]
	return _resample_df_to_nmin(tail, 15)
