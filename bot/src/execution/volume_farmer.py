"""Volume-farmer bar-driven paper session.

Fully self-contained, intentionally decoupled from :mod:`execution.paper_session`
and the mean-reversion strategy stack so this experiment cannot affect the
proven 15m/5m paths.
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from strategy.indicators import (
	bollinger_bands as bb_indicator,
	ema as ema_indicator,
	rsi as rsi_indicator,
	wavetrend as wt_indicator,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class FarmerEvent:
	"""Event emitted by the volume-farmer session."""

	kind: str  # 'entry' | 'exit' | 'skip' | 'halt' | 'milestone'
	time: pd.Timestamp
	payload: Dict[str, Any]


@dataclass
class _Position:
	side: str           # 'long' | 'short'
	entry_price: float
	entry_time: pd.Timestamp
	notional: float
	tp: float
	sl: float
	bars_held: int = 0
	tp_bps: float = 0.0
	sl_bps: float = 0.0


@dataclass
class VolumeFarmerSession:
	"""Bar-by-bar paper volume farmer.

	Emits an event callback for every lifecycle transition. Callers poll with
	:meth:`on_new_candle` once per closed bar.
	"""

	config: Dict[str, Any]
	event_callback: Optional[Callable[[FarmerEvent], None]] = None

	# mutable state -----------------------------------------------------
	equity: float = field(init=False)
	peak_equity: float = field(init=False)
	start_equity: float = field(init=False)
	position: Optional[_Position] = field(default=None, init=False)
	total_volume_usd: float = field(default=0.0, init=False)
	total_fees_gross: float = field(default=0.0, init=False)
	total_rebate_accrued: float = field(default=0.0, init=False)
	total_rebate_transferred: float = field(default=0.0, init=False)
	total_pnl: float = field(default=0.0, init=False)
	wins: int = field(default=0, init=False)
	losses: int = field(default=0, init=False)
	round_trips: int = field(default=0, init=False)
	consec_losses: int = field(default=0, init=False)
	cooldown_bars_left: int = field(default=0, init=False)
	ledger: List[Dict[str, Any]] = field(default_factory=list, init=False)
	halted: bool = field(default=False, init=False)
	halt_reason: str = field(default="", init=False)
	last_side: str = field(default="", init=False)       # for alternation
	daily_pnl: float = field(default=0.0, init=False)
	daily_pnl_date: str = field(default="", init=False)
	day_start_equity: float = field(default=0.0, init=False)
	milestones_hit: List[float] = field(default_factory=list, init=False)
	last_rebate_reminder_at: Optional[str] = field(default=None, init=False)
	# Campaign clock for the pace controller: ISO timestamp of the first bar
	# processed (persisted), so required-volume-per-day math survives restarts.
	campaign_start_iso: str = field(default="", init=False)

	def __post_init__(self) -> None:
		f = self.config["farmer"]
		self.equity = float(f["capital_usd"])
		self.peak_equity = self.equity
		self.start_equity = self.equity
		self._leverage = float(f.get("leverage", 0))  # 0 = auto
		self._margin_frac = float(f.get("margin_fraction_per_trade", 0.05))
		# Dynamic sizing (ebirth.net formula): lev = risk$ / (margin$ * SL%)
		sizing_cfg = f.get("sizing", {})
		self._dynamic_leverage = bool(sizing_cfg.get("dynamic_leverage", False))
		self._risk_pct = float(sizing_cfg.get("risk_per_trade_pct", 0.025))
		self._max_leverage = float(sizing_cfg.get("max_leverage", 125))
		self._min_leverage = float(sizing_cfg.get("min_leverage", 5))
		self._tp_bps = float(f["tp_bps"])
		self._sl_bps = float(f["sl_bps"])
		# Time-stop: closes non-winning trades after N bars at the current bar
		# close. When farmer.time_stop.enabled, max_hold_bars from that block
		# wins; otherwise we fall back to the legacy top-level max_hold_bars.
		ts_cfg = f.get("time_stop", {}) or {}
		self._time_stop_enabled = bool(ts_cfg.get("enabled", False))
		if self._time_stop_enabled:
			self._max_hold = int(ts_cfg.get("max_hold_bars", f.get("max_hold_bars", 2)))
		else:
			self._max_hold = int(f.get("max_hold_bars", 999))
		# Maker-exit: when enabled, the time_stop close path is accounted as a
		# maker fill (matches the live re-peg loop). Wide-band SL hits are still
		# taker — those are emergency exits, not the steady-state behavior.
		self._maker_exit_enabled = bool((f.get("maker_exit", {}) or {}).get("enabled", False))
		entry_cfg = f.get("entry", {})
		self._entry_mode = str(entry_cfg.get("mode", "micro_momentum"))
		self._min_range_bps = float(entry_cfg.get("min_bar_range_bps", 0.0))
		self._max_range_bps = float(entry_cfg.get("max_bar_range_bps", 1_000.0))
		# RSI / WaveTrend params (used when entry_mode == 'rsi_wt')
		rsi_cfg = entry_cfg.get("rsi", {})
		self._rsi_period = int(rsi_cfg.get("period", 14))
		self._rsi_upper = float(rsi_cfg.get("upper", 70.0))
		self._rsi_lower = float(rsi_cfg.get("lower", 30.0))
		wt_cfg = entry_cfg.get("wavetrend", {})
		self._wt_channel = int(wt_cfg.get("channel_len", 9))
		self._wt_avg = int(wt_cfg.get("average_len", 12))
		self._wt_signal = int(wt_cfg.get("signal_len", 3))
		self._wt_oversold = float(wt_cfg.get("oversold", -53.0))
		self._wt_overbought = float(wt_cfg.get("overbought", 53.0))
		self._require_wt_confirm = bool(entry_cfg.get("require_wt_confirmation", False))
		self._wt_lookback = int(entry_cfg.get("wt_confluence_lookback", 3))
		# Bollinger-fade params (used when entry_mode == 'bollinger_fade')
		bb_cfg = entry_cfg.get("bollinger", {})
		self._bb_period = int(bb_cfg.get("period", 20))
		self._bb_std = float(bb_cfg.get("std_dev", 2.0))
		self._bb_rsi_long_max = float(bb_cfg.get("rsi_long_max", 35.0))
		self._bb_rsi_short_min = float(bb_cfg.get("rsi_short_min", 65.0))
		self._bb_width_min_bps = float(bb_cfg.get("width_min_bps", 20.0))   # skip squeezed
		self._bb_width_max_bps = float(bb_cfg.get("width_max_bps", 200.0))  # skip exploded
		self._bb_ema_len = int(bb_cfg.get("ema_trend_len", 200))
		self._bb_ema_filter = bool(bb_cfg.get("ema_trend_filter", False))
		# Multi-timeframe entry params (used when entry_mode == 'multi_timeframe')
		mtf_cfg = entry_cfg.get("mtf", {})
		self._mtf_lookback = int(mtf_cfg.get("sma_lookback", 5))
		self._mtf_min_bars = int(mtf_cfg.get("min_5m_bars", 20))
		# New MTF API v2: 3m micro + 15m macro alignment.
		# Triggered when primary_3m_lookback_bars > 0 in entry config.
		self._mtf_micro_lookback = int(entry_cfg.get("primary_3m_lookback_bars", 0))
		self._mtf_macro_lookback = int(entry_cfg.get("macro_15m_lookback_bars", 5))
		self._mtf_allow_neutral_micro = bool(entry_cfg.get("allow_neutral_micro", True))
		self._mtf_skip_neutral_macro = bool(entry_cfg.get("skip_neutral_macro", True))
		# 1h-veto post-filter (orthogonal to entry mode; blocks trades that
		# fight a strong opposing 1h trend).
		veto_cfg = f.get("h1_veto", {})
		self._h1_veto_enabled = bool(veto_cfg.get("enabled", False))
		self._h1_veto_ema = int(veto_cfg.get("ema_period", 20))
		self._h1_veto_slope_bars = int(veto_cfg.get("slope_bars", 2))
		self._h1_veto_min_1h_bars = int(veto_cfg.get("min_1h_bars", 25))
		self._h1_veto_blocks = 0  # diagnostic counter
		self._entries_considered = 0   # times flat+ready → _decide_entry called
		self._mtf_skips = 0            # times MTF alignment blocked an entry
		# Default FALSE: defaulting alternation ON silently inverted every other
		# micro_momentum signal in any config that omitted the key.
		self._alternate = bool(f.get("alternate_direction", False))
		# Same-bar re-entry: when a position closes at bar close (time-stop/TP)
		# the bot is flat with a fresh signal in hand — waiting a full extra bar
		# halves the volume ceiling for nothing. Off by default (legacy).
		self._reentry_same_bar = bool(f.get("reentry_same_bar", False))
		# Session-hour filter: set of UTC hours to skip entries (e.g. low-liquidity windows)
		self._skip_hours: set = set(int(h) for h in entry_cfg.get("skip_hours", []))
		self._session_blocks = 0   # how many bars were silently skipped by hour filter
		# Dull-market notification: emit a 'skip' event when we go N consecutive
		# bars without an entry signal. Helps explain long quiet stretches on 1m.
		notif_cfg = self.config.get("notifications", {}).get("telegram", {}) or {}
		self._dull_threshold = int(notif_cfg.get("dull_market_bars", 30))
		self._dull_repeat = int(notif_cfg.get("dull_market_repeat_bars", 60))
		self._flat_skip_bars = 0
		self._dull_last_emit_at = 0  # value of _flat_skip_bars at last emit
		# Trend-break early exit: cut losers short before they reach full SL
		tb_cfg = f.get("trend_break", {})
		self._tb_enabled = bool(tb_cfg.get("enabled", False))
		self._tb_min_bars = int(tb_cfg.get("min_bars_held", 2))
		self._tb_adverse_bps = float(tb_cfg.get("adverse_bps", 20.0))
		# ATR-relative TP/SL and session filter
		atr_cfg = f.get("atr", {})
		self._atr_period = int(atr_cfg.get("period", 14))
		self._atr_relative = bool(atr_cfg.get("relative", False))
		self._atr_tp_mult = float(atr_cfg.get("tp_mult", 0.5))
		self._atr_sl_mult = float(atr_cfg.get("sl_mult", 1.5))
		self._atr_tp_bps_min = float(atr_cfg.get("tp_bps_min", 5.0))
		self._atr_sl_bps_min = float(atr_cfg.get("sl_bps_min", 8.0))
		self._atr_min_usd = float(atr_cfg.get("min_usd", 0.0))
		# Fee-aware, achievability-capped TP. The TP target is raised so a filled
		# TP always clears a full round-trip of fees with a profit buffer, then
		# clamped so a high-ATR bar can't push the TP past where price actually
		# travels within the hold (which would turn a would-be TP win into a
		# time-stop taker loss). All three default to no-ops, so configs that
		# don't set them are byte-identical to the old behaviour.
		self._tp_fee_cover_mult = float(atr_cfg.get("tp_fee_cover_mult", 0.0))
		self._tp_profit_buffer_bps = float(atr_cfg.get("tp_profit_buffer_bps", 0.0))
		self._tp_bps_max = float(atr_cfg.get("tp_bps_max", 0.0))  # <=0 disables cap
		self._limit_tp = bool(f.get("limit_tp", False))
		# Firm effective-TP override: pin the TP to a fixed bps after the ATR
		# bracket + floor/cap. 0 = disabled.
		self._force_tp_bps = float(f.get("force_tp_bps", 0.0))
		self._pending_tp_bps: float = self._tp_bps
		self._pending_sl_bps: float = self._sl_bps
		fees_cfg = self.config.get("fees", {})
		self._maker_rate = float(fees_cfg.get("maker", 0.0001))
		self._taker_rate = float(fees_cfg.get("taker", 0.0005))
		self._rebate_pct = float(fees_cfg.get("rebate_pct", 0.0))
		# Whether the 40% rebate applies to ALL fee legs (this campaign: yes)
		# or to maker legs only. Drives rebate accrual and the round-trip fee
		# floor; getting it wrong over/under-states reclaimable cash.
		self._rebate_all_legs = bool(fees_cfg.get("rebate_all_legs", True))
		risk_cfg = self.config.get("risk", {})
		self._daily_loss_limit = float(risk_cfg.get("daily_loss_limit_pct", 0.05))
		self._max_dd = float(risk_cfg.get("max_drawdown_pct", 0.25))
		self._consec_loss_limit = int(risk_cfg.get("consecutive_losses_limit", 3))
		self._consec_loss_cooldown_bars = int(risk_cfg.get("consecutive_losses_cooldown_bars", 24))
		# Which close reasons advance the consecutive-loss streak. Time-stop
		# scratches lose a few bps of fees by design — counting them used to
		# trip a 1h entry pause every handful of trades (the single biggest
		# self-inflicted throughput loss found in live data).
		self._consec_loss_reasons = set(
			risk_cfg.get("consecutive_losses_count_reasons", ["sl", "sl_ambiguous"])
		)
		self._stop_on_target = bool(risk_cfg.get("stop_on_volume_target", True))
		target_cfg = self.config.get("target", {})
		self._volume_target = float(target_cfg.get("volume_usd", 1_000_000.0))
		# Overshoot buffer: session-counted volume can lag the exchange's
		# official counter (close legs differ, abandoned entries, etc.), so we
		# halt a little PAST the nominal target rather than risk missing the
		# bounty by a rounding margin.
		self._volume_target_buffer = float(target_cfg.get("volume_buffer_pct", 0.02))
		self._milestones = [0.1, 0.25, 0.5, 0.75, 1.0]
		# Pace controller: scale per-trade margin so the projected month-end
		# volume converges on the target. Bounded multiplicatively so a slow
		# first day can't triple position size out of the gate.
		pace_cfg = f.get("pace", {}) or {}
		self._pace_enabled = bool(pace_cfg.get("enabled", False))
		self._pace_campaign_days = float(pace_cfg.get("campaign_days", 30.0))
		self._pace_min_frac = float(pace_cfg.get("min_margin_fraction", self._margin_frac * 0.5))
		self._pace_max_frac = float(pace_cfg.get("max_margin_fraction", self._margin_frac * 2.0))
		self._pace_warmup_trips = int(pace_cfg.get("warmup_trips", 10))
		self._last_pace: Dict[str, Any] = {}
		# Auto rebate-top-up reminder: pings configured members on Telegram
		# when the wallet looks shaky and there's enough available rebate to
		# rescue it. Cooldown prevents spam.
		notif_tg_cfg = self.config.get("notifications", {}).get("telegram", {}) or {}
		rr_cfg = notif_tg_cfg.get("rebate_reminder", {}) or {}
		self._rr_enabled = bool(rr_cfg.get("enabled", True))
		self._rr_equity_floor_pct = float(rr_cfg.get("equity_floor_pct", 0.85))
		self._rr_drawdown_pct = float(rr_cfg.get("drawdown_pct", 0.08))
		self._rr_min_rebate_usd = float(rr_cfg.get("min_rebate_usd", 5.0))
		self._rr_cooldown_minutes = int(rr_cfg.get("cooldown_minutes", 60))
		self._rr_mention_handles = list(rr_cfg.get("mention_handles", []))
		# Daily cadence: ping the group once per UTC day to reclaim rebate even
		# when the wallet is healthy, so it never sits idle off-wallet.
		self._rr_daily_enabled = bool(rr_cfg.get("daily", True))
		self._rr_last_daily_date = ""

	# ------------------------------------------------------------------
	def _emit(self, evt: FarmerEvent) -> None:
		if self.event_callback:
			try:
				self.event_callback(evt)
			except Exception as exc:  # noqa: BLE001
				LOGGER.exception("event_callback raised: %s", exc)

	# ------------------------------------------------------------------
	def _maybe_reset_daily(self, bar_time: pd.Timestamp) -> None:
		d = bar_time.strftime("%Y-%m-%d")
		if d != self.daily_pnl_date:
			self.daily_pnl_date = d
			self.daily_pnl = 0.0
			# Day-start equity anchors the daily-loss limit; denominating by
			# session START equity made the limit drift with cumulative PnL.
			self.day_start_equity = self.equity

	# ------------------------------------------------------------------
	def _check_halt(self, bar_time: pd.Timestamp) -> bool:
		if self.halted:
			return True
		# absolute drawdown — HARD halt
		dd = (self.peak_equity - self.equity) / max(self.peak_equity, 1e-9)
		if dd >= self._max_dd:
			self.halted = True
			self.halt_reason = f"max_drawdown {dd*100:.2f}%"
		# daily loss — HARD halt (account protection)
		day_base = self.day_start_equity if self.day_start_equity > 0 else self.start_equity
		day_loss = -self.daily_pnl / max(day_base, 1e-9)
		if day_loss >= self._daily_loss_limit:
			self.halted = True
			self.halt_reason = f"daily_loss {day_loss*100:.2f}%"
		# volume target reached — halt (with overshoot buffer so the exchange's
		# official counter, which we may undercount, is safely past the target)
		halt_volume = self._volume_target * (1.0 + self._volume_target_buffer)
		if self._stop_on_target and self.total_volume_usd >= halt_volume:
			self.halted = True
			self.halt_reason = f"volume_target_reached {self.total_volume_usd:,.0f}"
		if self.halted:
			self._emit(FarmerEvent(
				kind="halt", time=bar_time,
				payload={"reason": self.halt_reason, "equity": self.equity,
						 "volume": self.total_volume_usd},
			))
		return self.halted

	# ------------------------------------------------------------------
	def _set_pending_bracket(self, history: pd.DataFrame, ref_price: float) -> None:
		"""Compute the ATR-relative, fee-floored TP/SL bracket for the next entry.

		Every entry mode MUST route through this before opening — modes that
		returned early used to trade on a stale/unfloored bracket, silently
		opening positions whose TP could not clear fees.
		"""
		atr_val = self._compute_atr(history)
		if self._atr_relative and not np.isnan(atr_val) and ref_price > 0:
			atr_bps = atr_val / ref_price * 10_000
			self._pending_tp_bps = self._apply_tp_floor_cap(
				max(self._atr_tp_bps_min, self._atr_tp_mult * atr_bps)
			)
			self._pending_sl_bps = max(self._atr_sl_bps_min, self._atr_sl_mult * atr_bps)
		else:
			self._pending_tp_bps = self._apply_tp_floor_cap(self._tp_bps)
			self._pending_sl_bps = self._sl_bps
		# Firm effective-TP override — pin TP after the ATR bracket + floor/cap
		# so nothing downstream can drift it. SL stays ATR-relative.
		if self._force_tp_bps > 0:
			self._pending_tp_bps = self._force_tp_bps

	# ------------------------------------------------------------------
	def _decide_entry(self, history: pd.DataFrame) -> Optional[str]:
		"""Return 'long' | 'short' | None based on entry signal."""

		if history.empty:
			return None

		# RSI + WaveTrend entry (based on WTF Pine Scripts)
		if self._entry_mode == "rsi_wt":
			side = self._rsi_wt_signal(history)
			if side is not None:
				self._set_pending_bracket(history, float(history.iloc[-1]["close"]))
			return side

		# Bollinger-band fade with RSI + optional EMA trend filter
		if self._entry_mode == "bollinger_fade":
			side = self._bollinger_fade_signal(history)
			if side is not None:
				self._set_pending_bracket(history, float(history.iloc[-1]["close"]))
			return side

		# Multi-timeframe trend alignment.
		if self._entry_mode == "multi_timeframe":
			if self._mtf_micro_lookback > 0:
				# New API v2: 3m micro + 15m macro with range gate.
				last = history.iloc[-1]
				open_p = float(last["open"])
				close_p = float(last["close"])
				if open_p <= 0:
					return None
				bar_range_bps = abs(close_p - open_p) / open_p * 10_000
				if bar_range_bps < self._min_range_bps:
					return None
				if bar_range_bps > self._max_range_bps:
					return None
				from strategy.mtf_direction import (
					get_aligned_direction, get_micro_bars, get_macro_bars,
				)
				tf = self.config.get("exchange", {}).get("timeframe", "5m")
				micro = get_micro_bars(history, tf, self._mtf_micro_lookback)
				macro = get_macro_bars(history, tf, self._mtf_macro_lookback)
				direction = get_aligned_direction(
					bars_3m=micro,
					bars_15m=macro,
					allow_neutral_micro=self._mtf_allow_neutral_micro,
					skip_neutral_macro=self._mtf_skip_neutral_macro,
				)
				if direction is None:
					self._mtf_skips += 1
				else:
					self._set_pending_bracket(history, close_p)
				return direction
			else:
				# Old API (legacy): 5m + 15m SMA alignment, bypass range gate.
				from strategy.mtf_direction import detect_direction
				direction = detect_direction(
					history,
					lookback=self._mtf_lookback,
					min_5m_bars=self._mtf_min_bars,
				)
				if direction is not None:
					self._set_pending_bracket(history, float(history.iloc[-1]["close"]))
				return direction

		last = history.iloc[-1]
		open_p = float(last["open"])
		close_p = float(last["close"])
		if open_p <= 0:
			return None
		bar_range_bps = abs(close_p - open_p) / open_p * 10_000
		if bar_range_bps < self._min_range_bps:
			return None
		if bar_range_bps > self._max_range_bps:
			return None

		# ATR session filter + ATR-relative pending TP/SL
		atr_val = self._compute_atr(history)
		if self._atr_min_usd > 0 and (np.isnan(atr_val) or atr_val < self._atr_min_usd):
			return None  # skip: ATR below high-vol threshold
		self._set_pending_bracket(history, open_p)

		if self._entry_mode == "micro_momentum":
			bias = "long" if close_p > open_p else "short"
		elif self._entry_mode == "mean_revert":
			bias = "short" if close_p > open_p else "long"
		else:  # 'alternate' or anything else
			bias = "short" if self.last_side == "long" else "long"

		# Force alternation if configured (prevents one-sided drift)
		if self._alternate and self.last_side == bias:
			bias = "short" if bias == "long" else "long"
		return bias

	# ------------------------------------------------------------------
	def _rsi_wt_signal(self, history: pd.DataFrame) -> Optional[str]:
		"""RSI crossback signal (WTF Pine Script).

		Long  when RSI[-2] < lower and RSI[-1] >= lower (crossback up).
		Short when RSI[-2] > upper and RSI[-1] <= upper (crossback down).
		Optionally require a WT green/red dot within the last N bars.
		"""
		min_bars = max(self._rsi_period, self._wt_channel) + 5
		if len(history) < min_bars:
			return None
		close = history["close"].astype(float).values
		rsi_series = rsi_indicator(close, self._rsi_period)
		if len(rsi_series) < 2:
			return None
		r_prev = rsi_series[-2]
		r_now = rsi_series[-1]
		if np.isnan(r_prev) or np.isnan(r_now):
			return None

		rsi_buy = (r_prev < self._rsi_lower) and (r_now >= self._rsi_lower)
		rsi_sell = (r_prev > self._rsi_upper) and (r_now <= self._rsi_upper)
		if not (rsi_buy or rsi_sell):
			return None

		if self._require_wt_confirm:
			wt1, wt2 = wt_indicator(
				high=history["high"].astype(float).values,
				low=history["low"].astype(float).values,
				close=close,
				channel_len=self._wt_channel,
				average_len=self._wt_avg,
				signal_len=self._wt_signal,
			)
			lb = min(self._wt_lookback + 1, len(wt1))
			wt1_tail = wt1[-lb:]
			wt2_tail = wt2[-lb:]
			green = red = False
			for i in range(1, len(wt1_tail)):
				if np.isnan(wt1_tail[i]) or np.isnan(wt1_tail[i-1]):
					continue
				cross_up = wt1_tail[i-1] <= wt2_tail[i-1] and wt1_tail[i] > wt2_tail[i]
				cross_dn = wt1_tail[i-1] >= wt2_tail[i-1] and wt1_tail[i] < wt2_tail[i]
				if cross_up and wt2_tail[i] <= self._wt_oversold:
					green = True
				if cross_dn and wt2_tail[i] >= self._wt_overbought:
					red = True
			if rsi_buy and not green:
				return None
			if rsi_sell and not red:
				return None

		return "long" if rsi_buy else "short"

	# ------------------------------------------------------------------
	def _bollinger_fade_signal(self, history: pd.DataFrame) -> Optional[str]:
		"""Mean-reversion fade at BB extremes with RSI + BB-width gate.

		Long  : close <= lower BB AND RSI < rsi_long_max AND width in band
				[AND close > EMA200 if ema_trend_filter]
		Short : close >= upper BB AND RSI > rsi_short_min AND width in band
				[AND close < EMA200 if ema_trend_filter]
		"""
		need = max(self._bb_period, self._rsi_period, self._bb_ema_len if self._bb_ema_filter else 0) + 5
		if len(history) < need:
			return None
		close = history["close"].astype(float).values
		mid, upper, lower = bb_indicator(close, self._bb_period, self._bb_std)
		c = close[-1]
		m = mid[-1]; up = upper[-1]; lo = lower[-1]
		if np.isnan(m) or np.isnan(up) or np.isnan(lo) or m <= 0:
			return None
		width_bps = (up - lo) / m * 10_000
		if width_bps < self._bb_width_min_bps or width_bps > self._bb_width_max_bps:
			return None
		rsi_series = rsi_indicator(close, self._rsi_period)
		r_now = rsi_series[-1]
		if np.isnan(r_now):
			return None

		long_ok = (c <= lo) and (r_now < self._bb_rsi_long_max)
		short_ok = (c >= up) and (r_now > self._bb_rsi_short_min)
		if not (long_ok or short_ok):
			return None

		if self._bb_ema_filter:
			ema_series = ema_indicator(close, self._bb_ema_len)
			e_now = ema_series[-1]
			if np.isnan(e_now):
				return None
			if long_ok and c <= e_now:
				return None
			if short_ok and c >= e_now:
				return None

		return "long" if long_ok else "short"

	# ------------------------------------------------------------------
	def _compute_atr(self, history: pd.DataFrame) -> float:
		"""Wilder ATR-14 from bar history. Returns NaN if insufficient data."""
		if len(history) < self._atr_period + 1:
			return float("nan")
		tail = history.tail(max(50, self._atr_period * 3))
		hi = tail["high"].astype(float).values
		lo = tail["low"].astype(float).values
		cl = tail["close"].astype(float).values
		prev_cl = np.concatenate([[cl[0]], cl[:-1]])
		tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_cl), np.abs(lo - prev_cl)))
		period = self._atr_period
		atr = float(tr[:period].mean())
		alpha = 1.0 / period
		for k in range(period, len(tr)):
			atr = atr * (1.0 - alpha) + tr[k] * alpha
		return atr

	# ------------------------------------------------------------------
	def _calc_leverage(self, margin: float, sl_bps: Optional[float] = None) -> float:
		"""Dynamic leverage: lev = risk$ / (margin$ * SL_fraction).

		Mirrors the ebirth.net leverage calculator.
		Caps at exchange max; floors at min_leverage.
		"""
		if not self._dynamic_leverage or margin <= 0:
			return self._leverage if self._leverage > 0 else 20.0
		risk_usd = self.equity * self._risk_pct
		_sl = sl_bps if sl_bps is not None else self._sl_bps
		sl_frac = _sl / 10_000.0
		if sl_frac <= 0:
			return self._max_leverage
		lev = risk_usd / (margin * sl_frac)
		return max(self._min_leverage, min(lev, self._max_leverage))

	# ------------------------------------------------------------------
	def _round_trip_fee_bps(self) -> float:
		"""Round-trip fee (bps) for one TP round trip.

		The open leg is always maker (post_only limit entry). The close leg is
		maker when limit_tp is on (OKX limit TP fills maker), else taker. The
		rebate reduces maker legs always; taker legs too when the program
		rebates all fees (fees.rebate_all_legs, the default for this campaign).
		"""
		maker_eff = self._maker_rate * (1.0 - self._rebate_pct)
		taker_eff = self._taker_rate * (1.0 - self._rebate_pct) if self._rebate_all_legs else self._taker_rate
		open_leg = maker_eff
		close_leg = maker_eff if self._limit_tp else taker_eff
		return (open_leg + close_leg) * 10_000.0

	# ------------------------------------------------------------------
	def _rebate_for(self, fee: float, fee_type: str) -> float:
		"""Rebate accrued on one fee leg under the configured program rules."""
		if fee_type == "maker" or self._rebate_all_legs:
			return fee * self._rebate_pct
		return 0.0

	def _apply_tp_floor_cap(self, tp_bps: float) -> float:
		"""Raise tp_bps to the fee-aware floor, then clamp to the achievability
		cap.

		floor = tp_fee_cover_mult * round_trip_fee_bps + tp_profit_buffer_bps.
		Order matters: the floor is applied first (guarantees a filled TP clears
		fees), the cap last (guarantees the TP stays reachable). Backward
		compatible: mult/buffer 0 => floor is 0; tp_bps_max <= 0 => no cap.
		Operators must keep tp_bps_max >= the fee floor, else the cap silently
		re-creates an unprofitable TP.
		"""
		fee_floor_bps = (
			self._tp_fee_cover_mult * self._round_trip_fee_bps()
			+ self._tp_profit_buffer_bps
		)
		tp_bps = max(tp_bps, fee_floor_bps)
		if self._tp_bps_max > 0:
			tp_bps = min(tp_bps, self._tp_bps_max)
		return tp_bps

	# ------------------------------------------------------------------
	def _pace_margin_frac(self, bar_time: pd.Timestamp) -> float:
		"""Margin fraction for the next trade, scaled by campaign pace.

		factor = (volume/day still required) / (volume/day achieved so far),
		clamped to [0.5, 3.0], applied to the base margin fraction and bounded
		by [min_margin_fraction, max_margin_fraction]. Until warmup (first
		trips/first hours) the base fraction is used unchanged — early-rate
		noise must not drive sizing.
		"""
		base = self._margin_frac
		if not self.campaign_start_iso:
			self.campaign_start_iso = bar_time.isoformat()
		if not self._pace_enabled:
			return base
		try:
			start = pd.Timestamp(self.campaign_start_iso)
			elapsed_days = max((bar_time - start).total_seconds() / 86_400.0, 1e-6)
		except Exception:  # noqa: BLE001
			return base
		days_left = max(self._pace_campaign_days - elapsed_days, 0.25)
		remaining = max(self._volume_target - self.total_volume_usd, 0.0)
		required_per_day = remaining / days_left
		achieved_per_day = self.total_volume_usd / elapsed_days
		self._last_pace = {
			"elapsed_days": round(elapsed_days, 2),
			"days_left": round(days_left, 2),
			"required_per_day": round(required_per_day, 0),
			"achieved_per_day": round(achieved_per_day, 0),
			"projected_total": round(achieved_per_day * self._pace_campaign_days, 0),
			"margin_fraction": base,
		}
		if remaining <= 0:
			return self._pace_min_frac
		if self.round_trips < self._pace_warmup_trips or elapsed_days < 0.5 or achieved_per_day <= 0:
			return base
		factor = max(0.5, min(required_per_day / achieved_per_day, 3.0))
		frac = max(self._pace_min_frac, min(base * factor, self._pace_max_frac))
		self._last_pace["margin_fraction"] = round(frac, 5)
		return frac

	# ------------------------------------------------------------------
	def _open_position(self, side: str, price: float, bar_time: pd.Timestamp) -> None:
		# Reset dull-market counter — we're entering a trade.
		self._flat_skip_bars = 0
		self._dull_last_emit_at = 0
		margin_frac = self._pace_margin_frac(bar_time)
		margin = self.equity * margin_frac
		leverage = self._calc_leverage(margin, sl_bps=self._pending_sl_bps)
		notional = margin * leverage
		open_fee = notional * self._maker_rate  # limit order = maker
		self.total_fees_gross += open_fee
		self.total_rebate_accrued += self._rebate_for(open_fee, "maker")
		self.total_volume_usd += notional
		self.equity -= open_fee  # fee paid on open
		tp = price * (1 + self._pending_tp_bps / 10_000) if side == "long" else price * (1 - self._pending_tp_bps / 10_000)
		sl = price * (1 - self._pending_sl_bps / 10_000) if side == "long" else price * (1 + self._pending_sl_bps / 10_000)
		self.position = _Position(
			side=side, entry_price=price, entry_time=bar_time,
			notional=notional, tp=tp, sl=sl,
			tp_bps=self._pending_tp_bps, sl_bps=self._pending_sl_bps,
		)
		self._emit(FarmerEvent(
			kind="entry", time=bar_time,
			payload={
				"side": side, "price": price, "notional": notional,
				"margin": round(margin, 4),
				"margin_fraction": round(margin_frac, 5),
				"leverage": round(leverage, 1),
				"open_fee": open_fee,
				"fee_type": "maker",
				"tp": tp, "sl": sl,
				"tp_bps": self._pending_tp_bps,
				"sl_bps": self._pending_sl_bps,
				"equity": self.equity,
				"volume": self.total_volume_usd,
				"volume_target": self._volume_target,
				"round_trips": self.round_trips,
				"capital": self.equity,
				"wins": self.wins,
				"losses": self.losses,
				"pace": dict(self._last_pace),
			},
		))
		self._check_milestones(bar_time)

	# ------------------------------------------------------------------
	def _close_position(
		self, exit_price: float, bar_time: pd.Timestamp, reason: str
	) -> None:
		assert self.position is not None
		p = self.position
		if p.side == "long":
			pnl_pct = (exit_price - p.entry_price) / p.entry_price
		else:
			pnl_pct = (p.entry_price - exit_price) / p.entry_price
		gross_pnl = pnl_pct * p.notional
		# TP exits: use maker rate when limit_tp is enabled (OKX supports limit TP orders).
		# Time-stop exits use maker rate when maker_exit is enabled (matches the
		# live re-peg loop). SL / trend_break / sl_ambiguous stay taker — those
		# are fast/forced exits that cannot be guaranteed maker.
		if self._limit_tp and reason == "tp":
			close_fee = p.notional * self._maker_rate
			close_fee_type = "maker"
		elif self._maker_exit_enabled and reason == "time_stop":
			close_fee = p.notional * self._maker_rate
			close_fee_type = "maker"
		else:
			close_fee = p.notional * self._taker_rate
			close_fee_type = "taker"
		open_fee = p.notional * self._maker_rate
		# Net PnL must charge BOTH legs' fees. Charging only the close fee
		# inflated total_pnl/win-rate and weakened the daily-loss halt (the
		# demo state file showed equity +$11.10 vs reported PnL +$22.39 —
		# the gap was exactly the sum of uncounted open fees).
		net_pnl = gross_pnl - open_fee - close_fee
		self.total_fees_gross += close_fee
		self.total_rebate_accrued += self._rebate_for(close_fee, close_fee_type)
		# Close-leg volume at the EXIT price: the exchange counts the closing
		# fill's notional, not the entry's. Same contracts, different price.
		close_leg_notional = p.notional * (exit_price / max(p.entry_price, 1e-9))
		self.total_volume_usd += close_leg_notional
		self.total_pnl += net_pnl
		self.equity += gross_pnl - close_fee
		self.peak_equity = max(self.peak_equity, self.equity)
		self._maybe_reset_daily(bar_time)
		self.daily_pnl += net_pnl
		self.round_trips += 1
		self.last_side = p.side
		if net_pnl > 0:
			self.wins += 1
			self.consec_losses = 0
		else:
			self.losses += 1
			# Only configured reasons advance the cooldown streak: a 1-bar
			# time-stop scratch is a designed fee-cost exit, not an adverse
			# move — counting scratches paused entries for 1h after every 3
			# non-winners and starved the volume target.
			if reason in self._consec_loss_reasons:
				self.consec_losses += 1
				if self.consec_losses >= self._consec_loss_limit:
					self.cooldown_bars_left = self._consec_loss_cooldown_bars
					self.consec_losses = 0  # reset streak so next losses start fresh
		fee_overage = close_fee - p.notional * self._maker_rate
		wr_running = round(self.wins / self.round_trips * 100.0, 1) if self.round_trips else 0.0
		self.ledger.append({
			"entry_time": p.entry_time.isoformat(),
			"exit_time": bar_time.isoformat(),
			"side": p.side,
			"entry_price": p.entry_price,
			"exit_price": exit_price,
			"notional": p.notional,
			"gross_pnl": gross_pnl,
			"open_fee": open_fee,
			"close_fee": close_fee,
			"close_fee_type": close_fee_type,
			"total_fee": open_fee + close_fee,
			"fee_overage": fee_overage,
			"net_pnl": net_pnl,
			"reason": reason,
		})
		self._emit(FarmerEvent(
			kind="exit", time=bar_time,
			payload={
				"side": p.side, "entry_price": p.entry_price,
				"exit_price": exit_price, "reason": reason,
				"gross_pnl": gross_pnl,
				"open_fee": open_fee,
				"close_fee": close_fee,
				"close_fee_type": close_fee_type,
				"fee_total": open_fee + close_fee,
				"fee_overage": fee_overage,
				"net_pnl": net_pnl,
				"notional": p.notional,
				"bars_held": p.bars_held,
				"equity": self.equity, "round_trips": self.round_trips,
				"volume": self.total_volume_usd,
				"volume_target": self._volume_target,
				"wins": self.wins, "losses": self.losses,
				"win_rate_pct": wr_running,
				"consec_losses": self.consec_losses,
				"capital": self.equity,
			},
		))
		self.position = None
		self._check_milestones(bar_time)

	# ------------------------------------------------------------------
	def _check_milestones(self, bar_time: pd.Timestamp) -> None:
		for m in self._milestones:
			thresh = m * self._volume_target
			if self.total_volume_usd >= thresh and m not in self.milestones_hit:
				self.milestones_hit.append(m)
				self._emit(FarmerEvent(
					kind="milestone", time=bar_time,
					payload={
						"pct": m, "volume": self.total_volume_usd,
						"equity": self.equity,
						"fees_gross": self.total_fees_gross, "pnl": self.total_pnl,
					},
				))

	# ------------------------------------------------------------------
	def on_new_candle(self, history: pd.DataFrame) -> None:
		"""Process one newly closed 5m candle.

		Assumes candle's H/L are reachable intrabar.
		"""

		if history.empty:
			return
		last = history.iloc[-1]
		bar_time = pd.Timestamp(last["open_time"])
		self._maybe_reset_daily(bar_time)

		if self._check_halt(bar_time):
			return

		self._check_rebate_reminder(bar_time)

		high = float(last["high"])
		low = float(last["low"])
		close = float(last["close"])

		# Step 1: if in position, check TP/SL intrabar then time-stop
		if self.position is not None:
			p = self.position
			p.bars_held += 1
			tp_hit = (p.side == "long" and high >= p.tp) or (p.side == "short" and low <= p.tp)
			sl_hit = (p.side == "long" and low <= p.sl) or (p.side == "short" and high >= p.sl)

			# If both hit in same bar, assume worst case (SL first)
			if sl_hit and tp_hit:
				self._close_position(p.sl, bar_time, reason="sl_ambiguous")
			elif tp_hit:
				self._close_position(p.tp, bar_time, reason="tp")
			elif sl_hit:
				self._close_position(p.sl, bar_time, reason="sl")
			elif (
				self._tb_enabled
				and p.bars_held >= self._tb_min_bars
				and (
					(p.side == "long"  and (p.entry_price - close) / p.entry_price * 1e4 >= self._tb_adverse_bps)
					or
					(p.side == "short" and (close - p.entry_price) / p.entry_price * 1e4 >= self._tb_adverse_bps)
				)
			):
				self._close_position(close, bar_time, reason="trend_break")
			elif p.bars_held >= self._max_hold:
				self._close_position(close, bar_time, reason="time_stop")
			# Same-bar re-entry: if we just closed and the config allows it,
			# fall through to the entry logic on this same closed bar — the
			# minimum round-trip cycle drops from 2 bars to 1, doubling the
			# volume ceiling. Halt/cooldown gates below still apply.
			if not (self._reentry_same_bar and self.position is None):
				return  # legacy: don't open a new trade on the close bar
			if self._check_halt(bar_time):
				return

		# Step 2: flat — look for entry (respect cooldown)
		if self.cooldown_bars_left > 0:
			self.cooldown_bars_left -= 1
			return
		# Session-hour filter: silently skip entries during specified UTC hours.
		if self._skip_hours and bar_time.hour in self._skip_hours:
			self._session_blocks += 1
			return
		self._entries_considered += 1
		side = self._decide_entry(history)
		if side is None:
			self._flat_skip_bars += 1
			n = self._flat_skip_bars
			# Emit a dull-market notice on first crossing of threshold,
			# then again every _dull_repeat bars while still flat.
			if self._dull_threshold > 0 and (
				n == self._dull_threshold
				or (n > self._dull_threshold
				    and (n - self._dull_last_emit_at) >= self._dull_repeat)
			):
				self._dull_last_emit_at = n
				# Compute median bar range over the last min(60, n) bars
				try:
					tail = history.tail(min(60, n)).copy()
					o = tail["open"].astype(float)
					c = tail["close"].astype(float)
					rng_bps = ((c - o).abs() / o.replace(0, float("nan")) * 10_000)
					med_range = float(rng_bps.median()) if len(rng_bps) else 0.0
				except Exception:  # noqa: BLE001
					med_range = 0.0
				self._emit(FarmerEvent(
					kind="skip", time=bar_time,
					payload={
						"reason": "dull_market",
						"flat_bars": n,
						"min_range_bps": self._min_range_bps,
						"median_range_bps": round(med_range, 2),
						"timeframe": str(self.config.get("exchange", {}).get("timeframe", "")),
					},
				))
			return
		# 1h-veto post-filter (orthogonal to entry mode).
		if self._h1_veto_enabled:
			from strategy.h1_veto import is_blocked
			if is_blocked(
				side,
				history,
				ema_period=self._h1_veto_ema,
				slope_bars=self._h1_veto_slope_bars,
				min_1h_bars=self._h1_veto_min_1h_bars,
			):
				self._h1_veto_blocks += 1
				return
		self._open_position(side, close, bar_time)

	# ------------------------------------------------------------------
	def _check_rebate_reminder(self, bar_time: pd.Timestamp) -> None:
		"""Emit a `rebate_reminder` event when wallet is shaky and rebate could rescue it.

		Fires when (a) the wallet has fallen below the safety floor or peak
		drawdown exceeds the configured threshold, (b) at least
		`min_rebate_usd` of unclaimed rebate sits in the pool, and (c) the
		cooldown since the last reminder has elapsed. Suggests transferring
		whichever is smaller — the available rebate, or the gap back to
		start equity.
		"""
		if not self._rr_enabled or self.halted:
			return
		available = self.available_rebate
		if available < self._rr_min_rebate_usd:
			return
		dd = (self.peak_equity - self.equity) / max(self.peak_equity, 1e-9)
		below_floor = self.equity < self.start_equity * self._rr_equity_floor_pct
		over_dd = dd >= self._rr_drawdown_pct
		# Daily cadence: fire once per UTC day regardless of wallet health, so the
		# group is reminded to move the 40% rebate back into the wallet routinely
		# (not only in a drawdown). Naturally rate-limited to once/day, so it
		# bypasses the minute-cooldown that guards the shaky-wallet triggers.
		today = bar_time.strftime("%Y-%m-%d")
		daily_due = self._rr_daily_enabled and (today != self._rr_last_daily_date)
		if not (below_floor or over_dd or daily_due):
			return
		# Cooldown (applies only to the shaky-wallet triggers, not the daily ping)
		if not daily_due and self.last_rebate_reminder_at:
			try:
				last = pd.Timestamp(self.last_rebate_reminder_at)
				if (bar_time - last) < pd.Timedelta(minutes=self._rr_cooldown_minutes):
					return
			except Exception:  # noqa: BLE001
				pass
		gap_to_start = max(0.0, self.start_equity - self.equity)
		suggested = min(available, gap_to_start) if gap_to_start > 0 else available
		if suggested < self._rr_min_rebate_usd:
			suggested = available
		# Daily ping suggests reclaiming the WHOLE available pool (nothing to
		# rescue — just keep it flowing back to the wallet).
		if daily_due and not (below_floor or over_dd):
			suggested = available
		reasons: List[str] = []
		if below_floor:
			reasons.append(
				f"balance ${self.equity:.2f} below floor "
				f"({self._rr_equity_floor_pct*100:.0f}% of start)"
			)
		if over_dd:
			reasons.append(f"drawdown {dd*100:.1f}% from peak")
		if daily_due:
			reasons.append("daily rebate top-up")
			self._rr_last_daily_date = today
		self.last_rebate_reminder_at = bar_time.isoformat()
		self._emit(FarmerEvent(
			kind="rebate_reminder", time=bar_time,
			payload={
				"equity": self.equity,
				"start_equity": self.start_equity,
				"peak_equity": self.peak_equity,
				"drawdown_pct": dd,
				"available_rebate": available,
				"suggested_amount": round(suggested, 4),
				"reasons": reasons,
				"mention_handles": list(self._rr_mention_handles),
			},
		))

	# ------------------------------------------------------------------
	@property
	def available_rebate(self) -> float:
		"""Rebate accrued from fees but not yet transferred to equity."""
		return max(0.0, self.total_rebate_accrued - self.total_rebate_transferred)

	# ------------------------------------------------------------------
	def transfer_rebate(self, amount: float) -> Dict[str, Any]:
		"""Move up to *amount* USD from the rebate pool into trading equity.

		The transferred amount is capped at :pyattr:`available_rebate`. Equity
		(and peak equity) are bumped so subsequent trade sizing immediately
		reflects the top-up, but :pyattr:`start_equity` is left untouched so
		PnL accounting stays honest.
		"""
		req = float(amount)
		if req <= 0:
			return {
				"requested": req, "transferred": 0.0,
				"available_rebate": self.available_rebate,
				"equity": self.equity, "reason": "non_positive_amount",
			}
		moved = min(req, self.available_rebate)
		if moved <= 0:
			return {
				"requested": req, "transferred": 0.0,
				"available_rebate": self.available_rebate,
				"equity": self.equity, "reason": "no_rebate_available",
			}
		self.total_rebate_transferred += moved
		self.equity += moved
		self.peak_equity = max(self.peak_equity, self.equity)
		return {
			"requested": req, "transferred": moved,
			"available_rebate": self.available_rebate,
			"equity": self.equity, "reason": "ok",
		}

	# ------------------------------------------------------------------
	def summary(self) -> Dict[str, Any]:
		wr = (self.wins / self.round_trips * 100.0) if self.round_trips else 0.0
		# Net fees from the per-leg accrual (respects rebate_all_legs) rather
		# than a blanket gross*(1-rebate) that assumed every leg rebates.
		net_fees = self.total_fees_gross - self.total_rebate_accrued
		# fee_cover_pct: how much of gross fees the trade PnL covers
		fee_cover = (self.total_pnl / self.total_fees_gross * 100) if self.total_fees_gross > 0 else 0.0
		cost_per_1m = (net_fees / self.total_volume_usd * 1_000_000) if self.total_volume_usd > 0 else 0.0
		return {
			"equity": round(self.equity, 4),
			"start_equity": self.start_equity,
			"equity_delta": round(self.equity - self.start_equity, 4),
			"volume_usd": round(self.total_volume_usd, 2),
			"volume_target_pct": round(self.total_volume_usd / max(self._volume_target, 1e-9) * 100, 2),
			"fees_gross": round(self.total_fees_gross, 4),
			"fees_net": round(net_fees, 4),
			"net_cost_per_1m": round(cost_per_1m, 2),
			"pace": dict(self._last_pace),
			"campaign_start": self.campaign_start_iso,
			"rebate_estimate": round(self.total_rebate_accrued, 4),
			"rebate_accrued": round(self.total_rebate_accrued, 4),
			"rebate_transferred": round(self.total_rebate_transferred, 4),
			"rebate_available": round(self.available_rebate, 4),
			"total_pnl": round(self.total_pnl, 4),
			"fee_cover_pct": round(fee_cover, 2),
			"round_trips": self.round_trips,
			"wins": self.wins,
			"losses": self.losses,
			"win_rate_pct": round(wr, 2),
			"halted": self.halted,
			"halt_reason": self.halt_reason,
			"entries_considered": self._entries_considered,
			"mtf_skips": self._mtf_skips,
			"session_blocks": self._session_blocks,
		}

	# ------------------------------------------------------------------
	def save_state(self, path: pathlib.Path) -> None:
		pos_data: Optional[Dict[str, Any]] = None
		if self.position is not None:
			p = self.position
			pos_data = {
				"side": p.side,
				"entry_price": p.entry_price,
				"entry_time": p.entry_time.isoformat(),
				"notional": p.notional,
				"tp": p.tp,
				"sl": p.sl,
				"bars_held": p.bars_held,
				"tp_bps": p.tp_bps,
				"sl_bps": p.sl_bps,
			}
		state = {
			"equity": self.equity,
			"peak_equity": self.peak_equity,
			"start_equity": self.start_equity,
			"total_volume_usd": self.total_volume_usd,
			"total_fees_gross": self.total_fees_gross,
			"total_rebate_accrued": self.total_rebate_accrued,
			"total_rebate_transferred": self.total_rebate_transferred,
			"total_pnl": self.total_pnl,
			"wins": self.wins,
			"losses": self.losses,
			"round_trips": self.round_trips,
			"consec_losses": self.consec_losses,
			"cooldown_bars_left": self.cooldown_bars_left,
			"halted": self.halted,
			"halt_reason": self.halt_reason,
			"last_side": self.last_side,
			"daily_pnl": self.daily_pnl,
			"daily_pnl_date": self.daily_pnl_date,
			"day_start_equity": self.day_start_equity,
			"milestones_hit": self.milestones_hit,
			"last_rebate_reminder_at": self.last_rebate_reminder_at,
			"_rr_last_daily_date": self._rr_last_daily_date,
			"campaign_start_iso": self.campaign_start_iso,
			"position": pos_data,
			"ledger": self.ledger[-500:],  # cap
			"saved_at": datetime.now(tz=timezone.utc).isoformat(),
		}
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(json.dumps(state, indent=2), encoding="utf-8")

	def load_state(self, path: pathlib.Path) -> None:
		if not path.exists():
			return
		s = json.loads(path.read_text(encoding="utf-8"))
		for key in [
			"equity", "peak_equity", "start_equity", "total_volume_usd",
			"total_fees_gross", "total_rebate_accrued",
			"total_rebate_transferred", "total_pnl", "wins", "losses",
			"round_trips", "consec_losses", "cooldown_bars_left", "halted",
			"halt_reason", "last_side", "daily_pnl", "daily_pnl_date",
			"day_start_equity", "milestones_hit", "last_rebate_reminder_at",
			"_rr_last_daily_date", "campaign_start_iso",
		]:
			if key in s:
				setattr(self, key, s[key])
		if "ledger" in s:
			self.ledger = list(s["ledger"])
		if s.get("position") is not None:
			p = s["position"]
			self.position = _Position(
				side=str(p["side"]),
				entry_price=float(p["entry_price"]),
				entry_time=pd.Timestamp(p["entry_time"]),
				notional=float(p["notional"]),
				tp=float(p["tp"]),
				sl=float(p["sl"]),
				bars_held=int(p.get("bars_held", 0)),
				tp_bps=float(p.get("tp_bps", 0.0)),
				sl_bps=float(p.get("sl_bps", 0.0)),
			)
			LOGGER.info(
				"load_state: restored open %s position  entry=%.2f  tp=%.2f  sl=%.2f",
				self.position.side, self.position.entry_price,
				self.position.tp, self.position.sl,
			)

	def reconcile_orphan_position(self, bars: "pd.DataFrame") -> Optional[str]:
		"""Walk candle history to resolve an open position saved from a prior run.

		Filters bars to those after position entry, then applies normal
		TP/SL logic. Calls _close_position (which fires the exit event callback)
		if the level was hit. Returns "tp", "sl", or None (still open).
		"""
		if self.position is None:
			return None
		pos = self.position
		after = bars[bars["open_time"] > pos.entry_time].copy()
		if after.empty:
			LOGGER.info("reconcile: no bars after entry — position stays open")
			return None
		# Bars processed BEFORE the restart are already reflected in the
		# persisted bars_held — the replay window overlaps them (it filters by
		# entry_time, not by last-processed bar). Re-incrementing for those
		# would double-count the hold and fire the time-stop prematurely.
		already_held = max(int(pos.bars_held), 0)
		seen = 0
		for _, bar in after.iterrows():
			if not bar.get("closed", True):
				continue
			hi = float(bar["high"])
			lo = float(bar["low"])
			cl = float(bar["close"])
			bar_time = bar["open_time"]
			seen += 1
			if seen > already_held:
				pos.bars_held += 1
			tp_hit = (pos.side == "long" and hi >= pos.tp) or (pos.side == "short" and lo <= pos.tp)
			sl_hit = (pos.side == "long" and lo <= pos.sl) or (pos.side == "short" and hi >= pos.sl)
			# Same rules as the live bar loop: worst-case SL when both levels
			# touched, then TP, then SL, then the time-stop. The replay used to
			# skip the time-stop entirely, so an orphan restored after restart
			# rode to the catastrophic 6xATR taker SL — the exact loss profile
			# the 1-bar time-stop exists to prevent.
			if tp_hit and sl_hit:
				LOGGER.info("reconcile: orphan %s both-touched bar — worst-case SL at %.2f (bar %s)",
							pos.side.upper(), pos.sl, bar_time)
				self._close_position(pos.sl, bar_time, "sl_ambiguous")
				return "sl_ambiguous"
			if tp_hit:
				LOGGER.info("reconcile: orphan %s hit TP at %.2f (bar %s)", pos.side.upper(), pos.tp, bar_time)
				self._close_position(pos.tp, bar_time, "tp")
				return "tp"
			if sl_hit:
				LOGGER.info("reconcile: orphan %s hit SL at %.2f (bar %s)", pos.side.upper(), pos.sl, bar_time)
				self._close_position(pos.sl, bar_time, "sl")
				return "sl"
			if pos.bars_held >= self._max_hold:
				LOGGER.info("reconcile: orphan %s time-stopped at %.2f (bar %s)", pos.side.upper(), cl, bar_time)
				self._close_position(cl, bar_time, "time_stop")
				return "time_stop"
		LOGGER.info("reconcile: position still open (tp=%.2f sl=%.2f) — resuming", pos.tp, pos.sl)
		return None
