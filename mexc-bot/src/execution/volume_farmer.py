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
	milestones_hit: List[float] = field(default_factory=list, init=False)

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
		self._max_hold = int(f["max_hold_bars"])
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
		self._alternate = bool(f.get("alternate_direction", True))
		fees_cfg = self.config.get("fees", {})
		self._maker_rate = float(fees_cfg.get("maker", 0.0001))
		self._taker_rate = float(fees_cfg.get("taker", 0.0005))
		self._rebate_pct = float(fees_cfg.get("rebate_pct", 0.0))
		risk_cfg = self.config.get("risk", {})
		self._daily_loss_limit = float(risk_cfg.get("daily_loss_limit_pct", 0.05))
		self._max_dd = float(risk_cfg.get("max_drawdown_pct", 0.25))
		self._consec_loss_limit = int(risk_cfg.get("consecutive_losses_limit", 3))
		self._consec_loss_cooldown_bars = int(risk_cfg.get("consecutive_losses_cooldown_bars", 24))
		self._stop_on_target = bool(risk_cfg.get("stop_on_volume_target", True))
		target_cfg = self.config.get("target", {})
		self._volume_target = float(target_cfg.get("volume_usd", 1_000_000.0))
		self._milestones = [0.1, 0.25, 0.5, 0.75, 1.0]

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
		day_loss = -self.daily_pnl / max(self.start_equity, 1e-9)
		if day_loss >= self._daily_loss_limit:
			self.halted = True
			self.halt_reason = f"daily_loss {day_loss*100:.2f}%"
		# volume target reached — halt
		if self._stop_on_target and self.total_volume_usd >= self._volume_target:
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
	def _decide_entry(self, history: pd.DataFrame) -> Optional[str]:
		"""Return 'long' | 'short' | None based on entry signal."""

		if history.empty:
			return None

		# RSI + WaveTrend entry (based on WTF Pine Scripts)
		if self._entry_mode == "rsi_wt":
			return self._rsi_wt_signal(history)

		# Bollinger-band fade with RSI + optional EMA trend filter
		if self._entry_mode == "bollinger_fade":
			return self._bollinger_fade_signal(history)

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
	def _calc_leverage(self, margin: float) -> float:
		"""Dynamic leverage: lev = risk$ / (margin$ * SL_fraction).

		Mirrors the ebirth.net leverage calculator.
		Caps at exchange max; floors at min_leverage.
		"""
		if not self._dynamic_leverage or margin <= 0:
			return self._leverage if self._leverage > 0 else 20.0
		risk_usd = self.equity * self._risk_pct
		sl_frac = self._sl_bps / 10_000.0
		if sl_frac <= 0:
			return self._max_leverage
		lev = risk_usd / (margin * sl_frac)
		return max(self._min_leverage, min(lev, self._max_leverage))

	# ------------------------------------------------------------------
	def _open_position(self, side: str, price: float, bar_time: pd.Timestamp) -> None:
		margin = self.equity * self._margin_frac
		leverage = self._calc_leverage(margin)
		notional = margin * leverage
		open_fee = notional * self._maker_rate  # limit order = maker
		self.total_fees_gross += open_fee
		self.total_volume_usd += notional
		self.equity -= open_fee  # fee paid on open
		tp = price * (1 + self._tp_bps / 10_000) if side == "long" else price * (1 - self._tp_bps / 10_000)
		sl = price * (1 - self._sl_bps / 10_000) if side == "long" else price * (1 + self._sl_bps / 10_000)
		self.position = _Position(
			side=side, entry_price=price, entry_time=bar_time,
			notional=notional, tp=tp, sl=sl,
		)
		self._emit(FarmerEvent(
			kind="entry", time=bar_time,
			payload={
				"side": side, "price": price, "notional": notional,
				"margin": round(margin, 4),
				"leverage": round(leverage, 1),
				"open_fee": open_fee,
				"fee_type": "maker",
				"tp": tp, "sl": sl,
				"tp_bps": self._tp_bps,
				"sl_bps": self._sl_bps,
				"equity": self.equity,
				"volume": self.total_volume_usd,
				"volume_target": self._volume_target,
				"round_trips": self.round_trips,
				"capital": self.equity,
				"wins": self.wins,
				"losses": self.losses,
			},
		))

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
		# TP exits are limit orders (maker fee); SL/time_stop are market (taker fee)
		if reason == "tp":
			close_fee = p.notional * self._maker_rate
		else:
			close_fee = p.notional * self._taker_rate
		net_pnl = gross_pnl - close_fee
		self.total_fees_gross += close_fee
		self.total_volume_usd += p.notional
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
			self.consec_losses += 1
			if self.consec_losses >= self._consec_loss_limit:
				self.cooldown_bars_left = self._consec_loss_cooldown_bars
				self.consec_losses = 0  # reset streak so next losses start fresh
		open_fee = p.notional * self._maker_rate
		close_fee_type = "maker" if reason == "tp" else "taker"
		fee_overage = (close_fee - p.notional * self._maker_rate) if reason != "tp" else 0.0
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
			elif p.bars_held >= self._max_hold:
				self._close_position(close, bar_time, reason="time_stop")
			return  # don't open new trade on same bar we closed one

		# Step 2: flat — look for entry (respect cooldown)
		if self.cooldown_bars_left > 0:
			self.cooldown_bars_left -= 1
			return
		side = self._decide_entry(history)
		if side is None:
			return
		self._open_position(side, close, bar_time)

	# ------------------------------------------------------------------
	def summary(self) -> Dict[str, Any]:
		wr = (self.wins / self.round_trips * 100.0) if self.round_trips else 0.0
		net_fees = self.total_fees_gross * (1.0 - self._rebate_pct)
		# fee_cover_pct: how much of gross fees the trade PnL covers
		fee_cover = (self.total_pnl / self.total_fees_gross * 100) if self.total_fees_gross > 0 else 0.0
		return {
			"equity": round(self.equity, 4),
			"start_equity": self.start_equity,
			"equity_delta": round(self.equity - self.start_equity, 4),
			"volume_usd": round(self.total_volume_usd, 2),
			"volume_target_pct": round(self.total_volume_usd / max(self._volume_target, 1e-9) * 100, 2),
			"fees_gross": round(self.total_fees_gross, 4),
			"fees_net": round(net_fees, 4),
			"rebate_estimate": round(self.total_fees_gross * self._rebate_pct, 4),
			"total_pnl": round(self.total_pnl, 4),
			"fee_cover_pct": round(fee_cover, 2),
			"round_trips": self.round_trips,
			"wins": self.wins,
			"losses": self.losses,
			"win_rate_pct": round(wr, 2),
			"halted": self.halted,
			"halt_reason": self.halt_reason,
		}

	# ------------------------------------------------------------------
	def save_state(self, path: pathlib.Path) -> None:
		state = {
			"equity": self.equity,
			"peak_equity": self.peak_equity,
			"start_equity": self.start_equity,
			"total_volume_usd": self.total_volume_usd,
			"total_fees_gross": self.total_fees_gross,
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
			"milestones_hit": self.milestones_hit,
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
			"total_fees", "total_pnl", "wins", "losses", "round_trips",
			"consec_losses", "cooldown_bars_left", "halted", "halt_reason",
			"last_side", "daily_pnl", "daily_pnl_date", "milestones_hit",
		]:
			if key in s:
				setattr(self, key, s[key])
		if "ledger" in s:
			self.ledger = list(s["ledger"])
