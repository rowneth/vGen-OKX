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

import pandas as pd

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
		self._leverage = float(f["leverage"])
		self._margin_frac = float(f["margin_fraction_per_trade"])
		self._tp_bps = float(f["tp_bps"])
		self._sl_bps = float(f["sl_bps"])
		self._max_hold = int(f["max_hold_bars"])
		entry_cfg = f.get("entry", {})
		self._entry_mode = str(entry_cfg.get("mode", "micro_momentum"))
		self._min_range_bps = float(entry_cfg.get("min_bar_range_bps", 0.0))
		self._max_range_bps = float(entry_cfg.get("max_bar_range_bps", 1_000.0))
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
	def _open_position(self, side: str, price: float, bar_time: pd.Timestamp) -> None:
		margin = self.equity * self._margin_frac
		notional = margin * self._leverage
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
				"fee": open_fee, "tp": tp, "sl": sl,
				"equity": self.equity,
				"volume": self.total_volume_usd,
				"round_trips": self.round_trips,
				"capital": self.equity,
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
		close_fee = p.notional * self._taker_rate  # market order = taker
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
			"total_fee": open_fee + close_fee,
			"net_pnl": net_pnl,
			"reason": reason,
		})
		self._emit(FarmerEvent(
			kind="exit", time=bar_time,
			payload={
				"side": p.side, "entry_price": p.entry_price,
				"exit_price": exit_price, "reason": reason,
				"gross_pnl": gross_pnl, "fee": open_fee + close_fee,
				"net_pnl": net_pnl,
				"equity": self.equity, "round_trips": self.round_trips,
				"volume": self.total_volume_usd,
				"wins": self.wins, "losses": self.losses,
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
