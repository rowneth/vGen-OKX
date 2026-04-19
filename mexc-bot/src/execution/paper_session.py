"""Stateful paper trading session.

Mirrors the backtest engine's exit logic (TP1 → TP2 → runner, with
breakeven-after-TP1 and time-stop) but operates bar-by-bar so it can run
live against a streaming/poll-based MEXC kline source.

The session is fully in-memory; callers should call :meth:`snapshot` /
:meth:`load_snapshot` to persist/restore across restarts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import pandas as pd

from risk.sizing import risk_based_position_size
from strategy.base import Strategy, TradeSignal

LOGGER = logging.getLogger(__name__)


EventCallback = Callable[["PaperEvent"], Awaitable[None]]


@dataclass(frozen=True)
class PaperEvent:
	"""One lifecycle event emitted by the session."""

	kind: str  # signal|entry|tp1|tp2|exit|error
	time: datetime
	payload: Dict[str, Any]


@dataclass
class PaperPosition:
	"""In-memory open position state."""

	side: str
	symbol: str
	qty: float
	initial_qty: float
	entry_index: int
	entry_time: datetime
	entry_price: float
	stop_price: float
	take_profit_price: float
	tp1_price: Optional[float]
	tp1_qty: float
	tp2_price: Optional[float]
	tp2_qty: float
	move_stop_to_be: bool
	tp1_hit: bool = False
	tp1_hit_index: int = -1
	tp2_hit: bool = False
	tp2_hit_index: int = -1
	realized_partial_pnl: float = 0.0
	realized_partial_fee: float = 0.0
	entry_fee: float = 0.0


@dataclass
class PaperTradeLedgerEntry:
	"""One closed trade in the session ledger."""

	entry_time: str
	exit_time: str
	symbol: str
	side: str
	qty: float
	initial_qty: float
	entry_price: float
	exit_price: float
	reason: str
	gross_pnl: float
	partial_pnl: float
	entry_fee: float
	exit_fee: float
	net_pnl: float
	equity_after: float
	bars_held: int
	tp1_hit: bool
	tp2_hit: bool
	notional: float


class PaperSession:
	"""Event-driven paper trading session.

	Usage::

	    session = PaperSession(config, strategy, event_callback=...)
	    session.seed(history_df)             # warm up indicators
	    await session.on_new_candle(df)      # call each time a 15m bar closes
	"""

	def __init__(
		self,
		config: Dict[str, Any],
		strategy: Strategy,
		event_callback: Optional[EventCallback] = None,
	) -> None:
		self._config = config
		self._strategy = strategy
		self._event_cb = event_callback

		bt = config.get("backtest", {})
		risk = config.get("risk", {})
		fees = config.get("fees", {})
		exits = config.get("strategy", {}).get("exits", {})

		self._initial_equity = float(bt.get("initial_equity", 10_000.0))
		self.equity: float = self._initial_equity
		self._risk_pct = float(risk.get("risk_per_trade_pct", 0.01))
		self._max_leverage = float(risk.get("max_leverage", 5))
		self._time_stop_bars = int(exits.get("time_stop_candles", 10))
		self._maker_fee = float(fees.get("maker", 0.0001))
		self._taker_fee = float(fees.get("taker", 0.0005))
		self._symbol = str(config.get("exchange", {}).get("symbol", "BTC_USDT"))

		self.position: Optional[PaperPosition] = None
		self.ledger: List[PaperTradeLedgerEntry] = []
		self._last_processed_time: Optional[pd.Timestamp] = None

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------
	async def on_new_candle(self, history: pd.DataFrame) -> None:
		"""Process one newly-closed candle.

		Args:
			history: Dataframe containing at least the last ~250 closed candles
				ending with the bar that just closed. Must include columns
				``open_time, open, high, low, close, volume``.
		"""
		if len(history) < 2:
			return

		frame = self._strategy.prepare(history).reset_index(drop=True)
		last = frame.iloc[-1]
		now = _row_time(last)

		if self._last_processed_time is not None and now <= self._last_processed_time:
			return  # duplicate / out-of-order bar
		self._last_processed_time = now

		# 1. Manage open position first.
		if self.position is not None:
			await self._process_exits(frame=frame, index=len(frame) - 1)

		# 2. If flat, look for a new signal on this closed bar.
		if self.position is None:
			signal = self._strategy.generate_signal(frame, len(frame) - 1)
			if signal is not None:
				await self._open_position(frame=frame, index=len(frame) - 1, signal=signal)

	def snapshot(self) -> Dict[str, Any]:
		"""Serializable snapshot of session state."""

		return {
			"equity": self.equity,
			"position": asdict(self.position) if self.position else None,
			"ledger": [asdict(t) for t in self.ledger],
			"last_processed_time": (
				self._last_processed_time.isoformat() if self._last_processed_time else None
			),
		}

	def save_state(self, path: Path) -> None:
		"""Persist snapshot to JSON."""

		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(json.dumps(self.snapshot(), default=_json_default, indent=2))

	def load_state(self, path: Path) -> None:
		"""Restore snapshot from JSON (best-effort)."""

		if not path.exists():
			return
		data = json.loads(path.read_text())
		self.equity = float(data.get("equity", self._initial_equity))
		self.ledger = [PaperTradeLedgerEntry(**row) for row in data.get("ledger", [])]
		pos = data.get("position")
		if pos:
			pos = dict(pos)
			pos["entry_time"] = _parse_time(pos["entry_time"])
			self.position = PaperPosition(**pos)
		lpt = data.get("last_processed_time")
		self._last_processed_time = pd.Timestamp(lpt) if lpt else None

	def daily_summary(self, date_utc: str) -> Dict[str, float]:
		"""Compute per-day stats from the ledger.

		Args:
			date_utc: Date in ``YYYY-MM-DD`` UTC.
		"""

		day_trades = [t for t in self.ledger if t.exit_time.startswith(date_utc)]
		wins = sum(1 for t in day_trades if t.net_pnl > 0)
		losses = sum(1 for t in day_trades if t.net_pnl < 0)
		gross = sum(t.gross_pnl + t.partial_pnl for t in day_trades)
		fees = sum(t.entry_fee + t.exit_fee for t in day_trades)
		net = sum(t.net_pnl for t in day_trades)
		volume = sum(t.notional for t in day_trades)
		eq_start = day_trades[0].equity_after - day_trades[0].net_pnl if day_trades else self.equity
		eq_end = day_trades[-1].equity_after if day_trades else self.equity
		wr = wins / len(day_trades) if day_trades else 0.0
		return {
			"trades": len(day_trades),
			"wins": wins,
			"losses": losses,
			"win_rate": wr,
			"gross_pnl": gross,
			"fees": fees,
			"net_pnl": net,
			"equity_start": eq_start,
			"equity_end": eq_end,
			"volume": volume,
		}

	# ------------------------------------------------------------------
	# Internals
	# ------------------------------------------------------------------
	async def _emit(self, event: PaperEvent) -> None:
		if self._event_cb is None:
			return
		try:
			await self._event_cb(event)
		except Exception as exc:  # noqa: BLE001
			LOGGER.error("Event callback failed: %s", exc)

	async def _open_position(
		self,
		frame: pd.DataFrame,
		index: int,
		signal: TradeSignal,
	) -> None:
		row = frame.iloc[index]
		entry_price = float(signal.entry_price)
		qty = risk_based_position_size(
			equity=self.equity,
			risk_per_trade_pct=self._risk_pct,
			entry_price=entry_price,
			stop_price=signal.stop_price,
			max_leverage=self._max_leverage,
		)
		if qty <= 0:
			return

		tp1_qty = qty * float(signal.tp1_size_fraction or 0.0) if signal.tp1_price else 0.0
		tp2_qty = qty * float(signal.tp2_size_fraction or 0.0) if signal.tp2_price else 0.0
		entry_fee = abs(entry_price * qty) * self._maker_fee

		self.position = PaperPosition(
			side=signal.side,
			symbol=self._symbol,
			qty=qty,
			initial_qty=qty,
			entry_index=index,
			entry_time=_row_time(row),
			entry_price=entry_price,
			stop_price=float(signal.stop_price),
			take_profit_price=float(signal.take_profit_price),
			tp1_price=float(signal.tp1_price) if signal.tp1_price is not None else None,
			tp1_qty=tp1_qty,
			tp2_price=float(signal.tp2_price) if signal.tp2_price is not None else None,
			tp2_qty=tp2_qty,
			move_stop_to_be=bool(signal.move_stop_to_breakeven_after_tp1),
			entry_fee=entry_fee,
		)

		payload = {
			"side": signal.side,
			"symbol": self._symbol,
			"entry_price": entry_price,
			"stop_price": float(signal.stop_price),
			"take_profit_price": float(signal.take_profit_price),
			"tp1_price": signal.tp1_price,
			"tp2_price": signal.tp2_price,
			"qty": qty,
			"equity": self.equity,
		}
		await self._emit(PaperEvent(kind="signal", time=_row_time(row), payload=payload))
		await self._emit(
			PaperEvent(
				kind="entry",
				time=_row_time(row),
				payload={**payload, "fill_price": entry_price},
			)
		)

	async def _process_exits(self, frame: pd.DataFrame, index: int) -> None:
		pos = self.position
		assert pos is not None
		row = frame.iloc[index]
		high = float(row["high"])
		low = float(row["low"])
		close = float(row["close"])
		now = _row_time(row)

		# --- Partial TP1 ---
		if (
			not pos.tp1_hit
			and pos.tp1_price is not None
			and pos.tp1_qty > 0.0
		):
			hit = (
				(pos.side == "long" and high >= pos.tp1_price)
				or (pos.side == "short" and low <= pos.tp1_price)
			)
			if hit:
				partial_qty = min(pos.tp1_qty, pos.qty)
				partial_pnl = _pnl(pos.side, partial_qty, pos.entry_price, pos.tp1_price)
				partial_fee = abs(pos.tp1_price * partial_qty) * self._maker_fee
				pos.realized_partial_pnl += partial_pnl
				pos.realized_partial_fee += partial_fee
				pos.qty = max(0.0, pos.qty - partial_qty)
				pos.tp1_hit = True
				pos.tp1_hit_index = index
				await self._emit(
					PaperEvent(
						kind="tp1",
						time=now,
						payload={
							"symbol": pos.symbol,
							"side": pos.side,
							"price": pos.tp1_price,
							"partial_qty": partial_qty,
							"partial_pnl": partial_pnl - partial_fee,
							"new_stop": pos.entry_price if pos.move_stop_to_be else pos.stop_price,
						},
					)
				)
				if pos.qty <= 1e-12:
					await self._close_out(pos, now, pos.tp1_price, "take_profit_partial_only", index)
					return
				# Defer any further checks to the next bar.
				return

		# --- Partial TP2 ---
		if (
			pos.tp1_hit
			and not pos.tp2_hit
			and pos.tp2_price is not None
			and pos.tp2_qty > 0.0
			and pos.qty > 1e-12
		):
			hit = (
				(pos.side == "long" and high >= pos.tp2_price)
				or (pos.side == "short" and low <= pos.tp2_price)
			)
			if hit:
				partial_qty = min(pos.tp2_qty, pos.qty)
				partial_pnl = _pnl(pos.side, partial_qty, pos.entry_price, pos.tp2_price)
				partial_fee = abs(pos.tp2_price * partial_qty) * self._maker_fee
				pos.realized_partial_pnl += partial_pnl
				pos.realized_partial_fee += partial_fee
				pos.qty = max(0.0, pos.qty - partial_qty)
				pos.tp2_hit = True
				pos.tp2_hit_index = index
				new_stop = pos.tp1_price if pos.tp1_price is not None else pos.entry_price
				await self._emit(
					PaperEvent(
						kind="tp2",
						time=now,
						payload={
							"symbol": pos.symbol,
							"side": pos.side,
							"price": pos.tp2_price,
							"partial_qty": partial_qty,
							"partial_pnl": partial_pnl - partial_fee,
							"new_stop": new_stop,
						},
					)
				)
				if pos.qty <= 1e-12:
					await self._close_out(pos, now, pos.tp2_price, "take_profit_partial_only", index)
					return
				pos.stop_price = new_stop
				return

		# --- Apply breakeven stop shift once TP1 fired on a prior bar ---
		if pos.tp1_hit and pos.move_stop_to_be and index > pos.tp1_hit_index:
			if pos.side == "long" and pos.stop_price < pos.entry_price:
				pos.stop_price = pos.entry_price
			elif pos.side == "short" and pos.stop_price > pos.entry_price:
				pos.stop_price = pos.entry_price

		# --- Final exits: stop > final TP > time-stop ---
		bars_held = index - pos.entry_index
		reason = ""
		exit_price = close
		if pos.side == "long":
			if low <= pos.stop_price:
				reason, exit_price = "stop_loss", pos.stop_price
			elif high >= pos.take_profit_price:
				reason, exit_price = "take_profit", pos.take_profit_price
			elif bars_held >= self._time_stop_bars:
				reason, exit_price = "time_stop", close
		else:
			if high >= pos.stop_price:
				reason, exit_price = "stop_loss", pos.stop_price
			elif low <= pos.take_profit_price:
				reason, exit_price = "take_profit", pos.take_profit_price
			elif bars_held >= self._time_stop_bars:
				reason, exit_price = "time_stop", close

		if not reason:
			return

		await self._close_out(pos, now, exit_price, reason, index)

	async def _close_out(
		self,
		pos: PaperPosition,
		now: datetime,
		exit_price: float,
		reason: str,
		index: int,
	) -> None:
		fee_rate = self._maker_fee if reason == "take_profit" else self._taker_fee
		gross = _pnl(pos.side, pos.qty, pos.entry_price, exit_price)
		exit_fee = abs(exit_price * pos.qty) * fee_rate
		total_gross = gross + pos.realized_partial_pnl
		total_exit_fee = exit_fee + pos.realized_partial_fee
		net = total_gross - pos.entry_fee - total_exit_fee
		self.equity += net

		entry_notional = abs(pos.entry_price * pos.initial_qty)
		tp1_notional = abs((pos.tp1_price or 0.0) * (pos.tp1_qty if pos.tp1_hit else 0.0))
		tp2_notional = abs((pos.tp2_price or 0.0) * (pos.tp2_qty if pos.tp2_hit else 0.0))
		exit_notional = abs(exit_price * pos.qty)
		notional = entry_notional + tp1_notional + tp2_notional + exit_notional

		entry = PaperTradeLedgerEntry(
			entry_time=pos.entry_time.isoformat(),
			exit_time=now.isoformat(),
			symbol=pos.symbol,
			side=pos.side,
			qty=pos.qty,
			initial_qty=pos.initial_qty,
			entry_price=pos.entry_price,
			exit_price=exit_price,
			reason=reason,
			gross_pnl=total_gross,
			partial_pnl=pos.realized_partial_pnl,
			entry_fee=pos.entry_fee,
			exit_fee=total_exit_fee,
			net_pnl=net,
			equity_after=self.equity,
			bars_held=index - pos.entry_index,
			tp1_hit=pos.tp1_hit,
			tp2_hit=pos.tp2_hit,
			notional=notional,
		)
		self.ledger.append(entry)
		self.position = None

		pct_vs_start = (net / self._initial_equity) if self._initial_equity > 0 else 0.0
		await self._emit(
			PaperEvent(
				kind="exit",
				time=now,
				payload={
					"symbol": pos.symbol,
					"side": pos.side,
					"reason": reason,
					"exit_price": exit_price,
					"net_pnl": net,
					"total_pnl_pct": pct_vs_start,
					"equity_after": self.equity,
					"bars_held": index - pos.entry_index,
				},
			)
		)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _pnl(side: str, qty: float, entry: float, exit_price: float) -> float:
	if side == "long":
		return (exit_price - entry) * qty
	return (entry - exit_price) * qty


def _row_time(row: pd.Series) -> datetime:
	for key in ("open_time", "close_time", "time"):
		if key in row.index and pd.notna(row[key]):
			ts = pd.Timestamp(row[key])
			if ts.tzinfo is None:
				ts = ts.tz_localize("UTC")
			return ts.to_pydatetime()
	return datetime.now(tz=timezone.utc)


def _parse_time(value: Any) -> datetime:
	ts = pd.Timestamp(value)
	if ts.tzinfo is None:
		ts = ts.tz_localize("UTC")
	return ts.to_pydatetime()


def _json_default(value: Any) -> Any:
	if isinstance(value, (datetime, pd.Timestamp)):
		return pd.Timestamp(value).isoformat()
	raise TypeError(f"Not serializable: {type(value)!r}")


__all__ = [
	"PaperEvent",
	"PaperPosition",
	"PaperSession",
	"PaperTradeLedgerEntry",
]
