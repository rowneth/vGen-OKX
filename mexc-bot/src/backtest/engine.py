"""Event-driven backtesting engine for strategy validation."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from risk.limits import RiskLimits
from risk.sizing import risk_based_position_size
from strategy.base import Strategy, TradeSignal

LOGGER = logging.getLogger(__name__)


@dataclass
class Position:
	"""Represents an open backtest position."""

	side: str
	qty: float
	entry_index: int
	entry_time: datetime
	entry_price: float
	stop_price: float
	take_profit_price: float
	entry_fee: float


@dataclass
class BacktestResult:
	"""Backtest run outputs."""

	trades: pd.DataFrame
	equity_curve: pd.DataFrame
	decisions: pd.DataFrame
	summary: Dict[str, float]


class BacktestEngine:
	"""Runs a deterministic bar-by-bar backtest simulation."""

	def __init__(self, config: Dict[str, object], seed: int = 42) -> None:
		"""Initialize backtest engine.

		Args:
			config: Global configuration dictionary.
			seed: Random seed for deterministic fill simulation.
		"""
		self.config = config
		self.rng = random.Random(seed)

	def run(self, candles: pd.DataFrame, strategy: Strategy) -> BacktestResult:
		"""Run backtest on provided candles.

		Args:
			candles: OHLCV dataframe.
			strategy: Strategy object.

		Returns:
			BacktestResult with trades, curve, and summary stats.
		"""
		frame = strategy.prepare(candles).reset_index(drop=True)

		initial_equity = float(self.config["backtest"]["initial_equity"])
		equity = initial_equity
		fees_paid = 0.0

		risk_cfg = self.config["risk"]
		limits = RiskLimits(
			daily_drawdown_limit_pct=float(risk_cfg["daily_drawdown_limit_pct"]),
			consecutive_losses_limit=int(risk_cfg["consecutive_losses_limit"]),
			consecutive_losses_pause_hours=int(risk_cfg["consecutive_losses_pause_hours"]),
		)

		first_time = _row_time(frame.iloc[0])
		risk_state = limits.reset_for_day(now=first_time, equity=equity)

		open_position: Optional[Position] = None
		trades: List[Dict[str, object]] = []
		decisions: List[Dict[str, object]] = []
		equity_points: List[Dict[str, object]] = []

		avg_vol_lookback = int(self.config["backtest"]["average_volume_lookback"])
		max_spread_pct = float(risk_cfg["max_spread_pct"])

		for i in range(len(frame)):
			row = frame.iloc[i]
			now = _row_time(row)

			if now.date() != risk_state.day:
				risk_state = limits.reset_for_day(now=now, equity=equity)
				decisions.append({"time": now, "type": "day_reset", "message": "Daily risk reset"})

			if open_position is not None:
				close_record = self._try_close_position(
					frame=frame,
					index=i,
					position=open_position,
					avg_vol_lookback=avg_vol_lookback,
				)
				if close_record is not None:
					gross_pnl = float(close_record["gross_pnl"])
					exit_fee = float(close_record["exit_fee"])
					net_pnl = gross_pnl - open_position.entry_fee - exit_fee
					fees_paid += open_position.entry_fee + exit_fee
					equity += net_pnl

					close_record["entry_fee"] = open_position.entry_fee
					close_record["net_pnl"] = net_pnl
					close_record["equity_after"] = equity
					trades.append(close_record)
					limits.register_trade_result(risk_state, now=now, pnl=net_pnl)
					decisions.append(
						{
							"time": now,
							"type": "exit",
							"message": f"Closed {open_position.side} on {close_record['reason']}",
						}
					)
					open_position = None

			equity_points.append(
				{
					"time": now,
					"equity": equity,
					"close": float(row["close"]),
				}
			)

			if open_position is not None:
				continue

			if limits.check_daily_drawdown(risk_state, equity):
				decisions.append({"time": now, "type": "risk_block", "message": "Daily drawdown limit hit"})
				continue

			if limits.is_paused(risk_state, now):
				decisions.append({"time": now, "type": "risk_block", "message": "Consecutive loss pause active"})
				continue

			if i >= len(frame) - 1:
				continue

			signal = strategy.generate_signal(frame, i)
			if signal is None:
				continue

			spread_pct = self._estimate_spread_pct(frame.iloc[i])
			if spread_pct > max_spread_pct:
				decisions.append(
					{
						"time": now,
						"type": "signal_reject",
						"message": f"Spread filter failed ({spread_pct:.6f})",
					}
				)
				continue

			qty = risk_based_position_size(
				equity=equity,
				risk_per_trade_pct=float(risk_cfg["risk_per_trade_pct"]),
				entry_price=signal.entry_price,
				stop_price=signal.stop_price,
				max_leverage=float(risk_cfg["max_leverage"]),
			)
			if qty <= 0:
				decisions.append({"time": now, "type": "signal_reject", "message": "Zero position size"})
				continue

			entry_fill = self._simulate_entry_fill(
				frame=frame,
				signal=signal,
				avg_vol_lookback=avg_vol_lookback,
			)
			if entry_fill is None:
				decisions.append(
					{
						"time": now,
						"type": "signal_reject",
						"message": "Maker entry not filled; canceled",
					}
				)
				continue

			entry_index = int(entry_fill["entry_index"])
			entry_row = frame.iloc[entry_index]
			entry_price = float(entry_fill["entry_price"])

			maker_fee_rate = float(self.config["fees"]["maker"])
			entry_fee = abs(entry_price * qty) * maker_fee_rate

			open_position = Position(
				side=signal.side,
				qty=qty,
				entry_index=entry_index,
				entry_time=_row_time(entry_row),
				entry_price=entry_price,
				stop_price=signal.stop_price,
				take_profit_price=signal.take_profit_price,
				entry_fee=entry_fee,
			)
			decisions.append(
				{
					"time": _row_time(entry_row),
					"type": "entry",
					"message": f"Entered {signal.side} qty={qty:.6f} price={entry_price:.2f}",
				}
			)

		if open_position is not None:
			last = frame.iloc[-1]
			side = open_position.side
			close_price = float(last["close"])
			close_price = self._apply_slippage(
				price=close_price,
				side="sell" if side == "long" else "buy",
				low_volume=False,
			)
			gross = self._compute_gross_pnl(
				side=side,
				qty=open_position.qty,
				entry_price=open_position.entry_price,
				exit_price=close_price,
			)
			taker_fee = float(self.config["fees"]["taker"])
			exit_fee = abs(close_price * open_position.qty) * taker_fee
			net = gross - open_position.entry_fee - exit_fee
			equity += net
			fees_paid += open_position.entry_fee + exit_fee
			trades.append(
				{
					"entry_time": open_position.entry_time,
					"exit_time": _row_time(last),
					"side": side,
					"qty": open_position.qty,
					"entry_price": open_position.entry_price,
					"exit_price": close_price,
					"reason": "end_of_data",
					"gross_pnl": gross,
					"entry_fee": open_position.entry_fee,
					"exit_fee": exit_fee,
					"net_pnl": net,
					"bars_held": len(frame) - 1 - open_position.entry_index,
					"equity_after": equity,
				}
			)

		trades_df = pd.DataFrame(trades)
		equity_df = pd.DataFrame(equity_points)
		decisions_df = pd.DataFrame(decisions)
		summary = {
			"initial_equity": initial_equity,
			"final_equity": equity,
			"total_return_pct": ((equity / initial_equity) - 1.0) * 100.0,
			"total_fees_paid": fees_paid,
			"total_trades": float(len(trades_df)),
		}
		return BacktestResult(
			trades=trades_df,
			equity_curve=equity_df,
			decisions=decisions_df,
			summary=summary,
		)

	def _simulate_entry_fill(
		self,
		frame: pd.DataFrame,
		signal: TradeSignal,
		avg_vol_lookback: int,
	) -> Optional[Dict[str, float]]:
		"""Simulate maker-entry fill probability on next candle.

		Args:
			frame: Backtest frame.
			signal: Candidate trade signal.
			avg_vol_lookback: Lookback for average volume.

		Returns:
			Filled entry details or None.
		"""
		entry_index = signal.entry_index + 1
		if entry_index >= len(frame):
			return None

		row = frame.iloc[entry_index]
		entry_price = signal.entry_price
		tick_size = float(self.config["exchange"].get("tick_size", 0.1))

		candle_low = float(row["low"])
		candle_high = float(row["high"])
		candle_range = max(tick_size, candle_high - candle_low)

		if signal.side == "long" and candle_low > entry_price:
			return None
		if signal.side == "short" and candle_high < entry_price:
			return None

		if signal.side == "long":
			depth_ratio = max(0.0, min(1.0, (entry_price - candle_low) / candle_range))
		else:
			depth_ratio = max(0.0, min(1.0, (candle_high - entry_price) / candle_range))

		maker_model = self.config["execution"]["maker_fill_model"]
		base_prob = float(maker_model["base_fill_probability"])
		min_prob = float(maker_model["min_fill_probability"])
		max_prob = float(maker_model["max_fill_probability"])
		fill_prob = max(min_prob, min(max_prob, base_prob + 0.35 * depth_ratio))
		if self.rng.random() > fill_prob:
			return None

		low_volume = self._is_low_volume(frame, entry_index, avg_vol_lookback)
		side = "buy" if signal.side == "long" else "sell"
		fill_price = self._apply_slippage(entry_price, side=side, low_volume=low_volume)
		return {"entry_index": float(entry_index), "entry_price": fill_price}

	def _try_close_position(
		self,
		frame: pd.DataFrame,
		index: int,
		position: Position,
		avg_vol_lookback: int,
	) -> Optional[Dict[str, object]]:
		"""Close position if any exit rule is hit.

		Args:
			frame: Backtest frame.
			index: Current bar index.
			position: Open position.
			avg_vol_lookback: Lookback for low volume slippage flag.

		Returns:
			Close record if position exits on this bar.
		"""
		row = frame.iloc[index]
		low = float(row["low"])
		high = float(row["high"])
		close = float(row["close"])

		time_stop_bars = int(self.config["strategy"]["exits"]["time_stop_candles"])
		bars_held = index - position.entry_index

		reason = ""
		exit_price = close
		fee_rate = float(self.config["fees"]["taker"])

		if position.side == "long":
			stop_hit = low <= position.stop_price
			tp_hit = high >= position.take_profit_price
			if stop_hit:
				reason = "stop_loss"
				exit_price = position.stop_price
				fee_rate = float(self.config["fees"]["taker"])
			elif tp_hit:
				reason = "take_profit"
				exit_price = position.take_profit_price
				fee_rate = float(self.config["fees"]["maker"])
			elif bars_held >= time_stop_bars:
				reason = "time_stop"
				exit_price = close
				fee_rate = float(self.config["fees"]["taker"])
		else:
			stop_hit = high >= position.stop_price
			tp_hit = low <= position.take_profit_price
			if stop_hit:
				reason = "stop_loss"
				exit_price = position.stop_price
				fee_rate = float(self.config["fees"]["taker"])
			elif tp_hit:
				reason = "take_profit"
				exit_price = position.take_profit_price
				fee_rate = float(self.config["fees"]["maker"])
			elif bars_held >= time_stop_bars:
				reason = "time_stop"
				exit_price = close
				fee_rate = float(self.config["fees"]["taker"])

		if not reason:
			return None

		low_volume = self._is_low_volume(frame, index, avg_vol_lookback)
		exit_side = "sell" if position.side == "long" else "buy"
		slipped_exit = self._apply_slippage(exit_price, side=exit_side, low_volume=low_volume)
		gross = self._compute_gross_pnl(
			side=position.side,
			qty=position.qty,
			entry_price=position.entry_price,
			exit_price=slipped_exit,
		)
		exit_fee = abs(slipped_exit * position.qty) * fee_rate
		return {
			"entry_time": position.entry_time,
			"exit_time": _row_time(row),
			"side": position.side,
			"qty": position.qty,
			"entry_price": position.entry_price,
			"exit_price": slipped_exit,
			"reason": reason,
			"gross_pnl": gross,
			"exit_fee": exit_fee,
			"bars_held": bars_held,
		}

	def _is_low_volume(self, frame: pd.DataFrame, index: int, lookback: int) -> bool:
		"""Detect low-volume condition for extra slippage tick.

		Args:
			frame: Backtest frame.
			index: Current bar index.
			lookback: Number of bars for average volume.

		Returns:
			True if current bar volume is below threshold.
		"""
		if index <= 0:
			return False
		start = max(0, index - lookback)
		window = frame.iloc[start:index]
		if window.empty:
			return False
		avg_volume = float(window["volume"].mean())
		ratio = float(self.config["execution"]["slippage"]["low_volume_threshold_ratio"])
		current_volume = float(frame.iloc[index]["volume"])
		return current_volume < avg_volume * ratio

	def _apply_slippage(self, price: float, side: str, low_volume: bool) -> float:
		"""Apply directional slippage in ticks.

		Args:
			price: Raw price.
			side: ``buy`` or ``sell``.
			low_volume: Whether to add one extra slippage tick.

		Returns:
			Slippage-adjusted price.
		"""
		slip_cfg = self.config["execution"]["slippage"]
		ticks = int(slip_cfg["base_ticks_per_fill"])
		if low_volume:
			ticks += int(slip_cfg["extra_ticks_if_low_volume"])
		tick_size = float(self.config["exchange"].get("tick_size", 0.1))
		distance = ticks * tick_size

		if side == "buy":
			return price + distance
		if side == "sell":
			return max(0.0, price - distance)
		raise ValueError("side must be buy or sell")

	def _estimate_spread_pct(self, row: pd.Series) -> float:
		"""Estimate spread percentage from candle range proxy.

		Args:
			row: Candle row.

		Returns:
			Estimated spread percentage.
		"""
		high = float(row["high"])
		low = float(row["low"])
		close = float(row["close"])
		if close <= 0:
			return 1.0
		range_pct = (high - low) / close
		return max(0.00005, range_pct * 0.01)

	@staticmethod
	def _compute_gross_pnl(side: str, qty: float, entry_price: float, exit_price: float) -> float:
		"""Compute gross PnL before fees.

		Args:
			side: ``long`` or ``short``.
			qty: Position quantity.
			entry_price: Entry price.
			exit_price: Exit price.

		Returns:
			Gross PnL.
		"""
		if side == "long":
			return (exit_price - entry_price) * qty
		if side == "short":
			return (entry_price - exit_price) * qty
		raise ValueError("side must be long or short")


def _row_time(row: pd.Series) -> datetime:
	"""Extract timestamp from canonical candle row."""
	if "close_time" in row and pd.notna(row["close_time"]):
		return pd.Timestamp(row["close_time"]).to_pydatetime()
	return pd.Timestamp(row["open_time"]).to_pydatetime()
