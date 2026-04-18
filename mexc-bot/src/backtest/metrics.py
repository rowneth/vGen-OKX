"""Backtest performance metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult


def calculate_metrics(
	result: BacktestResult,
	win_rate_drop_pp: float = 5.0,
) -> Dict[str, object]:
	"""Calculate key strategy and risk metrics from backtest output.

	Args:
		result: Backtest result object.
		win_rate_drop_pp: Stress-test win-rate drop in percentage points.

	Returns:
		Dictionary with scalar metrics and dataframe breakdowns.
	"""
	trades = result.trades.copy()
	equity = result.equity_curve.copy()

	if not equity.empty:
		equity["time"] = pd.to_datetime(equity["time"], utc=True)
		equity = equity.sort_values("time").reset_index(drop=True)
		equity["returns"] = equity["equity"].pct_change().fillna(0.0)
	else:
		equity = pd.DataFrame(columns=["time", "equity", "close", "returns"])

	if not trades.empty:
		trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
		trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
		trades = trades.sort_values("exit_time").reset_index(drop=True)

	total_trades = len(trades)
	winning = trades[trades["net_pnl"] > 0] if total_trades else trades
	losing = trades[trades["net_pnl"] < 0] if total_trades else trades

	win_rate = float(len(winning) / total_trades) if total_trades else 0.0
	avg_trade = float(trades["net_pnl"].mean()) if total_trades else 0.0
	avg_winner = float(winning["net_pnl"].mean()) if len(winning) else 0.0
	avg_loser = float(losing["net_pnl"].mean()) if len(losing) else 0.0

	gross_profit = float(winning["net_pnl"].sum()) if len(winning) else 0.0
	gross_loss = float(losing["net_pnl"].sum()) if len(losing) else 0.0
	profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else float("inf")

	sharpe = _annualized_sharpe(equity["returns"].to_numpy(dtype=float)) if not equity.empty else 0.0
	max_dd = _max_drawdown(equity["equity"].to_numpy(dtype=float)) if not equity.empty else 0.0

	monthly_returns = _monthly_returns_table(equity)
	trades_per_day = _trades_per_day_distribution(trades)
	win_rate_by_month = _win_rate_by_month(trades)

	total_fees = float(trades["entry_fee"].sum() + trades["exit_fee"].sum()) if total_trades else 0.0
	stress = _stress_test(
		initial_equity=float(result.summary["initial_equity"]),
		total_trades=total_trades,
		win_rate=win_rate,
		avg_winner=avg_winner,
		avg_loser=avg_loser,
		win_rate_drop_pp=win_rate_drop_pp,
	)

	return {
		"initial_equity": float(result.summary["initial_equity"]),
		"final_equity": float(result.summary["final_equity"]),
		"total_return_pct": float(result.summary["total_return_pct"]),
		"sharpe_ratio": sharpe,
		"max_drawdown_pct": max_dd,
		"win_rate": win_rate,
		"profit_factor": profit_factor,
		"average_trade": avg_trade,
		"average_winner": avg_winner,
		"average_loser": avg_loser,
		"total_fees_paid": total_fees,
		"total_trades": total_trades,
		"monthly_returns": monthly_returns,
		"trades_per_day_distribution": trades_per_day,
		"win_rate_by_month": win_rate_by_month,
		"stress_test": stress,
	}


def _annualized_sharpe(returns: np.ndarray) -> float:
	if returns.size == 0:
		return 0.0
	mean = float(np.mean(returns))
	std = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
	if std == 0.0:
		return 0.0
	periods_per_year = 365 * 24 * 4
	return (mean / std) * np.sqrt(periods_per_year)


def _max_drawdown(equity: np.ndarray) -> float:
	if equity.size == 0:
		return 0.0
	peaks = np.maximum.accumulate(equity)
	drawdowns = (equity - peaks) / peaks
	return abs(float(drawdowns.min()))


def _monthly_returns_table(equity: pd.DataFrame) -> pd.DataFrame:
	if equity.empty:
		return pd.DataFrame(columns=["month", "return_pct"])
	monthly = equity.set_index("time")["equity"].resample("ME").last().dropna()
	monthly_ret = monthly.pct_change().dropna() * 100.0
	out = monthly_ret.reset_index()
	out.columns = ["month", "return_pct"]
	out["month"] = out["month"].dt.strftime("%Y-%m")
	return out


def _trades_per_day_distribution(trades: pd.DataFrame) -> pd.DataFrame:
	if trades.empty:
		return pd.DataFrame(columns=["trades_in_day", "days"])
	per_day = trades.groupby(trades["exit_time"].dt.date).size()
	dist = per_day.value_counts().sort_index().reset_index()
	dist.columns = ["trades_in_day", "days"]
	return dist


def _win_rate_by_month(trades: pd.DataFrame) -> pd.DataFrame:
	if trades.empty:
		return pd.DataFrame(columns=["month", "win_rate"])
	month_series = trades["exit_time"].dt.strftime("%Y-%m")
	grouped = trades.groupby(month_series)
	out = grouped.apply(lambda g: (g["net_pnl"] > 0).mean()).reset_index()
	out.columns = ["month", "win_rate"]
	return out


def _stress_test(
	initial_equity: float,
	total_trades: int,
	win_rate: float,
	avg_winner: float,
	avg_loser: float,
	win_rate_drop_pp: float,
) -> Dict[str, float]:
	stressed_win_rate = max(0.0, win_rate - (win_rate_drop_pp / 100.0))
	expectancy = stressed_win_rate * avg_winner + (1.0 - stressed_win_rate) * avg_loser
	stressed_net = expectancy * total_trades
	stressed_final = initial_equity + stressed_net
	stressed_return_pct = ((stressed_final / initial_equity) - 1.0) * 100.0 if initial_equity > 0 else 0.0
	return {
		"win_rate_drop_pp": win_rate_drop_pp,
		"stressed_win_rate": stressed_win_rate,
		"stressed_final_equity": stressed_final,
		"stressed_return_pct": stressed_return_pct,
	}
