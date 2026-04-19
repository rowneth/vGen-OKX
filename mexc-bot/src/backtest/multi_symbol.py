"""Multi-symbol portfolio backtesting.

Runs the same strategy independently on several symbols, scores them, and
combines the top-N into an aggregated portfolio. Designed for the dashboard
so the user can see:

- Per-symbol leaderboard (score, WR, PF, return, trades, fees, volume).
- Aggregate portfolio equity curve and KPIs.
- Concurrent-position awareness via scaled risk-per-trade.

The scoring formula favours strategies that are profitable, consistent, and
low-drawdown, while rewarding higher trade frequency (more fee coverage):

    score = max(0, PF - 1) * win_rate * (1 - max_drawdown) * log(1 + trades)

so a PF of 1.0 scores 0 (break-even), higher PF and WR increase it, and
deeper drawdowns / tiny trade counts penalise it.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from backtest.metrics import calculate_metrics
from strategy.base import Strategy
from strategy.bollinger import BollingerMeanReversionStrategy
from strategy.cipher_confluence import CipherConfluenceStrategy


def _build_strategy(config: dict) -> Strategy:
	name = str(config.get("strategy", {}).get("name", "cipher_confluence")).lower()
	if name == "cipher_confluence":
		return CipherConfluenceStrategy(config)
	if name in ("bollinger", "bollinger_mean_reversion"):
		return BollingerMeanReversionStrategy(config)
	raise ValueError(f"Unknown strategy '{name}'")


def _load_symbol_ohlcv(symbol: str, timeframe: str, data_dir: Path) -> Optional[pd.DataFrame]:
	path = data_dir / f"{symbol}_{timeframe}.parquet"
	if not path.exists():
		return None
	df = pd.read_parquet(path)
	df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
	df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
	return df.sort_values("open_time").reset_index(drop=True)


@dataclass
class SymbolResult:
	symbol: str
	trades: pd.DataFrame
	equity_curve: pd.DataFrame
	summary: dict
	metrics: dict
	score: float


def compute_score(metrics: dict, summary: dict, trades: int) -> float:
	"""Composite quality score (higher is better).

	Penalises break-even or worse (``PF <= 1``), tiny samples, and deep
	drawdowns; rewards more trades (up to log-scale).
	"""

	pf = float(metrics.get("profit_factor", 0.0) or 0.0)
	wr = float(metrics.get("win_rate", 0.0) or 0.0)
	dd = float(metrics.get("max_drawdown_pct", 0.0) or 0.0)
	pf_edge = max(0.0, pf - 1.0)
	dd_factor = max(0.0, 1.0 - dd)
	freq_factor = math.log1p(max(0.0, trades))
	return float(pf_edge * wr * dd_factor * freq_factor)


def run_symbol_backtest(
	symbol: str,
	ohlcv: pd.DataFrame,
	config: dict,
	initial_equity: float,
) -> SymbolResult:
	"""Run a single backtest for one symbol with its own initial equity."""

	cfg = copy.deepcopy(config)
	cfg["backtest"] = dict(cfg.get("backtest", {}))
	cfg["backtest"]["initial_equity"] = float(initial_equity)
	strategy = _build_strategy(cfg)
	engine = BacktestEngine(cfg)
	result = engine.run(ohlcv, strategy)
	metrics = calculate_metrics(
		result=result,
		win_rate_drop_pp=float(
			cfg["backtest"]["stress_test"]["win_rate_drop_percentage_points"]
		),
	)

	trades = result.trades.copy()
	if len(trades):
		trades["symbol"] = symbol
		trades["entry_notional"] = trades["entry_price"] * trades["initial_qty"]
		trades["exit_notional"] = trades["exit_price"] * trades["qty"]
		partial_qty = (trades["initial_qty"] - trades["qty"]).clip(lower=0.0)
		trades["partial_notional"] = trades["entry_price"] * partial_qty
		trades["total_traded_notional"] = (
			trades["entry_notional"] + trades["exit_notional"] + trades["partial_notional"]
		)

	score = compute_score(metrics, result.summary, int(len(trades)))
	return SymbolResult(
		symbol=symbol,
		trades=trades,
		equity_curve=result.equity_curve,
		summary=result.summary,
		metrics=metrics,
		score=score,
	)


def run_multi_symbol_backtest(
	symbols: Iterable[str],
	config: dict,
	data_dir: Path,
	initial_equity: float,
	timeframe: str = "15m",
	top_n: Optional[int] = None,
	risk_scale_by_slots: bool = True,
) -> Dict[str, object]:
	"""Backtest each symbol, rank, and aggregate the top-N into a portfolio.

	Args:
		symbols: Candidate symbols to evaluate.
		config: Full config dict.
		data_dir: Directory containing ``<SYMBOL>_<TF>.parquet``.
		initial_equity: Total portfolio starting capital.
		timeframe: Timeframe suffix for parquet files.
		top_n: If set, aggregate only the top-N ranked symbols.
			Otherwise aggregate all that traded.
		risk_scale_by_slots: If True and ``top_n`` is provided, divide the
			per-trade risk pct by the number of concurrent slots so total
			portfolio risk stays bounded.

	Returns:
		Dict with keys: ``per_symbol`` (List[SymbolResult]), ``leaderboard``
		(DataFrame), ``aggregate_equity`` (DataFrame), ``aggregate_trades``
		(DataFrame), ``aggregate_summary`` (dict), ``aggregate_metrics`` (dict).
	"""

	symbols = list(symbols)
	slots = int(top_n) if top_n else max(1, len(symbols))

	# Per-symbol run. Risk pct is scaled so that N concurrent positions at
	# full size would still respect the original per-trade risk budget.
	symbol_equity = initial_equity / max(1, slots)
	cfg = copy.deepcopy(config)
	if risk_scale_by_slots and "risk" in cfg:
		cfg["risk"] = dict(cfg["risk"])
		# Keep the per-trade risk pct as-is but shrink each symbol's equity
		# to 1/slots of the portfolio — equivalent to dividing risk.
	per_symbol: List[SymbolResult] = []
	for symbol in symbols:
		df = _load_symbol_ohlcv(symbol, timeframe, data_dir)
		if df is None or len(df) < 300:
			continue
		res = run_symbol_backtest(symbol, df, cfg, symbol_equity)
		per_symbol.append(res)

	per_symbol.sort(key=lambda r: r.score, reverse=True)

	# Leaderboard table.
	rows = []
	for res in per_symbol:
		rows.append(
			{
				"symbol": res.symbol,
				"score": res.score,
				"trades": int(len(res.trades)),
				"win_rate": res.metrics.get("win_rate", 0.0),
				"profit_factor": res.metrics.get("profit_factor", 0.0),
				"return_pct": res.summary.get("total_return_pct", 0.0),
				"sharpe": res.metrics.get("sharpe_ratio", 0.0),
				"max_dd": res.metrics.get("max_drawdown_pct", 0.0),
				"fees": res.metrics.get("total_fees_paid", 0.0),
				"volume": (
					float(res.trades["total_traded_notional"].sum())
					if "total_traded_notional" in res.trades.columns
					else 0.0
				),
				"final_equity": res.summary.get("final_equity", 0.0),
			}
		)
	leaderboard = pd.DataFrame(rows)

	# Aggregate only the selected (top-N) symbols.
	selected = per_symbol[:slots] if top_n else per_symbol

	aggregate_trades = _aggregate_trades(selected)
	aggregate_equity = _aggregate_equity(selected, initial_equity)
	aggregate_summary = {
		"initial_equity": float(initial_equity),
		"final_equity": (
			float(aggregate_equity["equity"].iloc[-1]) if len(aggregate_equity) else float(initial_equity)
		),
	}
	aggregate_summary["total_return_pct"] = (
		(aggregate_summary["final_equity"] / aggregate_summary["initial_equity"]) - 1.0
	) * 100.0
	aggregate_summary["total_fees_paid"] = sum(
		float(r.metrics.get("total_fees_paid", 0.0)) for r in selected
	)
	aggregate_summary["total_trades"] = float(len(aggregate_trades))

	aggregate_metrics = _aggregate_metrics(aggregate_trades, aggregate_equity)

	return {
		"per_symbol": per_symbol,
		"selected_symbols": [r.symbol for r in selected],
		"leaderboard": leaderboard,
		"aggregate_trades": aggregate_trades,
		"aggregate_equity": aggregate_equity,
		"aggregate_summary": aggregate_summary,
		"aggregate_metrics": aggregate_metrics,
	}


def _aggregate_trades(selected: List[SymbolResult]) -> pd.DataFrame:
	frames = [r.trades for r in selected if len(r.trades)]
	if not frames:
		return pd.DataFrame()
	combined = pd.concat(frames, ignore_index=True)
	combined["entry_time"] = pd.to_datetime(combined["entry_time"], utc=True)
	combined = combined.sort_values("entry_time").reset_index(drop=True)
	return combined


def _aggregate_equity(
	selected: List[SymbolResult], initial_equity: float
) -> pd.DataFrame:
	"""Sum per-symbol equity curves aligned by time."""

	if not selected:
		return pd.DataFrame({"time": [], "equity": []})

	curves = []
	for r in selected:
		eq = r.equity_curve.copy()
		if len(eq) == 0:
			continue
		eq["time"] = pd.to_datetime(eq["time"], utc=True)
		eq = eq.set_index("time")[["equity"]]
		eq = eq[~eq.index.duplicated(keep="last")]
		eq = eq.rename(columns={"equity": r.symbol})
		curves.append(eq)
	if not curves:
		return pd.DataFrame({"time": [], "equity": []})

	merged = pd.concat(curves, axis=1).sort_index().ffill().fillna(0.0)
	# For each symbol, replace any leading zeros (before first fill) with
	# that symbol's starting equity so the total stays sane.
	for col in merged.columns:
		first_valid = merged[col][merged[col] > 0].index.min()
		if pd.notna(first_valid):
			seed_val = merged.loc[first_valid, col]
			merged.loc[merged.index < first_valid, col] = seed_val
	merged["equity"] = merged.sum(axis=1)
	return merged.reset_index()[["time", "equity"]]


def _aggregate_metrics(
	aggregate_trades: pd.DataFrame, aggregate_equity: pd.DataFrame
) -> dict:
	total_trades = int(len(aggregate_trades))
	if total_trades == 0:
		return {
			"win_rate": 0.0,
			"profit_factor": 0.0,
			"sharpe_ratio": 0.0,
			"max_drawdown_pct": 0.0,
			"total_fees_paid": 0.0,
			"total_trades": 0.0,
			"average_trade": 0.0,
			"average_winner": 0.0,
			"average_loser": 0.0,
		}

	net = aggregate_trades["net_pnl"].astype(float)
	winners = net[net > 0]
	losers = net[net <= 0]
	wr = float((net > 0).mean())
	pf = float(winners.sum() / abs(losers.sum())) if losers.sum() < 0 else float("inf")

	# Max drawdown on combined equity.
	eq = aggregate_equity["equity"].astype(float) if len(aggregate_equity) else pd.Series([])
	dd_pct = 0.0
	sharpe = 0.0
	if len(eq):
		peak = eq.cummax()
		dd = (eq / peak) - 1.0
		dd_pct = float(abs(dd.min()))
		rets = eq.pct_change().dropna()
		if len(rets) > 1 and rets.std() > 0:
			# Annualise assuming 15m bars (96/day, ~365 days).
			sharpe = float(rets.mean() / rets.std() * math.sqrt(96 * 365))

	total_fees = float(
		aggregate_trades.get("entry_fee", pd.Series([0.0])).sum()
		+ aggregate_trades.get("exit_fee", pd.Series([0.0])).sum()
	)

	return {
		"win_rate": wr,
		"profit_factor": pf if math.isfinite(pf) else 0.0,
		"sharpe_ratio": sharpe,
		"max_drawdown_pct": dd_pct,
		"total_fees_paid": total_fees,
		"total_trades": float(total_trades),
		"average_trade": float(net.mean()),
		"average_winner": float(winners.mean()) if len(winners) else 0.0,
		"average_loser": float(losers.mean()) if len(losers) else 0.0,
	}
