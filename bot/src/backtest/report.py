"""Backtest report generation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtest.engine import BacktestResult


def generate_markdown_report(
	result: BacktestResult,
	metrics: Dict[str, object],
	output_path: Path,
) -> None:
	"""Generate markdown report from backtest outputs.

	Args:
		result: Backtest result object.
		metrics: Metrics dictionary.
		output_path: Destination markdown file path.
	"""
	output_path.parent.mkdir(parents=True, exist_ok=True)

	monthly_returns = metrics["monthly_returns"]
	trades_per_day = metrics["trades_per_day_distribution"]
	win_rate_by_month = metrics["win_rate_by_month"]
	stress = metrics["stress_test"]

	lines = [
		"# Backtest Report",
		"",
		"## Summary",
		f"- Initial equity: {metrics['initial_equity']:.2f}",
		f"- Final equity: {metrics['final_equity']:.2f}",
		f"- Total return: {metrics['total_return_pct']:.2f}%",
		f"- Sharpe ratio: {metrics['sharpe_ratio']:.4f}",
		f"- Max drawdown: {metrics['max_drawdown_pct'] * 100.0:.2f}%",
		f"- Win rate: {metrics['win_rate'] * 100.0:.2f}%",
		f"- Profit factor: {metrics['profit_factor']:.4f}",
		f"- Average trade: {metrics['average_trade']:.4f}",
		f"- Average winner: {metrics['average_winner']:.4f}",
		f"- Average loser: {metrics['average_loser']:.4f}",
		f"- Total fees paid: {metrics['total_fees_paid']:.4f}",
		f"- Total trades: {metrics['total_trades']}",
		"",
		"## Stress Test",
		f"- Win-rate drop assumption: {stress['win_rate_drop_pp']:.2f} percentage points",
		f"- Stressed win rate: {stress['stressed_win_rate'] * 100.0:.2f}%",
		f"- Stressed final equity: {stress['stressed_final_equity']:.2f}",
		f"- Stressed return: {stress['stressed_return_pct']:.2f}%",
		"",
		"## Monthly Returns",
		_to_markdown_table(monthly_returns),
		"",
		"## Trades Per Day Distribution",
		_to_markdown_table(trades_per_day),
		"",
		"## Win Rate by Month",
		_to_markdown_table(win_rate_by_month),
		"",
		"## Recent Trades",
		_to_markdown_table(result.trades.tail(20)),
	]

	output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_html_report(
	result: BacktestResult,
	metrics: Dict[str, object],
	output_path: Path,
) -> None:
	"""Generate interactive HTML report with tables and charts.

	Args:
		result: Backtest result object.
		metrics: Metrics dictionary.
		output_path: Destination html file path.
	"""
	output_path.parent.mkdir(parents=True, exist_ok=True)

	equity = result.equity_curve.copy()
	equity["time"] = pd.to_datetime(equity["time"], utc=True)

	fig = make_subplots(
		rows=2,
		cols=1,
		shared_xaxes=True,
		vertical_spacing=0.12,
		subplot_titles=("Equity Curve vs BTC Price", "Trade PnL Distribution"),
	)

	fig.add_trace(
		go.Scatter(x=equity["time"], y=equity["equity"], name="Equity", line=dict(color="#1f77b4")),
		row=1,
		col=1,
	)
	fig.add_trace(
		go.Scatter(x=equity["time"], y=equity["close"], name="BTC Close", line=dict(color="#ff7f0e"), yaxis="y2"),
		row=1,
		col=1,
	)

	if not result.trades.empty:
		fig.add_trace(
			go.Histogram(x=result.trades["net_pnl"], nbinsx=50, name="Net PnL"),
			row=2,
			col=1,
		)

	fig.update_layout(
		title="MEXC Bollinger Mean Reversion Backtest",
		template="plotly_white",
		height=900,
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
	)

	summary_table = _summary_html(metrics)
	monthly_table = metrics["monthly_returns"].to_html(index=False) if not metrics["monthly_returns"].empty else "<p>No monthly data.</p>"
	win_rate_month = (
		metrics["win_rate_by_month"].to_html(index=False) if not metrics["win_rate_by_month"].empty else "<p>No monthly win-rate data.</p>"
	)
	trades_per_day = (
		metrics["trades_per_day_distribution"].to_html(index=False)
		if not metrics["trades_per_day_distribution"].empty
		else "<p>No trades-per-day data.</p>"
	)

	html = f"""
	<html>
	  <head>
		<meta charset=\"utf-8\" />
		<title>Backtest Report</title>
		<style>
		  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 24px; }}
		  h1, h2 {{ color: #222; }}
		  table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
		  th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 12px; }}
		  th {{ background: #f7f7f7; }}
		</style>
	  </head>
	  <body>
		<h1>Backtest Report</h1>
		<h2>Summary</h2>
		{summary_table}
		<h2>Charts</h2>
		{fig.to_html(full_html=False, include_plotlyjs='cdn')}
		<h2>Monthly Returns</h2>
		{monthly_table}
		<h2>Win Rate by Month</h2>
		{win_rate_month}
		<h2>Trades Per Day Distribution</h2>
		{trades_per_day}
	  </body>
	</html>
	"""
	output_path.write_text(html, encoding="utf-8")


def _to_markdown_table(frame: pd.DataFrame) -> str:
	if frame.empty:
		return "No data."
	try:
		return frame.to_markdown(index=False)
	except Exception:
		columns = list(frame.columns)
		header = "| " + " | ".join(columns) + " |"
		separator = "| " + " | ".join(["---"] * len(columns)) + " |"
		rows = [
			"| " + " | ".join(str(row[col]) for col in columns) + " |"
			for _, row in frame.iterrows()
		]
		return "\n".join([header, separator, *rows])


def _summary_html(metrics: Dict[str, object]) -> str:
	rows = [
		("Initial Equity", f"{metrics['initial_equity']:.2f}"),
		("Final Equity", f"{metrics['final_equity']:.2f}"),
		("Total Return (%)", f"{metrics['total_return_pct']:.2f}"),
		("Sharpe Ratio", f"{metrics['sharpe_ratio']:.4f}"),
		("Max Drawdown (%)", f"{metrics['max_drawdown_pct'] * 100.0:.2f}"),
		("Win Rate (%)", f"{metrics['win_rate'] * 100.0:.2f}"),
		("Profit Factor", f"{metrics['profit_factor']:.4f}"),
		("Average Trade", f"{metrics['average_trade']:.4f}"),
		("Average Winner", f"{metrics['average_winner']:.4f}"),
		("Average Loser", f"{metrics['average_loser']:.4f}"),
		("Total Fees Paid", f"{metrics['total_fees_paid']:.4f}"),
		("Total Trades", f"{metrics['total_trades']}"),
	]
	row_html = "\n".join([f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in rows])
	return f"<table>{row_html}</table>"
