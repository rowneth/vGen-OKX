"""Streamlit dashboard for the MEXC bot.

Launch with:

    /path/to/venv/bin/python -m streamlit run src/monitoring/dashboard_app.py

or via the helper script:

    python scripts/run_dashboard.py

The dashboard loads the historical parquet, runs a fresh in-memory backtest
using the current ``config/config.yaml``, and presents:

- **Overview**: equity curve, drawdown, KPI cards (WR / PF / Sharpe / fees).
- **Trades**: interactive table with PnL, entry/exit reasons, TP1 hit flag.
- **Chart Inspector**: price with Bollinger + Fib bands and trade markers;
  sub-panels for WaveTrend, MFI, RSI around any selected trade.
- **Strategy Config**: current parameters from ``config.yaml``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from backtest.engine import BacktestEngine  # noqa: E402
from backtest.metrics import calculate_metrics  # noqa: E402
from strategy.base import Strategy  # noqa: E402
from strategy.bollinger import BollingerMeanReversionStrategy  # noqa: E402
from strategy.cipher_confluence import CipherConfluenceStrategy  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data" / "historical"


def _resolve_data_path(config: dict) -> Path:
	"""Pick the parquet file that matches exchange.symbol + exchange.timeframe."""

	exch = config.get("exchange", {})
	symbol = str(exch.get("symbol", "BTC_USDT"))
	timeframe = str(exch.get("timeframe", "15m"))
	return DATA_DIR / f"{symbol}_{timeframe}.parquet"


# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_config() -> dict:
	with CONFIG_PATH.open("r", encoding="utf-8") as handle:
		return yaml.safe_load(handle)


@st.cache_data(show_spinner=False)
def load_ohlcv(data_path: str) -> pd.DataFrame:
	df = pd.read_parquet(data_path)
	df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
	df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
	df = df.sort_values("open_time").reset_index(drop=True)
	return df


def build_strategy(config: dict) -> Strategy:
	name = str(config.get("strategy", {}).get("name", "cipher_confluence")).lower()
	if name == "cipher_confluence":
		return CipherConfluenceStrategy(config)
	if name in ("bollinger", "bollinger_mean_reversion"):
		return BollingerMeanReversionStrategy(config)
	raise ValueError(f"Unknown strategy '{name}'")


@st.cache_data(show_spinner="Running backtest…")
def run_backtest(config_hash: str, initial_equity: float) -> dict:
	# config_hash is only used as a cache key; we reload config inside.
	del config_hash
	config = load_config()
	config["backtest"] = dict(config.get("backtest", {}))
	config["backtest"]["initial_equity"] = float(initial_equity)
	df = load_ohlcv(str(_resolve_data_path(config)))
	strategy = build_strategy(config)
	engine = BacktestEngine(config)
	result = engine.run(df, strategy)
	metrics = calculate_metrics(
		result=result,
		win_rate_drop_pp=float(
			config["backtest"]["stress_test"]["win_rate_drop_percentage_points"]
		),
	)

	prepared = strategy.prepare(df)

	# Enrich trades with notional (dollar volume) per fill.
	trades = result.trades.copy()
	if len(trades):
		trades["entry_notional"] = trades["entry_price"] * trades["initial_qty"]
		trades["exit_notional"] = trades["exit_price"] * trades["qty"]
		partial_fraction = (trades["initial_qty"] - trades["qty"]).clip(lower=0.0)
		trades["partial_notional"] = trades["entry_price"] * partial_fraction
		trades["total_traded_notional"] = (
			trades["entry_notional"] + trades["exit_notional"] + trades["partial_notional"]
		)

	return {
		"trades": trades,
		"equity_curve": result.equity_curve,
		"summary": result.summary,
		"metrics": metrics,
		"prepared": prepared,
	}


def config_fingerprint(config: dict) -> str:
	"""Stable string of strategy + fees + backtest sections for cache-busting."""

	relevant = {
		"strategy": config.get("strategy"),
		"fees": config.get("fees"),
		"backtest": config.get("backtest"),
	}
	return yaml.safe_dump(relevant, sort_keys=True)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------


def _kpi_row(metrics: dict, summary: dict, trades: pd.DataFrame) -> None:
	cols = st.columns(6)
	total_return = summary["total_return_pct"]
	cols[0].metric(
		"Total Return",
		f"{total_return:+.2f}%",
		delta=f"${summary['final_equity'] - summary['initial_equity']:+.2f}",
	)
	cols[1].metric(
		"Win Rate",
		f"{metrics['win_rate'] * 100.0:.1f}%",
		delta=f"{int(metrics['total_trades'])} trades",
	)
	cols[2].metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
	cols[3].metric("Sharpe", f"{metrics['sharpe_ratio']:.2f}")
	cols[4].metric(
		"Max DD",
		f"{metrics['max_drawdown_pct'] * 100.0:.2f}%",
		delta=f"Fees ${metrics['total_fees_paid']:.2f}",
		delta_color="off",
	)
	volume = (
		float(trades["total_traded_notional"].sum())
		if "total_traded_notional" in trades.columns
		else 0.0
	)
	avg_notional = (
		float(trades["entry_notional"].mean())
		if len(trades) and "entry_notional" in trades.columns
		else 0.0
	)
	cols[5].metric(
		"Total Volume",
		f"${volume:,.0f}",
		delta=f"Avg size ${avg_notional:,.2f}",
		delta_color="off",
	)


def _equity_chart(equity: pd.DataFrame) -> go.Figure:
	fig = make_subplots(
		rows=2,
		cols=1,
		shared_xaxes=True,
		row_heights=[0.7, 0.3],
		vertical_spacing=0.04,
		subplot_titles=("Equity Curve", "Drawdown %"),
	)
	fig.add_trace(
		go.Scatter(
			x=equity["time"],
			y=equity["equity"],
			mode="lines",
			name="Equity",
			line=dict(color="#2ecc71", width=2),
		),
		row=1,
		col=1,
	)
	peak = equity["equity"].cummax()
	dd = (equity["equity"] / peak - 1.0) * 100.0
	fig.add_trace(
		go.Scatter(
			x=equity["time"],
			y=dd,
			mode="lines",
			name="Drawdown",
			line=dict(color="#e74c3c", width=1.5),
			fill="tozeroy",
			fillcolor="rgba(231,76,60,0.2)",
		),
		row=2,
		col=1,
	)
	fig.update_layout(
		height=520,
		showlegend=False,
		margin=dict(l=20, r=20, t=40, b=20),
		template="plotly_dark",
	)
	fig.update_yaxes(title_text="USD", row=1, col=1)
	fig.update_yaxes(title_text="DD %", row=2, col=1)
	return fig


def _trade_inspector_chart(
	prepared: pd.DataFrame, trade: pd.Series, window_bars: int = 96
) -> go.Figure:
	entry_time = pd.to_datetime(trade["entry_time"], utc=True)
	exit_time = pd.to_datetime(trade["exit_time"], utc=True)
	matches = prepared.index[prepared["open_time"] == entry_time]
	if len(matches) == 0:
		entry_idx = (prepared["open_time"] - entry_time).abs().idxmin()
	else:
		entry_idx = int(matches[0])
	start = max(0, entry_idx - window_bars)
	end = min(len(prepared), entry_idx + window_bars)
	view = prepared.iloc[start:end].copy()

	fig = make_subplots(
		rows=3,
		cols=1,
		shared_xaxes=True,
		row_heights=[0.6, 0.2, 0.2],
		vertical_spacing=0.03,
		subplot_titles=("Price + Bollinger + Fib Bands", "WaveTrend", "MFI / RSI"),
	)

	fig.add_trace(
		go.Candlestick(
			x=view["open_time"],
			open=view["open"],
			high=view["high"],
			low=view["low"],
			close=view["close"],
			name="Price",
			increasing_line_color="#26a69a",
			decreasing_line_color="#ef5350",
		),
		row=1,
		col=1,
	)
	for col, color, name in [
		("bb_upper", "rgba(52,152,219,0.7)", "BB Upper"),
		("bb_middle", "rgba(52,152,219,0.9)", "BB Mid"),
		("bb_lower", "rgba(52,152,219,0.7)", "BB Lower"),
		("fib_upper", "rgba(241,196,15,0.7)", "Fib Upper"),
		("fib_middle", "rgba(241,196,15,0.9)", "Fib Mid (SSMA)"),
		("fib_lower", "rgba(241,196,15,0.7)", "Fib Lower"),
	]:
		if col in view.columns:
			fig.add_trace(
				go.Scatter(
					x=view["open_time"],
					y=view[col],
					mode="lines",
					name=name,
					line=dict(color=color, width=1),
				),
				row=1,
				col=1,
			)

	# Entry / exit / stop / TP markers.
	entry_color = "#2ecc71" if trade["side"] == "long" else "#e74c3c"
	exit_color = "#f39c12"
	fig.add_trace(
		go.Scatter(
			x=[entry_time],
			y=[trade["entry_price"]],
			mode="markers",
			marker=dict(
				symbol="triangle-up" if trade["side"] == "long" else "triangle-down",
				color=entry_color,
				size=14,
				line=dict(color="white", width=1),
			),
			name=f"Entry ({trade['side']})",
		),
		row=1,
		col=1,
	)
	fig.add_trace(
		go.Scatter(
			x=[exit_time],
			y=[trade["exit_price"]],
			mode="markers",
			marker=dict(symbol="x", color=exit_color, size=12, line=dict(width=2)),
			name=f"Exit ({trade['reason']})",
		),
		row=1,
		col=1,
	)

	# WaveTrend panel.
	if "wt1" in view.columns:
		fig.add_trace(
			go.Scatter(
				x=view["open_time"], y=view["wt1"], name="WT1", line=dict(color="#3498db")
			),
			row=2,
			col=1,
		)
		fig.add_trace(
			go.Scatter(
				x=view["open_time"], y=view["wt2"], name="WT2", line=dict(color="#e67e22")
			),
			row=2,
			col=1,
		)
		fig.add_hline(y=-53, line=dict(color="rgba(46,204,113,0.4)", dash="dot"), row=2, col=1)
		fig.add_hline(y=-75, line=dict(color="rgba(46,204,113,0.7)", dash="dot"), row=2, col=1)
		fig.add_hline(y=53, line=dict(color="rgba(231,76,60,0.4)", dash="dot"), row=2, col=1)
		fig.add_hline(y=75, line=dict(color="rgba(231,76,60,0.7)", dash="dot"), row=2, col=1)
		fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)"), row=2, col=1)

	# MFI / RSI panel.
	if "mfi" in view.columns:
		fig.add_trace(
			go.Scatter(
				x=view["open_time"], y=view["mfi"], name="MFI", line=dict(color="#9b59b6")
			),
			row=3,
			col=1,
		)
	if "rsi" in view.columns:
		fig.add_trace(
			go.Scatter(
				x=view["open_time"], y=view["rsi"], name="RSI", line=dict(color="#1abc9c")
			),
			row=3,
			col=1,
		)
	fig.add_hline(y=50, line=dict(color="rgba(255,255,255,0.2)"), row=3, col=1)
	fig.add_hline(y=30, line=dict(color="rgba(46,204,113,0.4)", dash="dot"), row=3, col=1)
	fig.add_hline(y=70, line=dict(color="rgba(231,76,60,0.4)", dash="dot"), row=3, col=1)

	# Entry / exit vertical bands.
	fig.add_vline(x=entry_time, line=dict(color=entry_color, dash="dash", width=1))
	fig.add_vline(x=exit_time, line=dict(color=exit_color, dash="dash", width=1))

	fig.update_layout(
		height=760,
		template="plotly_dark",
		margin=dict(l=20, r=20, t=40, b=20),
		xaxis_rangeslider_visible=False,
		legend=dict(orientation="h", y=1.08, x=0),
	)
	return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def main() -> None:
	st.set_page_config(
		page_title="MEXC Bot Dashboard", layout="wide", initial_sidebar_state="expanded"
	)

	st.sidebar.title("MEXC Cipher Bot")
	st.sidebar.caption("Cipher Confluence · BTC/USDT 15m")

	config = load_config()

	default_equity = float(config.get("backtest", {}).get("initial_equity", 10000.0))
	initial_equity = st.sidebar.number_input(
		"Starting capital ($)",
		min_value=10.0,
		max_value=10_000_000.0,
		value=float(st.session_state.get("initial_equity", default_equity)),
		step=100.0,
		format="%.2f",
		help="Simulate any account size. Position sizing is risk-% based so this scales proportionally.",
	)
	st.session_state["initial_equity"] = initial_equity

	preset_cols = st.sidebar.columns(3)
	if preset_cols[0].button("$100", use_container_width=True):
		st.session_state["initial_equity"] = 100.0
		st.rerun()
	if preset_cols[1].button("$1k", use_container_width=True):
		st.session_state["initial_equity"] = 1000.0
		st.rerun()
	if preset_cols[2].button("$10k", use_container_width=True):
		st.session_state["initial_equity"] = 10000.0
		st.rerun()

	if st.sidebar.button("🔄 Re-run backtest", use_container_width=True):
		run_backtest.clear()

	st.sidebar.markdown("---")
	st.sidebar.subheader("Fees & Commission")
	fees_cfg = config.get("fees", {})
	maker_pct = float(fees_cfg.get("maker", 0.0001)) * 100.0
	taker_pct = float(fees_cfg.get("taker", 0.0005)) * 100.0
	st.sidebar.markdown(
		f"- **Maker:** `{maker_pct:.2f}%`\n"
		f"- **Taker:** `{taker_pct:.2f}%`"
	)
	commission_pct = st.sidebar.slider(
		"Commission rebate (%)",
		min_value=0,
		max_value=100,
		value=70,
		step=5,
		help="Percentage of trading fees rebated back (e.g. referral / affiliate / VIP commission).",
	)

	data = run_backtest(config_fingerprint(config), initial_equity)
	trades: pd.DataFrame = data["trades"]
	equity: pd.DataFrame = data["equity_curve"]
	metrics: dict = data["metrics"]
	summary: dict = data["summary"]
	prepared: pd.DataFrame = data["prepared"]

	tab_overview, tab_trades, tab_inspector, tab_config = st.tabs(
		["📈 Overview", "📋 Trades", "🔬 Chart Inspector", "⚙️ Strategy Config"]
	)

	# ----- Overview tab -----
	with tab_overview:
		st.subheader("Performance")
		_kpi_row(metrics, summary, trades)
		st.plotly_chart(_equity_chart(equity), use_container_width=True)

		# Fees & Commission panel.
		st.subheader("Fees & Commission")
		gross_fees = float(metrics["total_fees_paid"])
		rebate = gross_fees * (commission_pct / 100.0)
		net_fees = gross_fees - rebate
		adjusted_final = float(summary["final_equity"]) + rebate
		adjusted_return_pct = (
			(adjusted_final / float(summary["initial_equity"])) - 1.0
		) * 100.0
		fee_cols = st.columns(5)
		fee_cols[0].metric("Maker Fee", f"{maker_pct:.2f}%")
		fee_cols[1].metric("Taker Fee", f"{taker_pct:.2f}%")
		fee_cols[2].metric(
			"Gross Fees Paid",
			f"${gross_fees:,.2f}",
			delta=f"on ${float(trades['total_traded_notional'].sum()) if 'total_traded_notional' in trades.columns else 0:,.0f} volume",
			delta_color="off",
		)
		fee_cols[3].metric(
			f"Commission Rebate ({commission_pct}%)",
			f"${rebate:,.2f}",
			delta=f"Net fees ${net_fees:,.2f}",
			delta_color="off",
		)
		fee_cols[4].metric(
			"Return after rebate",
			f"{adjusted_return_pct:+.2f}%",
			delta=f"${adjusted_final - float(summary['initial_equity']):+.2f}",
		)

		left, right = st.columns(2)
		with left:
			st.subheader("PnL distribution")
			if len(trades):
				fig = go.Figure()
				fig.add_trace(
					go.Histogram(
						x=trades["net_pnl"],
						nbinsx=30,
						marker_color="#3498db",
						name="Net PnL",
					)
				)
				fig.update_layout(
					template="plotly_dark",
					height=320,
					margin=dict(l=20, r=20, t=20, b=20),
					showlegend=False,
				)
				st.plotly_chart(fig, use_container_width=True)
		with right:
			st.subheader("Exits by reason")
			if len(trades):
				reasons = (
					trades.groupby("reason")
					.agg(count=("net_pnl", "size"), total_net=("net_pnl", "sum"))
					.reset_index()
				)
				fig = go.Figure()
				fig.add_trace(
					go.Bar(
						x=reasons["reason"],
						y=reasons["count"],
						text=[f"${v:+.0f}" for v in reasons["total_net"]],
						textposition="outside",
						marker_color=[
							"#2ecc71" if r == "take_profit" else "#e74c3c" if r == "stop_loss" else "#95a5a6"
							for r in reasons["reason"]
						],
					)
				)
				fig.update_layout(
					template="plotly_dark",
					height=320,
					margin=dict(l=20, r=20, t=20, b=20),
				)
				st.plotly_chart(fig, use_container_width=True)

		if len(trades) and "entry_notional" in trades.columns:
			st.subheader("Trading volume (notional $)")
			vol_df = trades.copy()
			vol_df["entry_time"] = pd.to_datetime(vol_df["entry_time"], utc=True)
			vol_df = vol_df.sort_values("entry_time")
			vol_df["cum_volume"] = vol_df["total_traded_notional"].cumsum()
			fig = make_subplots(specs=[[{"secondary_y": True}]])
			fig.add_trace(
				go.Bar(
					x=vol_df["entry_time"],
					y=vol_df["entry_notional"],
					name="Trade size ($)",
					marker_color="#3498db",
					opacity=0.7,
				),
				secondary_y=False,
			)
			fig.add_trace(
				go.Scatter(
					x=vol_df["entry_time"],
					y=vol_df["cum_volume"],
					name="Cumulative volume ($)",
					line=dict(color="#f1c40f", width=2),
				),
				secondary_y=True,
			)
			fig.update_layout(
				template="plotly_dark",
				height=340,
				margin=dict(l=20, r=20, t=20, b=20),
				legend=dict(orientation="h", y=1.1, x=0),
			)
			fig.update_yaxes(title_text="Per-trade $", secondary_y=False)
			fig.update_yaxes(title_text="Cumulative $", secondary_y=True)
			st.plotly_chart(fig, use_container_width=True)

	# ----- Trades tab -----
	with tab_trades:
		st.subheader(f"All trades ({len(trades)})")
		if len(trades) == 0:
			st.info("No trades produced by the current configuration.")
		else:
			show = trades.copy()
			show["entry_time"] = pd.to_datetime(show["entry_time"], utc=True).dt.strftime(
				"%Y-%m-%d %H:%M"
			)
			show["exit_time"] = pd.to_datetime(show["exit_time"], utc=True).dt.strftime(
				"%Y-%m-%d %H:%M"
			)
			display_cols = [
				"entry_time",
				"exit_time",
				"side",
				"entry_price",
				"exit_price",
				"qty",
				"entry_notional",
				"reason",
				"bars_held",
				"tp1_hit",
				"gross_pnl",
				"net_pnl",
				"equity_after",
			]
			display_cols = [c for c in display_cols if c in show.columns]
			st.dataframe(
				show[display_cols].style.format(
					{
						"entry_price": "{:.2f}",
						"exit_price": "{:.2f}",
						"qty": "{:.6f}",
						"entry_notional": "${:,.2f}",
						"gross_pnl": "{:+.2f}",
						"net_pnl": "{:+.2f}",
						"equity_after": "{:.2f}",
					}
				),
				use_container_width=True,
				height=560,
			)

	# ----- Chart inspector tab -----
	with tab_inspector:
		st.subheader("Trade chart inspector")
		if len(trades) == 0:
			st.info("No trades to inspect.")
		else:
			labeled = trades.copy().reset_index(drop=True)
			labeled["label"] = [
				f"#{i + 1}  {pd.to_datetime(row['entry_time']).strftime('%Y-%m-%d %H:%M')} "
				f"{row['side']}  net=${row['net_pnl']:+.1f}  ({row['reason']})"
				for i, row in labeled.iterrows()
			]
			c1, c2 = st.columns([3, 1])
			with c1:
				choice = st.selectbox(
					"Select a trade",
					options=labeled.index.tolist(),
					format_func=lambda i: labeled.loc[i, "label"],
				)
			with c2:
				window = st.slider(
					"Bars before/after entry",
					min_value=24,
					max_value=192,
					value=96,
					step=12,
				)
			trade = labeled.loc[int(choice)]
			st.plotly_chart(
				_trade_inspector_chart(prepared, trade, window_bars=int(window)),
				use_container_width=True,
			)

			st.markdown("**Trade details**")
			detail_cols = st.columns(4)
			detail_cols[0].metric("Side", str(trade["side"]).upper())
			detail_cols[1].metric("Net PnL", f"${trade['net_pnl']:+.2f}")
			detail_cols[2].metric("Reason", str(trade["reason"]))
			detail_cols[3].metric(
				"TP1 hit", "Yes" if bool(trade.get("tp1_hit", False)) else "No"
			)

	# ----- Config tab -----
	with tab_config:
		st.subheader("Current config/config.yaml")
		st.caption(str(CONFIG_PATH))
		st.code(yaml.safe_dump(config, sort_keys=False), language="yaml")


if __name__ == "__main__":
	main()
