"""Storage helpers for historical and audit data."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from backtest.engine import BacktestResult


def save_parquet(df: pd.DataFrame, output_path: Path, compression: str = "snappy") -> None:
	"""Persist a DataFrame as Parquet.

	Args:
		df: Data to persist.
		output_path: Destination path.
		compression: Parquet compression codec.
	"""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_parquet(output_path, index=False, compression=compression)


def load_parquet(input_path: Path) -> pd.DataFrame:
	"""Load a DataFrame from Parquet.

	Args:
		input_path: Path to Parquet file.

	Returns:
		Loaded DataFrame.
	"""
	return pd.read_parquet(input_path)


async def initialize_audit_db(sqlite_path: Path) -> None:
	"""Initialize SQLite schema for audit logs.

	Args:
		sqlite_path: Path to SQLite database file.
	"""
	sqlite_path.parent.mkdir(parents=True, exist_ok=True)
	await asyncio.to_thread(_initialize_audit_db_sync, sqlite_path)


async def persist_backtest_audit(
	result: BacktestResult,
	sqlite_path: Path,
	run_id: str,
) -> None:
	"""Persist decisions, orders, and fills from a backtest run.

	Args:
		result: Backtest result object.
		sqlite_path: SQLite file path.
		run_id: Unique run identifier.
	"""
	await asyncio.to_thread(_persist_backtest_audit_sync, result, sqlite_path, run_id)


def _initialize_audit_db_sync(sqlite_path: Path) -> None:
	conn = sqlite3.connect(sqlite_path)
	try:
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS decisions (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				run_id TEXT NOT NULL,
				decision_time TEXT,
				decision_type TEXT,
				message TEXT,
				created_at TEXT DEFAULT CURRENT_TIMESTAMP
			)
			"""
		)
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS orders (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				run_id TEXT NOT NULL,
				order_id TEXT NOT NULL,
				side TEXT,
				order_type TEXT,
				quantity REAL,
				price REAL,
				status TEXT,
				notes TEXT,
				created_at TEXT DEFAULT CURRENT_TIMESTAMP
			)
			"""
		)
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS fills (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				run_id TEXT NOT NULL,
				order_id TEXT NOT NULL,
				fill_time TEXT,
				side TEXT,
				quantity REAL,
				price REAL,
				fee REAL,
				pnl REAL,
				reason TEXT,
				created_at TEXT DEFAULT CURRENT_TIMESTAMP
			)
			"""
		)
		conn.commit()
	finally:
		conn.close()


def _persist_backtest_audit_sync(result: BacktestResult, sqlite_path: Path, run_id: str) -> None:
	_initialize_audit_db_sync(sqlite_path)
	conn = sqlite3.connect(sqlite_path)
	try:
		decision_rows = _build_decision_rows(result=result, run_id=run_id)
		if decision_rows:
			conn.executemany(
				"""
				INSERT INTO decisions (run_id, decision_time, decision_type, message)
				VALUES (:run_id, :decision_time, :decision_type, :message)
				""",
				decision_rows,
			)

		order_rows, fill_rows = _build_order_and_fill_rows(result=result, run_id=run_id)
		if order_rows:
			conn.executemany(
				"""
				INSERT INTO orders (run_id, order_id, side, order_type, quantity, price, status, notes)
				VALUES (:run_id, :order_id, :side, :order_type, :quantity, :price, :status, :notes)
				""",
				order_rows,
			)
		if fill_rows:
			conn.executemany(
				"""
				INSERT INTO fills (run_id, order_id, fill_time, side, quantity, price, fee, pnl, reason)
				VALUES (:run_id, :order_id, :fill_time, :side, :quantity, :price, :fee, :pnl, :reason)
				""",
				fill_rows,
			)

		conn.commit()
	finally:
		conn.close()


def _build_decision_rows(result: BacktestResult, run_id: str) -> List[Dict[str, Any]]:
	if result.decisions.empty:
		return []

	rows: List[Dict[str, Any]] = []
	for _, row in result.decisions.iterrows():
		rows.append(
			{
				"run_id": run_id,
				"decision_time": _to_iso(row.get("time")),
				"decision_type": str(row.get("type", "")),
				"message": str(row.get("message", "")),
			}
		)
	return rows


def _build_order_and_fill_rows(result: BacktestResult, run_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
	if result.trades.empty:
		return [], []

	orders: List[Dict[str, Any]] = []
	fills: List[Dict[str, Any]] = []

	for index, row in result.trades.reset_index(drop=True).iterrows():
		trade_id = index + 1
		side = str(row.get("side", ""))
		entry_order_id = f"{run_id}-T{trade_id}-ENTRY"
		exit_order_id = f"{run_id}-T{trade_id}-EXIT"

		qty = float(row.get("qty", 0.0))
		entry_price = float(row.get("entry_price", 0.0))
		exit_price = float(row.get("exit_price", 0.0))
		reason = str(row.get("reason", ""))

		orders.append(
			{
				"run_id": run_id,
				"order_id": entry_order_id,
				"side": side,
				"order_type": "limit_post_only",
				"quantity": qty,
				"price": entry_price,
				"status": "filled",
				"notes": "backtest_entry",
			}
		)
		orders.append(
			{
				"run_id": run_id,
				"order_id": exit_order_id,
				"side": "sell" if side == "long" else "buy",
				"order_type": "exit",
				"quantity": qty,
				"price": exit_price,
				"status": "filled",
				"notes": reason,
			}
		)

		fills.append(
			{
				"run_id": run_id,
				"order_id": entry_order_id,
				"fill_time": _to_iso(row.get("entry_time")),
				"side": side,
				"quantity": qty,
				"price": entry_price,
				"fee": float(row.get("entry_fee", 0.0)),
				"pnl": 0.0,
				"reason": "entry_fill",
			}
		)
		fills.append(
			{
				"run_id": run_id,
				"order_id": exit_order_id,
				"fill_time": _to_iso(row.get("exit_time")),
				"side": "sell" if side == "long" else "buy",
				"quantity": qty,
				"price": exit_price,
				"fee": float(row.get("exit_fee", 0.0)),
				"pnl": float(row.get("net_pnl", 0.0)),
				"reason": reason,
			}
		)

	return orders, fills


def _to_iso(value: Any) -> str:
	if value is None:
		return ""
	if hasattr(value, "isoformat"):
		return str(value.isoformat())
	return str(value)
