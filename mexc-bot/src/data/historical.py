"""Historical data download and normalization utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from exchange.mexc_client import MEXCClient

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadRequest:
	"""Represents a historical download query."""

	symbol: str
	interval: str
	start_time: datetime
	end_time: datetime
	chunk_days: int = 7


class HistoricalDataDownloader:
	"""Downloads and normalizes futures kline history from MEXC."""

	def __init__(self, client: MEXCClient) -> None:
		"""Initialize downloader.

		Args:
			client: Configured MEXC client.
		"""
		self._client = client

	async def download_klines(self, request: DownloadRequest) -> pd.DataFrame:
		"""Download kline data for a time range.

		Args:
			request: Download parameters.

		Returns:
			Normalized and time-sorted DataFrame.
		"""
		start = request.start_time.astimezone(timezone.utc)
		end = request.end_time.astimezone(timezone.utc)
		chunk = timedelta(days=request.chunk_days)

		rows: List[Dict[str, Any]] = []
		cursor = start
		while cursor < end:
			chunk_end = min(cursor + chunk, end)
			start_sec = int(cursor.timestamp())
			end_sec = int(chunk_end.timestamp())

			LOGGER.info(
				"Downloading %s %s klines from %s to %s",
				request.symbol,
				request.interval,
				cursor.isoformat(),
				chunk_end.isoformat(),
			)

			payload = await self._client.get_klines(
				symbol=request.symbol,
				interval=request.interval,
				start=start_sec,
				end=end_sec,
			)
			rows.extend(_normalize_kline_payload(payload))
			cursor = chunk_end

		df = pd.DataFrame(rows)
		if df.empty:
			return df

		df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
		return df


def _normalize_kline_payload(payload: Any) -> List[Dict[str, Any]]:
	"""Normalize MEXC kline response variants into a standard schema.

	Args:
		payload: Raw response data from the kline endpoint.

	Returns:
		List of normalized row dictionaries.
	"""
	if payload is None:
		return []

	records: List[Any]
	if isinstance(payload, dict) and _is_columnar_kline_dict(payload):
		return _normalize_columnar_kline_dict(payload)
	if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
		records = payload["data"]
	elif isinstance(payload, list):
		records = payload
	else:
		LOGGER.warning("Unexpected kline payload shape: %s", type(payload).__name__)
		return []

	out: List[Dict[str, Any]] = []
	for item in records:
		row = _normalize_one_kline(item)
		if row is not None:
			out.append(row)
	return out


def _normalize_one_kline(item: Any) -> Optional[Dict[str, Any]]:
	"""Normalize one raw kline record.

	Args:
		item: Raw kline row (dict or list format).

	Returns:
		Normalized row or None if unsupported.
	"""
	try:
		if isinstance(item, dict):
			ts = int(item.get("time") or item.get("openTime") or item.get("timestamp"))
			open_price = float(item.get("open"))
			high_price = float(item.get("high"))
			low_price = float(item.get("low"))
			close_price = float(item.get("close"))
			volume = float(item.get("vol") or item.get("volume") or 0.0)
			turnover = float(item.get("amount") or item.get("turnover") or 0.0)
		elif isinstance(item, (list, tuple)) and len(item) >= 6:
			ts = int(item[0])
			open_price = float(item[1])
			high_price = float(item[2])
			low_price = float(item[3])
			close_price = float(item[4])
			volume = float(item[5])
			turnover = float(item[6]) if len(item) > 6 else 0.0
		else:
			return None
	except (TypeError, ValueError):
		return None

	if ts < 10_000_000_000:
		ts = ts * 1000
	open_dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
	close_dt = open_dt + timedelta(minutes=15)
	return {
		"open_time": open_dt,
		"close_time": close_dt,
		"open": open_price,
		"high": high_price,
		"low": low_price,
		"close": close_price,
		"volume": volume,
		"turnover": turnover,
	}


def _is_columnar_kline_dict(payload: Dict[str, Any]) -> bool:
	required = {"time", "open", "high", "low", "close", "vol", "amount"}
	return required.issubset(payload.keys()) and all(isinstance(payload[key], list) for key in required)


def _normalize_columnar_kline_dict(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
	lengths = [len(payload[key]) for key in ["time", "open", "high", "low", "close", "vol", "amount"]]
	row_count = min(lengths) if lengths else 0
	out: List[Dict[str, Any]] = []
	for i in range(row_count):
		row = _normalize_one_kline(
			{
				"time": payload["time"][i],
				"open": payload["open"][i],
				"high": payload["high"][i],
				"low": payload["low"][i],
				"close": payload["close"][i],
				"vol": payload["vol"][i],
				"amount": payload["amount"][i],
			}
		)
		if row is not None:
			out.append(row)
	return out
