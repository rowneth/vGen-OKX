"""Live WebSocket feed for MEXC market data."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

import websockets

LOGGER = logging.getLogger(__name__)


LiveFeedCallback = Callable[["LiveFeedMessage"], Awaitable[None]]


@dataclass(frozen=True)
class LiveFeedMessage:
	"""Represents one parsed live feed event."""

	event_type: str
	symbol: str
	timestamp: datetime
	payload: Dict[str, Any]
	raw: Dict[str, Any]


class MEXCLiveFeed:
	"""MEXC WebSocket feed wrapper with reconnect and heartbeat protection."""

	def __init__(
		self,
		ws_url: str,
		symbol: str,
		interval: str,
		heartbeat_timeout_seconds: int = 30,
		reconnect_delay_seconds: int = 5,
	) -> None:
		"""Initialize live feed client.

		Args:
			ws_url: WebSocket URL.
			symbol: Trading symbol (for example, BTC_USDT).
			interval: Kline interval (for example, Min15).
			heartbeat_timeout_seconds: Max silence duration before safety halt.
			reconnect_delay_seconds: Delay before reconnect attempt.
		"""
		self._ws_url = ws_url
		self._symbol = symbol
		self._interval = interval
		self._heartbeat_timeout_seconds = heartbeat_timeout_seconds
		self._reconnect_delay_seconds = reconnect_delay_seconds
		self._stop_event = asyncio.Event()
		self._last_message_at: Optional[datetime] = None

	@property
	def last_message_at(self) -> Optional[datetime]:
		"""Return timestamp of the most recent received message."""
		return self._last_message_at

	def stop(self) -> None:
		"""Signal the run loop to stop."""
		self._stop_event.set()

	async def run(self, callback: LiveFeedCallback, max_messages: Optional[int] = None) -> None:
		"""Connect and stream messages to callback with reconnect handling.

		Args:
			callback: Async callback for parsed messages.
			max_messages: Optional limit for processed messages.

		Raises:
			RuntimeError: If heartbeat timeout is breached.
		"""
		processed = 0
		while not self._stop_event.is_set():
			try:
				async with websockets.connect(self._ws_url, ping_interval=20, ping_timeout=10) as socket:
					LOGGER.info("WebSocket connected url=%s symbol=%s", self._ws_url, self._symbol)
					await self._subscribe(socket)
					while not self._stop_event.is_set():
						raw_text = await asyncio.wait_for(
							socket.recv(),
							timeout=self._heartbeat_timeout_seconds,
						)
						self._last_message_at = datetime.now(tz=timezone.utc)
						message = self._parse_message(raw_text)
						if message is None:
							continue
						await callback(message)
						processed += 1
						if max_messages is not None and processed >= max_messages:
							LOGGER.info("Reached max_messages=%d, stopping feed loop", max_messages)
							self.stop()
							return
			except asyncio.TimeoutError as exc:
				raise RuntimeError(
					f"WebSocket heartbeat timeout exceeded {self._heartbeat_timeout_seconds}s"
				) from exc
			except Exception as exc:  # noqa: BLE001
				if self._stop_event.is_set():
					return
				LOGGER.warning("WebSocket error: %s. Reconnecting in %ss", exc, self._reconnect_delay_seconds)
				await asyncio.sleep(self._reconnect_delay_seconds)

	async def _subscribe(self, socket: websockets.WebSocketClientProtocol) -> None:
		"""Send kline and trade subscriptions.

		Args:
			socket: Active websocket connection.
		"""
		for payload in self._subscription_payloads():
			await socket.send(json.dumps(payload))

	def _subscription_payloads(self) -> List[Dict[str, Any]]:
		"""Return default subscription payloads for kline and trade streams."""
		return [
			{
				"method": "sub.kline",
				"param": {"symbol": self._symbol, "interval": self._interval},
			},
			{
				"method": "sub.deal",
				"param": {"symbol": self._symbol},
			},
		]

	def _parse_message(self, raw_text: str) -> Optional[LiveFeedMessage]:
		"""Parse one websocket message.

		Args:
			raw_text: Raw JSON string.

		Returns:
			Parsed LiveFeedMessage when mappable, else None.
		"""
		try:
			raw: Dict[str, Any] = json.loads(raw_text)
		except json.JSONDecodeError:
			return None

		if raw.get("channel") == "pong" or raw.get("msg") == "pong":
			return None

		channel = str(raw.get("channel", ""))
		data = raw.get("data")
		if not channel and data is None:
			return None

		event_type = "unknown"
		if "kline" in channel.lower():
			event_type = "kline"
		elif "deal" in channel.lower() or "trade" in channel.lower():
			event_type = "trade"

		payload = data if isinstance(data, dict) else {"value": data}
		ts = payload.get("ts") or payload.get("timestamp") or raw.get("ts")
		timestamp = _coerce_timestamp(ts)
		symbol = str(payload.get("symbol") or raw.get("symbol") or self._symbol)

		return LiveFeedMessage(
			event_type=event_type,
			symbol=symbol,
			timestamp=timestamp,
			payload=payload,
			raw=raw,
		)


def _coerce_timestamp(value: Any) -> datetime:
	if value is None:
		return datetime.now(tz=timezone.utc)
	try:
		ts = int(value)
	except (TypeError, ValueError):
		return datetime.now(tz=timezone.utc)
	if ts > 10_000_000_000:
		return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
	return datetime.fromtimestamp(ts, tz=timezone.utc)
