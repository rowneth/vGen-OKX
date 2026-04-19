"""Unit tests for MEXCClient order and trigger-order wrappers.

These tests replace the aiohttp ClientSession with an in-memory fake so we can
assert on URL, HTTP method, body payload, and signing headers without making
any real network calls.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from exchange.mexc_client import MEXCClient  # noqa: E402


class _FakeResponse:
	def __init__(self, payload: Dict[str, Any]) -> None:
		self._payload = payload
		self.status = 200

	async def text(self) -> str:
		return json.dumps(self._payload)

	def raise_for_status(self) -> None:
		return None

	async def __aenter__(self) -> "_FakeResponse":
		return self

	async def __aexit__(self, *_: object) -> None:
		return None


class _FakeSession:
	"""Collects outgoing request metadata without touching the network."""

	def __init__(self) -> None:
		self.calls: List[Dict[str, Any]] = []
		self.response_payload: Dict[str, Any] = {"success": True, "code": 0, "data": {"orderId": 12345}}

	def request(
		self,
		method: str,
		url: str,
		params: Optional[Dict[str, Any]] = None,
		json: Optional[Any] = None,  # noqa: A002 - mirror aiohttp signature
		headers: Optional[Dict[str, str]] = None,
	) -> _FakeResponse:
		self.calls.append(
			{
				"method": method,
				"url": url,
				"params": params,
				"json": json,
				"headers": dict(headers or {}),
			}
		)
		return _FakeResponse(self.response_payload)

	async def close(self) -> None:
		return None


def _make_client(session: _FakeSession) -> MEXCClient:
	client = MEXCClient(
		api_key="test-key",
		api_secret="test-secret",
		session=session,  # type: ignore[arg-type]
	)
	return client


def _run_with_client(
	session: _FakeSession,
	coro_factory,
) -> Any:
	"""Instantiate the client inside an event loop (Py3.9 asyncio.Lock needs it)."""

	async def _runner() -> Any:
		client = _make_client(session)
		return await coro_factory(client)

	return asyncio.run(_runner())


def _expected_signature(api_key: str, api_secret: str, request_time: str, param_string: str) -> str:
	payload = f"{api_key}{request_time}{param_string}".encode("utf-8")
	return hmac.new(api_secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def test_submit_order_with_attached_sl_tp_sends_expected_body() -> None:
	session = _FakeSession()

	async def _call(client: MEXCClient) -> Any:
		return await client.submit_order(
			symbol="BTC_USDT",
			side=1,
			order_type=1,
			vol=0.01,
			price=65000.0,
			stop_loss_price=64000.0,
			take_profit_price=67000.0,
			open_type=1,
			leverage=5,
		)

	result = _run_with_client(session, _call)
	assert result == {"orderId": 12345}
	assert len(session.calls) == 1
	call = session.calls[0]
	assert call["method"] == "POST"
	assert call["url"].endswith("/api/v1/private/order/submit")
	body = call["json"]
	assert body["symbol"] == "BTC_USDT"
	assert body["stopLossPrice"] == 64000.0
	assert body["takeProfitPrice"] == 67000.0
	assert body["leverage"] == 5
	assert {"ApiKey", "Request-Time", "Signature"}.issubset(call["headers"])


def test_place_trigger_order_builds_plan_order_body() -> None:
	session = _FakeSession()

	async def _call(client: MEXCClient) -> Any:
		return await client.place_trigger_order(
			symbol="BTC_USDT",
			side=4,  # close long
			vol=0.01,
			trigger_price=64000.0,
			trigger_type=1,  # <= trigger
			order_type=5,  # market once triggered
			open_type=1,
		)

	_run_with_client(session, _call)
	call = session.calls[0]
	assert call["url"].endswith("/api/v1/private/planorder/place")
	body = call["json"]
	assert body["triggerPrice"] == 64000.0
	assert body["triggerType"] == 1
	assert body["orderType"] == 5
	assert body["vol"] == 0.01


def test_change_stop_price_requires_at_least_one_price() -> None:
	session = _FakeSession()

	async def _call(client: MEXCClient) -> Any:
		return await client.change_stop_price(stop_order_id=1)

	with pytest.raises(ValueError):
		_run_with_client(session, _call)


def test_change_stop_price_posts_both_prices_and_signs_correctly() -> None:
	session = _FakeSession()

	async def _call(client: MEXCClient) -> Any:
		return await client.change_stop_price(
			stop_order_id=999,
			stop_loss_price=65000.0,
			take_profit_price=68000.0,
		)

	_run_with_client(session, _call)
	call = session.calls[0]
	assert call["url"].endswith("/api/v1/private/stoporder/change_price")
	body = call["json"]
	assert body == {"orderId": 999, "stopLossPrice": 65000.0, "takeProfitPrice": 68000.0}

	# Recompute expected signature from the canonical JSON and ensure header matches.
	canonical = json.dumps(body, separators=(",", ":"), sort_keys=True)
	request_time = call["headers"]["Request-Time"]
	expected = _expected_signature("test-key", "test-secret", request_time, canonical)
	assert call["headers"]["Signature"] == expected
	assert call["headers"]["ApiKey"] == "test-key"


def test_cancel_trigger_order_sends_list_body() -> None:
	session = _FakeSession()

	async def _call(client: MEXCClient) -> Any:
		return await client.cancel_trigger_order(order_ids=[10, 20])

	_run_with_client(session, _call)
	call = session.calls[0]
	assert call["url"].endswith("/api/v1/private/planorder/cancel")
	assert call["json"] == [{"orderId": 10}, {"orderId": 20}]

	# Signing over a list body must be stable (insertion-ordered JSON).
	canonical = json.dumps(call["json"], separators=(",", ":"))
	request_time = call["headers"]["Request-Time"]
	expected = _expected_signature("test-key", "test-secret", request_time, canonical)
	assert call["headers"]["Signature"] == expected


def test_cancel_stop_order_sends_stop_plan_order_ids() -> None:
	session = _FakeSession()

	async def _call(client: MEXCClient) -> Any:
		return await client.cancel_stop_order(stop_plan_order_ids=[77])

	_run_with_client(session, _call)
	call = session.calls[0]
	assert call["url"].endswith("/api/v1/private/stoporder/cancel")
	assert call["json"] == [{"stopPlanOrderId": 77}]


def test_get_open_positions_passes_symbol_filter() -> None:
	session = _FakeSession()

	async def _call(client: MEXCClient) -> Any:
		return await client.get_open_positions(symbol="BTC_USDT")

	_run_with_client(session, _call)
	call = session.calls[0]
	assert call["method"] == "GET"
	assert call["url"].endswith("/api/v1/private/position/open_positions")
	assert call["params"] == {"symbol": "BTC_USDT"}
