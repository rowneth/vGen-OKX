"""Async MEXC futures REST client with signed endpoint support."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import aiohttp

from exchange.rate_limiter import TokenBucketRateLimiter

LOGGER = logging.getLogger(__name__)


class MEXCClient:
	"""Minimal MEXC futures REST API client.

	The signing implementation follows the public MEXC futures documentation
	convention of hashing ``api_key + request_time + request_param_string``.
	"""

	def __init__(
		self,
		api_key: str,
		api_secret: str,
		base_url: str = "https://contract.mexc.com",
		timeout_seconds: int = 15,
		requests_per_second: float = 10.0,
		burst_capacity: int = 20,
		session: Optional[aiohttp.ClientSession] = None,
	) -> None:
		"""Initialize a MEXC futures client.

		Args:
			api_key: MEXC API key.
			api_secret: MEXC API secret.
			base_url: REST base URL.
			timeout_seconds: HTTP timeout in seconds.
			requests_per_second: Rate limiter refill speed.
			burst_capacity: Rate limiter max burst size.
			session: Optional injected aiohttp session.
		"""
		self._api_key = api_key
		self._api_secret = api_secret
		self._base_url = base_url.rstrip("/")
		self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
		self._session = session
		self._owns_session = session is None
		self._limiter = TokenBucketRateLimiter(
			rate_per_second=requests_per_second,
			burst_capacity=burst_capacity,
		)

	async def __aenter__(self) -> "MEXCClient":
		"""Create an internal session when using async context manager."""
		if self._session is None:
			self._session = aiohttp.ClientSession(timeout=self._timeout)
			self._owns_session = True
		return self

	async def __aexit__(self, *_: object) -> None:
		"""Close internally managed session."""
		await self.close()

	async def close(self) -> None:
		"""Close underlying HTTP session if this client created it."""
		if self._owns_session and self._session is not None:
			await self._session.close()
			self._session = None

	async def get_klines(
		self,
		symbol: str,
		interval: str = "Min15",
		start: Optional[int] = None,
		end: Optional[int] = None,
		limit: Optional[int] = None,
	) -> Any:
		"""Fetch historical futures klines.

		Args:
			symbol: Contract symbol like ``BTC_USDT``.
			interval: Kline interval such as ``Min15``.
			start: Optional start timestamp in milliseconds.
			end: Optional end timestamp in milliseconds.
			limit: Optional server-side limit when supported.

		Returns:
			Decoded response payload from the endpoint.
		"""
		params: Dict[str, Any] = {"interval": interval}
		if start is not None:
			params["start"] = start
		if end is not None:
			params["end"] = end
		if limit is not None:
			params["limit"] = limit

		path = f"/api/v1/contract/kline/{symbol}"
		return await self._request("GET", path, params=params, auth=False)

	async def get_contract_detail(self, symbol: Optional[str] = None) -> Any:
		"""Fetch contract specification (contractSize, scales, min vol, fees).

		Args:
			symbol: Optional contract symbol filter.

		Returns:
			Decoded contract-detail payload (list of contracts when no symbol).
		"""
		params: Dict[str, Any] = {}
		if symbol is not None:
			params["symbol"] = symbol
		return await self._request("GET", "/api/v1/contract/detail", params=params, auth=False)

	async def get_ticker(self, symbol: str) -> Any:
		"""Fetch latest ticker (last/bid/ask) for a contract.

		Args:
			symbol: Contract symbol.

		Returns:
			Decoded ticker payload.
		"""
		return await self._request(
			"GET", "/api/v1/contract/ticker", params={"symbol": symbol}, auth=False
		)

	async def get_order_by_external_oid(self, symbol: str, external_oid: str) -> Any:
		"""Fetch a specific order by its external (client) order id.

		Args:
			symbol: Contract symbol.
			external_oid: External order id supplied at submit time.

		Returns:
			Decoded order payload.
		"""
		path = f"/api/v1/private/order/external/{symbol}/{external_oid}"
		return await self._request("GET", path, auth=True)

	async def get_account_info(self) -> Any:
		"""Fetch private futures account assets/margin info.

		Returns:
			Decoded account payload from MEXC private endpoint.
		"""
		path = "/api/v1/private/account/assets"
		return await self._request("GET", path, auth=True)

	# ------------------------------------------------------------------
	# Order + trigger-order wrappers
	# ------------------------------------------------------------------

	async def submit_order(
		self,
		symbol: str,
		side: int,
		order_type: int,
		vol: float,
		price: Optional[float] = None,
		leverage: Optional[int] = None,
		open_type: int = 1,
		position_id: Optional[int] = None,
		external_oid: Optional[str] = None,
		stop_loss_price: Optional[float] = None,
		take_profit_price: Optional[float] = None,
		reduce_only: bool = False,
	) -> Any:
		"""Submit a contract order, optionally with attached SL/TP.

		Args:
			symbol: Contract symbol (e.g. ``BTC_USDT``).
			side: MEXC side code (1=open long, 2=close short, 3=open short, 4=close long).
			order_type: 1=limit, 5=market, 6=post-only (per MEXC contract API).
			vol: Order size in contracts.
			price: Limit price (required for non-market types).
			leverage: Optional leverage override.
			open_type: 1=isolated, 2=cross.
			position_id: Existing position id for close-side operations.
			external_oid: Client-supplied external order id (idempotency).
			stop_loss_price: Optional server-side SL attached on fill.
			take_profit_price: Optional server-side TP attached on fill.
			reduce_only: Reduce-only flag.

		Returns:
			Decoded exchange response (typically an order id).
		"""
		body: Dict[str, Any] = {
			"symbol": symbol,
			"side": int(side),
			"type": int(order_type),
			"vol": float(vol),
			"openType": int(open_type),
			"reduceOnly": bool(reduce_only),
		}
		if price is not None:
			body["price"] = float(price)
		if leverage is not None:
			body["leverage"] = int(leverage)
		if position_id is not None:
			body["positionId"] = int(position_id)
		if external_oid is not None:
			body["externalOid"] = str(external_oid)
		if stop_loss_price is not None:
			body["stopLossPrice"] = float(stop_loss_price)
		if take_profit_price is not None:
			body["takeProfitPrice"] = float(take_profit_price)

		return await self._request(
			"POST", "/api/v1/private/order/submit", body=body, auth=True
		)

	async def place_trigger_order(
		self,
		symbol: str,
		side: int,
		vol: float,
		trigger_price: float,
		trigger_type: int,
		execute_cycle: int = 1,
		order_type: int = 5,
		price: Optional[float] = None,
		leverage: Optional[int] = None,
		open_type: int = 1,
		trend: int = 1,
	) -> Any:
		"""Place a standalone plan/trigger order (stop-loss or take-profit).

		Args:
			symbol: Contract symbol.
			side: MEXC side code (same scheme as ``submit_order``).
			vol: Size in contracts.
			trigger_price: Price that fires the order.
			trigger_type: 1=price <= trigger (stop for long / TP for short),
				2=price >= trigger (TP for long / stop for short).
			execute_cycle: 1=24h, 2=7d validity.
			order_type: Resulting order type after trigger (5=market, 1=limit).
			price: Limit price (only when order_type=1).
			leverage: Optional leverage.
			open_type: 1=isolated, 2=cross.
			trend: 1=latest price, 2=fair price, 3=index price.

		Returns:
			Decoded plan-order id payload.
		"""
		body: Dict[str, Any] = {
			"symbol": symbol,
			"side": int(side),
			"vol": float(vol),
			"triggerPrice": float(trigger_price),
			"triggerType": int(trigger_type),
			"executeCycle": int(execute_cycle),
			"orderType": int(order_type),
			"openType": int(open_type),
			"trend": int(trend),
		}
		if price is not None:
			body["price"] = float(price)
		if leverage is not None:
			body["leverage"] = int(leverage)

		return await self._request(
			"POST", "/api/v1/private/planorder/place", body=body, auth=True
		)

	async def change_stop_price(
		self,
		stop_order_id: int,
		*,
		stop_loss_price: Optional[float] = None,
		take_profit_price: Optional[float] = None,
	) -> Any:
		"""Modify the SL/TP price of an existing position-bound stop order.

		Use this to trail a stop after TP1/TP2 without cancelling + re-placing.

		Args:
			stop_order_id: The stop order id returned by MEXC when the SL/TP
				was first attached (or ``positionId`` for position-level SL/TP
				depending on MEXC account mode).
			stop_loss_price: New SL price, or None to leave unchanged.
			take_profit_price: New TP price, or None to leave unchanged.

		Returns:
			Decoded exchange response.

		Raises:
			ValueError: If neither price is provided.
		"""
		if stop_loss_price is None and take_profit_price is None:
			raise ValueError("Provide stop_loss_price and/or take_profit_price")

		body: Dict[str, Any] = {"orderId": int(stop_order_id)}
		if stop_loss_price is not None:
			body["stopLossPrice"] = float(stop_loss_price)
		if take_profit_price is not None:
			body["takeProfitPrice"] = float(take_profit_price)

		return await self._request(
			"POST",
			"/api/v1/private/stoporder/change_price",
			body=body,
			auth=True,
		)

	async def cancel_trigger_order(self, order_ids: list) -> Any:
		"""Cancel one or more pending plan/trigger orders.

		Args:
			order_ids: List of plan-order ids to cancel.

		Returns:
			Decoded exchange response.
		"""
		body = [{"orderId": int(oid)} for oid in order_ids]
		return await self._request(
			"POST",
			"/api/v1/private/planorder/cancel",
			body=body,
			auth=True,
		)

	async def cancel_stop_order(self, stop_plan_order_ids: list) -> Any:
		"""Cancel position-bound stop orders (SL/TP attached to open positions).

		Args:
			stop_plan_order_ids: List of stop-plan order ids to cancel.

		Returns:
			Decoded exchange response.
		"""
		body = [{"stopPlanOrderId": int(oid)} for oid in stop_plan_order_ids]
		return await self._request(
			"POST", "/api/v1/private/stoporder/cancel", body=body, auth=True
		)

	async def get_open_positions(self, symbol: Optional[str] = None) -> Any:
		"""Fetch currently open futures positions.

		Args:
			symbol: Optional contract symbol filter.

		Returns:
			Decoded positions payload.
		"""
		params: Dict[str, Any] = {}
		if symbol is not None:
			params["symbol"] = symbol
		return await self._request(
			"GET", "/api/v1/private/position/open_positions", params=params, auth=True
		)

	async def get_api_permission_snapshot(self) -> Dict[str, Any]:
		"""Fetch API-key permission payload from private account endpoints.

		Returns:
			Dictionary containing source path and decoded payload.

		Raises:
			RuntimeError: If no permission-capable endpoint can be queried.
		"""
		candidate_paths = [
			"/api/v1/private/account/api_key",
			"/api/v1/private/account/apiKey",
			"/api/v1/private/account/security",
			"/api/v1/private/account/info",
		]

		for path in candidate_paths:
			try:
				payload = await self._request("GET", path, auth=True)
				return {"source": path, "payload": payload}
			except aiohttp.ClientResponseError as exc:
				if exc.status in {400, 401, 403, 404, 405}:
					LOGGER.debug("Permission endpoint unavailable path=%s status=%s", path, exc.status)
					continue
				raise
			except RuntimeError:
				continue

		# Fallback to account info endpoint in case permissions are embedded there.
		try:
			payload = await self.get_account_info()
		except Exception as exc:  # noqa: BLE001
			raise RuntimeError("Unable to retrieve API permission metadata from MEXC") from exc

		return {"source": "/api/v1/private/account/assets", "payload": payload}

	async def _request(
		self,
		method: str,
		path: str,
		params: Optional[Dict[str, Any]] = None,
		body: Optional[Any] = None,
		auth: bool = False,
	) -> Any:
		"""Perform a rate-limited HTTP request.

		Args:
			method: HTTP method.
			path: Relative REST path.
			params: URL query parameters.
			body: JSON body for non-GET endpoints.
			auth: Whether to attach auth headers.

		Returns:
			Response payload.

		Raises:
			RuntimeError: If response format indicates an API-level failure.
		"""
		if self._session is None:
			self._session = aiohttp.ClientSession(timeout=self._timeout)
			self._owns_session = True

		await self._limiter.acquire()

		url = f"{self._base_url}{path}"
		headers: Dict[str, str] = {"Content-Type": "application/json"}
		json_body: Optional[Any] = body
		request_param_string = self._build_request_param_string(params=params, body=body)

		if auth:
			headers.update(self._build_auth_headers(request_param_string=request_param_string))

		LOGGER.debug(
			"Sending %s request to %s with params=%s auth=%s",
			method,
			path,
			params,
			auth,
		)

		assert self._session is not None
		async with self._session.request(
			method=method,
			url=url,
			params=params,
			json=json_body,
			headers=headers,
		) as response:
			text = await response.text()
			if response.status >= 400:
				snippet = text[:1000] if text else "<empty body>"
				raise RuntimeError(
					f"MEXC HTTP {response.status} on {method} {path} — body: {snippet}"
				)

			payload: Any
			try:
				payload = json.loads(text)
			except json.JSONDecodeError:
				payload = text

		if isinstance(payload, dict):
			# MEXC futures responses typically return either success/code or direct data.
			success = payload.get("success", True)
			code = payload.get("code", 0)
			if success is False or (isinstance(code, int) and code != 0):
				msg = payload.get("message") or payload.get("msg") or "unknown api error"
				raise RuntimeError(f"MEXC API error code={code}: {msg}")
			if "data" in payload:
				return payload["data"]
		return payload

	def _build_auth_headers(self, request_param_string: str) -> Dict[str, str]:
		"""Build signed headers for private MEXC requests.

		Args:
			request_param_string: Canonical query/body string for signature.

		Returns:
			Header dictionary without exposing sensitive values in logs.
		"""
		request_time = str(int(time.time() * 1000))
		signature_payload = f"{self._api_key}{request_time}{request_param_string}"
		signature = hmac.new(
			self._api_secret.encode("utf-8"),
			signature_payload.encode("utf-8"),
			hashlib.sha256,
		).hexdigest()

		return {
			"ApiKey": self._api_key,
			"Request-Time": request_time,
			"Signature": signature,
		}

	@staticmethod
	def _build_request_param_string(
		params: Optional[Dict[str, Any]] = None,
		body: Optional[Any] = None,
	) -> str:
		"""Create canonical request parameter string for signing.

		Args:
			params: Query parameters.
			body: JSON body (dict or list).

		Returns:
			Canonical string used by MEXC signing algorithm.
		"""
		if body is not None:
			if isinstance(body, dict):
				return json.dumps(body, separators=(",", ":"), sort_keys=True)
			return json.dumps(body, separators=(",", ":"))
		if not params:
			return ""
		normalized = {key: params[key] for key in sorted(params.keys())}
		return urlencode(normalized, doseq=True)
