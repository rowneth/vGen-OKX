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

	async def get_account_info(self) -> Any:
		"""Fetch private futures account assets/margin info.

		Returns:
			Decoded account payload from MEXC private endpoint.
		"""
		path = "/api/v1/private/account/assets"
		return await self._request("GET", path, auth=True)

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
		body: Optional[Dict[str, Any]] = None,
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
		json_body: Optional[Dict[str, Any]] = body
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
			response.raise_for_status()

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
		body: Optional[Dict[str, Any]] = None,
	) -> str:
		"""Create canonical request parameter string for signing.

		Args:
			params: Query parameters.
			body: JSON body.

		Returns:
			Canonical string used by MEXC signing algorithm.
		"""
		if body:
			return json.dumps(body, separators=(",", ":"), sort_keys=True)
		if not params:
			return ""
		normalized = {key: params[key] for key in sorted(params.keys())}
		return urlencode(normalized, doseq=True)
