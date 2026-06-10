"""Async OKX V5 REST client.

Covers the endpoints needed by the volume-farmer bot:
  - Public:  candles, instruments, ticker
  - Private: balance, positions, set leverage, set position mode,
             place order (with attached TP/SL algo orders), cancel,
             get order, list pending orders

Sign convention (per OKX V5 docs):
    timestamp = ISO 8601 UTC with milliseconds, e.g. "2026-05-30T12:34:56.789Z"
    prestring = timestamp + method + requestPath + body
    signature = base64( HMAC-SHA256(secret, prestring) )

Simulated trading (OKX demo) is engaged by passing ``simulated=True`` to the
constructor; that adds the ``x-simulated-trading: 1`` header to every request.
The endpoints, signing, and credentials are otherwise identical.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import aiohttp

LOGGER = logging.getLogger(__name__)

# OKX rate-limit / transient-busy codes that are safe to retry with backoff.
# 50011 = "Requests too frequent"; 50013 = "System busy, try again later".
_RATE_LIMIT_CODES = {"50011", "50013"}


# Map our internal symbol convention to OKX SWAP convention.
def to_okx_inst_id(symbol: str) -> str:
    """``BTC_USDT`` -> ``BTC-USDT-SWAP``. Pass-through if already in OKX form."""
    if "-SWAP" in symbol:
        return symbol
    if "_" in symbol:
        return symbol.replace("_", "-") + "-SWAP"
    if "-" in symbol:
        return symbol + "-SWAP"
    return symbol


# Map our timeframe strings to OKX bar codes.
_TIMEFRAME_TO_OKX = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H",
    "1d": "1D", "1w": "1W",
}


def to_okx_bar(timeframe: str) -> str:
    """``5m`` -> ``5m``, ``1h`` -> ``1H``. OKX uses uppercase for ≥1h."""
    return _TIMEFRAME_TO_OKX.get(timeframe.lower(), timeframe)


class OKXClient:
    """Minimal OKX V5 REST API client (futures / perpetual swaps)."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        *,
        base_url: str = "https://www.okx.com",
        timeout_seconds: int = 15,
        simulated: bool = False,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self._key = api_key
        self._secret = api_secret
        self._passphrase = passphrase
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._simulated = bool(simulated)
        self._session = session
        self._owns_session = session is None
        # Rate-limit (HTTP 429 / OKX 50011) retry policy. Confirmation polls
        # (get_order / get_positions / orders-history) must not surface a
        # transient throttle as "no fill" / "still open" to the executor, so we
        # retry them a bounded number of times with exponential backoff.
        # The same policy covers network-level errors (connection reset,
        # timeout) on GETs — equally transient, equally idempotent.
        self._max_retries = 3
        self._retry_base_s = 0.25
        self._retry_max_s = 2.0
        # Hot-path polls (ticker/order/position inside re-peg loops) get a
        # short per-call timeout so one hung socket cannot eat most of a
        # 20s maker-exit budget waiting on the blanket 15s client timeout.
        self._hot_timeout = aiohttp.ClientTimeout(total=6)

    async def __aenter__(self) -> "OKXClient":
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
            self._owns_session = True
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Public market data
    # ------------------------------------------------------------------

    async def get_candles(
        self,
        symbol: str,
        timeframe: str = "5m",
        *,
        limit: int = 300,
        before: Optional[int] = None,
        after: Optional[int] = None,
        history: bool = False,
    ) -> List[List[str]]:
        """Recent (or history) candlesticks for a SWAP instrument.

        OKX returns raw lists of strings:
            [ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
        where ts is millisecond epoch (string) and confirm is "1" for closed.

        Args:
            symbol: Symbol in our convention (``BTC_USDT``) or OKX form.
            timeframe: e.g. ``5m``, ``1h``.
            limit: max 300 (OKX cap).
            before: paginate backwards (older).
            after: paginate forward (newer).
            history: Use ``/market/history-candles`` for deeper backfill.
        """
        path = "/api/v5/market/history-candles" if history else "/api/v5/market/candles"
        params: Dict[str, Any] = {
            "instId": to_okx_inst_id(symbol),
            "bar": to_okx_bar(timeframe),
            "limit": str(min(int(limit), 300)),
        }
        if before is not None:
            params["before"] = str(int(before))
        if after is not None:
            params["after"] = str(int(after))
        resp = await self._request("GET", path, params=params, auth=False)
        if str(resp.get("code", "0")) != "0":
            # An error payload must be distinguishable from "no new candles":
            # the poll loop treats [] as a quiet bar and would silently stall
            # on a persistent endpoint error.
            raise RuntimeError(f"OKX candles error code={resp.get('code')}: {resp.get('msg')}")
        return list(resp.get("data") or [])

    async def get_instrument(self, symbol: str) -> Dict[str, Any]:
        """Contract spec for one SWAP instrument (ctVal, tickSz, lotSz, ...)."""
        path = "/api/v5/public/instruments"
        params = {"instType": "SWAP", "instId": to_okx_inst_id(symbol)}
        resp = await self._request("GET", path, params=params, auth=False)
        data = resp.get("data") or []
        if not data:
            raise RuntimeError(f"OKX: no instrument data for {symbol}")
        return data[0]

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Latest ticker (last, bid, ask, ...). Hot-path: short timeout."""
        path = "/api/v5/market/ticker"
        params = {"instId": to_okx_inst_id(symbol)}
        resp = await self._request("GET", path, params=params, auth=False, hot=True)
        data = resp.get("data") or []
        if not data:
            raise RuntimeError(f"OKX: no ticker data for {symbol}")
        return data[0]

    async def get_trade_fee(self, symbol: str, *, inst_type: str = "SWAP") -> Dict[str, Any]:
        """ACTUAL fee rates for this account from GET /account/trade-fee.

        Returns the raw row: ``makerU``/``takerU`` (USDT-margined contracts,
        falling back to ``maker``/``taker``) as SIGNED rates — OKX reports
        fees you PAY as negative (e.g. "-0.0002" = 2bps cost). Used at
        startup to verify the config's assumed maker/taker against reality:
        the whole campaign cost model rests on those two numbers, so they
        must never be trusted from a yaml comment alone.
        """
        resp = await self._request(
            "GET", "/api/v5/account/trade-fee",
            params={"instType": inst_type, "instId": to_okx_inst_id(symbol)},
            auth=True,
        )
        data = resp.get("data") or []
        if not data:
            raise RuntimeError("OKX: no trade-fee data")
        return data[0]

    async def get_server_time_ms(self) -> int:
        """OKX server time (ms epoch). Used to sanity-check local clock drift
        at startup — a drift past OKX's ~30s signing window breaks every
        private call with code 50102."""
        resp = await self._request("GET", "/api/v5/public/time", auth=False)
        data = resp.get("data") or []
        if not data:
            raise RuntimeError("OKX: no server time")
        return int(data[0]["ts"])

    # ------------------------------------------------------------------
    # Account / positions
    # ------------------------------------------------------------------

    async def get_balance(self, ccy: Optional[str] = None) -> Dict[str, Any]:
        path = "/api/v5/account/balance"
        params = {"ccy": ccy} if ccy else None
        return await self._request("GET", path, params=params, auth=True)

    async def get_positions(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        path = "/api/v5/account/positions"
        params: Dict[str, Any] = {"instType": "SWAP"}
        if symbol is not None:
            params["instId"] = to_okx_inst_id(symbol)
        return await self._request("GET", path, params=params, auth=True, hot=True)

    async def set_position_mode(self, mode: str) -> Dict[str, Any]:
        """Set hedge ('long_short_mode') vs one-way ('net_mode') globally."""
        path = "/api/v5/account/set-position-mode"
        body = {"posMode": mode}
        return await self._request("POST", path, body=body, auth=True)

    async def get_account_config(self) -> Dict[str, Any]:
        """Current account-mode/level (``acctLv``: 1 spot, 2 spot+futures, 3 multi-ccy, 4 portfolio)."""
        return await self._request("GET", "/api/v5/account/config", auth=True)

    async def set_account_level(self, acct_lv: str) -> Dict[str, Any]:
        """Upgrade account mode. ``acct_lv``: '1'..'4'. Perpetual swaps need >= '2'."""
        body = {"acctLv": str(acct_lv)}
        return await self._request(
            "POST", "/api/v5/account/set-account-level", body=body, auth=True,
        )

    async def set_leverage(
        self, symbol: str, leverage: int, *, mgn_mode: str = "isolated",
        pos_side: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = "/api/v5/account/set-leverage"
        body: Dict[str, Any] = {
            "instId": to_okx_inst_id(symbol),
            "lever": str(int(leverage)),
            "mgnMode": mgn_mode,
        }
        if pos_side is not None:
            body["posSide"] = pos_side
        return await self._request("POST", path, body=body, auth=True)

    # ------------------------------------------------------------------
    # Trading
    # ------------------------------------------------------------------

    async def place_order(
        self,
        symbol: str,
        *,
        side: str,                # "buy" or "sell"
        pos_side: Optional[str],  # "long"/"short" in hedge mode, None in one-way
        td_mode: str = "isolated",
        ord_type: str = "post_only",
        sz: str = "1",            # size in CONTRACTS (not USD)
        px: Optional[str] = None,
        client_oid: Optional[str] = None,
        reduce_only: bool = False,
        tp_trigger_px: Optional[str] = None,
        tp_ord_px: Optional[str] = None,         # set to price for MAKER TP; "-1" for market
        sl_trigger_px: Optional[str] = None,
        sl_ord_px: Optional[str] = None,         # set to "-1" for market SL
        tp_trigger_px_type: str = "last",        # "last" | "mark" | "index"
        sl_trigger_px_type: str = "last",
    ) -> Dict[str, Any]:
        """Place an order, optionally with an attached TP/SL algo bundle.

        The 'maker-TP' win: when ``tp_ord_px`` is a price (not "-1"), the TP
        trigger fires a LIMIT order at that price -> fills as maker.
        """
        body: Dict[str, Any] = {
            "instId": to_okx_inst_id(symbol),
            "tdMode": td_mode,
            "side": side,
            "ordType": ord_type,
            "sz": str(sz),
        }
        if pos_side is not None:
            body["posSide"] = pos_side
        if px is not None:
            body["px"] = str(px)
        if client_oid:
            body["clOrdId"] = client_oid
        if reduce_only:
            body["reduceOnly"] = True
        if tp_trigger_px is not None or sl_trigger_px is not None:
            attach: Dict[str, Any] = {}
            if tp_trigger_px is not None:
                attach["tpTriggerPx"] = str(tp_trigger_px)
                attach["tpOrdPx"] = str(tp_ord_px if tp_ord_px is not None else "-1")
                attach["tpTriggerPxType"] = tp_trigger_px_type
            if sl_trigger_px is not None:
                attach["slTriggerPx"] = str(sl_trigger_px)
                attach["slOrdPx"] = str(sl_ord_px if sl_ord_px is not None else "-1")
                attach["slTriggerPxType"] = sl_trigger_px_type
            body["attachAlgoOrds"] = [attach]
        return await self._request("POST", "/api/v5/trade/order", body=body, auth=True)

    async def cancel_order(
        self, symbol: str, *,
        ord_id: Optional[str] = None, client_oid: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not (ord_id or client_oid):
            raise ValueError("cancel_order: need ord_id or client_oid")
        body: Dict[str, Any] = {"instId": to_okx_inst_id(symbol)}
        if ord_id:
            body["ordId"] = ord_id
        if client_oid:
            body["clOrdId"] = client_oid
        return await self._request("POST", "/api/v5/trade/cancel-order", body=body, auth=True)

    async def get_order(
        self, symbol: str, *,
        ord_id: Optional[str] = None, client_oid: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not (ord_id or client_oid):
            raise ValueError("get_order: need ord_id or client_oid")
        params: Dict[str, Any] = {"instId": to_okx_inst_id(symbol)}
        if ord_id:
            params["ordId"] = ord_id
        if client_oid:
            params["clOrdId"] = client_oid
        return await self._request("GET", "/api/v5/trade/order", params=params, auth=True, hot=True)

    async def amend_order(
        self, symbol: str, *,
        ord_id: Optional[str] = None, client_oid: Optional[str] = None,
        new_px: Optional[str] = None, new_sz: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Re-price (and/or re-size) a resting order in ONE call.

        Replaces the cancel+replace re-peg dance: half the round trips, no
        naked window with nothing resting on the book, and the amended order
        keeps its ordId so fill polling continues seamlessly. ``cxlOnFail``
        is deliberately False — on amend failure the original order stays
        resting and the caller falls back to cancel+replace explicitly.
        """
        if not (ord_id or client_oid):
            raise ValueError("amend_order: need ord_id or client_oid")
        if not (new_px or new_sz):
            raise ValueError("amend_order: need new_px or new_sz")
        body: Dict[str, Any] = {"instId": to_okx_inst_id(symbol)}
        if ord_id:
            body["ordId"] = ord_id
        if client_oid:
            body["clOrdId"] = client_oid
        if new_px is not None:
            body["newPx"] = str(new_px)
        if new_sz is not None:
            body["newSz"] = str(new_sz)
        return await self._request("POST", "/api/v5/trade/amend-order", body=body, auth=True)

    async def get_pending_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"instType": "SWAP"}
        if symbol is not None:
            params["instId"] = to_okx_inst_id(symbol)
        return await self._request("GET", "/api/v5/trade/orders-pending", params=params, auth=True)

    async def get_pending_algos(
        self, symbol: Optional[str] = None, *, algo_type: str = "conditional",
    ) -> Dict[str, Any]:
        """List currently-resting algo orders (TP/SL attached or standalone)."""
        params: Dict[str, Any] = {"ordType": algo_type, "instType": "SWAP"}
        if symbol is not None:
            params["instId"] = to_okx_inst_id(symbol)
        return await self._request("GET", "/api/v5/trade/orders-algo-pending", params=params, auth=True)

    async def cancel_algos(
        self, symbol: str, algo_ids: List[str],
    ) -> Dict[str, Any]:
        """Cancel algo orders, chunked at OKX's 10-per-call cap.

        Previously truncated silently at 10, leaving stale TP/SL algos
        resting; now every id is cancelled across as many calls as needed
        and the last payload is returned (per-chunk errors are logged).
        """
        if not algo_ids:
            return {"code": "0", "data": []}
        last: Dict[str, Any] = {"code": "0", "data": []}
        for i in range(0, len(algo_ids), 10):
            chunk = algo_ids[i:i + 10]
            body: List[Dict[str, Any]] = [
                {"instId": to_okx_inst_id(symbol), "algoId": str(a)} for a in chunk
            ]
            last = await self._request(
                "POST", "/api/v5/trade/cancel-algos", body=body, auth=True,
            )
            if str(last.get("code", "")) not in ("0", ""):
                LOGGER.warning("cancel_algos chunk failed code=%s ids=%s",
                               last.get("code"), chunk)
        return last

    async def close_position_market(
        self, symbol: str, *,
        mgn_mode: str = "isolated",
        pos_side: Optional[str] = None,
        ccy: Optional[str] = None,
        auto_cxl: bool = False,
    ) -> Dict[str, Any]:
        """Force-close the entire position at market (taker). Last-resort fallback.

        ``auto_cxl`` maps to OKX ``autoCxl``: when True, any pending close orders
        on the instrument (e.g. a resting reduceOnly maker-exit limit) are
        cancelled automatically so the market close is not rejected by a
        conflicting order. Use it for an operator-triggered manual close.
        """
        body: Dict[str, Any] = {
            "instId": to_okx_inst_id(symbol),
            "mgnMode": mgn_mode,
        }
        if pos_side is not None:
            body["posSide"] = pos_side
        if ccy is not None:
            body["ccy"] = ccy
        if auto_cxl:
            body["autoCxl"] = True
        return await self._request(
            "POST", "/api/v5/trade/close-position", body=body, auth=True,
        )

    async def get_orders_history(
        self, symbol: Optional[str] = None, *,
        inst_type: str = "SWAP",
        ord_id: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 30,
        archive: bool = False,
    ) -> Dict[str, Any]:
        """Recent finished orders (7d) or the 3-month archive.

        Used to resolve a closing fill's avgPx / fee / realized pnl. Pass
        ``ord_id`` to fetch one specific order; otherwise returns the most
        recent ``limit`` orders for the instrument. ``pnl`` on a closing order
        is OKX's authoritative GROSS realized PnL (fees are separate in ``fee``,
        which is negative when paid).
        """
        path = ("/api/v5/trade/orders-history-archive" if archive
                else "/api/v5/trade/orders-history")
        params: Dict[str, Any] = {"instType": inst_type, "limit": str(int(limit))}
        if symbol is not None:
            params["instId"] = to_okx_inst_id(symbol)
        if ord_id:
            params["ordId"] = ord_id
        if state:
            params["state"] = state
        return await self._request("GET", path, params=params, auth=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sign(self, ts: str, method: str, request_path: str, body: str) -> str:
        msg = f"{ts}{method.upper()}{request_path}{body}"
        mac = hmac.new(self._secret.encode(), msg.encode(), hashlib.sha256).digest()
        return base64.b64encode(mac).decode()

    @staticmethod
    def _iso_ts() -> str:
        now = time.time()
        secs = time.gmtime(now)
        millis = int((now - int(now)) * 1000)
        return time.strftime("%Y-%m-%dT%H:%M:%S", secs) + f".{millis:03d}Z"

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,  # dict for most endpoints; list for batch endpoints
        auth: bool = False,
        hot: bool = False,           # hot-path poll: short per-call timeout
    ) -> Dict[str, Any]:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
            self._owns_session = True

        query = ""
        if params:
            query = "?" + urlencode({k: v for k, v in params.items() if v is not None})
        request_path = path + query
        body_str = json.dumps(body, separators=(",", ":")) if body is not None else ""

        url = self._base_url + request_path
        # ONLY GETs are retried (throttles AND network errors). GETs
        # (get_order / get_positions / orders-history) are idempotent.
        # Re-sending a POST (e.g. place_order) could double-submit or, if the
        # first silently landed, orphan an order — so POSTs surface errors to
        # the caller instead.
        retryable = method.upper() == "GET"
        attempt = 0
        while True:
            # Headers (and the auth signature) are rebuilt every attempt: the
            # signature embeds a timestamp OKX rejects once stale, so a retried
            # request must be re-signed.
            headers: Dict[str, str] = {"Content-Type": "application/json"}
            if auth:
                if not (self._key and self._secret and self._passphrase):
                    raise RuntimeError("OKX auth requires key, secret AND passphrase")
                ts = self._iso_ts()
                headers["OK-ACCESS-KEY"] = self._key
                headers["OK-ACCESS-SIGN"] = self._sign(ts, method, request_path, body_str)
                headers["OK-ACCESS-TIMESTAMP"] = ts
                headers["OK-ACCESS-PASSPHRASE"] = self._passphrase
            if self._simulated:
                headers["x-simulated-trading"] = "1"

            kwargs: Dict[str, Any] = {"headers": headers}
            if body_str:
                kwargs["data"] = body_str
            if hot:
                kwargs["timeout"] = self._hot_timeout

            try:
                async with self._session.request(method.upper(), url, **kwargs) as resp:
                    status = resp.status
                    text = await resp.text()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                # Network-level transient (connection reset, DNS hiccup, hung
                # socket). Same treatment as a throttle: bounded retry for
                # idempotent GETs so confirmation polls don't surface a socket
                # blip as "no fill" / "still open".
                if retryable and attempt < self._max_retries:
                    delay = min(self._retry_base_s * (2 ** attempt), self._retry_max_s)
                    LOGGER.warning(
                        "OKX network error (%s: %s) — retry %d/%d in %.2fs path=%s",
                        type(exc).__name__, exc, attempt + 1, self._max_retries, delay, request_path,
                    )
                    attempt += 1
                    await asyncio.sleep(delay)
                    continue
                raise

            # Rate-limit backoff BEFORE body parsing: edge throttles
            # (Cloudflare) return HTML bodies on 429/5xx — exactly the case
            # the retry exists for, so the status check must not depend on
            # the body being JSON.
            if retryable and status in (429, 502, 503, 504) and attempt < self._max_retries:
                delay = min(self._retry_base_s * (2 ** attempt), self._retry_max_s)
                LOGGER.warning(
                    "OKX throttled/unavailable (HTTP %s) — retry %d/%d in %.2fs path=%s",
                    status, attempt + 1, self._max_retries, delay, request_path,
                )
                attempt += 1
                await asyncio.sleep(delay)
                continue

            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                raise RuntimeError(f"OKX non-JSON response: HTTP {status}: {text[:200]}")
            code = str(payload.get("code", ""))
            # OKX-level too-frequent/busy codes arrive with HTTP 200.
            if retryable and code in _RATE_LIMIT_CODES and attempt < self._max_retries:
                delay = min(self._retry_base_s * (2 ** attempt), self._retry_max_s)
                LOGGER.warning(
                    "OKX rate-limited (code=%s) — retry %d/%d in %.2fs path=%s",
                    code, attempt + 1, self._max_retries, delay, request_path,
                )
                attempt += 1
                await asyncio.sleep(delay)
                continue
            if status >= 400:
                raise RuntimeError(f"OKX HTTP {status}: {payload}")
            if code != "0" and code != "":
                # OKX returns code=0 on success. Non-zero is an API-level error.
                # Some endpoints (place_order) return code=0 outer with per-item
                # errors in data[].sCode; we let the caller handle those.
                if code in {"50112", "50113", "50111", "50114", "50102"}:
                    raise RuntimeError(f"OKX auth error code={code}: {payload.get('msg')}")
                LOGGER.warning("OKX returned code=%s msg=%s path=%s",
                               code, payload.get("msg"), request_path)
            return payload
