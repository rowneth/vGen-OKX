# Multi-Exchange Adapter Guide
## Adding Bybit (and any future exchange) without touching strategy code

**Target audience:** AI coding agents and developers  
**Scope:** Plugging this bot into Bybit Futures while keeping `VolumeFarmerSession`, all signal logic, and the Telegram/logging stack completely unchanged.

---

## 1. How the Current System Works

Before building anything, understand the existing data flow:

```
MEXC REST API
    ↓  get_klines()
run_volume_farmer.py  ←── config YAML (exchange.timeframe, entry.*, risk.*)
    ↓  on_new_candle(df)
VolumeFarmerSession              ← STRATEGY LIVES HERE — never touch this
    ↓  emits FarmerEvent("entry" | "exit" | "halt" | ...)
LiveVolumeExecutor               ← ORDER EXECUTION — exchange-specific
    ↓  submit_order / get_open_positions / get_order_by_external_oid
MEXCClient                       ← EXCHANGE CLIENT — only exchange-specific layer
```

**Key insight:** The strategy (`VolumeFarmerSession`) sees **only a pandas DataFrame** of OHLCV candles. It never talks to any exchange. All exchange coupling lives in three files:

| File | What it does | Must change for new exchange |
|---|---|---|
| `src/exchange/mexc_client.py` | MEXC REST signing, endpoints | ✅ Yes — write a new client |
| `src/exchange/mexc_models.py` | Typed dataclasses (`Candle`, `Order`) | ❌ No — already generic |
| `src/execution/live_volume_executor.py` | Bridges FarmerEvents → real orders | ⚠️ Partially — thin adapter needed |
| `src/execution/volume_farmer.py` | Core strategy | ❌ Never touch |
| `scripts/run_volume_farmer.py` | Wiring + kline polling loop | ⚠️ Small config section only |

---

## 2. The Exchange Contract (Interface to Implement)

Every exchange adapter must provide **exactly these 7 async methods**. This is the interface `LiveVolumeExecutor` calls:

```python
# src/exchange/exchange_client_base.py  (create this file)

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ExchangeClientBase(ABC):
    """Minimum interface every exchange adapter must implement.
    
    All methods return raw dicts/lists — LiveVolumeExecutor normalises them
    via ExchangeAdapter (see Section 5).
    """

    # ── Market data ──────────────────────────────────────────────────

    @abstractmethod
    async def get_klines(
        self,
        symbol: str,
        interval: str,           # exchange-native interval string
        start: Optional[int] = None,   # ms timestamp
        end: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Any:
        """Return OHLCV candle list. Shape normalised by adapter."""

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Any:
        """Return dict with at minimum: bid, ask, last price fields."""

    @abstractmethod
    async def get_contract_detail(self, symbol: Optional[str] = None) -> Any:
        """Return contract spec: contractSize, tick size, min order qty."""

    # ── Account ──────────────────────────────────────────────────────

    @abstractmethod
    async def get_account_info(self) -> Any:
        """Return account assets. Must be parseable for USDT equity."""

    # ── Orders ───────────────────────────────────────────────────────

    @abstractmethod
    async def submit_order(
        self,
        symbol: str,
        side: int,               # 1=open long, 3=open short (internal code)
        order_type: int,         # 2=post-only (internal code)
        vol: float,              # contracts
        price: Optional[float],
        leverage: Optional[int],
        open_type: int,          # 1=isolated, 2=cross
        external_oid: Optional[str],
        stop_loss_price: Optional[float],
        take_profit_price: Optional[float],
        reduce_only: bool = False,
    ) -> Any:
        """Place an order. Return order id (str or dict with orderId)."""

    @abstractmethod
    async def get_order_by_external_oid(self, symbol: str, external_oid: str) -> Any:
        """Fetch a single order by client order id. Return dict with 'state' field:
            3 = filled, 4 = rejected/cancelled (use these codes regardless of exchange).
        """

    @abstractmethod
    async def get_open_positions(self, symbol: Optional[str] = None) -> Any:
        """Return list of open positions. Each item must have:
            symbol, positionType (1=long, 2=short), state (1/2=open), holdVol.
        """
```

> **Implementation rule:** Each adapter internally translates its exchange's native API format into the field names shown above. The `LiveVolumeExecutor` only ever reads those specific keys.

---

## 3. Build the Bybit Client

### 3.1 Create `src/exchange/bybit_client.py`

Bybit Unified Margin V5 is the current API. Key differences vs MEXC:

| Concern | MEXC | Bybit V5 |
|---|---|---|
| Auth signing | `HMAC-SHA256(api_key + timestamp + param_string)` | `HMAC-SHA256(timestamp + api_key + recv_window + param_string)` |
| Base URL | `https://contract.mexc.com` | `https://api.bybit.com` |
| Klines endpoint | `GET /api/v1/contract/kline/{symbol}` | `GET /v5/market/kline` |
| Submit order | `POST /api/v1/private/order/create` | `POST /v5/order/create` |
| Side codes | `1=open long, 3=open short` | `Buy` / `Sell` + `openOnly=true` flag |
| Order state | `3=filled, 4=rejected` | `status: Filled / Rejected / Cancelled` |
| Ticker | `bid1, ask1` | `bid1Price, ask1Price` |
| Positions | `holdVol, positionType` | `size, side` (Buy=long, Sell=short) |
| TP/SL on open | `stopLossPrice, takeProfitPrice` in order body | `stopLoss, takeProfit` in order body |
| Post-only type | `type=2` | `orderType=Limit` + `timeInForce=PostOnly` |

```python
# src/exchange/bybit_client.py

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import aiohttp

from exchange.exchange_client_base import ExchangeClientBase
from exchange.rate_limiter import TokenBucketRateLimiter

LOGGER = logging.getLogger(__name__)

_BYBIT_INTERVAL = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15",
    "30m": "30", "1h": "60", "4h": "240", "1d": "D",
}


class BybitClient(ExchangeClientBase):
    """Bybit Unified Margin V5 async REST client."""

    BASE_URL = "https://api.bybit.com"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        timeout_seconds: int = 15,
        requests_per_second: float = 10.0,
        burst_capacity: int = 20,
        recv_window: int = 5000,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = "https://api-testnet.bybit.com" if testnet else self.BASE_URL
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._recv_window = recv_window
        self._session = session
        self._owns_session = session is None
        self._limiter = TokenBucketRateLimiter(
            rate_per_second=requests_per_second,
            burst_capacity=burst_capacity,
        )

    async def __aenter__(self) -> "BybitClient":
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

    # ── Signing ──────────────────────────────────────────────────────

    def _sign(self, timestamp: str, payload_str: str) -> str:
        msg = timestamp + self._api_key + str(self._recv_window) + payload_str
        return hmac.new(
            self._api_secret.encode(), msg.encode(), hashlib.sha256
        ).hexdigest()

    def _auth_headers(self, payload_str: str) -> Dict[str, str]:
        ts = str(int(time.time() * 1000))
        return {
            "X-BAPI-API-KEY": self._api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-SIGN": self._sign(ts, payload_str),
            "X-BAPI-RECV-WINDOW": str(self._recv_window),
            "Content-Type": "application/json",
        }

    # ── Internal request ─────────────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        auth: bool = False,
    ) -> Any:
        await self._limiter.acquire()
        url = self._base_url + path
        headers = {}
        if auth:
            if method == "GET":
                payload_str = urlencode(params or {})
            else:
                payload_str = json.dumps(body or {})
            headers = self._auth_headers(payload_str)

        async with self._session.request(
            method, url, params=params, json=body, headers=headers,
        ) as resp:
            data = await resp.json()
            if data.get("retCode") not in (0, None):
                LOGGER.warning("Bybit API error %s: %s", data.get("retCode"), data.get("retMsg"))
            return data

    # ── ExchangeClientBase implementation ────────────────────────────

    async def get_klines(
        self,
        symbol: str,
        interval: str = "5",
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Any:
        params: Dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "interval": _BYBIT_INTERVAL.get(interval, interval),
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if limit:
            params["limit"] = limit
        return await self._request("GET", "/v5/market/kline", params=params)

    async def get_ticker(self, symbol: str) -> Any:
        params = {"category": "linear", "symbol": symbol}
        return await self._request("GET", "/v5/market/tickers", params=params)

    async def get_contract_detail(self, symbol: Optional[str] = None) -> Any:
        params: Dict[str, Any] = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/v5/market/instruments-info", params=params)

    async def get_account_info(self) -> Any:
        params = {"accountType": "UNIFIED"}
        return await self._request("GET", "/v5/account/wallet-balance", params=params, auth=True)

    async def submit_order(
        self,
        symbol: str,
        side: int,
        order_type: int,
        vol: float,
        price: Optional[float] = None,
        leverage: Optional[int] = None,
        open_type: int = 1,
        external_oid: Optional[str] = None,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        reduce_only: bool = False,
        **_: Any,
    ) -> Any:
        # Translate internal side codes to Bybit strings
        # Internal: 1=open long, 2=close short, 3=open short, 4=close long
        if side in (1, 4):
            bybit_side = "Buy"
        else:
            bybit_side = "Sell"

        body: Dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "side": bybit_side,
            "orderType": "Limit",
            "timeInForce": "PostOnly" if order_type == 2 else "GTC",
            "qty": str(vol),
            "reduceOnly": reduce_only,
        }
        if price is not None:
            body["price"] = str(price)
        if external_oid is not None:
            body["orderLinkId"] = external_oid
        if stop_loss_price is not None:
            body["stopLoss"] = str(stop_loss_price)
        if take_profit_price is not None:
            body["takeProfit"] = str(take_profit_price)

        # Set leverage before placing order (Bybit requires a separate call or position mode)
        if leverage is not None:
            await self._set_leverage(symbol, leverage)

        return await self._request("POST", "/v5/order/create", body=body, auth=True)

    async def get_order_by_external_oid(self, symbol: str, external_oid: str) -> Any:
        params = {
            "category": "linear",
            "symbol": symbol,
            "orderLinkId": external_oid,
        }
        raw = await self._request("GET", "/v5/order/realtime", params=params, auth=True)
        # Normalise to the field names LiveVolumeExecutor expects
        items = (raw.get("result") or {}).get("list") or []
        if not items:
            # Check history (filled/cancelled orders move to history)
            hist_params = {
                "category": "linear",
                "symbol": symbol,
                "orderLinkId": external_oid,
                "limit": 1,
            }
            hist = await self._request("GET", "/v5/order/history", params=hist_params, auth=True)
            items = (hist.get("result") or {}).get("list") or []
        if not items:
            return {}
        order = items[0]
        # Map Bybit order status → internal state codes (3=filled, 4=rejected)
        bybit_status = order.get("orderStatus", "")
        if bybit_status == "Filled":
            state = 3
        elif bybit_status in ("Rejected", "Cancelled", "Deactivated"):
            state = 4
        else:
            state = 0  # still pending
        return {
            "state": state,
            "dealAvgPrice": order.get("avgPrice") or order.get("price"),
            "makerFee": order.get("cumExecFee", 0),
            "takerFee": 0,
            "orderLinkId": external_oid,
            "_raw": order,
        }

    async def get_open_positions(self, symbol: Optional[str] = None) -> Any:
        params: Dict[str, Any] = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol
        raw = await self._request("GET", "/v5/position/list", params=params, auth=True)
        items = (raw.get("result") or {}).get("list") or []
        # Normalise to the field names LiveVolumeExecutor expects
        normalised = []
        for pos in items:
            size = float(pos.get("size") or 0)
            if size <= 0:
                continue
            bybit_side = pos.get("side", "")
            normalised.append({
                "symbol": pos.get("symbol"),
                "positionType": 1 if bybit_side == "Buy" else 2,
                "state": 1,        # Bybit only returns open positions in this endpoint
                "holdVol": size,
                "openAvgPrice": pos.get("avgPrice"),
                "_raw": pos,
            })
        return normalised

    # ── Bybit-specific helpers ────────────────────────────────────────

    async def _set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for a symbol (Bybit requires explicit call)."""
        body = {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }
        await self._request("POST", "/v5/position/set-leverage", body=body, auth=True)
```

---

## 4. Normalise Kline Data

Bybit klines have a different field order than MEXC. The runner calls `_normalize_kline_payload` which currently knows only MEXC format. Extend it:

```python
# src/data/historical.py  — add this function alongside _normalize_kline_payload

def _normalize_bybit_kline_payload(payload: Any) -> pd.DataFrame:
    """Normalise Bybit V5 /v5/market/kline response into standard OHLCV DataFrame.
    
    Bybit kline list format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
    """
    items = (payload.get("result") or {}).get("list") or []
    if not items:
        return pd.DataFrame()

    rows = []
    for item in items:
        # item is [startTime_ms, open, high, low, close, volume, turnover]
        ts_ms = int(item[0])
        rows.append({
            "open_time":  pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
            "close_time": pd.Timestamp(ts_ms, unit="ms", tz="UTC"),  # Bybit gives start; close = start + interval - 1ms
            "open":   float(item[1]),
            "high":   float(item[2]),
            "low":    float(item[3]),
            "close":  float(item[4]),
            "volume": float(item[5]),
        })

    df = pd.DataFrame(rows)
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
```

---

## 5. Add Thin Exchange Adapter Layer (Normalisation)

Instead of modifying `LiveVolumeExecutor`, add an adapter class that translates any client's raw responses into the field names the executor expects. This keeps `LiveVolumeExecutor` exchange-agnostic:

```python
# src/exchange/exchange_adapter.py

from __future__ import annotations
from typing import Any, Optional
from exchange.exchange_client_base import ExchangeClientBase


class ExchangeAdapter:
    """Wraps any ExchangeClientBase and exposes a MEXCClient-compatible interface.
    
    LiveVolumeExecutor will accept this in place of MEXCClient directly.
    The adapter's job: ensure every method returns data in the exact dict shape
    that LiveVolumeExecutor already knows how to parse.
    """

    def __init__(self, client: ExchangeClientBase) -> None:
        self._client = client

    def __getattr__(self, name: str) -> Any:
        """Delegate all calls directly to the underlying client.
        
        Since BybitClient already normalises field names in its methods,
        this passthrough is sufficient for most calls.
        """
        return getattr(self._client, name)
```

Then update the type hint in `LiveVolumeExecutor`:

```python
# src/execution/live_volume_executor.py — change ONE import line

# Before:
from exchange.mexc_client import MEXCClient
# ...
@dataclass
class LiveVolumeExecutor:
    client: MEXCClient

# After:
from exchange.exchange_client_base import ExchangeClientBase
# ...
@dataclass  
class LiveVolumeExecutor:
    client: ExchangeClientBase   # accepts MEXCClient OR BybitClient
```

---

## 6. Config for Bybit

Create `config/config_volume_farmer_bybit.yaml` — identical to the current filter4h config except the `exchange` section:

```yaml
app:
  name: volume-farmer-bybit
  environment: production
  timezone: Asia/Colombo
  log_level: INFO

exchange:
  symbol: BTCUSDT          # ← Bybit uses no underscore
  timeframe: 5m
  provider: bybit          # ← new field: tells runner which client to instantiate

fees:
  maker: 0.0002            # Bybit linear perpetual maker fee (0.02%)
  taker: 0.0055            # Bybit linear perpetual taker fee (0.055%)
  rebate_pct: 0.00         # Bybit does NOT have a rebate program — set to 0

farmer:
  capital_usd: 30.0
  leverage: 0
  margin_fraction_per_trade: 0.05
  sizing:
    dynamic_leverage: true
    risk_per_trade_pct: 0.025
    max_leverage: 100       # Bybit max for BTC perpetual
    min_leverage: 5
  tp_bps: 8.0
  sl_bps: 50.0
  max_hold_bars: 999
  entry:
    mode: micro_momentum
    min_bar_range_bps: 3.0
    max_bar_range_bps: 40.0
    skip_hours: [1, 6, 12, 22]
  alternate_direction: true
  trend_break:
    enabled: true
    min_bars_held: 3
    adverse_bps: 20.0

risk:
  daily_loss_limit_pct: 0.50
  max_drawdown_pct: 0.95
  consecutive_losses_limit: 10
  consecutive_losses_cooldown_bars: 12
  stop_on_volume_target: false

notifications:
  telegram:
    enabled: true
    send_daily: true
    send_errors: true
    send_milestones: true
    daily_report_hour: 0
```

> **Fee note:** Bybit does not have a volume rebate program like MEXC's $80/M bonus. The economics are different — Bybit profit comes from raw P&L only. At 76.7% WR with TP=8bps and SL=50bps, the bot will bleed. Consider raising TP to at least 15bps on Bybit (backtest first) to make the strategy self-sustaining without bonus income.

---

## 7. Update the Runner to Support Multiple Exchanges

Edit `scripts/run_volume_farmer.py` — the only section that needs to change is the client instantiation block:

```python
# scripts/run_volume_farmer.py
# Replace the client creation section (~line 755) with this:

from exchange.mexc_client import MEXCClient
from exchange.bybit_client import BybitClient   # ← add this import

# ... inside _run() function, replace the client block:

provider = str(config.get("exchange", {}).get("provider", "mexc")).lower()

if provider == "bybit":
    api_key    = os.getenv("BYBIT_API_KEY", "").strip()
    api_secret = os.getenv("BYBIT_API_SECRET", "").strip()
    testnet    = str(config.get("exchange", {}).get("testnet", "false")).lower() == "true"
    client = BybitClient(
        api_key=api_key or "paper",
        api_secret=api_secret or "paper",
        testnet=testnet,
    )
    # Bybit uses different kline normalisation
    from data.historical import _normalize_bybit_kline_payload
    _normalize_fn = _normalize_bybit_kline_payload
else:
    # Default: MEXC
    api_key    = os.getenv("MEXC_API_KEY", "").strip()
    api_secret = os.getenv("MEXC_API_SECRET", "").strip()
    client = MEXCClient(
        api_key=api_key or "paper",
        api_secret=api_secret or "paper",
        base_url="https://contract.mexc.com",
    )
    from data.historical import _normalize_kline_payload
    _normalize_fn = _normalize_kline_payload
```

Also update `_fetch_candles` to accept `normalize_fn` as a parameter so the correct parser is used for each exchange.

Also update the timeframe→interval map:

```python
_TIMEFRAME_TO_MEXC = {
    "1m": "Min1", "5m": "Min5", "15m": "Min15",
    "30m": "Min30", "1h": "Min60", "4h": "Hour4", "1d": "Day1",
}

_TIMEFRAME_TO_BYBIT = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15",
    "30m": "30", "1h": "60", "4h": "240", "1d": "D",
}
```

---

## 8. Environment Variables

Add Bybit credentials to `.env` (never commit real keys):

```bash
# .env  — add alongside existing MEXC vars

BYBIT_API_KEY=your_bybit_api_key_here
BYBIT_API_SECRET=your_bybit_api_secret_here

# Optional: Bybit testnet flag (set in config YAML as exchange.testnet: true)
```

Add a `LIVE_FARMER_ACK=I_UNDERSTAND` guard (already exists) — it applies regardless of exchange.

---

## 9. What Changes for Future Exchanges (e.g. Binance, OKX)

Follow the same pattern:

1. Create `src/exchange/{exchange}_client.py` implementing `ExchangeClientBase`
2. Normalise field names in its methods (`state: 3/4`, `positionType: 1/2`, `holdVol`, etc.)
3. Create `src/data/historical.py` normalisation function for that exchange's kline format
4. Add `provider: {exchange}` to the config YAML's `exchange` section
5. Add a branch in `_run()` inside `run_volume_farmer.py` to instantiate that client
6. Add env vars for API credentials

**Nothing else changes.** Strategy, signals, Telegram, risk management — all untouched.

---

## 10. Implementation Checklist for an AI Agent

Work through these in order. Each step is independently testable before proceeding.

- [ ] **Step 1** — Create `src/exchange/exchange_client_base.py` with the abstract base class from Section 2
- [ ] **Step 2** — Create `src/exchange/bybit_client.py` from Section 3 skeleton; fill in `_request` and all 7 methods
- [ ] **Step 3** — Smoke test: run `python -c "import asyncio; from exchange.bybit_client import BybitClient; ..."` to call `get_klines` with paper credentials against testnet — confirm OHLCV rows come back
- [ ] **Step 4** — Create `src/data/historical.py::_normalize_bybit_kline_payload` from Section 4; confirm DataFrame has `open_time`, `open`, `high`, `low`, `close`, `volume` columns
- [ ] **Step 5** — Change `LiveVolumeExecutor.client` type from `MEXCClient` to `ExchangeClientBase`; run `pytest tests/` — all existing tests must still pass
- [ ] **Step 6** — Create `config/config_volume_farmer_bybit.yaml` from Section 6; adjust fees for Bybit
- [ ] **Step 7** — Update `run_volume_farmer.py` to branch on `exchange.provider` per Section 7
- [ ] **Step 8** — Paper-trade test: run `python scripts/run_volume_farmer.py --config config/config_volume_farmer_bybit.yaml` (no `--live` flag) — confirm candles seed, session fires entry/exit events, Telegram messages arrive
- [ ] **Step 9** — Live testnet test: add `exchange.testnet: true` to config; run with `--live --live-dry-run` — confirm orders are submitted to Bybit testnet and normalised close events fire
- [ ] **Step 10** — Live production: remove testnet flag, set real credentials, run with `--live`

---

## 11. Critical Differences: MEXC vs Bybit Strategy Economics

| Metric | MEXC (current) | Bybit |
|---|---|---|
| Maker fee | 0.01% | 0.02% |
| Taker fee | 0.05% | 0.055% |
| Rebate | 70% of fees | None |
| Volume bonus | $80/million | None |
| Break-even WR (TP=8, SL=50) | ~89% raw, ~76% with rebate+bonus | ~89% raw, no offset |
| Live WR achieved | 76.7% | Unknown — backtest needed |
| Net at 76.7% WR | Profitable (bonus covers bleed) | **Loss-making** |

**Action required before going live on Bybit:** Run backtest with Bybit fee structure to find a TP level where raw P&L is break-even or better. Based on the math: TP needs to be at least **15–20bps** to achieve break-even WR around 76–78% on Bybit without any bonus income.

```bash
# Quick re-backtest command with Bybit fees:
python scripts/backtest_volume_farmer.py \
  --config config/config_volume_farmer_bybit.yaml \
  --tp 15 --sl 50
```
