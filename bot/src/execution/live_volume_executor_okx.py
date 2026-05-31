"""Bridge between VolumeFarmerSession and real OKX perpetual-swap orders.

Strategy: when the session emits an ``entry`` event, place a post-only limit
order at the session's chosen entry price with an attached TP/SL algo bundle.
The TP is a LIMIT order at the TP price (``tpOrdPx`` = price) → fills as maker
when triggered. The SL is a market order (``slOrdPx`` = ``-1``) because price
must move *through* the trigger to flag it as an SL hit.

Lifecycle:
    1. Session emits entry → executor converts USD notional to contracts using
       the cached ``ctVal`` and submits a post-only order.
    2. Executor polls the order state. If ``live`` after ``post_only_timeout``,
       it's cancelled (post-only would have crossed → market moved).
    3. Once ``filled``, the attached algo orders are armed on OKX. Executor
       polls ``/account/positions`` until the position closes.
    4. On close, executor reads the last reduceOnly fill from order history
       and logs the real-money trade to a separate jsonl (so paper and live
       can be compared apples-to-apples).

The session itself continues to simulate paper fills from candles. The two
streams (paper + live) diverge by slippage and post-only miss rate — that's
the entire point: comparing them tells you the cost of real execution.

Operator MUST set ``LIVE_OKX_ACK=I_UNDERSTAND`` in env for real-money mode.
Demo mode (``simulated=True`` on the client) does not require the ack.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import pathlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from exchange.okx_client import OKXClient, to_okx_inst_id

LOGGER = logging.getLogger(__name__)


def _round_to_tick(x: float, tick: float) -> float:
    if tick <= 0:
        return x
    return round(round(x / tick) * tick, 10)


def _round_size(x: float, lot: float) -> float:
    if lot <= 0:
        return x
    # Round DOWN to lot to avoid over-allocating margin.
    return math.floor(x / lot) * lot


@dataclass
class LiveTradeOKX:
    """Bookkeeping for one live OKX trade lifecycle."""

    cl_ord_id: str
    side: str                       # "long" | "short"
    entry_px_req: float
    notional_usd: float
    sz_contracts: float
    tp_px: float
    sl_px: float
    tp_bps: float
    sl_bps: float
    placed_at: float
    ord_id: Optional[str] = None
    filled: bool = False
    fill_px: float = 0.0
    fill_ts: int = 0
    canceled: bool = False
    cancel_reason: str = ""
    closed: bool = False
    close_px: float = 0.0
    close_reason: str = ""           # "tp" | "sl" | "manual"
    close_ts: int = 0
    real_gross_pnl: float = 0.0
    real_open_fee: float = 0.0
    real_close_fee: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveVolumeExecutorOKX:
    """Translate FarmerEvent lifecycle into OKX orders."""

    client: OKXClient
    symbol: str = "BTC_USDT"
    pos_mode: str = "net"            # "net" (one-way) or "hedge"
    mgn_mode: str = "isolated"
    post_only_timeout_s: float = 30.0
    order_poll_s: float = 1.5
    position_poll_s: float = 3.0
    position_watch_max_s: float = 6 * 3600.0   # 6 hours max
    max_live_trades: int = 1_000_000
    dry_run: bool = False
    log_dir: pathlib.Path = field(default_factory=lambda: pathlib.Path("data/logs"))

    # Cached instrument metadata
    _ct_val: float = field(default=0.01, init=False)
    _ct_val_ccy: str = field(default="BTC", init=False)
    _tick_sz: float = field(default=0.1, init=False)
    _lot_sz: float = field(default=0.01, init=False)
    _min_sz: float = field(default=0.01, init=False)

    # Mutable state
    _open_trade: Optional[LiveTradeOKX] = field(default=None, init=False)
    _trade_count: int = field(default=0, init=False)
    _watcher_task: Optional[asyncio.Task] = field(default=None, init=False)
    _initialized: bool = field(default=False, init=False)

    async def initialize(self, leverage_cap: int = 50) -> None:
        """One-time setup: load instrument spec, set leverage."""
        if self._initialized:
            return
        inst = await self.client.get_instrument(self.symbol)
        self._ct_val = float(inst["ctVal"])
        self._ct_val_ccy = str(inst["ctValCcy"])
        self._tick_sz = float(inst["tickSz"])
        self._lot_sz = float(inst["lotSz"])
        self._min_sz = float(inst["minSz"])
        LOGGER.info(
            "OKX instrument %s: ctVal=%s %s tickSz=%s lotSz=%s minSz=%s",
            to_okx_inst_id(self.symbol), self._ct_val, self._ct_val_ccy,
            self._tick_sz, self._lot_sz, self._min_sz,
        )
        if not self.dry_run:
            try:
                resp = await self.client.set_leverage(
                    self.symbol, leverage_cap, mgn_mode=self.mgn_mode,
                )
                LOGGER.info("set leverage cap %sx (%s): code=%s msg=%r",
                            leverage_cap, self.mgn_mode, resp.get("code"), resp.get("msg"))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("set_leverage failed (continuing): %s", exc)
        self._initialized = True

    # ------------------------------------------------------------------
    # Hook called by the session's event_callback.
    # ------------------------------------------------------------------
    def consume_session_event(self, evt) -> None:  # type: FarmerEvent
        """Sync hook — schedules async work on the running loop."""
        if evt.kind != "entry":
            return
        if self._trade_count >= self.max_live_trades:
            LOGGER.warning("max_live_trades=%d reached; ignoring entry",
                           self.max_live_trades)
            return
        if self._open_trade is not None and not self._open_trade.closed:
            LOGGER.warning("ignoring entry — prior live trade still open: %s",
                           self._open_trade.cl_ord_id)
            return
        asyncio.create_task(self._handle_entry(evt.payload))

    # ------------------------------------------------------------------
    async def _handle_entry(self, payload: Dict[str, Any]) -> None:
        """Place a real OKX order matching the session's entry decision."""
        side = str(payload["side"]).lower()                 # long | short
        entry_req = float(payload["price"])
        notional = float(payload["notional"])
        tp_px = float(payload["tp"])
        sl_px = float(payload["sl"])
        tp_bps = float(payload["tp_bps"])
        sl_bps = float(payload["sl_bps"])

        # Notional USD -> BTC -> contracts (BTC-USDT-SWAP: 1 ct = 0.01 BTC)
        # btc_qty = notional / entry_req
        # contracts = btc_qty / ctVal
        btc_qty = notional / entry_req
        contracts = btc_qty / self._ct_val
        contracts = _round_size(contracts, self._lot_sz)
        if contracts < self._min_sz:
            LOGGER.warning("sized %.4f contracts < minSz %.4f; skipping",
                           contracts, self._min_sz)
            return

        entry_px = _round_to_tick(entry_req, self._tick_sz)
        tp_px_r = _round_to_tick(tp_px, self._tick_sz)
        sl_px_r = _round_to_tick(sl_px, self._tick_sz)

        cl_ord_id = "vfox" + uuid.uuid4().hex[:14]
        ok_side = "buy" if side == "long" else "sell"
        pos_side = "long" if side == "long" else "short" if self.pos_mode == "hedge" else None

        trade = LiveTradeOKX(
            cl_ord_id=cl_ord_id, side=side,
            entry_px_req=entry_px, notional_usd=notional,
            sz_contracts=contracts, tp_px=tp_px_r, sl_px=sl_px_r,
            tp_bps=tp_bps, sl_bps=sl_bps, placed_at=time.time(),
        )
        self._open_trade = trade
        self._trade_count += 1

        LOGGER.info(
            "LIVE entry: %s %s sz=%s ct (notional=$%.2f) px=%s  TP=%s (maker)  SL=%s (market)  cid=%s",
            side.upper(), to_okx_inst_id(self.symbol),
            contracts, notional, entry_px, tp_px_r, sl_px_r, cl_ord_id,
        )

        if self.dry_run:
            LOGGER.info("DRY-RUN: not submitting")
            return

        try:
            resp = await self.client.place_order(
                symbol=self.symbol,
                side=ok_side, pos_side=pos_side,
                td_mode=self.mgn_mode, ord_type="post_only",
                sz=str(contracts), px=str(entry_px),
                client_oid=cl_ord_id,
                tp_trigger_px=str(tp_px_r), tp_ord_px=str(tp_px_r),       # MAKER TP
                sl_trigger_px=str(sl_px_r), sl_ord_px="-1",                # MARKET SL
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("place_order failed: %s", exc)
            trade.canceled = True
            trade.cancel_reason = f"submit_error: {exc}"
            self._log_trade(trade)
            self._open_trade = None
            return

        data = (resp.get("data") or [{}])[0]
        if str(data.get("sCode", "0")) != "0":
            LOGGER.error("OKX rejected order: sCode=%s sMsg=%r",
                         data.get("sCode"), data.get("sMsg"))
            trade.canceled = True
            trade.cancel_reason = f"sCode={data.get('sCode')}:{data.get('sMsg')}"
            self._log_trade(trade)
            self._open_trade = None
            return

        trade.ord_id = data.get("ordId")
        LOGGER.info("submitted ordId=%s (cid=%s)", trade.ord_id, cl_ord_id)

        await self._watch_entry_then_position(trade)

    async def _watch_entry_then_position(self, trade: LiveTradeOKX) -> None:
        """Poll until entry fills or times out, then watch position until close."""
        deadline = trade.placed_at + self.post_only_timeout_s
        while time.time() < deadline and not trade.filled and not trade.canceled:
            await asyncio.sleep(self.order_poll_s)
            try:
                r = await self.client.get_order(
                    self.symbol, ord_id=trade.ord_id, client_oid=trade.cl_ord_id,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("get_order failed: %s", exc)
                continue
            data = (r.get("data") or [{}])[0]
            state = str(data.get("state", ""))
            if state == "filled":
                trade.filled = True
                trade.fill_px = float(data.get("avgPx") or trade.entry_px_req)
                trade.fill_ts = int(data.get("fillTime") or int(time.time() * 1000))
                trade.real_open_fee = abs(float(data.get("fee") or 0.0))
                LOGGER.info("FILL  cid=%s  px=%.2f  fee=%.6f",
                            trade.cl_ord_id, trade.fill_px, trade.real_open_fee)
                break
            if state in {"canceled", "mmp_canceled"}:
                trade.canceled = True
                trade.cancel_reason = f"state={state}"
                LOGGER.info("CANCEL cid=%s state=%s", trade.cl_ord_id, state)
                break

        if not trade.filled and not trade.canceled:
            # Timeout — cancel and move on
            try:
                await self.client.cancel_order(self.symbol, ord_id=trade.ord_id)
                LOGGER.info("cancelled stale post-only cid=%s", trade.cl_ord_id)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("cancel after timeout failed: %s", exc)
            trade.canceled = True
            trade.cancel_reason = "post_only_timeout"

        if trade.canceled:
            self._log_trade(trade)
            self._open_trade = None
            return

        # Position is now open; wait for it to close via attached TP/SL.
        await self._watch_position(trade)
        self._log_trade(trade)
        self._open_trade = None

    async def _watch_position(self, trade: LiveTradeOKX) -> None:
        start = time.time()
        last_seen_open = False
        while time.time() - start < self.position_watch_max_s:
            await asyncio.sleep(self.position_poll_s)
            try:
                r = await self.client.get_positions(self.symbol)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("get_positions failed: %s", exc)
                continue
            positions = r.get("data") or []
            qty = 0.0
            for p in positions:
                pos_str = p.get("pos") or "0"
                try:
                    qty = float(pos_str)
                except ValueError:
                    qty = 0.0
                if qty != 0.0:
                    break
            if qty != 0.0:
                last_seen_open = True
                continue
            if last_seen_open:
                # Was open, now flat → position closed.
                await self._resolve_close(trade)
                return
        LOGGER.warning("position watch timed out for cid=%s", trade.cl_ord_id)
        trade.close_reason = "watch_timeout"

    async def _resolve_close(self, trade: LiveTradeOKX) -> None:
        """Look up the last reduceOnly fill to identify exit price and reason."""
        try:
            # The attached algo's fill appears in order history as a reduceOnly
            # order. Pull the most recent one matching our side.
            r = await self.client._request(  # noqa: SLF001 — fallback to raw
                "GET", "/api/v5/trade/orders-history",
                params={"instType": "SWAP",
                        "instId": to_okx_inst_id(self.symbol),
                        "limit": "10"},
                auth=True,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("orders-history fetch failed: %s", exc)
            trade.closed = True
            trade.close_reason = "history_fetch_failed"
            return

        for o in r.get("data") or []:
            if str(o.get("reduceOnly", "false")).lower() != "true":
                continue
            if str(o.get("state", "")) != "filled":
                continue
            # Match: opposite side, after our fill_ts
            exp_close_side = "sell" if trade.side == "long" else "buy"
            if str(o.get("side", "")) != exp_close_side:
                continue
            close_ts = int(o.get("fillTime") or 0)
            if close_ts < trade.fill_ts:
                continue
            trade.close_px = float(o.get("avgPx") or 0.0)
            trade.close_ts = close_ts
            trade.real_close_fee = abs(float(o.get("fee") or 0.0))
            # Identify TP vs SL by proximity (tolerant of slippage)
            d_tp = abs(trade.close_px - trade.tp_px)
            d_sl = abs(trade.close_px - trade.sl_px)
            trade.close_reason = "tp" if d_tp < d_sl else "sl"
            break

        if trade.close_px == 0.0:
            LOGGER.warning("could not match close fill for cid=%s", trade.cl_ord_id)
            trade.closed = True
            trade.close_reason = "unmatched_close"
            return

        # Realized gross PnL
        if trade.side == "long":
            trade.real_gross_pnl = (trade.close_px - trade.fill_px) * trade.sz_contracts * self._ct_val
        else:
            trade.real_gross_pnl = (trade.fill_px - trade.close_px) * trade.sz_contracts * self._ct_val
        trade.closed = True
        LOGGER.info(
            "CLOSE cid=%s reason=%s  fill_px=%.2f -> close_px=%.2f  gross=%+.4f  fees=%.6f+%.6f",
            trade.cl_ord_id, trade.close_reason, trade.fill_px, trade.close_px,
            trade.real_gross_pnl, trade.real_open_fee, trade.real_close_fee,
        )

    def _log_trade(self, trade: LiveTradeOKX) -> None:
        rec = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "symbol": to_okx_inst_id(self.symbol),
            "cl_ord_id": trade.cl_ord_id,
            "ord_id": trade.ord_id,
            "side": trade.side.upper(),
            "sz_contracts": trade.sz_contracts,
            "notional_usd": trade.notional_usd,
            "entry_px_req": trade.entry_px_req,
            "filled": trade.filled,
            "fill_px": trade.fill_px if trade.filled else None,
            "canceled": trade.canceled,
            "cancel_reason": trade.cancel_reason or None,
            "closed": trade.closed,
            "close_reason": trade.close_reason or None,
            "close_px": trade.close_px if trade.closed else None,
            "tp_px": trade.tp_px, "sl_px": trade.sl_px,
            "tp_bps": trade.tp_bps, "sl_bps": trade.sl_bps,
            "real_gross_pnl": trade.real_gross_pnl,
            "real_open_fee": trade.real_open_fee,
            "real_close_fee": trade.real_close_fee,
            "fill_ts": trade.fill_ts, "close_ts": trade.close_ts,
        }
        try:
            live_dir = self.log_dir / "live_trades"
            live_dir.mkdir(parents=True, exist_ok=True)
            day = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
            with (live_dir / f"{day}.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("live trade log write failed: %s", exc)
