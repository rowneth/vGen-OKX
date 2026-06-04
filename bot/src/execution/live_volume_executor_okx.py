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
from execution.maker_exit import (
    ExitResult,
    MakerExitConfig,
    cancel_attached_algos,
    close_position_maker,
)

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


def _lot_decimals(lot: float) -> int:
    """How many decimal places ``lot`` carries. e.g. 0.01 -> 2, 0.1 -> 1, 1 -> 0."""
    if lot <= 0:
        return 8
    # log10 may yield -1.9999… for 0.01 due to float repr; round before negating.
    return max(0, -int(round(math.log10(lot))))


def _fmt_size(x: float, lot: float) -> str:
    """Format a contract size as a string OKX will accept.

    OKX validates ``sz`` textually against ``lotSz``: ``2.5500000000000003``
    is rejected as not a clean multiple even though it numerically is. Format
    to the lot's decimal precision so the string matches the grid exactly.
    """
    return f"{x:.{_lot_decimals(lot)}f}"


@dataclass
class EntryRepegConfig:
    """Tuning for the entry post_only re-peg loop.

    Each "peg" places a post_only at the current touch, waits ``repeg_ms`` for
    fill, and on no-fill cancels and re-prices. ``max_repegs`` and
    ``max_entry_seconds`` bound the total chase. ``taker_fallback`` (off by
    default) sends a market order after exhausting the maker loop; leaving it
    off means an unfilled entry is simply abandoned for the next signal — the
    volume-preserving option since the next 5m bar still produces a signal.
    """

    enabled: bool = True
    repeg_ms: int = 8_000
    max_repegs: int = 6
    max_entry_seconds: float = 60.0
    poll_ms: int = 150
    taker_fallback: bool = False


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
    # Exit-execution metrics (populated when maker_exit / time_stop path runs).
    exit_path: str = "native_tp_sl"     # "native_tp_sl" | "maker_exit" | "taker_fallback"
    exit_repegs: int = 0
    exit_ttf_s: float = 0.0
    exit_maker_qty: float = 0.0
    exit_taker_qty: float = 0.0
    exit_fee_bps: float = 0.0
    exit_adverse_bps: float = 0.0
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
    # Feature 1 — time-stop (close non-winning trades at touch instead of letting
    # SL trigger taker close). Defaults are off to preserve current behavior.
    time_stop_enabled: bool = False
    max_hold_bars: int = 2
    bar_seconds: int = 300
    # Feature 2 — maker-only exit re-peg loop. Off by default; when on, the
    # time-stop path uses it instead of a market close.
    maker_exit_enabled: bool = False
    maker_exit_cfg: MakerExitConfig = field(default_factory=MakerExitConfig)
    # Entry re-peg loop (B). When enabled, replaces the single-shot post_only
    # entry with a re-peg loop that follows the touch. Solves the post_only
    # timeout-cancel problem in trending markets where the bar-close reference
    # price goes stale within seconds.
    entry_repeg_cfg: EntryRepegConfig = field(default_factory=EntryRepegConfig)
    # Fires once per resolved trade with the real OKX fill record. Caller can
    # use it to emit a Telegram message reflecting actual fills (paper exit
    # messages quote simulated TP/SL prices, which diverge from reality).
    on_real_fill: Optional[Callable[[LiveTradeOKX], Awaitable[None]]] = field(default=None)

    # Per-trade entry-repeg metrics (most-recent attempt only — for log line)
    _last_entry_repegs: int = field(default=0, init=False)

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

        # Pin the entry to the current top-of-book so post_only lands at the
        # front of the maker queue. The bar-close reference price is stale by
        # the time we submit (5–25s after the bar close); placing AT the close
        # leaves us deep in the book and post_only timeout-cancels with no fill.
        # We never cross — best_ask-tick for long, best_bid+tick for short are
        # both guaranteed post_only-valid. Falls back to bar close if ticker
        # fetch fails. TP/SL targets stay anchored to the session's reference
        # so paper/live ledgers don't diverge on those.
        ref_px = entry_req
        try:
            tk = await self.client.get_ticker(self.symbol)
            best_bid = float(tk.get("bidPx") or 0.0)
            best_ask = float(tk.get("askPx") or 0.0)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("ticker fetch failed, using bar close as entry px: %s", exc)
            best_bid = best_ask = 0.0
        if best_bid > 0 and best_ask > 0:
            if side == "long":
                # Top of bid book: 1 tick below best ask, but not above the signal
                ref_px = min(best_ask - self._tick_sz, entry_req)
            else:
                # Top of ask book: 1 tick above best bid, but not below the signal
                ref_px = max(best_bid + self._tick_sz, entry_req)
            LOGGER.info(
                "entry pin: side=%s bar_close=%.2f best_bid=%.2f best_ask=%.2f -> entry=%.2f (drift=%+.2f)",
                side, entry_req, best_bid, best_ask, ref_px, ref_px - entry_req,
            )
        entry_px = _round_to_tick(ref_px, self._tick_sz)
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

        sz_payload = _fmt_size(contracts, self._lot_sz)
        LOGGER.info(
            "LIVE entry: %s %s sz=%s ct (notional=$%.2f) px=%s  TP=%s (maker)  SL=%s (market)  cid=%s",
            side.upper(), to_okx_inst_id(self.symbol),
            sz_payload, notional, entry_px, tp_px_r, sl_px_r, cl_ord_id,
        )

        if self.dry_run:
            LOGGER.info("DRY-RUN: not submitting")
            return

        # Entry placement — re-peg loop (B) by default, single-shot legacy if
        # entry_repeg_cfg.enabled is False. The loop keeps the order at the
        # current touch as the book moves so we don't sit unfilled while the
        # market trends away from a stale bar-close price.
        if self.entry_repeg_cfg.enabled:
            ok = await self._place_entry_with_repeg(
                trade=trade, ok_side=ok_side, pos_side=pos_side,
                contracts=contracts, sz_payload=sz_payload,
                tp_px_r=tp_px_r, sl_px_r=sl_px_r,
            )
            if not ok:
                # entry abandoned (re-pegs exhausted) — clean up and skip
                self._log_trade(trade)
                self._open_trade = None
                return
        else:
            # Legacy single-shot post_only with timeout watcher.
            try:
                resp = await self.client.place_order(
                    symbol=self.symbol,
                    side=ok_side, pos_side=pos_side,
                    td_mode=self.mgn_mode, ord_type="post_only",
                    sz=sz_payload, px=str(entry_px),
                    client_oid=trade.cl_ord_id,
                    tp_trigger_px=str(tp_px_r), tp_ord_px=str(tp_px_r),
                    sl_trigger_px=str(sl_px_r), sl_ord_px="-1",
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
            LOGGER.info("submitted ordId=%s (cid=%s)", trade.ord_id, trade.cl_ord_id)
            filled = await self._wait_for_entry_fill_or_timeout(trade)
            if not filled:
                self._log_trade(trade)
                self._open_trade = None
                return

        # Position is open — start the position + time-stop watch.
        await self._watch_open_position(trade)

    async def _place_entry_with_repeg(
        self, *, trade: LiveTradeOKX, ok_side: str, pos_side: Optional[str],
        contracts: float, sz_payload: str, tp_px_r: float, sl_px_r: float,
    ) -> bool:
        """Run the entry-side maker re-peg loop. Returns True on a fill.

        Each iteration: fetch current touch, place post_only at it with the
        attached TP/SL algo bundle, wait up to ``repeg_ms`` for fill, on
        timeout cancel + re-pin. Bound by ``max_repegs`` and
        ``max_entry_seconds``. Optional ``taker_fallback`` at the end (off by
        default). If everything is exhausted without a fill, returns False and
        the caller abandons the entry — next signal becomes the next chance.
        """
        cfg = self.entry_repeg_cfg
        side = trade.side
        deadline_wall = time.time() + cfg.max_entry_seconds
        repegs = 0

        while time.time() < deadline_wall and repegs < cfg.max_repegs:
            # 1. Fetch current touch
            try:
                tk = await self.client.get_ticker(self.symbol)
                best_bid = float(tk.get("bidPx") or 0.0)
                best_ask = float(tk.get("askPx") or 0.0)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("entry_repeg: ticker failed: %s", exc)
                await asyncio.sleep(cfg.poll_ms / 1000.0)
                continue
            if best_bid <= 0 or best_ask <= 0:
                await asyncio.sleep(cfg.poll_ms / 1000.0)
                continue
            # 2. Touch-pinned post_only price
            if side == "long":
                px = best_ask - self._tick_sz
            else:
                px = best_bid + self._tick_sz
            entry_px = _round_to_tick(px, self._tick_sz)

            # 3. Fresh client order id per peg
            cl_ord = "vfox" + uuid.uuid4().hex[:14]
            trade.cl_ord_id = cl_ord
            trade.entry_px_req = entry_px

            try:
                resp = await self.client.place_order(
                    symbol=self.symbol,
                    side=ok_side, pos_side=pos_side,
                    td_mode=self.mgn_mode, ord_type="post_only",
                    sz=sz_payload, px=str(entry_px),
                    client_oid=cl_ord,
                    tp_trigger_px=str(tp_px_r), tp_ord_px=str(tp_px_r),
                    sl_trigger_px=str(sl_px_r), sl_ord_px="-1",
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("entry_repeg: place_order error: %s", exc)
                repegs += 1
                await asyncio.sleep(cfg.poll_ms / 1000.0)
                continue

            data = (resp.get("data") or [{}])[0]
            scode = str(data.get("sCode", "0"))
            if scode != "0":
                # 51280 = would-cross. Touch moved during placement; re-peg.
                LOGGER.info(
                    "entry_repeg #%d sCode=%s sMsg=%r — re-peg",
                    repegs + 1, scode, data.get("sMsg"),
                )
                repegs += 1
                await asyncio.sleep(cfg.poll_ms / 1000.0)
                continue

            trade.ord_id = data.get("ordId")
            LOGGER.info(
                "entry_repeg #%d submitted ordId=%s cid=%s px=%.2f (bid=%.2f ask=%.2f)",
                repegs + 1, trade.ord_id, cl_ord, entry_px, best_bid, best_ask,
            )

            # 4. Wait up to repeg_ms for fill or terminal state
            wait_deadline = time.time() + cfg.repeg_ms / 1000.0
            filled_this_peg = False
            while time.time() < wait_deadline and time.time() < deadline_wall:
                await asyncio.sleep(cfg.poll_ms / 1000.0)
                try:
                    r = await self.client.get_order(
                        self.symbol, ord_id=trade.ord_id, client_oid=cl_ord,
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("entry_repeg: get_order failed: %s", exc)
                    continue
                od = (r.get("data") or [{}])[0]
                state = str(od.get("state", ""))
                if state == "filled":
                    trade.filled = True
                    trade.fill_px = float(od.get("avgPx") or entry_px)
                    trade.fill_ts = int(od.get("fillTime") or int(time.time() * 1000))
                    trade.real_open_fee = abs(float(od.get("fee") or 0.0))
                    self._last_entry_repegs = repegs
                    LOGGER.info(
                        "FILL  cid=%s  px=%.2f  fee=%.6f  repegs=%d",
                        trade.cl_ord_id, trade.fill_px, trade.real_open_fee, repegs,
                    )
                    filled_this_peg = True
                    break
                if state in {"canceled", "mmp_canceled"}:
                    break

            if filled_this_peg:
                return True

            # 5. Window elapsed without fill — cancel and re-peg
            try:
                await self.client.cancel_order(self.symbol, ord_id=trade.ord_id)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("entry_repeg: cancel failed: %s", exc)
            # Race window: a fill might have landed between our last poll and
            # the cancel. Check once more before counting this as a missed peg.
            try:
                r = await self.client.get_order(
                    self.symbol, ord_id=trade.ord_id, client_oid=cl_ord,
                )
                od = (r.get("data") or [{}])[0]
                if str(od.get("state", "")) == "filled":
                    trade.filled = True
                    trade.fill_px = float(od.get("avgPx") or entry_px)
                    trade.fill_ts = int(od.get("fillTime") or int(time.time() * 1000))
                    trade.real_open_fee = abs(float(od.get("fee") or 0.0))
                    self._last_entry_repegs = repegs
                    LOGGER.info(
                        "FILL  cid=%s  px=%.2f  fee=%.6f  repegs=%d (race-fill)",
                        trade.cl_ord_id, trade.fill_px, trade.real_open_fee, repegs,
                    )
                    return True
            except Exception:  # noqa: BLE001
                pass

            repegs += 1

        # All pegs exhausted. Optional taker fallback (off by default — we
        # prefer to skip a signal rather than pay taker on entry).
        self._last_entry_repegs = repegs
        if not cfg.taker_fallback:
            LOGGER.warning(
                "entry abandoned after %d re-pegs, %.1fs elapsed (no taker fallback)",
                repegs, time.time() - (deadline_wall - cfg.max_entry_seconds),
            )
            trade.canceled = True
            trade.cancel_reason = f"repeg_exhausted_after_{repegs}_attempts"
            return False

        LOGGER.warning(
            "entry: re-pegs exhausted (%d) — TAKER fallback engaged", repegs,
        )
        cl_ord = "vfoxk" + uuid.uuid4().hex[:13]
        trade.cl_ord_id = cl_ord
        # OKX V5 quirk: market orders in net_mode want posSide explicitly set
        # to "net" (post_only orders tolerate it being omitted). Without this,
        # OKX returns sCode=51000 "Parameter posSide error" and the fallback
        # never lands.
        pos_side_for_market = pos_side if pos_side is not None else (
            "net" if self.pos_mode == "net" else None
        )
        try:
            resp = await self.client.place_order(
                symbol=self.symbol,
                side=ok_side, pos_side=pos_side_for_market,
                td_mode=self.mgn_mode, ord_type="market",
                sz=sz_payload, client_oid=cl_ord,
                tp_trigger_px=str(tp_px_r), tp_ord_px=str(tp_px_r),
                sl_trigger_px=str(sl_px_r), sl_ord_px="-1",
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("entry taker fallback place_order failed: %s", exc)
            trade.canceled = True
            trade.cancel_reason = f"taker_submit_error: {exc}"
            return False
        data = (resp.get("data") or [{}])[0]
        if str(data.get("sCode", "0")) != "0":
            LOGGER.error(
                "entry taker fallback rejected sCode=%s sMsg=%r",
                data.get("sCode"), data.get("sMsg"),
            )
            trade.canceled = True
            trade.cancel_reason = f"taker_sCode={data.get('sCode')}"
            return False
        trade.ord_id = data.get("ordId")
        for _ in range(40):
            await asyncio.sleep(0.1)
            try:
                r = await self.client.get_order(
                    self.symbol, ord_id=trade.ord_id, client_oid=cl_ord,
                )
            except Exception:  # noqa: BLE001
                continue
            od = (r.get("data") or [{}])[0]
            if str(od.get("state", "")) == "filled":
                trade.filled = True
                trade.fill_px = float(od.get("avgPx") or 0.0)
                trade.fill_ts = int(od.get("fillTime") or int(time.time() * 1000))
                trade.real_open_fee = abs(float(od.get("fee") or 0.0))
                trade.extras["entry_taker_fallback"] = True
                LOGGER.info(
                    "FILL (taker fallback)  cid=%s  px=%.2f  fee=%.6f  repegs=%d",
                    trade.cl_ord_id, trade.fill_px, trade.real_open_fee, repegs,
                )
                return True
        LOGGER.error("entry taker fallback no fill — abandoning")
        trade.canceled = True
        trade.cancel_reason = "taker_no_fill"
        return False

    async def _wait_for_entry_fill_or_timeout(self, trade: LiveTradeOKX) -> bool:
        """Legacy single-shot watcher (used only when entry_repeg is disabled)."""
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
                return True
            if state in {"canceled", "mmp_canceled"}:
                trade.canceled = True
                trade.cancel_reason = f"state={state}"
                return False
        if not trade.filled and not trade.canceled:
            try:
                await self.client.cancel_order(self.symbol, ord_id=trade.ord_id)
                LOGGER.info("cancelled stale post-only cid=%s", trade.cl_ord_id)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("cancel after timeout failed: %s", exc)
            trade.canceled = True
            trade.cancel_reason = "post_only_timeout"
        return trade.filled

    async def _watch_open_position(self, trade: LiveTradeOKX) -> None:
        """Race native TP/SL watcher vs time-stop watcher until position closes."""
        tasks: List[asyncio.Task] = [
            asyncio.create_task(self._watch_position(trade), name="watch_position"),
        ]
        if self.time_stop_enabled:
            tasks.append(asyncio.create_task(
                self._watch_time_stop(trade), name="watch_time_stop",
            ))
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        for t in pending:
            try:
                await t
            except asyncio.CancelledError:
                pass
        self._log_trade(trade)
        if self.on_real_fill is not None and trade.filled and trade.closed:
            try:
                await self.on_real_fill(trade)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("on_real_fill callback failed: %s", exc)
        self._open_trade = None

    async def _watch_time_stop(self, trade: LiveTradeOKX) -> None:
        """Fire a maker-first close once ``max_hold_bars`` have elapsed since fill.

        Sleeps until ``fill_ts + max_hold_bars * bar_seconds`` (in UTC monotonic
        terms — we re-anchor off the wall clock since fill_ts is exchange ms).
        If the position is still open at deadline, cancel any resting algo
        orders and run :func:`close_position_maker`.
        """
        if not self.time_stop_enabled:
            return
        # fill_ts is exchange-side milliseconds; convert to local wallclock.
        fill_wall = trade.fill_ts / 1000.0 if trade.fill_ts else time.time()
        deadline_wall = fill_wall + max(1, self.max_hold_bars) * max(1, self.bar_seconds)
        sleep_for = deadline_wall - time.time()
        if sleep_for > 0:
            try:
                await asyncio.sleep(sleep_for)
            except asyncio.CancelledError:
                raise

        # Verify the position is actually still open before we touch the book.
        try:
            r = await self.client.get_positions(self.symbol)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("time_stop: positions fetch failed: %s", exc)
            return
        positions = r.get("data") or []
        qty = 0.0
        for p in positions:
            try:
                qty = float(p.get("pos") or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
            if qty != 0.0:
                break
        if qty == 0.0:
            LOGGER.info("time_stop: position already flat, no action (cid=%s)",
                        trade.cl_ord_id)
            return

        LOGGER.info(
            "TIME-STOP firing: cid=%s held >= %d bars (%ds each); closing via maker exit",
            trade.cl_ord_id, self.max_hold_bars, self.bar_seconds,
        )
        await cancel_attached_algos(self.client, self.symbol)
        await self._maker_close(trade, reason="time_stop", abs_qty=abs(qty))

    async def _maker_close(
        self, trade: LiveTradeOKX, *, reason: str, abs_qty: float,
    ) -> None:
        """Run the maker-exit re-peg loop and reconcile the trade record."""
        if not self.maker_exit_enabled:
            # Fallback: native market close (taker). Still records the path.
            LOGGER.info("maker_exit disabled — falling back to market close cid=%s",
                        trade.cl_ord_id)
            try:
                pos_side_param = (
                    trade.side if self.pos_mode == "hedge" else None
                )
                await self.client.close_position_market(
                    self.symbol, mgn_mode=self.mgn_mode, pos_side=pos_side_param,
                )
                trade.exit_path = "taker_fallback"
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("close_position_market failed: %s", exc)
            await self._resolve_close(trade)
            if not trade.close_reason:
                trade.close_reason = reason
            return

        intended_px = trade.tp_px if reason == "tp" else trade.fill_px
        pos_side_param = trade.side if self.pos_mode == "hedge" else None
        result: ExitResult = await close_position_maker(
            self.client,
            symbol=self.symbol,
            position_side=trade.side,
            sz_contracts=abs_qty,
            tick_sz=self._tick_sz,
            lot_sz=self._lot_sz,
            intended_exit_px=intended_px,
            pos_side_param=pos_side_param,
            td_mode=self.mgn_mode,
            cfg=self.maker_exit_cfg,
        )
        trade.exit_path = "maker_exit" if not result.used_taker else "maker_then_taker"
        trade.exit_repegs = result.repegs
        trade.exit_ttf_s = result.ttf_seconds
        trade.exit_maker_qty = result.maker_qty
        trade.exit_taker_qty = result.taker_qty
        trade.exit_fee_bps = result.realized_fee_bps
        trade.exit_adverse_bps = result.adverse_bps

        if result.filled_qty > 0:
            trade.close_px = result.avg_fill_px
            trade.close_ts = int(time.time() * 1000)
            trade.real_close_fee = sum(f.fee for f in result.fills)
            if trade.side == "long":
                trade.real_gross_pnl = (
                    trade.close_px - trade.fill_px
                ) * trade.sz_contracts * self._ct_val
            else:
                trade.real_gross_pnl = (
                    trade.fill_px - trade.close_px
                ) * trade.sz_contracts * self._ct_val
            trade.closed = True
            trade.close_reason = reason
            LOGGER.info(
                "CLOSE cid=%s reason=%s exit_path=%s avg_px=%.2f repegs=%d ttf=%.2fs "
                "maker=%s taker=%s fee_bps=%.3f adverse=%.1fbps fallback=%r",
                trade.cl_ord_id, reason, trade.exit_path, trade.close_px,
                result.repegs, result.ttf_seconds, result.maker_qty,
                result.taker_qty, result.realized_fee_bps, result.adverse_bps,
                result.fallback_reason,
            )
        else:
            LOGGER.error(
                "maker_exit: nothing filled (cid=%s err=%s) — leaving position open",
                trade.cl_ord_id, result.error,
            )

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
            "exit_path": trade.exit_path,
            "exit_repegs": trade.exit_repegs,
            "exit_ttf_s": round(trade.exit_ttf_s, 4),
            "exit_maker_qty": trade.exit_maker_qty,
            "exit_taker_qty": trade.exit_taker_qty,
            "exit_fee_bps": round(trade.exit_fee_bps, 4),
            "exit_adverse_bps": round(trade.exit_adverse_bps, 4),
        }
        try:
            live_dir = self.log_dir / "live_trades"
            live_dir.mkdir(parents=True, exist_ok=True)
            day = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
            with (live_dir / f"{day}.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("live trade log write failed: %s", exc)
