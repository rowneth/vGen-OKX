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
import random
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
    # Abandon the chase when the touch has drifted more than this many bps
    # from the signal price (0 = no limit). The TP/SL bracket stays anchored
    # to the SIGNAL, so an uncapped chase can fill with the TP nearly through
    # price (or the SL effectively wider than configured) — bounding the chase
    # bounds that bracket skew.
    max_chase_bps: float = 0.0
    # Re-price the resting order via amend-order (1 REST call, no naked
    # window) instead of cancel+replace (2 calls + a gap with nothing resting).
    # Falls back to cancel+replace automatically when an amend is rejected.
    use_amend: bool = True


@dataclass
class DemoRealismConfig:
    """DEMO-ONLY execution-friction overlay. Never constructed under --live, so
    these frictions can physically never touch real-money execution."""

    enabled: bool = True
    seed: Optional[int] = None
    entry_fill_prob: float = 0.85
    entry_slip_bps: float = 0.4
    entry_taker_fee_mult: float = 2.5
    sl_slip_bps: float = 3.0
    taker_exit_fee_mult: float = 2.5
    tp_fill_prob: float = 0.90
    tp_miss_giveback_bps: float = 6.0


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
    # When > 0, size each real order off the LIVE OKX wallet balance
    # (notional = balance × margin_frac × leverage) instead of the simulation's
    # seeded/drifting equity — so position size always tracks the real account
    # and never a hardcoded capital_usd. Leverage is equity-independent so the
    # SIM's leverage is reused. 0 disables (falls back to the SIM's notional).
    margin_frac: float = 0.0
    # When > 0, CAP the sizing base at this many USDT: the order is sized off
    # min(real_wallet, working_capital_usd). Lets a small "working" account run on
    # a larger (e.g. demo) wallet — every trade risks off this fixed capital, not
    # the full balance. 0 = no cap (use the full wallet).
    working_capital_usd: float = 0.0
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
    # Feature 3 — RESTING maker TP. The attached trigger-TP fires a limit only
    # when price touches the trigger; on a gap that limit is already marketable
    # and fills TAKER (live: ~47% of TP wins leaked taker). When enabled, the
    # entry attaches ONLY the market SL, and a reduce-only POST_ONLY limit is
    # rested at tp_px the moment the entry fills — it sits in the book and fills
    # as MAKER by construction (post_only is rejected if it would cross).
    # Backtested 2026-06-14: TP-taker 47%->0%, fill-rate -2-3%, ~$7-18/1M saved.
    resting_tp_enabled: bool = False
    # Entry re-peg loop (B). When enabled, replaces the single-shot post_only
    # entry with a re-peg loop that follows the touch. Solves the post_only
    # timeout-cancel problem in trending markets where the bar-close reference
    # price goes stale within seconds.
    entry_repeg_cfg: EntryRepegConfig = field(default_factory=EntryRepegConfig)
    # DEMO-ONLY realism overlay. None under --live (the runner only constructs it
    # when the OKX client is in simulated mode), so frictions can't reach real money.
    demo_realism: Optional[DemoRealismConfig] = None
    # ------------------------------------------------------------------
    # OKX-confirmed lifecycle callbacks. Each fires ONLY at a real exchange
    # confirmation point so Telegram never announces a trade the exchange did
    # not actually make. This is the spine of the "real OKX panel" messaging:
    #   on_entry_filled    -> a position is CONFIRMED open (get_order filled)
    #   on_entry_abandoned -> the entry never filled (repeg-exhausted/reject)
    #   on_close_confirmed -> the position is CONFIRMED flat AND priced
    #   on_close_unverified-> closed-but-unpriced, or close failed (still open)
    # ------------------------------------------------------------------
    on_entry_filled: Optional[Callable[[LiveTradeOKX], Awaitable[None]]] = field(default=None)
    on_entry_abandoned: Optional[Callable[[LiveTradeOKX], Awaitable[None]]] = field(default=None)
    on_close_confirmed: Optional[Callable[[LiveTradeOKX], Awaitable[None]]] = field(default=None)
    on_close_unverified: Optional[Callable[[LiveTradeOKX], Awaitable[None]]] = field(default=None)
    # Back-compat alias for the close card. When on_close_confirmed is unset the
    # finalizer falls back to on_real_fill so existing wiring keeps working.
    on_real_fill: Optional[Callable[[LiveTradeOKX], Awaitable[None]]] = field(default=None)
    # Real-money safety bounds for the confirmation machinery:
    #   callback_timeout_s — a hung Telegram/account fetch can never wedge the
    #     close watcher (which must reset _open_trade to unblock new entries).
    #   fast_poll_window_s — poll faster right after a fill so a fast TP that
    #     closes within one normal poll is still seen (no missed close card).
    #   flat_confirm_polls — require N consecutive flat polls before treating a
    #     never-seen-open position as closed (guards positions-lag after entry).
    #   flat_giveup_polls  — after this many confirmed-flat-but-unpriceable
    #     polls, finalize as unverified instead of hanging the gate.
    callback_timeout_s: float = 15.0
    fast_poll_window_s: float = 8.0
    flat_confirm_polls: int = 2
    flat_giveup_polls: int = 12
    # Close-confirmation: how long to wait for /positions to read flat after a
    # close fills (it lags the fill by a second or two — polling until flat,
    # rather than a couple of quick checks, stops that lag from falsely
    # tripping the "still open" hold). And how often the recovery watcher
    # re-checks a held position so the entry gate is NEVER wedged permanently.
    flat_timeout_s: float = 12.0
    recover_poll_s: float = 10.0
    # Backoff base for the last-resort retried market close (_force_market_close);
    # attempts sleep base, 2×base, 4×base … capped at 4s. Small in tests.
    force_close_base_delay_s: float = 0.5
    # GLOBAL single-position gate across instruments: when the runner trades
    # multiple symbols, every executor gets the SAME list (itself included).
    # An entry is skipped while ANY peer has a pending claim or an open trade —
    # strictly one position at a time account-wide, by construction. The check
    # is race-safe because consume_session_event is synchronous and sets
    # _entry_pending before returning control to the event loop.
    peer_executors: List["LiveVolumeExecutorOKX"] = field(default_factory=list)
    # Breaker DELEGATION: all executors trade ONE wallet, so wallet-level risk
    # state (daily anchor, peak, loss streak, halts) must live in ONE place.
    # Secondaries point here at the primary; their closes feed the primary's
    # breaker and their entry gate asks the primary. Without this, per-symbol
    # anchors diluted the daily/MDD limits up to ~2x and an alternating
    # BTC/DOGE loss streak never tripped the cooldown.
    breaker_delegate: Optional["LiveVolumeExecutorOKX"] = None

    # MILESTONE 0 — process-wide kill switch (file-backed). When engaged, every
    # live entry is blocked here, the single sync chokepoint. Shared across all
    # symbol executors so one /kill halts the whole account. None = no switch.
    kill_switch: Optional[Any] = None
    # Wall-clock of the last CONFIRMED entry fill — read by the dead-man monitor
    # (no confirmed trade in N hours on --live => something is wedged => halt).
    _last_fill_wall: float = field(default=0.0, init=False)

    # ------------------------------------------------------------------
    # REAL-WALLET circuit breaker. Unlike the session's paper-equity guards,
    # these gate live entries on the ACTUAL OKX balance and real per-trade P&L:
    #   * loss-streak  — N consecutive real losing closes -> pause entries cooldown_s
    #   * daily-loss   — real wallet down daily_loss_pct from the UTC-day open -> halt for the day
    #   * max-drawdown — real wallet down max_dd_pct from peak -> HARD halt (manual restart)
    # on_breaker fires a Telegram card on each trip. All checks read the live
    # balance via _fetch_wallet_usdt so a real blow-up is caught even if the
    # simulation's equity diverges from the wallet.
    # ------------------------------------------------------------------
    breaker_consec_loss_limit: int = 3
    # Only these close reasons advance the consecutive-loss streak. A stop-loss
    # hit is a real adverse move and counts; a time-stop scratch, manual flatten
    # or trend-break exit is small/intentional and must NOT trip the breaker.
    breaker_loss_streak_reasons: frozenset = frozenset({"sl"})
    breaker_cooldown_s: float = 1 * 3600.0
    breaker_daily_loss_pct: float = 0.02
    breaker_max_drawdown_pct: float = 0.10
    on_breaker: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = field(default=None)
    _consec_real_losses: int = field(default=0, init=False)
    _entry_paused_until: float = field(default=0.0, init=False)   # wall time
    _daily_halt_date: str = field(default="", init=False)        # UTC date entries halted
    _daily_anchor_eq: float = field(default=0.0, init=False)
    _daily_anchor_date: str = field(default="", init=False)
    _peak_real_eq: float = field(default=0.0, init=False)
    _hard_halted: bool = field(default=False, init=False)
    _halt_reason: str = field(default="", init=False)

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
    # Synchronous gate claim: set in consume_session_event BEFORE the entry
    # task is scheduled, cleared when _handle_entry finishes claiming (or
    # bails). Without it, two entry events landing in the same loop tick both
    # passed the _open_trade check and opened two concurrent positions.
    _entry_pending: bool = field(default=False, init=False)
    _trade_count: int = field(default=0, init=False)        # FILLED entries
    _attempt_count: int = field(default=0, init=False)      # placement attempts
    _watcher_task: Optional[asyncio.Task] = field(default=None, init=False)
    # Strong refs to fire-and-forget tasks (entry lifecycle, recovery watcher,
    # cards). The event loop holds only weak refs — without this set a GC pass
    # can collect an in-flight task mid-trade (documented asyncio pitfall).
    _bg_tasks: set = field(default_factory=set, init=False)
    # DEMO realism RNG (seeded once in initialize) + per-peg fill decisions.
    _demo_rng: Optional["random.Random"] = field(default=None, init=False)
    _demo_fill_decisions: Dict[str, bool] = field(default_factory=dict, init=False)
    _initialized: bool = field(default=False, init=False)
    # Serializes the strategy's own close path (time-stop / maker-exit) against an
    # operator-triggered manual close from Telegram, so the two can't fire
    # competing close orders / cancels on the same position concurrently.
    _close_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

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
        if self.demo_realism is not None and self.demo_realism.enabled:
            self._demo_rng = random.Random(self.demo_realism.seed)
            LOGGER.info(
                "DEMO REALISM overlay active: entry_fill_prob=%.2f entry_slip=%.1fbps "
                "sl_slip=%.1fbps tp_fill_prob=%.2f seed=%s",
                self.demo_realism.entry_fill_prob, self.demo_realism.entry_slip_bps,
                self.demo_realism.sl_slip_bps, self.demo_realism.tp_fill_prob,
                self.demo_realism.seed,
            )
        # Restore persisted breaker state (hard halt / anchors survive restarts).
        if not self.dry_run:
            self._load_breaker_state()
        self._initialized = True

    # ------------------------------------------------------------------
    # DEMO realism overlay (demo-only; physically inert under --live)
    # ------------------------------------------------------------------
    def _demo_on(self) -> bool:
        """True only when running against the OKX simulated endpoint AND a realism
        overlay is configured. Double-gated: ``demo_realism`` is None under --live."""
        return (
            bool(getattr(self.client, "_simulated", False))
            and self.demo_realism is not None
            and self.demo_realism.enabled
            and self._demo_rng is not None
        )

    def _apply_demo_entry_slip(self, side: str, raw_px: float) -> float:
        """Adverse entry slippage vs the demo avgPx (long fills higher, short lower)."""
        slip = raw_px * (self.demo_realism.entry_slip_bps / 10_000.0)
        px = raw_px + slip if side == "long" else raw_px - slip
        return _round_to_tick(px, self._tick_sz)

    # ------------------------------------------------------------------
    # OKX-confirmation helpers (shared by entry + close paths)
    # ------------------------------------------------------------------
    @staticmethod
    def _position_qty(resp: Dict[str, Any]) -> float:
        """Signed contract qty of the open position in a /positions response (0 if flat)."""
        for p in resp.get("data") or []:
            try:
                q = float(p.get("pos") or 0.0)
            except (TypeError, ValueError):
                q = 0.0
            if q != 0.0:
                return q
        return 0.0

    async def _safe_callback(
        self, cb: Optional[Callable], trade: LiveTradeOKX, name: str,
    ) -> None:
        """Fire a user callback bounded by a timeout; never propagate or hang.

        A confirmation callback does Telegram I/O and a real-account fetch. It
        must never be able to wedge the close watcher — if it hung, the
        ``_open_trade`` reset that unblocks new entries would never run. So we
        time-box it and swallow everything.
        """
        if cb is None:
            return
        try:
            await asyncio.wait_for(cb(trade), timeout=self.callback_timeout_s)
        except asyncio.TimeoutError:
            LOGGER.error("%s callback timed out after %.0fs (cid=%s)",
                         name, self.callback_timeout_s, trade.cl_ord_id)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("%s callback failed: %s", name, exc)

    async def _confirm_flat(
        self, *, timeout_s: Optional[float] = None, poll_s: float = 0.5,
    ) -> Optional[bool]:
        """Poll /positions until the position reads flat, or until timeout.

        Returns True as soon as ANY poll reads flat (the position genuinely
        closed), False only if it never goes flat within the window (a real
        stuck position), None if every poll errored. Polling-until-flat rather
        than a couple of quick checks is the fix for the gate-wedge bug: after a
        maker-exit fills, /positions lags the fill by a second or two, and a
        single lagging "still open" read used to trip a permanent hold. We now
        wait the lag out (returning the instant it reads flat, so no slowdown in
        the normal case).
        """
        deadline = time.time() + (self.flat_timeout_s if timeout_s is None else timeout_s)
        seen = False
        while time.time() < deadline:
            try:
                r = await self.client.get_positions(self.symbol)
                seen = True
                if self._position_qty(r) == 0.0:
                    return True
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("_confirm_flat: positions fetch failed: %s", exc)
            await asyncio.sleep(poll_s)
        return False if seen else None

    async def _recover_stuck_position(self, trade: LiveTradeOKX) -> None:
        """Re-poll a 'still open' position until OKX reads flat, then release the
        entry gate. The hold is NEVER permanent: a slow/lagging close clears in
        seconds, a genuinely stuck position clears whenever it actually closes,
        and either way trading resumes automatically instead of staying dead
        until a restart.

        Critically, this does not just WAIT — on every cycle that still reads
        open it RE-ATTEMPTS a market close. The maker-exit's single best-effort
        taker fallback can be rejected or rate-limited (OKX 50011 isn't retried
        on POSTs) and give up; this keeps slamming a reduceOnly market close
        until the exchange actually reports flat, so a real-money position can't
        sit open just because one close request bounced.
        """
        LOGGER.warning("recovery watcher armed for cid=%s — retrying market close until flat",
                       trade.cl_ord_id)
        pos_side_param = trade.side if self.pos_mode == "hedge" else None
        while self._open_trade is trade:
            await asyncio.sleep(self.recover_poll_s)
            try:
                r = await self.client.get_positions(self.symbol)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("recovery poll failed: %s", exc)
                continue
            if self._position_qty(r) == 0.0:
                LOGGER.info("cid=%s now flat on OKX — releasing entry gate, resuming trading",
                            trade.cl_ord_id)
                if self._open_trade is trade:
                    self._open_trade = None
                return
            # Still open — actively re-attempt a market close instead of waiting.
            if self.dry_run:
                continue
            try:
                resp = await self.client.close_position_market(
                    self.symbol, mgn_mode=self.mgn_mode,
                    pos_side=pos_side_param, auto_cxl=True,
                )
                cdata = (resp.get("data") or [{}])[0]
                LOGGER.warning("recovery: re-attempted market close cid=%s sCode=%s sMsg=%r",
                               trade.cl_ord_id, cdata.get("sCode"), cdata.get("sMsg"))
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("recovery close attempt failed: %s", exc)

    async def _force_market_close(self, trade: LiveTradeOKX, *, attempts: int = 4) -> bool:
        """Last-resort backstop: hammer a market close until /positions reads flat.

        The maker-exit's single best-effort taker fallback can be rejected or
        rate-limited and then silently give up, stranding a real-money position.
        This retries close_position_market with exponential backoff, re-checking
        flat between attempts. Returns True once OKX confirms flat, False if it
        still shows open after all attempts (caller then routes to the alert +
        recovery watcher, which keeps retrying).
        """
        if self.dry_run:
            return False
        pos_side_param = trade.side if self.pos_mode == "hedge" else None
        delay = self.force_close_base_delay_s
        for i in range(max(1, attempts)):
            try:
                r = await self.client.get_positions(self.symbol)
                if self._position_qty(r) == 0.0:
                    return True
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("force_close: positions check failed: %s", exc)
            try:
                resp = await self.client.close_position_market(
                    self.symbol, mgn_mode=self.mgn_mode,
                    pos_side=pos_side_param, auto_cxl=True,
                )
                cdata = (resp.get("data") or [{}])[0]
                cscode = str(cdata.get("sCode", "0"))
                if cscode != "0":
                    LOGGER.warning(
                        "force_close attempt %d/%d rejected sCode=%s sMsg=%r (cid=%s)",
                        i + 1, attempts, cscode, cdata.get("sMsg"), trade.cl_ord_id,
                    )
                else:
                    trade.extras["forced_market_close"] = True
                    LOGGER.warning("force_close attempt %d/%d sent market close (cid=%s)",
                                   i + 1, attempts, trade.cl_ord_id)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("force_close attempt %d/%d failed: %s", i + 1, attempts, exc)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 4.0)
        try:
            r = await self.client.get_positions(self.symbol)
            return self._position_qty(r) == 0.0
        except Exception:  # noqa: BLE001
            return False

    async def _fire_entry_filled(self, trade: LiveTradeOKX) -> None:
        """Announce a CONFIRMED entry fill exactly once."""
        # Set the guard synchronously (before any await) so concurrent fill
        # sites / retries can never double-send the entry card.
        if trade.extras.get("entry_card_sent"):
            return
        trade.extras["entry_card_sent"] = True
        await self._safe_callback(self.on_entry_filled, trade, "on_entry_filled")

    async def _fire_entry_abandoned(self, trade: LiveTradeOKX, reason: str) -> None:
        """Signal an entry that never filled (abandoned/rejected). Idempotent.

        By default the runner leaves ``on_entry_abandoned`` unset, so this is
        silent (log only) — the chosen behaviour for no-fills. It remains a
        clean hook if visibility into miss rate is wanted later.
        """
        if trade.extras.get("abandon_fired"):
            return
        trade.extras["abandon_fired"] = True
        if not trade.cancel_reason:
            trade.cancel_reason = reason
        await self._safe_callback(self.on_entry_abandoned, trade, "on_entry_abandoned")

    # ------------------------------------------------------------------
    # REAL-WALLET circuit breaker
    # ------------------------------------------------------------------
    def _entry_blocked_reason(self) -> Optional[str]:
        """Reason new entries are blocked right now, or None if clear.

        Sync (called from the sync entry hook). Reads only local breaker state;
        the state itself is refreshed from the live wallet on each close.
        """
        # Kill switch is process-wide and checked on EVERY executor (not via the
        # delegate) so an engaged halt blocks entries regardless of which symbol
        # asks. Cheapest possible guard — a file existence check, entries are
        # minutes apart.
        if self.kill_switch is not None and self.kill_switch.is_engaged():
            return f"killed:{self.kill_switch.reason() or 'manual'}"
        if self.breaker_delegate is not None and self.breaker_delegate is not self:
            return self.breaker_delegate._entry_blocked_reason()  # noqa: SLF001
        if self._hard_halted:
            return f"hard_halt:{self._halt_reason}"
        now = time.time()
        if now < self._entry_paused_until:
            return f"loss_streak_cooldown:{(self._entry_paused_until - now) / 60:.0f}m_left"
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        if self._daily_halt_date == today:
            return "daily_loss_halt"
        return None

    async def _fire_breaker(self, kind: str, detail: str, **extra: Any) -> None:
        LOGGER.warning("REAL-WALLET BREAKER [%s] %s", kind, detail)
        if self.on_breaker is None:
            return
        payload = {"kind": kind, "detail": detail, **extra}
        try:
            await asyncio.wait_for(self.on_breaker(payload), timeout=self.callback_timeout_s)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("on_breaker callback failed: %s", exc)

    async def _update_breaker_on_close(self, trade: LiveTradeOKX) -> None:
        """Update the real-wallet breaker after a VERIFIED, priced close.

        With a breaker_delegate set (multi-instrument), the close feeds the
        DELEGATE's wallet-level state instead of a per-symbol copy.

        Counts consecutive real losing closes (loss-streak cooldown), and reads
        the live wallet to enforce the daily-loss and max-drawdown halts. Runs at
        most once per trade. Never raises — risk gating must not wedge the close.
        """
        if trade.extras.get("breaker_counted"):
            return
        trade.extras["breaker_counted"] = True
        if self.breaker_delegate is not None and self.breaker_delegate is not self:
            trade.extras.pop("breaker_counted", None)
            await self.breaker_delegate._update_breaker_on_close(trade)  # noqa: SLF001
            return
        try:
            net = trade.real_gross_pnl - trade.real_open_fee - trade.real_close_fee
            reason = (trade.close_reason or "").lower()
            if net >= 0:
                # any profitable / break-even close clears the streak
                self._consec_real_losses = 0
            elif reason in self.breaker_loss_streak_reasons:
                # a "real" loss (default: a stop-loss hit) advances the streak
                self._consec_real_losses += 1
            else:
                # losing close that isn't a streak reason (time_stop scratch,
                # manual flatten, trend_break) — noise; leave the streak as-is so
                # it neither trips nor resets the breaker.
                LOGGER.info(
                    "breaker: not counting %s loss toward streak "
                    "(net=%+.4f, streak stays at %d)",
                    reason or "unknown", net, self._consec_real_losses,
                )
            if (self._consec_real_losses >= self.breaker_consec_loss_limit
                    and time.time() >= self._entry_paused_until):
                n = self._consec_real_losses
                self._entry_paused_until = time.time() + self.breaker_cooldown_s
                self._consec_real_losses = 0
                self._save_breaker_state()
                await self._fire_breaker(
                    "loss_streak",
                    f"{n} consecutive REAL losing closes — pausing entries "
                    f"{self.breaker_cooldown_s / 3600:.1f}h",
                    consec=n, cooldown_s=self.breaker_cooldown_s,
                )

            eq = await self._fetch_wallet_usdt()
            if not eq or eq <= 0:
                return
            today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
            if self._daily_anchor_date != today:
                self._daily_anchor_date = today
                self._daily_anchor_eq = eq
            self._peak_real_eq = max(self._peak_real_eq, eq) if self._peak_real_eq > 0 else eq

            if self._daily_anchor_eq > 0:
                day_loss = (self._daily_anchor_eq - eq) / self._daily_anchor_eq
                if day_loss >= self.breaker_daily_loss_pct and self._daily_halt_date != today:
                    self._daily_halt_date = today
                    await self._fire_breaker(
                        "daily_loss",
                        f"real wallet down {day_loss * 100:.1f}% today "
                        f"(${eq:.2f} from ${self._daily_anchor_eq:.2f}) — entries halted until UTC tomorrow",
                        equity=eq, day_loss_pct=day_loss,
                    )

            if self._peak_real_eq > 0:
                dd = (self._peak_real_eq - eq) / self._peak_real_eq
                if dd >= self.breaker_max_drawdown_pct and not self._hard_halted:
                    self._hard_halted = True
                    self._halt_reason = f"drawdown_{dd * 100:.1f}pct"
                    await self._fire_breaker(
                        "max_drawdown",
                        f"real wallet down {dd * 100:.1f}% from peak "
                        f"(${eq:.2f} from ${self._peak_real_eq:.2f}) — HARD HALT, manual restart required",
                        equity=eq, drawdown_pct=dd,
                    )
            self._save_breaker_state()
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("breaker update failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Breaker persistence. All breaker state used to be memory-only, so any
    # restart — including the keep-alive supervisor's automatic one — wiped
    # the 40% hard halt ("manual restart required" was untrue), re-baselined
    # the peak at the depleted wallet (each crash granted a fresh -40%), and
    # reset the daily anchor mid-day. Now it survives restarts.
    # ------------------------------------------------------------------
    def _breaker_state_file(self) -> pathlib.Path:
        # Per-symbol: with multiple instruments, two executors sharing one
        # file would clobber each other's anchors.
        return self.log_dir / f"breaker_state_{to_okx_inst_id(self.symbol)}.json"

    def _save_breaker_state(self) -> None:
        try:
            state = {
                "peak_real_eq": self._peak_real_eq,
                "daily_anchor_eq": self._daily_anchor_eq,
                "daily_anchor_date": self._daily_anchor_date,
                "daily_halt_date": self._daily_halt_date,
                "hard_halted": self._hard_halted,
                "halt_reason": self._halt_reason,
                "entry_paused_until": self._entry_paused_until,
                "consec_real_losses": self._consec_real_losses,
            }
            p = self._breaker_state_file()
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=1))
            os.replace(tmp, p)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("breaker state save failed: %s", exc)

    def _load_breaker_state(self) -> None:
        p = self._breaker_state_file()
        if not p.exists():
            # Migrate the pre-multi-symbol file so a persisted HARD HALT or
            # real anchors survive this upgrade instead of silently resetting.
            legacy = self.log_dir / "breaker_state.json"
            if legacy.exists():
                try:
                    os.replace(legacy, p)
                    LOGGER.info("migrated legacy breaker state -> %s", p.name)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("legacy breaker migration failed: %s", exc)
            if not p.exists():
                return
        try:
            s = json.loads(p.read_text())
            self._peak_real_eq = float(s.get("peak_real_eq", 0.0))
            self._daily_anchor_eq = float(s.get("daily_anchor_eq", 0.0))
            self._daily_anchor_date = str(s.get("daily_anchor_date", ""))
            self._daily_halt_date = str(s.get("daily_halt_date", ""))
            self._hard_halted = bool(s.get("hard_halted", False))
            self._halt_reason = str(s.get("halt_reason", ""))
            self._entry_paused_until = float(s.get("entry_paused_until", 0.0))
            self._consec_real_losses = int(s.get("consec_real_losses", 0))
            if self._hard_halted:
                LOGGER.error(
                    "BREAKER STATE RESTORED: HARD HALT is ACTIVE (%s). Entries stay "
                    "blocked — delete %s after funding/reviewing to resume.",
                    self._halt_reason, p,
                )
            else:
                LOGGER.info("breaker state restored: peak=$%.4f anchor=$%.4f (%s)",
                            self._peak_real_eq, self._daily_anchor_eq, self._daily_anchor_date)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("breaker state load failed (fresh anchors): %s", exc)

    # ------------------------------------------------------------------
    # Hook called by the session's event_callback.
    # ------------------------------------------------------------------
    def consume_session_event(self, evt) -> None:  # type: FarmerEvent
        """Sync hook — schedules async work on the running loop."""
        if evt.kind != "entry":
            return
        blocked = self._entry_blocked_reason()
        if blocked is not None:
            LOGGER.warning("entry blocked by REAL-wallet breaker: %s", blocked)
            return
        if self._trade_count >= self.max_live_trades:
            LOGGER.warning("max_live_trades=%d reached; ignoring entry",
                           self.max_live_trades)
            return
        for peer in self.peer_executors:
            if peer is self:
                continue
            if peer._entry_pending or peer._open_trade is not None:  # noqa: SLF001
                LOGGER.info(
                    "entry skipped on %s — global gate held by %s",
                    self.symbol, peer.symbol,
                )
                return
        if self._entry_pending or self._open_trade is not None:
            # Gate on ANY prior trade, including closed-but-not-finalized.
            # The old `not closed` carve-out reopened the gate the instant a
            # close was priced — but the prior trade's _confirm_flat was still
            # polling, and with same-bar re-entry a NEW position could open
            # inside that window, be read as "old position still open", and
            # get force-market-closed by the old trade's finalizer. The gate
            # now stays shut the few seconds until finalize releases it.
            LOGGER.warning("ignoring entry — prior live trade still open/finalizing: %s",
                           self._open_trade.cl_ord_id if self._open_trade else "<claiming>")
            return
        # Claim the single-position gate SYNCHRONOUSLY — _handle_entry only
        # assigns _open_trade after several awaits, so without this claim two
        # back-to-back events could both pass the gate above.
        self._entry_pending = True
        self._spawn(self._handle_entry(evt.payload), name="handle_entry")

    # ------------------------------------------------------------------
    def _spawn(self, coro, *, name: str) -> asyncio.Task:
        """create_task with a strong reference held until completion."""
        task = asyncio.create_task(coro, name=name)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    # ------------------------------------------------------------------
    async def _fetch_wallet_usdt(self) -> Optional[float]:
        """Live USDT equity from /account/balance (None on failure)."""
        try:
            bal = await self.client.get_balance("USDT")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("balance fetch failed: %s", exc)
            return None
        bd = (bal.get("data") or [{}])[0]
        for det in (bd.get("details") or []):
            if det.get("ccy") == "USDT":
                try:
                    return float(det.get("eq") or 0.0)
                except (TypeError, ValueError):
                    return None
        try:
            return float(bd.get("totalEq")) if bd.get("totalEq") else None
        except (TypeError, ValueError):
            return None

    async def _resize_on_real_balance(self, payload: Dict[str, Any], sim_notional: float) -> float:
        """Re-base the trade notional on the LIVE OKX wallet balance.

        notional = real_balance × margin_frac × leverage. Leverage is
        equity-independent (risk%/(margin_frac×SL)), so the SIM's leverage is
        reused verbatim. Falls back to the SIM's notional if sizing is disabled
        (margin_frac == 0), in dry-run, or if the balance fetch fails — never a
        hardcoded capital figure.
        """
        if self.margin_frac <= 0 or self.dry_run:
            return sim_notional
        lev = float(payload.get("leverage") or 0.0)
        if lev <= 0:
            return sim_notional
        # The session's pace controller scales margin per trade to stay on the
        # campaign target; honor its fraction when present (bounded by config),
        # falling back to the static config value.
        margin_frac = self.margin_frac
        payload_frac = float(payload.get("margin_fraction") or 0.0)
        if payload_frac > 0:
            margin_frac = min(payload_frac, max(self.margin_frac * 3.0, self.margin_frac))
        real_bal = await self._fetch_wallet_usdt()
        if not real_bal or real_bal <= 0:
            LOGGER.warning("real balance unavailable; falling back to SIM notional $%.2f", sim_notional)
            return sim_notional
        # Cap the sizing base at the configured working capital so a small account
        # can run on a larger (demo) wallet — risk off min(wallet, cap), not all of it.
        base = real_bal
        capped = self.working_capital_usd > 0 and real_bal > self.working_capital_usd
        if capped:
            base = self.working_capital_usd
        notional = base * margin_frac * lev
        LOGGER.info(
            "sizing on %s: $%.4f × margin_frac %.3f × lev %.1f = notional $%.2f (wallet $%.2f, sim was $%.2f)",
            "WORKING CAP" if capped else "REAL wallet",
            base, margin_frac, lev, notional, real_bal, sim_notional,
        )
        return notional

    async def _handle_entry(self, payload: Dict[str, Any]) -> None:
        """Place a real OKX order matching the session's entry decision."""
        try:
            trade, ok_side, pos_side, sz_payload = await self._prepare_entry(payload)
        finally:
            # Gate ownership transfers from the synchronous _entry_pending
            # claim to _open_trade (set inside _prepare_entry) — or is
            # released entirely when sizing bailed.
            self._entry_pending = False
        if trade is None:
            return
        try:
            await self._submit_and_watch(trade, payload, ok_side, pos_side, sz_payload)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("entry lifecycle error (cid=%s): %s", trade.cl_ord_id, exc)
            # Never let an unexpected placement error wedge the entry gate.
            # (Once filled, _watch_open_position owns gate release.)
            if self._open_trade is trade and not trade.filled and not trade.extras.get("still_open"):
                self._open_trade = None

    async def _prepare_entry(self, payload: Dict[str, Any]):
        """Size + build the trade and claim the gate. Returns (None, …) to skip."""
        side = str(payload["side"]).lower()                 # long | short
        entry_req = float(payload["price"])
        notional = float(payload["notional"])
        tp_px = float(payload["tp"])
        sl_px = float(payload["sl"])
        tp_bps = float(payload["tp_bps"])
        sl_bps = float(payload["sl_bps"])

        # Re-base the size on the REAL OKX wallet (not the simulation's equity).
        notional = await self._resize_on_real_balance(payload, notional)

        # Notional USD -> BTC -> contracts (BTC-USDT-SWAP: 1 ct = 0.01 BTC)
        # btc_qty = notional / entry_req
        # contracts = btc_qty / ctVal
        btc_qty = notional / entry_req
        contracts = btc_qty / self._ct_val
        contracts = _round_size(contracts, self._lot_sz)
        if contracts < self._min_sz:
            LOGGER.warning("sized %.4f contracts < minSz %.4f; skipping",
                           contracts, self._min_sz)
            return None, "", None, ""

        # Entry reference price. With the re-peg loop enabled (the default)
        # the loop pins to the live touch itself per peg, so a pre-pin ticker
        # fetch here would be fetched-and-discarded — skip the wasted call and
        # the extra RTT of pre-fill latency. The legacy single-shot path still
        # pins once below.
        ref_px = entry_req
        if not self.entry_repeg_cfg.enabled:
            # Pin the entry to the current top-of-book so post_only lands at the
            # front of the maker queue. The bar-close reference price is stale by
            # the time we submit; placing AT the close leaves us deep in the book
            # and post_only timeout-cancels with no fill. We never cross. Falls
            # back to bar close if the ticker fetch fails. TP/SL targets stay
            # anchored to the session's reference so ledgers don't diverge.
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
        # posSide is ONLY valid in hedge ("long_short") mode. In net (one-way)
        # mode OKX rejects any posSide with sCode=51000 "Parameter posSide error",
        # so it must be omitted entirely (None) for BOTH directions. The previous
        # one-liner mis-parsed by precedence and always sent "long" for longs.
        pos_side = (
            ("long" if side == "long" else "short")
            if self.pos_mode == "hedge"
            else None
        )

        trade = LiveTradeOKX(
            cl_ord_id=cl_ord_id, side=side,
            entry_px_req=entry_px, notional_usd=notional,
            sz_contracts=contracts, tp_px=tp_px_r, sl_px=sl_px_r,
            tp_bps=tp_bps, sl_bps=sl_bps, placed_at=time.time(),
        )
        trade.extras["signal_px"] = entry_req
        # Carried for the Telegram entry card: margin/leverage actually used.
        lev_used = float(payload.get("leverage") or 0.0)
        trade.extras["leverage"] = lev_used
        trade.extras["margin_usd"] = (notional / lev_used) if lev_used > 0 else 0.0
        trade.extras["margin_fraction"] = float(payload.get("margin_fraction") or 0.0)
        self._open_trade = trade
        self._attempt_count += 1

        sz_payload = _fmt_size(contracts, self._lot_sz)
        LOGGER.info(
            "LIVE entry: %s %s sz=%s ct (notional=$%.2f) px=%s  TP=%s (maker)  SL=%s (market)  cid=%s",
            side.upper(), to_okx_inst_id(self.symbol),
            sz_payload, notional, entry_px, tp_px_r, sl_px_r, cl_ord_id,
        )
        return trade, ok_side, pos_side, sz_payload

    def _attach_tp_args(self, tp_px_r: float) -> Dict[str, Any]:
        """TP attachment for the entry order. With resting_tp on, the TP is a
        SEPARATE resting post_only order placed after fill (always maker), so
        the entry attaches ONLY the SL — no trigger-TP (those fill taker on a
        gap). With it off, the legacy attached trigger-limit TP."""
        if self.resting_tp_enabled:
            return {"tp_trigger_px": None, "tp_ord_px": None}
        return {"tp_trigger_px": str(tp_px_r), "tp_ord_px": str(tp_px_r)}

    async def _place_resting_tp(self, trade: LiveTradeOKX) -> None:
        """Rest a reduce-only post_only limit at tp_px so a winning close fills
        MAKER. Stores the ordId for cancellation on any non-TP exit. If the
        price has ALREADY gapped past the TP (post_only would cross), capture
        the profit immediately at market rather than give it back."""
        if not self.resting_tp_enabled or self.dry_run or trade.tp_px <= 0:
            return
        side = "sell" if trade.side == "long" else "buy"
        pos_side = trade.side if self.pos_mode == "hedge" else None
        cl = "vftp" + uuid.uuid4().hex[:12]
        sz = _fmt_size(trade.sz_contracts, self._lot_sz)
        px = str(_round_to_tick(trade.tp_px, self._tick_sz))
        try:
            resp = await self.client.place_order(
                symbol=self.symbol, side=side, pos_side=pos_side,
                td_mode=self.mgn_mode, ord_type="post_only", sz=sz, px=px,
                client_oid=cl, reduce_only=True,
            )
            data = (resp.get("data") or [{}])[0]
            sc = str(data.get("sCode", "0"))
            if sc == "0":
                trade.extras["resting_tp_ord_id"] = data.get("ordId")
                LOGGER.info("resting TP placed: %s %s @ %s ordId=%s",
                            side, sz, px, data.get("ordId"))
            elif sc == "51280":
                # would cross → price already past TP → we're in profit; take it.
                LOGGER.info("resting TP would cross (price past TP) — market-taking profit (cid=%s)",
                            trade.cl_ord_id)
                try:
                    await self.client.close_position_market(
                        self.symbol, mgn_mode=self.mgn_mode, pos_side=pos_side, auto_cxl=True)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("resting TP profit-take market close failed: %s", exc)
            else:
                LOGGER.warning("resting TP rejected sCode=%s sMsg=%r (the time-stop will exit)",
                               sc, data.get("sMsg"))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("resting TP placement failed (time-stop will exit): %s", exc)

    async def _cancel_resting_tp(self, trade: LiveTradeOKX) -> None:
        """Cancel the resting TP if still open. Idempotent — a no-op (or a
        harmless 'already filled/canceled' error) once it has filled. MUST run
        on every non-TP exit so a stale reduce-only order can't fill against a
        later position."""
        oid = trade.extras.pop("resting_tp_ord_id", None)
        if not oid:
            return
        try:
            await self.client.cancel_order(self.symbol, ord_id=oid)
            LOGGER.debug("resting TP cancelled ordId=%s", oid)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("resting TP cancel (likely already filled/gone): %s", exc)

    async def _submit_and_watch(
        self, trade: LiveTradeOKX, payload: Dict[str, Any],
        ok_side: str, pos_side: Optional[str], sz_payload: str,
    ) -> None:
        entry_px = trade.entry_px_req
        tp_px_r = trade.tp_px
        sl_px_r = trade.sl_px
        contracts = trade.sz_contracts

        if self.dry_run:
            LOGGER.info("DRY-RUN: not submitting")
            # Release the gate — leaving the unfilled dry-run trade claimed
            # wedged every subsequent entry of the run ("prior trade still open").
            self._open_trade = None
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
                if trade.extras.get("sweep_failed"):
                    # The cancel could not be CONFIRMED: a full-size bracketed
                    # order may still rest on the book. Hold the gate and hand
                    # off to the recovery watcher — releasing here would let a
                    # second position stack on top of a late fill.
                    await self._fire_entry_abandoned(trade, "sweep_failed_recovering")
                    self._log_trade(trade)
                    self._spawn(self._recover_unswept_entry(trade), name="recover_unswept")
                    return
                # entry abandoned (re-pegs exhausted) — no position exists, so
                # the gate is released; the next signal is the next chance.
                await self._fire_entry_abandoned(trade, trade.cancel_reason or "repeg_exhausted")
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
                    **self._attach_tp_args(tp_px_r),
                    sl_trigger_px=str(sl_px_r), sl_ord_px="-1",
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("place_order failed: %s", exc)
                trade.canceled = True
                trade.cancel_reason = f"submit_error: {exc}"
                await self._fire_entry_abandoned(trade, trade.cancel_reason)
                self._log_trade(trade)
                self._open_trade = None
                return
            data = (resp.get("data") or [{}])[0]
            if str(data.get("sCode", "0")) != "0":
                LOGGER.error("OKX rejected order: sCode=%s sMsg=%r",
                             data.get("sCode"), data.get("sMsg"))
                trade.canceled = True
                trade.cancel_reason = f"sCode={data.get('sCode')}:{data.get('sMsg')}"
                await self._fire_entry_abandoned(trade, trade.cancel_reason)
                self._log_trade(trade)
                self._open_trade = None
                return
            trade.ord_id = data.get("ordId")
            LOGGER.info("submitted ordId=%s (cid=%s)", trade.ord_id, trade.cl_ord_id)
            filled = await self._wait_for_entry_fill_or_timeout(trade)
            if not filled:
                await self._fire_entry_abandoned(trade, trade.cancel_reason or "post_only_timeout")
                self._log_trade(trade)
                self._open_trade = None
                return

        # Entry is CONFIRMED filled on OKX (only reachable when a real position
        # exists). The fill — not the placement attempt — consumes the
        # max_live_trades budget, so a string of no-fills can't exhaust it.
        self._trade_count += 1
        self._last_fill_wall = time.time()   # dead-man monitor heartbeat
        # Rest the maker TP IMMEDIATELY (before the watch) so it's in the book
        # the instant price can reach it — fills as maker instead of the
        # attached trigger-TP's taker-on-gap. No-op when resting_tp is off.
        await self._place_resting_tp(trade)
        # Fire the entry card CONCURRENTLY — the card does a chart render +
        # account fetch + photo upload (up to callback_timeout_s); awaiting it
        # here used to delay the first position poll, so a fast TP inside that
        # window was detected late and the gate sat closed for nothing.
        self._spawn(self._fire_entry_filled(trade), name="entry_card")
        # Position is open — start the position + time-stop watch.
        await self._watch_open_position(trade)

    def _adopt_order_fill(self, trade: LiveTradeOKX, od: Dict[str, Any], note: str = "") -> bool:
        """Adopt an order-state read into ``trade`` as the live entry fill.

        Handles BOTH a full fill (state=='filled') and a TERMINAL PARTIAL —
        canceled with accFillSz>0. The partial contracts are a REAL position
        on OKX (with the attached TP/SL covering the filled part); treating
        them as no-fill used to either abandon the entry with an unmanaged
        position left behind, or re-place a FULL-size order on top (up to 2x
        intended). Mirrors the same fix this commit applied in maker_exit's
        taker fallback. On partial adoption the trade is resized to what
        actually filled so every downstream exit/PnL/volume figure is honest.
        """
        state = str(od.get("state", ""))
        try:
            acc = float(od.get("accFillSz") or 0.0)
        except (TypeError, ValueError):
            acc = 0.0
        full = state == "filled"
        partial_terminal = state in {"canceled", "mmp_canceled"} and acc > 0
        if not (full or partial_terminal):
            return False
        if partial_terminal:
            ratio = (acc / trade.sz_contracts) if trade.sz_contracts > 0 else 1.0
            trade.extras["partial_entry"] = True
            trade.extras["requested_sz"] = trade.sz_contracts
            trade.sz_contracts = acc
            trade.notional_usd *= ratio
            note = (note + " partial-adopt").strip()
        trade.filled = True
        _raw = float(od.get("avgPx") or trade.entry_px_req)
        trade.extras["demo_raw_fill_px"] = _raw
        trade.fill_px = self._apply_demo_entry_slip(trade.side, _raw) if self._demo_on() else _raw
        trade.fill_ts = int(od.get("fillTime") or int(time.time() * 1000))
        trade.real_open_fee = abs(float(od.get("fee") or 0.0))
        LOGGER.info(
            "FILL  cid=%s  px=%.2f  sz=%s  fee=%.6f%s",
            trade.cl_ord_id, trade.fill_px, trade.sz_contracts,
            trade.real_open_fee, f" ({note})" if note else "",
        )
        return True

    async def _recover_unswept_entry(self, trade: LiveTradeOKX) -> None:
        """Resolve an entry order whose cancel could not be confirmed.

        The order may still be resting (and could fill any moment), may have
        filled already, or may have actually been canceled. Poll until a
        terminal state, re-attempting the cancel each round; adopt a fill and
        run the normal position watch, or release the gate on a clean cancel.
        The gate stays held the whole time so no second position can stack.
        """
        for _ in range(60):                      # ~10 min at 10s
            await asyncio.sleep(10.0)
            try:
                r = await self.client.get_order(
                    self.symbol, ord_id=trade.ord_id, client_oid=trade.cl_ord_id,
                )
                od = (r.get("data") or [{}])[0]
                if self._adopt_order_fill(trade, od, note="unswept-recovery"):
                    self._trade_count += 1
                    self._last_fill_wall = time.time()
                    await self._place_resting_tp(trade)
                    self._spawn(self._fire_entry_filled(trade), name="entry_card")
                    await self._watch_open_position(trade)
                    return
                if str(od.get("state", "")) in {"canceled", "mmp_canceled"}:
                    LOGGER.info("unswept entry resolved: canceled clean (cid=%s)", trade.cl_ord_id)
                    if self._open_trade is trade:
                        self._open_trade = None
                    return
                # Still live — try the cancel again.
                await self.client.cancel_order(self.symbol, ord_id=trade.ord_id)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("unswept-entry recovery poll failed: %s", exc)
        LOGGER.error(
            "unswept entry NOT resolved after first recovery window (cid=%s ordId=%s) — "
            "gate stays held; continuing to poll every 60s (/orders, /cancel to inspect)",
            trade.cl_ord_id, trade.ord_id,
        )
        # NEVER give up while the gate is held: a silent return left the bot
        # permanently idle. Slow indefinite polling, same terminal handling.
        while self._open_trade is trade:
            await asyncio.sleep(60.0)
            try:
                r = await self.client.get_order(
                    self.symbol, ord_id=trade.ord_id, client_oid=trade.cl_ord_id,
                )
                od = (r.get("data") or [{}])[0]
                if self._adopt_order_fill(trade, od, note="unswept-recovery-slow"):
                    self._trade_count += 1
                    self._last_fill_wall = time.time()
                    await self._place_resting_tp(trade)
                    self._spawn(self._fire_entry_filled(trade), name="entry_card")
                    await self._watch_open_position(trade)
                    return
                if str(od.get("state", "")) in {"canceled", "mmp_canceled"}:
                    if self._open_trade is trade:
                        self._open_trade = None
                    return
                await self.client.cancel_order(self.symbol, ord_id=trade.ord_id)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("unswept-entry slow recovery poll failed: %s", exc)

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
        # Rejection storm guard (mirrors maker_exit): placement errors and
        # would-cross rejections no longer burn the re-peg budget — only pegs
        # that actually rested an order count.
        rejects = 0
        max_rejects = max(cfg.max_repegs * 3, 12)
        signal_px = float(trade.extras.get("signal_px") or 0.0)
        resting = False        # a live post_only currently rests on the book
        last_px = 0.0

        def _capture_fill(od: Dict[str, Any], note: str = "") -> bool:
            """Adopt a fill (full OR terminal partial) into the trade."""
            if self._adopt_order_fill(trade, od, note=note):
                self._last_entry_repegs = repegs
                return True
            return False

        async def _sweep_resting() -> bool:
            """Cancel the resting peg; True if it actually filled in the race
            window. MUST run on every exit path — an order left resting after
            an 'abandoned' entry would fill later as an unmanaged position.

            A cancel that cannot be CONFIRMED (POSTs are not retried at the
            client layer, and a 429/timeout here is exactly the transient the
            rest of this commit hardens against) must NOT clear ``resting`` —
            doing so let the loop place a second full-size order, or abandon
            the entry with a live bracketed order still on the book. On
            persistent failure we set extras['sweep_failed'] and the caller
            holds the gate + hands off to _recover_unswept_entry.
            """
            nonlocal resting
            if not resting or not trade.ord_id:
                return False
            confirmed_off_book = False
            for attempt in range(4):
                try:
                    resp = await self.client.cancel_order(self.symbol, ord_id=trade.ord_id)
                    cdata = (resp.get("data") or [{}])[0]
                    cscode = str(cdata.get("sCode", "0"))
                    # 0 = cancel accepted; 51400/51401/51402/51410 = already
                    # canceled / completed — either way nothing rests anymore.
                    if cscode in {"0", "51400", "51401", "51402", "51410"}:
                        confirmed_off_book = True
                        break
                    LOGGER.warning("entry_repeg: cancel rejected sCode=%s (try %d/4)",
                                   cscode, attempt + 1)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("entry_repeg: cancel failed (try %d/4): %s",
                                   attempt + 1, exc)
                await asyncio.sleep(0.4 * (attempt + 1))
                # Between attempts the order may have reached a terminal state
                # on its own (filled, or our earlier cancel actually landed).
                try:
                    r = await self.client.get_order(
                        self.symbol, ord_id=trade.ord_id, client_oid=trade.cl_ord_id,
                    )
                    od = (r.get("data") or [{}])[0]
                    if _capture_fill(od, "race-fill"):
                        resting = False
                        return True
                    if str(od.get("state", "")) in {"canceled", "mmp_canceled"}:
                        confirmed_off_book = True
                        break
                except Exception:  # noqa: BLE001
                    pass
            if not confirmed_off_book:
                trade.extras["sweep_failed"] = True
                LOGGER.error(
                    "entry_repeg: SWEEP FAILED — order %s may still rest on the "
                    "book; holding the entry gate (recovery watcher takes over)",
                    trade.ord_id,
                )
                return False
            resting = False
            try:
                r = await self.client.get_order(
                    self.symbol, ord_id=trade.ord_id, client_oid=trade.cl_ord_id,
                )
                return _capture_fill((r.get("data") or [{}])[0], "race-fill")
            except Exception:  # noqa: BLE001
                return False

        while time.time() < deadline_wall and repegs < cfg.max_repegs and rejects < max_rejects:
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

            # 2b. Chase guard: the TP/SL bracket stays anchored to the SIGNAL
            # price, so chasing too far skews the effective bracket (a TP can
            # end up nearly through price at fill). Abandon past the limit.
            if cfg.max_chase_bps > 0 and signal_px > 0:
                drift_bps = ((entry_px - signal_px) if side == "long" else (signal_px - entry_px)) / signal_px * 10_000.0
                if drift_bps > cfg.max_chase_bps:
                    if await _sweep_resting():
                        return True
                    LOGGER.warning(
                        "entry abandoned: touch drifted %+.1fbps from signal (max_chase %.1f)",
                        drift_bps, cfg.max_chase_bps,
                    )
                    trade.canceled = True
                    trade.cancel_reason = f"chase_limit_{drift_bps:.1f}bps"
                    self._last_entry_repegs = repegs
                    return False

            # DEMO: model post-only queue risk. Decide once per peg; on a "miss"
            # rest several ticks deeper so the demo engine genuinely won't fill —
            # the cancel/amend re-peg loop then runs unchanged (no desync).
            if self._demo_on():
                demo_key = f"{trade.cl_ord_id}#{repegs}"
                hit = self._demo_fill_decisions.get(demo_key)
                if hit is None:
                    hit = self._demo_rng.random() <= self.demo_realism.entry_fill_prob
                    if len(self._demo_fill_decisions) > 2_000:
                        self._demo_fill_decisions.clear()   # bound long-run growth
                    self._demo_fill_decisions[demo_key] = hit
                if not hit:
                    px2 = (best_ask - 3 * self._tick_sz) if side == "long" else (best_bid + 3 * self._tick_sz)
                    entry_px = _round_to_tick(px2, self._tick_sz)

            # 3. Rest an order at entry_px: amend the existing one (1 call, no
            # naked window) or place a fresh post_only with the TP/SL bundle.
            if resting and cfg.use_amend and entry_px != last_px:
                try:
                    aresp = await self.client.amend_order(
                        self.symbol, ord_id=trade.ord_id, new_px=str(entry_px),
                    )
                    adata = (aresp.get("data") or [{}])[0]
                    ascode = str(adata.get("sCode", "0"))
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("entry_repeg: amend error: %s", exc)
                    ascode = "exc"
                if ascode == "0":
                    last_px = entry_px
                    trade.entry_px_req = entry_px
                    LOGGER.info("entry_repeg #%d amended ordId=%s px=%.2f (bid=%.2f ask=%.2f)",
                                repegs + 1, trade.ord_id, entry_px, best_bid, best_ask)
                else:
                    # Amend rejected: the order may have just filled, may have
                    # been canceled, or the amend would cross. Poll once; on a
                    # fill we're done, otherwise cancel + fall back to replace.
                    try:
                        r = await self.client.get_order(
                            self.symbol, ord_id=trade.ord_id, client_oid=trade.cl_ord_id,
                        )
                        od = (r.get("data") or [{}])[0]
                        if _capture_fill(od, "amend-race"):
                            return True
                        if str(od.get("state", "")) not in {"canceled", "mmp_canceled"}:
                            await _sweep_resting()
                            if trade.filled:
                                return True
                        else:
                            resting = False
                    except Exception:  # noqa: BLE001
                        await _sweep_resting()
                        if trade.filled:
                            return True
                    rejects += 1
                    continue
            elif not resting:
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
                        **self._attach_tp_args(tp_px_r),
                        sl_trigger_px=str(sl_px_r), sl_ord_px="-1",
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("entry_repeg: place_order error: %s", exc)
                    rejects += 1
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
                    rejects += 1
                    await asyncio.sleep(cfg.poll_ms / 1000.0)
                    continue

                trade.ord_id = data.get("ordId")
                resting = True
                last_px = entry_px
                LOGGER.info(
                    "entry_repeg #%d submitted ordId=%s cid=%s px=%.2f (bid=%.2f ask=%.2f)",
                    repegs + 1, trade.ord_id, cl_ord, entry_px, best_bid, best_ask,
                )

            # 4. Wait up to repeg_ms for fill or terminal state
            wait_deadline = time.time() + cfg.repeg_ms / 1000.0
            while time.time() < wait_deadline and time.time() < deadline_wall:
                await asyncio.sleep(cfg.poll_ms / 1000.0)
                try:
                    r = await self.client.get_order(
                        self.symbol, ord_id=trade.ord_id, client_oid=trade.cl_ord_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("entry_repeg: get_order failed: %s", exc)
                    continue
                od = (r.get("data") or [{}])[0]
                if _capture_fill(od):
                    return True
                if str(od.get("state", "")) in {"canceled", "mmp_canceled"}:
                    resting = False
                    break

            # 5. Window elapsed without fill. With amend enabled the order keeps
            # resting (queue position at the old price is forfeited on the next
            # amend anyway, but there is never a moment with nothing on the
            # book); legacy mode cancels and replaces.
            if resting and not cfg.use_amend:
                if await _sweep_resting():
                    return True
            repegs += 1

        # Loop exhausted (deadline / max repegs / reject storm). Sweep any
        # still-resting order FIRST — on every path below, an order left on the
        # book could fill later as a completely unmanaged position.
        if await _sweep_resting():
            return True
        if trade.extras.get("sweep_failed"):
            # The maker order may STILL be resting — the taker fallback below
            # would market-fill a second full-size position on top of it.
            # Bail; the caller hands off to the unswept-entry recovery.
            trade.canceled = True
            trade.cancel_reason = "sweep_failed"
            self._last_entry_repegs = repegs
            return False

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
        # posSide must match the account position mode: long/short in hedge mode,
        # omitted in net (one-way) mode. Sending "net" (or any posSide) on a
        # net-mode account makes OKX reject the market fallback with sCode=51000
        # "Parameter posSide error", so just pass the same posSide we computed for
        # the maker entry (None in net mode).
        pos_side_for_market = pos_side
        try:
            resp = await self.client.place_order(
                symbol=self.symbol,
                side=ok_side, pos_side=pos_side_for_market,
                td_mode=self.mgn_mode, ord_type="market",
                sz=sz_payload, client_oid=cl_ord,
                **self._attach_tp_args(tp_px_r),
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
                _raw = float(data.get("avgPx") or trade.entry_px_req)
                trade.extras["demo_raw_fill_px"] = _raw
                trade.fill_px = self._apply_demo_entry_slip(trade.side, _raw) if self._demo_on() else _raw
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
        """Race native TP/SL watcher vs time-stop watcher, then finalize + announce.

        The close card is gated on a VERIFIED close (OKX shows flat AND we have
        a real close price/PnL) so we never announce a close that did not happen
        or print fabricated zeros. The _open_trade reset runs in ``finally`` so a
        hung callback cannot wedge the new-entry gate — EXCEPT when the exchange
        still shows the position open (a failed close), where we deliberately
        hold the gate so a second position can never be stacked on top.
        """
        try:
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
            await self._finalize_and_announce(trade)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("watch_open_position error (cid=%s): %s", trade.cl_ord_id, exc)
        finally:
            # Release the gate, but ONLY if it still points at THIS trade — never
            # clobber a different trade's gate. Hold it when OKX still shows this
            # position open (never stack a second position; the unverified alert
            # already prompted the operator). If a manual close already released
            # and a new trade took the slot, `_open_trade is trade` is False so we
            # correctly leave the new trade's gate intact.
            if self._open_trade is trade and not trade.extras.get("still_open"):
                self._open_trade = None

    async def _finalize_and_announce(self, trade: LiveTradeOKX) -> None:
        """Decide verified-close vs unverified, log once, and fire one card.

        Runs exactly once per trade. A close is VERIFIED only when OKX confirms
        the position is flat (re-poll) AND we have a real close price. Verified
        closes prefer OKX's authoritative realized pnl. Everything else routes
        to the unverified-close alert with the reason — never a fake fill card.
        """
        if trade.extras.get("close_finalized"):
            return
        trade.extras["close_finalized"] = True

        # Cancel any resting maker TP so it can't linger on the book and fill
        # against a later position (idempotent: a no-op once it has filled).
        await self._cancel_resting_tp(trade)

        flat = await self._confirm_flat()
        have_price = bool(trade.closed and trade.close_px > 0)
        # Confirmed flat but never priced (e.g. time-stop saw it already flat and
        # cancelled the price watcher) — make one final attempt to price it.
        if flat is True and not have_price:
            if await self._resolve_close(trade):
                have_price = bool(trade.closed and trade.close_px > 0)

        if flat is True and have_price:
            await self._reconcile_okx_pnl(trade)
            await self._update_breaker_on_close(trade)
            trade.extras["close_verified"] = True
            self._log_trade(trade)
            await self._safe_callback(
                self.on_close_confirmed or self.on_real_fill, trade, "on_close_confirmed",
            )
            return

        # ---- Flat state UNKNOWN (every poll errored — API/network outage,
        # which correlates with exactly the volatile moments a time-stop
        # fires). Treating this like "flat" used to RELEASE the entry gate on
        # a possibly-open position whose protective algos were already
        # cancelled — an unmanaged naked position while the bot kept entering.
        # Treat unknown as still-open: hold the gate, arm the recovery watcher
        # (its first action is a positions poll, so a false alarm self-clears
        # in ~10s), and say "unknown", not "flat", on the card.
        if flat is None:
            trade.extras["still_open"] = True
            trade.closed = False
            trade.close_reason = "flat_state_unknown"
            LOGGER.error(
                "CLOSE UNVERIFIED — flat state UNKNOWN (API outage) cid=%s; "
                "holding entries, recovery watcher armed", trade.cl_ord_id,
            )
            self._spawn(self._recover_stuck_position(trade), name="recover_stuck")
            self._log_trade(trade)
            await self._safe_callback(self.on_close_unverified, trade, "on_close_unverified")
            return

        # ---- Close not verified flat. Before declaring it stuck, FORCE it.
        if flat is False:
            # The maker-exit's lone taker fallback may have been rejected/throttled
            # and silently given up. Hammer a retried market close — the guaranteed
            # flatten the operator would otherwise have to do by hand.
            if await self._force_market_close(trade):
                LOGGER.info("force_close flattened the position (cid=%s) — finalizing",
                            trade.cl_ord_id)
                if not (trade.closed and trade.close_px > 0):
                    await self._resolve_close(trade)
                if trade.closed and trade.close_px > 0:
                    await self._reconcile_okx_pnl(trade)
                    await self._update_breaker_on_close(trade)
                    trade.extras["close_verified"] = True
                    self._log_trade(trade)
                    await self._safe_callback(
                        self.on_close_confirmed or self.on_real_fill,
                        trade, "on_close_confirmed",
                    )
                    return
                # Flat now but couldn't price it → flat-but-unpriced unverified.
                flat = True
            else:
                trade.extras["still_open"] = True
                trade.closed = False
                trade.close_reason = "close_failed_still_open"
                LOGGER.error(
                    "CLOSE UNVERIFIED — OKX still shows a position open after force-close "
                    "attempts (cid=%s); pausing entries until flat (recovery keeps retrying)",
                    trade.cl_ord_id,
                )
                # Self-healing hold: the recovery watcher keeps re-attempting the
                # close AND releases the gate the moment OKX reports flat. Strong
                # ref via _spawn — losing this task to GC meant a stuck REAL
                # position with nothing retrying the close.
                self._spawn(self._recover_stuck_position(trade), name="recover_stuck")

        if flat is not False:
            # flat is True (incl. forced-flat-but-unpriced) or None (all polls errored)
            trade.close_reason = trade.close_reason or "unverified_close"
            LOGGER.warning(
                "CLOSE UNVERIFIED — flat=%s close_px=%.2f (cid=%s reason=%s)",
                flat, trade.close_px, trade.cl_ord_id, trade.close_reason,
            )
        self._log_trade(trade)
        await self._safe_callback(self.on_close_unverified, trade, "on_close_unverified")

    async def _reconcile_okx_pnl(self, trade: LiveTradeOKX) -> None:
        """Best-effort: replace gross PnL with OKX's authoritative realized pnl.

        Sums ``pnl`` over all filled closing orders (opposite side, fillTime >=
        entry) — for a multi-peg maker exit that is the sum across the close
        ordIds. Leaves the existing (recomputed) gross untouched if history is
        unavailable. Because the executor holds one position at a time and this
        runs immediately after the close, the only opposite-side fills after the
        entry timestamp belong to THIS trade.
        """
        # DEMO: the realism overlay already recomputed gross from the slipped
        # close_px in _resolve_close. OKX's authoritative `pnl` ignores our
        # frictions, so adopting it here would erase them — skip in demo.
        if self._demo_on():
            trade.extras["pnl_source"] = "recomputed_demo"
            return
        exp_close_side = "sell" if trade.side == "long" else "buy"
        try:
            r = await self.client.get_orders_history(self.symbol, limit=50)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("_reconcile_okx_pnl: history fetch failed: %s", exc)
            trade.extras.setdefault("pnl_source", "recomputed")
            return
        pnl_sum = 0.0
        fee_sum = 0.0
        matched = 0
        for o in r.get("data") or []:
            if str(o.get("state", "")) != "filled":
                continue
            if str(o.get("side", "")) != exp_close_side:
                continue
            if int(o.get("fillTime") or 0) < trade.fill_ts:
                continue
            try:
                pnl_sum += float(o.get("pnl") or 0.0)
                fee_sum += abs(float(o.get("fee") or 0.0))
                matched += 1
            except (TypeError, ValueError):
                continue
        if matched:
            trade.real_gross_pnl = pnl_sum
            if fee_sum > 0:
                trade.real_close_fee = fee_sum
            trade.extras["pnl_source"] = "okx"
        else:
            trade.extras.setdefault("pnl_source", "recomputed")

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
        # Retry transient failures — a single socket blip here used to disarm
        # the time-stop entirely, leaving the trade to ride to the catastrophic
        # 6xATR taker SL (the exact loss the time-stop exists to prevent).
        r = None
        for attempt in range(4):
            try:
                r = await self.client.get_positions(self.symbol)
                break
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("time_stop: positions fetch failed (try %d/4): %s",
                               attempt + 1, exc)
                await asyncio.sleep(0.5 * (attempt + 1))
        if r is None:
            LOGGER.error("time_stop: positions unavailable after retries — "
                         "closing blind via maker exit (reduceOnly is safe on flat)")
            async with self._close_lock:
                if trade.close_reason == "manual" or trade.closed:
                    return
                await self._cancel_resting_tp(trade)
                await cancel_attached_algos(self.client, self.symbol)
                await self._maker_close(trade, reason="time_stop", abs_qty=trade.sz_contracts)
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
        # Serialize against an operator manual close. Re-check flat INSIDE the
        # lock: the operator may have flattened it (or tagged it "manual") in the
        # window between the qty read above and acquiring the lock.
        async with self._close_lock:
            if trade.close_reason == "manual" or trade.closed:
                LOGGER.info("time_stop: superseded by manual close (cid=%s)", trade.cl_ord_id)
                return
            try:
                r2 = await self.client.get_positions(self.symbol)
                if self._position_qty(r2) == 0.0:
                    LOGGER.info("time_stop: flat after lock (manual close?) cid=%s",
                                trade.cl_ord_id)
                    return
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("time_stop: re-check positions failed: %s", exc)
            await self._cancel_resting_tp(trade)
            await cancel_attached_algos(self.client, self.symbol)
            await self._maker_close(trade, reason="time_stop", abs_qty=abs(qty))

    async def _maker_close(
        self, trade: LiveTradeOKX, *, reason: str, abs_qty: float,
    ) -> None:
        """Close the position. With maker_exit disabled this is a straight market
        (taker) close — chosen for reliability; with it enabled, the maker-first
        re-peg loop runs.
        """
        if not self.maker_exit_enabled:
            # Direct market (taker) close — the configured exit policy. autoCxl
            # sweeps any lingering close order so the market close can't be
            # rejected by a conflict. A bounced order is caught by the finalizer's
            # retried force-close backstop, so this can't strand the position.
            LOGGER.info("maker_exit disabled — closing at market (taker) cid=%s",
                        trade.cl_ord_id)
            try:
                pos_side_param = (
                    trade.side if self.pos_mode == "hedge" else None
                )
                resp = await self.client.close_position_market(
                    self.symbol, mgn_mode=self.mgn_mode, pos_side=pos_side_param,
                    auto_cxl=True,
                )
                cdata = (resp.get("data") or [{}])[0]
                cscode = str(cdata.get("sCode", "0"))
                if cscode != "0":
                    LOGGER.error("close_position_market rejected sCode=%s sMsg=%r (cid=%s)",
                                 cscode, cdata.get("sMsg"), trade.cl_ord_id)
                trade.exit_path = "taker_market"
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("close_position_market failed: %s", exc)
            # Stamp the real exit reason (e.g. "time_stop") BEFORE resolving, so
            # the tp/sl proximity matcher in _resolve_close doesn't relabel a
            # time-stop taker loss as "tp" just because the drifted exit price
            # landed nearer tp_px than the (much wider) sl_px.
            if not trade.close_reason:
                trade.close_reason = reason
            await self._resolve_close(trade)
            return

        intended_px = trade.tp_px if reason == "tp" else trade.fill_px
        pos_side_param = trade.side if self.pos_mode == "hedge" else None
        # Stamp the reason BEFORE the maker exit runs: _watch_position keeps
        # polling during the re-peg loop, and when a peg fills it can resolve
        # the close first — with an empty reason the proximity matcher used to
        # relabel a time-stop scratch as "sl", advancing the breaker's loss
        # streak (1h entry pause) off pure noise and mislabeling the Telegram
        # card. With the reason pre-stamped, _resolve_close preserves it.
        if not trade.close_reason:
            trade.close_reason = reason
        result: ExitResult = await close_position_maker(
            self.client,
            symbol=self.symbol,
            position_side=trade.side,
            sz_contracts=abs_qty,
            tick_sz=self._tick_sz,
            lot_sz=self._lot_sz,
            ct_val=self._ct_val,
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
            # Gross over the qty that ACTUALLY closed at this price — a partial
            # maker fill's residual closes later (force-close) at a different
            # price; computing over full size fed wrong numbers to the breaker
            # (and demo skips the OKX-pnl reconciliation that fixed it in live).
            closed_qty = min(result.filled_qty, trade.sz_contracts)
            if trade.side == "long":
                trade.real_gross_pnl = (
                    trade.close_px - trade.fill_px
                ) * closed_qty * self._ct_val
            else:
                trade.real_gross_pnl = (
                    trade.fill_px - trade.close_px
                ) * closed_qty * self._ct_val
            if result.error == "residual_unfilled":
                trade.extras["partial_close_qty"] = closed_qty
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
            # Nothing filled. The native TP/SL were already cancelled before
            # this maker exit ran, so the ONLY remaining actor is the
            # finalizer's force-close — which is still flattening for THIS
            # reason (e.g. time_stop). Keep the pre-stamped reason so the
            # forced close is labelled honestly; do NOT blank it (blanking
            # let the proximity matcher relabel a losing time-stop as "tp").
            LOGGER.error(
                "maker_exit: nothing filled (cid=%s err=%s) — leaving position "
                "open for force-close (reason stays %s)",
                trade.cl_ord_id, result.error, trade.close_reason or reason,
            )

    async def _watch_position(self, trade: LiveTradeOKX) -> None:
        """Poll /positions until the position closes, then resolve its price/PnL.

        Two hardenings over the naive open->flat edge watcher:
          * Fast poll for the first ``fast_poll_window_s`` after the fill so a
            quick TP that closes inside one normal poll is still seen — without
            this a fast fill produced ZERO close cards and pinned the gate.
          * A never-seen-open position is treated as closed only after
            ``flat_confirm_polls`` consecutive flat reads (guards positions-lag
            right after entry). If a close can't be priced after
            ``flat_giveup_polls`` confirmed-flat polls we stop so the finalizer
            can route it to the unverified path instead of hanging for hours.
        """
        start = time.time()
        last_seen_open = False
        consecutive_flat = 0
        while time.time() - start < self.position_watch_max_s:
            elapsed = time.time() - start
            sleep_s = 0.5 if elapsed < self.fast_poll_window_s else self.position_poll_s
            await asyncio.sleep(sleep_s)
            try:
                r = await self.client.get_positions(self.symbol)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("get_positions failed: %s", exc)
                continue
            qty = self._position_qty(r)
            if qty != 0.0:
                last_seen_open = True
                consecutive_flat = 0
                continue
            # qty == 0
            consecutive_flat += 1
            # Don't act on a single flat read before we've ever seen the position
            # open — that is usually positions-lag right after entry, not a close.
            if not last_seen_open and consecutive_flat < self.flat_confirm_polls:
                continue
            # Looks closed — try to price the close from order history.
            if await self._resolve_close(trade):
                return
            # Couldn't price it. If flat has persisted long enough, give up so
            # the finalizer routes to the unverified-close alert (no hang).
            if consecutive_flat >= self.flat_giveup_polls:
                trade.close_reason = trade.close_reason or "unmatched_close"
                return
            # else: history lag or a false flat — keep watching.
        LOGGER.warning("position watch timed out for cid=%s — final resolve attempt",
                       trade.cl_ord_id)
        trade.close_reason = trade.close_reason or "watch_timeout"
        await self._resolve_close(trade)

    async def _resolve_close(self, trade: LiveTradeOKX) -> bool:
        """Match the closing fill in order history for exit price/fee/pnl.

        Returns True ONLY when a real closing fill is matched (a priced close).
        On no-match or a fetch error it returns False WITHOUT marking the trade
        closed — the caller decides whether to keep watching or route to the
        unverified-close path. (Previously this set ``closed=True`` with a zero
        price on failure, which surfaced as a fabricated "$0.00 REAL FILL".)

        OKX net (one-way) mode does NOT flag the maker-exit / TP / time-stop
        close as reduceOnly, so we match purely on: filled, opposite side, on or
        after our entry fill. The executor holds one position at a time and
        gates new entries until this trade is finalized, so the newest such
        order IS our close. order-history can lag /positions by a few hundred ms,
        hence the bounded retries.
        """
        okx_pnl_str = None
        for attempt in range(3):
            try:
                r = await self.client.get_orders_history(self.symbol, limit=30)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("orders-history fetch failed: %s", exc)
                if attempt < 2:
                    await asyncio.sleep(0.5)
                continue

            for o in r.get("data") or []:
                if str(o.get("state", "")) != "filled":
                    continue
                exp_close_side = "sell" if trade.side == "long" else "buy"
                if str(o.get("side", "")) != exp_close_side:
                    continue
                close_ts = int(o.get("fillTime") or 0)
                if close_ts < trade.fill_ts:
                    continue
                trade.close_px = float(o.get("avgPx") or 0.0)
                trade.close_ts = close_ts
                trade.real_close_fee = abs(float(o.get("fee") or 0.0))
                okx_pnl_str = o.get("pnl")
                # Identify TP vs SL by proximity (tolerant of slippage) — but
                # only when no explicit exit reason is already set. An operator
                # "manual" flatten, or a strategy-driven "time_stop"/"trend_break"
                # close, carries the true reason; preserve it so the card/stats
                # don't mislabel it as a TP/SL price exit. Proximity only guesses
                # for genuine native TP/SL fills, which arrive with no reason.
                if trade.close_reason not in (
                    "manual", "time_stop", "trend_break",
                    "watch_timeout", "unmatched_close",
                ):
                    # PnL-aware label. A real TP is a profit by construction
                    # (the limit sits at a fee-cleared profit); an SL is a
                    # loss. So the sign of realized PnL is the GROUND TRUTH —
                    # proximity only breaks ties within the correct sign. This
                    # stops a forced/drifted close from being called "TP HIT"
                    # while it lost money (or "SL" while it won).
                    if trade.side == "long":
                        gross_now = (trade.close_px - trade.fill_px)
                    else:
                        gross_now = (trade.fill_px - trade.close_px)
                    if gross_now > 0:
                        trade.close_reason = "tp"
                    elif gross_now < 0:
                        trade.close_reason = "sl"
                    else:
                        d_tp = abs(trade.close_px - trade.tp_px)
                        d_sl = abs(trade.close_px - trade.sl_px)
                        trade.close_reason = "tp" if d_tp < d_sl else "sl"
                break

            if trade.close_px != 0.0:
                break
            if attempt < 2:
                await asyncio.sleep(0.5)   # let order-history catch up, then refetch

        if trade.close_px == 0.0:
            LOGGER.warning("could not match close fill for cid=%s (will retry/verify)",
                           trade.cl_ord_id)
            return False

        # DEMO: SL/taker exits slip adversely through the trigger; native limit-TPs
        # occasionally miss and give back profit (degrading to a worse taker exit).
        # Applied AFTER the reason is decided (won't reclassify) and BEFORE the
        # gross recompute below. Idempotent via the demo_exit_adj flag.
        if self._demo_on() and not trade.extras.get("demo_exit_adj"):
            trade.extras["demo_exit_adj"] = True
            trade.extras["demo_raw_close_px"] = trade.close_px
            r = self.demo_realism
            if trade.close_reason == "sl":
                # SL market exit slips worse (it already carries a taker fee in demo;
                # don't uplift the fee or it double-counts — slippage only).
                bump = trade.close_px * (r.sl_slip_bps / 10_000.0)
                trade.close_px = _round_to_tick(
                    trade.close_px - bump if trade.side == "long" else trade.close_px + bump,
                    self._tick_sz)
            elif trade.close_reason == "tp":
                if self._demo_rng.random() > r.tp_fill_prob:
                    # TP "missed" → chased to a worse exit + maker→taker fee.
                    gb = trade.close_px * (r.tp_miss_giveback_bps / 10_000.0)
                    trade.close_px = _round_to_tick(
                        trade.close_px - gb if trade.side == "long" else trade.close_px + gb,
                        self._tick_sz)
                    trade.real_close_fee *= r.taker_exit_fee_mult
                    trade.extras["demo_tp_missed"] = True
            if trade.close_px <= 0:  # never drive price non-positive
                trade.close_px = trade.extras["demo_raw_close_px"]

        # Realized gross PnL — prefer OKX's own authoritative `pnl`; recompute
        # from fill/close prices if it's absent OR the literal "0"/"" string
        # (don't silently log $0 on a real close). In DEMO, always recompute so
        # the slipped close_px above flows into PnL (the demo `pnl` ignores it).
        gross = None
        if not self._demo_on() and okx_pnl_str not in (None, "", "0"):
            try:
                gross = float(okx_pnl_str)
            except (TypeError, ValueError):
                gross = None
        if gross is None:
            if trade.side == "long":
                gross = (trade.close_px - trade.fill_px) * trade.sz_contracts * self._ct_val
            else:
                gross = (trade.fill_px - trade.close_px) * trade.sz_contracts * self._ct_val
        trade.real_gross_pnl = gross
        trade.closed = True
        LOGGER.info(
            "CLOSE cid=%s reason=%s  fill_px=%.2f -> close_px=%.2f  gross=%+.4f  fees=%.6f+%.6f",
            trade.cl_ord_id, trade.close_reason, trade.fill_px, trade.close_px,
            trade.real_gross_pnl, trade.real_open_fee, trade.real_close_fee,
        )
        return True

    def _log_trade(self, trade: LiveTradeOKX) -> None:
        rec = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            # Real money vs OKX demo (x-simulated-trading). Demo fills land in
            # the same live_trades/ log, so tag each record and let readers
            # (Telegram /status /fees /trades) count only real-money trades.
            "live": (not getattr(self.client, "_simulated", False)) and not self.dry_run,  # noqa: SLF001
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
