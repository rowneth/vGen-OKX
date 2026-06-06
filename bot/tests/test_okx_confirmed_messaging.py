"""Tests for OKX-confirmed Telegram messaging in LiveVolumeExecutorOKX.

These lock in the contract that drives the "real OKX panel" rewire:
  * the entry card fires exactly once, and ONLY on a confirmed fill;
  * a no-fill entry is silent by default (no entry card);
  * the close card (on_close_confirmed) fires ONLY when OKX confirms the
    position is flat AND the close is priced;
  * an unpriced-but-flat close, or a close that left the position open, routes
    to on_close_unverified — never a fabricated "$0.00 REAL FILL";
  * a still-open position holds the new-entry gate (never stacks);
  * confirmation callbacks are time-boxed (a hung callback can't wedge the gate);
  * realized PnL prefers OKX's authoritative `pnl`.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from execution.live_volume_executor_okx import LiveTradeOKX, LiveVolumeExecutorOKX


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# Scriptable fake OKX client (only the methods the close/confirm paths touch)
# --------------------------------------------------------------------------
class FakeClient:
    def __init__(
        self,
        *,
        positions: Any,
        history: Optional[Dict[str, Any]] = None,
        simulated: bool = False,
    ) -> None:
        # positions: a fixed response dict, or a list used as a per-call queue
        # (the last element repeats once exhausted).
        self._positions = positions
        self._history = history or {"data": []}
        self._simulated = simulated
        self.calls = {"get_positions": 0, "get_orders_history": 0, "close_position_market": 0}

    async def get_positions(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        self.calls["get_positions"] += 1
        if isinstance(self._positions, list):
            idx = min(self.calls["get_positions"] - 1, len(self._positions) - 1)
            return self._positions[idx]
        return self._positions

    async def get_orders_history(self, symbol: Optional[str] = None, **kw: Any) -> Dict[str, Any]:
        self.calls["get_orders_history"] += 1
        return self._history

    async def close_position_market(self, *a: Any, **k: Any) -> Dict[str, Any]:
        self.calls["close_position_market"] += 1
        return {"data": [{"sCode": "0"}]}


def _flat() -> Dict[str, Any]:
    return {"data": [{"pos": "0"}]}


def _open() -> Dict[str, Any]:
    return {"data": [{"pos": "0.03", "posSide": "net", "upl": "0.10",
                      "avgPx": "100.0", "markPx": "100.1"}]}


def _close_hist_long(pnl: str = "0.50", px: str = "101.0", fee: str = "-0.01",
                     fill_time: str = "2000") -> Dict[str, Any]:
    """Order-history with a closing SELL fill (closes a long), after entry."""
    return {"data": [{"state": "filled", "side": "sell", "fillTime": fill_time,
                      "avgPx": px, "fee": fee, "pnl": pnl}]}


def _make_trade(side: str = "long", closed: bool = True,
                close_px: float = 101.0, fill_ts: int = 1000) -> LiveTradeOKX:
    t = LiveTradeOKX(
        cl_ord_id="vfoxtest", side=side, entry_px_req=100.0, notional_usd=50.0,
        sz_contracts=0.03, tp_px=101.0, sl_px=98.0, tp_bps=8.0, sl_bps=50.0,
        placed_at=0.0,
    )
    t.ord_id = "ord1"
    t.filled = True
    t.fill_px = 100.0
    t.fill_ts = fill_ts
    t.real_open_fee = 0.01
    t.closed = closed
    t.close_px = close_px
    return t


def _make_executor(client: FakeClient, tmp_path):
    ex = LiveVolumeExecutorOKX(client=client, log_dir=tmp_path)
    # keep close-confirmation snappy in tests (real default is 12s / 10s)
    ex.flat_timeout_s = 0.4
    ex.recover_poll_s = 0.05
    ex.force_close_base_delay_s = 0.0  # no backoff sleeps in tests
    fired = {"entry": 0, "abandon": 0, "confirmed": 0, "unverified": 0}

    async def on_entry(_t):
        fired["entry"] += 1

    async def on_conf(_t):
        fired["confirmed"] += 1

    async def on_unver(_t):
        fired["unverified"] += 1

    ex.on_entry_filled = on_entry
    ex.on_close_confirmed = on_conf
    ex.on_close_unverified = on_unver
    # on_entry_abandoned intentionally left None (silent no-fill default)
    return ex, fired


# --------------------------------------------------------------------------
# Verified close → on_close_confirmed
# --------------------------------------------------------------------------
def test_verified_close_fires_confirmed_once(tmp_path):
    client = FakeClient(positions=_flat(), history=_close_hist_long())
    ex, fired = _make_executor(client, tmp_path)
    trade = _make_trade()
    ex._open_trade = trade

    _run(ex._finalize_and_announce(trade))
    assert fired["confirmed"] == 1
    assert fired["unverified"] == 0
    assert trade.extras.get("close_verified") is True

    # Idempotent: a second finalize must not re-announce.
    _run(ex._finalize_and_announce(trade))
    assert fired["confirmed"] == 1


def test_verified_close_prefers_okx_pnl(tmp_path):
    client = FakeClient(positions=_flat(), history=_close_hist_long(pnl="0.75"))
    ex, fired = _make_executor(client, tmp_path)
    trade = _make_trade()
    trade.real_gross_pnl = 0.50  # a recomputed/price-derived value
    ex._open_trade = trade

    _run(ex._finalize_and_announce(trade))
    assert fired["confirmed"] == 1
    assert abs(trade.real_gross_pnl - 0.75) < 1e-9
    assert trade.extras.get("pnl_source") == "okx"


# --------------------------------------------------------------------------
# Flat but unpriced → on_close_unverified (no fabricated zero card)
# --------------------------------------------------------------------------
def test_flat_but_unpriced_routes_unverified(tmp_path):
    client = FakeClient(positions=_flat(), history={"data": []})
    ex, fired = _make_executor(client, tmp_path)
    trade = _make_trade(closed=False, close_px=0.0)
    ex._open_trade = trade

    _run(ex._finalize_and_announce(trade))
    assert fired["confirmed"] == 0
    assert fired["unverified"] == 1
    assert trade.extras.get("still_open") is not True
    assert trade.extras.get("close_verified") is not True


# --------------------------------------------------------------------------
# Still open on OKX → unverified, gate held, "closed" never claimed
# --------------------------------------------------------------------------
def test_still_open_routes_unverified_and_marks_flag(tmp_path):
    client = FakeClient(positions=_open(), history=_close_hist_long())
    ex, fired = _make_executor(client, tmp_path)
    trade = _make_trade(closed=True, close_px=101.0)  # we THINK we closed
    ex._open_trade = trade

    _run(ex._finalize_and_announce(trade))
    assert fired["confirmed"] == 0
    assert fired["unverified"] == 1
    assert trade.extras.get("still_open") is True
    assert trade.closed is False


class _FlipOnCloseClient(FakeClient):
    """Reads the position OPEN until close_position_market is called, then FLAT.

    Models the real backstop scenario: the maker-exit's lone taker bounced
    (position still open), and a retried market close actually flattens it.
    """

    def __init__(self, **kw: Any) -> None:
        super().__init__(positions=_open(), **kw)
        self._flat_now = False

    async def get_positions(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        self.calls["get_positions"] += 1
        return _flat() if self._flat_now else _open()

    async def close_position_market(self, *a: Any, **k: Any) -> Dict[str, Any]:
        self.calls["close_position_market"] += 1
        self._flat_now = True
        return {"data": [{"sCode": "0"}]}


def test_force_market_close_retries_then_flat(tmp_path):
    # open, open, flat -> succeeds on the 3rd positions read
    client = FakeClient(positions=[_open(), _open(), _flat()])
    ex, _ = _make_executor(client, tmp_path)
    assert _run(ex._force_market_close(_make_trade(), attempts=4)) is True
    assert client.calls["close_position_market"] >= 1


def test_force_market_close_gives_up_when_never_flat(tmp_path):
    client = FakeClient(positions=_open())  # never flat
    ex, _ = _make_executor(client, tmp_path)
    assert _run(ex._force_market_close(_make_trade(), attempts=3)) is False
    assert client.calls["close_position_market"] == 3


def test_finalize_force_closes_then_confirms(tmp_path):
    # _confirm_flat reads open (False); the backstop market close flattens it;
    # then it prices from history and routes to CONFIRMED — not the stuck alert.
    client = _FlipOnCloseClient(history=_close_hist_long())
    ex, fired = _make_executor(client, tmp_path)
    ex.flat_timeout_s = 0.05
    trade = _make_trade(closed=False, close_px=0.0)
    ex._open_trade = trade
    _run(ex._finalize_and_announce(trade))
    assert client.calls["close_position_market"] >= 1
    assert fired["confirmed"] == 1
    assert fired["unverified"] == 0
    assert trade.extras.get("forced_market_close") is True


def test_finalize_force_close_gives_up_routes_still_open(tmp_path):
    # Backstop tried but the position stays open -> still_open alert (as before).
    client = FakeClient(positions=_open())
    ex, fired = _make_executor(client, tmp_path)
    ex.flat_timeout_s = 0.05
    trade = _make_trade(closed=True, close_px=101.0)
    ex._open_trade = trade
    _run(ex._finalize_and_announce(trade))
    assert client.calls["close_position_market"] >= 1   # backstop attempted
    assert fired["unverified"] == 1
    assert fired["confirmed"] == 0
    assert trade.extras.get("still_open") is True


def test_maker_close_disabled_uses_taker_market_with_autocxl(tmp_path):
    # Exit policy = always taker: maker_exit disabled -> straight market close,
    # with autoCxl so a lingering order can't reject it.
    captured: Dict[str, Any] = {}

    class _Cap(FakeClient):
        async def close_position_market(self, symbol: Any, **k: Any) -> Dict[str, Any]:
            self.calls["close_position_market"] += 1
            captured.update(k)
            return {"data": [{"sCode": "0"}]}

    client = _Cap(positions=_flat(), history=_close_hist_long())
    ex, _ = _make_executor(client, tmp_path)
    assert ex.maker_exit_enabled is False  # dataclass default; live config sets it
    trade = _make_trade(closed=False, close_px=0.0)
    _run(ex._maker_close(trade, reason="time_stop", abs_qty=0.03))
    assert client.calls["close_position_market"] == 1
    assert captured.get("auto_cxl") is True
    assert trade.exit_path == "taker_market"


def test_recovery_reattempts_market_close_until_flat(tmp_path):
    # The recovery watcher must actively RE-CLOSE, not just wait.
    client = FakeClient(positions=[_open(), _open(), _flat()])
    ex, _ = _make_executor(client, tmp_path)
    trade = _make_trade()
    ex._open_trade = trade
    _run(ex._recover_stuck_position(trade))
    assert ex._open_trade is None
    assert client.calls["close_position_market"] >= 1


def test_watch_open_position_holds_gate_when_still_open(tmp_path):
    client = FakeClient(positions=_open())
    ex, _ = _make_executor(client, tmp_path)
    ex.time_stop_enabled = False
    trade = _make_trade()
    ex._open_trade = trade

    async def _noop(_t):
        return

    async def _finalize_still_open(t):
        t.extras["still_open"] = True
        t.closed = False

    ex._watch_position = _noop
    ex._finalize_and_announce = _finalize_still_open

    _run(ex._watch_open_position(trade))
    assert ex._open_trade is trade  # gate HELD — no second position can stack


def test_watch_open_position_releases_gate_on_normal_close(tmp_path):
    client = FakeClient(positions=_flat())
    ex, _ = _make_executor(client, tmp_path)
    ex.time_stop_enabled = False
    trade = _make_trade()
    ex._open_trade = trade

    async def _noop(_t):
        return

    async def _finalize_ok(t):
        t.extras["close_verified"] = True

    ex._watch_position = _noop
    ex._finalize_and_announce = _finalize_ok

    _run(ex._watch_open_position(trade))
    assert ex._open_trade is None  # gate released for the next signal


# --------------------------------------------------------------------------
# Entry card: fires once on confirmed fill
# --------------------------------------------------------------------------
def test_entry_filled_fires_once(tmp_path):
    client = FakeClient(positions=_flat())
    ex, fired = _make_executor(client, tmp_path)
    trade = _make_trade()

    _run(ex._fire_entry_filled(trade))
    _run(ex._fire_entry_filled(trade))  # idempotent
    assert fired["entry"] == 1
    assert trade.extras.get("entry_card_sent") is True


# --------------------------------------------------------------------------
# No-fill is silent by default; fires once when explicitly wired
# --------------------------------------------------------------------------
def test_abandon_silent_by_default(tmp_path):
    client = FakeClient(positions=_flat())
    ex, fired = _make_executor(client, tmp_path)  # on_entry_abandoned is None
    trade = _make_trade(closed=False, close_px=0.0)

    _run(ex._fire_entry_abandoned(trade, "repeg_exhausted"))
    assert fired["abandon"] == 0
    assert trade.cancel_reason == "repeg_exhausted"


def test_abandon_fires_once_when_wired(tmp_path):
    client = FakeClient(positions=_flat())
    ex, fired = _make_executor(client, tmp_path)

    async def on_ab(_t):
        fired["abandon"] += 1

    ex.on_entry_abandoned = on_ab
    trade = _make_trade()

    _run(ex._fire_entry_abandoned(trade, "r1"))
    _run(ex._fire_entry_abandoned(trade, "r2"))
    assert fired["abandon"] == 1


# --------------------------------------------------------------------------
# _confirm_flat semantics
# --------------------------------------------------------------------------
def test_confirm_flat_true_false_none(tmp_path):
    # flat -> True immediately
    ex_flat, _ = _make_executor(FakeClient(positions=_flat()), tmp_path)
    assert _run(ex_flat._confirm_flat(timeout_s=0.3, poll_s=0.05)) is True

    # persistently open -> False after the timeout window
    ex_open, _ = _make_executor(FakeClient(positions=_open()), tmp_path)
    assert _run(ex_open._confirm_flat(timeout_s=0.3, poll_s=0.05)) is False

    # all polls error -> None
    class ErrClient(FakeClient):
        async def get_positions(self, symbol=None):
            raise RuntimeError("boom")

    ex_err, _ = _make_executor(ErrClient(positions=_flat()), tmp_path)
    assert _run(ex_err._confirm_flat(timeout_s=0.3, poll_s=0.05)) is None


def test_confirm_flat_tolerates_post_close_lag(tmp_path):
    # /positions shows the position open on the first 2 polls (lag), then flat.
    # The new poll-until-flat behaviour must return True, NOT a false "still open"
    # — this is the exact bug that wedged the live gate for 3 hours.
    client = FakeClient(positions=[_open(), _open(), _flat(), _flat()])
    ex, _ = _make_executor(client, tmp_path)
    assert _run(ex._confirm_flat(timeout_s=2.0, poll_s=0.05)) is True


def test_recovery_releases_gate_when_position_clears(tmp_path):
    # A held (still-open) trade must auto-release once OKX reads flat.
    client = FakeClient(positions=[_open(), _open(), _flat()])
    ex, _ = _make_executor(client, tmp_path)
    trade = _make_trade()
    ex._open_trade = trade
    _run(ex._recover_stuck_position(trade))
    assert ex._open_trade is None  # gate released, trading resumes


# --------------------------------------------------------------------------
# Sizing tracks the REAL OKX wallet, never a hardcoded capital
# --------------------------------------------------------------------------
class _BalClient(FakeClient):
    def __init__(self, *, usdt_eq, **kw):
        super().__init__(positions=_flat(), **kw)
        self._usdt_eq = usdt_eq
    async def get_balance(self, ccy=None):
        return {"data": [{"details": [{"ccy": "USDT", "eq": str(self._usdt_eq)}]}]}


def test_resize_uses_real_wallet_not_sim_notional(tmp_path):
    ex, _ = _make_executor(_BalClient(usdt_eq=1500.0), tmp_path)
    ex.margin_frac = 0.03
    # SIM thought equity was $500 -> notional $1500 at lev 100; real wallet is $1500
    payload = {"leverage": 100.0, "notional": 1500.0}
    notional = _run(ex._resize_on_real_balance(payload, 1500.0))
    assert abs(notional - 1500.0 * 0.03 * 100.0) < 1e-6   # = $4500, off the REAL wallet
    assert notional != 1500.0


def test_resize_falls_back_when_balance_unavailable(tmp_path):
    class _NoBal(FakeClient):
        async def get_balance(self, ccy=None):
            raise RuntimeError("api down")
    ex, _ = _make_executor(_NoBal(positions=_flat()), tmp_path)
    ex.margin_frac = 0.03
    # balance fetch fails -> keep the SIM notional, never invent a number
    notional = _run(ex._resize_on_real_balance({"leverage": 100.0}, 153.0))
    assert notional == 153.0


def test_resize_disabled_when_margin_frac_zero(tmp_path):
    ex, _ = _make_executor(_BalClient(usdt_eq=1500.0), tmp_path)
    ex.margin_frac = 0.0  # disabled -> SIM notional passes through
    notional = _run(ex._resize_on_real_balance({"leverage": 100.0}, 153.0))
    assert notional == 153.0


# --------------------------------------------------------------------------
# A hung confirmation callback must not propagate / wedge the caller
# --------------------------------------------------------------------------
def test_safe_callback_timeout_is_swallowed(tmp_path):
    ex, _ = _make_executor(FakeClient(positions=_flat()), tmp_path)
    ex.callback_timeout_s = 0.05

    async def _slow(_t):
        await asyncio.sleep(5)

    # Must return promptly without raising.
    _run(ex._safe_callback(_slow, _make_trade(), "slow"))


def test_safe_callback_swallows_exceptions(tmp_path):
    ex, _ = _make_executor(FakeClient(positions=_flat()), tmp_path)

    async def _boom(_t):
        raise ValueError("kaboom")

    _run(ex._safe_callback(_boom, _make_trade(), "boom"))  # no raise


# --------------------------------------------------------------------------
# Fast-fill: resolve on a confirmed flat even if we never saw it open
# --------------------------------------------------------------------------
def test_watch_position_resolves_on_confirmed_flat(tmp_path):
    client = FakeClient(positions=_flat(), history=_close_hist_long())
    ex, _ = _make_executor(client, tmp_path)
    ex.fast_poll_window_s = 100.0   # always fast-poll
    ex.flat_confirm_polls = 1       # a fast fill that closed before first poll
    ex.position_watch_max_s = 5.0
    trade = _make_trade(closed=False, close_px=0.0)

    _run(ex._watch_position(trade))
    assert trade.closed is True
    assert trade.close_px == 101.0


# --------------------------------------------------------------------------
# _resolve_close returns False (not a fake close) when nothing matches
# --------------------------------------------------------------------------
def test_resolve_close_no_match_returns_false_without_closing(tmp_path):
    client = FakeClient(positions=_flat(), history={"data": []})
    ex, _ = _make_executor(client, tmp_path)
    trade = _make_trade(closed=False, close_px=0.0)

    ok = _run(ex._resolve_close(trade))
    assert ok is False
    assert trade.closed is False
    assert trade.close_px == 0.0
    # No fabricated close_reason of "history_fetch_failed"/"unmatched_close" set
    assert trade.close_reason in ("", None)
