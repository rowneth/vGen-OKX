"""Unit tests for the RESTING maker-TP lifecycle (Feature 3).

The attached trigger-TP fires a limit only on touch; on a gap that limit is
already marketable and fills TAKER. The resting-TP path instead attaches ONLY
the SL to the entry and rests a reduce-only POST_ONLY limit at tp_px the moment
the entry fills, so a winning close fills MAKER by construction. These tests
exercise the three executor helpers that implement it:

  * _attach_tp_args   — what TP (if any) rides on the entry order
  * _place_resting_tp — rest the post_only limit after fill (incl. would-cross)
  * _cancel_resting_tp — idempotent, race-safe cancel on every non-TP exit
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from execution.live_volume_executor_okx import (
    LiveTradeOKX,
    LiveVolumeExecutorOKX,
)


class StubOKX:
    """Scriptable OKX stub: records placements / cancels / market closes."""

    def __init__(self, place_results: List[Dict[str, Any]] | None = None) -> None:
        self.place_results: List[Dict[str, Any]] = place_results or []
        self.placed_orders: List[Dict[str, Any]] = []
        self.cancel_calls: List[Dict[str, Any]] = []
        self.market_close_calls: List[Dict[str, Any]] = []
        self._simulated = False

    async def place_order(self, **kwargs):
        self.placed_orders.append(kwargs)
        if self.place_results:
            return self.place_results.pop(0)
        return {"data": [{"sCode": "0", "ordId": f"o{len(self.placed_orders)}"}]}

    async def cancel_order(self, symbol, *, ord_id=None, client_oid=None):
        self.cancel_calls.append({"symbol": symbol, "ord_id": ord_id})
        return {"code": "0", "data": []}

    async def close_position_market(self, symbol, **kwargs):
        self.market_close_calls.append({"symbol": symbol, **kwargs})
        return {"code": "0", "data": []}


def _run(coro):
    return asyncio.run(coro)


def _executor(stub: StubOKX, *, resting_tp_enabled: bool = True,
              pos_mode: str = "net") -> LiveVolumeExecutorOKX:
    ex = LiveVolumeExecutorOKX(
        client=stub, symbol="BTC_USDT", pos_mode=pos_mode,
        resting_tp_enabled=resting_tp_enabled,
    )
    # prepare() (which hits the API) is skipped in unit tests — set the
    # instrument precision the helpers need directly.
    ex._tick_sz = 0.1
    ex._lot_sz = 0.01
    return ex


def _trade(side: str = "long", tp_px: float = 101.0) -> LiveTradeOKX:
    return LiveTradeOKX(
        cl_ord_id="vf_test_1", side=side, entry_px_req=100.0,
        notional_usd=1500.0, sz_contracts=1.0, tp_px=tp_px, sl_px=99.5,
        tp_bps=8.0, sl_bps=50.0, placed_at=0.0, filled=True, fill_px=100.0,
    )


# ---------------------------------------------------------------------------
# _attach_tp_args
# ---------------------------------------------------------------------------

def test_attach_tp_args_omits_trigger_when_resting_on():
    """resting_tp on → entry carries NO trigger TP (only the SL rides along)."""
    ex = _executor(StubOKX(), resting_tp_enabled=True)
    args = ex._attach_tp_args(101.0)
    assert args == {"tp_trigger_px": None, "tp_ord_px": None}


def test_attach_tp_args_legacy_trigger_when_resting_off():
    """resting_tp off → legacy attached trigger-limit TP at tp_px."""
    ex = _executor(StubOKX(), resting_tp_enabled=False)
    args = ex._attach_tp_args(101.0)
    assert args == {"tp_trigger_px": "101.0", "tp_ord_px": "101.0"}


# ---------------------------------------------------------------------------
# _place_resting_tp
# ---------------------------------------------------------------------------

def test_place_resting_tp_rests_reduce_only_post_only_maker():
    """Happy path: a reduce-only post_only limit is rested at tp_px and its
    ordId is stored for later cancellation."""
    stub = StubOKX([{"data": [{"sCode": "0", "ordId": "TP123"}]}])
    ex = _executor(stub)
    trade = _trade(side="long", tp_px=101.0)
    _run(ex._place_resting_tp(trade))

    assert len(stub.placed_orders) == 1
    o = stub.placed_orders[0]
    assert o["side"] == "sell"            # closing a long => sell
    assert o["ord_type"] == "post_only"   # MAKER by construction
    assert o["reduce_only"] is True
    assert o["px"] == "101.0"             # rounded to tick
    assert o["pos_side"] is None          # net (one-way) mode
    assert trade.extras["resting_tp_ord_id"] == "TP123"


def test_place_resting_tp_short_closes_with_buy():
    """A short position's resting TP is a BUY (and hedge mode carries pos_side)."""
    stub = StubOKX([{"data": [{"sCode": "0", "ordId": "TP9"}]}])
    ex = _executor(stub, pos_mode="hedge")
    trade = _trade(side="short", tp_px=99.0)
    _run(ex._place_resting_tp(trade))
    o = stub.placed_orders[0]
    assert o["side"] == "buy"
    assert o["pos_side"] == "short"


def test_place_resting_tp_noop_when_disabled():
    """resting_tp off → never touches the book (legacy trigger TP handles it)."""
    stub = StubOKX()
    ex = _executor(stub, resting_tp_enabled=False)
    _run(ex._place_resting_tp(_trade()))
    assert stub.placed_orders == []


def test_place_resting_tp_noop_in_dry_run():
    stub = StubOKX()
    ex = _executor(stub)
    ex.dry_run = True
    _run(ex._place_resting_tp(_trade()))
    assert stub.placed_orders == []


def test_place_resting_tp_would_cross_takes_profit_at_market():
    """51280 (post_only would cross) means price has ALREADY gapped past the TP
    → we're in profit → capture it at market rather than give it back. No
    resting ordId is stored (nothing is resting)."""
    stub = StubOKX([{"data": [{"sCode": "51280", "sMsg": "would cross"}]}])
    ex = _executor(stub)
    trade = _trade(side="long", tp_px=101.0)
    _run(ex._place_resting_tp(trade))
    assert len(stub.market_close_calls) == 1
    assert "resting_tp_ord_id" not in trade.extras


def test_place_resting_tp_other_reject_leaves_position_for_time_stop():
    """Any other reject → log + leave it; the time-stop still exits. No ordId,
    no market close (the position is NOT in profit)."""
    stub = StubOKX([{"data": [{"sCode": "51000", "sMsg": "param error"}]}])
    ex = _executor(stub)
    trade = _trade()
    _run(ex._place_resting_tp(trade))
    assert "resting_tp_ord_id" not in trade.extras
    assert stub.market_close_calls == []


# ---------------------------------------------------------------------------
# _cancel_resting_tp
# ---------------------------------------------------------------------------

def test_cancel_resting_tp_cancels_open_order():
    stub = StubOKX()
    ex = _executor(stub)
    trade = _trade()
    trade.extras["resting_tp_ord_id"] = "TP123"
    _run(ex._cancel_resting_tp(trade))
    assert len(stub.cancel_calls) == 1
    assert stub.cancel_calls[0]["ord_id"] == "TP123"
    # popped → a second cancel is a no-op (idempotent).
    assert "resting_tp_ord_id" not in trade.extras


def test_cancel_resting_tp_noop_without_order():
    stub = StubOKX()
    ex = _executor(stub)
    _run(ex._cancel_resting_tp(_trade()))
    assert stub.cancel_calls == []


def test_cancel_resting_tp_swallows_already_filled_error():
    """Cancelling an already-filled/gone order must not raise (the TP win path
    races the finalizer's cancel)."""
    class _Raises(StubOKX):
        async def cancel_order(self, symbol, *, ord_id=None, client_oid=None):
            raise RuntimeError("order already filled")

    stub = _Raises()
    ex = _executor(stub)
    trade = _trade()
    trade.extras["resting_tp_ord_id"] = "TP123"
    _run(ex._cancel_resting_tp(trade))  # must not raise


def test_cancel_resting_tp_race_safe_single_fire():
    """pop-before-await: two concurrent cancels fire exactly ONE real cancel —
    the loser pops None and returns. Guards the finalizer-vs-time-stop race."""
    stub = StubOKX()
    ex = _executor(stub)
    trade = _trade()
    trade.extras["resting_tp_ord_id"] = "TP123"

    async def go():
        await asyncio.gather(
            ex._cancel_resting_tp(trade),
            ex._cancel_resting_tp(trade),
        )

    _run(go())
    assert len(stub.cancel_calls) == 1


# ---------------------------------------------------------------------------
# end-to-end attach→place→cancel coherence
# ---------------------------------------------------------------------------

def test_full_cycle_entry_attaches_no_tp_then_rests_then_cancels():
    """The contract that makes the leak fix sound: the entry carries no trigger
    TP, the resting limit is placed after fill, and a non-TP exit cancels it."""
    stub = StubOKX([{"data": [{"sCode": "0", "ordId": "TPxyz"}]}])
    ex = _executor(stub)
    trade = _trade()

    # entry attaches only the SL — no trigger TP that could fill taker on a gap
    assert ex._attach_tp_args(trade.tp_px)["tp_trigger_px"] is None
    # after fill: rest the maker limit
    _run(ex._place_resting_tp(trade))
    assert trade.extras["resting_tp_ord_id"] == "TPxyz"
    # a time-stop / manual exit cancels it before closing
    _run(ex._cancel_resting_tp(trade))
    assert stub.cancel_calls[0]["ord_id"] == "TPxyz"
    assert "resting_tp_ord_id" not in trade.extras
