"""Unit tests for the maker-only exit re-peg loop."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from execution.maker_exit import (
    ExitResult,
    MakerExitConfig,
    close_position_maker,
)


class StubOKX:
    """In-memory OKX stub that lets us script ticker / order outcomes per call."""

    def __init__(self) -> None:
        self.tickers: List[Dict[str, str]] = []
        self.place_results: List[Dict[str, Any]] = []
        self.order_states: List[List[Dict[str, Any]]] = []   # per-call list of get_order responses
        self.cancel_calls: int = 0
        self.market_close_calls: int = 0
        self._order_state_iters: List[Any] = []
        self.placed_orders: List[Dict[str, Any]] = []

    async def get_ticker(self, symbol: str):
        if not self.tickers:
            return {"askPx": "100.0", "bidPx": "99.9"}
        return self.tickers.pop(0)

    async def place_order(self, **kwargs):
        self.placed_orders.append(kwargs)
        resp = self.place_results.pop(0) if self.place_results else {
            "data": [{"sCode": "0", "ordId": f"o{len(self.placed_orders)}"}],
        }
        # Prime the order-state iterator for this order if scripted
        if self.order_states:
            self._order_state_iters.append(iter(self.order_states.pop(0)))
        else:
            self._order_state_iters.append(iter([{"data": [{"state": "live", "accFillSz": "0"}]}]))
        return resp

    async def get_order(self, symbol, *, ord_id=None, client_oid=None):
        if not self._order_state_iters:
            return {"data": [{"state": "live", "accFillSz": "0"}]}
        it = self._order_state_iters[-1]
        try:
            return next(it)
        except StopIteration:
            return {"data": [{"state": "live", "accFillSz": "0"}]}

    async def cancel_order(self, symbol, *, ord_id=None, client_oid=None):
        self.cancel_calls += 1
        return {"code": "0", "data": []}

    async def close_position_market(self, symbol, **kwargs):
        self.market_close_calls += 1
        return {"code": "0", "data": []}


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def fast_cfg():
    # Tiny durations so tests don't take seconds
    return MakerExitConfig(
        repeg_ms=30, max_repegs=8, max_exit_seconds=0.6,
        max_adverse_bps=15.0, poll_ms=10,
    )


def test_maker_exit_fills_first_try(fast_cfg):
    """Touch is right at intended; one post_only fills cleanly → pure maker."""
    stub = StubOKX()
    stub.tickers = [{"askPx": "100.0", "bidPx": "99.9"}]
    stub.place_results = [{"data": [{"sCode": "0", "ordId": "ok1"}]}]
    stub.order_states = [[
        {"data": [{"state": "live", "accFillSz": "0"}]},
        {"data": [{"state": "filled", "accFillSz": "1.0",
                   "avgPx": "100.0", "fee": "-0.02"}]},
    ]]
    res: ExitResult = _run(close_position_maker(
        stub,  # type: ignore[arg-type]
        symbol="BTC_USDT", position_side="long",
        sz_contracts=1.0, tick_sz=0.1,
        intended_exit_px=100.0, cfg=fast_cfg,
    ))
    assert res.fill_class == "maker"
    assert res.filled_qty == 1.0
    assert res.maker_qty == 1.0
    assert res.taker_qty == 0.0
    assert res.fallback_reason == ""
    assert stub.market_close_calls == 0


def test_maker_exit_repegs_then_fills(fast_cfg):
    """First post_only stays live (no fill) → cancel + re-peg → second fills."""
    stub = StubOKX()
    stub.tickers = [
        {"askPx": "100.0", "bidPx": "99.9"},
        {"askPx": "100.1", "bidPx": "100.0"},
    ]
    stub.place_results = [
        {"data": [{"sCode": "0", "ordId": "ok1"}]},
        {"data": [{"sCode": "0", "ordId": "ok2"}]},
    ]
    # First order: never fills inside the window.
    # Second order: fills.
    stub.order_states = [
        [
            {"data": [{"state": "live", "accFillSz": "0"}]},
            {"data": [{"state": "live", "accFillSz": "0"}]},
        ],
        [
            {"data": [{"state": "filled", "accFillSz": "1.0",
                       "avgPx": "100.1", "fee": "-0.02"}]},
        ],
    ]
    res = _run(close_position_maker(
        stub,  # type: ignore[arg-type]
        symbol="BTC_USDT", position_side="long",
        sz_contracts=1.0, tick_sz=0.1,
        intended_exit_px=100.0, cfg=fast_cfg,
    ))
    assert res.filled_qty == 1.0
    assert res.maker_qty == 1.0
    assert res.repegs >= 1
    assert stub.cancel_calls >= 1
    assert res.fallback_reason == ""


def test_maker_exit_falls_back_to_taker_on_max_repegs(fast_cfg):
    """All re-pegs fail to fill → eventually hits max_repegs → taker market close."""
    stub = StubOKX()
    stub.tickers = [{"askPx": "100.0", "bidPx": "99.9"} for _ in range(10)]
    stub.place_results = [{"data": [{"sCode": "0", "ordId": f"o{i}"}]} for i in range(10)]
    # max_repegs=2 → 2 maker attempts (both stay live), then taker fallback (fills).
    stub.order_states = [
        [{"data": [{"state": "live", "accFillSz": "0"}]}],
        [{"data": [{"state": "live", "accFillSz": "0"}]}],
        [{"data": [{"state": "filled", "accFillSz": "1.0",
                    "avgPx": "100.0", "fee": "-0.05"}]}],
    ]
    cfg = MakerExitConfig(repeg_ms=10, max_repegs=2, max_exit_seconds=1.0,
                          max_adverse_bps=200.0, poll_ms=5)
    res = _run(close_position_maker(
        stub,  # type: ignore[arg-type]
        symbol="BTC_USDT", position_side="long",
        sz_contracts=1.0, tick_sz=0.1,
        intended_exit_px=100.0, cfg=cfg,
    ))
    assert res.filled_qty == 1.0
    assert res.taker_qty == 1.0
    assert res.maker_qty == 0.0
    assert res.fallback_reason.startswith("max_repegs")
    assert res.used_taker


def test_maker_exit_aborts_on_adverse_excursion(fast_cfg):
    """Touch drifts far from intended → adverse trigger fires before max_repegs."""
    stub = StubOKX()
    # Position: long. Closing means SELL. Intended px = 100.0. Touch falling below
    # 100.0 means we'd sell into worse price → adverse for the long-close.
    stub.tickers = [{"askPx": "99.0", "bidPx": "98.9"}]
    cfg = MakerExitConfig(repeg_ms=10, max_repegs=8, max_exit_seconds=1.0,
                          max_adverse_bps=15.0, poll_ms=5)
    stub.order_states.append([
        {"data": [{"state": "filled", "accFillSz": "1.0",
                   "avgPx": "99.0", "fee": "-0.05"}]},
    ])
    res = _run(close_position_maker(
        stub,  # type: ignore[arg-type]
        symbol="BTC_USDT", position_side="long",
        sz_contracts=1.0, tick_sz=0.1,
        intended_exit_px=100.0, cfg=cfg,
    ))
    assert res.fallback_reason.startswith("adverse_")
    assert res.adverse_bps > 15.0
    assert res.used_taker  # residual taken via market fallback
