"""Tests for the real-money ops helpers: 24h fee window + fee-tier verification.

These guard the "real facts only" contract: every number in the live rebate
reminder and the startup fee check comes from the exchange API, errors render
as "unavailable" (never a fake $0), and a fee-tier mismatch is called out.
"""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import sys
import time
from typing import Any, Dict, Optional

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
_spec = importlib.util.spec_from_file_location(
    "run_volume_farmer_okx", _ROOT / "scripts" / "run_volume_farmer_okx.py",
)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def _run(coro):
    return asyncio.run(coro)


class FakeFeeClient:
    """Scriptable client for _fees_last_24h / _verify_fee_tier."""

    def __init__(self, orders: Optional[list] = None,
                 trade_fee: Optional[Dict[str, Any]] = None,
                 fail_history: bool = False, fail_fee: bool = False) -> None:
        self._orders = orders or []
        self._trade_fee = trade_fee
        self._fail_history = fail_history
        self._fail_fee = fail_fee

    async def _request(self, method: str, path: str, **kw: Any) -> Dict[str, Any]:
        assert "orders-history" in path
        if self._fail_history:
            raise RuntimeError("api down")
        return {"code": "0", "data": self._orders}

    async def get_trade_fee(self, symbol: str, **kw: Any) -> Dict[str, Any]:
        if self._fail_fee or self._trade_fee is None:
            raise RuntimeError("api down")
        return self._trade_fee


def _order(fill_ago_s: float, fee: str, state: str = "filled") -> Dict[str, Any]:
    return {"state": state, "fee": fee, "ordId": "o1",
            "fillTime": str(int((time.time() - fill_ago_s) * 1000))}


# --------------------------------------------------------------------------
# _fees_last_24h — real-window fee sum
# --------------------------------------------------------------------------

def test_fees_last_24h_sums_only_recent_filled():
    client = FakeFeeClient(orders=[
        _order(60, "-0.10"),                 # 1 min ago — counted
        _order(3600, "-0.25"),               # 1 h ago — counted
        _order(90_000, "-9.99"),             # 25 h ago — outside window
        _order(120, "-5.00", state="canceled"),  # not filled — ignored
    ])
    total = _run(R._fees_last_24h(client, "BTC_USDT"))
    assert total is not None
    assert abs(total - 0.35) < 1e-9


def test_fees_last_24h_unavailable_returns_none_not_zero():
    client = FakeFeeClient(fail_history=True)
    assert _run(R._fees_last_24h(client, "BTC_USDT")) is None


# --------------------------------------------------------------------------
# _verify_fee_tier — config vs the account's ACTUAL rates
# --------------------------------------------------------------------------

_CFG = {"fees": {"maker": 0.0002, "taker": 0.0005, "rebate_pct": 0.40}}


def test_fee_tier_verified_when_matching():
    client = FakeFeeClient(trade_fee={"makerU": "-0.0002", "takerU": "-0.0005"})
    note = _run(R._verify_fee_tier(client, "BTC_USDT", _CFG))
    assert "verified" in note
    assert "MISMATCH" not in note


def test_fee_tier_mismatch_is_called_out():
    # Account actually pays 3/6bps while the config assumes 2/5 — the whole
    # cost model would be wrong; the note must scream, not soothe.
    client = FakeFeeClient(trade_fee={"makerU": "-0.0003", "takerU": "-0.0006"})
    note = _run(R._verify_fee_tier(client, "BTC_USDT", _CFG))
    assert "MISMATCH" in note
    assert "3.0" in note and "6.0" in note     # real rates shown in bps


def test_fee_tier_api_down_is_unverified_not_fake():
    client = FakeFeeClient(fail_fee=True)
    note = _run(R._verify_fee_tier(client, "BTC_USDT", _CFG))
    assert "unverified" in note
