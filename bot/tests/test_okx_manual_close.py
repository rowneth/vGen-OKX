"""Tests for the Telegram manual-position-management commands.

Covers the /position, /orders, /close (market + maker-limit) and /cancel flow
added so the operator can see and flatten a stuck ("EXIT FAILED · still OPEN")
position directly from Telegram, plus the routing in _handle_command.

The runner lives in ``bot/scripts`` (not on the default test pythonpath), so we
add it explicitly. It imports only stdlib + pandas/yaml at module load, all of
which are in the venv.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_volume_farmer_okx as R  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# Scriptable fake OKX client
# --------------------------------------------------------------------------
class FakeClient:
    def __init__(
        self,
        *,
        positions: Any,
        ticker: Optional[Dict[str, str]] = None,
        pending_orders: Optional[Dict[str, Any]] = None,
        pending_algos: Optional[Dict[str, Any]] = None,
        close_resp: Optional[Dict[str, Any]] = None,
        place_resp: Optional[Dict[str, Any]] = None,
    ) -> None:
        # positions: fixed dict, or a list used as a per-call queue (last repeats).
        self._positions = positions
        self._ticker = ticker or {"bidPx": "60000.0", "askPx": "60000.1"}
        self._pending_orders = pending_orders or {"data": []}
        self._pending_algos = pending_algos or {"data": []}
        self._close_resp = close_resp or {"data": [{"sCode": "0"}]}
        self._place_resp = place_resp or {"data": [{"sCode": "0", "ordId": "O123"}]}
        self.calls: Dict[str, int] = {}
        self.close_kwargs: Optional[Dict[str, Any]] = None
        self.place_kwargs: Optional[Dict[str, Any]] = None
        self.cancelled_orders: List[str] = []
        self.cancelled_algos: List[List[str]] = []

    def _bump(self, name: str) -> int:
        self.calls[name] = self.calls.get(name, 0) + 1
        return self.calls[name]

    async def get_positions(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        n = self._bump("get_positions")
        if isinstance(self._positions, list):
            idx = min(n - 1, len(self._positions) - 1)
            return self._positions[idx]
        return self._positions

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        self._bump("get_ticker")
        return self._ticker

    async def close_position_market(self, symbol: str, **kw: Any) -> Dict[str, Any]:
        self._bump("close_position_market")
        self.close_kwargs = kw
        return self._close_resp

    async def place_order(self, **kw: Any) -> Dict[str, Any]:
        self._bump("place_order")
        self.place_kwargs = kw
        return self._place_resp

    async def get_pending_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        self._bump("get_pending_orders")
        return self._pending_orders

    async def get_pending_algos(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        self._bump("get_pending_algos")
        return self._pending_algos

    async def cancel_order(self, symbol: str, *, ord_id: str = "", **kw: Any) -> Dict[str, Any]:
        self._bump("cancel_order")
        self.cancelled_orders.append(ord_id)
        return {"data": [{"sCode": "0"}]}

    async def cancel_algos(self, symbol: str, algo_ids: List[str]) -> Dict[str, Any]:
        self._bump("cancel_algos")
        self.cancelled_algos.append(list(algo_ids))
        return {"data": [{"sCode": "0"}]}


def _pos_open(pos: str = "0.25", pos_side: str = "net", upl: str = "-0.17") -> Dict[str, Any]:
    return {"data": [{"pos": pos, "posSide": pos_side, "upl": upl,
                      "avgPx": "60708.7", "markPx": "60700.0"}]}


def _pos_flat() -> Dict[str, Any]:
    return {"data": [{"pos": "0"}]}


def _no_sleep(monkeypatch):
    async def _noop(*_a, **_k):
        return
    monkeypatch.setattr(R.asyncio, "sleep", _noop)


class FakeTrade:
    def __init__(self) -> None:
        self.closed = False
        self.close_reason = ""
        self.extras: Dict[str, Any] = {}


class FakeExec:
    """Minimal stand-in for LiveVolumeExecutorOKX (only the attrs the runner reads)."""

    def __init__(self, open_trade: Optional[FakeTrade] = None) -> None:
        self._open_trade = open_trade
        self._close_lock = asyncio.Lock()
        self.mgn_mode = "isolated"
        self._lot_sz = 0.01
        self._tick_sz = 0.1
        self.pos_mode = "net"


# --------------------------------------------------------------------------
# _close_direction
# --------------------------------------------------------------------------
def test_close_direction_net_long():
    d = R._close_direction({"pos": "0.25", "posSide": "net"}, "net")
    assert d == {"is_long": True, "close_side": "sell", "order_pos_side": None, "size": 0.25}


def test_close_direction_net_short():
    d = R._close_direction({"pos": "-0.25", "posSide": "net"}, "net")
    assert d == {"is_long": False, "close_side": "buy", "order_pos_side": None, "size": 0.25}


def test_close_direction_hedge_long_keeps_posside():
    d = R._close_direction({"pos": "0.25", "posSide": "long"}, "hedge")
    assert d["close_side"] == "sell"
    assert d["order_pos_side"] == "long"


def test_close_direction_hedge_short_keeps_posside():
    d = R._close_direction({"pos": "0.25", "posSide": "short"}, "hedge")
    assert d["close_side"] == "buy"
    assert d["order_pos_side"] == "short"


# --------------------------------------------------------------------------
# _parse_close_price
# --------------------------------------------------------------------------
def test_parse_close_price_variants():
    assert R._parse_close_price("/close limit 60123.4") == ("ok", 60123.4)
    assert R._parse_close_price("/close limit $60,000") == ("ok", 60000.0)
    assert R._parse_close_price("/close limit") == ("none", None)
    assert R._parse_close_price("/close market") == ("none", None)
    assert R._parse_close_price("/close limit 6o000") == ("bad", None)
    assert R._parse_close_price("/close limit abc") == ("bad", None)
    assert R._parse_close_price("") == ("none", None)


def test_code_sanitizer_strips_backslash_and_backtick():
    # Inside a code span only backtick/backslash matter — _code neutralizes both.
    assert "`" not in R._code("would cross `x`")
    assert "\\" not in R._code("a\\b path")
    assert R._code("ok (51121)") == "ok (51121)"


# --------------------------------------------------------------------------
# Market close
# --------------------------------------------------------------------------
def test_market_close_net_sends_autocxl_and_confirms_flat(monkeypatch):
    _no_sleep(monkeypatch)
    # open before close, flat afterwards
    client = FakeClient(positions=[_pos_open(), _pos_flat()])
    text = _run(R._do_market_close(client, "BTC_USDT", "net", "isolated", "LIVE", "okx-live"))
    assert client.calls["close_position_market"] == 1
    assert client.close_kwargs["auto_cxl"] is True
    assert client.close_kwargs["pos_side"] is None  # net mode omits posSide
    assert "closed" in text.lower()


def test_market_close_hedge_passes_posside(monkeypatch):
    _no_sleep(monkeypatch)
    client = FakeClient(positions=[_pos_open(pos_side="long"), _pos_flat()])
    _run(R._do_market_close(client, "BTC_USDT", "hedge", "isolated", "LIVE", "okx-live"))
    assert client.close_kwargs["pos_side"] == "long"


def test_market_close_already_flat_short_circuits(monkeypatch):
    _no_sleep(monkeypatch)
    client = FakeClient(positions=_pos_flat())
    text = _run(R._do_market_close(client, "BTC_USDT", "net", "isolated", "LIVE", "okx-live"))
    assert "close_position_market" not in client.calls
    assert "flat" in text.lower()


def test_market_close_rejected_scode(monkeypatch):
    _no_sleep(monkeypatch)
    client = FakeClient(
        positions=_pos_open(),
        close_resp={"data": [{"sCode": "51000", "sMsg": "Parameter error"}]},
    )
    text = _run(R._do_market_close(client, "BTC_USDT", "net", "isolated", "LIVE", "okx-live"))
    assert "rejected" in text.lower()
    assert "51000" in text


def test_market_close_still_open_after_send(monkeypatch):
    _no_sleep(monkeypatch)
    client = FakeClient(positions=_pos_open())  # never goes flat
    text = _run(R._do_market_close(client, "BTC_USDT", "net", "isolated", "LIVE", "okx-live"))
    assert "sent" in text.lower()


# --------------------------------------------------------------------------
# Limit close
# --------------------------------------------------------------------------
def test_limit_close_explicit_price_long_is_reduceonly_sell():
    client = FakeClient(positions=_pos_open(pos="0.25"))
    text = _run(R._do_limit_close(
        client, "BTC_USDT", "net", "isolated", 0.01, 0.1, 60500.0, "LIVE", "okx-live",
    ))
    kw = client.place_kwargs
    assert kw["side"] == "sell"
    assert kw["reduce_only"] is True
    # post_only even with an explicit price: the UI promises a MAKER close, and
    # a plain 'limit' at/through the touch fills instantly as taker (5bps).
    assert kw["ord_type"] == "post_only"
    assert kw["sz"] == "0.25"
    assert kw["px"] == "60500.0"
    assert kw["pos_side"] is None
    assert "O123" in text


def test_limit_close_no_price_uses_touch_post_only_long():
    # closing a long => SELL maker rests at best ask
    client = FakeClient(positions=_pos_open(pos="0.25"),
                        ticker={"bidPx": "60000.0", "askPx": "60010.5"})
    _run(R._do_limit_close(
        client, "BTC_USDT", "net", "isolated", 0.01, 0.1, None, "LIVE", "okx-live",
    ))
    kw = client.place_kwargs
    assert kw["ord_type"] == "post_only"
    assert kw["side"] == "sell"
    assert kw["px"] == "60010.5"


def test_limit_close_no_price_uses_touch_post_only_short():
    # closing a short => BUY maker rests at best bid
    client = FakeClient(positions=_pos_open(pos="-0.25"),
                        ticker={"bidPx": "59990.0", "askPx": "60000.0"})
    _run(R._do_limit_close(
        client, "BTC_USDT", "net", "isolated", 0.01, 0.1, None, "LIVE", "okx-live",
    ))
    kw = client.place_kwargs
    assert kw["side"] == "buy"
    assert kw["px"] == "59990.0"


def test_limit_close_price_rounds_to_tick():
    client = FakeClient(positions=_pos_open(pos="0.25"))
    _run(R._do_limit_close(
        client, "BTC_USDT", "net", "isolated", 0.01, 0.1, 60123.456, "LIVE", "okx-live",
    ))
    assert client.place_kwargs["px"] == "60123.5"


def test_limit_close_already_flat():
    client = FakeClient(positions=_pos_flat())
    text = _run(R._do_limit_close(
        client, "BTC_USDT", "net", "isolated", 0.01, 0.1, 60500.0, "LIVE", "okx-live",
    ))
    assert "place_order" not in client.calls
    assert "flat" in text.lower()


def test_limit_close_rejected_scode():
    client = FakeClient(
        positions=_pos_open(),
        place_resp={"data": [{"sCode": "51121", "sMsg": "would cross"}]},
    )
    text = _run(R._do_limit_close(
        client, "BTC_USDT", "net", "isolated", 0.01, 0.1, 60500.0, "LIVE", "okx-live",
    ))
    assert "rejected" in text.lower()
    assert "51121" in text


def test_limit_close_sweeps_resting_orders_first():
    # MED fix: a leftover reduceOnly limit must be pulled before placing a new one.
    tr = FakeTrade()
    ex = FakeExec(open_trade=tr)
    client = FakeClient(positions=_pos_open(), pending_orders={"data": [{"ordId": "OLD"}]})
    _run(R._do_limit_close(
        client, "BTC_USDT", "net", "isolated", 0.01, 0.1, 60500.0, "LIVE", "okx-live", ex,
    ))
    assert "OLD" in client.cancelled_orders   # swept before the new order
    assert tr.close_reason == "manual"        # tagged as operator close


# --------------------------------------------------------------------------
# Manual-close tagging + gate release (executor coordination)
# --------------------------------------------------------------------------
def test_market_close_tags_manual_and_releases_gate(monkeypatch):
    _no_sleep(monkeypatch)
    tr = FakeTrade()
    ex = FakeExec(open_trade=tr)
    client = FakeClient(positions=[_pos_open(), _pos_flat()])
    _run(R._do_market_close(client, "BTC_USDT", "net", "isolated", "LIVE", "okx-live", ex))
    assert tr.close_reason == "manual"
    assert tr.extras.get("manual_close") is True
    assert ex._open_trade is None             # gate released after confirmed flat


def test_market_close_does_not_release_gate_when_still_open(monkeypatch):
    _no_sleep(monkeypatch)
    tr = FakeTrade()
    tr.extras["still_open"] = True            # executor's recovery watcher owns release
    ex = FakeExec(open_trade=tr)
    client = FakeClient(positions=_pos_open())  # never goes flat
    _run(R._do_market_close(client, "BTC_USDT", "net", "isolated", "LIVE", "okx-live", ex))
    assert ex._open_trade is tr               # gate NOT yanked from under recovery


# --------------------------------------------------------------------------
# Cancel all — STOP-LOSS SAFE
# --------------------------------------------------------------------------
def test_cancel_all_keeps_sl_while_position_open():
    # HIGH-severity fix: an open position must NOT have its TP/SL algos cancelled.
    client = FakeClient(
        positions=_pos_open(),
        pending_orders={"data": [{"ordId": "A"}, {"ordId": "B"}]},
        pending_algos={"data": [{"algoId": "Z1"}, {"algoId": "Z2"}]},
    )
    text = _run(R._do_cancel_all(client, "BTC_USDT", "LIVE", "okx-live"))
    assert client.cancelled_orders == ["A", "B"]      # limit orders pulled
    assert client.cancelled_algos == []               # protective algos KEPT
    assert "kept" in text.lower()
    assert "still in place" in text.lower()


def test_cancel_all_sweeps_everything_when_flat():
    client = FakeClient(
        positions=_pos_flat(),
        pending_orders={"data": [{"ordId": "A"}]},
        pending_algos={"data": [{"algoId": "Z1"}]},
    )
    text = _run(R._do_cancel_all(client, "BTC_USDT", "LIVE", "okx-live"))
    assert client.cancelled_orders == ["A"]
    assert client.cancelled_algos == [["Z1"]]
    assert "flat" in text.lower()


def test_cancel_all_nothing_resting_open():
    client = FakeClient(positions=_pos_open())
    text = _run(R._do_cancel_all(client, "BTC_USDT", "LIVE", "okx-live"))
    assert "`0`" in text


# --------------------------------------------------------------------------
# Orders view
# --------------------------------------------------------------------------
def test_build_orders_text_renders_orders_and_algos():
    oo = {
        "orders": [{"side": "sell", "sz": "0.25", "px": "60500", "ordType": "limit",
                    "reduceOnly": "true", "ordId": "ORD9"}],
        "algos": [{"tpTriggerPx": "61000", "slTriggerPx": "60000", "algoId": "ALG1"}],
    }
    text = R._build_orders_text(oo, "BTC_USDT", "okx-live", "LIVE")
    assert "ORD9" in text
    assert "ALG1" in text
    assert "60500" in text


def test_build_orders_text_empty():
    text = R._build_orders_text({"orders": [], "algos": []}, "BTC_USDT", "okx-live", "LIVE")
    assert "No resting orders" in text


# --------------------------------------------------------------------------
# _handle_command routing
# --------------------------------------------------------------------------
def _hc_kwargs(client, mode_label="LIVE", live_executor=None):
    return dict(
        session=None, client=client, symbol="BTC_USDT", tf="5m", cfg={},
        mode_label=mode_label, bot_label="okx-live",
        start_time=datetime.now(tz=timezone.utc),
        log_dir=None, live_executor=live_executor, pos_mode="net",
    )


def test_handle_help_returns_text_no_buttons():
    text, buttons = _run(R._handle_command("help", "", **_hc_kwargs(FakeClient(positions=_pos_flat()))))
    assert buttons is None
    assert "Commands" in text


def test_handle_dismiss_is_noop():
    text, buttons = _run(R._handle_command("dismiss", "", **_hc_kwargs(FakeClient(positions=_pos_flat()))))
    assert text is None and buttons is None


def test_handle_close_returns_confirm_buttons():
    client = FakeClient(positions=_pos_open())
    text, buttons = _run(R._handle_command("close", "", **_hc_kwargs(client)))
    assert buttons is R._CLOSE_CONFIRM_BUTTONS
    assert "Confirm close" in text


def test_handle_manage_blocked_in_paper():
    client = FakeClient(positions=_pos_flat())
    text, buttons = _run(R._handle_command("close", "", **_hc_kwargs(client, mode_label="PAPER")))
    assert buttons is None
    assert "live/demo" in text.lower()
    assert "close_position_market" not in client.calls


def test_handle_close_market_executes(monkeypatch):
    _no_sleep(monkeypatch)
    client = FakeClient(positions=[_pos_open(), _pos_flat()])
    text, buttons = _run(R._handle_command("close_market", "", **_hc_kwargs(client)))
    assert client.calls["close_position_market"] == 1
    assert buttons is None


def test_handle_close_limit_passes_typed_price():
    client = FakeClient(positions=_pos_open())
    _run(R._handle_command("close_limit", "/close limit 60500", **_hc_kwargs(client)))
    assert client.place_kwargs["px"] == "60500.0"


def test_handle_close_limit_bad_price_warns_and_sends_nothing():
    client = FakeClient(positions=_pos_open())
    text, buttons = _run(R._handle_command("close_limit", "/close limit 6o000", **_hc_kwargs(client)))
    assert buttons is None
    assert "place_order" not in client.calls   # no order placed on a typo
    assert "price" in text.lower()
