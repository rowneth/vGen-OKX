"""Paper-side coverage for the time_stop + maker_exit config wiring.

Verifies that:
  * farmer.time_stop.enabled wins over legacy max_hold_bars
  * farmer.maker_exit.enabled charges the time_stop exit at maker fees,
    not taker — matching what the live re-peg loop will do.
"""

from __future__ import annotations

import pandas as pd

from execution.volume_farmer import VolumeFarmerSession


def _bar(ts: str, o: float, h: float, l: float, c: float) -> dict:
    return {"open_time": pd.Timestamp(ts), "open": o, "high": h, "low": l, "close": c, "volume": 1.0}


def _cfg(*, time_stop=False, maker_exit=False, max_hold=999):
    return {
        "exchange": {"symbol": "BTC_USDT", "timeframe": "5m"},
        "fees": {"maker": 0.0002, "taker": 0.0005, "rebate_pct": 0.40},
        "farmer": {
            "capital_usd": 500.0,
            "leverage": 20,
            "margin_fraction_per_trade": 0.03,
            "tp_bps": 30.0,
            "sl_bps": 50.0,
            "max_hold_bars": max_hold,
            "time_stop": {"enabled": time_stop, "max_hold_bars": 2, "bar_seconds": 300},
            "maker_exit": {"enabled": maker_exit},
            "entry": {"mode": "micro_momentum", "min_bar_range_bps": 0.0, "max_bar_range_bps": 1_000.0},
            "alternate_direction": False,
        },
        "risk": {
            "daily_loss_limit_pct": 0.5,
            "max_drawdown_pct": 0.99,
            "consecutive_losses_limit": 99,
            "stop_on_volume_target": True,
        },
        "target": {"volume_usd": 1_000_000.0, "min_fee_cover_pct": 0.30},
    }


def _drift_sequence(start: str = "2026-04-01T00:00:00Z"):
    """3 bars: green entry, then two flat bars within +/-10bps so neither TP nor SL hits."""
    base = pd.Timestamp(start)
    return [
        _bar(base.isoformat(),                                100.0, 100.5, 99.9, 100.4),  # entry green
        _bar((base + pd.Timedelta(minutes=5)).isoformat(),   100.4, 100.42, 100.38, 100.40),
        _bar((base + pd.Timedelta(minutes=10)).isoformat(),  100.4, 100.42, 100.38, 100.41),
    ]


def test_time_stop_disabled_no_close_within_two_bars():
    s = VolumeFarmerSession(config=_cfg(time_stop=False, max_hold=999))
    rows = _drift_sequence()
    hist = pd.DataFrame([rows[0]])
    s.on_new_candle(hist)
    assert s.position is not None
    for r in rows[1:]:
        hist = pd.concat([hist, pd.DataFrame([r])], ignore_index=True)
        s.on_new_candle(hist)
    # Legacy behavior — wide max_hold leaves position open
    assert s.position is not None
    assert s.round_trips == 0


def test_time_stop_enabled_closes_after_max_hold_bars():
    s = VolumeFarmerSession(config=_cfg(time_stop=True, maker_exit=False, max_hold=999))
    events = []
    s.event_callback = events.append
    rows = _drift_sequence()
    hist = pd.DataFrame([rows[0]])
    s.on_new_candle(hist)
    assert s.position is not None
    for r in rows[1:]:
        hist = pd.concat([hist, pd.DataFrame([r])], ignore_index=True)
        s.on_new_candle(hist)
    assert s.position is None
    assert s.round_trips == 1
    # Latest exit event should be reason=time_stop, fee_type=taker (no maker_exit flag)
    exits = [e for e in events if e.kind == "exit"]
    assert exits, "expected an exit event"
    last = exits[-1].payload
    assert last["reason"] == "time_stop"
    assert last["close_fee_type"] == "taker"


def test_maker_exit_charges_time_stop_as_maker():
    s = VolumeFarmerSession(config=_cfg(time_stop=True, maker_exit=True, max_hold=999))
    events = []
    s.event_callback = events.append
    rows = _drift_sequence()
    hist = pd.DataFrame([rows[0]])
    s.on_new_candle(hist)
    for r in rows[1:]:
        hist = pd.concat([hist, pd.DataFrame([r])], ignore_index=True)
        s.on_new_candle(hist)
    assert s.position is None
    exits = [e for e in events if e.kind == "exit"]
    last = exits[-1].payload
    assert last["reason"] == "time_stop"
    # When maker_exit is enabled, the time_stop close path is accounted as maker.
    assert last["close_fee_type"] == "maker"
    # Fee should be notional * maker rate, not taker rate.
    notional = last["notional"]
    expected_maker_fee = notional * 0.0002
    assert abs(last["close_fee"] - expected_maker_fee) < 1e-9


def test_real_sl_still_taker_with_maker_exit():
    """Genuine SL hits should remain taker even when maker_exit is on."""
    s = VolumeFarmerSession(config=_cfg(time_stop=True, maker_exit=True, max_hold=999))
    events = []
    s.event_callback = events.append
    base = pd.Timestamp("2026-04-01T00:00:00Z")
    hist = pd.DataFrame([_bar(base.isoformat(), 100.0, 100.5, 99.9, 100.4)])
    s.on_new_candle(hist)
    # Smash through SL on next bar
    sl_bar = _bar((base + pd.Timedelta(minutes=5)).isoformat(), 100.4, 100.45, 99.0, 99.5)
    hist = pd.concat([hist, pd.DataFrame([sl_bar])], ignore_index=True)
    s.on_new_candle(hist)
    exits = [e for e in events if e.kind == "exit"]
    last = exits[-1].payload
    assert last["reason"] == "sl"
    assert last["close_fee_type"] == "taker"
