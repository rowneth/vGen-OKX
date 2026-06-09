"""Tests for the DEMO realism overlay + firm-TP override.

Covers the safety-critical property (frictions are OFF under --live) and the
direction of the frictions, plus the firm 12bps effective-TP override.
"""
import pathlib
import random
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from execution.live_volume_executor_okx import (  # noqa: E402
    DemoRealismConfig,
    LiveVolumeExecutorOKX,
)
from execution.volume_farmer import VolumeFarmerSession  # noqa: E402


class _FakeClient:
    def __init__(self, simulated: bool):
        self._simulated = simulated


def _mk_exec(simulated: bool, realism):
    e = LiveVolumeExecutorOKX(client=_FakeClient(simulated), demo_realism=realism)
    e._tick_sz = 0.1
    if realism is not None and realism.enabled:
        e._demo_rng = random.Random(realism.seed or 1)
    return e


def test_demo_gate_is_live_safe():
    # demo client + realism overlay -> active
    assert _mk_exec(True, DemoRealismConfig(seed=1))._demo_on() is True
    # LIVE client (not simulated) -> frictions OFF even if a realism object exists
    assert _mk_exec(False, DemoRealismConfig(seed=1))._demo_on() is False
    # no realism object (the live path) -> OFF
    assert _mk_exec(True, None)._demo_on() is False


def test_entry_slip_is_adverse():
    e = _mk_exec(True, DemoRealismConfig(seed=1, entry_slip_bps=10.0))
    raw = 100.0
    assert e._apply_demo_entry_slip("long", raw) > raw    # long fills higher = worse
    assert e._apply_demo_entry_slip("short", raw) < raw   # short fills lower = worse


def _cfg(force_tp: float):
    return {
        "exchange": {"symbol": "BTC_USDT", "timeframe": "5m"},
        "fees": {"maker": 0.0002, "taker": 0.0005, "rebate_pct": 0.4},
        "farmer": {
            "capital_usd": 500, "leverage": 0, "margin_fraction_per_trade": 0.03,
            "sizing": {"dynamic_leverage": True, "risk_per_trade_pct": 0.025,
                       "max_leverage": 68, "min_leverage": 5},
            "tp_bps": 8.0, "sl_bps": 50.0, "force_tp_bps": force_tp, "max_hold_bars": 999,
            "entry": {"mode": "micro_momentum", "min_bar_range_bps": 0.0,
                      "max_bar_range_bps": 100000.0},
            "alternate_direction": False, "limit_tp": True,
            "atr": {"period": 14, "relative": True, "tp_mult": 0.5, "sl_mult": 1.5,
                    "tp_bps_min": 5.0, "sl_bps_min": 8.0, "min_usd": 0.0},
            "trend_break": {"enabled": False},
        },
        "risk": {"daily_loss_limit_pct": 0.5, "max_drawdown_pct": 0.95,
                 "consecutive_losses_limit": 10, "consecutive_losses_cooldown_bars": 12,
                 "stop_on_volume_target": True},
        "target": {"volume_usd": 5_000_000},
    }


def _first_entry(force_tp: float):
    s = VolumeFarmerSession(config=_cfg(force_tp))
    events = []
    s.event_callback = events.append
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    base = 60000.0
    for i in range(60):
        o = base + i * 10.0
        c = o + 20.0           # up bar -> micro_momentum long
        rows.append({"open_time": t0 + pd.Timedelta(minutes=5 * i),
                     "open": o, "high": max(o, c) + 5, "low": min(o, c) - 5, "close": c})
    s.on_new_candle(pd.DataFrame(rows))
    return next((e for e in events if e.kind == "entry"), None)


def test_force_tp_bps_pins_effective_tp_to_12():
    evt = _first_entry(force_tp=12.0)
    assert evt is not None, "expected an entry"
    assert abs(evt.payload["tp_bps"] - 12.0) < 1e-6   # firm 12 bps regardless of ATR
    assert evt.payload["sl_bps"] > 0                  # SL stays ATR-relative (not pinned)


def test_no_override_uses_atr_bracket():
    evt = _first_entry(force_tp=0.0)
    assert evt is not None
    # ATR-relative TP (tp_mult 0.5) is well under 12 for this low-vol synthetic feed
    assert evt.payload["tp_bps"] < 12.0
