"""Milestone-0 safety-rail tests.

Covers: config schema validation (accepts the shipped config, rejects unsafe
edits), the fee-floor / timeframe math, the liquidation-safety check, the
file-backed kill switch lifecycle, and the strategy-level bounds (SL cap +
age-ramped pace factor) and the executor kill-switch entry gate.
"""

from __future__ import annotations

import copy
import pathlib

import pandas as pd
import pytest
import yaml

from execution.safety import (
    KillSwitch,
    atr_bps_median,
    check_liquidation_safety,
    deadman_idle_exceeded,
    fee_floor_bps,
    liq_distance_bps,
    round_trip_fee_bps,
    tf_seconds,
    validate_config,
)

CONFIG_PATH = pathlib.Path(__file__).resolve().parents[1] / "config" / "config_volume_farmer_okx_v3.yaml"


@pytest.fixture
def cfg():
    return yaml.safe_load(CONFIG_PATH.read_text())


# ---------------------------------------------------------------------------
# config schema validation
# ---------------------------------------------------------------------------

def test_shipped_config_is_valid(cfg):
    """The single most important guard: the live config MUST pass, else a
    restart refuses to start and halts the campaign."""
    assert validate_config(cfg) == []


def test_missing_section_blocks(cfg):
    del cfg["fees"]
    errs = validate_config(cfg)
    assert any("fees" in e for e in errs)


def test_margin_typo_is_caught(cfg):
    cfg["farmer"]["margin_fration_per_trade"] = cfg["farmer"].pop("margin_fraction_per_trade")
    errs = validate_config(cfg)
    assert any("margin_fraction_per_trade" in e and "MISSING" in e for e in errs)


def test_overleverage_blocked(cfg):
    cfg["farmer"]["sizing"]["max_leverage"] = 75
    assert any("max_leverage" in e and "maximum" in e for e in validate_config(cfg))


def test_min_above_max_leverage_blocked(cfg):
    cfg["farmer"]["sizing"]["min_leverage"] = 20  # > max_leverage 15
    assert any("min_leverage" in e for e in validate_config(cfg))


def test_tp_cap_below_fee_floor_blocked(cfg):
    cfg["farmer"]["atr"]["tp_bps_max"] = 5.0   # floor is ~8.4
    assert any("tp_bps_max" in e and "floor" in e for e in validate_config(cfg))


def test_unknown_sizing_key_blocked(cfg):
    cfg["farmer"]["sizing"]["max_levrage"] = 20
    assert any("unknown key" in e for e in validate_config(cfg))


def test_pace_min_above_max_blocked(cfg):
    cfg["farmer"]["pace"]["min_margin_fraction"] = 0.30  # > max 0.20
    assert any("min_margin_fraction" in e for e in validate_config(cfg))


def test_breaker_bounds_enforced(cfg):
    cfg["risk"]["live_breaker_max_drawdown_pct"] = 1.5  # above 0.99
    assert any("live_breaker_max_drawdown_pct" in e for e in validate_config(cfg))


def test_bad_timeframe_blocked(cfg):
    cfg["exchange"]["timeframe"] = "5x"
    assert any("timeframe" in e for e in validate_config(cfg))


def test_rebate_out_of_range_blocked(cfg):
    cfg["fees"]["rebate_pct"] = 1.5
    assert any("rebate_pct" in e for e in validate_config(cfg))


# ---------------------------------------------------------------------------
# fee / timeframe math
# ---------------------------------------------------------------------------

def test_tf_seconds():
    assert tf_seconds("5m") == 300
    assert tf_seconds("1h") == 3600
    with pytest.raises(ValueError):
        tf_seconds("5x")
    with pytest.raises(ValueError):
        tf_seconds("m")


def test_fee_floor_matches_config(cfg):
    # maker 2bps, rebate 20%, rebate_all_legs, close maker (resting/limit tp):
    # round_trip = (1.6 + 1.6) = 3.2bps; floor = 2.0*3.2 + 2.0 = 8.4bps
    assert round_trip_fee_bps(cfg) == pytest.approx(3.2, abs=1e-6)
    assert fee_floor_bps(cfg) == pytest.approx(8.4, abs=1e-6)


# ---------------------------------------------------------------------------
# liquidation safety
# ---------------------------------------------------------------------------

def test_liq_distance_15x():
    # (1/15 - 0.005) * 1e4 ≈ 617 bps
    assert liq_distance_bps(15) == pytest.approx(617.0, abs=2.0)


def test_liq_safe_at_15x_typical_atr():
    errs, _ = check_liquidation_safety("BTC", 14.0, max_leverage=15, sl_mult=9.0, safety_factor=2.0)
    assert errs == []


def test_liq_unsafe_when_overleveraged():
    # DOGE-like median ATR 23bps at 30x: sl=207, liq=283 → 1.37x < 2x → ERROR
    errs, _ = check_liquidation_safety("DOGE", 23.0, max_leverage=30, sl_mult=9.0, safety_factor=2.0)
    assert errs and "UNSAFE" in errs[0]


def test_liq_skipped_without_data():
    errs, warns = check_liquidation_safety("X", 0.0, max_leverage=15, sl_mult=9.0)
    assert errs == [] and warns


def test_atr_bps_median_sane_then_zero():
    # ~30 bars of BTC-ish 5m candles → a positive, plausible bps figure.
    rows = []
    px = 50000.0
    for i in range(40):
        hi, lo = px * 1.001, px * 0.999
        rows.append({"high": hi, "low": lo, "close": px})
        px *= 1.0002
    med = atr_bps_median(pd.DataFrame(rows), 14)
    assert 5.0 < med < 60.0
    # too few bars → 0.0 (cannot compute)
    assert atr_bps_median(pd.DataFrame(rows[:5]), 14) == 0.0


# ---------------------------------------------------------------------------
# kill switch
# ---------------------------------------------------------------------------

def test_kill_switch_lifecycle(tmp_path):
    ks = KillSwitch(tmp_path / "halt.flag")
    assert ks.is_engaged() is False
    ks.engage("manual /kill")
    assert ks.is_engaged() is True
    assert "manual" in ks.reason()
    assert ks.clear() is True
    assert ks.is_engaged() is False
    # clearing again is a harmless no-op
    assert ks.clear() is False


def test_kill_switch_engage_is_idempotent(tmp_path):
    ks = KillSwitch(tmp_path / "halt.flag")
    ks.engage("first reason")
    ks.engage("second reason")          # must NOT overwrite the original
    assert ks.reason() == "first reason"


# ---------------------------------------------------------------------------
# dead-man firing rule (breaker-aware baseline)
# ---------------------------------------------------------------------------

def test_deadman_fires_when_gate_open_and_idle():
    now, idle_s = 100_000.0, 3 * 3600.0
    # No fill, no recent block, started 4h ago → fire.
    assert deadman_idle_exceeded(now, last_fill=0.0, last_blocked_seen=0.0,
                                 start_wall=now - 4 * 3600.0, idle_s=idle_s) is True


def test_deadman_quiet_within_window_does_not_fire():
    now, idle_s = 100_000.0, 3 * 3600.0
    assert deadman_idle_exceeded(now, last_fill=now - 3600.0, last_blocked_seen=0.0,
                                 start_wall=now - 10 * 3600.0, idle_s=idle_s) is False


def test_deadman_does_not_fire_during_breaker_halt():
    # A daily-loss halt keeps refreshing last_blocked_seen; even with no fills
    # for 10h the dead-man must NOT fire (the gate is intentionally shut).
    now, idle_s = 100_000.0, 3 * 3600.0
    assert deadman_idle_exceeded(now, last_fill=now - 10 * 3600.0,
                                 last_blocked_seen=now - 60.0,
                                 start_wall=now - 50 * 3600.0, idle_s=idle_s) is False


def test_deadman_loop_noops_when_disabled(tmp_path):
    import asyncio
    import sys
    _scripts = str(pathlib.Path(__file__).resolve().parents[1] / "scripts")
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)
    from run_volume_farmer_okx import _deadman_loop

    ks = KillSwitch(tmp_path / "halt.flag")
    stop = asyncio.Event()

    async def go():
        # idle_hours <= 0 must return immediately without engaging.
        await _deadman_loop(None, [object()], ks, "lbl", stop,
                            idle_hours=0.0, start_wall=0.0)
        assert ks.is_engaged() is False
    asyncio.run(go())


# ---------------------------------------------------------------------------
# config parsing of the new safety knobs
# ---------------------------------------------------------------------------

def test_new_safety_knobs_present_in_config(cfg):
    assert cfg["farmer"]["safety"]["deadman_hours"] == 3.0
    assert cfg["farmer"]["atr"]["sl_bps_cap"] == 1000.0
    assert cfg["farmer"]["pace"]["factor_ceiling_day1"] == 1.5


# ---------------------------------------------------------------------------
# strategy bounds (SL cap + age-ramped pace factor)
# ---------------------------------------------------------------------------

def _session_cfg(**farmer):
    base = {
        "exchange": {"symbol": "BTC_USDT", "timeframe": "5m"},
        "fees": {"maker": 0.0002, "taker": 0.0005, "rebate_pct": 0.20, "rebate_all_legs": True},
        "farmer": {
            "capital_usd": 100.0, "leverage": 15, "margin_fraction_per_trade": 0.8,
            "tp_bps": 10.0, "sl_bps": 10.0, "max_hold_bars": 2,
            "entry": {"mode": "micro_momentum", "min_bar_range_bps": 0.0, "max_bar_range_bps": 1e4},
            "alternate_direction": False,
        },
        "risk": {"daily_loss_limit_pct": 0.5, "max_drawdown_pct": 0.99,
                 "consecutive_losses_limit": 99, "stop_on_volume_target": True},
        "target": {"volume_usd": 1_000_000.0, "min_fee_cover_pct": 0.30},
    }
    base["farmer"].update(farmer)
    return base


def test_sl_bps_cap_clamps_runaway_stop():
    from execution.volume_farmer import VolumeFarmerSession
    cfg = _session_cfg(sl_bps=5000.0, atr={"relative": False, "sl_bps_cap": 1000.0})
    s = VolumeFarmerSession(config=cfg)
    s._set_pending_bracket(pd.DataFrame([{"open": 100, "high": 100, "low": 100, "close": 100}]), 100.0)  # noqa: SLF001
    assert s._pending_sl_bps == 1000.0  # noqa: SLF001  (clamped from 5000)


def test_pace_factor_age_ramp():
    from execution.volume_farmer import VolumeFarmerSession
    # max_margin high so the FACTOR CEILING (not the max bound) is what binds.
    cfg = _session_cfg(margin_fraction_per_trade=0.8, pace={
        "enabled": True, "campaign_days": 30, "min_margin_fraction": 0.1,
        "max_margin_fraction": 5.0, "warmup_trips": 0,
        "factor_ceiling_day1": 1.5, "factor_ceiling_day2": 2.0, "factor_ceiling": 3.0,
    })
    s = VolumeFarmerSession(config=cfg)
    s.total_volume_usd = 1_000.0   # far behind → raw factor is huge → ceiling binds
    s.round_trips = 50

    def frac_at(elapsed_days):
        start = pd.Timestamp("2026-04-01T00:00:00Z")
        s.campaign_start_iso = start.isoformat()
        return s._pace_margin_frac(start + pd.Timedelta(days=elapsed_days))  # noqa: SLF001

    assert frac_at(0.6) == pytest.approx(0.8 * 1.5, abs=1e-6)   # day-1 ceiling
    assert frac_at(1.5) == pytest.approx(0.8 * 2.0, abs=1e-6)   # day-2 ceiling
    assert frac_at(3.0) == pytest.approx(0.8 * 3.0, abs=1e-6)   # steady ceiling


# ---------------------------------------------------------------------------
# executor kill-switch entry gate
# ---------------------------------------------------------------------------

def test_executor_entry_blocked_when_killed(tmp_path):
    import asyncio
    from execution.live_volume_executor_okx import LiveVolumeExecutorOKX

    ks = KillSwitch(tmp_path / "halt.flag")
    ex = LiveVolumeExecutorOKX(client=None, symbol="BTC_USDT", dry_run=True, kill_switch=ks)
    assert ex._entry_blocked_reason() is None  # noqa: SLF001  (clear)
    ks.engage("manual /kill")
    reason = ex._entry_blocked_reason()  # noqa: SLF001
    assert reason is not None and reason.startswith("killed")

    class _Evt:
        kind = "entry"
        payload = {"side": "long", "price": 100.0, "notional": 50.0,
                   "tp": 101.0, "sl": 98.0, "tp_bps": 8.0, "sl_bps": 50.0, "leverage": 15.0}

    async def go():
        ex.consume_session_event(_Evt())   # killed → must NOT claim the gate
        assert ex._entry_pending is False   # noqa: SLF001
        assert ex._open_trade is None       # noqa: SLF001
    asyncio.run(go())
