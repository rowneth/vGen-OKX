"""Tests for the volume-farmer session."""

from __future__ import annotations

import pandas as pd
import pytest

from execution.volume_farmer import VolumeFarmerSession


def _make_config(**overrides):
	cfg = {
		"exchange": {"symbol": "BTC_USDT", "timeframe": "5m"},
		"fees": {"maker": 0.0001, "taker": 0.0005, "rebate_pct": 0.70},
		"farmer": {
			"capital_usd": 100.0,
			"leverage": 20,
			"margin_fraction_per_trade": 0.8,
			"tp_bps": 10.0,
			"sl_bps": 10.0,
			"max_hold_bars": 2,
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
	for k, v in overrides.items():
		cfg[k] = v
	return cfg


def _bar(ts: str, o: float, h: float, l: float, c: float) -> dict:
	return {"open_time": pd.Timestamp(ts), "open": o, "high": h, "low": l, "close": c, "volume": 1.0}


def test_entry_emits_on_green_bar_then_tp_hit():
	events = []
	s = VolumeFarmerSession(config=_make_config(), event_callback=events.append)
	# Bar 1: green (entry trigger)
	hist = pd.DataFrame([_bar("2026-04-01T00:00:00Z", 100.0, 100.5, 99.9, 100.4)])
	s.on_new_candle(hist)
	assert s.position is not None and s.position.side == "long"
	# Bar 2: spikes high enough to hit TP (+0.10% = 100.50)
	hist = pd.concat([hist, pd.DataFrame([_bar("2026-04-01T00:05:00Z", 100.4, 100.6, 100.3, 100.55)])], ignore_index=True)
	s.on_new_candle(hist)
	assert s.position is None
	assert s.wins == 1
	assert s.round_trips == 1
	kinds = [e.kind for e in events]
	assert "entry" in kinds and "exit" in kinds


def test_sl_hit_counts_as_loss():
	s = VolumeFarmerSession(config=_make_config())
	# Green bar → long
	hist = pd.DataFrame([_bar("2026-04-01T00:00:00Z", 100.0, 100.5, 99.9, 100.4)])
	s.on_new_candle(hist)
	assert s.position is not None
	# Next bar dumps through SL (-0.10% from entry 100.4 = 100.30) — low 99.8
	hist = pd.concat([hist, pd.DataFrame([_bar("2026-04-01T00:05:00Z", 100.4, 100.45, 99.8, 100.2)])], ignore_index=True)
	s.on_new_candle(hist)
	assert s.losses == 1
	assert s.wins == 0


def test_volume_target_halts():
	cfg = _make_config()
	cfg["target"]["volume_usd"] = 3_000.0  # tiny target so we hit it instantly
	s = VolumeFarmerSession(config=cfg)
	hist = pd.DataFrame([_bar("2026-04-01T00:00:00Z", 100.0, 100.5, 99.9, 100.4)])
	s.on_new_candle(hist)
	# entry opened $1600 notional → already over $3k? No, need round-trip.
	hist = pd.concat([hist, pd.DataFrame([_bar("2026-04-01T00:05:00Z", 100.4, 100.6, 100.3, 100.55)])], ignore_index=True)
	s.on_new_candle(hist)
	# After round-trip: volume = 2 * 1600 = 3200 >= 3000 → halt on next bar
	hist = pd.concat([hist, pd.DataFrame([_bar("2026-04-01T00:10:00Z", 100.55, 100.7, 100.4, 100.6)])], ignore_index=True)
	s.on_new_candle(hist)
	assert s.halted is True
	assert "volume_target_reached" in s.halt_reason
