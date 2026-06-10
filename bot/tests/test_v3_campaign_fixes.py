"""Regression tests for the v3 5M-campaign fixes in the volume-farmer session.

Covers the accounting + throughput bugs found in the 2026-06 audit:
  * net PnL must charge BOTH fee legs (the open fee was omitted)
  * loss-streak cooldown counts only configured reasons (time-stop scratches
    used to trigger 1h entry pauses)
  * same-bar re-entry (cycle floor 1 bar instead of 2)
  * pace controller margin scaling
  * volume-target overshoot buffer
  * rebate accrual rules (all legs vs maker-only)
  * close-leg volume counted at the exit price
  * orphan reconcile applies the time-stop
"""

from __future__ import annotations

import pandas as pd

from execution.volume_farmer import VolumeFarmerSession


def _make_config(**overrides):
	cfg = {
		"exchange": {"symbol": "BTC_USDT", "timeframe": "5m"},
		"fees": {"maker": 0.0002, "taker": 0.0005, "rebate_pct": 0.40},
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
		if isinstance(v, dict) and isinstance(cfg.get(k), dict):
			cfg[k].update(v)
		else:
			cfg[k] = v
	return cfg


def _bar(ts: str, o: float, h: float, l: float, c: float) -> dict:
	return {"open_time": pd.Timestamp(ts), "open": o, "high": h, "low": l, "close": c, "volume": 1.0}


def _feed(s: VolumeFarmerSession, bars: list) -> pd.DataFrame:
	hist = pd.DataFrame(bars[:1])
	s.on_new_candle(hist)
	for b in bars[1:]:
		hist = pd.concat([hist, pd.DataFrame([b])], ignore_index=True)
		s.on_new_candle(hist)
	return hist


# ---------------------------------------------------------------------------
# net PnL includes BOTH fee legs
# ---------------------------------------------------------------------------

def test_net_pnl_charges_open_and_close_fee():
	s = VolumeFarmerSession(config=_make_config())
	_feed(s, [
		_bar("2026-04-01T00:00:00Z", 100.0, 100.5, 99.9, 100.4),     # entry long
		_bar("2026-04-01T00:05:00Z", 100.4, 100.6, 100.3, 100.55),   # TP hit
	])
	assert s.round_trips == 1
	rec = s.ledger[-1]
	expected_net = rec["gross_pnl"] - rec["open_fee"] - rec["close_fee"]
	assert abs(rec["net_pnl"] - expected_net) < 1e-9
	# total_pnl must reconcile with the equity delta (this is exactly the
	# inflation the demo state file exposed: equity +11.10 vs pnl +22.39).
	assert abs((s.equity - s.start_equity) - s.total_pnl) < 1e-9


# ---------------------------------------------------------------------------
# loss-streak cooldown: reason-gated
# ---------------------------------------------------------------------------

def _scratch_three_times(cfg) -> VolumeFarmerSession:
	"""Three consecutive time-stop scratches (flat bars, no TP/SL touch)."""
	cfg["farmer"]["max_hold_bars"] = 1
	cfg["farmer"]["tp_bps"] = 500.0   # unreachable
	cfg["farmer"]["sl_bps"] = 500.0   # unreachable
	cfg["risk"]["consecutive_losses_limit"] = 3
	cfg["risk"]["consecutive_losses_cooldown_bars"] = 12
	s = VolumeFarmerSession(config=cfg)
	bars = []
	px = 100.0
	t0 = pd.Timestamp("2026-04-01T00:00:00Z")
	for i in range(8):
		# every bar slightly green: always a fresh long signal, never a touch
		bars.append({
			"open_time": t0 + pd.Timedelta(minutes=5 * i),
			"open": px, "high": px + 0.02, "low": px - 0.02, "close": px + 0.01,
			"volume": 1.0,
		})
	_feed(s, bars)
	return s


def test_time_stop_scratches_do_not_trigger_cooldown():
	cfg = _make_config()
	cfg["risk"]["consecutive_losses_count_reasons"] = ["sl", "sl_ambiguous"]
	s = _scratch_three_times(cfg)
	# scratches lose fees => losses recorded, but NO cooldown
	assert s.losses >= 3
	assert s.cooldown_bars_left == 0


def test_sl_losses_still_trigger_cooldown():
	cfg = _make_config()
	cfg["risk"]["consecutive_losses_count_reasons"] = ["sl", "sl_ambiguous"]
	cfg["risk"]["consecutive_losses_limit"] = 2
	cfg["risk"]["consecutive_losses_cooldown_bars"] = 12
	cfg["farmer"]["tp_bps"] = 500.0
	cfg["farmer"]["sl_bps"] = 5.0
	cfg["farmer"]["max_hold_bars"] = 99
	s = VolumeFarmerSession(config=cfg)
	bars = [
		_bar("2026-04-01T00:00:00Z", 100.0, 100.5, 99.9, 100.4),   # long entry
		_bar("2026-04-01T00:05:00Z", 100.4, 100.41, 99.0, 100.0),  # SL hit (1)
		_bar("2026-04-01T00:10:00Z", 100.0, 100.5, 99.95, 100.3),  # re-entry
		_bar("2026-04-01T00:15:00Z", 100.3, 100.31, 99.0, 99.9),   # SL hit (2)
	]
	_feed(s, bars)
	assert s.cooldown_bars_left == 12


# ---------------------------------------------------------------------------
# same-bar re-entry
# ---------------------------------------------------------------------------

def test_same_bar_reentry_opens_on_close_bar():
	cfg = _make_config()
	cfg["farmer"]["reentry_same_bar"] = True
	cfg["farmer"]["max_hold_bars"] = 1
	cfg["farmer"]["tp_bps"] = 500.0
	cfg["farmer"]["sl_bps"] = 500.0
	s = VolumeFarmerSession(config=cfg)
	_feed(s, [
		_bar("2026-04-01T00:00:00Z", 100.0, 100.1, 99.95, 100.05),  # entry
		_bar("2026-04-01T00:05:00Z", 100.05, 100.1, 100.0, 100.08), # time-stop + re-entry
	])
	assert s.round_trips == 1
	assert s.position is not None   # re-entered on the SAME closed bar


def test_no_same_bar_reentry_by_default():
	cfg = _make_config()
	cfg["farmer"]["max_hold_bars"] = 1
	cfg["farmer"]["tp_bps"] = 500.0
	cfg["farmer"]["sl_bps"] = 500.0
	s = VolumeFarmerSession(config=cfg)
	_feed(s, [
		_bar("2026-04-01T00:00:00Z", 100.0, 100.1, 99.95, 100.05),
		_bar("2026-04-01T00:05:00Z", 100.05, 100.1, 100.0, 100.08),
	])
	assert s.round_trips == 1
	assert s.position is None       # legacy: wait for the next bar


# ---------------------------------------------------------------------------
# pace controller
# ---------------------------------------------------------------------------

def test_pace_controller_scales_up_when_behind():
	cfg = _make_config()
	cfg["farmer"]["pace"] = {
		"enabled": True, "campaign_days": 30,
		"min_margin_fraction": 0.4, "max_margin_fraction": 1.6,
		"warmup_trips": 0,
	}
	s = VolumeFarmerSession(config=cfg)
	# Pretend the campaign started 10 days ago with almost no volume: we are
	# far behind, so the controller should push margin to the max bound.
	t = pd.Timestamp("2026-04-11T00:00:00Z")
	s.campaign_start_iso = "2026-04-01T00:00:00Z"
	s.total_volume_usd = 1_000.0
	s.round_trips = 50
	frac = s._pace_margin_frac(t)  # noqa: SLF001
	assert frac == 1.6


def test_pace_controller_uses_base_during_warmup():
	cfg = _make_config()
	cfg["farmer"]["pace"] = {
		"enabled": True, "campaign_days": 30,
		"min_margin_fraction": 0.4, "max_margin_fraction": 1.6,
		"warmup_trips": 10,
	}
	s = VolumeFarmerSession(config=cfg)
	t = pd.Timestamp("2026-04-11T00:00:00Z")
	s.campaign_start_iso = "2026-04-01T00:00:00Z"
	s.total_volume_usd = 1_000.0
	s.round_trips = 3                # below warmup
	assert s._pace_margin_frac(t) == 0.8  # noqa: SLF001  (base fraction)


# ---------------------------------------------------------------------------
# volume target buffer
# ---------------------------------------------------------------------------

def test_halt_waits_for_buffer_past_target():
	cfg = _make_config()
	cfg["target"]["volume_usd"] = 1_000.0
	cfg["target"]["volume_buffer_pct"] = 0.10
	s = VolumeFarmerSession(config=cfg)
	s.total_volume_usd = 1_050.0     # past target but inside the 10% buffer
	assert s._check_halt(pd.Timestamp("2026-04-01T00:00:00Z")) is False  # noqa: SLF001
	s.total_volume_usd = 1_101.0     # past target*(1+buffer)
	assert s._check_halt(pd.Timestamp("2026-04-01T00:05:00Z")) is True   # noqa: SLF001
	assert "volume_target_reached" in s.halt_reason


# ---------------------------------------------------------------------------
# rebate rules
# ---------------------------------------------------------------------------

def test_rebate_all_legs_vs_maker_only():
	cfg_all = _make_config(fees={"maker": 0.0002, "taker": 0.0005,
								 "rebate_pct": 0.40, "rebate_all_legs": True})
	cfg_mk = _make_config(fees={"maker": 0.0002, "taker": 0.0005,
								"rebate_pct": 0.40, "rebate_all_legs": False})
	s_all = VolumeFarmerSession(config=cfg_all)
	s_mk = VolumeFarmerSession(config=cfg_mk)
	assert s_all._rebate_for(10.0, "taker") == 4.0   # noqa: SLF001
	assert s_mk._rebate_for(10.0, "taker") == 0.0    # noqa: SLF001
	assert s_all._rebate_for(10.0, "maker") == 4.0   # noqa: SLF001
	assert s_mk._rebate_for(10.0, "maker") == 4.0    # noqa: SLF001


# ---------------------------------------------------------------------------
# close-leg volume at exit price
# ---------------------------------------------------------------------------

def test_close_leg_volume_uses_exit_price():
	s = VolumeFarmerSession(config=_make_config())
	_feed(s, [
		_bar("2026-04-01T00:00:00Z", 100.0, 100.5, 99.9, 100.4),     # entry long
		_bar("2026-04-01T00:05:00Z", 100.4, 100.6, 100.3, 100.55),   # TP at +10bps
	])
	rec = s.ledger[-1]
	open_leg = rec["notional"]
	close_leg = rec["notional"] * (rec["exit_price"] / rec["entry_price"])
	assert abs(s.total_volume_usd - (open_leg + close_leg)) < 1e-6
	assert s.total_volume_usd > 2 * open_leg  # TP exit => slightly MORE volume


# ---------------------------------------------------------------------------
# orphan reconcile applies the time-stop
# ---------------------------------------------------------------------------

def test_orphan_reconcile_time_stops():
	cfg = _make_config()
	cfg["farmer"]["max_hold_bars"] = 1
	cfg["farmer"]["tp_bps"] = 500.0
	cfg["farmer"]["sl_bps"] = 500.0
	s = VolumeFarmerSession(config=cfg)
	# Open a position on bar 1.
	hist = pd.DataFrame([_bar("2026-04-01T00:00:00Z", 100.0, 100.1, 99.95, 100.05)])
	s.on_new_candle(hist)
	assert s.position is not None
	# Simulate a restart: replay two quiet bars AFTER the entry. The old code
	# only checked TP/SL here, so the orphan rode on forever; now the 1-bar
	# time-stop must close it at the first replayed bar's close.
	replay = pd.DataFrame([
		{**_bar("2026-04-01T00:05:00Z", 100.05, 100.10, 100.0, 100.06), "closed": True},
		{**_bar("2026-04-01T00:10:00Z", 100.06, 100.09, 100.0, 100.04), "closed": True},
	])
	result = s.reconcile_orphan_position(replay)
	assert result == "time_stop"
	assert s.position is None
	assert s.ledger[-1]["reason"] == "time_stop"
