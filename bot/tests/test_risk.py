"""Tests for risk sizing and limit logic."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from risk.limits import RiskLimits
from risk.sizing import risk_based_position_size


def test_risk_based_position_size_caps_by_risk() -> None:
	qty = risk_based_position_size(
		equity=10000.0,
		risk_per_trade_pct=0.02,
		entry_price=100.0,
		stop_price=95.0,
		max_leverage=5.0,
	)
	assert qty == 40.0


def test_risk_based_position_size_caps_by_leverage() -> None:
	qty = risk_based_position_size(
		equity=1000.0,
		risk_per_trade_pct=0.02,
		entry_price=1000.0,
		stop_price=999.0,
		max_leverage=2.0,
	)
	assert qty == 2.0


def test_consecutive_loss_pause_trigger() -> None:
	limits = RiskLimits(
		daily_drawdown_limit_pct=0.03,
		consecutive_losses_limit=3,
		consecutive_losses_pause_hours=4,
	)
	now = datetime.now(tz=timezone.utc)
	state = limits.reset_for_day(now=now, equity=10000.0)
	limits.register_trade_result(state, now=now, pnl=-10.0)
	limits.register_trade_result(state, now=now, pnl=-20.0)
	assert not limits.is_paused(state, now)
	limits.register_trade_result(state, now=now, pnl=-30.0)
	assert limits.is_paused(state, now + timedelta(minutes=1))


def test_daily_drawdown_halt() -> None:
	limits = RiskLimits(
		daily_drawdown_limit_pct=0.03,
		consecutive_losses_limit=3,
		consecutive_losses_pause_hours=4,
	)
	now = datetime.now(tz=timezone.utc)
	state = limits.reset_for_day(now=now, equity=10000.0)
	assert not limits.check_daily_drawdown(state, equity=9800.0)
	assert limits.check_daily_drawdown(state, equity=9700.0)
