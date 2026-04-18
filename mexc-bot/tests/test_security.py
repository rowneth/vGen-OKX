"""Tests for startup API permission safety checks."""

from __future__ import annotations

import asyncio
import importlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

import pytest

_security = importlib.import_module("exchange.security")
extract_permission_status = _security.extract_permission_status
verify_startup_permissions = _security.verify_startup_permissions


class _FakeClient:
	def __init__(self, payload: object) -> None:
		self._payload = payload

	async def get_api_permission_snapshot(self) -> dict:
		return {"source": "fake", "payload": self._payload}


def test_extract_permission_status_from_nested_dict() -> None:
	payload = {
		"permissions": {
			"contractTrade": True,
			"canWithdraw": False,
		}
	}
	status = extract_permission_status(payload, source="unit")
	assert status is not None
	assert status.futures_trade_enabled is True
	assert status.withdrawal_enabled is False


def test_extract_permission_status_from_permission_string() -> None:
	payload = {
		"apiPermission": "CONTRACT_TRADE,READ_ONLY",
		"withdrawEnabled": "false",
	}
	status = extract_permission_status(payload, source="unit")
	assert status is not None
	assert status.futures_trade_enabled is True
	assert status.withdrawal_enabled is False


def test_verify_startup_permissions_blocks_withdrawal_enabled() -> None:
	payload = {
		"contractTradeEnabled": True,
		"withdrawalEnabled": True,
	}
	with pytest.raises(RuntimeError):
		asyncio.run(
			verify_startup_permissions(
				client=_FakeClient(payload),
				require_futures_trade_permission=True,
				require_withdrawal_disabled=True,
			)
		)
