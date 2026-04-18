"""Entrypoint for live trading phase with hard safety guard."""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from exchange.mexc_client import MEXCClient
from exchange.security import verify_startup_permissions


def _load_config() -> dict:
	with (PROJECT_ROOT / "config" / "config.yaml").open("r", encoding="utf-8") as handle:
		return yaml.safe_load(handle)


def _require_env(name: str) -> str:
	value = os.getenv(name, "").strip()
	if not value:
		raise RuntimeError(f"Required environment variable {name} is missing.")
	return value


async def _startup_checks() -> None:
	config = _load_config()
	load_dotenv(PROJECT_ROOT / ".env", override=False)

	api_key = _require_env("MEXC_API_KEY")
	api_secret = _require_env("MEXC_API_SECRET")

	startup_cfg = config["risk"]["startup_checks"]
	require_futures = bool(startup_cfg["require_futures_trade_permission"])
	require_withdraw_disabled = bool(startup_cfg["require_withdrawal_disabled"])

	async with MEXCClient(
		api_key=api_key,
		api_secret=api_secret,
		base_url=str(config["exchange"]["base_url_rest"]),
		timeout_seconds=int(config["exchange"]["request_timeout_seconds"]),
		requests_per_second=float(config["exchange"]["rate_limits"]["requests_per_second"]),
		burst_capacity=int(config["exchange"]["rate_limits"]["burst_capacity"]),
	) as client:
		status = await verify_startup_permissions(
			client=client,
			require_futures_trade_permission=require_futures,
			require_withdrawal_disabled=require_withdraw_disabled,
		)

	print(
		"Startup checks passed. "
		f"futures_trade_enabled={status.futures_trade_enabled} "
		f"withdrawal_enabled={status.withdrawal_enabled} source={status.source}"
	)


def main() -> None:
	"""Run live trading placeholder with explicit operator confirmation."""
	ack = os.getenv("LIVE_TRADING_ACKNOWLEDGED", "")
	if ack != "I_UNDERSTAND_AND_ACCEPT_RISK":
		raise RuntimeError(
			"Live trading is blocked. Set LIVE_TRADING_ACKNOWLEDGED=I_UNDERSTAND_AND_ACCEPT_RISK "
			"only after completing backtest and paper phases."
		)

	asyncio.run(_startup_checks())

	raise RuntimeError("Live trading is intentionally not implemented in this session.")


if __name__ == "__main__":
	main()
