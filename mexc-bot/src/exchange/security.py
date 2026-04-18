"""Startup security checks for API permission enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from exchange.mexc_client import MEXCClient


@dataclass(frozen=True)
class ApiPermissionStatus:
	"""Normalized API permission status for startup safety checks."""

	futures_trade_enabled: bool
	withdrawal_enabled: bool
	source: str
	raw: Dict[str, Any]


async def verify_startup_permissions(
	client: MEXCClient,
	require_futures_trade_permission: bool,
	require_withdrawal_disabled: bool,
) -> ApiPermissionStatus:
	"""Verify API key permissions and fail closed if verification is ambiguous.

	Args:
		client: Initialized MEXC REST client.
		require_futures_trade_permission: Whether futures trade permission is mandatory.
		require_withdrawal_disabled: Whether withdrawal permission must be disabled.

	Returns:
		Normalized permission status.

	Raises:
		RuntimeError: If permissions cannot be verified or violate policy.
	"""
	snapshot = await client.get_api_permission_snapshot()
	source = str(snapshot.get("source", "unknown"))
	payload = snapshot.get("payload", {})
	status = extract_permission_status(payload, source=source)
	if status is None:
		raise RuntimeError(
			"Unable to verify API key permissions from exchange response. "
			"Refusing to start to protect funds."
		)

	if require_futures_trade_permission and not status.futures_trade_enabled:
		raise RuntimeError("API key is missing futures-trade permission; refusing to start.")

	if require_withdrawal_disabled and status.withdrawal_enabled:
		raise RuntimeError("API key has withdrawal permission enabled; refusing to start.")

	return status


def extract_permission_status(payload: Any, source: str = "unknown") -> Optional[ApiPermissionStatus]:
	"""Extract normalized permission flags from varied API payload shapes.

	Args:
		payload: Raw payload returned by exchange endpoint.
		source: Endpoint identifier used for extraction.

	Returns:
		ApiPermissionStatus when inferable, otherwise None.
	"""
	flat: Dict[str, Any] = {}
	_flatten_payload(payload=payload, prefix="", out=flat)

	futures_trade_enabled: Optional[bool] = None
	withdrawal_enabled: Optional[bool] = None

	for key, value in flat.items():
		normalized_key = key.lower().replace("-", "_")
		parsed = _parse_bool_like(value)
		if parsed is None:
			continue

		if _key_matches_withdrawal(normalized_key):
			withdrawal_enabled = parsed
		if _key_matches_futures_trade(normalized_key):
			futures_trade_enabled = parsed

	if futures_trade_enabled is None:
		futures_trade_enabled = _extract_from_permission_strings(flat, target="futures_trade")
	if withdrawal_enabled is None:
		withdrawal_enabled = _extract_from_permission_strings(flat, target="withdrawal")

	if futures_trade_enabled is None or withdrawal_enabled is None:
		return None

	return ApiPermissionStatus(
		futures_trade_enabled=futures_trade_enabled,
		withdrawal_enabled=withdrawal_enabled,
		source=source,
		raw={"flattened": flat},
	)


def _flatten_payload(payload: Any, prefix: str, out: Dict[str, Any]) -> None:
	if isinstance(payload, dict):
		for key, value in payload.items():
			next_prefix = f"{prefix}.{key}" if prefix else str(key)
			_flatten_payload(payload=value, prefix=next_prefix, out=out)
		return
	if isinstance(payload, list):
		for index, item in enumerate(payload):
			next_prefix = f"{prefix}[{index}]"
			_flatten_payload(payload=item, prefix=next_prefix, out=out)
		return
	out[prefix] = payload


def _parse_bool_like(value: Any) -> Optional[bool]:
	if isinstance(value, bool):
		return value
	if isinstance(value, int) and value in {0, 1}:
		return bool(value)
	if isinstance(value, str):
		v = value.strip().lower()
		if v in {"true", "1", "yes", "enabled", "allow", "allowed", "on"}:
			return True
		if v in {"false", "0", "no", "disabled", "deny", "denied", "off"}:
			return False
	return None


def _key_matches_futures_trade(normalized_key: str) -> bool:
	patterns = [
		"contract_trade",
		"contracttrade",
		"future_trade",
		"futures_trade",
		"futurestrade",
		"futurestrading",
		"can_trade",
		"cantrade",
		"trade_enabled",
		"tradeenabled",
	]
	return any(pattern in normalized_key for pattern in patterns)


def _key_matches_withdrawal(normalized_key: str) -> bool:
	patterns = [
		"withdraw",
		"withdrawal",
		"can_withdraw",
		"canwithdraw",
		"withdraw_enabled",
		"withdrawenabled",
	]
	return any(pattern in normalized_key for pattern in patterns)


def _extract_from_permission_strings(flat: Dict[str, Any], target: str) -> Optional[bool]:
	for key, value in flat.items():
		normalized_key = key.lower().replace("-", "_")
		if "permission" not in normalized_key and "perm" not in normalized_key:
			continue
		if not isinstance(value, str):
			continue
		tokens = {token.strip().lower() for token in value.replace(";", ",").split(",") if token.strip()}
		if target == "futures_trade":
			if "contract_trade" in tokens or "futures_trade" in tokens:
				return True
		if target == "withdrawal":
			if "withdraw" in tokens or "withdrawal" in tokens:
				return True
	if target == "withdrawal":
		return False
	return None
