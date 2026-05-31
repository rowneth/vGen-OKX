"""Kill switch logic for emergency halting conditions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class KillSwitchState:
	"""Tracks heartbeat and halt state."""

	last_heartbeat_at: datetime
	halted: bool = False
	reason: str = ""


def should_halt_on_heartbeat(
	state: KillSwitchState,
	now: datetime,
	timeout_seconds: int,
) -> bool:
	"""Return whether heartbeat timeout requires halt.

	Args:
		state: Kill switch state.
		now: Current timestamp.
		timeout_seconds: Timeout threshold in seconds.

	Returns:
		True if timeout exceeded.
	"""
	return now - state.last_heartbeat_at > timedelta(seconds=timeout_seconds)
