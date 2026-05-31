"""Live broker placeholder.

Real order placement is intentionally not implemented in this session.
"""

from __future__ import annotations


class LiveBroker:
	"""Placeholder for future live broker integration."""

	def place_order(self, *_: object, **__: object) -> None:
		"""Block live execution until later rollout phase."""
		raise NotImplementedError("Live order placement is intentionally disabled in this phase.")
