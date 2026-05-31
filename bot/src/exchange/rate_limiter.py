"""Async token bucket rate limiter for exchange REST calls."""

from __future__ import annotations

import asyncio
import time


class TokenBucketRateLimiter:
	"""A simple async token bucket.

	This limiter is intentionally minimal and predictable for auditability.
	"""

	def __init__(self, rate_per_second: float, burst_capacity: int) -> None:
		"""Initialize a token bucket limiter.

		Args:
			rate_per_second: Number of tokens added per second.
			burst_capacity: Maximum number of tokens the bucket can hold.
		"""
		if rate_per_second <= 0:
			raise ValueError("rate_per_second must be > 0")
		if burst_capacity <= 0:
			raise ValueError("burst_capacity must be > 0")

		self._rate_per_second = rate_per_second
		self._capacity = float(burst_capacity)
		self._tokens = float(burst_capacity)
		self._last_refill = time.monotonic()
		self._lock = asyncio.Lock()

	async def acquire(self, tokens: float = 1.0) -> None:
		"""Wait until enough tokens are available and consume them.

		Args:
			tokens: Number of tokens to consume.
		"""
		if tokens <= 0:
			raise ValueError("tokens must be > 0")

		while True:
			async with self._lock:
				self._refill()
				if self._tokens >= tokens:
					self._tokens -= tokens
					return
				missing = tokens - self._tokens
				sleep_for = missing / self._rate_per_second
			await asyncio.sleep(sleep_for)

	def _refill(self) -> None:
		now = time.monotonic()
		elapsed = now - self._last_refill
		if elapsed <= 0:
			return
		self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_per_second)
		self._last_refill = now
