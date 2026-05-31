"""Working-capital clamp for the volume-farmer bot.

The bot is funded with a *single* MEXC futures wallet.  We do **not** move funds
between wallets — that would require manual transfers and risk leaving funds
stranded.  Instead this clamp implements a **logical** split:

* ``working_capital`` — the slice the trading session is allowed to size from
  (default $30).  ``session.equity`` is pinned to this value.
* ``reserve``         — everything else in the wallet.  Held back so a losing
  streak does not auto-compound losses by raising leverage off a bigger pot.

Behaviour:

1. **Top-up on loss** — if ``working`` falls below ``target`` and ``reserve``
   has funds, we logically transfer the deficit from ``reserve`` -> ``working``
   so the bot keeps trading at full size.
2. **No auto profit sweep** — winnings stay in ``working`` until the operator
   intervenes (per user request, top-up only).
3. **Deposit detection** — when total wallet grows, the surplus goes to
   ``reserve`` (not ``working``); ``working`` only changes via top-up.
4. **Reserve depletion** — when ``reserve`` reaches 0, ``working`` is rebound
   to the actual remaining wallet balance and we run "single-pot" until the
   operator deposits again.

Telegram notifications are emitted via the supplied callable on every state
change (top-up, deposit, depletion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

NotifyFn = Callable[[str], Awaitable[None]]


@dataclass
class CapitalClamp:
	"""Pure logic — no I/O.  Caller polls wallet, calls ``observe()``."""

	working_target: float = 30.0  # USDT — desired working_capital
	min_topup_step: float = 0.50   # ignore sub-cent fluctuations
	working_capital: float = 0.0
	reserve: float = 0.0
	last_total: float = 0.0
	single_pot: bool = False       # True once reserve permanently depleted
	_initialised: bool = False

	# ------------------------------------------------------------------
	def initialise(self, wallet_total: float) -> None:
		"""Seed the clamp from the first wallet snapshot at process start.

		If the wallet already exceeds ``working_target``, the surplus becomes
		``reserve``.  Otherwise the entire wallet is the working pot and we
		go straight into single-pot mode.
		"""
		wallet_total = max(0.0, float(wallet_total))
		self.last_total = wallet_total
		if wallet_total >= self.working_target:
			self.working_capital = self.working_target
			self.reserve = wallet_total - self.working_target
			self.single_pot = False
		else:
			self.working_capital = wallet_total
			self.reserve = 0.0
			self.single_pot = True
		self._initialised = True

	# ------------------------------------------------------------------
	def observe(self, wallet_total: float) -> list[str]:
		"""Reconcile the clamp with the latest wallet snapshot.

		Returns a list of human-readable Telegram-ready messages describing
		state changes (deposit, top-up, depletion).  Each message is plain
		text — caller is responsible for MarkdownV2 escaping.
		"""
		if not self._initialised:
			self.initialise(wallet_total)
			return [
				f"🔒 Capital clamp armed: working ${self.working_capital:,.2f} / "
				f"reserve ${self.reserve:,.2f} "
				f"({'single-pot' if self.single_pot else 'split'})"
			]

		wallet_total = max(0.0, float(wallet_total))
		messages: list[str] = []
		delta = wallet_total - self.last_total

		# --- 1. Deposit detection (positive delta beyond threshold) -------
		if delta >= self.min_topup_step:
			# Anything above the working_target lands in reserve.  If we are
			# in single-pot mode, the deposit first refills working back up
			# to target, the rest goes to reserve and we exit single-pot.
			if self.single_pot:
				deficit = max(0.0, self.working_target - self.working_capital)
				to_working = min(delta, deficit)
				to_reserve = delta - to_working
				self.working_capital += to_working
				self.reserve += to_reserve
				if self.reserve > 0 or self.working_capital >= self.working_target:
					self.single_pot = False
			else:
				self.reserve += delta
			messages.append(
				f"💰 Deposit detected: +${delta:,.2f} (wallet ${wallet_total:,.2f}). "
				f"Working ${self.working_capital:,.2f} / "
				f"Reserve ${self.reserve:,.2f}."
			)

		# --- 2. Top-up: working below target, reserve has funds -----------
		if not self.single_pot and self.working_capital + 1e-9 < self.working_target:
			deficit = self.working_target - self.working_capital
			topup = min(deficit, self.reserve)
			if topup >= self.min_topup_step:
				self.working_capital += topup
				self.reserve -= topup
				messages.append(
					f"🔁 Top-up from reserve: +${topup:,.2f}. "
					f"Working ${self.working_capital:,.2f} / "
					f"Reserve ${self.reserve:,.2f}."
				)

		# --- 3. Reserve depletion ------------------------------------------
		if not self.single_pot and self.reserve <= self.min_topup_step:
			self.reserve = 0.0
			# Only flip to single-pot once working has also fallen below
			# target — i.e. the operator is now riding the actual wallet.
			if self.working_capital + 1e-9 < self.working_target:
				self.single_pot = True
				# Rebind working to whatever cash actually remains.
				self.working_capital = wallet_total
				messages.append(
					f"⚠️ Reserve empty. Switching to single-pot mode at "
					f"${self.working_capital:,.2f}. Future losses will "
					f"reduce trade size directly."
				)

		# --- 4. In single-pot mode, working tracks the wallet --------------
		if self.single_pot:
			self.working_capital = wallet_total

		self.last_total = wallet_total
		return messages

	# ------------------------------------------------------------------
	def snapshot(self) -> dict:
		return {
			"working_capital": round(self.working_capital, 4),
			"reserve": round(self.reserve, 4),
			"last_total": round(self.last_total, 4),
			"working_target": self.working_target,
			"single_pot": self.single_pot,
		}


# ----------------------------------------------------------------------
async def emit(notify: Optional[NotifyFn], escape, messages: list[str]) -> None:
	"""Send each message through Telegram with proper MarkdownV2 escape.

	``escape`` should be ``TelegramNotifier.escape``.
	"""
	if notify is None or not messages:
		return
	for m in messages:
		try:
			await notify(escape(m))
		except Exception:  # noqa: BLE001
			# never let a notification kill the bot
			pass
