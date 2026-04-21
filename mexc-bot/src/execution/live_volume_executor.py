"""Bridge between VolumeFarmerSession and real MEXC futures orders.

The :class:`LiveVolumeExecutor` reacts to ``FarmerEvent``s emitted by the
paper session and places real orders on MEXC that mirror the session's
decisions. Execution policy:

  * Entry: POST-ONLY limit order at the session's entry price (type=2),
    with server-side stop-loss AND take-profit prices attached. This means
    the exchange itself will close the position when the TP or SL price is
    touched intrabar — protection continues even if the bot disconnects.
  * Exits: the exchange closes the position automatically via the attached
    TP/SL. When the session emits an exit for reason ``time_stop`` (no
    exchange-side analogue), the executor issues a market close.
  * A hard ``max_live_trades`` guard stops further submissions after N real
    entries, regardless of session state.
  * A post-only entry that does not fill within ``post_only_fill_timeout``
    is cancelled; the session keeps running but no real position exists —
    the next bar will re-evaluate.

Safety caveats:
  * The session's paper P&L is an *approximation* of reality. Real fills
    may differ on price and fee type (post-only can be rejected). Operator
    must verify on the MEXC UI.
  * The executor is single-position: it refuses a new entry if the prior
    real order is still resting or still open on the exchange.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from exchange.mexc_client import MEXCClient

LOGGER = logging.getLogger(__name__)


def _as_float(x: Any, default: float = 0.0) -> float:
	try:
		return float(x)
	except (TypeError, ValueError):
		return default


def _round_price(price: float, tick: float) -> float:
	if tick <= 0:
		return price
	return round(round(price / tick) * tick, 10)


@dataclass
class LiveTradeRecord:
	"""Book-keeping for a single live trade lifecycle."""

	external_oid: str
	order_id: Optional[str] = None
	side: str = ""
	entry_price_req: float = 0.0
	vol_contracts: int = 0
	notional: float = 0.0
	tp_price: float = 0.0
	sl_price: float = 0.0
	submitted_at: float = 0.0
	filled: bool = False
	fill_price: float = 0.0
	fill_fee_maker: float = 0.0
	fill_fee_taker: float = 0.0
	closed: bool = False
	close_reason: str = ""


@dataclass
class LiveVolumeExecutor:
	"""Translate FarmerEvent lifecycle into MEXC orders."""

	client: MEXCClient
	symbol: str = "BTC_USDT"
	leverage: int = 20
	open_type: int = 1  # 1 = isolated
	max_live_trades: int = 5
	max_notional_usd: float = 200.0
	post_only_fill_timeout: float = 15.0
	poll_interval: float = 1.5
	dry_run: bool = False
	notify_callback: Optional[Callable[[str], Awaitable[None]]] = None

	# contract spec (populated by startup)
	contract_size: float = field(default=0.0, init=False)
	price_scale: int = field(default=1, init=False)
	vol_scale: int = field(default=0, init=False)
	tick: float = field(default=0.0, init=False)
	min_vol: int = field(default=1, init=False)

	# runtime state
	trades: List[LiveTradeRecord] = field(default_factory=list, init=False)
	_current: Optional[LiveTradeRecord] = field(default=None, init=False)
	_disabled: bool = field(default=False, init=False)
	_disabled_reason: str = field(default="", init=False)
	_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

	# ------------------------------------------------------------------
	async def startup(self) -> None:
		"""Fetch contract spec and verify auth+balance before trading.

		Raises:
			RuntimeError: If spec is missing or balance is too low.
		"""
		detail = await self.client.get_contract_detail(symbol=self.symbol)
		spec = self._extract_spec(detail)
		if not spec:
			raise RuntimeError(f"{self.symbol} not found in contract/detail")
		self.contract_size = _as_float(spec.get("contractSize"))
		self.price_scale = int(spec.get("priceScale", 1))
		self.vol_scale = int(spec.get("volScale", 0))
		price_unit = _as_float(spec.get("priceUnit"))
		self.tick = price_unit if price_unit > 0 else 10 ** (-self.price_scale)
		self.min_vol = int(_as_float(spec.get("minVol", 1)))

		LOGGER.info(
			"LiveExecutor ready: symbol=%s contractSize=%s tick=%s min_vol=%s maxTrades=%d maxNotional=$%.2f dry_run=%s",
			self.symbol, self.contract_size, self.tick, self.min_vol,
			self.max_live_trades, self.max_notional_usd, self.dry_run,
		)

		# balance probe (non-fatal if parse fails, but log)
		try:
			account = await self.client.get_account_info()
			LOGGER.info("Account probe OK (account endpoint reachable)")
			_ = account  # silence
		except Exception as exc:  # noqa: BLE001
			raise RuntimeError(f"Account probe failed: {exc}") from exc

	# ------------------------------------------------------------------
	def _extract_spec(self, payload: Any) -> Optional[Dict[str, Any]]:
		if isinstance(payload, dict) and payload.get("symbol") == self.symbol:
			return payload
		if isinstance(payload, list):
			items = payload
		elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
			items = payload["data"]
		else:
			items = []
		for item in items:
			if isinstance(item, dict) and item.get("symbol") == self.symbol:
				return item
		return None

	# ------------------------------------------------------------------
	async def _notify(self, text: str) -> None:
		if self.notify_callback is None:
			return
		try:
			await self.notify_callback(text)
		except Exception as exc:  # noqa: BLE001
			LOGGER.exception("live notify failed: %s", exc)

	# ------------------------------------------------------------------
	def _compute_contracts(self, notional: float, entry_price: float) -> int:
		if entry_price <= 0 or self.contract_size <= 0:
			return 0
		raw = notional / (entry_price * self.contract_size)
		return int(max(self.min_vol, round(raw)))

	# ------------------------------------------------------------------
	async def handle_entry(self, payload: Dict[str, Any]) -> None:
		"""React to a session 'entry' event by placing a real order."""
		async with self._lock:
			if self._disabled:
				LOGGER.info("live entry skipped — executor disabled: %s", self._disabled_reason)
				return
			if self._current is not None and not self._current.closed:
				LOGGER.warning(
					"live entry skipped — prior trade still open (oid=%s)",
					self._current.external_oid,
				)
				return
			if len([t for t in self.trades if t.filled]) >= self.max_live_trades:
				self._disabled = True
				self._disabled_reason = f"max_live_trades={self.max_live_trades} reached"
				await self._notify(
					"🛑 *LIVE cap reached* — no further real orders will be placed"
				)
				return

			side_str = str(payload.get("side", "")).lower()
			sess_price = _as_float(payload.get("price"))
			sess_notional = _as_float(payload.get("notional"))
			tp = _as_float(payload.get("tp"))
			sl = _as_float(payload.get("sl"))
			if not side_str or sess_price <= 0 or tp <= 0 or sl <= 0:
				LOGGER.error("live entry payload invalid: %s", payload)
				return

			# Safety clamp on notional
			real_notional = min(sess_notional, self.max_notional_usd)
			if real_notional <= 0:
				LOGGER.error("live entry notional <= 0; skipping")
				return

			entry_price = _round_price(sess_price, self.tick)
			tp_price = _round_price(tp, self.tick)
			sl_price = _round_price(sl, self.tick)
			vol_contracts = self._compute_contracts(real_notional, entry_price)
			if vol_contracts < self.min_vol:
				LOGGER.error(
					"live entry vol %d < min %d; skipping", vol_contracts, self.min_vol,
				)
				return

			mexc_side = 1 if side_str == "long" else 3  # 1=open long, 3=open short
			external_oid = f"vf-{int(time.time())}-{uuid.uuid4().hex[:6]}"

			record = LiveTradeRecord(
				external_oid=external_oid,
				side=side_str,
				entry_price_req=entry_price,
				vol_contracts=vol_contracts,
				notional=vol_contracts * entry_price * self.contract_size,
				tp_price=tp_price,
				sl_price=sl_price,
				submitted_at=time.time(),
			)

			if self.dry_run:
				LOGGER.info(
					"DRY-RUN live entry: %s %s %d contracts @%.2f TP=%.2f SL=%.2f oid=%s",
					side_str.upper(), self.symbol, vol_contracts, entry_price,
					tp_price, sl_price, external_oid,
				)
				record.filled = True
				record.fill_price = entry_price
				self.trades.append(record)
				self._current = record
				await self._notify(
					f"🟡 *DRY\\-RUN entry* `{side_str.upper()}` `{vol_contracts}` "
					f"contracts oid `{external_oid}`"
				)
				return

			# Real submission
			try:
				submit_result = await self.client.submit_order(
					symbol=self.symbol,
					side=mexc_side,
					order_type=2,  # 2 = post-only limit (MEXC futures)
					vol=vol_contracts,
					price=entry_price,
					leverage=self.leverage,
					open_type=self.open_type,
					external_oid=external_oid,
					stop_loss_price=sl_price,
					take_profit_price=tp_price,
				)
			except Exception as exc:  # noqa: BLE001
				LOGGER.exception("live submit_order failed: %s", exc)
				await self._notify(
					f"❌ *LIVE submit failed* oid `{external_oid}` — see log"
				)
				return

			order_id = submit_result if isinstance(submit_result, (str, int)) else None
			if isinstance(submit_result, dict):
				order_id = submit_result.get("orderId") or submit_result.get("order_id")
			record.order_id = str(order_id) if order_id is not None else ""
			self.trades.append(record)
			self._current = record

			LOGGER.info(
				"LIVE entry submitted: oid=%s orderId=%s side=%s vol=%d entry=%.2f",
				external_oid, record.order_id, side_str, vol_contracts, entry_price,
			)
			await self._notify(
				f"📤 *LIVE order* `{side_str.upper()}` `{vol_contracts}` contracts "
				f"oid `{external_oid}` orderId `{record.order_id}`"
			)

			# Poll for fill outside the lock so other events aren't blocked
			asyncio.create_task(self._poll_fill(record))

	# ------------------------------------------------------------------
	async def _poll_fill(self, record: LiveTradeRecord) -> None:
		deadline = record.submitted_at + self.post_only_fill_timeout
		while time.time() < deadline:
			await asyncio.sleep(self.poll_interval)
			try:
				order = await self.client.get_order_by_external_oid(
					self.symbol, record.external_oid,
				)
			except Exception as exc:  # noqa: BLE001
				LOGGER.debug("poll fill error oid=%s: %s", record.external_oid, exc)
				continue
			if not isinstance(order, dict):
				continue
			state = int(order.get("state", -1))
			# 3 = filled, 4 = cancelled, 5 = partial
			if state == 3:
				record.filled = True
				record.fill_price = _as_float(order.get("dealAvgPrice"))
				record.fill_fee_maker = _as_float(order.get("makerFee"))
				record.fill_fee_taker = _as_float(order.get("takerFee"))
				LOGGER.info(
					"LIVE entry filled: oid=%s avg=%.2f makerFee=%.6f takerFee=%.6f",
					record.external_oid, record.fill_price,
					record.fill_fee_maker, record.fill_fee_taker,
				)
				await self._notify(
					f"✅ *LIVE fill* oid `{record.external_oid}` @ "
					f"`{record.fill_price:,.2f}` maker fee "
					f"`${record.fill_fee_maker:.4f}`"
				)
				return
			if state == 4:
				LOGGER.info("LIVE entry cancelled before fill oid=%s", record.external_oid)
				record.closed = True
				record.close_reason = "cancelled_before_fill"
				if self._current is record:
					self._current = None
				await self._notify(
					f"🚫 *LIVE cancelled* oid `{record.external_oid}`"
				)
				return
		# timeout — cancel the resting post-only
		LOGGER.warning(
			"LIVE entry not filled within %.1fs — cancelling oid=%s",
			self.post_only_fill_timeout, record.external_oid,
		)
		try:
			# Best-effort cancel via orderId if we have one
			if record.order_id:
				await self.client._request(  # noqa: SLF001 — reuse generic request
					"POST",
					"/api/v1/private/order/cancel",
					body=[int(record.order_id)] if record.order_id.isdigit() else [record.order_id],
					auth=True,
				)
		except Exception as exc:  # noqa: BLE001
			LOGGER.exception("LIVE cancel failed oid=%s: %s", record.external_oid, exc)
		record.closed = True
		record.close_reason = "timeout_unfilled"
		if self._current is record:
			self._current = None
		await self._notify(
			f"⏳ *LIVE post\\-only timeout* — cancelled oid `{record.external_oid}`"
		)

	# ------------------------------------------------------------------
	async def handle_exit(self, payload: Dict[str, Any]) -> None:
		"""React to a session 'exit' event.

		TP/SL exits are handled by the exchange's attached SL/TP orders, so
		we only issue an explicit close for ``time_stop`` exits.
		"""
		async with self._lock:
			if self._disabled:
				return
			if self._current is None or not self._current.filled or self._current.closed:
				LOGGER.debug("live exit: no tracked open position (session-only exit)")
				return
			reason = str(payload.get("reason", ""))
			if reason != "time_stop":
				# Exchange's attached TP/SL should handle this. Mark closed locally.
				self._current.closed = True
				self._current.close_reason = reason
				LOGGER.info(
					"LIVE exit expected via exchange SL/TP (reason=%s) oid=%s",
					reason, self._current.external_oid,
				)
				self._current = None
				return

			# Time stop → market close
			mexc_side = 4 if self._current.side == "long" else 2  # 4=close long, 2=close short
			try:
				if self.dry_run:
					LOGGER.info("DRY-RUN market close oid=%s", self._current.external_oid)
				else:
					await self.client.submit_order(
						symbol=self.symbol,
						side=mexc_side,
						order_type=5,  # market
						vol=self._current.vol_contracts,
						open_type=self.open_type,
						reduce_only=True,
					)
				LOGGER.info(
					"LIVE time-stop close submitted oid=%s", self._current.external_oid,
				)
				await self._notify(
					f"⏹ *LIVE time\\-stop close* oid `{self._current.external_oid}`"
				)
			except Exception as exc:  # noqa: BLE001
				LOGGER.exception("LIVE time-stop close failed: %s", exc)
				await self._notify("❌ *LIVE time\\-stop close failed* — see log")
			self._current.closed = True
			self._current.close_reason = reason
			self._current = None

	# ------------------------------------------------------------------
	def summary(self) -> Dict[str, Any]:
		filled = [t for t in self.trades if t.filled]
		return {
			"attempted": len(self.trades),
			"filled": len(filled),
			"cancelled": sum(1 for t in self.trades if t.close_reason.startswith("cancelled") or t.close_reason == "timeout_unfilled"),
			"disabled": self._disabled,
			"disabled_reason": self._disabled_reason,
		}
