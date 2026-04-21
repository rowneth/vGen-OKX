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
	tp_bps: float = 0.0
	sl_bps: float = 0.0
	leverage: int = 0
	mexc_side: int = 0
	reprice_attempts: int = 0
	submitted_at: float = 0.0
	filled: bool = False
	fill_price: float = 0.0
	fill_fee_maker: float = 0.0
	fill_fee_taker: float = 0.0
	fill_time_ms: int = 0
	closed: bool = False
	close_reason: str = ""


@dataclass
class LiveVolumeExecutor:
	"""Translate FarmerEvent lifecycle into MEXC orders."""

	client: MEXCClient
	symbol: str = "BTC_USDT"
	leverage: int = 125        # max cap — actual leverage comes from session payload
	open_type: int = 1         # 1 = isolated
	max_live_trades: int = 5
	max_notional_usd: float = 5000.0  # high cap — session controls real sizing
	post_only_fill_timeout: float = 60.0
	poll_interval: float = 1.5
	reprice_drift_ticks: int = 3   # cancel & repost if market drifts this many ticks from our price
	max_reprice_attempts: int = 10
	position_watch_interval: float = 3.0   # how often to poll open_positions after fill
	position_watch_timeout: float = 3600.0  # give up watching after this many seconds
	dry_run: bool = False
	notify_callback: Optional[Callable[[str], Awaitable[None]]] = None
	real_close_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None

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
			tp_bps = _as_float(payload.get("tp_bps"))
			sl_bps = _as_float(payload.get("sl_bps"))
			if not side_str or sess_price <= 0 or tp_bps <= 0 or sl_bps <= 0:
				LOGGER.error("live entry payload invalid: %s", payload)
				return

			# Fetch live bid/ask so we place at the current market, not stale bar close.
			try:
				ticker = await self.client.get_ticker(self.symbol)
				if isinstance(ticker, list) and ticker:
					ticker = ticker[0]
				live_bid = _as_float(ticker.get("bid1") or ticker.get("bidPrice") or ticker.get("bid"))
				live_ask = _as_float(ticker.get("ask1") or ticker.get("askPrice") or ticker.get("ask"))
				live_last = _as_float(ticker.get("lastPrice") or ticker.get("last"))
			except Exception as exc:  # noqa: BLE001
				LOGGER.warning("ticker fetch failed, falling back to bar close: %s", exc)
				live_bid = live_ask = live_last = sess_price

			# POST-ONLY SAFE: place ONE TICK INSIDE the book on the passive side.
			#   long  -> best_bid - 1 tick   (can never cross the ask -> MEXC accepts)
			#   short -> best_ask + 1 tick   (can never cross the bid -> MEXC accepts)
			# This eliminates the "submitted at bid but ask lifted in-flight -> rejected"
			# race that causes immediate 'LIVE cancelled' on volatile BTC ticks.
			if side_str == "long":
				raw_price = (live_bid - self.tick) if live_bid > 0 else 0.0
			else:
				raw_price = (live_ask + self.tick) if live_ask > 0 else 0.0
			if raw_price <= 0:
				raw_price = live_last if live_last > 0 else sess_price

			# Recalculate TP/SL from the live entry price
			tp = raw_price * (1 + tp_bps / 10_000) if side_str == "long" else raw_price * (1 - tp_bps / 10_000)
			sl = raw_price * (1 - sl_bps / 10_000) if side_str == "long" else raw_price * (1 + sl_bps / 10_000)

			# Safety clamp on notional
			real_notional = min(sess_notional, self.max_notional_usd)
			if real_notional <= 0:
				LOGGER.error("live entry notional <= 0; skipping")
				return

			# Use LIVE bid/ask (raw_price) for the actual order — not stale bar close
			entry_price = _round_price(raw_price, self.tick)
			tp_price = _round_price(tp, self.tick)
			sl_price = _round_price(sl, self.tick)
			vol_contracts = self._compute_contracts(real_notional, entry_price)
			if vol_contracts < self.min_vol:
				LOGGER.error(
					"live entry vol %d < min %d; skipping", vol_contracts, self.min_vol,
				)
				return

			# Use session-computed dynamic leverage; cap at exchange max (self.leverage)
			sess_leverage = int(round(_as_float(payload.get("leverage", self.leverage))))
			order_leverage = min(sess_leverage, self.leverage)

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
				tp_bps=tp_bps,
				sl_bps=sl_bps,
				leverage=order_leverage,
				mexc_side=mexc_side,
				submitted_at=time.time(),
			)

			if self.dry_run:
				LOGGER.info(
					"DRY-RUN live entry: %s %s %d contracts @%.2f TP=%.2f SL=%.2f lev=%dx oid=%s",
					side_str.upper(), self.symbol, vol_contracts, entry_price,
					tp_price, sl_price, order_leverage, external_oid,
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
					leverage=order_leverage,
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
				"LIVE entry submitted: oid=%s orderId=%s side=%s vol=%d entry=%.2f lev=%dx",
				external_oid, record.order_id, side_str, vol_contracts, entry_price, order_leverage,
			)
			await self._notify(
				f"📤 *LIVE order* `{side_str.upper()}` `{vol_contracts}` contracts "
				f"oid `{external_oid}` orderId `{record.order_id}`"
			)

			# Poll for fill outside the lock so other events aren't blocked
			asyncio.create_task(self._poll_fill(record))

	# ------------------------------------------------------------------
	async def _cancel_order(self, record: LiveTradeRecord) -> None:
		"""Best-effort cancel of the resting order on MEXC."""
		if not record.order_id:
			return
		try:
			await self.client._request(  # noqa: SLF001
				"POST",
				"/api/v1/private/order/cancel",
				body=[int(record.order_id)] if record.order_id.isdigit() else [record.order_id],
				auth=True,
			)
		except Exception as exc:  # noqa: BLE001
			LOGGER.exception("LIVE cancel failed oid=%s: %s", record.external_oid, exc)

	# ------------------------------------------------------------------
	async def _reprice(self, record: LiveTradeRecord, new_price: float) -> bool:
		"""Cancel the current order and resubmit at ``new_price``.

		Returns True if a new order was placed.
		"""
		await self._cancel_order(record)
		# Re-enter one tick INSIDE the book on the passive side so post-only
		# cannot be rejected by a cross on arrival.
		if record.side == "long":
			passive = new_price - self.tick
			tp = passive * (1 + record.tp_bps / 10_000)
			sl = passive * (1 - record.sl_bps / 10_000)
		else:
			passive = new_price + self.tick
			tp = passive * (1 - record.tp_bps / 10_000)
			sl = passive * (1 + record.sl_bps / 10_000)

		new_entry = _round_price(passive, self.tick)
		new_tp = _round_price(tp, self.tick)
		new_sl = _round_price(sl, self.tick)
		new_oid = f"vf-{int(time.time())}-{uuid.uuid4().hex[:6]}"

		try:
			submit_result = await self.client.submit_order(
				symbol=self.symbol,
				side=record.mexc_side,
				order_type=2,
				vol=record.vol_contracts,
				price=new_entry,
				leverage=record.leverage,
				open_type=self.open_type,
				external_oid=new_oid,
				stop_loss_price=new_sl,
				take_profit_price=new_tp,
			)
		except Exception as exc:  # noqa: BLE001
			LOGGER.exception("LIVE reprice submit failed oid=%s: %s", record.external_oid, exc)
			return False

		order_id = submit_result if isinstance(submit_result, (str, int)) else None
		if isinstance(submit_result, dict):
			order_id = submit_result.get("orderId") or submit_result.get("order_id")

		old_oid = record.external_oid
		record.external_oid = new_oid
		record.order_id = str(order_id) if order_id is not None else ""
		record.entry_price_req = new_entry
		record.tp_price = new_tp
		record.sl_price = new_sl
		record.submitted_at = time.time()
		record.reprice_attempts += 1

		LOGGER.info(
			"LIVE reprice #%d: old_oid=%s new_oid=%s new_price=%.2f",
			record.reprice_attempts, old_oid, new_oid, new_entry,
		)
		return True

	# ------------------------------------------------------------------
	async def _poll_fill(self, record: LiveTradeRecord) -> None:
		while True:
			deadline = record.submitted_at + self.post_only_fill_timeout
			repriced = False
			while time.time() < deadline:
				await asyncio.sleep(self.poll_interval)
				# Check fill state
				try:
					order = await self.client.get_order_by_external_oid(
						self.symbol, record.external_oid,
					)
				except Exception as exc:  # noqa: BLE001
					LOGGER.debug("poll fill error oid=%s: %s", record.external_oid, exc)
					order = None
				if isinstance(order, dict):
					state = int(order.get("state", -1))
					if state == 3:
						record.filled = True
						record.fill_price = _as_float(order.get("dealAvgPrice"))
						record.fill_fee_maker = _as_float(order.get("makerFee"))
						record.fill_fee_taker = _as_float(order.get("takerFee"))
						record.fill_time_ms = int(time.time() * 1000)
						LOGGER.info(
							"LIVE entry filled: oid=%s avg=%.2f makerFee=%.6f takerFee=%.6f reprices=%d",
							record.external_oid, record.fill_price,
							record.fill_fee_maker, record.fill_fee_taker,
							record.reprice_attempts,
						)
						await self._notify(
							f"✅ *LIVE fill* oid `{record.external_oid}` @ "
							f"`{record.fill_price:,.2f}` maker fee "
							f"`${record.fill_fee_maker:.4f}`"
						)
						# Start watching the exchange position for real-time close detection
						asyncio.create_task(self._watch_position_close(record))
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

				# Check drift — if market moved past our price, reprice
				if record.reprice_attempts >= self.max_reprice_attempts:
					continue
				try:
					ticker = await self.client.get_ticker(self.symbol)
					if isinstance(ticker, list) and ticker:
						ticker = ticker[0]
					best = _as_float(
						(ticker.get("bid1") or ticker.get("bidPrice") or ticker.get("bid"))
						if record.side == "long"
						else (ticker.get("ask1") or ticker.get("askPrice") or ticker.get("ask"))
					)
				except Exception:  # noqa: BLE001
					best = 0.0
				if best <= 0 or self.tick <= 0:
					continue
				# Compare against the PASSIVE target (one tick inside current book),
				# not the raw best, since we always rest one tick inside.
				if record.side == "long":
					passive_target = best - self.tick
					behind = passive_target > record.entry_price_req
				else:
					passive_target = best + self.tick
					behind = passive_target < record.entry_price_req
				drift = abs(passive_target - record.entry_price_req) / self.tick
				if behind and drift >= self.reprice_drift_ticks:
					LOGGER.info(
						"LIVE drift %.1f ticks — repricing oid=%s from %.2f to %.2f",
						drift, record.external_oid, record.entry_price_req, best,
					)
					if await self._reprice(record, best):
						repriced = True
						break  # restart outer with new deadline

			if repriced:
				continue  # retry with fresh deadline after reprice
			break  # timed out — fall through to cancel path

		# timeout — cancel the resting post-only
		LOGGER.warning(
			"LIVE entry not filled (reprices=%d) — cancelling oid=%s",
			record.reprice_attempts, record.external_oid,
		)
		await self._cancel_order(record)
		record.closed = True
		record.close_reason = "timeout_unfilled"
		if self._current is record:
			self._current = None
		await self._notify(
			f"⏳ *LIVE post\\-only timeout* — cancelled oid `{record.external_oid}` "
			f"\\(after {record.reprice_attempts} reprice attempts\\)"
		)

	# ------------------------------------------------------------------
	async def _watch_position_close(self, record: LiveTradeRecord) -> None:
		"""Poll open_positions after fill; when position disappears, find the
		closing order in history and fire ``real_close_callback``.

		This gives us real-time TP/SL exit detection (MEXC closes the position
		via attached TP/SL orders the instant price is touched), instead of
		waiting for the next 5m bar to tell us.
		"""
		start = time.time()
		position_seen = False
		while time.time() - start < self.position_watch_timeout:
			await asyncio.sleep(self.position_watch_interval)
			try:
				payload = await self.client.get_open_positions(symbol=self.symbol)
			except Exception as exc:  # noqa: BLE001
				LOGGER.debug("open_positions poll failed: %s", exc)
				continue

			items = payload if isinstance(payload, list) else []
			if isinstance(payload, dict):
				items = payload.get("data") if isinstance(payload.get("data"), list) else []

			# Find our position: matching symbol + side
			our_side_code = 1 if record.side == "long" else 2  # MEXC position_type: 1=long, 2=short
			found = None
			for pos in items or []:
				if not isinstance(pos, dict):
					continue
				if pos.get("symbol") != self.symbol:
					continue
				if int(pos.get("positionType", pos.get("position_type", 0))) != our_side_code:
					continue
				vol = _as_float(pos.get("holdVol", pos.get("hold_vol", 0)))
				if vol > 0:
					found = pos
					break

			if found is not None:
				position_seen = True
				continue

			# Position no longer open
			if not position_seen:
				# Race: our fill may not yet be reflected in positions. Keep waiting a bit.
				if time.time() - start < 10.0:
					continue
				LOGGER.warning(
					"watch_position: never saw position for oid=%s — aborting watcher",
					record.external_oid,
				)
				return

			# Position closed — fetch history orders to find the close
			await self._notify_real_close(record)
			return

		LOGGER.warning(
			"watch_position: timeout after %.0fs oid=%s", self.position_watch_timeout, record.external_oid,
		)

	# ------------------------------------------------------------------
	async def _notify_real_close(self, record: LiveTradeRecord) -> None:
		"""Fetch the close order from history and invoke ``real_close_callback``."""
		close_side = 4 if record.side == "long" else 2  # 4=close long, 2=close short
		exit_price = 0.0
		exit_fee = 0.0
		order_found: Optional[Dict[str, Any]] = None
		try:
			# Query orders in the window since fill
			start_ms = max(record.fill_time_ms - 5000, int(record.submitted_at * 1000))
			hist = await self.client.get_history_orders(
				symbol=self.symbol,
				start_time=start_ms,
				page_size=50,
			)
			items = hist if isinstance(hist, list) else (
				hist.get("data") if isinstance(hist, dict) and isinstance(hist.get("data"), list) else []
			)
			for o in items or []:
				if not isinstance(o, dict):
					continue
				if o.get("symbol") != self.symbol:
					continue
				if int(o.get("side", 0)) != close_side:
					continue
				if int(o.get("state", -1)) != 3:  # filled
					continue
				vol = int(_as_float(o.get("vol", 0)))
				if vol != record.vol_contracts:
					continue
				# Must be AFTER our fill timestamp
				create_time = int(_as_float(o.get("createTime") or o.get("create_time") or 0))
				if create_time and create_time < record.fill_time_ms - 1000:
					continue
				order_found = o
				exit_price = _as_float(o.get("dealAvgPrice"))
				exit_fee = _as_float(o.get("takerFee"))
				break
		except Exception as exc:  # noqa: BLE001
			LOGGER.exception("history_orders fetch failed oid=%s: %s", record.external_oid, exc)

		# Decide reason by comparing exit price to our TP/SL
		reason = "unknown"
		if exit_price > 0 and self.tick > 0:
			tp_dist = abs(exit_price - record.tp_price)
			sl_dist = abs(exit_price - record.sl_price)
			if tp_dist <= sl_dist:
				reason = "tp"
			else:
				reason = "sl"

		# Compute net PnL (approx; precise figure comes from session)
		side_sign = 1 if record.side == "long" else -1
		gross_pnl = (exit_price - record.fill_price) * side_sign * record.vol_contracts * self.contract_size
		net_pnl = gross_pnl - record.fill_fee_maker - exit_fee

		record.closed = True
		record.close_reason = f"real_{reason}"
		if self._current is record:
			self._current = None

		LOGGER.info(
			"LIVE real close detected: oid=%s reason=%s entry=%.2f exit=%.2f gross=%+.4f fees=%.4f net=%+.4f",
			record.external_oid, reason, record.fill_price, exit_price,
			gross_pnl, record.fill_fee_maker + exit_fee, net_pnl,
		)

		if self.real_close_callback is not None:
			try:
				await self.real_close_callback({
					"symbol": self.symbol,
					"side": record.side,
					"reason": reason,
					"entry_price": record.fill_price,
					"exit_price": exit_price,
					"vol_contracts": record.vol_contracts,
					"notional": record.notional,
					"open_fee": record.fill_fee_maker,
					"close_fee": exit_fee,
					"gross_pnl": gross_pnl,
					"net_pnl": net_pnl,
					"tp_price": record.tp_price,
					"sl_price": record.sl_price,
					"external_oid": record.external_oid,
					"order": order_found,
				})
			except Exception as exc:  # noqa: BLE001
				LOGGER.exception("real_close_callback failed: %s", exc)

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
