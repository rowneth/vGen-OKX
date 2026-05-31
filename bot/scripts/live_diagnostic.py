"""One-shot live diagnostic for MEXC futures on $30 capital.

What this script does, in order:

    1. Loads API credentials from .env
    2. Verifies API key permissions (futures-trade yes, withdraw no)
    3. Fetches and prints USDT balance from the futures wallet
    4. Fetches and prints BTC_USDT contract spec (contractSize, minVol, fee rates)
    5. Fetches the current last/bid/ask price
    6. Computes the correct contract quantity for ~$150 notional
    7. Computes TP and SL prices from tp_bps / sl_bps
    8. Prints the FULL intended order spec and WAITS for the operator to type YES
    9. Submits ONE post-only limit buy with server-side TP + SL attached
   10. Polls the order status and prints the real fill price + fee
   11. Prints the currently open position state
   12. Exits — the position stays open on the exchange with SL/TP protecting it

Safety gates:
    * Env var LIVE_DIAGNOSTIC_ACK must equal "I_UNDERSTAND_AND_ACCEPT_RISK"
    * Operator must type YES at the confirm prompt
    * Notional is capped at $200 regardless of input
    * Only BTC_USDT supported
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
import time
import uuid
from typing import Any, Dict, Optional

from dotenv import load_dotenv

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from exchange.mexc_client import MEXCClient  # noqa: E402
from exchange.security import verify_startup_permissions  # noqa: E402
from monitoring.telegram_notifier import TelegramNotifier  # noqa: E402


MAX_NOTIONAL_USD = 200.0
SYMBOL = "BTC_USDT"
POLL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 120


_TG_ESCAPE_CHARS = "_*[]()~`>#+-=|{}.!\\"


def _tg(text: Any) -> str:
	s = str(text) if text is not None else "-"
	for ch in _TG_ESCAPE_CHARS:
		s = s.replace(ch, f"\\{ch}")
	return s


async def _tg_send(notifier: Optional[TelegramNotifier], text: str) -> None:
	if notifier is None:
		return
	try:
		await notifier.send_raw(text)
	except Exception as exc:  # noqa: BLE001
		print(f"  telegram send failed: {exc}")


def _require_env(name: str) -> str:
	v = os.getenv(name, "").strip()
	if not v:
		raise RuntimeError(f"Missing env var {name}")
	return v


def _line(char: str = "─", n: int = 56) -> str:
	return char * n


def _section(title: str) -> None:
	print(f"\n{_line('━')}\n  {title}\n{_line('━')}")


def _kv(label: str, value: Any, width: int = 22) -> None:
	print(f"  {label:<{width}} {value}")


def _as_float(x: Any, default: float = 0.0) -> float:
	try:
		return float(x)
	except (TypeError, ValueError):
		return default


def _extract_usdt_balance(account_payload: Any) -> Optional[Dict[str, float]]:
	"""Find the USDT row in whatever shape MEXC returns."""

	def _walk(node: Any):
		if isinstance(node, dict):
			cur = str(node.get("currency") or node.get("coin") or "").upper()
			if cur == "USDT":
				yield node
			for v in node.values():
				yield from _walk(v)
		elif isinstance(node, list):
			for v in node:
				yield from _walk(v)

	for row in _walk(account_payload):
		return {
			"available": _as_float(row.get("availableBalance", row.get("available", 0))),
			"equity": _as_float(row.get("equity", row.get("balance", 0))),
			"unrealized": _as_float(row.get("unrealized", 0)),
		}
	return None


def _extract_contract_spec(detail_payload: Any, symbol: str) -> Optional[Dict[str, Any]]:
	"""Return the row matching the symbol from contract/detail payload."""
	# MEXC returns a single dict when called with ?symbol=..., a list otherwise.
	if isinstance(detail_payload, dict) and detail_payload.get("symbol") == symbol:
		return detail_payload
	if isinstance(detail_payload, list):
		items = detail_payload
	elif isinstance(detail_payload, dict) and isinstance(detail_payload.get("data"), list):
		items = detail_payload["data"]
	else:
		items = []
	for item in items:
		if isinstance(item, dict) and item.get("symbol") == symbol:
			return item
	return None


def _extract_ticker(ticker_payload: Any) -> Dict[str, float]:
	t = ticker_payload
	if isinstance(t, list) and t:
		t = t[0]
	if not isinstance(t, dict):
		return {}
	return {
		"last": _as_float(t.get("lastPrice", t.get("last", 0))),
		"bid": _as_float(t.get("bid1", t.get("bidPrice", 0))),
		"ask": _as_float(t.get("ask1", t.get("askPrice", 0))),
	}


def _round_price(price: float, tick: float) -> float:
	if tick <= 0:
		return price
	return round(round(price / tick) * tick, 10)


async def run(args: argparse.Namespace) -> None:
	load_dotenv(PROJECT_ROOT / ".env", override=False)

	if os.getenv("LIVE_DIAGNOSTIC_ACK", "") != "I_UNDERSTAND_AND_ACCEPT_RISK":
		raise RuntimeError(
			"LIVE_DIAGNOSTIC_ACK=I_UNDERSTAND_AND_ACCEPT_RISK must be set in env "
			"to enable real-money order submission."
		)

	api_key = _require_env("MEXC_API_KEY")
	api_secret = _require_env("MEXC_API_SECRET")

	target_notional = min(float(args.notional), MAX_NOTIONAL_USD)

	notifier: Optional[TelegramNotifier] = None
	if not args.no_telegram:
		notifier = TelegramNotifier()
		await notifier.start()
		await _tg_send(notifier, (
			f"🧪 *LIVE DIAGNOSTIC starting*\n"
			f"Symbol `{_tg(SYMBOL)}`   Notional `\\${_tg(f'{target_notional:.2f}')}`\n"
			f"Leverage `{args.leverage}x`   "
			f"TP `\\+{_tg(f'{args.tp_bps:.1f}')} bps`   "
			f"SL `\\-{_tg(f'{args.sl_bps:.1f}')} bps`"
		))

	async with MEXCClient(
		api_key=api_key,
		api_secret=api_secret,
		base_url="https://contract.mexc.com",
		timeout_seconds=15,
		requests_per_second=10.0,
		burst_capacity=20,
	) as client:

		# 1 ─────────────────────────────────────── permissions
		_section("1. API KEY PERMISSIONS")
		if args.skip_permission_check:
			print("  ⚠️  Skipping strict permission check (--skip-permission-check).")
			print("  Probing account endpoint instead to verify read+auth works.")
			try:
				_ = await client.get_account_info()
				print("  ✅ Auth + read OK (account endpoint reachable).")
				print("  NOTE: You manually confirmed withdrawal=OFF when creating the key.")
			except Exception as exc:  # noqa: BLE001
				raise RuntimeError(f"Account probe failed: {exc}") from exc
		else:
			perm = await verify_startup_permissions(
				client=client,
				require_futures_trade_permission=True,
				require_withdrawal_disabled=True,
			)
			_kv("Futures trade", perm.futures_trade_enabled)
			_kv("Withdrawal",    perm.withdrawal_enabled)
			_kv("Source",        perm.source)

		# 2 ─────────────────────────────────────── balance
		_section("2. FUTURES WALLET BALANCE")
		account = await client.get_account_info()
		usdt = _extract_usdt_balance(account)
		if not usdt:
			print("  Could not parse USDT row. Raw payload:")
			print(json.dumps(account, indent=2)[:2000])
			raise RuntimeError("USDT balance not found in account response")
		_kv("Available", f"${usdt['available']:.4f} USDT")
		_kv("Equity",    f"${usdt['equity']:.4f} USDT")
		_kv("Unrealized",f"${usdt['unrealized']:.4f} USDT")
		if usdt["available"] < 5.0:
			raise RuntimeError(f"Insufficient USDT: ${usdt['available']:.2f} < $5.00")

		# 3 ─────────────────────────────────────── contract spec
		_section(f"3. CONTRACT SPEC · {SYMBOL}")
		detail = await client.get_contract_detail(symbol=SYMBOL)
		spec = _extract_contract_spec(detail, SYMBOL)
		if not spec:
			print(json.dumps(detail, indent=2)[:2000])
			raise RuntimeError(f"{SYMBOL} not found in contract/detail")
		contract_size = _as_float(spec.get("contractSize"))
		price_scale   = int(spec.get("priceScale", 1))
		vol_scale     = int(spec.get("volScale", 0))
		price_unit    = _as_float(spec.get("priceUnit"))
		min_vol       = _as_float(spec.get("minVol", 1))
		maker_fee     = _as_float(spec.get("makerFeeRate"))
		taker_fee     = _as_float(spec.get("takerFeeRate"))
		max_leverage  = _as_float(spec.get("maxLeverage", 125))
		_kv("contractSize",  f"{contract_size} BTC per contract")
		_kv("priceUnit",     price_unit)
		_kv("priceScale",    price_scale)
		_kv("volScale",      vol_scale)
		_kv("minVol",        f"{min_vol} contracts")
		_kv("maker fee",     f"{maker_fee*100:.4f}% (paper assumes 0.0100%)")
		_kv("taker fee",     f"{taker_fee*100:.4f}% (paper assumes 0.0500%)")
		_kv("maxLeverage",   max_leverage)

		# 4 ─────────────────────────────────────── price
		_section(f"4. LIVE PRICE · {SYMBOL}")
		ticker = _extract_ticker(await client.get_ticker(SYMBOL))
		last = ticker.get("last", 0.0)
		bid = ticker.get("bid", 0.0)
		ask = ticker.get("ask", 0.0)
		if last <= 0:
			raise RuntimeError("Invalid ticker price")
		_kv("Last", f"${last:,.{price_scale}f}")
		_kv("Bid",  f"${bid:,.{price_scale}f}")
		_kv("Ask",  f"${ask:,.{price_scale}f}")
		_kv("Spread", f"{(ask - bid) / last * 10000:.2f} bps" if ask and bid else "n/a")

		# 5 ─────────────────────────────────────── sizing
		_section("5. ORDER SIZING")
		leverage = int(args.leverage)
		tp_bps = float(args.tp_bps)
		sl_bps = float(args.sl_bps)

		# post-only long at bid (safest for maker fill)
		entry_price_raw = bid if bid > 0 else last
		tick = price_unit if price_unit > 0 else 10 ** (-price_scale)
		entry_price = _round_price(entry_price_raw, tick)

		# contracts for target notional
		contracts_float = target_notional / (entry_price * contract_size) if contract_size > 0 else 0.0
		vol_contracts = int(max(min_vol, round(contracts_float)))
		actual_notional = vol_contracts * entry_price * contract_size
		required_margin = actual_notional / leverage

		tp_price = _round_price(entry_price * (1 + tp_bps / 10_000), tick)
		sl_price = _round_price(entry_price * (1 - sl_bps / 10_000), tick)

		_kv("Side",             "OPEN LONG (post-only limit)")
		_kv("Entry price",      f"${entry_price:,.{price_scale}f} (at bid)")
		_kv("Contracts",        f"{vol_contracts} (min {min_vol})")
		_kv("Actual notional",  f"${actual_notional:,.2f}")
		_kv("Leverage",         f"{leverage}x")
		_kv("Required margin",  f"${required_margin:,.4f}")
		_kv("TP price",         f"${tp_price:,.{price_scale}f} (+{tp_bps:.1f} bps)")
		_kv("SL price",         f"${sl_price:,.{price_scale}f} (-{sl_bps:.1f} bps)")
		_kv("Est. open fee",    f"${actual_notional * maker_fee:.4f} (if maker)")

		if required_margin > usdt["available"]:
			raise RuntimeError(
				f"Required margin ${required_margin:.4f} exceeds "
				f"available ${usdt['available']:.4f}"
			)

		# 6 ─────────────────────────────────────── confirm
		_section("6. CONFIRMATION REQUIRED")
		print("  This will place a REAL order on MEXC with REAL funds.")
		print("  Type YES (uppercase) to proceed, anything else to abort.")
		sys.stdout.write("  > ")
		sys.stdout.flush()
		reply = sys.stdin.readline().strip()
		if reply != "YES":
			print("\n  Aborted. No order submitted.")
			await _tg_send(notifier, "🚫 *Diagnostic aborted* \\(no YES confirmation\\)")
			return

		# 7 ─────────────────────────────────────── submit
		_section("7. SUBMITTING ORDER")
		external_oid = f"diag-{int(time.time())}-{uuid.uuid4().hex[:6]}"
		_kv("externalOid", external_oid)
		submit_body = {
			"symbol": SYMBOL, "side": 1, "type": 2, "vol": vol_contracts,
			"price": entry_price, "leverage": leverage, "openType": 1,
			"externalOid": external_oid,
			"stopLossPrice": sl_price, "takeProfitPrice": tp_price,
		}
		print("  Request body being sent:")
		print(f"    {json.dumps(submit_body)}")
		await _tg_send(notifier, (
			f"📤 *Submitting order*\n"
			f"Side `LONG`   Vol `{vol_contracts}` contracts\n"
			f"Entry `\\${_tg(f'{entry_price:,.{price_scale}f}')}` \\(post\\-only\\)\n"
			f"TP `\\${_tg(f'{tp_price:,.{price_scale}f}')}`   "
			f"SL `\\${_tg(f'{sl_price:,.{price_scale}f}')}`\n"
			f"Notional `\\${_tg(f'{actual_notional:,.2f}')}`   "
			f"Margin `\\${_tg(f'{required_margin:,.4f}')}`\n"
			f"`oid={_tg(external_oid)}`"
		))
		try:
			submit_result = await client.submit_order(
				symbol=SYMBOL,
				side=1,                  # 1 = open long
				order_type=2,            # 2 = post-only limit (MEXC convention)
				vol=vol_contracts,
				price=entry_price,
				leverage=leverage,
				open_type=1,             # 1 = isolated
				external_oid=external_oid,
				stop_loss_price=sl_price,
				take_profit_price=tp_price,
			)
		except Exception as exc:  # noqa: BLE001
			print()
			print(f"  ❌ submit_order failed: {exc}")
			print()
			print("  ─── FOR MEXC SUPPORT ───")
			print(f"  Endpoint      : POST https://contract.mexc.com/api/v1/private/order/create")
			print(f"  externalOid   : {external_oid}")
			print(f"  Request body  : {json.dumps(submit_body)}")
			print(f"  Error         : {exc}")
			print("  ─────────────────────────")
			await _tg_send(notifier, (
				f"❌ *Submit failed*\n`{_tg(str(exc))[:300]}`"
			))
			raise
		order_id = submit_result if isinstance(submit_result, (int, str)) else submit_result
		_kv("Exchange response", order_id)
		await _tg_send(notifier, f"✅ *Submitted* exchange orderId `{_tg(order_id)}`")

		# 8 ─────────────────────────────────────── poll fill
		_section("8. POLLING FILL STATUS")
		deadline = time.time() + POLL_TIMEOUT_SECONDS
		final_order: Optional[Dict[str, Any]] = None
		while time.time() < deadline:
			try:
				order = await client.get_order_by_external_oid(SYMBOL, external_oid)
				state = int(order.get("state", -1)) if isinstance(order, dict) else -1
				# MEXC futures order states: 1=uninitialized 2=open 3=filled 4=cancelled 5=partial
				state_label = {
					1: "uninitialized", 2: "open (resting)", 3: "filled",
					4: "cancelled", 5: "partially filled",
				}.get(state, f"state={state}")
				deal_avg = _as_float(order.get("dealAvgPrice", 0)) if isinstance(order, dict) else 0
				deal_vol = _as_float(order.get("dealVol", 0)) if isinstance(order, dict) else 0
				fee_amt  = _as_float(order.get("takerFee", 0)) + _as_float(order.get("makerFee", 0)) if isinstance(order, dict) else 0
				print(f"  [{int(time.time())%1000:03d}] {state_label:>18}  "
					  f"dealVol={deal_vol}  avgPx={deal_avg:.{price_scale}f}  fee={fee_amt:.6f}")
				if state in (3, 4):
					final_order = order if isinstance(order, dict) else None
					break
			except Exception as exc:  # noqa: BLE001
				print(f"  poll error: {exc}")
			await asyncio.sleep(POLL_SECONDS)

		# 9 ─────────────────────────────────────── fill report
		_section("9. FILL REPORT")
		if not final_order:
			print("  Order did not reach terminal state within timeout.")
			print("  It may still be resting on the book — check MEXC UI.")
			await _tg_send(notifier, (
				f"⏳ *Fill timeout*\n"
				f"Order still resting after {POLL_TIMEOUT_SECONDS}s\\. "
				f"Check MEXC UI\\.\n`oid={_tg(external_oid)}`"
			))
		else:
			state = int(final_order.get("state", -1))
			deal_avg = _as_float(final_order.get("dealAvgPrice"))
			deal_vol = _as_float(final_order.get("dealVol"))
			maker_fee_paid = _as_float(final_order.get("makerFee"))
			taker_fee_paid = _as_float(final_order.get("takerFee"))
			was_maker = maker_fee_paid > 0 and taker_fee_paid == 0
			_kv("Final state",    state)
			_kv("Filled contracts", deal_vol)
			_kv("Avg fill price", f"${deal_avg:,.{price_scale}f}")
			_kv("Maker fee paid", f"${maker_fee_paid:.6f}")
			_kv("Taker fee paid", f"${taker_fee_paid:.6f}")
			_kv("Fee type",       "MAKER ✅" if was_maker else "TAKER ⚠️  (post-only rejected)")
			eff_rate = 0.0
			if deal_vol > 0 and deal_avg > 0:
				eff_rate = (maker_fee_paid + taker_fee_paid) / (deal_vol * deal_avg * contract_size)
				_kv("Effective rate", f"{eff_rate*100:.4f}%")
			fee_badge = "MAKER ✅ 0\\.01%" if was_maker else "TAKER ⚠️ 0\\.05%"
			await _tg_send(notifier, (
				f"🎯 *ENTRY FILLED*\n"
				f"Filled `{_tg(deal_vol)}` contracts at `\\${_tg(f'{deal_avg:,.{price_scale}f}')}`\n"
				f"Fee `\\${_tg(f'{maker_fee_paid + taker_fee_paid:.6f}')}`   {fee_badge}\n"
				f"Effective rate `{_tg(f'{eff_rate*100:.4f}')}%`\n"
				f"Now watching for TP/SL\\.\\.\\."
			))

		# 10 ────────────────────────────────────── open position
		_section("10. OPEN POSITIONS")
		positions = await client.get_open_positions(symbol=SYMBOL)
		pos_rows = []
		if not positions:
			print("  No open positions reported.")
		else:
			pos_list = positions if isinstance(positions, list) else [positions]
			for pos in pos_list:
				if not isinstance(pos, dict):
					continue
				pos_rows.append(pos)
				_kv("positionId",  pos.get("positionId"))
				_kv("holdVol",     pos.get("holdVol"))
				_kv("holdAvgPrice",pos.get("holdAvgPrice"))
				_kv("leverage",    pos.get("leverage"))
				_kv("openType",    f"{pos.get('openType')} (1=isolated)")
				_kv("state",       pos.get("state"))
				print()
		if pos_rows:
			p0 = pos_rows[0]
			await _tg_send(notifier, (
				f"🟢 *POSITION OPEN on MEXC*\n"
				f"positionId `{_tg(p0.get('positionId'))}`\n"
				f"holdVol `{_tg(p0.get('holdVol'))}`   "
				f"avgPrice `\\${_tg(p0.get('holdAvgPrice'))}`\n"
				f"leverage `{_tg(p0.get('leverage'))}x`   "
				f"openType `{_tg(p0.get('openType'))}` \\(1\\=isolated\\)\n"
				f"Monitoring until TP/SL resolves\\.\\.\\."
			))

		# 11 ────────────────────────────────────── watch until closed
		if pos_rows and args.watch_minutes > 0:
			_section(f"11. WATCHING POSITION (max {args.watch_minutes}m)")
			await _watch_until_closed(
				client=client,
				notifier=notifier,
				symbol=SYMBOL,
				position_id=pos_rows[0].get("positionId"),
				entry_price=_as_float(pos_rows[0].get("holdAvgPrice")),
				contract_size=contract_size,
				price_scale=price_scale,
				max_minutes=args.watch_minutes,
			)

		_section("DONE")
		print(f"  Saved externalOid: {external_oid}")
		await _tg_send(notifier, f"🏁 *Diagnostic script finished*\n`oid={_tg(external_oid)}`")
		if notifier is not None:
			await notifier.stop()


async def _watch_until_closed(
	client: MEXCClient,
	notifier: Optional[TelegramNotifier],
	symbol: str,
	position_id: Any,
	entry_price: float,
	contract_size: float,
	price_scale: int,
	max_minutes: int,
	poll_seconds: int = 10,
) -> None:
	"""Poll open_positions until the target position disappears, then report."""
	deadline = time.time() + max_minutes * 60
	last_print = 0.0
	while time.time() < deadline:
		try:
			positions = await client.get_open_positions(symbol=symbol)
			pos_list = positions if isinstance(positions, list) else ([positions] if positions else [])
			still_open = any(
				isinstance(p, dict) and p.get("positionId") == position_id
				for p in pos_list
			)
			if not still_open:
				# position closed — fetch account to show current equity
				acct = await client.get_account_info()
				usdt = _extract_usdt_balance(acct)
				equity = usdt["equity"] if usdt else 0.0
				available = usdt["available"] if usdt else 0.0
				_section("POSITION CLOSED")
				_kv("Wallet equity",    f"${equity:.4f}")
				_kv("Wallet available", f"${available:.4f}")
				equity_str = _tg(f"{equity:.4f}")
				avail_str = _tg(f"{available:.4f}")
				await _tg_send(notifier, (
					f"🔔 *POSITION CLOSED*\n"
					f"positionId `{_tg(position_id)}`\n"
					f"Equity `\\${equity_str}`   "
					f"Available `\\${avail_str}`\n"
					f"_Check MEXC order history for exit price and fee breakdown\\._"
				))
				return
			# still open — heartbeat log every ~60s
			if time.time() - last_print > 60:
				print(f"  [{int((time.time()-deadline+max_minutes*60)):4d}s] position still open, polling...")
				last_print = time.time()
		except Exception as exc:  # noqa: BLE001
			print(f"  watch error: {exc}")
		await asyncio.sleep(poll_seconds)
	_section("WATCH TIMEOUT")
	print("  Position still open at end of watch window — monitor on MEXC UI.")
	await _tg_send(notifier, (
		f"⏱ *Watch timeout* after {max_minutes}m\\.\n"
		f"Position may still be open — check MEXC UI\\."
	))


def _parse() -> argparse.Namespace:
	p = argparse.ArgumentParser()
	p.add_argument("--notional", type=float, default=150.0, help="Target notional USD (capped at 200)")
	p.add_argument("--leverage", type=int, default=100)
	p.add_argument("--tp-bps",   type=float, default=5.0)
	p.add_argument("--sl-bps",   type=float, default=50.0)
	p.add_argument("--watch-minutes", type=int, default=30,
				   help="After fill, watch the position until closed or N minutes elapse (0 = skip)")
	p.add_argument("--no-telegram", action="store_true", help="Disable Telegram notifications")
	p.add_argument("--skip-permission-check", action="store_true",
				   help="Skip strict permission parse (MEXC futures API doesn't expose flags). "
						"Use only after manually verifying withdraw=OFF on the API key.")
	return p.parse_args()


if __name__ == "__main__":
	asyncio.run(run(_parse()))
