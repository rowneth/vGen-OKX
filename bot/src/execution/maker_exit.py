"""Maker-only close routine with re-peg loop and taker fallback.

Used to exit OKX perpetual positions while staying on the passive (maker) side
of the book wherever possible. The strategy: place a reduceOnly post_only order
at the touch (best ask for closing long, best bid for closing short). If it
rejects (would-cross) or doesn't fill in time, cancel and re-peg to follow the
book. Switch to a market order only when re-peg count, elapsed time, or adverse
excursion crosses a limit — guaranteeing the position can always be closed even
during a fast move.

Designed to be called from ``LiveVolumeExecutorOKX`` when:
  * a TIME-STOP fires and we want to close at touch (maker) instead of taker
  * an explicit manual close is requested
  * any future "soft" exit (e.g. break-even drag) that should prefer maker
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from exchange.okx_client import OKXClient, to_okx_inst_id

LOGGER = logging.getLogger(__name__)


@dataclass
class MakerExitConfig:
    repeg_ms: int = 750
    max_repegs: int = 8
    max_exit_seconds: float = 20.0
    max_adverse_bps: float = 15.0
    poll_ms: int = 120  # how often to check order state inside one re-peg window


@dataclass
class ExitFill:
    px: float
    qty: float
    fee: float
    fee_type: str  # "maker" | "taker"


@dataclass
class ExitResult:
    """Outcome of a maker-exit attempt."""

    filled_qty: float = 0.0
    avg_fill_px: float = 0.0
    maker_qty: float = 0.0
    taker_qty: float = 0.0
    repegs: int = 0
    ttf_seconds: float = 0.0
    realized_fee_bps: float = 0.0
    adverse_bps: float = 0.0
    fallback_reason: str = ""           # "" if pure maker
    fills: List[ExitFill] = field(default_factory=list)
    error: str = ""

    @property
    def used_taker(self) -> bool:
        return self.taker_qty > 0

    @property
    def fill_class(self) -> str:
        if self.error:
            return "error"
        if self.maker_qty > 0 and self.taker_qty == 0:
            return "maker"
        if self.maker_qty == 0 and self.taker_qty > 0:
            return "taker"
        return "mixed"


def _round_to_tick(x: float, tick: float) -> float:
    if tick <= 0:
        return x
    return round(round(x / tick) * tick, 10)


def _decimals_from_step(step: float) -> int:
    """Decimal places implied by a price/size step (e.g. 0.01 -> 2)."""
    if step <= 0:
        return 8
    import math as _math
    return max(0, -int(round(_math.log10(step))))


def _fmt_size(x: float, lot: float) -> str:
    """OKX validates ``sz`` textually against lotSz; format to lot decimals."""
    return f"{x:.{_decimals_from_step(lot)}f}"


async def close_position_maker(
    client: OKXClient,
    *,
    symbol: str,
    position_side: str,                # "long" | "short" — side held
    sz_contracts: float,
    tick_sz: float,
    intended_exit_px: float,
    lot_sz: float = 0.0,               # format sz to this grid; 0 = no formatting
    ct_val: float = 1.0,               # USD notional per contract (BTC-USDT-SWAP: 0.01 BTC × price)
    pos_side_param: Optional[str] = None,  # "long"/"short" in hedge mode, else None
    td_mode: str = "isolated",
    cfg: MakerExitConfig = MakerExitConfig(),
    cl_ord_prefix: str = "mex",
) -> ExitResult:
    """Close ``sz_contracts`` of an open position using maker-first re-peg loop.

    ``position_side`` is the side currently HELD. The closing order is the
    opposite side: long position closes by SELL; short position closes by BUY.

    Re-peg loop pins the post_only price to the current touch (best ask when
    closing long, best bid when closing short). Re-pegging stays passive — we
    never escalate to cross the spread. Taker fallback fires only when one of
    the configured limits is breached.
    """
    side_to_close = position_side.lower()
    if side_to_close not in ("long", "short"):
        raise ValueError(f"position_side must be long|short, got {position_side!r}")
    close_side = "sell" if side_to_close == "long" else "buy"

    result = ExitResult()
    if sz_contracts <= 0:
        result.error = "non_positive_size"
        return result

    remaining = float(sz_contracts)
    started_at = time.monotonic()
    deadline = started_at + cfg.max_exit_seconds

    def _adverse_bps_from(touch_px: float) -> float:
        # adverse = exit worse than intended, in our direction-of-favor frame
        if intended_exit_px <= 0:
            return 0.0
        if side_to_close == "long":
            # closing long, lower price is worse (less profit). Adverse if touch < intended.
            return max(0.0, (intended_exit_px - touch_px) / intended_exit_px * 10_000.0)
        # closing short, higher price is worse. Adverse if touch > intended.
        return max(0.0, (touch_px - intended_exit_px) / intended_exit_px * 10_000.0)

    cumulative_notional = 0.0
    while remaining > 0:
        # ---- evaluate fallback triggers BEFORE placing the next maker peg
        if result.repegs >= cfg.max_repegs:
            result.fallback_reason = f"max_repegs_{cfg.max_repegs}"
            break
        if time.monotonic() >= deadline:
            result.fallback_reason = "max_exit_seconds"
            break

        # ---- fetch touch
        try:
            tk = await client.get_ticker(symbol)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("maker_exit: ticker fetch failed: %s", exc)
            await asyncio.sleep(cfg.poll_ms / 1000.0)
            continue
        try:
            best_ask = float(tk.get("askPx") or 0.0)
            best_bid = float(tk.get("bidPx") or 0.0)
        except (TypeError, ValueError):
            best_ask = best_bid = 0.0
        if best_ask <= 0 or best_bid <= 0:
            await asyncio.sleep(cfg.poll_ms / 1000.0)
            continue
        touch = best_ask if side_to_close == "long" else best_bid
        touch = _round_to_tick(touch, tick_sz)

        adverse = _adverse_bps_from(touch)
        result.adverse_bps = max(result.adverse_bps, adverse)
        if adverse > cfg.max_adverse_bps:
            result.fallback_reason = f"adverse_{adverse:.1f}bps"
            break

        cl_ord = cl_ord_prefix + uuid.uuid4().hex[:14]
        sz_str = _fmt_size(remaining, lot_sz) if lot_sz > 0 else str(remaining)
        try:
            resp = await client.place_order(
                symbol=symbol,
                side=close_side,
                pos_side=pos_side_param,
                td_mode=td_mode,
                ord_type="post_only",
                sz=sz_str,
                px=str(touch),
                client_oid=cl_ord,
                reduce_only=True,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("maker_exit: place_order error: %s", exc)
            await asyncio.sleep(cfg.poll_ms / 1000.0)
            result.repegs += 1
            continue

        data = (resp.get("data") or [{}])[0]
        scode = str(data.get("sCode", "0"))
        ord_id = data.get("ordId")
        if scode != "0":
            # 51280: post_only would cross. Treat as "re-peg, don't escalate."
            LOGGER.info("maker_exit: post_only rejected sCode=%s sMsg=%r — re-pegging",
                        scode, data.get("sMsg"))
            result.repegs += 1
            await asyncio.sleep(cfg.poll_ms / 1000.0)
            continue

        # ---- wait up to repeg_ms for a (partial) fill
        order_deadline = time.monotonic() + cfg.repeg_ms / 1000.0
        filled_this_round = 0.0
        avg_px_this_round = 0.0
        fee_this_round = 0.0
        terminal = False
        while time.monotonic() < order_deadline and time.monotonic() < deadline:
            await asyncio.sleep(cfg.poll_ms / 1000.0)
            try:
                r = await client.get_order(
                    symbol, ord_id=ord_id, client_oid=cl_ord,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("maker_exit: get_order failed: %s", exc)
                continue
            od = (r.get("data") or [{}])[0]
            state = str(od.get("state", ""))
            try:
                filled_sz = float(od.get("accFillSz") or 0.0)
                fill_avg = float(od.get("avgPx") or 0.0)
                fee_total = abs(float(od.get("fee") or 0.0))
            except (TypeError, ValueError):
                filled_sz = 0.0
                fill_avg = 0.0
                fee_total = 0.0
            filled_this_round = filled_sz
            avg_px_this_round = fill_avg
            fee_this_round = fee_total
            if state == "filled":
                terminal = True
                break
            if state in {"canceled", "mmp_canceled"}:
                terminal = True
                break

        if not terminal:
            # Window elapsed without terminal state — cancel and re-peg.
            try:
                await client.cancel_order(symbol, ord_id=ord_id)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("maker_exit: cancel after repeg failed: %s", exc)
            # Poll once more to capture any last-moment fill before cancel landed.
            try:
                r = await client.get_order(
                    symbol, ord_id=ord_id, client_oid=cl_ord,
                )
                od = (r.get("data") or [{}])[0]
                filled_sz = float(od.get("accFillSz") or 0.0)
                fill_avg = float(od.get("avgPx") or 0.0)
                fee_total = abs(float(od.get("fee") or 0.0))
                filled_this_round = max(filled_this_round, filled_sz)
                if fill_avg > 0:
                    avg_px_this_round = fill_avg
                fee_this_round = max(fee_this_round, fee_total)
            except Exception:  # noqa: BLE001
                pass

        if filled_this_round > 0:
            # Maker fill (post_only can only fill as maker)
            result.maker_qty += filled_this_round
            result.filled_qty += filled_this_round
            cumulative_notional += filled_this_round * avg_px_this_round
            remaining = max(0.0, remaining - filled_this_round)
            result.fills.append(ExitFill(
                px=avg_px_this_round, qty=filled_this_round,
                fee=fee_this_round, fee_type="maker",
            ))
        result.repegs += 1

    # ---- taker fallback for residual
    if remaining > 0 and result.fallback_reason:
        LOGGER.warning(
            "maker_exit: falling back to taker for residual=%s reason=%s",
            remaining, result.fallback_reason,
        )
        cl_ord = cl_ord_prefix + "_tk_" + uuid.uuid4().hex[:10]
        sz_str = _fmt_size(remaining, lot_sz) if lot_sz > 0 else str(remaining)
        try:
            resp = await client.place_order(
                symbol=symbol,
                side=close_side,
                pos_side=pos_side_param,
                td_mode=td_mode,
                ord_type="market",
                sz=sz_str,
                client_oid=cl_ord,
                reduce_only=True,
            )
            data = (resp.get("data") or [{}])[0]
            ord_id = data.get("ordId")
            # Brief poll for fill details (market should fill near-instantly)
            fill_avg = 0.0
            fee_total = 0.0
            filled_sz = 0.0
            for _ in range(20):
                await asyncio.sleep(0.1)
                try:
                    r = await client.get_order(
                        symbol, ord_id=ord_id, client_oid=cl_ord,
                    )
                except Exception:  # noqa: BLE001
                    continue
                od = (r.get("data") or [{}])[0]
                if str(od.get("state", "")) == "filled":
                    filled_sz = float(od.get("accFillSz") or 0.0)
                    fill_avg = float(od.get("avgPx") or 0.0)
                    fee_total = abs(float(od.get("fee") or 0.0))
                    break
            if filled_sz > 0:
                result.taker_qty += filled_sz
                result.filled_qty += filled_sz
                cumulative_notional += filled_sz * fill_avg
                remaining = max(0.0, remaining - filled_sz)
                result.fills.append(ExitFill(
                    px=fill_avg, qty=filled_sz, fee=fee_total, fee_type="taker",
                ))
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("maker_exit: taker fallback failed: %s", exc)
            result.error = f"taker_fallback_error: {exc}"

    result.ttf_seconds = time.monotonic() - started_at
    if result.filled_qty > 0:
        result.avg_fill_px = cumulative_notional / result.filled_qty
        total_fee = sum(f.fee for f in result.fills)
        # cumulative_notional is in (contracts × price) units; for instruments
        # with ctVal != 1 (e.g. BTC-USDT-SWAP where ctVal=0.01 BTC), multiply
        # by ct_val to get USD notional. Without this the bps calc reads
        # ~100x too small.
        notional_filled_usd = cumulative_notional * ct_val
        if notional_filled_usd > 0:
            result.realized_fee_bps = total_fee / notional_filled_usd * 10_000.0
    if remaining > 0 and not result.error:
        result.error = "residual_unfilled"
    return result


async def cancel_attached_algos(
    client: OKXClient, symbol: str,
) -> List[str]:
    """Cancel all currently-pending conditional algo orders for ``symbol``.

    Returns the list of algo_ids that were submitted for cancellation. Failures
    are swallowed and logged; the caller should treat best-effort.
    """
    try:
        r = await client.get_pending_algos(symbol)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("cancel_attached_algos: pending fetch failed: %s", exc)
        return []
    pending = r.get("data") or []
    algo_ids = [str(p.get("algoId")) for p in pending if p.get("algoId")]
    if not algo_ids:
        return []
    try:
        await client.cancel_algos(symbol, algo_ids)
        LOGGER.info("cancelled %d attached algo order(s) for %s",
                    len(algo_ids), to_okx_inst_id(symbol))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("cancel_algos failed: %s", exc)
    return algo_ids
