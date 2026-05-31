"""End-to-end smoke test of the OKX live executor in DRY-RUN mode.

Exercises every code path that matters without submitting any order:
  - Instrument metadata fetch (public)
  - Order construction from a synthetic FarmerEvent
  - Notional -> contracts conversion
  - Price tick rounding
  - clOrdId generation

Prints the would-be order payload so you can verify it looks correct
before flipping to --demo or --live.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys
from types import SimpleNamespace

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from exchange.okx_client import OKXClient  # noqa: E402
from execution.live_volume_executor_okx import LiveVolumeExecutorOKX  # noqa: E402


def _load_env() -> None:
    p = PROJECT_ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


async def main() -> int:
    _load_env()
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    async with OKXClient() as client:  # public-only for the smoke test
        executor = LiveVolumeExecutorOKX(
            client=client, symbol="BTC_USDT",
            dry_run=True,
            log_dir=PROJECT_ROOT / "data/logs",
        )
        print("\n--- 1. initialize() — load instrument metadata ---")
        await executor.initialize(leverage_cap=50)
        print(f"    ctVal={executor._ct_val} {executor._ct_val_ccy}")
        print(f"    tickSz={executor._tick_sz}  lotSz={executor._lot_sz}  minSz={executor._min_sz}")

        # Fetch current price so the synthetic entry uses a realistic price
        t = await client.get_ticker("BTC_USDT")
        last_px = float(t["last"])
        print(f"    last price: {last_px}")

        # Build a synthetic entry event matching what VolumeFarmerSession emits
        tp_bps, sl_bps = 18.0, 30.0
        notional = 30.0 * 100  # $30 capital × 100x leverage = $3000 notional
        evt = SimpleNamespace(
            kind="entry",
            time=None,
            payload={
                "side": "long",
                "price": last_px,
                "notional": notional,
                "leverage": 100,
                "open_fee": notional * 0.0002,
                "tp": last_px * (1 + tp_bps / 1e4),
                "sl": last_px * (1 - sl_bps / 1e4),
                "tp_bps": tp_bps,
                "sl_bps": sl_bps,
                "equity": 30.0,
                "volume": 0,
                "volume_target": 1_000_000,
                "round_trips": 0,
                "wins": 0, "losses": 0,
                "margin": notional / 100,
            },
        )

        print("\n--- 2. consume_session_event(entry) — dry-run order build ---")
        executor.consume_session_event(evt)
        # The handler schedules an async task — give it a moment to complete
        # the synchronous parts before exit
        await asyncio.sleep(0.5)

        trade = executor._open_trade
        if trade is None:
            print("    NO TRADE BUILT — order was rejected before placement")
            return 1
        print(f"    cl_ord_id: {trade.cl_ord_id}")
        print(f"    side:      {trade.side}")
        print(f"    notional:  ${trade.notional_usd:.2f}")
        print(f"    contracts: {trade.sz_contracts}")
        print(f"    entry px:  {trade.entry_px_req}  (tick-aligned)")
        print(f"    tp px:     {trade.tp_px}  (MAKER limit)")
        print(f"    sl px:     {trade.sl_px}  (MARKET on trigger)")
        print(f"    canceled:  {trade.canceled} ({trade.cancel_reason or 'n/a'})")

        # What would the OKX body look like?
        from exchange.okx_client import to_okx_inst_id
        body = {
            "instId": to_okx_inst_id("BTC_USDT"),
            "tdMode": "isolated",
            "side": "buy" if trade.side == "long" else "sell",
            "ordType": "post_only",
            "sz": str(trade.sz_contracts),
            "px": str(trade.entry_px_req),
            "clOrdId": trade.cl_ord_id,
            "attachAlgoOrds": [{
                "tpTriggerPx": str(trade.tp_px),
                "tpOrdPx": str(trade.tp_px),       # <-- MAKER TP
                "tpTriggerPxType": "last",
                "slTriggerPx": str(trade.sl_px),
                "slOrdPx": "-1",                    # <-- MARKET SL
                "slTriggerPxType": "last",
            }],
        }
        print("\n--- 3. would-be OKX POST /api/v5/trade/order body ---")
        import json
        print(json.dumps(body, indent=2))

        print("\n--- 4. SUMMARY ---")
        notional_btc = trade.sz_contracts * executor._ct_val
        print(f"    $30 capital × 100x leverage = ${notional:.2f} notional")
        print(f"    = {notional_btc:.4f} BTC = {trade.sz_contracts} contracts (lot {executor._lot_sz})")
        print(f"    TP +{tp_bps:.0f} bps as LIMIT @ {trade.tp_px}  (maker fill if triggered)")
        print(f"    SL -{sl_bps:.0f} bps as MARKET @ {trade.sl_px}  (taker fill)")
        print(f"    Expected round-trip fee at OKX 0.02%/0.06%: "
              f"open={notional*0.0002:.4f} + close_tp={notional*0.0002:.4f} = ${notional*0.0004:.4f}")
        print(f"                                            or close_sl={notional*0.0006:.4f}")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
