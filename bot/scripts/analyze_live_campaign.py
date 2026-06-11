"""Pull the FULL live-campaign record from OKX (exchange truth, read-only)
and dump a machine-readable summary for analysis.

Reads creds from env (OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE).
Fetches, for each campaign instrument:
  * complete order history (7d endpoint + 3-month archive, fully paginated)
  * account bills (trade fees, transfers, deposits — the wallet trail)
  * current balance
Prints JSON to stdout. No orders are placed; every call is a GET.
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys
import time
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from exchange.okx_client import OKXClient, to_okx_inst_id  # noqa: E402

SYMBOLS = ["BTC_USDT", "DOGE_USDT"]


async def fetch_orders(client: OKXClient, symbol: str) -> list:
    inst = to_okx_inst_id(symbol)
    seen: dict = {}
    for path in ("/api/v5/trade/orders-history",
                 "/api/v5/trade/orders-history-archive"):
        after = ""
        for _ in range(60):                      # up to 6,000 orders/endpoint
            params = {"instType": "SWAP", "instId": inst, "limit": "100"}
            if after:
                params["after"] = after
            r = await client._request("GET", path, params=params, auth=True)  # noqa: SLF001
            if str(r.get("code")) != "0":
                break
            data = r.get("data") or []
            if not data:
                break
            for o in data:
                seen[o.get("ordId")] = o
            after = data[-1].get("ordId") or ""
            if not after or len(data) < 100:
                break
            await asyncio.sleep(0.12)
    return list(seen.values())


async def fetch_bills(client: OKXClient) -> list:
    """Account bills, 7d window (campaign is younger than that). Shows every
    balance change: trade fee+pnl, transfers in/out, deposits."""
    out: list = []
    after = ""
    for _ in range(40):
        params = {"limit": "100"}
        if after:
            params["after"] = after
        r = await client._request("GET", "/api/v5/account/bills", params=params, auth=True)  # noqa: SLF001
        if str(r.get("code")) != "0":
            break
        data = r.get("data") or []
        if not data:
            break
        out.extend(data)
        after = data[-1].get("billId") or ""
        if not after or len(data) < 100:
            break
        await asyncio.sleep(0.12)
    return out


async def main() -> int:
    k = os.environ.get("OKX_API_KEY", "")
    s = os.environ.get("OKX_API_SECRET", "")
    p = os.environ.get("OKX_API_PASSPHRASE", "")
    if not (k and s and p):
        print(json.dumps({"error": "missing creds"})); return 2
    out: dict = {"fetched_at_ms": int(time.time() * 1000)}
    async with OKXClient(api_key=k, api_secret=s, passphrase=p, simulated=False) as c:
        bal = await c.get_balance("USDT")
        bd = (bal.get("data") or [{}])[0]
        usdt = next((d for d in (bd.get("details") or []) if d.get("ccy") == "USDT"), {})
        out["wallet"] = {"eq": usdt.get("eq"), "avail": usdt.get("availBal"),
                         "totalEq": bd.get("totalEq")}
        out["orders"] = {}
        for sym in SYMBOLS:
            orders = await fetch_orders(c, sym)
            # keep only the fields we need, filled or partially-filled
            keep = []
            for o in orders:
                try:
                    acc = float(o.get("accFillSz") or 0)
                except (TypeError, ValueError):
                    acc = 0.0
                if o.get("state") == "filled" or acc > 0:
                    keep.append({k2: o.get(k2) for k2 in (
                        "ordId", "state", "side", "posSide", "ordType",
                        "accFillSz", "avgPx", "fee", "pnl", "fillTime",
                        "cTime", "ctVal", "lever", "reduceOnly", "instId")})
            out["orders"][sym] = keep
            await asyncio.sleep(0.2)
        bills = await fetch_bills(c)
        out["bills"] = [{k2: b.get(k2) for k2 in (
            "billId", "ts", "type", "subType", "instId", "balChg", "bal",
            "pnl", "fee", "ccy")} for b in bills]
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
