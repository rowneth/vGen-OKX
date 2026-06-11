"""LIVE deployment diagnostic — run on the droplet to find why authenticated
OKX calls fail ("Wallet unavailable" / "fee tier unavailable").

  .venv/bin/python scripts/diag_live.py

Loads .env, hits the live endpoints with the LIVE keys, and prints the exact
OKX code + a plain-English cause for each. Prints NO secret values — only key
NAMES, lengths, and API results.
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from exchange.okx_client import OKXClient  # noqa: E402

CODE_CAUSE = {
    "0": "OK",
    "50110": "IP NOT WHITELISTED — this droplet's IP is not on the API key's allowed list. "
             "Fix on OKX: API management → edit key → add this server's IP (or set 'no IP restriction').",
    "50111": "Invalid signature — API SECRET is wrong/mismatched for this key.",
    "50113": "Invalid signature/timestamp.",
    "50102": "Timestamp expired — server clock drift > 30s. Fix NTP on the droplet.",
    "50119": "API key does not exist — wrong key, or a DEMO key used against LIVE.",
    "50100": "Demo/simulated key used on live endpoint (or vice-versa).",
    "50101": "API key / passphrase mismatch — wrong passphrase.",
    "51008": "Insufficient balance (account reachable, but no funds in trading account).",
}


def _load_env(path: pathlib.Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


async def main() -> int:
    _load_env(pathlib.Path(__file__).resolve().parents[1] / ".env")
    k = os.environ.get("OKX_API_KEY", "")
    s = os.environ.get("OKX_API_SECRET", "")
    p = os.environ.get("OKX_API_PASSPHRASE", "")
    print("=== KEYS (live) ===")
    print(f"  OKX_API_KEY        len={len(k)}  {'set' if k else 'MISSING'}")
    print(f"  OKX_API_SECRET     len={len(s)}  {'set' if s else 'MISSING'}")
    print(f"  OKX_API_PASSPHRASE len={len(p)}  {'set' if p else 'MISSING'}")
    print(f"  TELEGRAM_BOT_TOKEN ...{os.environ.get('TELEGRAM_BOT_TOKEN','')[-6:]}")
    print(f"  TELEGRAM_CHAT_ID   {os.environ.get('TELEGRAM_CHAT_ID','(unset)')}")
    if not (k and s and p):
        print("\n>>> live keys incomplete — fix .env first."); return 2

    async with OKXClient(api_key=k, api_secret=s, passphrase=p, simulated=False) as c:
        # server time / clock
        try:
            import time as _t
            srv = await c.get_server_time_ms()
            drift = abs(srv/1000 - _t.time())
            print(f"\n=== CLOCK ===\n  drift vs OKX: {drift:.2f}s  {'OK' if drift < 10 else '>>> FIX NTP'}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n=== CLOCK ===\n  server-time check failed: {exc}")

        async def probe(name, coro):
            try:
                r = await coro
                code = str(r.get("code", "?")) if isinstance(r, dict) else "0"
                cause = CODE_CAUSE.get(code, f"see OKX docs for code {code}")
                msg = r.get("msg", "") if isinstance(r, dict) else ""
                print(f"  {name:22} code={code:6} {cause}{(' | '+msg) if msg else ''}")
                return r
            except Exception as exc:  # noqa: BLE001
                # _request raises on HTTP>=400/auth codes — surface the embedded code
                txt = str(exc)
                hit = next((cc for cc in CODE_CAUSE if cc in txt and cc != "0"), None)
                print(f"  {name:22} EXC    {CODE_CAUSE.get(hit, txt[:120])}")
                return None

        print("\n=== AUTHENTICATED CALLS (live) ===")
        bal = await probe("balance", c.get_balance("USDT"))
        await probe("account-config", c.get_account_config())
        await probe("trade-fee", c.get_trade_fee("BTC_USDT"))
        await probe("positions", c.get_positions("BTC_USDT"))

        # Where are the funds?
        if isinstance(bal, dict) and str(bal.get("code")) == "0":
            print("\n=== TRADING-ACCOUNT BALANCE ===")
            data = (bal.get("data") or [{}])[0]
            details = data.get("details") or []
            usdt = next((d for d in details if d.get("ccy") == "USDT"), None)
            if usdt:
                print(f"  USDT eq={usdt.get('eq')}  avail={usdt.get('availBal')}  totalEq={data.get('totalEq')}")
                print("  >>> if eq looks right (~your deposit), the bot should size fine.")
            else:
                print(f"  NO USDT in the trading account. totalEq={data.get('totalEq')}")
                print("  >>> Your funds are likely in the FUNDING account. On OKX: Assets → "
                      "transfer USDT from Funding → Trading (Unified). The bot trades the "
                      "Trading/Unified balance only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
