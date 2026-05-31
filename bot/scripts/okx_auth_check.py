"""Standalone OKX auth verification — run once to confirm .env creds work.

    python3 scripts/okx_auth_check.py

Prints HTTP status and code/msg, but never echoes any credential value.
Safe to commit; safe to run repeatedly.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import pathlib
import sys
import time

try:
    import requests
except ImportError:
    sys.stderr.write("pip install requests first\n")
    sys.exit(2)


def _load_env(path: pathlib.Path) -> None:
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def main() -> int:
    env_path = pathlib.Path(__file__).resolve().parents[1] / ".env"
    _load_env(env_path)

    try:
        key = os.environ["OKX_API_KEY"]
        sec = os.environ["OKX_API_SECRET"]
        phr = os.environ["OKX_API_PASSPHRASE"]
    except KeyError as e:
        print(f"missing env var: {e}")
        return 2

    base = "https://www.okx.com"
    path = "/api/v5/account/balance"
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time()*1000)%1000:03d}Z"
    msg = f"{ts}GET{path}"
    sign = base64.b64encode(hmac.new(sec.encode(), msg.encode(), hashlib.sha256).digest()).decode()

    r = requests.get(base + path, timeout=10, headers={
        "OK-ACCESS-KEY": key,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": ts,
        "OK-ACCESS-PASSPHRASE": phr,
    })
    j = r.json()
    code, m = j.get("code"), j.get("msg", "")
    print(f"HTTP {r.status_code}  code={code}  msg={m!r}")

    diagnostics = {
        "0":     "AUTH OK — credentials work",
        "50111": "INVALID API KEY",
        "50112": "INVALID PASSPHRASE",
        "50113": "INVALID SIGNATURE (secret wrong, or clock skew)",
        "50114": "INVALID AUTHORITY (key lacks required permission)",
        "50102": "TIMESTAMP TOO OLD (clock drift)",
    }
    print(diagnostics.get(code, "unrecognized — see OKX docs"))
    return 0 if code == "0" else 1


if __name__ == "__main__":
    sys.exit(main())
