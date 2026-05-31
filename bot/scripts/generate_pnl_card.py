"""Generate a MEXC P&L card image from your live account history.

Fetches closed positions from MEXC via API and renders a styled PNG card
(dark mode, MEXC-style) showing total PnL, win rate, stats grid, and
cumulative PnL curve.

Usage:
    cd bot/
    python scripts/generate_pnl_card.py
    python scripts/generate_pnl_card.py --symbol ETH_USDT --days 30
    python scripts/generate_pnl_card.py --days 7 --out data/my_card.png
    python scripts/generate_pnl_card.py --days 7 --send-telegram

Required env vars (in .env):
    MEXC_API_KEY
    MEXC_API_SECRET
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pathlib
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from exchange.mexc_client import MEXCClient  # noqa: E402
from monitoring.pnl_card import build_pnl_card  # noqa: E402

LOGGER = logging.getLogger("generate_pnl_card")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a P&L card PNG from MEXC history.")
    p.add_argument("--symbol",   default="BTC_USDT",  help="Contract symbol (default: BTC_USDT)")
    p.add_argument("--days",     type=float, default=30.0, help="Look-back window in days (default: 30)")
    p.add_argument("--max-positions", type=int, default=500, help="Max positions to fetch (default: 500)")
    p.add_argument("--out",      default=None, help="Output PNG path (default: data/pnl_card_<symbol>_<date>.png)")
    p.add_argument("--label",    default="", help="Optional strategy label shown on card")
    p.add_argument("--send-telegram", action="store_true", help="Send the card to Telegram after rendering")
    p.add_argument("--config",   default="config/config_volume_farmer_filter4h.yaml",
                   help="Config YAML to load Telegram settings from (only needed with --send-telegram)")
    return p.parse_args()


def _default_out(symbol: str) -> pathlib.Path:
    date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    return PROJECT_ROOT / "data" / f"pnl_card_{symbol}_{date_str}.png"


async def _fetch_all_positions(
    client: MEXCClient,
    symbol: str,
    start_ms: int,
    end_ms: int,
    max_positions: int,
) -> List[Dict[str, Any]]:
    """Page through history_positions until we have everything in the window."""
    all_records: List[Dict[str, Any]] = []
    page = 1
    page_size = 100

    while len(all_records) < max_positions:
        LOGGER.info("Fetching page %d (got %d so far) …", page, len(all_records))
        resp = await client.get_history_positions(
            symbol=symbol,
            start_time=start_ms,
            end_time=end_ms,
            page_num=page,
            page_size=page_size,
        )

        # MEXC returns a plain list directly for this endpoint
        if isinstance(resp, list):
            records = resp
        elif isinstance(resp, dict):
            inner = resp.get("data") or resp
            if isinstance(inner, list):
                records = inner
            else:
                records = inner.get("resultList") or inner.get("items") or []
        else:
            records = []

        if not records:
            break  # no more pages

        all_records.extend(records)

        # If we got a full page, there may be more
        if len(records) < page_size:
            break
        page += 1

    LOGGER.info("Total positions fetched: %d", len(all_records))
    return all_records[:max_positions]


async def _run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key    = os.getenv("MEXC_API_KEY", "").strip()
    api_secret = os.getenv("MEXC_API_SECRET", "").strip()

    if not api_key or not api_secret:
        LOGGER.error("MEXC_API_KEY / MEXC_API_SECRET not set in .env — cannot fetch positions.")
        sys.exit(1)

    now_ms   = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(tz=timezone.utc) - timedelta(days=args.days)).timestamp() * 1000)

    LOGGER.info(
        "Fetching %s positions for the last %.0f days (since %s) …",
        args.symbol, args.days,
        datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
    )

    async with MEXCClient(api_key=api_key, api_secret=api_secret) as client:
        positions = await _fetch_all_positions(
            client, args.symbol, start_ms, now_ms, args.max_positions,
        )

    if not positions:
        LOGGER.error("No closed positions found for %s in the last %.0f days.", args.symbol, args.days)
        sys.exit(1)

    LOGGER.info("Building P&L card from %d positions …", len(positions))
    card_bytes = build_pnl_card(positions, symbol=args.symbol, label=args.label)
    if card_bytes is None:
        LOGGER.error("Card render failed.")
        sys.exit(1)

    out_path = pathlib.Path(args.out) if args.out else _default_out(args.symbol)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(card_bytes)
    LOGGER.info("Card saved → %s", out_path)

    if args.send_telegram:
        await _send_telegram(card_bytes, args, positions, out_path)


async def _send_telegram(
    card_bytes: bytes,
    args: argparse.Namespace,
    positions: List[Dict[str, Any]],
    out_path: pathlib.Path,
) -> None:
    """Send the card to Telegram as a photo with a caption."""
    try:
        from monitoring.telegram_notifier import TelegramNotifier
    except ImportError:
        LOGGER.error("TelegramNotifier not available — cannot send.")
        return

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        LOGGER.error("Config not found: %s", config_path)
        return

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    notif_cfg = (config.get("notifications") or {}).get("telegram") or {}
    if not notif_cfg.get("enabled"):
        LOGGER.warning("Telegram not enabled in config — skipping send.")
        return

    notifier = TelegramNotifier()
    await notifier.start()
    try:
        wins   = sum(1 for p in positions if (float(p.get("realised", 0)) - float(p.get("totalFee") or abs(float(p.get("fee", 0))))) >= 0)
        losses = len(positions) - wins
        wr     = wins / len(positions) * 100 if positions else 0
        total  = sum(float(p.get("realised", 0)) - float(p.get("totalFee") or abs(float(p.get("fee", 0)))) for p in positions)
        sign   = "+" if total >= 0 else ""
        caption = (
            f"📊 *P\\&L Card · {args.symbol.replace('_', '/')}*\n"
            f"Last `{int(args.days)}` days  ·  `{len(positions)}` trades\n"
            f"Net: *{sign}{total:,.4f} USDT*   WR: `{wr:.1f}%` \\({wins}W / {losses}L\\)"
        )
        await notifier.send_photo(card_bytes, caption=caption)
        LOGGER.info("Card sent to Telegram.")
    except Exception as exc:
        LOGGER.exception("Telegram send failed: %s", exc)
    finally:
        await notifier.stop()


def main() -> None:
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
