"""Paper-trading runner for the volume-farmer bot using OKX candles.

Polls OKX V5 public candles for ``BTC-USDT-SWAP`` every N seconds, builds a
rolling history dataframe, and feeds it to ``VolumeFarmerSession``. Fills are
simulated locally (intrabar high/low check, same as the backtest).

No authentication needed — uses only the public candles endpoint. Add
``--demo`` to enable OKX simulated-trading mode for the future LIVE path,
which requires the three OKX_* credentials in .env.

Trade events are appended to ``data/logs/trades/YYYY-MM-DD.jsonl`` with the
same schema as the MEXC runner.

Usage:
    python3 scripts/run_volume_farmer_okx.py \
        --config config/config_volume_farmer_okx_5m_tp18.yaml \
        --label okx-paper \
        --duration-days 7
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pathlib
import signal
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from exchange.okx_client import OKXClient, to_okx_bar  # noqa: E402
from execution.live_volume_executor_okx import LiveVolumeExecutorOKX  # noqa: E402
from execution.volume_farmer import FarmerEvent, VolumeFarmerSession  # noqa: E402
from monitoring.telegram_notifier import TelegramNotifier  # noqa: E402

LOGGER = logging.getLogger("vf_okx")


SEED_CANDLES = 100
DEFAULT_POLL_SECONDS = 20


def _load_env(path: pathlib.Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def _load_config(path: pathlib.Path) -> Dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh)


def _parse_timeframe_seconds(tf: str) -> int:
    n = int("".join(ch for ch in tf if ch.isdigit()))
    unit = "".join(ch for ch in tf if ch.isalpha()).lower()
    return n * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


def _normalize_okx_candles(raw: list) -> pd.DataFrame:
    """OKX returns DESCENDING [ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm].

    We return an ASCENDING dataframe with columns matching what the rest of
    the bot expects: open_time (UTC datetime), open, high, low, close, volume,
    plus a 'closed' bool. Unclosed (current) bars are filtered out.
    """
    if not raw:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "closed"])
    rows = []
    for r in raw:
        ts = int(r[0])
        closed = (r[8] == "1")
        rows.append({
            "open_time": pd.Timestamp(ts, unit="ms", tz="UTC"),
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "volume": float(r[5]),
            "closed": closed,
        })
    df = pd.DataFrame(rows).sort_values("open_time").reset_index(drop=True)
    return df


async def _seed_history(client: OKXClient, symbol: str, tf: str, n: int) -> pd.DataFrame:
    raw = await client.get_candles(symbol, tf, limit=min(n + 5, 300))
    df = _normalize_okx_candles(raw)
    closed = df[df["closed"]].copy()
    if len(closed) > n:
        closed = closed.tail(n).reset_index(drop=True)
    return closed


async def _poll_loop(
    client: OKXClient,
    session: VolumeFarmerSession,
    symbol: str,
    tf: str,
    poll_sec: int,
    history: pd.DataFrame,
    end_at: Optional[datetime],
    log_dir: pathlib.Path,
    on_event,
) -> None:
    bar_seconds = _parse_timeframe_seconds(tf)
    last_bar_ts: Optional[pd.Timestamp] = (
        history["open_time"].iloc[-1] if not history.empty else None
    )
    LOGGER.info("seeded %d closed bars, last=%s", len(history), last_bar_ts)
    LOGGER.info("polling %s %s every %ds (bar size %ds)", symbol, tf, poll_sec, bar_seconds)

    while True:
        if end_at is not None and datetime.now(tz=timezone.utc) >= end_at:
            LOGGER.info("duration elapsed; stopping")
            break
        if session.halted:
            LOGGER.warning("session halted: %s", session.halt_reason)
            break

        try:
            raw = await client.get_candles(symbol, tf, limit=10)
            df_new = _normalize_okx_candles(raw)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("candle fetch failed: %s", exc)
            await asyncio.sleep(poll_sec)
            continue

        closed_new = df_new[df_new["closed"]].copy()
        if not closed_new.empty:
            if last_bar_ts is not None:
                closed_new = closed_new[closed_new["open_time"] > last_bar_ts]
            if not closed_new.empty:
                # append each newly closed bar to history and feed the session
                for _, row in closed_new.iterrows():
                    history = pd.concat([history, row.to_frame().T], ignore_index=True)
                    history = history.tail(500).reset_index(drop=True)
                    last_bar_ts = row["open_time"]
                    session.on_new_candle(history)
                    if session.halted:
                        break

        await asyncio.sleep(poll_sec)


def _write_daily_trade_log(log_dir: pathlib.Path, record: Dict[str, Any]) -> None:
    try:
        trades_dir = log_dir / "trades"
        trades_dir.mkdir(parents=True, exist_ok=True)
        day = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        with (trades_dir / f"{day}.jsonl").open("a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("trade log write failed: %s", exc)


def _make_event_handler(log_dir: pathlib.Path, symbol: str, session: VolumeFarmerSession,
                        live_executor: Optional["LiveVolumeExecutorOKX"] = None,
                        notifier: Optional[TelegramNotifier] = None,
                        mode_label: str = "PAPER"):
    _pending_entry: Dict[str, Any] = {}

    def _tg(msg: str) -> None:
        if notifier is None or not notifier.enabled:
            return
        try:
            asyncio.create_task(notifier.send_raw(msg))
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("tg send failed: %s", exc)

    def on_event(evt: FarmerEvent) -> None:
        if live_executor is not None:
            try:
                live_executor.consume_session_event(evt)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("live executor error: %s", exc)
        if evt.kind == "entry":
            p = evt.payload
            _pending_entry.update({
                "ts_entry": evt.time.isoformat(),
                "side": p.get("side", "").upper(),
                "entry_price": p.get("price"),
                "notional": p.get("notional"),
                "leverage": p.get("leverage"),
                "open_fee": p.get("open_fee"),
                "tp": p.get("tp"),
                "sl": p.get("sl"),
            })
            LOGGER.info(
                "ENTRY %s %s  price=%.2f  notional=$%.2f  lev=%.1fx  tp=%.2f  sl=%.2f",
                p["side"].upper(), symbol, p["price"], p["notional"],
                p.get("leverage", 0), p["tp"], p["sl"],
            )
            esc = TelegramNotifier.escape
            e_mode = esc(mode_label)
            e_side = esc(p["side"].upper())
            e_sym = esc(symbol)
            e_px = esc(f"{p['price']:.2f}")
            e_notional = esc(f"${p['notional']:.2f}")
            e_tp = esc(f"{p['tp']:.2f}")
            e_sl = esc(f"{p['sl']:.2f}")
            e_lev = esc(f"{p.get('leverage', 0):.0f}x")
            _tg(
                f"📈 *{e_mode} ENTRY*\n"
                f"`{e_side}` {e_sym}\n"
                f"px {e_px}  notional {e_notional}\n"
                f"TP {e_tp}  SL {e_sl}  lev {e_lev}"
            )
        elif evt.kind == "exit":
            p = evt.payload
            record = {
                "ts": evt.time.isoformat(),
                "symbol": symbol,
                "side": p["side"].upper(),
                "reason": p["reason"],
                "entry_price": p["entry_price"],
                "exit_price": p["exit_price"],
                "notional": p["notional"],
                "open_fee": _pending_entry.get("open_fee", p.get("open_fee")),
                "close_fee": p.get("close_fee"),
                "gross_pnl": p["gross_pnl"],
                "net_pnl": p["net_pnl"],
                "trade_num": p["round_trips"],
                "wins": p["wins"],
                "losses": p["losses"],
                "win_rate_pct": p.get("win_rate_pct"),
                "equity": p["equity"],
                "total_volume_usd": p.get("volume"),
                "bars_held": p["bars_held"],
            }
            _write_daily_trade_log(log_dir, record)
            sign = "+" if p["net_pnl"] >= 0 else ""
            LOGGER.info(
                "EXIT  %s  %s  %s -> %s  net=%s$%.4f  eq=$%.4f  vol=$%.0f  wr=%.1f%%  trade #%d",
                p["reason"], p["side"].upper(), p["entry_price"], p["exit_price"],
                sign, p["net_pnl"], p["equity"], p.get("volume", 0),
                p.get("win_rate_pct", 0), p["round_trips"],
            )
            emoji = "✅" if p["net_pnl"] >= 0 else "❌"
            esc = TelegramNotifier.escape
            e_mode = esc(mode_label)
            e_reason = esc(p["reason"])
            e_side = esc(p["side"].upper())
            e_entry = esc(f"{p['entry_price']:.2f}")
            e_exit = esc(f"{p['exit_price']:.2f}")
            e_net = esc(f"{sign}${p['net_pnl']:.4f}")
            e_eq = esc(f"${p['equity']:.4f}")
            e_vol = esc(f"${p.get('volume', 0):,.0f}")
            e_wr = esc(f"{p.get('win_rate_pct', 0):.1f}%")
            _tg(
                f"{emoji} *{e_mode} EXIT* `{e_reason}`\n"
                f"{e_side} {e_entry} → {e_exit}\n"
                f"net `{e_net}`  eq `{e_eq}`\n"
                f"vol `{e_vol}`  wr `{e_wr}`  trade `#{p['round_trips']}`"
            )
            _pending_entry.clear()
        elif evt.kind == "halt":
            LOGGER.warning("HALT: %s", evt.payload.get("reason"))
            esc = TelegramNotifier.escape
            e_mode = esc(mode_label)
            e_reason = esc(evt.payload.get("reason", "?"))
            e_eq = esc(f"${evt.payload.get('equity', 0):.4f}")
            e_vol = esc(f"${evt.payload.get('volume', 0):,.0f}")
            _tg(
                f"🛑 *{e_mode} HALT*\n"
                f"reason: `{e_reason}`\n"
                f"equity: `{e_eq}`\n"
                f"volume: `{e_vol}`"
            )
        elif evt.kind == "milestone":
            LOGGER.info("MILESTONE %s%%  volume=$%.0f", int(evt.payload.get("pct", 0) * 100),
                        evt.payload.get("volume", 0))
            esc = TelegramNotifier.escape
            e_pct = esc(f"{int(evt.payload.get('pct', 0) * 100)}%")
            e_vol = esc(f"${evt.payload.get('volume', 0):,.0f}")
            _tg(f"🎯 *MILESTONE* `{e_pct}` of volume target — `{e_vol}`")
    return on_event


def _save_state(session: VolumeFarmerSession, path: pathlib.Path) -> None:
    state = {
        "equity": session.equity,
        "peak_equity": session.peak_equity,
        "start_equity": session.start_equity,
        "total_volume_usd": session.total_volume_usd,
        "total_fees_gross": session.total_fees_gross,
        "total_pnl": session.total_pnl,
        "wins": session.wins,
        "losses": session.losses,
        "round_trips": session.round_trips,
        "halted": session.halted,
        "halt_reason": session.halt_reason,
        "saved_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


async def main_async(args: argparse.Namespace) -> int:
    cfg_path = pathlib.Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = _load_config(cfg_path)
    symbol = cfg["exchange"]["symbol"]
    tf = cfg["exchange"]["timeframe"]

    _load_env(PROJECT_ROOT / ".env")

    log_dir = pathlib.Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"volume_farmer_okx_{args.label or 'paper'}.log"
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, encoding="utf-8")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers, force=True,
    )

    state_path = (
        pathlib.Path(args.state_file) if args.state_file
        else PROJECT_ROOT / f"data/volume_farmer_okx_{args.label or 'paper'}_state.json"
    )

    end_at: Optional[datetime] = None
    if args.duration_days and args.duration_days > 0:
        from datetime import timedelta
        end_at = datetime.now(tz=timezone.utc) + timedelta(days=args.duration_days)

    LOGGER.info("config=%s symbol=%s tf=%s", cfg_path.name, symbol, tf)
    LOGGER.info("end_at=%s", end_at.isoformat() if end_at else "unbounded")
    LOGGER.info("logs=%s  state=%s", log_dir, state_path)

    # Telegram notifier — enabled when TG creds present in env, unless --no-telegram.
    os.environ.setdefault("BOT_LABEL", args.label or "okx")
    notifier = TelegramNotifier()
    if args.no_telegram:
        notifier._enabled = False  # noqa: SLF001
    if notifier.enabled:
        await notifier.start()

    session = VolumeFarmerSession(config=cfg)
    # Provisional handler — replaced below once execution mode is decided.
    handler = _make_event_handler(log_dir, symbol, session, notifier=notifier, mode_label="PAPER")
    session.event_callback = handler

    stop_evt = asyncio.Event()
    def _stop(*_): stop_evt.set()
    try:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _stop)
    except NotImplementedError:
        pass

    # Determine execution mode: paper (default), demo (OKX simulated), live (real money).
    is_live_or_demo = args.live or args.demo
    if args.live:
        ack = os.environ.get("LIVE_OKX_ACK", "")
        if ack != "I_UNDERSTAND" and not args.live_dry_run:
            LOGGER.error("--live requires LIVE_OKX_ACK=I_UNDERSTAND in env (or use --live-dry-run)")
            return 2
        LOGGER.warning("=== LIVE MODE: REAL MONEY ===")
    elif args.demo:
        LOGGER.info("=== DEMO MODE: OKX simulated trading (virtual USDT) ===")

    # The OKX client connects to simulated endpoints when simulated=True. For
    # --demo we want simulated=True; for --live we want simulated=False.
    client_simulated = args.demo and not args.live

    async with OKXClient(
        api_key=os.environ.get("OKX_API_KEY", ""),
        api_secret=os.environ.get("OKX_API_SECRET", ""),
        passphrase=os.environ.get("OKX_API_PASSPHRASE", ""),
        simulated=client_simulated,
    ) as client:
        live_executor: Optional[LiveVolumeExecutorOKX] = None
        if is_live_or_demo:
            live_executor = LiveVolumeExecutorOKX(
                client=client,
                symbol=symbol,
                pos_mode=args.pos_mode,
                mgn_mode="isolated",
                max_live_trades=args.max_live_trades,
                dry_run=args.live_dry_run,
                log_dir=log_dir,
            )
            try:
                await live_executor.initialize(leverage_cap=args.live_leverage)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("live executor init failed: %s", exc)
                return 2

        try:
            history = await _seed_history(client, symbol, tf, SEED_CANDLES)
        except Exception as exc:
            LOGGER.error("failed to seed history: %s", exc)
            return 2

        # Decide mode label for messages.
        mode_label = (
            "LIVE" if args.live else
            "DEMO" if args.demo else
            "PAPER"
        )
        # Rebuild handler now that we have the live_executor (if any).
        handler = _make_event_handler(log_dir, symbol, session, live_executor,
                                      notifier=notifier, mode_label=mode_label)
        session.event_callback = handler

        # STARTUP message — go.
        if notifier.enabled:
            last_bar_ts = history["open_time"].iloc[-1] if not history.empty else "n/a"
            esc = TelegramNotifier.escape
            tp_bps = cfg["farmer"]["tp_bps"]
            sl_bps = cfg["farmer"]["sl_bps"]
            cap = cfg["farmer"]["capital_usd"]
            maker_bps = cfg["fees"]["maker"] * 1e4
            taker_bps = cfg["fees"]["taker"] * 1e4
            rebate_pct = cfg["fees"]["rebate_pct"] * 100
            dur = f"{args.duration_days}d" if args.duration_days else "unbounded"
            startup_msg = (
                f"🚀 *vGen OKX bot started* `{esc(mode_label)}`\n"
                f"config: `{esc(cfg_path.name)}`\n"
                f"symbol: `{esc(symbol)}`  tf: `{esc(tf)}`\n"
                f"tp/sl: `{esc(f'{tp_bps:.0f}/{sl_bps:.0f}bps')}`  cap: `{esc(f'${cap:.2f}')}`\n"
                f"fees: `{esc(f'{maker_bps:.0f}/{taker_bps:.0f}bps')}`  "
                f"rebate: `{esc(f'{rebate_pct:.0f}%')}`\n"
                f"seeded `{len(history)}` bars, last `{esc(str(last_bar_ts))}`\n"
                f"poll: `{esc(f'{args.poll_seconds}s')}`  duration: `{esc(dur)}`"
            )
            await notifier.send_raw(startup_msg)

        poll_task = asyncio.create_task(_poll_loop(
            client, session, symbol, tf, args.poll_seconds,
            history, end_at, log_dir, handler,
        ))
        try:
            done, _ = await asyncio.wait(
                {poll_task, asyncio.create_task(stop_evt.wait())},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            poll_task.cancel()
            try:
                await poll_task
            except asyncio.CancelledError:
                pass

    _save_state(session, state_path)
    s = session.summary()
    LOGGER.info("FINAL: trades=%d  wr=%.2f%%  eq=$%.4f  vol=$%.0f  pnl=$%.4f",
                s["round_trips"], s["win_rate_pct"], s["equity"], s["volume_usd"], s["total_pnl"])
    if notifier.enabled:
        try:
            esc = TelegramNotifier.escape
            e_wr = esc(f"{s['win_rate_pct']:.2f}%")
            e_eq = esc(f"${s['equity']:.4f}")
            e_vol = esc(f"${s['volume_usd']:,.0f}")
            e_pnl = esc(f"${s['total_pnl']:.4f}")
            await notifier.send_raw(
                f"⏹️ *vGen OKX bot stopped*\n"
                f"trades: `{s['round_trips']}`  wr: `{e_wr}`\n"
                f"eq: `{e_eq}`  vol: `{e_vol}`  pnl: `{e_pnl}`"
            )
            await notifier.stop()
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("notifier shutdown failed: %s", exc)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="OKX paper-trade volume farmer")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--label", type=str, default="paper")
    p.add_argument("--state-file", type=str, default=None)
    p.add_argument("--log-dir", type=str, default="data/logs")
    p.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    p.add_argument("--duration-days", type=float, default=0.0,
                   help="Auto-stop after N days; 0 = run until Ctrl-C")
    p.add_argument("--demo", action="store_true",
                   help="Submit real orders to OKX simulated-trading (virtual USDT)")
    p.add_argument("--live", action="store_true",
                   help="Submit real orders with REAL MONEY (needs LIVE_OKX_ACK=I_UNDERSTAND)")
    p.add_argument("--live-dry-run", action="store_true",
                   help="With --live: build orders but do NOT submit")
    p.add_argument("--max-live-trades", type=int, default=100,
                   help="Hard cap on number of live/demo orders submitted")
    p.add_argument("--live-leverage", type=int, default=50,
                   help="Leverage cap to set on OKX for the symbol")
    p.add_argument("--pos-mode", type=str, default="net", choices=["net", "hedge"],
                   help="OKX position mode (net = one-way, hedge = both sides)")
    p.add_argument("--no-telegram", action="store_true",
                   help="Disable Telegram notifications even if creds are set")
    return asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
