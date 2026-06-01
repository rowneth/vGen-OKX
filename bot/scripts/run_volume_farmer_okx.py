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
import io
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

_SEP = "━" * 22

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


def _draw_entry_chart(
    bars: pd.DataFrame,
    side: str,
    entry_price: float,
    tp_price: float,
    sl_price: float,
) -> Optional[bytes]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        BG      = "#0b0e11"
        BULL    = "#089981"
        BEAR    = "#f23645"
        C_ENTRY = "#3b82f6"
        C_TP    = "#22c55e"
        C_SL    = "#ef4444"

        df = bars.tail(30).reset_index(drop=True)
        n = len(df)
        if n < 3:
            return None

        lows  = df["low"].astype(float)
        highs = df["high"].astype(float)
        all_p = list(lows) + list(highs) + [tp_price, sl_price, entry_price]
        p_range = max(all_p) - min(all_p)
        pad = p_range * 0.12
        y_lo = min(all_p) - pad
        y_hi = max(all_p) + pad

        fig, ax = plt.subplots(figsize=(10, 3.6))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        ax.set_xlim(-0.8, n + 4.5)
        ax.set_ylim(y_lo, y_hi)
        ax.axis("off")

        for i, row in df.iterrows():
            op, hi, lo, cl = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            color = BULL if cl >= op else BEAR
            ax.plot([i, i], [lo, hi], color=color, linewidth=0.7, zorder=1, solid_capstyle="round")
            body_lo = min(op, cl)
            height  = max(abs(cl - op), p_range * 0.004)
            ax.add_patch(mpatches.FancyBboxPatch(
                (i - 0.32, body_lo), 0.64, height,
                boxstyle="square,pad=0", facecolor=color, edgecolor="none", zorder=2,
            ))

        lx = n + 0.5  # label x
        for price, color, tag in [
            (tp_price,    C_TP,    f"TP  {tp_price:,.1f}"),
            (entry_price, C_ENTRY, f"    {entry_price:,.1f}"),
            (sl_price,    C_SL,    f"SL  {sl_price:,.1f}"),
        ]:
            ax.plot([-0.8, n - 0.5], [price, price], color=color,
                    linewidth=0.9, linestyle="--", alpha=0.8, zorder=3)
            ax.text(lx, price, tag, color=color, va="center",
                    fontsize=7.5, fontweight="bold", zorder=4,
                    fontfamily="monospace")

        mk = "^" if side.lower() == "long" else "v"
        ax.scatter([n - 1], [entry_price], marker=mk, color=C_ENTRY,
                   s=90, zorder=5, edgecolors="white", linewidths=0.4)

        fig.subplots_adjust(left=0.01, right=0.82, top=0.97, bottom=0.02)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140, facecolor=BG)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as exc:
        LOGGER.warning("chart draw failed: %s", exc)
        return None


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
    notifier: Optional[TelegramNotifier] = None,
    mode_label: str = "PAPER",
    bot_label: str = "",
    cfg: Optional[Dict[str, Any]] = None,
    intrabar: Optional[Dict[str, Any]] = None,
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

        # Intrabar TP/SL detection — fire Telegram immediately on hit, don't wait for bar close.
        if notifier and notifier.enabled and session.position is not None and cfg is not None and intrabar is not None:
            pos = session.position
            unclosed = df_new[~df_new["closed"]]
            if not unclosed.empty:
                cur_hi = float(unclosed["high"].iloc[-1])
                cur_lo = float(unclosed["low"].iloc[-1])
                hit: Optional[str] = None
                if pos.side == "long":
                    if cur_hi >= pos.tp:
                        hit = "tp"
                    elif cur_lo <= pos.sl:
                        hit = "sl"
                else:
                    if cur_lo <= pos.tp:
                        hit = "tp"
                    elif cur_hi >= pos.sl:
                        hit = "sl"
                ikey = f"{round(pos.entry_price, 4)}"
                if hit and intrabar.get(ikey) != hit:
                    intrabar[ikey] = hit
                    exit_px = pos.tp if hit == "tp" else pos.sl
                    limit_tp = cfg.get("farmer", {}).get("limit_tp", False)
                    maker_r = cfg["fees"]["maker"]
                    taker_r = cfg["fees"]["taker"]
                    close_r = maker_r if (hit == "tp" and limit_tp) else taker_r
                    fee_type = "maker" if (hit == "tp" and limit_tp) else "taker"
                    if pos.side == "long":
                        gross = (exit_px - pos.entry_price) / pos.entry_price * pos.notional
                    else:
                        gross = (pos.entry_price - exit_px) / pos.entry_price * pos.notional
                    close_fee = pos.notional * close_r
                    net = gross - close_fee
                    est_wins = session.wins + (1 if net > 0 else 0)
                    est_losses = session.losses + (0 if net > 0 else 1)
                    est_trips = session.round_trips + 1
                    est_wr = est_wins / est_trips * 100 if est_trips else 0
                    est_equity = session.equity + gross - close_fee
                    est_vol = session.total_volume_usd + pos.notional
                    vol_target = cfg.get("target", {}).get("volume_usd", 0)
                    vol_pct = est_vol / vol_target * 100 if vol_target else 0
                    _esc = TelegramNotifier.escape
                    g_str = f"+${abs(gross):.4f}" if gross >= 0 else f"-${abs(gross):.4f}"
                    n_str = f"+${abs(net):.4f}" if net >= 0 else f"-${abs(net):.4f}"
                    emoji = "✅" if hit == "tp" else "❌"
                    _lbl = f"\n{_SEP}\n\\[{_esc(bot_label)}\\]" if bot_label else ""
                    text = (
                        f"{emoji} *{_esc(hit.upper())} · {_esc(pos.side.upper())} \\#{_esc(str(est_trips))} · {_esc(mode_label)}*\n"
                        f"{_esc(symbol)}  `{pos.entry_price:,.2f}` → `{exit_px:,.2f}`\n"
                        f"{_SEP}\n"
                        f"Gross   `{g_str}`\n"
                        f"Fee     `${close_fee:.4f}` \\({_esc(fee_type)}\\)\n"
                        f"Net     `{n_str}`\n"
                        f"{_SEP}\n"
                        f"`{est_wins}W · {est_losses}L · {est_wr:.1f}% WR`\n"
                        f"{_SEP}\n"
                        f"Balance  `${est_equity:.4f}`\n"
                        f"{_SEP}\n"
                        f"Vol  `${est_vol:,.0f}` / `${vol_target:,.0f}`  \\[`{vol_pct:.1f}%`\\]"
                        f"{_lbl}"
                    )
                    entry_msg_id = intrabar.get(f"{ikey}_msg_id")
                    asyncio.create_task(notifier.send_and_get_id(text, reply_to_message_id=entry_msg_id))
                    LOGGER.info("INTRABAR %s detected for %s pos @ %.2f, notified", hit, pos.side, pos.entry_price)

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


def _make_event_handler(
    log_dir: pathlib.Path,
    symbol: str,
    session: VolumeFarmerSession,
    live_executor: Optional["LiveVolumeExecutorOKX"] = None,
    notifier: Optional[TelegramNotifier] = None,
    mode_label: str = "PAPER",
    client: Optional["OKXClient"] = None,
    tf: str = "5m",
    bot_label: str = "",
    intrabar: Optional[Dict[str, Any]] = None,
):
    _pending: Dict[str, Any] = {}
    _esc = TelegramNotifier.escape
    _lbl = f"\n{_SEP}\n\\[{_esc(bot_label)}\\]" if bot_label else ""

    async def _send_entry_with_chart(text: str, side: str, entry: float, tp: float, sl: float) -> None:
        chart: Optional[bytes] = None
        if client is not None:
            try:
                raw = await client.get_candles(symbol, tf, limit=35)
                bars = _normalize_okx_candles(raw)
                chart = _draw_entry_chart(bars[bars["closed"]], side, entry, tp, sl)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("chart generation failed: %s", exc)
        if chart:
            msg_id = await notifier.send_photo(chart, caption=text)
        else:
            msg_id = await notifier.send_and_get_id(text)
        _pending["msg_id"] = msg_id
        # Share entry msg_id with poll loop so intrabar reply threading works
        if intrabar is not None:
            ikey = f"{round(entry, 4)}"
            intrabar[f"{ikey}_msg_id"] = msg_id

    async def _send_exit(text: str, reply_to: Optional[int]) -> None:
        await notifier.send_and_get_id(text, reply_to_message_id=reply_to)

    def _tg(msg: str) -> None:
        if notifier and notifier.enabled:
            asyncio.create_task(notifier.send_raw(msg + _lbl))

    def on_event(evt: FarmerEvent) -> None:
        if live_executor is not None:
            try:
                live_executor.consume_session_event(evt)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("live executor error: %s", exc)

        if evt.kind == "entry":
            p = evt.payload
            side = p["side"]
            entry = p["price"]
            tp = p["tp"]
            sl = p["sl"]
            tp_bps = p.get("tp_bps", abs(tp - entry) / entry * 10_000)
            sl_bps = p.get("sl_bps", abs(sl - entry) / entry * 10_000)
            notional = p["notional"]
            margin = p.get("margin", 0.0)
            lev = p.get("leverage", 0)
            open_fee = p.get("open_fee", 0)
            equity = p.get("equity", session.equity)
            trade_num = p.get("round_trips", session.round_trips) + 1

            _pending.update({"side": side, "entry": entry, "tp": tp, "sl": sl,
                              "open_fee": open_fee, "msg_id": None})

            LOGGER.info(
                "ENTRY %s %s  price=%.2f  notional=$%.2f  lev=%.1fx  tp=%.2f  sl=%.2f",
                side.upper(), symbol, entry, notional, lev, tp, sl,
            )

            if notifier and notifier.enabled:
                arrow = "🟢" if side == "long" else "🔴"
                text = (
                    f"{arrow} *{_esc(side.upper())} \\#{_esc(str(trade_num))} · {_esc(mode_label)}*\n"
                    f"{_esc(symbol)} · {_esc(tf)}\n"
                    f"{_SEP}\n"
                    f"Entry →  `{entry:,.2f}`\n"
                    f"TP    →  `{tp:,.2f}`   `+{tp_bps:.1f}bps`\n"
                    f"SL    →  `{sl:,.2f}`   `-{sl_bps:.1f}bps`\n"
                    f"{_SEP}\n"
                    f"Margin  `${margin:,.2f}`  ·  Lev `{lev:.0f}×`\n"
                    f"Size    `${notional:,.2f}`\n"
                    f"{_SEP}\n"
                    f"Open fee  `${open_fee:.4f}` \\(maker\\)\n"
                    f"{_SEP}\n"
                    f"Balance  `${equity:.4f}`"
                    f"{_lbl}"
                )
                asyncio.create_task(_send_entry_with_chart(text, side, entry, tp, sl))

        elif evt.kind == "exit":
            p = evt.payload
            side = p["side"]
            reason = p["reason"]
            entry_px = p["entry_price"]
            exit_px = p["exit_price"]
            gross = p["gross_pnl"]
            net = p["net_pnl"]
            close_fee = p.get("close_fee", 0)
            close_fee_type = p.get("close_fee_type", "taker")
            equity = p["equity"]
            wins = p["wins"]
            losses = p["losses"]
            wr = p.get("win_rate_pct", 0)
            trade_num = p["round_trips"]
            vol_now = p.get("volume", 0)
            vol_target = p.get("volume_target", 0)

            record = {
                "ts": evt.time.isoformat(), "symbol": symbol,
                "side": side, "reason": reason,
                "entry_price": entry_px, "exit_price": exit_px,
                "notional": p["notional"],
                "open_fee": _pending.get("open_fee", p.get("open_fee")),
                "close_fee": close_fee, "gross_pnl": gross, "net_pnl": net,
                "trade_num": trade_num, "wins": wins, "losses": losses,
                "win_rate_pct": wr, "equity": equity,
                "total_volume_usd": vol_now, "bars_held": p["bars_held"],
            }
            _write_daily_trade_log(log_dir, record)

            sign = "+" if net >= 0 else ""
            LOGGER.info(
                "EXIT  %s  %s  %s -> %s  net=%s$%.4f  eq=$%.4f  vol=$%.0f  wr=%.1f%%  trade #%d",
                reason, side.upper(), entry_px, exit_px, sign, net, equity,
                vol_now, wr, trade_num,
            )

            ikey = f"{round(entry_px, 4)}"
            already_notified = intrabar is not None and intrabar.get(ikey) == reason
            if already_notified and intrabar is not None:
                intrabar.pop(ikey, None)
                intrabar.pop(f"{ikey}_msg_id", None)

            if notifier and notifier.enabled and not already_notified:
                emoji = "✅" if reason == "tp" else ("❌" if reason == "sl" else "⏱")
                g_str = f"+${abs(gross):.4f}" if gross >= 0 else f"-${abs(gross):.4f}"
                n_str = f"+${abs(net):.4f}" if net >= 0 else f"-${abs(net):.4f}"
                vol_pct = vol_now / vol_target * 100 if vol_target else 0
                text = (
                    f"{emoji} *{_esc(reason.upper())} · {_esc(side.upper())} \\#{_esc(str(trade_num))} · {_esc(mode_label)}*\n"
                    f"{_esc(symbol)}  `{entry_px:,.2f}` → `{exit_px:,.2f}`\n"
                    f"{_SEP}\n"
                    f"Gross   `{g_str}`\n"
                    f"Fee     `${close_fee:.4f}` \\({_esc(close_fee_type)}\\)\n"
                    f"Net     `{n_str}`\n"
                    f"{_SEP}\n"
                    f"`{wins}W · {losses}L · {wr:.1f}% WR`\n"
                    f"{_SEP}\n"
                    f"Balance  `${equity:.4f}`\n"
                    f"{_SEP}\n"
                    f"Vol  `${vol_now:,.0f}` / `${vol_target:,.0f}`  \\[`{vol_pct:.1f}%`\\]"
                    f"{_lbl}"
                )
                entry_msg_id = _pending.get("msg_id")
                asyncio.create_task(_send_exit(text, entry_msg_id))
            _pending.clear()

        elif evt.kind == "halt":
            LOGGER.warning("HALT: %s", evt.payload.get("reason"))
            if notifier and notifier.enabled:
                reason = evt.payload.get("reason", "?")
                eq = evt.payload.get("equity", 0)
                vol = evt.payload.get("volume", 0)
                _tg(
                    f"🛑 *HALT · {_esc(mode_label)}*\n"
                    f"{_SEP}\n"
                    f"Reason  `{_esc(reason)}`\n"
                    f"Equity  `${eq:.4f}`\n"
                    f"Volume  `${vol:,.0f}`"
                )

        elif evt.kind == "milestone":
            pct = int(evt.payload.get("pct", 0) * 100)
            vol = evt.payload.get("volume", 0)
            vol_target = evt.payload.get("volume_target", 0)
            LOGGER.info("MILESTONE %d%%  volume=$%.0f", pct, vol)
            if notifier and notifier.enabled:
                _tg(
                    f"🎯 *Milestone {_esc(str(pct))}%* of volume target\n"
                    f"Vol  `${vol:,.0f}` / `${vol_target:,.0f}`"
                )

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
    notifier._label = ""  # noqa: SLF001 — label appended manually at message bottom
    if args.no_telegram:
        notifier._enabled = False  # noqa: SLF001
    if notifier.enabled:
        await notifier.start()

    _intrabar: Dict[str, Any] = {}  # shared state: poll loop ↔ event handler

    session = VolumeFarmerSession(config=cfg)
    # Provisional handler — replaced below once execution mode is decided.
    handler = _make_event_handler(log_dir, symbol, session, notifier=notifier, mode_label="PAPER", tf=tf,
                                   bot_label=args.label or "okx-paper", intrabar=_intrabar)
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
                                      notifier=notifier, mode_label=mode_label,
                                      client=client, tf=tf, bot_label=args.label or "okx-paper",
                                      intrabar=_intrabar)
        session.event_callback = handler

        # STARTUP message.
        if notifier.enabled:
            _esc = TelegramNotifier.escape
            cap = cfg["farmer"]["capital_usd"]
            maker_bps = cfg["fees"]["maker"] * 1e4
            taker_bps = cfg["fees"]["taker"] * 1e4
            rebate_pct = cfg["fees"]["rebate_pct"] * 100
            dur = f"{args.duration_days}d" if args.duration_days else "unbounded"
            atr_cfg = cfg.get("farmer", {}).get("atr", {})
            atr_min = atr_cfg.get("min_usd", 0)
            atr_tp_mult = atr_cfg.get("tp_mult", 0.5)
            atr_sl_mult = atr_cfg.get("sl_mult", 1.5)
            limit_tp = cfg.get("farmer", {}).get("limit_tp", False)
            min_lev = cfg.get("farmer", {}).get("sizing", {}).get("min_leverage", 5)
            max_lev = cfg.get("farmer", {}).get("sizing", {}).get("max_leverage", 125)
            vol_target = cfg.get("target", {}).get("volume_usd", 0)
            startup_msg = (
                f"🚀 *vGen OKX · {_esc(mode_label)}*\n"
                f"{_esc(symbol)} · {_esc(tf)}\n"
                f"{_SEP}\n"
                f"Capital    `${cap:.2f}`\n"
                f"Lev        `{min_lev}-{max_lev}×` \\(dynamic\\)\n"
                f"ATR filter `≥${atr_min:.0f}` \\(Wilder\\-14\\)\n"
                f"TP `{atr_tp_mult}×` ATR  ·  SL `{atr_sl_mult}×` ATR\n"
                f"Limit TP   `{'✓' if limit_tp else '✗'}` \\(maker fill\\)\n"
                f"{_SEP}\n"
                f"Maker  `{maker_bps:.0f}bps`  ·  rebate `{rebate_pct:.0f}%`\n"
                f"Taker  `{taker_bps:.0f}bps`\n"
                f"{_SEP}\n"
                f"Target  `${vol_target:,.0f}` volume\n"
                f"Duration  `{_esc(dur)}`  ·  poll `{args.poll_seconds}s`\n"
                f"{_SEP}\n"
                f"\\[{_esc(args.label or 'okx-paper')}\\]"
            )
            await notifier.send_raw(startup_msg)

        poll_task = asyncio.create_task(_poll_loop(
            client, session, symbol, tf, args.poll_seconds,
            history, end_at, log_dir, handler,
            notifier=notifier, mode_label=mode_label,
            bot_label=args.label or "okx-paper", cfg=cfg, intrabar=_intrabar,
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
            pnl_str = f"+${s['total_pnl']:.4f}" if s["total_pnl"] >= 0 else f"-${abs(s['total_pnl']):.4f}"
            await notifier.send_raw(
                f"⏹ *vGen OKX stopped · {TelegramNotifier.escape(mode_label)}*\n"
                f"{_SEP}\n"
                f"Trades  `{s['round_trips']}`  ·  WR `{s['win_rate_pct']:.1f}%`\n"
                f"{_SEP}\n"
                f"Equity  `${s['equity']:.4f}`  ·  PnL  `{pnl_str}`\n"
                f"Vol     `${s['volume_usd']:,.0f}`\n"
                f"{_SEP}\n"
                f"\\[{TelegramNotifier.escape(args.label or 'okx-paper')}\\]"
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
