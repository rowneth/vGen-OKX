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

from exchange.okx_client import OKXClient, to_okx_bar, to_okx_inst_id  # noqa: E402
from execution.live_volume_executor_okx import (  # noqa: E402
    EntryRepegConfig,
    LiveVolumeExecutorOKX,
    _fmt_size,
    _round_to_tick,
)
from execution.maker_exit import MakerExitConfig  # noqa: E402
from execution.volume_farmer import FarmerEvent, VolumeFarmerSession  # noqa: E402
from monitoring.telegram_notifier import TelegramNotifier  # noqa: E402

LOGGER = logging.getLogger("vf_okx")

_SEP = "━" * 22

_MENU_BUTTONS = [
    [
        {"text": "📊 Status",    "callback_data": "cmd_status"},
        {"text": "📈 Position",  "callback_data": "cmd_position"},
    ],
    [
        {"text": "📋 Orders",    "callback_data": "cmd_orders"},
        {"text": "🛑 Close",     "callback_data": "cmd_close"},
    ],
    [
        {"text": "💰 Fees",      "callback_data": "cmd_fees"},
        {"text": "📋 Trades",    "callback_data": "cmd_trades"},
    ],
    [
        {"text": "❓ Help",      "callback_data": "cmd_help"},
    ],
]

# Confirm card buttons (shown by /close with no sub-command). Closing real money
# is two-tap: the user reaches this card, then taps one of these to execute.
_CLOSE_CONFIRM_BUTTONS = [
    [{"text": "🛑 Market close NOW (taker)", "callback_data": "cmd_close_market"}],
    [{"text": "✋ Maker limit @ touch",      "callback_data": "cmd_close_limit"}],
    [{"text": "✖️ Dismiss",                  "callback_data": "cmd_dismiss"}],
]

# Buttons attached to the EXIT-FAILED (position-still-open) alert.
_STUCK_BUTTONS = [
    [
        {"text": "📈 Position", "callback_data": "cmd_position"},
        {"text": "📋 Orders",   "callback_data": "cmd_orders"},
    ],
    [{"text": "🛑 Close position", "callback_data": "cmd_close"}],
]

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
    symbol: str = "BTC_USDT",
    tf: str = "5m",
) -> Optional[bytes]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.gridspec as gridspec

        BG      = "#0d1117"
        PANEL   = "#0d1117"
        BULL    = "#089981"
        BEAR    = "#f23645"
        C_ENTRY = "#f0c040"
        C_TP    = "#089981"
        C_SL    = "#f23645"
        GRID    = "#1c2333"
        TEXT    = "#c9d1d9"

        df = bars.tail(40).reset_index(drop=True)
        n = len(df)
        if n < 3:
            return None

        fig = plt.figure(figsize=(12, 6.2), facecolor=BG)
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.03, figure=fig)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        for ax in (ax1, ax2):
            ax.set_facecolor(PANEL)
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.tick_params(colors=TEXT, labelsize=8, which="both", direction="out", length=3)
            for spine in ax.spines.values():
                spine.set_edgecolor("#21262d")

        # Candlesticks + volume bars
        for i, row in df.iterrows():
            op  = float(row["open"])
            hi  = float(row["high"])
            lo  = float(row["low"])
            cl  = float(row["close"])
            vol = float(row.get("volume", 0))
            color = BULL if cl >= op else BEAR
            ax1.plot([i, i], [lo, hi], color=color, linewidth=0.75, zorder=2)
            body_lo = min(op, cl)
            height  = max(abs(cl - op), (hi - lo) * 0.008 + 0.5)
            ax1.add_patch(mpatches.FancyBboxPatch(
                (i - 0.36, body_lo), 0.72, height,
                boxstyle="square,pad=0", facecolor=color, edgecolor="none", zorder=3,
            ))
            ax2.bar(i, vol, width=0.72, color=color, alpha=0.85, edgecolor="none")

        # Price levels
        ax1.axhline(entry_price, color=C_ENTRY, linewidth=1.1, linestyle="--", zorder=4, alpha=0.9)
        ax1.axhline(tp_price,    color=C_TP,    linewidth=1.1, linestyle="-",  zorder=4, alpha=0.9)
        ax1.axhline(sl_price,    color=C_SL,    linewidth=1.1, linestyle="-",  zorder=4, alpha=0.9)

        # Grid
        ax1.grid(True, color=GRID, linewidth=0.5, linestyle="-", zorder=1)
        ax2.grid(True, color=GRID, linewidth=0.5, linestyle="-", axis="y", zorder=1)

        # Y-axis limits
        all_p = list(df["low"].astype(float)) + list(df["high"].astype(float)) + [tp_price, sl_price]
        p_range = max(all_p) - min(all_p)
        ax1.set_ylim(min(all_p) - p_range * 0.10, max(all_p) + p_range * 0.10)
        ax1.set_ylabel("Price (USDT)", color=TEXT, fontsize=8, rotation=270, labelpad=14)
        ax1.tick_params(axis="y", colors=TEXT, labelsize=8)

        # Volume y-axis
        ax2.set_ylabel("Volume", color=TEXT, fontsize=8, rotation=270, labelpad=14)
        ax2.tick_params(axis="y", colors=TEXT, labelsize=7)

        # X-axis timestamps
        step = max(1, n // 7)
        ticks = list(range(0, n, step))
        if "open_time" in df.columns and hasattr(df["open_time"].iloc[0], "strftime"):
            labels = [df["open_time"].iloc[i].strftime("%b %d, %H:%M") for i in ticks]
        else:
            labels = [str(i) for i in ticks]
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(labels, rotation=0, ha="center", fontsize=7.5, color=TEXT)
        plt.setp(ax1.get_xticklabels(), visible=False)

        # Legend (top-left of price panel)
        legend_elements = [
            mpatches.Patch(facecolor=C_ENTRY, edgecolor="none", label=f"Entry  {entry_price:,.1f}"),
            mpatches.Patch(facecolor=C_TP,    edgecolor="none", label=f"TP     {tp_price:,.1f}"),
            mpatches.Patch(facecolor=C_SL,    edgecolor="none", label=f"SL     {sl_price:,.1f}"),
        ]
        ax1.legend(
            handles=legend_elements, loc="upper left",
            framealpha=0.75, facecolor="#161b22", edgecolor="#30363d",
            labelcolor=TEXT, fontsize=8.5,
            handlelength=1.2, handleheight=0.9, borderpad=0.7, labelspacing=0.4,
            prop={"family": "monospace", "size": 8.5},
        )

        # Title
        arrow = "▼" if side.lower() == "short" else "▲"
        dir_label = "SHORT" if side.lower() == "short" else "LONG"
        fig.suptitle(
            f"{symbol}  {tf}  —  {arrow} {dir_label}  entry {entry_price:,.1f}",
            color=TEXT, fontsize=11, y=0.99, x=0.44,
        )

        ax1.set_xlim(-0.8, n - 0.2)
        fig.subplots_adjust(left=0.04, right=0.92, top=0.95, bottom=0.08)
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
        # PAPER only: this is a SIMULATED touch off local candles with simulated PnL. In
        # demo/live the authoritative exit notice is the executor's on_close_confirmed card
        # (real fills), so we suppress the simulated touch message to avoid quoting fake numbers.
        if (notifier and notifier.enabled and mode_label == "PAPER"
                and session.position is not None and cfg is not None and intrabar is not None):
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
    state_path: Optional[pathlib.Path] = None,
    cfg: Optional[Dict[str, Any]] = None,
):
    _pending: Dict[str, Any] = {}
    _esc = TelegramNotifier.escape
    _lbl = f"\n{_SEP}\n\\[{_esc(bot_label)}\\]" if bot_label else ""
    # In demo/live the balance/PnL shown must come from the REAL OKX account, not
    # the local paper simulation. PAPER mode keeps the simulated numbers.
    #
    # IMPORTANT: in live/demo the trade-lifecycle cards (entry/exit) are NOT
    # rendered here from the simulation. They are emitted by the executor at real
    # OKX confirmation points (on_entry_filled / on_close_confirmed). This handler
    # only drives them in PAPER mode. HALT/MILESTONE/REBATE are rebased on real
    # account numbers in live so no card ever quotes the simulation.
    is_real = mode_label != "PAPER"

    async def _send_entry_with_chart(
        text: str, side: str, entry: float, tp: float, sl: float, equity_sim: float = 0.0,
    ) -> None:
        # Append the balance footer: real demo/live account in real mode, else sim equity.
        if is_real and client is not None:
            acct = await _fetch_real_account(client, symbol)
            text = text + "\n" + "\n".join(_real_acct_lines(acct))
        else:
            text = text + f"\n{_SEP}\nBalance  `${equity_sim:.4f}`"
        text = text + _lbl
        chart: Optional[bytes] = None
        if client is not None:
            try:
                raw = await client.get_candles(symbol, tf, limit=35)
                bars = _normalize_okx_candles(raw)
                chart = _draw_entry_chart(bars[bars["closed"]], side, entry, tp, sl, symbol=symbol, tf=tf)
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
        # Attach entry msg_id to the executor's open trade so the real-fill
        # callback can thread its reply under the same entry message.
        if (
            live_executor is not None
            and live_executor._open_trade is not None  # noqa: SLF001
            and msg_id is not None
        ):
            live_executor._open_trade.extras["entry_msg_id"] = msg_id  # noqa: SLF001

    async def _send_exit(text: str, reply_to: Optional[int]) -> None:
        # Real mode: append the live demo/live account snapshot so the message
        # carries actual balance/position alongside the strategy's view.
        if is_real and client is not None:
            acct = await _fetch_real_account(client, symbol)
            text = text + "\n" + "\n".join(_real_acct_lines(acct))
        text = text + _lbl
        await notifier.send_and_get_id(text, reply_to_message_id=reply_to)

    def _tg(msg: str) -> None:
        if notifier and notifier.enabled:
            asyncio.create_task(notifier.send_raw(msg + _lbl))

    async def _send_auto_real(kind: str, payload: Dict[str, Any]) -> None:
        """HALT / MILESTONE / REBATE card rebased on REAL OKX numbers (live mode).

        The strategy's bookkeeping still decides WHEN to fire, but every number
        shown is fetched from the real account / order history so the card can
        never quote the simulation. Best-effort: a failed fetch degrades to
        "unavailable", never to a SIM figure.
        """
        if not (notifier and notifier.enabled):
            return
        acct: Dict[str, Any] = {}
        stats: Optional[Dict[str, Any]] = None
        if client is not None:
            try:
                acct = await _fetch_real_account(client, symbol)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("auto-card real acct fetch failed: %s", exc)
            try:
                stats = await _fetch_okx_trade_stats(client, symbol)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("auto-card real stats fetch failed: %s", exc)
        eq = acct.get("equity") if isinstance(acct, dict) else None
        eq_str = f"${eq:,.4f}" if eq is not None else "unavailable"
        real_vol = stats.get("volume") if stats else None
        vol_str = f"${real_vol:,.0f}" if real_vol is not None else "unavailable"
        real_fees = stats.get("fees") if stats else None
        rebate_pct = float(((cfg or {}).get("fees", {}) or {}).get("rebate_pct", 0.40))
        rebate_est = (real_fees * rebate_pct) if real_fees is not None else None

        if kind == "halt":
            reason = str(payload.get("reason", "?"))
            text = (
                f"🛑 *HALT · {_esc(mode_label)}*\n"
                f"{_SEP}\n"
                f"Reason  `{reason}`\n"
                f"Wallet  `{eq_str}`  \\(real\\)\n"
                f"Volume  `{vol_str}`  \\(real\\)"
            )
        elif kind == "milestone":
            vol_target = float(payload.get("volume_target", 0) or 0)
            real_pct = (real_vol / vol_target * 100) if (real_vol is not None and vol_target) else None
            pct_line = f"  \\[`{real_pct:.1f}%`\\]" if real_pct is not None else ""
            text = (
                f"🎯 *Volume milestone · {_esc(mode_label)}*\n"
                f"{_SEP}\n"
                f"Real volume  `{vol_str}` / `${vol_target:,.0f}`{pct_line}"
            )
        elif kind == "rebate_reminder":
            handles = list(payload.get("mention_handles", []) or [])
            # @rowneth holds the rebate wallet — always pinged first, even if the
            # config's mention list is empty or omits them.
            if "@rowneth" not in handles:
                handles.insert(0, "@rowneth")
            mention = " ".join(_esc(h) for h in handles)
            head = mention + "\n" if mention else ""
            # Amount to transfer: prefer the real-fee-derived rebate estimate;
            # fall back to the session's suggested amount so the ask is concrete.
            transfer = rebate_est if (rebate_est is not None and rebate_est > 0) \
                else float(payload.get("suggested_amount", 0.0) or 0.0)
            reb_str = f"${rebate_est:.4f}" if rebate_est is not None else "unavailable"
            fees_str = f"${real_fees:.4f}" if real_fees is not None else "unavailable"
            text = (
                f"{head}"
                f"⚠️ *Rebate top\\-up needed · {_esc(mode_label)}*\n"
                f"{_SEP}\n"
                f"Wallet            `{eq_str}`\n"
                f"Real fees paid    `{fees_str}`\n"
                f"Est rebate {int(rebate_pct*100)}%   `{reb_str}`\n"
                f"{_SEP}\n"
                f"*Please transfer* `${transfer:.2f}` *back to the main wallet*\n"
                f"Run: `/rebate {transfer:.2f}`"
            )
        else:
            return
        try:
            await notifier.send_raw(text + _lbl)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("auto-card send failed (%s): %s", kind, exc)

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

            # PAPER only: render the entry card from the simulated signal. In
            # live/demo the entry card is emitted by the executor's
            # on_entry_filled callback — i.e. ONLY after OKX confirms a real fill
            # — so we never announce an entry the exchange skipped/rejected.
            if notifier and notifier.enabled and not is_real:
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
                    f"Open fee  `${open_fee:.4f}` \\(maker\\)"
                )
                # Balance footer + bot label are appended inside the sender.
                asyncio.create_task(_send_entry_with_chart(text, side, entry, tp, sl, equity))
            if state_path is not None:
                try:
                    session.save_state(state_path)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("state save failed (entry): %s", exc)

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

            # PAPER only: render the exit card from the simulated touch. In
            # live/demo the ONE exit message is the executor's OKX-confirmed
            # close card (on_close_confirmed) — this SIM card is suppressed so
            # there are never two conflicting exit messages with different
            # numbers, and never a "closed" card while the real position is open.
            if notifier and notifier.enabled and not already_notified and not is_real:
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
                )
                entry_msg_id = _pending.get("msg_id")
                asyncio.create_task(_send_exit(text, entry_msg_id))
            _pending.clear()
            if state_path is not None:
                try:
                    session.save_state(state_path)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("state save failed (exit): %s", exc)

        elif evt.kind == "halt":
            LOGGER.warning("HALT: %s", evt.payload.get("reason"))
            if notifier and notifier.enabled:
                if is_real:
                    # Rebase on the REAL account: trigger is the strategy guard,
                    # numbers are the live wallet/volume.
                    asyncio.create_task(_send_auto_real("halt", evt.payload))
                else:
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
                if is_real:
                    asyncio.create_task(_send_auto_real("milestone", evt.payload))
                else:
                    _tg(
                        f"🎯 *Milestone {_esc(str(pct))}%* of volume target\n"
                        f"Vol  `${vol:,.0f}` / `${vol_target:,.0f}`"
                    )

        elif evt.kind == "rebate_reminder":
            p = evt.payload
            equity = float(p.get("equity", 0.0))
            available = float(p.get("available_rebate", 0.0))
            suggested = float(p.get("suggested_amount", available))
            dd_pct = float(p.get("drawdown_pct", 0.0)) * 100
            reasons = p.get("reasons", []) or []
            handles = p.get("mention_handles", []) or []
            reason_str = "; ".join(reasons) if reasons else "wallet low"
            LOGGER.warning(
                "REBATE REMINDER eq=$%.2f available=$%.2f suggested=$%.2f reason=%s",
                equity, available, suggested, reason_str,
            )
            if notifier and notifier.enabled:
                if is_real:
                    asyncio.create_task(_send_auto_real("rebate_reminder", evt.payload))
                    return
                mention_line = " ".join(_esc(h) for h in handles) if handles else ""
                head = mention_line + "\n" if mention_line else ""
                _tg(
                    f"{head}"
                    f"⚠️ *Rebate top\\-up needed*\n"
                    f"{_SEP}\n"
                    f"Wallet            `${equity:.2f}`\n"
                    f"Available rebate  `${available:.2f}`\n"
                    f"Drawdown          `{dd_pct:.1f}%`\n"
                    f"{_SEP}\n"
                    f"Please transfer  `${suggested:.2f}`  to main wallet\n"
                    f"Reason: {_esc(reason_str)}\n"
                    f"Run: `/rebate {suggested:.2f}`"
                )

    return on_event



async def _fetch_real_account(client: Any, symbol: str) -> Dict[str, Any]:
    """Best-effort snapshot of the REAL OKX (demo/live) account.

    Pulls USDT equity from /account/balance and the live open position from
    /account/positions so Telegram can quote the actual account instead of the
    local paper simulation. Never raises — returns Nones on failure so callers
    can fall back gracefully.
    """
    out: Dict[str, Any] = {
        "equity": None, "pos_side": None, "pos_sz": None, "upl": None,
        "avg_px": None, "mark_px": None, "upl_ratio": None,
    }
    try:
        bal = await client.get_balance("USDT")
        bd = (bal.get("data") or [{}])[0]
        for det in (bd.get("details") or []):
            if det.get("ccy") == "USDT":
                out["equity"] = float(det.get("eq") or 0.0)
                break
        if out["equity"] is None and bd.get("totalEq"):
            out["equity"] = float(bd["totalEq"])
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("real balance fetch failed: %s", exc)
    try:
        pos = await client.get_positions(symbol)
        for p in (pos.get("data") or []):
            sz = float(p.get("pos") or 0.0)
            if sz != 0.0:
                out["pos_side"] = p.get("posSide")
                out["pos_sz"] = sz
                out["upl"] = float(p.get("upl") or 0.0)
                out["avg_px"] = float(p.get("avgPx") or 0.0) or None
                out["mark_px"] = float(p.get("markPx") or 0.0) or None
                try:
                    out["upl_ratio"] = float(p.get("uplRatio") or 0.0)
                except (TypeError, ValueError):
                    out["upl_ratio"] = None
                break
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("real positions fetch failed: %s", exc)
    return out


def _real_acct_lines(acct: Dict[str, Any]) -> list:
    """Render the real-account snapshot as Telegram lines (already escaped-safe)."""
    lines = [_SEP]
    if acct.get("equity") is not None:
        lines.append(f"Wallet  `${acct['equity']:,.4f}`")
    else:
        lines.append("Wallet  `unavailable`")
    if acct.get("pos_side") and acct.get("pos_sz"):
        upl = acct.get("upl") or 0.0
        lines.append(
            f"Open pos  `{acct['pos_side']} {acct['pos_sz']:g} ct`  uPnL `{upl:+.4f}`"
        )
    else:
        lines.append("Open pos  `flat`")
    return lines


def _real_acct_block(acct: Optional[Dict[str, Any]]) -> str:
    """Trailing block (with separator) for status/position cards; '' if unavailable."""
    if not acct or acct.get("equity") is None:
        return ""
    block = f"Wallet  `${acct['equity']:,.4f}`\n"
    if acct.get("pos_side") and acct.get("pos_sz"):
        upl = acct.get("upl") or 0.0
        block += f"Open pos   `{acct['pos_side']} {acct['pos_sz']:g} ct`  uPnL `{upl:+.4f}`\n"
    else:
        block += "Open pos   `flat`\n"
    return block + f"{_SEP}\n"


def _read_live_trades(log_dir: pathlib.Path) -> list:
    """Read all REAL-money trade records from data/logs/live_trades/*.jsonl.

    Written by ``LiveVolumeExecutorOKX._log_trade`` — the authoritative record
    of real fills (real avgPx, exchange-reported fees, realized PnL). Returns
    filled records oldest→newest; tolerant of partial/garbage lines. Survives
    restarts because it reads the on-disk log, not in-memory state.

    Only ``live == True`` records count: OKX **demo** fills land in this same
    directory but carry ``live == False`` (or no flag, for pre-tag records), so
    the Telegram cards never mix demo/paper volume into real-money stats.
    """
    out: list = []
    trades_dir = log_dir / "live_trades"
    if not trades_dir.exists():
        return out
    for fp in sorted(trades_dir.glob("*.jsonl")):
        try:
            for line in fp.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("filled") and rec.get("live") is True:
                    out.append(rec)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("live_trades read failed for %s: %s", fp, exc)
    return out


def _real_trade_stats(records: list) -> Dict[str, Any]:
    """Aggregate REAL trade stats from live_trades records (filled only).

    Net PnL is real_gross − real_open_fee − real_close_fee per closed trade.
    Volume counts the entry leg for every filled trade plus the exit leg for
    every closed trade (≈ notional each), matching how the farmer counts volume.
    """
    closed = [r for r in records if r.get("closed") and r.get("close_px") is not None]
    n = len(closed)
    wins = losses = 0
    net_total = gross = open_fees = close_fees = volume = 0.0
    for r in records:
        notional = float(r.get("notional_usd") or 0.0)
        volume += notional  # entry leg
        # The open fee is real money already spent (or maker rebate already
        # accrued) at fill time — count it for every filled record, even one
        # not yet closed, so /fees doesn't under-report what was actually paid.
        of = float(r.get("real_open_fee") or 0.0)
        open_fees += of
        if r.get("closed") and r.get("close_px") is not None:
            volume += notional  # exit leg (≈ same notional)
            g = float(r.get("real_gross_pnl") or 0.0)
            cf = float(r.get("real_close_fee") or 0.0)
            net = g - of - cf
            net_total += net
            gross += g
            close_fees += cf
            if net > 0:
                wins += 1
            else:
                losses += 1
    wr = wins / n * 100.0 if n else 0.0
    return {
        "n": n, "wins": wins, "losses": losses, "wr": wr,
        "net": net_total, "gross": gross,
        "open_fees": open_fees, "close_fees": close_fees,
        "fees": open_fees + close_fees, "volume": volume,
        "closed": closed,
    }


async def _fetch_okx_trade_stats(
    client: Any, symbol: str, max_pages: int = 5,
) -> Optional[Dict[str, Any]]:
    """Build real-trade stats from OKX's OWN order history (the exchange truth).

    Merges /orders-history (7d) + /orders-history-archive (3mo), pairs each
    opening fill (pnl==0) with its closing fill (pnl!=0) into a round-trip,
    synthesizes a record in the live_trades schema, and runs it through
    ``_real_trade_stats`` so the Telegram cards render identically — but now
    sourced from the exchange, immune to any local-logging drift. Returns None
    on any API error so the caller can fall back to the local jsonl.
    """
    inst = to_okx_inst_id(symbol)
    ct_val = 0.01  # BTC-USDT-SWAP: 1 contract = 0.01 BTC
    merged: Dict[str, Any] = {}
    ok_any = False  # did ANY page return cleanly? if not, fall back to jsonl
    try:
        for path in ("/api/v5/trade/orders-history",
                     "/api/v5/trade/orders-history-archive"):
            after = ""
            for _ in range(max_pages):
                params = {"instType": "SWAP", "instId": inst, "limit": "100"}
                if after:
                    params["after"] = after
                r = await client._request("GET", path, params=params, auth=True)  # noqa: SLF001
                if str(r.get("code")) != "0":
                    LOGGER.warning("OKX order-history code=%s msg=%r",
                                   r.get("code"), r.get("msg"))
                    break
                ok_any = True
                data = r.get("data") or []
                if not data:
                    break
                for o in data:
                    merged[o.get("ordId")] = o
                after = data[-1].get("ordId") or ""
                if not after or len(data) < 100:
                    break
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("OKX order-history fetch failed: %s", exc)
        return None
    if not ok_any:
        # every endpoint errored (busy/auth/etc.) — don't report a false "0 trades"
        return None

    orders = [o for o in merged.values() if o.get("state") == "filled"]
    orders.sort(key=lambda o: int(o.get("cTime") or 0))

    opens: list = []
    records: list = []
    for o in orders:
        try:
            pnl = float(o.get("pnl") or 0.0)
            sz = float(o.get("accFillSz") or 0.0)
            px = float(o.get("avgPx") or 0.0)
            fee = float(o.get("fee") or 0.0)        # negative when paid
        except (TypeError, ValueError):
            continue
        close_notional = sz * ct_val * px
        if pnl == 0.0:
            opens.append({"side": o.get("side"), "fee": fee, "notional": close_notional})
            continue
        op = opens.pop() if opens else {
            "side": ("sell" if o.get("side") == "buy" else "buy"),
            "fee": 0.0, "notional": close_notional,
        }
        open_fee = -float(op.get("fee") or 0.0)   # fee<=0 -> positive amount paid
        close_fee = -fee
        net = pnl - open_fee - close_fee
        records.append({
            "filled": True, "closed": True, "live": True, "close_px": px,
            "side": "LONG" if op.get("side") == "buy" else "SHORT",
            "close_reason": "tp" if net > 0 else "sl",   # icon reflects NET win/loss
            # _real_trade_stats counts notional×2 (entry+exit); use the mean of the
            # two legs so the doubled total equals open+close notional exactly.
            "notional_usd": ((op.get("notional") or close_notional) + close_notional) / 2.0,
            "real_gross_pnl": pnl,
            "real_open_fee": open_fee,
            "real_close_fee": close_fee,
        })
    return _real_trade_stats(records)


def _make_entry_filled_callback(
    notifier: Optional[TelegramNotifier],
    symbol: str,
    mode_label: str,
    bot_label: str,
    tf: str,
    client: Any = None,
):
    """Build the executor's on_entry_filled callback — the REAL entry card.

    Fires only after OKX confirms a fill (real avgPx + exchange fee). This is
    the sole entry announcement in live/demo; the simulation's entry card is
    suppressed. It also writes ``entry_msg_id`` onto the trade so the close
    card threads under it (the simulated card used to be the only writer of
    that id — suppressing it here would otherwise break threading).
    """
    _esc = TelegramNotifier.escape
    _lbl = f"\n{_SEP}\n\\[{_esc(bot_label)}\\]" if bot_label else ""

    async def _on_entry_filled(trade) -> None:  # type: LiveTradeOKX
        if notifier is None or not notifier.enabled:
            return
        side = str(trade.side)
        arrow = "🟢" if side == "long" else "🔴"
        fill_px = float(trade.fill_px)
        notional = float(trade.notional_usd)
        open_fee = float(trade.real_open_fee)
        tp = float(trade.tp_px)
        sl = float(trade.sl_px)
        tp_bps = float(trade.tp_bps)
        sl_bps = float(trade.sl_bps)
        sz = trade.sz_contracts
        ordid = str(trade.ord_id or "—")
        text = (
            f"{arrow} *{_esc(side.upper())} FILLED · {_esc(mode_label)}*\n"
            f"{_esc(symbol)} · {_esc(tf)}\n"
            f"{_SEP}\n"
            f"Entry →  `{fill_px:,.2f}`  \\(real fill\\)\n"
            f"TP    →  `{tp:,.2f}`   `+{tp_bps:.1f}bps`\n"
            f"SL    →  `{sl:,.2f}`   `-{sl_bps:.1f}bps`\n"
            f"{_SEP}\n"
            f"Size    `${notional:,.2f}`  ·  `{sz:g}` ct\n"
            f"Open fee  `${open_fee:.4f}`\n"
            f"{_SEP}\n"
            f"`ordId {ordid}`"
        )
        if client is not None:
            try:
                acct = await _fetch_real_account(client, symbol)
                text = text + "\n" + "\n".join(_real_acct_lines(acct))
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("entry card acct fetch failed: %s", exc)
        text = text + _lbl
        chart: Optional[bytes] = None
        if client is not None:
            try:
                raw = await client.get_candles(symbol, tf, limit=35)
                bars = _normalize_okx_candles(raw)
                chart = _draw_entry_chart(
                    bars[bars["closed"]], side, fill_px, tp, sl, symbol=symbol, tf=tf,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("entry chart generation failed: %s", exc)
        try:
            if chart:
                msg_id = await notifier.send_photo(chart, caption=text)
            else:
                msg_id = await notifier.send_and_get_id(text)
            if msg_id is not None:
                trade.extras["entry_msg_id"] = msg_id
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("entry_filled send failed: %s", exc)

    return _on_entry_filled


def _make_close_unverified_callback(
    notifier: Optional[TelegramNotifier],
    symbol: str,
    mode_label: str,
    bot_label: str,
    client: Any = None,
):
    """Build the executor's on_close_unverified callback — the honest alert.

    Fired when the position is flat on OKX but the close couldn't be priced, OR
    when a close attempt failed and OKX still shows the position open. Never a
    "REAL FILL" card with fabricated zeros — it tells the operator to check OKX.
    """
    _esc = TelegramNotifier.escape
    _lbl = f"\n{_SEP}\n\\[{_esc(bot_label)}\\]" if bot_label else ""

    async def _on_close_unverified(trade) -> None:  # type: LiveTradeOKX
        if notifier is None or not notifier.enabled:
            return
        extras = trade.extras or {}
        still_open = bool(extras.get("still_open"))
        reason = trade.close_reason or "unverified"
        if still_open:
            head = "⚠️ *EXIT FAILED · position may still be OPEN*"
            detail = _esc(
                "The bot tried to close but OKX still shows a position. "
                "New entries are paused until it reads flat (auto-resumes — no "
                "restart needed). Close it yourself from here:"
            )
        else:
            head = "⚠️ *CLOSED · PnL unverified*"
            detail = _esc(
                "Position is flat on OKX but the closing fill could not be "
                "priced from order history. Check the OKX panel for the exact PnL."
            )
        lines = [
            head,
            f"{_esc(trade.side.upper())} · {_esc(symbol)} · {_esc(mode_label)}",
            _SEP,
            f"Reason  `{reason}`",
            f"Entry   `{trade.fill_px:,.2f}`",
            _SEP,
            detail,
        ]
        if client is not None:
            try:
                acct = await _fetch_real_account(client, symbol)
                lines.extend(_real_acct_lines(acct))
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("unverified card acct fetch failed: %s", exc)
        # When OKX still shows the position open, give the operator the exact
        # commands (and tap-buttons) to inspect and flatten it from Telegram.
        if still_open:
            lines += [
                _SEP,
                f"`/position` — {_esc('see the live position')}",
                f"`/orders` — {_esc('see resting orders')}",
                f"`/close` — {_esc('market or maker-limit close')}",
            ]
        body = "\n".join(lines) + _lbl
        entry_msg_id = extras.get("entry_msg_id")
        try:
            if still_open:
                await notifier.send_with_buttons(
                    body, _STUCK_BUTTONS, reply_to_message_id=entry_msg_id,
                )
            else:
                await notifier.send_and_get_id(body, reply_to_message_id=entry_msg_id)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("close_unverified send failed: %s", exc)

    return _on_close_unverified


def _make_real_fill_callback(
    notifier: Optional[TelegramNotifier],
    symbol: str,
    mode_label: str,
    bot_label: str,
    client: Any = None,
):
    """Build the executor's on_close_confirmed callback — the REAL close card.

    Emits the single exit message once OKX confirms the position is flat and the
    close is priced, threaded under the entry-filled card. Every number comes
    from real fills (avgPx, exchange-reported fees, OKX realized pnl). In live
    the simulation's exit card is suppressed, so this is the only exit notice.
    """
    _esc = TelegramNotifier.escape
    _lbl = f"\n{_SEP}\n\\[{_esc(bot_label)}\\]" if bot_label else ""

    async def _on_real_fill(trade) -> None:  # type: LiveTradeOKX
        if notifier is None or not notifier.enabled:
            return
        reason = trade.close_reason or "closed"
        is_manual = reason == "manual"
        emoji = {"tp": "✅", "sl": "❌", "time_stop": "⏱", "manual": "📤"}.get(reason, "📤")
        title = "MANUAL CLOSE" if is_manual else f"REAL FILL · {_esc(reason.upper())}"
        gross = float(trade.real_gross_pnl)
        open_fee = float(trade.real_open_fee)
        close_fee = float(trade.real_close_fee)
        net = gross - open_fee - close_fee
        g_str = f"+${abs(gross):.4f}" if gross >= 0 else f"-${abs(gross):.4f}"
        n_str = f"+${abs(net):.4f}" if net >= 0 else f"-${abs(net):.4f}"
        # Per-side bps slip vs the paper TP/SL (positive = we got a worse fill).
        # Meaningless for an operator close (no intended target), so skip it.
        intended_close = trade.tp_px if reason == "tp" else trade.sl_px
        if intended_close > 0 and trade.close_px > 0:
            if trade.side == "long":
                slip = (intended_close - trade.close_px) / intended_close * 10_000
            else:
                slip = (trade.close_px - intended_close) / intended_close * 10_000
        else:
            slip = 0.0
        lines = [
            f"{emoji} *{title} · "
            f"{_esc(trade.side.upper())} · {_esc(mode_label)}*",
            f"{_esc(symbol)}  `{trade.fill_px:,.2f}` → `{trade.close_px:,.2f}`",
            _SEP,
            f"Gross   `{g_str}`",
            f"Open fee  `${open_fee:.4f}`",
            f"Close fee `${close_fee:.4f}`",
            f"Net     `{n_str}`",
        ]
        if not is_manual:
            lines += [_SEP, f"Slip vs paper  `{slip:+.1f}bps`"]
        if trade.exit_path and trade.exit_path != "native_tp_sl":
            lines.append(
                f"Exit    `{_esc(trade.exit_path)}` · "
                f"repegs `{trade.exit_repegs}` · "
                f"ttf `{trade.exit_ttf_s:.1f}s`"
            )
            if trade.exit_adverse_bps:
                lines.append(f"Adverse `{trade.exit_adverse_bps:.1f}bps`")
        if client is not None:
            acct = await _fetch_real_account(client, symbol)
            lines.extend(_real_acct_lines(acct))
        body = "\n".join(lines) + _lbl
        entry_msg_id = trade.extras.get("entry_msg_id") if trade.extras else None
        try:
            await notifier.send_and_get_id(body, reply_to_message_id=entry_msg_id)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("real_fill send failed: %s", exc)

    return _on_real_fill


def _fmt_uptime(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h {m}m" if h else f"{m}m"


def _build_status_text(
    session: "VolumeFarmerSession",
    cfg: Dict[str, Any],
    mode_label: str,
    bot_label: str,
    start_time: datetime,
    real_acct: Optional[Dict[str, Any]] = None,
    real_stats: Optional[Dict[str, Any]] = None,
) -> str:
    _esc = TelegramNotifier.escape
    uptime = _fmt_uptime((datetime.now(tz=timezone.utc) - start_time).total_seconds())
    sym = cfg["exchange"]["symbol"]
    tf = cfg["exchange"]["timeframe"]
    status_str = "Halted" if session.halted else "Running"
    is_live = mode_label != "PAPER"

    # LIVE/DEMO: source everything from the REAL account + real-fill trade log,
    # NOT the local paper simulation.
    if is_live and real_acct is not None and real_stats is not None:
        rs = real_stats
        eq = real_acct.get("equity")
        eq_str = f"${eq:,.4f}" if eq is not None else "unavailable"
        upl = real_acct.get("upl") or 0.0
        realized = rs["net"]
        r_sign = "+" if realized >= 0 else "-"
        u_sign = "+" if upl >= 0 else "-"
        if real_acct.get("pos_side") and real_acct.get("pos_sz"):
            pos_line = (
                f"`{_esc(str(real_acct['pos_side']))} {real_acct['pos_sz']:g} ct`"
                f"  uPnL `{u_sign}${abs(upl):.4f}`"
            )
        else:
            pos_line = "`flat`"
        return (
            f"📊 *Status · {_esc(mode_label)}*\n"
            f"{_SEP}\n"
            f"Mode      `{_esc(mode_label)}`  ·  `{_esc(sym)}` `{_esc(tf)}`\n"
            f"Uptime    `{_esc(uptime)}`\n"
            f"{_SEP}\n"
            f"Wallet    `{eq_str}`\n"
            f"Position  {pos_line}\n"
            f"{_SEP}\n"
            f"Realized  `{r_sign}${abs(realized):.4f}`  \\(real fills\\)\n"
            f"Trades    `{rs['n']}`  ·  WR `{rs['wr']:.1f}%`\n"
            f"W / L     `{rs['wins']}` / `{rs['losses']}`\n"
            f"{_SEP}\n"
            f"{'🛑' if session.halted else '✅'} `{_esc(status_str)}`\n"
            f"{_SEP}\n"
            f"\\[{_esc(bot_label)}\\]"
        )

    # PAPER: simulated session view (unchanged).
    eq = session.equity
    start_eq = session.start_equity
    peak = session.peak_equity
    dd_pct = (peak - eq) / peak * 100 if peak > 0 else 0.0
    pnl = session.total_pnl
    pnl_pct = pnl / start_eq * 100 if start_eq > 0 else 0.0
    wr = session.wins / session.round_trips * 100 if session.round_trips else 0.0
    pos_str = f"{session.position.side.upper()} open" if session.position else "flat"
    pnl_sign = "+" if pnl >= 0 else ""
    return (
        f"📊 *Status*\n"
        f"{_SEP}\n"
        f"Mode      `{_esc(mode_label)}`  ·  `{_esc(sym)}` `{_esc(tf)}`\n"
        f"Uptime    `{_esc(uptime)}`\n"
        f"{_SEP}\n"
        f"Equity    `${eq:.4f}`  \\(start `${start_eq:.2f}`\\)\n"
        f"Peak      `${peak:.4f}`\n"
        f"Drawdown  `{dd_pct:.2f}%`\n"
        f"PnL       `{pnl_sign}${abs(pnl):.4f}`  \\(`{pnl_sign}{pnl_pct:.2f}%`\\)\n"
        f"{_SEP}\n"
        f"Trades    `{session.round_trips}`  ·  WR `{wr:.1f}%`\n"
        f"W / L     `{session.wins}` / `{session.losses}`\n"
        f"Position  `{_esc(pos_str)}`\n"
        f"{_SEP}\n"
        f"{'🛑' if session.halted else '✅'} `{_esc(status_str)}`\n"
        f"{_SEP}\n"
        f"{_real_acct_block(real_acct)}"
        f"\\[{_esc(bot_label)}\\]"
    )


async def _build_position_text(
    session: "VolumeFarmerSession",
    client: Any,
    symbol: str,
    tf: str,
    bot_label: str,
    mode_label: str = "PAPER",
    real_acct: Optional[Dict[str, Any]] = None,
    live_executor: Optional[Any] = None,
) -> str:
    _esc = TelegramNotifier.escape
    is_live = mode_label != "PAPER"

    # LIVE/DEMO: show the REAL OKX position, not the paper simulation.
    if is_live:
        acct = real_acct if real_acct is not None else await _fetch_real_account(client, symbol)
        if not (acct.get("pos_side") and acct.get("pos_sz")):
            return (
                f"📈 *Position · {_esc(mode_label)}*\n"
                f"{_SEP}\n"
                f"No open position\n"
                f"{_SEP}\n"
                f"\\[{_esc(bot_label)}\\]"
            )
        side = str(acct.get("pos_side") or "")
        sz = float(acct.get("pos_sz") or 0.0)
        avg = acct.get("avg_px")
        current = acct.get("mark_px")
        if current is None:
            try:
                raw = await client.get_candles(symbol, tf, limit=2)
                df_cur = _normalize_okx_candles(raw)
                current = float(df_cur["close"].iloc[-1])
            except Exception:  # noqa: BLE001
                current = avg
        upl = acct.get("upl") or 0.0
        upl_ratio = acct.get("upl_ratio")
        upl_pct = (upl_ratio * 100.0) if upl_ratio is not None else 0.0
        u_sign = "+" if upl >= 0 else "-"
        arrow = "🟢" if side.lower() == "long" else "🔴"
        tp = sl = None
        ot = getattr(live_executor, "_open_trade", None) if live_executor is not None else None
        if ot is not None and not getattr(ot, "closed", True):
            tp = getattr(ot, "tp_px", None) or None
            sl = getattr(ot, "sl_px", None) or None
        lines = [
            f"📈 *Position · {_esc(mode_label)}*",
            _SEP,
            f"{arrow} `{_esc(side.upper())}`  ·  `{_esc(symbol)}`",
            f"Size     `{sz:g} ct`",
            (f"Entry    `{avg:,.2f}`" if avg else "Entry    `—`"),
            (f"Mark     `{current:,.2f}`" if current else "Mark     `—`"),
        ]
        if tp and sl and current:
            dist_tp = abs(tp - current) / current * 10_000
            dist_sl = abs(sl - current) / current * 10_000
            lines += [
                _SEP,
                f"TP  `{tp:,.2f}`  \\({dist_tp:.1f}bps away\\)",
                f"SL  `{sl:,.2f}`  \\({dist_sl:.1f}bps away\\)",
            ]
        lines += [
            _SEP,
            f"uPnL  `{u_sign}${abs(upl):.4f}`  \\(`{u_sign}{abs(upl_pct):.2f}%`\\)",
            _SEP,
            f"\\[{_esc(bot_label)}\\]",
        ]
        return "\n".join(lines)

    # PAPER: simulated position (unchanged).
    if session.position is None:
        return (
            f"📈 *Position*\n"
            f"{_SEP}\n"
            f"No open position\n"
            f"{_SEP}\n"
            f"\\[{_esc(bot_label)}\\]"
        )
    pos = session.position
    current_price = pos.entry_price
    try:
        raw = await client.get_candles(symbol, tf, limit=2)
        df_cur = _normalize_okx_candles(raw)
        current_price = float(df_cur["close"].iloc[-1])
    except Exception:
        pass
    qty = pos.notional / pos.entry_price
    if pos.side == "long":
        upnl = (current_price - pos.entry_price) * qty
    else:
        upnl = (pos.entry_price - current_price) * qty
    upnl_pct = upnl / session.equity * 100 if session.equity > 0 else 0.0
    upnl_sign = "+" if upnl >= 0 else ""
    dist_tp = abs(pos.tp - current_price) / current_price * 10_000
    dist_sl = abs(pos.sl - current_price) / current_price * 10_000
    arrow = "🟢" if pos.side == "long" else "🔴"
    return (
        f"📈 *Position*\n"
        f"{_SEP}\n"
        f"{arrow} `{_esc(pos.side.upper())}`  ·  `{_esc(symbol)}`\n"
        f"Entry    `{pos.entry_price:,.2f}`\n"
        f"Current  `{current_price:,.2f}`\n"
        f"{_SEP}\n"
        f"TP  `{pos.tp:,.2f}`  \\({dist_tp:.1f}bps away\\)\n"
        f"SL  `{pos.sl:,.2f}`  \\({dist_sl:.1f}bps away\\)\n"
        f"Bars held  `{pos.bars_held}`\n"
        f"{_SEP}\n"
        f"Notional  `${pos.notional:,.2f}`\n"
        f"uPnL      `{upnl_sign}${abs(upnl):.4f}`  \\(`{upnl_sign}{upnl_pct:.2f}%`\\)\n"
        f"{_SEP}\n"
        f"\\[{_esc(bot_label)}\\]"
    )


def _build_fees_text(
    session: "VolumeFarmerSession",
    cfg: Dict[str, Any],
    bot_label: str,
    real_stats: Optional[Dict[str, Any]] = None,
    real_acct: Optional[Dict[str, Any]] = None,
    mode_label: str = "PAPER",
) -> str:
    _esc = TelegramNotifier.escape
    fees_cfg = cfg.get("fees", {})
    rebate_pct = float(fees_cfg.get("rebate_pct", 0.40))
    is_live = mode_label != "PAPER"

    # LIVE/DEMO: real volume + exchange-reported fees from the real-fill log.
    if is_live and real_stats is not None:
        rs = real_stats
        vol = rs["volume"]
        vol_target = float(cfg.get("target", {}).get("volume_usd", 5_000_000))
        vol_pct = vol / vol_target * 100 if vol_target else 0.0
        gross_fees = rs["fees"]
        rebate_est = gross_fees * rebate_pct
        maker_bps = float(fees_cfg.get("maker", 0.0002)) * 10_000
        taker_bps = float(fees_cfg.get("taker", 0.0005)) * 10_000
        bal = (real_acct or {}).get("equity")
        bal_str = f"${bal:,.4f}" if bal is not None else "unavailable"
        return (
            f"💰 *Fees & Rebate · {_esc(mode_label)}*\n"
            f"{_SEP}\n"
            f"Volume    `${vol:,.0f}`\n"
            f"Target    `${vol_target:,.0f}`  \\(`{vol_pct:.1f}%` done\\)\n"
            f"{_SEP}\n"
            f"Real fees       `${gross_fees:.4f}`  \\(open\\+close\\)\n"
            f"Est rebate {rebate_pct*100:.0f}%  `${rebate_est:.4f}`  \\(est, not moved\\)\n"
            f"{_SEP}\n"
            f"Wallet    `{bal_str}`\n"
            f"{_SEP}\n"
            f"Maker `{maker_bps:.0f}bps`  ·  Taker `{taker_bps:.0f}bps`\n"
            f"Trades  `{rs['n']}`\n"
            f"{_SEP}\n"
            f"\\[{_esc(bot_label)}\\]"
        )

    # PAPER: simulated session view (unchanged).
    rebate_accrued = session.total_rebate_accrued
    rebate_transferred = session.total_rebate_transferred
    rebate_available = session.available_rebate
    vol_target = float(cfg.get("target", {}).get("volume_usd", 5_000_000))
    vol_pct = session.total_volume_usd / vol_target * 100 if vol_target else 0.0
    maker_bps = float(fees_cfg.get("maker", 0.0002)) * 10_000
    taker_bps = float(fees_cfg.get("taker", 0.0005)) * 10_000
    return (
        f"💰 *Fees & Rebate*\n"
        f"{_SEP}\n"
        f"Volume    `${session.total_volume_usd:,.0f}`\n"
        f"Target    `${vol_target:,.0f}`  \\(`{vol_pct:.1f}%` done\\)\n"
        f"{_SEP}\n"
        f"Gross fees       `${session.total_fees_gross:.4f}`\n"
        f"Rebate {rebate_pct*100:.0f}% total `${rebate_accrued:.4f}`\n"
        f"Transferred      `${rebate_transferred:.4f}`\n"
        f"Available rebate `${rebate_available:.4f}`\n"
        f"{_SEP}\n"
        f"Balance   `${session.equity:.4f}`\n"
        f"{_SEP}\n"
        f"Maker `{maker_bps:.0f}bps`  ·  Taker `{taker_bps:.0f}bps`\n"
        f"Trades  `{session.round_trips}`\n"
        f"{_SEP}\n"
        f"Tip: `/rebate <amount>` tops up your balance\n"
        f"{_SEP}\n"
        f"\\[{_esc(bot_label)}\\]"
    )


def _build_trades_text(
    session: "VolumeFarmerSession",
    bot_label: str,
    real_stats: Optional[Dict[str, Any]] = None,
    mode_label: str = "PAPER",
) -> str:
    _esc = TelegramNotifier.escape
    is_live = mode_label != "PAPER"

    # LIVE/DEMO: recent REAL closed trades from the real-fill log.
    if is_live and real_stats is not None:
        rs = real_stats
        closed = rs["closed"]
        total = len(closed)
        header = f"📋 *Trades · {_esc(mode_label)}*  `{total}` total\n{_SEP}"
        lines = [header]
        if total == 0:
            lines.append("No closed trades yet")
        else:
            # Show ALL trades, newest first, bounded by Telegram's 4096-char
            # limit (headroom kept for the footer); truncate oldest if it overflows.
            used = len(header)
            shown = 0
            for i, t in enumerate(reversed(closed)):
                num = total - i
                reason = str(t.get("close_reason") or "").lower()
                icon = (
                    "✅" if reason == "tp"
                    else "❌" if reason == "sl"
                    else "📤" if reason == "manual"
                    else "⏱"
                )
                net = (
                    float(t.get("real_gross_pnl") or 0.0)
                    - float(t.get("real_open_fee") or 0.0)
                    - float(t.get("real_close_fee") or 0.0)
                )
                sign = "+" if net >= 0 else "-"
                side_ch = "L" if str(t.get("side") or "").upper().startswith("L") else "S"
                line = f"\\#{_esc(str(num))}  {icon} {_esc(side_ch)}  `{sign}${abs(net):.4f}`"
                if used + len(line) + 1 > 3500:
                    lines.append(f"… {total - shown} older not shown \\(see CSV\\)")
                    break
                lines.append(line)
                used += len(line) + 1
                shown += 1
        lines.append(_SEP)
        tsign = "+" if rs["net"] >= 0 else "-"
        lines.append(
            f"Total  `{tsign}${abs(rs['net']):.4f}`  ·  WR `{rs['wr']:.1f}%`"
        )
        lines.append(f"{_SEP}\n\\[{_esc(bot_label)}\\]")
        return "\n".join(lines)

    # PAPER: simulated ledger (unchanged).
    recent = list(reversed(session.ledger[-8:])) if session.ledger else []
    lines: list = [f"📋 *Recent Trades*\n{_SEP}"]
    if not recent:
        lines.append("No trades yet")
    else:
        base_num = session.round_trips
        for i, t in enumerate(recent):
            num = base_num - i
            icon = "✅" if t["reason"] == "tp" else "❌"
            net = t["net_pnl"]
            sign = "+" if net >= 0 else ""
            side_ch = "L" if t["side"] == "long" else "S"
            lines.append(
                f"\\#{_esc(str(num))}  {icon} {_esc(side_ch)}  `{sign}${abs(net):.4f}`"
            )
    lines.append(_SEP)
    total_net = sum(t["net_pnl"] for t in session.ledger)
    sign = "+" if total_net >= 0 else ""
    wr = session.wins / session.round_trips * 100 if session.round_trips else 0.0
    lines.append(
        f"Total  `{sign}${abs(total_net):.4f}`  ·  WR `{wr:.1f}%`"
    )
    lines.append(f"{_SEP}\n\\[{_esc(bot_label)}\\]")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Manual position management (live/demo): see the REAL position & resting orders,
# and close it from Telegram via a market or maker-limit order.
# ----------------------------------------------------------------------------
async def _fetch_position_raw(client: Any, symbol: str) -> Optional[Dict[str, Any]]:
    """Return the raw /positions item for the open position (None if flat)."""
    resp = await client.get_positions(symbol)
    for p in resp.get("data") or []:
        try:
            if float(p.get("pos") or 0.0) != 0.0:
                return p
        except (TypeError, ValueError):
            continue
    return None


def _close_direction(p: Dict[str, Any], pos_mode: str) -> Dict[str, Any]:
    """Derive close-order parameters from a raw /positions item.

    OKX reports ``pos`` in CONTRACTS (signed in net mode; positive with a
    ``posSide`` in hedge mode). To flatten we send the OPPOSITE side, reduceOnly.
    ``posSide`` is only valid in hedge mode — in net (one-way) mode OKX rejects
    any posSide, so it must be omitted there.
    """
    try:
        raw = float(p.get("pos") or 0.0)
    except (TypeError, ValueError):
        raw = 0.0
    ps = str(p.get("posSide") or "").lower()
    if ps in ("long", "short"):
        is_long = ps == "long"
    else:
        is_long = raw > 0
    close_side = "sell" if is_long else "buy"
    order_pos_side = ps if (pos_mode == "hedge" and ps in ("long", "short")) else None
    return {
        "is_long": is_long,
        "close_side": close_side,
        "order_pos_side": order_pos_side,
        "size": abs(raw),
    }


async def _fetch_open_orders(client: Any, symbol: str) -> Dict[str, Any]:
    """Resting limit orders + algo (TP/SL) orders for the instrument. Best-effort."""
    out: Dict[str, Any] = {"orders": [], "algos": []}
    try:
        r = await client.get_pending_orders(symbol)
        out["orders"] = list(r.get("data") or [])
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("pending orders fetch failed: %s", exc)
    try:
        r = await client.get_pending_algos(symbol)
        out["algos"] = list(r.get("data") or [])
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("pending algos fetch failed: %s", exc)
    return out


def _build_orders_text(
    open_orders: Dict[str, Any], symbol: str, bot_label: str, mode_label: str,
) -> str:
    """Render resting limit + algo orders (the 'what's going on' view)."""
    _esc = TelegramNotifier.escape
    ords = open_orders.get("orders") or []
    algos = open_orders.get("algos") or []
    lines = [f"📋 *Open orders · {_esc(mode_label)}*", _SEP]
    if not ords and not algos:
        lines.append("No resting orders on OKX")
    if ords:
        lines.append("*Limit / pending*")
        for o in ords:
            side = str(o.get("side") or "?").upper()[:1]
            sz = o.get("sz") or "-"
            px = o.get("px") or "-"
            otype = str(o.get("ordType") or "?")
            ro = " ·ro" if str(o.get("reduceOnly")).lower() == "true" else ""
            oid = str(o.get("ordId") or "-")
            lines.append(f"`{side} {sz}ct @ {px}`  `{otype}{ro}`")
            lines.append(f"   id `{oid}`")
    if algos:
        lines.append("*Algo TP / SL*")
        for a in algos:
            tp = a.get("tpTriggerPx") or "-"
            sl = a.get("slTriggerPx") or "-"
            aid = str(a.get("algoId") or "-")
            lines.append(f"`TP {tp}  SL {sl}`")
            lines.append(f"   id `{aid}`")
    lines += [
        _SEP,
        f"Cancel limit orders  `/cancel`",
        f"Close position       `/close`",
        _SEP,
        f"\\[{_esc(bot_label)}\\]",
    ]
    return "\n".join(lines)


def _build_help_text(mode_label: str, bot_label: str) -> str:
    """Command guide. Manage-position commands are shown only in live/demo."""
    _esc = TelegramNotifier.escape
    is_live = mode_label != "PAPER"
    lines = [
        f"🆘 *Commands · {_esc(mode_label)}*",
        _SEP,
        "*Look*",
        f"`/status` — {_esc('wallet, PnL, win-rate')}",
        f"`/position` — {_esc('live OKX position + TP/SL')}",
        f"`/orders` — {_esc('resting limit & TP/SL orders')}",
        f"`/fees` — {_esc('volume, fees, rebate')}",
        f"`/trades` — {_esc('recent real fills')}",
    ]
    if is_live:
        lines += [
            _SEP,
            "*Manage the position*",
            f"`/close` — {_esc('show position + close buttons')}",
            f"`/close market` — {_esc('close NOW at market (taker fee)')}",
            f"`/close limit <price>` — {_esc('maker limit close at that price')}",
            f"`/close limit` — {_esc('maker limit at the current touch')}",
            f"`/cancel` — {_esc('cancel resting limit orders (keeps your TP/SL)')}",
            _SEP,
            _esc("When you see 'EXIT FAILED — still OPEN': run /position to see it, "
                 "then /close to flatten it. Entries resume once OKX reads flat."),
        ]
    else:
        lines += [_SEP, f"`/rebate <amount>` — {_esc('top up wallet from rebate')}"]
    lines += [_SEP, f"\\[{_esc(bot_label)}\\]"]
    return "\n".join(lines)


async def _build_close_confirm_text(
    client: Any, symbol: str, tf: str, pos_mode: str,
    bot_label: str, mode_label: str, live_executor: Optional[Any] = None,
) -> str:
    """Position summary + the market-vs-limit explainer, shown above close buttons."""
    _esc = TelegramNotifier.escape
    p = await _fetch_position_raw(client, symbol)
    if p is None:
        return (
            f"📈 *Close · {_esc(mode_label)}*\n{_SEP}\n"
            f"No open position — nothing to close\\.\n{_SEP}\n"
            f"\\[{_esc(bot_label)}\\]"
        )
    d = _close_direction(p, pos_mode)
    side_lbl = "LONG" if d["is_long"] else "SHORT"
    avg = p.get("avgPx") or "-"
    mark = p.get("markPx") or "-"
    try:
        upl_f = float(p.get("upl") or 0.0)
    except (TypeError, ValueError):
        upl_f = 0.0
    u_sign = "+" if upl_f >= 0 else "-"
    lot_sz = float(getattr(live_executor, "_lot_sz", 0.01)) if live_executor is not None else 0.01
    sz_str = _fmt_size(d["size"], lot_sz)
    return (
        f"🛑 *Confirm close · {_esc(mode_label)}*\n{_SEP}\n"
        f"`{_esc(side_lbl)}`  ·  `{_esc(symbol)}`\n"
        f"Size   `{sz_str} ct`\n"
        f"Entry  `{avg}`   Mark `{mark}`\n"
        f"uPnL   `{u_sign}${abs(upl_f):.4f}`\n{_SEP}\n"
        f"{_esc('Market = instant, taker fee. Limit = maker @ touch, waits for price.')}\n{_SEP}\n"
        f"\\[{_esc(bot_label)}\\]"
    )


async def _quiet_cancel_algos(client: Any, symbol: str) -> None:
    """Best-effort cancel of any standalone resting algos (no message)."""
    try:
        r = await client.get_pending_algos(symbol)
        ids = [a.get("algoId") for a in (r.get("data") or []) if a.get("algoId")]
        if ids:
            await client.cancel_algos(symbol, ids)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("quiet algo cancel failed: %s", exc)


def _code(value: Any) -> str:
    """Sanitize arbitrary text for placement INSIDE a MarkdownV2 backtick span.

    Inside a code span only backtick and backslash are special, so we strip
    backslashes and swap backticks for apostrophes — leaving the message
    verbatim and readable (no stray escape slashes from the full _esc set).
    """
    return str(value)[:160].replace("\\", "").replace("`", "'")


async def _sweep_pending_limit_orders(client: Any, symbol: str) -> int:
    """Cancel resting LIMIT/pending orders (NOT algos). Returns count cancelled.

    Used before a manual maker-limit close so a leftover reduceOnly order from
    the executor's maker-exit (or a prior /close limit) can't stack a second
    full-size close order on the book. Protective TP/SL algos are left intact.
    """
    n = 0
    try:
        r = await client.get_pending_orders(symbol)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("sweep pending orders fetch failed: %s", exc)
        return 0
    for o in (r.get("data") or []):
        oid = o.get("ordId")
        if not oid:
            continue
        try:
            await client.cancel_order(symbol, ord_id=oid)
            n += 1
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("sweep cancel ordId=%s failed: %s", oid, exc)
    return n


def _mark_manual_close(live_executor: Optional[Any]) -> Optional[Any]:
    """Tag the executor's open trade as an operator close (if one is live).

    Returns the open trade (so the caller can release the gate after a confirmed
    flat) or None. Setting ``close_reason='manual'`` BEFORE the close order is
    sent means the executor's _resolve_close keeps "manual" instead of
    mislabeling the operator flatten as a strategy TP/SL exit.
    """
    ot = getattr(live_executor, "_open_trade", None) if live_executor is not None else None
    if ot is not None and not getattr(ot, "closed", True):
        try:
            ot.close_reason = "manual"
            ot.extras["manual_close"] = True
        except Exception:  # noqa: BLE001
            return ot
    return ot


def _release_gate_if_match(live_executor: Optional[Any], trade: Optional[Any]) -> None:
    """Proactively release the entry gate after a confirmed manual flat.

    Only clears the gate if it still points at the trade we closed and that
    trade isn't flagged still-open (in which case the executor's own recovery
    watcher owns the release). Safe no-op when nothing matches.
    """
    if live_executor is None or trade is None:
        return
    try:
        if getattr(live_executor, "_open_trade", None) is trade and not trade.extras.get("still_open"):
            live_executor._open_trade = None  # noqa: SLF001
    except Exception:  # noqa: BLE001
        pass


class _MaybeLock:
    """async-with helper: use the executor's close lock if present, else a no-op."""

    def __init__(self, live_executor: Optional[Any]) -> None:
        self._lock = getattr(live_executor, "_close_lock", None) if live_executor is not None else None

    async def __aenter__(self):
        if self._lock is not None:
            await self._lock.acquire()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        if self._lock is not None:
            self._lock.release()


async def _do_market_close(
    client: Any, symbol: str, pos_mode: str, mgn_mode: str,
    mode_label: str, bot_label: str, live_executor: Optional[Any] = None,
) -> str:
    """Flatten the position at market (taker), cancelling conflicting orders."""
    _esc = TelegramNotifier.escape
    p = await _fetch_position_raw(client, symbol)
    if p is None:
        return (
            f"✅ *Already flat · {_esc(mode_label)}*\n{_SEP}\n"
            f"No open position on OKX\\.\n{_SEP}\n\\[{_esc(bot_label)}\\]"
        )
    d = _close_direction(p, pos_mode)
    # Tag the executor's trade as a manual close (before sending), and serialize
    # against the executor's own close path via the shared lock.
    ot = _mark_manual_close(live_executor)
    async with _MaybeLock(live_executor):
        try:
            resp = await client.close_position_market(
                symbol, mgn_mode=mgn_mode, pos_side=d["order_pos_side"], auto_cxl=True,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("manual market close failed: %s", exc)
            return (
                f"❌ *Close failed · {_esc(mode_label)}*\n{_SEP}\n"
                f"`{_code(exc)}`\n{_SEP}\n"
                f"{_esc('Try /close market again, or close on the OKX app.')}\n{_SEP}\n"
                f"\\[{_esc(bot_label)}\\]"
            )
        data = resp.get("data") or []
        item = data[0] if data else {}
        s_code = str(item.get("sCode", resp.get("code", "")))
        if s_code not in ("0", ""):
            s_msg = str(item.get("sMsg") or resp.get("msg") or "")
            return (
                f"❌ *Close rejected · {_esc(mode_label)}*\n{_SEP}\n"
                f"sCode `{_code(s_code)}`\n`{_code(s_msg)}`\n{_SEP}\n"
                f"{_esc('Check the OKX app and close manually.')}\n{_SEP}\n"
                f"\\[{_esc(bot_label)}\\]"
            )
        # Resting attached algos auto-cancel with the position; sweep any leftovers.
        await _quiet_cancel_algos(client, symbol)
    # Confirm flat — /positions lags the fill by ~1s, so poll briefly.
    flat = False
    for _ in range(6):
        await asyncio.sleep(0.5)
        if await _fetch_position_raw(client, symbol) is None:
            flat = True
            break
    if flat:
        _release_gate_if_match(live_executor, ot)
        head = f"✅ *Position closed · {_esc(mode_label)}*"
        tail = _esc("OKX now shows flat. New entries resume within a few seconds.")
    else:
        head = f"🟡 *Close sent · {_esc(mode_label)}*"
        tail = _esc("Close order sent but OKX still shows a position — "
                    "give it a moment, then /position.")
    return (
        f"{head}\n{_SEP}\n"
        f"`market close (taker)`\n{_SEP}\n"
        f"{tail}\n{_SEP}\n\\[{_esc(bot_label)}\\]"
    )


async def _do_limit_close(
    client: Any, symbol: str, pos_mode: str, mgn_mode: str,
    lot_sz: float, tick_sz: float, price: Optional[float],
    mode_label: str, bot_label: str, live_executor: Optional[Any] = None,
) -> str:
    """Place a reduceOnly maker-limit order to close the position.

    With an explicit ``price`` we send a plain ``limit`` (fills when price
    reaches it). With no price we pin to the current touch (best ask to sell a
    long, best bid to buy back a short) as ``post_only`` → always rests as maker.
    Any resting limit order (e.g. a leftover maker-exit / prior /close limit) is
    swept first so two full-size reduceOnly close orders can never stack.
    """
    _esc = TelegramNotifier.escape
    p = await _fetch_position_raw(client, symbol)
    if p is None:
        return (
            f"✅ *Already flat · {_esc(mode_label)}*\n{_SEP}\n"
            f"No open position to close\\.\n{_SEP}\n\\[{_esc(bot_label)}\\]"
        )
    d = _close_direction(p, pos_mode)
    sz_str = _fmt_size(d["size"], lot_sz)
    ord_type = "limit"
    if price is None:
        try:
            tk = await client.get_ticker(symbol)
            best_bid = float(tk.get("bidPx") or 0.0)
            best_ask = float(tk.get("askPx") or 0.0)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("limit-close ticker fetch failed: %s", exc)
            best_bid = best_ask = 0.0
        touch = best_ask if d["is_long"] else best_bid
        if touch <= 0:
            return (
                f"❌ *Limit close failed · {_esc(mode_label)}*\n{_SEP}\n"
                f"{_esc('No touch price available. Give a price: /close limit <price>')}\n"
                f"{_SEP}\n\\[{_esc(bot_label)}\\]"
            )
        price = touch
        ord_type = "post_only"
    px_str = _fmt_size(_round_to_tick(price, tick_sz), tick_sz)
    ot = _mark_manual_close(live_executor)
    async with _MaybeLock(live_executor):
        # Clear any resting limit order first so we never stack two full-size
        # reduceOnly close orders (protective TP/SL algos are left untouched).
        await _sweep_pending_limit_orders(client, symbol)
        try:
            resp = await client.place_order(
                symbol=symbol, side=d["close_side"], pos_side=d["order_pos_side"],
                td_mode=mgn_mode, ord_type=ord_type, sz=sz_str, px=px_str,
                reduce_only=True,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("manual limit close failed: %s", exc)
            return (
                f"❌ *Limit close failed · {_esc(mode_label)}*\n{_SEP}\n"
                f"`{_code(exc)}`\n{_SEP}\n"
                f"{_esc('Try /close market, or /close limit <price>.')}\n{_SEP}\n"
                f"\\[{_esc(bot_label)}\\]"
            )
    data = resp.get("data") or []
    item = data[0] if data else {}
    s_code = str(item.get("sCode", resp.get("code", "")))
    if s_code not in ("0", ""):
        s_msg = str(item.get("sMsg") or resp.get("msg") or "")
        return (
            f"❌ *Limit close rejected · {_esc(mode_label)}*\n{_SEP}\n"
            f"sCode `{_code(s_code)}`\n`{_code(s_msg)}`\n{_SEP}\n"
            f"{_esc('post_only may have crossed. Retry /close limit <price>, or /close market.')}\n"
            f"{_SEP}\n\\[{_esc(bot_label)}\\]"
        )
    oid = str(item.get("ordId") or "-")
    side_lbl = "SELL" if d["close_side"] == "sell" else "BUY"
    return (
        f"✋ *Maker limit close sent · {_esc(mode_label)}*\n{_SEP}\n"
        f"`{side_lbl} {sz_str}ct @ {px_str}`  `{ord_type}`\n"
        f"id `{oid}`\n{_SEP}\n"
        f"{_esc('Rests as maker until price reaches it. /orders to watch, '
               '/cancel to pull it, /close market to force out.')}\n{_SEP}\n"
        f"\\[{_esc(bot_label)}\\]"
    )


async def _do_cancel_all(
    client: Any, symbol: str, mode_label: str, bot_label: str,
) -> str:
    """Cancel resting orders — SL-SAFE.

    While a position is OPEN, only resting LIMIT/pending orders are cancelled and
    the protective TP/SL algos are deliberately KEPT (cancelling them would leave
    the position with no stop-loss while the bot still believes it is protected).
    To remove the stop you must flatten the position with /close. When the
    instrument is already FLAT, every resting order + algo is swept as cleanup.
    """
    _esc = TelegramNotifier.escape
    has_position = await _fetch_position_raw(client, symbol) is not None
    n_ord = 0
    n_algo = 0
    errs: list = []
    try:
        r = await client.get_pending_orders(symbol)
        for o in (r.get("data") or []):
            oid = o.get("ordId")
            if not oid:
                continue
            try:
                await client.cancel_order(symbol, ord_id=oid)
                n_ord += 1
            except Exception as exc:  # noqa: BLE001
                errs.append(_code(exc)[:60])
    except Exception as exc:  # noqa: BLE001
        errs.append(f"orders: {_code(exc)[:60]}")
    # Only pull algos (the TP/SL protection) when the position is already flat.
    if not has_position:
        try:
            r = await client.get_pending_algos(symbol)
            ids = [a.get("algoId") for a in (r.get("data") or []) if a.get("algoId")]
            if ids:
                await client.cancel_algos(symbol, ids)
                n_algo = len(ids)
        except Exception as exc:  # noqa: BLE001
            errs.append(f"algos: {_code(exc)[:60]}")
    lines = [
        f"🧹 *Cancelled resting orders · {_esc(mode_label)}*",
        _SEP,
        f"Limit / pending  `{n_ord}`",
    ]
    if has_position:
        lines.append(f"Algo TP/SL       `kept (position open)`")
    else:
        lines.append(f"Algo TP/SL       `{n_algo}`")
    if errs:
        lines.append(_SEP)
        lines.append("Some cancels errored:")
        lines.extend(f"`{_code(e)}`" for e in errs[:4])
    lines.append(_SEP)
    if has_position:
        lines.append(_esc(
            "Your TP/SL stop is still in place. To remove it you must close the "
            "position — use /close."
        ))
    else:
        lines.append(_esc("Instrument is flat — all resting orders cleared."))
    lines += [_SEP, f"\\[{_esc(bot_label)}\\]"]
    return "\n".join(lines)


def _parse_close_price(raw_text: str) -> tuple:
    """Parse the price token in '/close limit ...'.

    Returns ``(status, price)``:
      * ``("none", None)``  — no price token given → caller pins to the touch
      * ``("ok", float)``   — a valid price
      * ``("bad", None)``   — a token was supplied but isn't a number (a typo) →
        the caller warns instead of silently pinning to the touch.
    """
    if not raw_text:
        return ("none", None)
    parts = raw_text.split()
    cand: Optional[str] = None
    for i, tok in enumerate(parts):
        if tok.lstrip("/").lower() == "limit" and i + 1 < len(parts):
            cand = parts[i + 1]
            break
    if cand is None:
        return ("none", None)
    try:
        return ("ok", float(cand.lstrip("$").replace(",", "")))
    except ValueError:
        return ("bad", None)


async def _handle_command(
    action: str,
    raw_text: str,
    *,
    session: "VolumeFarmerSession",
    client: Any,
    symbol: str,
    tf: str,
    cfg: Dict[str, Any],
    mode_label: str,
    bot_label: str,
    start_time: datetime,
    log_dir: Optional[pathlib.Path],
    live_executor: Optional[Any],
    pos_mode: str,
) -> tuple:
    """Resolve an action to ``(text, buttons)``. ``buttons`` is None for plain cards.

    Actions: status / position / fees / trades / orders / help / close /
    close_market / close_limit / cancel / dismiss. The position-management
    actions are live/demo only (no real OKX position exists in paper mode).
    """
    _esc = TelegramNotifier.escape
    is_live = mode_label != "PAPER"
    mgn_mode = getattr(live_executor, "mgn_mode", "isolated") if live_executor is not None else "isolated"
    lot_sz = float(getattr(live_executor, "_lot_sz", 0.01)) if live_executor is not None else 0.01
    tick_sz = float(getattr(live_executor, "_tick_sz", 0.1)) if live_executor is not None else 0.1

    if action == "help":
        return _build_help_text(mode_label, bot_label), None
    if action == "dismiss":
        return None, None

    _MANAGE = {"orders", "close", "close_market", "close_limit", "cancel"}
    if action in _MANAGE and not is_live:
        return (
            f"ℹ️ *{_esc(action.replace('_', ' '))}*\n{_SEP}\n"
            f"{_esc('Only available in live/demo (no real OKX position in paper mode).')}\n"
            f"{_SEP}\n\\[{_esc(bot_label)}\\]",
            None,
        )

    if action == "orders":
        return _build_orders_text(
            await _fetch_open_orders(client, symbol), symbol, bot_label, mode_label,
        ), None
    if action == "close":
        text = await _build_close_confirm_text(
            client, symbol, tf, pos_mode, bot_label, mode_label, live_executor,
        )
        return text, _CLOSE_CONFIRM_BUTTONS
    if action == "close_market":
        return await _do_market_close(
            client, symbol, pos_mode, mgn_mode, mode_label, bot_label, live_executor,
        ), None
    if action == "close_limit":
        status, price = _parse_close_price(raw_text)
        if status == "bad":
            warn = _esc(
                "Couldn't read that price. Send a number, e.g. /close limit 60500 "
                "(or bare /close limit for a maker order at the current touch)."
            )
            return (
                f"❌ *Bad price · {_esc(mode_label)}*\n{_SEP}\n"
                f"{warn}\n{_SEP}\n\\[{_esc(bot_label)}\\]",
                None,
            )
        return await _do_limit_close(
            client, symbol, pos_mode, mgn_mode, lot_sz, tick_sz, price,
            mode_label, bot_label, live_executor,
        ), None
    if action == "cancel":
        return await _do_cancel_all(client, symbol, mode_label, bot_label), None

    # Read-only cards (status / position / fees / trades).
    text = await _dispatch_cmd(
        f"cmd_{action}", session, client, symbol, tf, cfg,
        mode_label, bot_label, start_time, log_dir, live_executor,
    )
    return text, None


async def _dispatch_cmd(
    data: str,
    session: "VolumeFarmerSession",
    client: Any,
    symbol: str,
    tf: str,
    cfg: Dict[str, Any],
    mode_label: str,
    bot_label: str,
    start_time: datetime,
    log_dir: Optional[pathlib.Path] = None,
    live_executor: Optional[Any] = None,
) -> Optional[str]:
    # In LIVE/DEMO, every card is sourced from the REAL account + real-fill log,
    # fetched once here and threaded into the builders (no paper numbers leak).
    is_live = mode_label != "PAPER"
    real_acct = await _fetch_real_account(client, symbol) if is_live else None
    # Source the trade stats from OKX's own order history (exchange truth);
    # only if that API call fails do we fall back to the local jsonl log.
    real_stats = None
    if is_live:
        real_stats = await _fetch_okx_trade_stats(client, symbol)
        if real_stats is None and log_dir is not None:
            real_stats = _real_trade_stats(_read_live_trades(log_dir))
    if data == "cmd_status":
        return _build_status_text(
            session, cfg, mode_label, bot_label, start_time, real_acct, real_stats,
        )
    if data == "cmd_position":
        return await _build_position_text(
            session, client, symbol, tf, bot_label,
            mode_label, real_acct, live_executor,
        )
    if data == "cmd_fees":
        return _build_fees_text(session, cfg, bot_label, real_stats, real_acct, mode_label)
    if data == "cmd_trades":
        return _build_trades_text(session, bot_label, real_stats, mode_label)
    return None


def _build_rebate_text(
    session: "VolumeFarmerSession",
    result: Dict[str, Any],
    bot_label: str,
) -> str:
    _esc = TelegramNotifier.escape
    requested = float(result.get("requested", 0.0))
    transferred = float(result.get("transferred", 0.0))
    available = float(result.get("available_rebate", session.available_rebate))
    equity = float(result.get("equity", session.equity))
    reason = str(result.get("reason", "ok"))
    if reason == "ok" and transferred > 0:
        head = "💸 *Rebate transferred*"
        body = (
            f"Requested  `${requested:.4f}`\n"
            f"Transferred  `${transferred:.4f}`\n"
            f"{_SEP}\n"
            f"New wallet balance  `${equity:.4f}`\n"
            f"Available rebate    `${available:.4f}`"
        )
    elif reason == "no_rebate_available":
        head = "⚠️ *Rebate transfer skipped*"
        body = (
            f"No rebate available yet\n"
            f"{_SEP}\n"
            f"Wallet balance   `${equity:.4f}`\n"
            f"Available rebate `${available:.4f}`"
        )
    elif reason == "non_positive_amount":
        head = "⚠️ *Rebate transfer skipped*"
        body = (
            f"Amount must be > 0\n"
            f"Usage: `/rebate <amount>`"
        )
    else:
        head = "⚠️ *Rebate transfer*"
        body = (
            f"Requested  `${requested:.4f}`\n"
            f"Transferred  `${transferred:.4f}`\n"
            f"Wallet balance  `${equity:.4f}`\n"
            f"Available rebate `${available:.4f}`"
        )
    return (
        f"{head}\n"
        f"{_SEP}\n"
        f"{body}\n"
        f"{_SEP}\n"
        f"\\[{_esc(bot_label)}\\]"
    )


def _handle_rebate_command(
    session: "VolumeFarmerSession",
    raw_text: str,
    bot_label: str,
    state_path: Optional[pathlib.Path],
) -> str:
    parts = raw_text.split()
    amount: Optional[float] = None
    if len(parts) >= 2:
        try:
            amount = float(parts[1].lstrip("$").replace(",", ""))
        except ValueError:
            amount = None
    if amount is None:
        result = {
            "requested": 0.0, "transferred": 0.0,
            "available_rebate": session.available_rebate,
            "equity": session.equity, "reason": "non_positive_amount",
        }
    else:
        result = session.transfer_rebate(amount)
        if result.get("transferred", 0.0) > 0 and state_path is not None:
            try:
                session.save_state(state_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("state save failed (rebate): %s", exc)
    return _build_rebate_text(session, result, bot_label)


async def _build_rebate_text_live(
    client: Any,
    symbol: str,
    cfg: Dict[str, Any],
    bot_label: str,
    mode_label: str,
) -> str:
    """/rebate in live/demo: INFORMATIONAL ONLY — the bot moves no funds.

    Shows the real wallet, real fees paid, and the estimated rebate (real fees ×
    rebate_pct). The user transfers the rebate themselves on OKX. (The PAPER
    path's simulated internal wallet transfer would report a fake new balance.)
    """
    _esc = TelegramNotifier.escape
    acct = await _fetch_real_account(client, symbol)
    stats = await _fetch_okx_trade_stats(client, symbol)
    rebate_pct = float((cfg.get("fees", {}) or {}).get("rebate_pct", 0.40))
    eq = acct.get("equity") if isinstance(acct, dict) else None
    eq_str = f"${eq:,.4f}" if eq is not None else "unavailable"
    fees = stats.get("fees") if stats else None
    fees_str = f"${fees:.4f}" if fees is not None else "unavailable"
    reb = (fees * rebate_pct) if fees is not None else None
    reb_str = f"${reb:.4f}" if reb is not None else "unavailable"
    return (
        f"💸 *Rebate · {_esc(mode_label)}*\n"
        f"{_SEP}\n"
        f"Wallet           `{eq_str}`\n"
        f"Real fees paid   `{fees_str}`\n"
        f"Est rebate {int(rebate_pct*100)}%   `{reb_str}`\n"
        f"{_SEP}\n"
        f"{_esc('Informational only — the bot does not move funds. Transfer your rebate on the OKX app/website.')}\n"
        f"{_SEP}\n"
        f"\\[{_esc(bot_label)}\\]"
    )


async def _command_loop(
    notifier: TelegramNotifier,
    session: "VolumeFarmerSession",
    client: Any,
    symbol: str,
    tf: str,
    cfg: Dict[str, Any],
    mode_label: str,
    bot_label: str,
    stop_evt: asyncio.Event,
    start_time: datetime,
    state_path: Optional[pathlib.Path] = None,
    log_dir: Optional[pathlib.Path] = None,
    live_executor: Optional[Any] = None,
    pos_mode: str = "net",
) -> None:
    if not notifier.enabled:
        return
    _esc = TelegramNotifier.escape

    menu_msg_id: Optional[int] = await notifier.send_with_buttons(
        f"🎛 *Controls* — tap to query or manage the bot\n"
        f"{_SEP}\n"
        f"`/help` for the full command list\n"
        f"{_SEP}\n"
        f"\\[{_esc(bot_label)}\\]",
        _MENU_BUTTONS,
    )

    _hc = dict(
        session=session, client=client, symbol=symbol, tf=tf, cfg=cfg,
        mode_label=mode_label, bot_label=bot_label, start_time=start_time,
        log_dir=log_dir, live_executor=live_executor, pos_mode=pos_mode,
    )

    async def _reply(text: Optional[str], buttons: Optional[list], reply_to: Optional[int]) -> None:
        if text and buttons:
            await notifier.send_with_buttons(text, buttons, reply_to_message_id=reply_to)
        elif text:
            await notifier.send_and_get_id(text, reply_to_message_id=reply_to)

    # Money-moving actions are guarded so a panicked double-tap can't fire two
    # closes; everything is dispatched off the poll loop so a slow close never
    # blocks getUpdates (the bot stays responsive during the EXIT-FAILED flurry).
    _inflight: set = set()
    _tasks: set = set()

    def _busy_card() -> str:
        return (
            f"⏳ *Close already in progress · {_esc(mode_label)}*\n{_SEP}\n"
            f"{_esc('Wait for the current close to finish, then check /position.')}\n"
            f"{_SEP}\n\\[{_esc(bot_label)}\\]"
        )

    async def _compute(action: str, raw_text: str) -> tuple:
        if action == "rebate":
            if mode_label != "PAPER":
                # Live/demo: informational only — no funds moved.
                return await _build_rebate_text_live(
                    client, symbol, cfg, bot_label, mode_label,
                ), None
            return _handle_rebate_command(session, raw_text, bot_label, state_path), None
        return await _handle_command(action, raw_text, **_hc)

    async def _dispatch(action: str, raw_text: str, reply_to: Optional[int]) -> None:
        guarded = action in ("close_market", "close_limit", "cancel")
        if guarded and "close" in _inflight:
            await _reply(_busy_card(), None, reply_to)
            return
        if guarded:
            _inflight.add("close")
        try:
            text, buttons = await _compute(action, raw_text)
            await _reply(text, buttons, reply_to)
        except Exception as exc:
            LOGGER.warning("command dispatch error: %s", exc)
        finally:
            if guarded:
                _inflight.discard("close")

    def _spawn(coro) -> None:
        t = asyncio.create_task(coro)
        _tasks.add(t)
        t.add_done_callback(_tasks.discard)

    offset = 0
    while not stop_evt.is_set():
        try:
            updates = await notifier.get_updates(offset=offset, timeout=20)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.debug("command loop error: %s", exc)
            await asyncio.sleep(5)
            continue

        for upd in updates:
            # Telegram guarantees update_id, but never trust the wire: a malformed
            # update must be skipped, not crash the loop (which would tear down the
            # whole run and stop live position monitoring).
            if not isinstance(upd, dict):
                continue
            uid = upd.get("update_id")
            if uid is None:
                continue
            offset = uid + 1

            if "callback_query" in upd:
                cq = upd.get("callback_query") or {}
                cq_id = cq.get("id")
                if not cq_id:
                    continue
                data = cq.get("data", "") or ""
                await notifier.answer_callback(cq_id)
                action = data[4:] if data.startswith("cmd_") else data
                _spawn(_dispatch(action, "", menu_msg_id))

            elif "message" in upd:
                msg = upd.get("message") or {}
                raw_text = (msg.get("text") or "").strip()
                if not raw_text.startswith("/"):
                    continue
                cmd = raw_text.split()[0].lstrip("/").split("@")[0].lower()
                reply_to = msg.get("message_id")
                if cmd == "rebate":
                    _spawn(_dispatch("rebate", raw_text, reply_to))
                elif cmd == "close":
                    parts = raw_text.split()
                    sub = parts[1].lower() if len(parts) > 1 else ""
                    action = (
                        "close_market" if sub == "market"
                        else "close_limit" if sub == "limit"
                        else "close"
                    )
                    _spawn(_dispatch(action, raw_text, reply_to))
                elif cmd in (
                    "status", "position", "fees", "trades",
                    "orders", "cancel", "help",
                ):
                    _spawn(_dispatch(cmd, raw_text, reply_to))


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
                                   bot_label=args.label or "okx-paper", intrabar=_intrabar,
                                   state_path=state_path, cfg=cfg)
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

    # OKX demo and live use separate API key tiers. When --demo, prefer the
    # OKX_DEMO_* env vars; fall back to live keys only if demo keys are absent
    # (in which case authenticated calls will 401 with code=50120 and entries
    # will silently fail — visible warning emitted here).
    if client_simulated:
        api_key = os.environ.get("OKX_DEMO_API_KEY") or os.environ.get("OKX_API_KEY", "")
        api_secret = os.environ.get("OKX_DEMO_API_SECRET") or os.environ.get("OKX_API_SECRET", "")
        passphrase = os.environ.get("OKX_DEMO_API_PASSPHRASE") or os.environ.get("OKX_API_PASSPHRASE", "")
        if not os.environ.get("OKX_DEMO_API_KEY"):
            LOGGER.warning(
                "--demo selected but OKX_DEMO_API_KEY not set in .env; falling back to "
                "live keys which will 401 on authenticated demo calls. Generate demo keys at "
                "https://www.okx.com/account/demo-trading and add OKX_DEMO_API_KEY/SECRET/PASSPHRASE.",
            )
    else:
        api_key = os.environ.get("OKX_API_KEY", "")
        api_secret = os.environ.get("OKX_API_SECRET", "")
        passphrase = os.environ.get("OKX_API_PASSPHRASE", "")

    async with OKXClient(
        api_key=api_key, api_secret=api_secret, passphrase=passphrase,
        simulated=client_simulated,
    ) as client:
        live_executor: Optional[LiveVolumeExecutorOKX] = None
        if is_live_or_demo:
            farmer_cfg = cfg.get("farmer", {}) or {}
            ts_cfg = farmer_cfg.get("time_stop", {}) or {}
            mx_cfg = farmer_cfg.get("maker_exit", {}) or {}
            er_cfg = farmer_cfg.get("entry_repeg", {}) or {}
            risk_cfg = cfg.get("risk", {}) or {}
            tf_seconds = _parse_timeframe_seconds(tf)
            live_executor = LiveVolumeExecutorOKX(
                client=client,
                symbol=symbol,
                pos_mode=args.pos_mode,
                mgn_mode="isolated",
                max_live_trades=args.max_live_trades,
                # Size live orders off the REAL OKX wallet, not the SIM equity /
                # hardcoded capital_usd. margin_fraction_per_trade comes from the
                # same farmer config the SIM uses, so sizing stays consistent.
                margin_frac=float(farmer_cfg.get("margin_fraction_per_trade", 0.03)),
                dry_run=args.live_dry_run,
                log_dir=log_dir,
                time_stop_enabled=bool(ts_cfg.get("enabled", False)),
                max_hold_bars=int(ts_cfg.get(
                    "max_hold_bars", farmer_cfg.get("max_hold_bars", 2),
                )),
                bar_seconds=int(ts_cfg.get("bar_seconds", tf_seconds)),
                maker_exit_enabled=bool(mx_cfg.get("enabled", False)),
                maker_exit_cfg=MakerExitConfig(
                    repeg_ms=int(mx_cfg.get("repeg_ms", 750)),
                    max_repegs=int(mx_cfg.get("max_repegs", 8)),
                    max_exit_seconds=float(mx_cfg.get("max_exit_seconds", 20.0)),
                    max_adverse_bps=float(mx_cfg.get("max_adverse_bps", 15.0)),
                    poll_ms=int(mx_cfg.get("poll_ms", 120)),
                ),
                entry_repeg_cfg=EntryRepegConfig(
                    enabled=bool(er_cfg.get("enabled", True)),
                    repeg_ms=int(er_cfg.get("repeg_ms", 8_000)),
                    max_repegs=int(er_cfg.get("max_repegs", 6)),
                    max_entry_seconds=float(er_cfg.get("max_entry_seconds", 60.0)),
                    poll_ms=int(er_cfg.get("poll_ms", 150)),
                    taker_fallback=bool(er_cfg.get("taker_fallback", False)),
                ),
                # REAL-wallet circuit breaker — gates live entries on the actual
                # OKX balance (not the simulation's paper equity).
                breaker_consec_loss_limit=int(risk_cfg.get("consecutive_losses_limit", 3)),
                breaker_cooldown_s=float(risk_cfg.get("consecutive_losses_cooldown_bars", 72)) * tf_seconds,
                breaker_daily_loss_pct=float(risk_cfg.get("daily_loss_limit_pct", 0.02)),
                breaker_max_drawdown_pct=float(risk_cfg.get("max_drawdown_pct", 0.10)),
            )
            LOGGER.info(
                "live executor: time_stop=%s (max_hold=%d bars, %ds each), maker_exit=%s, "
                "entry_repeg=%s (max_repegs=%d, repeg_ms=%d, max_secs=%.0f, taker_fallback=%s)",
                live_executor.time_stop_enabled, live_executor.max_hold_bars,
                live_executor.bar_seconds, live_executor.maker_exit_enabled,
                live_executor.entry_repeg_cfg.enabled,
                live_executor.entry_repeg_cfg.max_repegs,
                live_executor.entry_repeg_cfg.repeg_ms,
                live_executor.entry_repeg_cfg.max_entry_seconds,
                live_executor.entry_repeg_cfg.taker_fallback,
            )
            try:
                await live_executor.initialize(leverage_cap=args.live_leverage)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("live executor init failed: %s", exc)
                return 2
            # OKX-confirmed lifecycle cards: every trade message is emitted by the
            # executor at a real exchange confirmation point, not the simulation.
            _exec_mode = "LIVE" if args.live else "DEMO"
            _exec_lbl = args.label or "okx-paper"
            live_executor.on_entry_filled = _make_entry_filled_callback(
                notifier, symbol, mode_label=_exec_mode, bot_label=_exec_lbl,
                tf=tf, client=client,
            )
            # The confirmed close card (real avgPx + exchange fees + OKX pnl).
            live_executor.on_close_confirmed = _make_real_fill_callback(
                notifier, symbol, mode_label=_exec_mode, bot_label=_exec_lbl,
                client=client,
            )
            # Honest alert when a close can't be priced or the position is still
            # open — never a fabricated "$0.00 REAL FILL".
            live_executor.on_close_unverified = _make_close_unverified_callback(
                notifier, symbol, mode_label=_exec_mode, bot_label=_exec_lbl,
                client=client,
            )
            # REAL-wallet circuit-breaker alerts: loss-streak pause, daily-loss
            # halt, and max-drawdown HARD halt — all fired off the actual OKX
            # balance, not the simulation's paper equity.
            if notifier is not None and notifier.enabled:
                _esc_brk = TelegramNotifier.escape

                async def _on_breaker(info) -> None:
                    kind = str(info.get("kind", "?"))
                    detail = str(info.get("detail", ""))
                    icon = {"loss_streak": "⏸️", "daily_loss": "🛑",
                            "max_drawdown": "⛔"}.get(kind, "⚠️")
                    try:
                        await notifier.send_raw(
                            f"{icon} *RISK BREAKER · {_esc_brk(_exec_mode)}*\n"
                            f"{_SEP}\n"
                            f"`{_esc_brk(kind)}`\n"
                            f"{_esc_brk(detail)}\n"
                            f"\\[{_esc_brk(_exec_lbl)}\\]"
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning("breaker telegram send failed: %s", exc)

                live_executor.on_breaker = _on_breaker
            # on_entry_abandoned is left unset on purpose: no-fill entries are
            # silent (log only) per the chosen behaviour.

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
                                      intrabar=_intrabar, state_path=state_path, cfg=cfg)
        session.event_callback = handler

        # Load persisted state and reconcile any orphan position.
        _esc_resume = TelegramNotifier.escape
        _bot_lbl = args.label or "okx-paper"
        had_position = False
        if state_path.exists():
            try:
                session.load_state(state_path)
                LOGGER.info(
                    "state loaded: trades=%d  eq=%.4f  vol=%.0f  position=%s",
                    session.round_trips, session.equity, session.total_volume_usd,
                    session.position.side if session.position else "none",
                )
                had_position = session.position is not None
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("state load failed (starting fresh): %s", exc)

        orphan_result: Optional[str] = None
        if session.position is not None:
            closed_bars = history[history["closed"]].copy()
            orphan_result = session.reconcile_orphan_position(closed_bars)

        # STARTUP message.
        if notifier.enabled:
            _esc = TelegramNotifier.escape
            # Show the REAL OKX wallet in live/demo, not the hardcoded capital_usd.
            cap = cfg["farmer"]["capital_usd"]
            cap_label = "Capital"
            if mode_label != "PAPER" and client is not None:
                _acct = await _fetch_real_account(client, symbol)
                if _acct.get("equity") is not None:
                    cap = float(_acct["equity"]); cap_label = "Wallet "
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
            okx_cap = args.live_leverage if is_live_or_demo else None
            okx_cap_line = f"  ·  OKX cap `{okx_cap}×`" if okx_cap else ""
            vol_target = cfg.get("target", {}).get("volume_usd", 0)
            startup_msg = (
                f"🚀 *vGen OKX · {_esc(mode_label)}*\n"
                f"{_esc(symbol)} · {_esc(tf)}\n"
                f"{_SEP}\n"
                f"{cap_label}    `${cap:,.2f}`\n"
                f"Lev sizing `{min_lev}-{max_lev}×` \\(dynamic\\){okx_cap_line}\n"
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
            if mode_label != "PAPER" and client is not None:
                # Live/demo: report the REAL OKX position on restart, not the SIM
                # orphan reconciliation (which never queried the exchange).
                try:
                    acct = await _fetch_real_account(client, symbol)
                except Exception as exc:  # noqa: BLE001
                    acct = {}
                    LOGGER.debug("orphan banner real fetch failed: %s", exc)
                if acct.get("pos_sz"):
                    lines = ["♻️ *Resuming · open position on OKX*"]
                    lines.extend(_real_acct_lines(acct))
                    lines.append(f"{_SEP}")
                    lines.append(
                        _esc_resume(
                            "Note: the bot is not yet watching this pre-restart "
                            "position. Manage/close it on OKX; new entries continue."
                        )
                    )
                    lines.append(f"\\[{_esc_resume(_bot_lbl)}\\]")
                    await notifier.send_raw("\n".join(lines))
                # else: flat on OKX → nothing to resume, no banner.
            elif had_position and orphan_result is not None:
                icon = "✅" if orphan_result == "tp" else "❌"
                await notifier.send_raw(
                    f"{icon} *Orphan position resolved on restart*\n"
                    f"{_SEP}\n"
                    f"Result  `{_esc_resume(orphan_result.upper())}`\n"
                    f"Equity  `${session.equity:.4f}`\n"
                    f"{_SEP}\n"
                    f"\\[{_esc_resume(_bot_lbl)}\\]"
                )
            elif had_position and orphan_result is None:
                pos = session.position
                await notifier.send_raw(
                    f"⏸ *Resuming with open position*\n"
                    f"{_SEP}\n"
                    f"`{_esc_resume(pos.side.upper())}`  entry `{pos.entry_price:,.2f}`\n"
                    f"TP `{pos.tp:,.2f}`  ·  SL `{pos.sl:,.2f}`\n"
                    f"{_SEP}\n"
                    f"\\[{_esc_resume(_bot_lbl)}\\]"
                )

        start_time = datetime.now(tz=timezone.utc)
        poll_task = asyncio.create_task(_poll_loop(
            client, session, symbol, tf, args.poll_seconds,
            history, end_at, log_dir, handler,
            notifier=notifier, mode_label=mode_label,
            bot_label=args.label or "okx-paper", cfg=cfg, intrabar=_intrabar,
        ))
        cmd_task = asyncio.create_task(_command_loop(
            notifier, session, client, symbol, tf, cfg,
            mode_label, bot_label=args.label or "okx-paper",
            stop_evt=stop_evt, start_time=start_time,
            state_path=state_path,
            log_dir=log_dir, live_executor=live_executor,
            pos_mode=args.pos_mode,
        ))
        try:
            await asyncio.wait(
                {poll_task, cmd_task, asyncio.create_task(stop_evt.wait())},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for t in (poll_task, cmd_task):
                t.cancel()
            for t in (poll_task, cmd_task):
                try:
                    await t
                except asyncio.CancelledError:
                    pass

    session.save_state(state_path)
    s = session.summary()
    LOGGER.info("FINAL (sim view): trades=%d  wr=%.2f%%  eq=$%.4f  vol=$%.0f  pnl=$%.4f",
                s["round_trips"], s["win_rate_pct"], s["equity"], s["volume_usd"], s["total_pnl"])
    if notifier.enabled:
        try:
            # LIVE: summarise REAL trades from the on-disk live-fill log (the OKX
            # client session is already closed here, so we can't hit the API).
            # DEMO/PAPER keep the simulated session summary.
            if mode_label == "LIVE":
                rs = _real_trade_stats(_read_live_trades(log_dir))
                net = float(rs.get("net", 0.0))
                pnl_str = f"+${net:.4f}" if net >= 0 else f"-${abs(net):.4f}"
                await notifier.send_raw(
                    f"⏹ *vGen OKX stopped · {TelegramNotifier.escape(mode_label)}*\n"
                    f"{_SEP}\n"
                    f"Real trades  `{rs.get('n', 0)}`  ·  WR `{rs.get('wr', 0.0):.1f}%`\n"
                    f"{_SEP}\n"
                    f"Net PnL  `{pnl_str}`  ·  fees `${rs.get('fees', 0.0):.4f}`\n"
                    f"Volume   `${rs.get('volume', 0.0):,.0f}`\n"
                    f"{_SEP}\n"
                    f"\\[{TelegramNotifier.escape(args.label or 'okx-paper')}\\]"
                )
            else:
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
