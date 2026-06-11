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
import time
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
    DemoRealismConfig,
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
        {"text": "🗺 Plan",      "callback_data": "cmd_plan"},
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

# Strong refs for fire-and-forget sends. The event loop keeps only weak refs
# to tasks — without this, a GC pass could collect an in-flight Telegram send
# (entry/exit/halt/milestone cards) and the message silently vanished.
_BG_TASKS: set = set()


def _bg(coro) -> "asyncio.Task":
    t = asyncio.create_task(coro)
    _BG_TASKS.add(t)
    t.add_done_callback(_BG_TASKS.discard)
    return t


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
    initial_delay: float = 0.0,
) -> None:
    bar_seconds = _parse_timeframe_seconds(tf)
    if initial_delay > 0:
        # Secondary symbols start their poll a few seconds later so that on a
        # shared 5m boundary the PRIMARY signal reaches the global gate first.
        await asyncio.sleep(initial_delay)
    last_bar_ts: Optional[pd.Timestamp] = (
        history["open_time"].iloc[-1] if not history.empty else None
    )
    LOGGER.info("seeded %d closed bars, last=%s", len(history), last_bar_ts)
    LOGGER.info("polling %s %s every %ds (bar size %ds)", symbol, tf, poll_sec, bar_seconds)

    fail_streak = 0
    while True:
        if end_at is not None and datetime.now(tz=timezone.utc) >= end_at:
            LOGGER.info("duration elapsed; stopping")
            break
        if session.halted:
            # BOUNTY GUARD: the session halts on SIM-counted volume, which
            # overcounts reality (abandoned entries, breaker-blocked entries
            # still book sim volume). Stopping on sim 5.1M with real exchange
            # volume short of 5M would silently forfeit the reward — verify
            # against OKX's own numbers before honoring a volume-target halt.
            if (mode_label != "PAPER"
                    and str(session.halt_reason).startswith("volume_target_reached")):
                tgt = float((cfg or {}).get("target", {}).get("volume_usd", 0) or 0)
                stats = None
                try:
                    stats = await _fetch_okx_trade_stats(client, symbol)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("real-volume check failed: %s", exc)
                real_vol = float(stats.get("volume", 0.0)) if stats else None
                if tgt > 0 and real_vol is not None and real_vol < tgt:
                    session.halted = False
                    session.halt_reason = ""
                    LOGGER.warning(
                        "sim volume target hit but REAL volume $%.0f < target $%.0f — "
                        "continuing until the exchange counter agrees", real_vol, tgt,
                    )
                    if notifier and notifier.enabled:
                        _e_ = TelegramNotifier.escape
                        _bg(notifier.send_raw(
                            f"⚠️ *Target check · {_e_(mode_label)}*\n{_SEP}\n"
                            f"Sim counter hit the target, but OKX real volume is "
                            f"`${real_vol:,.0f}` of `${tgt:,.0f}`\\. Continuing until "
                            f"the EXCHANGE counter crosses the line — the reward pays "
                            f"on their number, not ours\\.\n"
                            f"{_SEP}\n\\[{_e_(bot_label)}\\]"
                        ))
                    continue
            LOGGER.warning("session halted: %s", session.halt_reason)
            break

        try:
            # Gap backfill: after an outage longer than 10 bars the old fixed
            # limit=10 silently dropped bars forever (an open position's TP/SL
            # touch inside the gap was never evaluated). Fetch what's missing.
            need = 10
            if last_bar_ts is not None:
                behind = (pd.Timestamp.now(tz="UTC") - last_bar_ts).total_seconds() / bar_seconds
                need = int(min(300, max(10, behind + 5)))
            raw = await client.get_candles(symbol, tf, limit=need)
            df_new = _normalize_okx_candles(raw)
            fail_streak = 0
        except Exception as exc:  # noqa: BLE001
            fail_streak += 1
            LOGGER.warning("candle fetch failed (streak %d): %s", fail_streak, exc)
            # The operator must hear about a persistent feed outage — the bot
            # is flying blind on an open position. Alert at 5 consecutive
            # failures, re-alert every 50.
            if notifier and notifier.enabled and (fail_streak == 5 or fail_streak % 50 == 0):
                _esc_ = TelegramNotifier.escape
                _bg(notifier.send_raw(
                    f"🚨 *Market data outage · {_esc_(mode_label)}*\n{_SEP}\n"
                    f"{fail_streak} consecutive candle fetch failures\\.\n"
                    f"Last error: `{_esc_(str(exc)[:150])}`\n"
                    f"{_SEP}\n\\[{_esc_(bot_label)}\\]"
                ))
            await asyncio.sleep(min(poll_sec, 10))
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
                    _bg(notifier.send_and_get_id(text, reply_to_message_id=entry_msg_id))
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

        # Bar-aligned cadence. The fixed 20s sleep made ~15 redundant candle
        # fetches per 5m bar AND delayed the entry signal up to 20s after bar
        # close — a worse maker-queue position on every single entry. Sleep to
        # just past the next bar close instead; PAPER keeps the short cadence
        # too (its intrabar TP/SL cards need mid-bar polls); an overdue bar
        # (OKX confirms late) re-polls every 2s until it lands.
        if last_bar_ts is None:
            sleep_s = float(poll_sec)
        else:
            next_close = last_bar_ts + pd.Timedelta(seconds=2 * bar_seconds)
            wait = (next_close - pd.Timestamp.now(tz="UTC")).total_seconds() + 1.0
            if wait <= 0:
                wait = 2.0
            if mode_label == "PAPER":
                sleep_s = min(float(poll_sec), wait)
            else:
                sleep_s = min(wait, bar_seconds + 5.0)
        await asyncio.sleep(sleep_s)


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
                # matplotlib render is ~0.5-2s of synchronous CPU; run it off
                # the event loop so candle polling / order watching / Telegram
                # long-poll never freeze during an entry card.
                chart = await asyncio.to_thread(
                    _draw_entry_chart, bars[bars["closed"]], side, entry, tp, sl,
                    symbol=symbol, tf=tf,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("chart generation failed: %s", exc)
        if chart:
            msg_id = await notifier.send_photo(chart, caption=text)
            if msg_id is None:
                # Photo upload failed (429/caption too long/network): fall back
                # to a plain text card rather than losing the entry message.
                msg_id = await notifier.send_and_get_id(text)
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
            _bg(notifier.send_raw(msg + _lbl))

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
                _bg(_send_entry_with_chart(text, side, entry, tp, sl, equity))
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
            intrabar_guess = intrabar.get(ikey) if intrabar is not None else None
            already_notified = intrabar_guess == reason
            corrects_estimate = intrabar_guess is not None and intrabar_guess != reason
            entry_msg_threaded: Optional[int] = None
            if intrabar is not None:
                # ALWAYS pop, match or not. A mismatched key used to leak
                # forever AND wrongly suppress the intrabar card of a future
                # trade at the same rounded entry price.
                intrabar.pop(ikey, None)
                entry_msg_threaded = intrabar.pop(f"{ikey}_msg_id", None)

            # PAPER only: render the exit card from the simulated touch. In
            # live/demo the ONE exit message is the executor's OKX-confirmed
            # close card (on_close_confirmed) — this SIM card is suppressed so
            # there are never two conflicting exit messages with different
            # numbers, and never a "closed" card while the real position is open.
            if notifier and notifier.enabled and not already_notified and not is_real:
                # Honest at-a-glance icon per reason — sl_ambiguous and
                # trend_break used to render with the time-stop icon.
                emoji = {
                    "tp": "✅", "sl": "❌", "sl_ambiguous": "❌",
                    "time_stop": "⏱", "trend_break": "✂️", "manual": "📤",
                }.get(reason, "⏱")
                g_str = f"+${abs(gross):.4f}" if gross >= 0 else f"-${abs(gross):.4f}"
                n_str = f"+${abs(net):.4f}" if net >= 0 else f"-${abs(net):.4f}"
                open_fee_disp = p.get("open_fee", 0.0) or 0.0
                correction = (
                    f"⚠️ _corrects earlier intrabar *{_esc(str(intrabar_guess).upper())}* estimate_\n"
                    if corrects_estimate else ""
                )
                text = (
                    f"{emoji} *{_esc(reason.upper())} · {_esc(side.upper())} \\#{_esc(str(trade_num))} · {_esc(mode_label)}*\n"
                    f"{correction}"
                    f"{_esc(symbol)}  `{entry_px:,.2f}` → `{exit_px:,.2f}`\n"
                    f"{_SEP}\n"
                    f"Gross   `{g_str}`\n"
                    f"Fees    `${open_fee_disp:.4f}` \\+ `${close_fee:.4f}` \\({_esc(close_fee_type)}\\)\n"
                    f"Net     `{n_str}`\n"
                    f"{_SEP}\n"
                    f"`{wins}W · {losses}L · {wr:.1f}% WR`\n"
                    f"{_SEP}\n"
                    f"Balance  `${equity:.4f}`"
                )
                text = text + "\n" + "\n".join(_journey_lines(session, cfg))
                entry_msg_id = _pending.get("msg_id") or entry_msg_threaded
                _bg(_send_exit(text, entry_msg_id))
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
                    _bg(_send_auto_real("halt", evt.payload))
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
                    _bg(_send_auto_real("milestone", evt.payload))
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
                    # Live/demo rebate reminders are owned by _live_ops_loop,
                    # which triggers on the REAL wallet + REAL 24h fees. The
                    # session's trigger keys off simulated paper equity and
                    # its ask compounded lifetime fees — not real facts.
                    LOGGER.debug("sim rebate_reminder suppressed in %s (ops loop owns it)", mode_label)
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
        # pos_ok=False means the positions call FAILED: renderers must show
        # "unavailable", never "flat" (which fabricates zero-exposure).
        "pos_ok": False,
    }
    # Balance + positions concurrently — these two sequential awaits sat on
    # the critical path of every card and command.
    bal_res, pos_res = await asyncio.gather(
        client.get_balance("USDT"),
        client.get_positions(symbol),
        return_exceptions=True,
    )
    try:
        if isinstance(bal_res, Exception):
            raise bal_res
        bal = bal_res
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
        if isinstance(pos_res, Exception):
            raise pos_res
        pos = pos_res
        out["pos_ok"] = True
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
    if not acct.get("pos_ok", True):
        # The positions call FAILED. Saying "flat" here fabricated a fact —
        # hiding live leveraged exposure exactly when the operator must act.
        lines.append("Open pos  `unavailable` \\(positions API error\\)")
    elif acct.get("pos_side") and acct.get("pos_sz"):
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
    if not acct.get("pos_ok", True):
        block += "Open pos   `unavailable` \\(positions API error\\)\n"
    elif acct.get("pos_side") and acct.get("pos_sz"):
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
    # Exit-reason composition (TP / SL / TimeStop / …). Keyed by the REAL
    # close_reason logged at exit — independent of win/lose, so a fee-eroded TP
    # still counts as a TP. Skipped for OKX-history-sourced records, whose reason
    # is only synthesized from the PnL sign (reason_synthetic) and would just
    # mirror W/L. Each bucket tracks count, net P&L and how many were net-wins.
    by_reason: Dict[str, Dict[str, Any]] = {}
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
            if not r.get("reason_synthetic"):
                reason = str(r.get("close_reason") or "other").lower()
                br = by_reason.setdefault(reason, {"n": 0, "net": 0.0, "wins": 0})
                br["n"] += 1
                br["net"] += net
                if net > 0:
                    br["wins"] += 1
    wr = wins / n * 100.0 if n else 0.0
    return {
        "n": n, "wins": wins, "losses": losses, "wr": wr,
        "net": net_total, "gross": gross,
        "open_fees": open_fees, "close_fees": close_fees,
        "fees": open_fees + close_fees, "volume": volume,
        "closed": closed, "by_reason": by_reason,
    }


# Display order + short labels for the exit-reason breakdown. Anything not
# listed here is grouped under "OTH" so new reasons never silently vanish.
_REASON_LABELS = [
    ("tp", "TP"), ("sl", "SL"), ("time_stop", "TS"),
    ("trend_break", "TB"), ("manual", "MAN"),
]


def _format_exit_breakdown(by_reason: Dict[str, Any]) -> str:
    """One-line exit-reason composition: ``TP 12 (40%) · SL 9 (30%) · TS 9 (30%)``.

    Ratios are share-of-closed-trades, by REAL exit reason — not win/lose. Returns
    "" when there's nothing to show (no real-reason trades yet).
    """
    if not by_reason:
        return ""
    total = sum(v.get("n", 0) for v in by_reason.values())
    if total <= 0:
        return ""
    parts, seen = [], set()
    for key, lab in _REASON_LABELS:
        v = by_reason.get(key)
        if not v or not v.get("n"):
            continue
        seen.add(key)
        parts.append(f"{lab} {v['n']} ({v['n'] / total * 100:.0f}%)")
    other = sum(v.get("n", 0) for k, v in by_reason.items() if k not in seen)
    if other:
        parts.append(f"OTH {other} ({other / total * 100:.0f}%)")
    return "  ·  ".join(parts)


# TTL cache for the expensive real-stats fetch: up to 50 sequential
# authenticated GETs per call, previously re-run from scratch on EVERY
# /status //fees //trades command AND every live halt/milestone/rebate card —
# burning auth rate limit exactly when an EXIT-FAILED flurry needs headroom.
_STATS_CACHE: Dict[str, Any] = {"ts": 0.0, "key": "", "val": None}
_STATS_CACHE_TTL_S = 30.0
# Instrument ctVal cache (was hardcoded 0.01 = BTC-only; wrong stats for any
# other instrument).
_CT_VAL_CACHE: Dict[str, float] = {}


async def _instrument_ct_val(client: Any, symbol: str) -> float:
    inst = to_okx_inst_id(symbol)
    if inst in _CT_VAL_CACHE:
        return _CT_VAL_CACHE[inst]
    try:
        spec = await client.get_instrument(symbol)
        ct = float(spec.get("ctVal") or 0.01)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("ctVal fetch failed (defaulting 0.01): %s", exc)
        return 0.01
    _CT_VAL_CACHE[inst] = ct
    return ct


def _merge_stats(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Merge two _real_trade_stats dicts (multi-symbol campaign totals)."""
    if a is None:
        return b
    if b is None:
        return a
    out = dict(a)
    for k in ("n", "wins", "losses", "volume", "fees", "gross", "net"):
        out[k] = (a.get(k) or 0) + (b.get(k) or 0)
    n = out.get("n") or 0
    out["wr"] = (out.get("wins", 0) / n * 100.0) if n else 0.0
    br = dict(a.get("by_reason") or {})
    for k, v in (b.get("by_reason") or {}).items():
        br[k] = br.get(k, 0) + v
    out["by_reason"] = br
    return out


async def _fetch_okx_trade_stats_multi(client: Any, symbols: list) -> Optional[Dict[str, Any]]:
    """Exchange-truth stats summed across every campaign instrument."""
    total: Optional[Dict[str, Any]] = None
    for sym in symbols:
        st = await _fetch_okx_trade_stats(client, sym)
        total = _merge_stats(total, st)
    return total


async def _fetch_okx_trade_stats(
    client: Any, symbol: str, max_pages: int = 25,
) -> Optional[Dict[str, Any]]:
    """Build real-trade stats from OKX's OWN order history (the exchange truth).

    Merges /orders-history (7d) + /orders-history-archive (3mo), pairs each
    opening fill (pnl==0) with its closing fill (pnl!=0) into a round-trip,
    synthesizes a record in the live_trades schema, and runs it through
    ``_real_trade_stats`` so the Telegram cards render identically — but now
    sourced from the exchange, immune to any local-logging drift. Returns None
    on any API error so the caller can fall back to the local jsonl.

    max_pages=25 ⇒ up to 2,500 orders per endpoint. The old cap of 5 (~500
    orders) silently under-reported volume for most of a 5M campaign (~1,600+
    orders), so milestones fired late and the rebate ask was undersized.
    Results are cached for ``_STATS_CACHE_TTL_S`` seconds.
    """
    now = time.time()
    cache_key = to_okx_inst_id(symbol)
    if (
        _STATS_CACHE["val"] is not None
        and _STATS_CACHE["key"] == cache_key
        and now - _STATS_CACHE["ts"] < _STATS_CACHE_TTL_S
    ):
        return _STATS_CACHE["val"]
    inst = to_okx_inst_id(symbol)
    ct_val = await _instrument_ct_val(client, symbol)
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
            # OKX history has no exit reason — this is a W/L guess, not a real
            # TP/SL/TimeStop label, so keep it out of the exit-reason breakdown.
            "reason_synthetic": True,
            # _real_trade_stats counts notional×2 (entry+exit); use the mean of the
            # two legs so the doubled total equals open+close notional exactly.
            "notional_usd": ((op.get("notional") or close_notional) + close_notional) / 2.0,
            "real_gross_pnl": pnl,
            "real_open_fee": open_fee,
            "real_close_fee": close_fee,
        })
    stats = _real_trade_stats(records)
    _STATS_CACHE.update({"ts": time.time(), "key": cache_key, "val": stats})
    return stats


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
        extras = trade.extras or {}
        lev = float(extras.get("leverage") or 0.0)
        margin_usd = float(extras.get("margin_usd") or 0.0)
        if margin_usd <= 0 and lev > 0:
            margin_usd = notional / lev
        margin_line = (
            f"Margin  `${margin_usd:,.2f}`  ·  Lev `{lev:.0f}×`\n"
            if lev > 0 else ""
        )
        text = (
            f"{arrow} *{_esc(side.upper())} FILLED · {_esc(mode_label)}*\n"
            f"{_esc(symbol)} · {_esc(tf)}\n"
            f"{_SEP}\n"
            f"Entry →  `{fill_px:,.2f}`  \\(real fill\\)\n"
            f"TP    →  `{tp:,.2f}`   `+{tp_bps:.1f}bps`\n"
            f"SL    →  `{sl:,.2f}`   `-{sl_bps:.1f}bps`\n"
            f"{_SEP}\n"
            f"{margin_line}"
            f"Size    `${notional:,.2f}`  ·  `{sz:g}` ct\n"
            f"Open fee  `${open_fee:.4f}`\n"
            f"{_SEP}\n"
            f"`ordId {ordid}`"
        )
        chart: Optional[bytes] = None
        if client is not None:
            # Account snapshot and chart candles concurrently — these were
            # 3 sequential REST round trips on the entry-card path.
            acct_res, candles_res = await asyncio.gather(
                _fetch_real_account(client, symbol),
                client.get_candles(symbol, tf, limit=35),
                return_exceptions=True,
            )
            if isinstance(acct_res, dict):
                text = text + "\n" + "\n".join(_real_acct_lines(acct_res))
            try:
                if isinstance(candles_res, Exception):
                    raise candles_res
                bars = _normalize_okx_candles(candles_res)
                # Render off the event loop: the synchronous matplotlib pass
                # used to freeze position polling for up to ~2s per entry.
                chart = await asyncio.to_thread(
                    _draw_entry_chart, bars[bars["closed"]], side, fill_px, tp, sl,
                    symbol=symbol, tf=tf,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("entry chart generation failed: %s", exc)
        text = text + _lbl
        try:
            msg_id = None
            if chart:
                msg_id = await notifier.send_photo(chart, caption=text)
            if msg_id is None:
                # No chart, or the photo upload failed — never lose the card.
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


def _journey_lines(session: Any, cfg: Optional[Dict[str, Any]]) -> list:
    """Compact campaign-progress block for exit/close/status cards.

    Answers "where are we heading": % of target, pace vs required, ETA,
    realized net cost per 1M, and projected total fees vs the reward.
    """
    try:
        if isinstance(session, (list, tuple)):
            summs = [s_.summary() for s_ in session]
            summ = dict(summs[0])
            for k in ("volume_usd", "fees_net", "fees_gross", "total_pnl"):
                summ[k] = sum(float(x.get(k) or 0.0) for x in summs)
            vol = summ["volume_usd"]
            summ["net_cost_per_1m"] = (summ["fees_net"] / vol * 1e6) if vol > 0 else 0.0
            # pace comes from the primary (first) session's controller
        else:
            summ = session.summary()
    except Exception:  # noqa: BLE001
        return []
    vol = float(summ.get("volume_usd") or 0.0)
    tgt_cfg = (cfg or {}).get("target", {}) or {}
    tgt = float(tgt_cfg.get("volume_usd", 0) or 0)
    if tgt <= 0 or vol <= 0:
        return []
    reward = float(tgt_cfg.get("reward_usd", 1500.0) or 0)
    pct = vol / tgt * 100.0
    fees_net = float(summ.get("fees_net") or 0.0)
    per_1m = float(summ.get("net_cost_per_1m") or 0.0)
    proj_fees = per_1m * tgt / 1_000_000.0
    pace = summ.get("pace") or {}
    # The block is built from the SESSION ledger (sim view). In live mode the
    # real exchange-counted volume lives on /fees; this is pacing guidance and
    # is labeled as such so it can never be read as an account fact.
    lines = [_SEP, f"📍 *Journey \\(sim est\\)*  `{pct:.1f}%`  ·  `${vol:,.0f}` / `${tgt:,.0f}`"]
    req = pace.get("required_per_day")
    ach = pace.get("achieved_per_day")
    days_left = pace.get("days_left")
    if req is not None and ach:
        status = "🟢 on pace" if float(ach) >= float(req) else "🔴 behind"
        lines.append(f"Pace  `${float(ach):,.0f}/day`  need `${float(req):,.0f}/day`  {status}")
        if days_left is not None and float(ach) > 0:
            eta_days = (tgt - vol) / float(ach)
            lines.append(f"ETA  `{eta_days:.1f}d`  \\(window `{float(days_left):.1f}d`\\)")
    lines.append(f"Net fees  `${fees_net:,.2f}`  ·  `${per_1m:,.0f}/1M`")
    if reward > 0 and proj_fees > 0:
        lines.append(f"At target  fees `${proj_fees:,.0f}`  vs reward `${reward:,.0f}`")
    return lines


def _make_real_fill_callback(
    notifier: Optional[TelegramNotifier],
    symbol: str,
    mode_label: str,
    bot_label: str,
    client: Any = None,
    session: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
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
        gross = float(trade.real_gross_pnl)
        open_fee = float(trade.real_open_fee)
        close_fee = float(trade.real_close_fee)
        net = gross - open_fee - close_fee
        # The leading emoji reflects the actual money outcome, NOT the exit
        # reason: a time-stop / forced close that LOST must never wear a green
        # ✅. WIN = net > 0, LOSS = net < 0, scratch ≈ 0. The reason is still
        # named in the title so you can tell a TP from a time-stop.
        if net > 0.0001:
            emoji = "🟢"          # WIN
        elif net < -0.0001:
            emoji = "🔴"          # LOSS
        else:
            emoji = "⚪"          # flat scratch
        # Name the exit. A "tp"/"sl" label is only honest now that _resolve_close
        # is PnL-aware (a profitable close is tp, a losing close is sl); a
        # time-stop/forced close keeps its own descriptive label.
        _REASON_TITLE = {
            "tp": "TP", "sl": "SL", "sl_ambiguous": "SL (worst case)",
            "time_stop": "TIME STOP", "trend_break": "TREND EXIT",
            "manual": "MANUAL CLOSE", "watch_timeout": "FORCED CLOSE",
            "unmatched_close": "FORCED CLOSE", "closed": "CLOSED",
        }
        outcome = "WIN" if net > 0.0001 else ("LOSS" if net < -0.0001 else "SCRATCH")
        label = _REASON_TITLE.get(reason, reason.upper())
        title = label if is_manual else f"{outcome} · {_esc(label)}"
        g_str = f"+${abs(gross):.4f}" if gross >= 0 else f"-${abs(gross):.4f}"
        n_str = f"+${abs(net):.4f}" if net >= 0 else f"-${abs(net):.4f}"
        # Per-side bps slip vs the paper TP/SL (positive = we got a worse fill).
        # Only meaningful for genuine TP/SL exits — a time-stop/trend-break
        # close has no intended price target, so comparing it to the (much
        # wider) SL printed a large, meaningless "slip" figure.
        intended_close = trade.tp_px if reason == "tp" else trade.sl_px
        if reason in ("tp", "sl") and intended_close > 0 and trade.close_px > 0:
            if trade.side == "long":
                slip = (intended_close - trade.close_px) / intended_close * 10_000
            else:
                slip = (trade.close_px - intended_close) / intended_close * 10_000
        else:
            slip = None
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
        if slip is not None:
            lines += [_SEP, f"Slip vs paper  `{slip:+.1f}bps`"]
        if trade.exit_path and trade.exit_path != "native_tp_sl":
            lines.append(
                f"Exit    `{_esc(trade.exit_path)}` · "
                f"repegs `{trade.exit_repegs}` · "
                f"ttf `{trade.exit_ttf_s:.1f}s`"
            )
            if trade.exit_adverse_bps:
                lines.append(f"Adverse `{trade.exit_adverse_bps:.1f}bps`")
        if session is not None:
            lines.extend(_journey_lines(session, cfg))
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
    live_executor: Optional[Any] = None,
    all_sessions: Optional[list] = None,
) -> str:
    _esc = TelegramNotifier.escape
    uptime = _fmt_uptime((datetime.now(tz=timezone.utc) - start_time).total_seconds())
    sym = cfg["exchange"]["symbol"]
    tf = cfg["exchange"]["timeframe"]
    status_str = "Halted" if session.halted else "Running"
    # The EXECUTOR can be hard-halted / daily-halted / cooling down / holding
    # the gate while the session looks fine — "✅ Running" during a 40% MDD
    # hard halt hid an idle bot indefinitely. Surface the real gate state.
    exec_note = ""
    if live_executor is not None:
        try:
            _br = live_executor._entry_blocked_reason()  # noqa: SLF001
        except Exception:  # noqa: BLE001
            _br = None
        _ot = getattr(live_executor, "_open_trade", None)
        if _br:
            exec_note = f"\n⛔ entries blocked: `{_esc(str(_br))}`"
        elif _ot is not None and (getattr(_ot, "extras", {}) or {}).get("still_open"):
            exec_note = "\n⏸ gate held: `close unverified — recovery running`"
    _j = _journey_lines(all_sessions or session, cfg)
    journey_block = ("\n".join(_j) + "\n") if _j else ""
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
        exit_brk = _format_exit_breakdown(rs.get("by_reason") or {})
        exit_line = f"Exits     `{_esc(exit_brk)}`\n" if exit_brk else ""
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
            f"{exit_line}"
            f"{_SEP}\n"
            f"{'🛑' if session.halted else '✅'} `{_esc(status_str)}`{exec_note}\n"
            f"{journey_block}"
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
        f"{'🛑' if session.halted else '✅'} `{_esc(status_str)}`{exec_note}\n"
        f"{journey_block}"
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
        if not acct.get("pos_ok", True):
            return (
                f"📈 *Position · {_esc(mode_label)}*\n"
                f"{_SEP}\n"
                f"Position state `unavailable` \\(positions API error\\) — check OKX\n"
                f"{_SEP}\n"
                f"\\[{_esc(bot_label)}\\]"
            )
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
                f"TP  `{tp:,.2f}`  \\(`{dist_tp:.1f}bps` away\\)",
                f"SL  `{sl:,.2f}`  \\(`{dist_sl:.1f}bps` away\\)",
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
        f"TP  `{pos.tp:,.2f}`  \\(`{dist_tp:.1f}bps` away\\)\n"
        f"SL  `{pos.sl:,.2f}`  \\(`{dist_sl:.1f}bps` away\\)\n"
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
    # post_only ALWAYS: the help text promises a maker close, but a plain
    # 'limit' at/through the touch executes instantly as TAKER (0.05% vs
    # 0.02%). post_only with an explicit crossing price is rejected with
    # sCode 51280 instead — the rejection card already tells the operator to
    # retry with a different price or use /close market, which is the honest
    # behavior for a button labeled "maker".
    ord_type = "post_only"
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
    symbols: Optional[list] = None,
    sessions: Optional[list] = None,
) -> tuple:
    """Resolve an action to ``(text, buttons)``. ``buttons`` is None for plain cards.

    Actions: status / position / fees / trades / orders / help / close /
    close_market / close_limit / cancel / dismiss. The position-management
    actions are live/demo only (no real OKX position exists in paper mode).
    """
    _esc = TelegramNotifier.escape
    if action == "roadmap":
        action = "plan"
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
        symbols=symbols, sessions=sessions,
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
    symbols: Optional[list] = None,
    sessions: Optional[list] = None,
) -> Optional[str]:
    # In LIVE/DEMO, every card is sourced from the REAL account + real-fill log,
    # fetched once here and threaded into the builders (no paper numbers leak).
    is_live = mode_label != "PAPER"
    multi = bool(symbols and len(symbols) > 1)
    # symbol=None → positions across ALL campaign instruments
    real_acct = await _fetch_real_account(client, None if multi else symbol) if is_live else None
    # Source the trade stats from OKX's own order history (exchange truth);
    # only if that API call fails do we fall back to the local jsonl log.
    real_stats = None
    if is_live:
        real_stats = (await _fetch_okx_trade_stats_multi(client, symbols)
                      if multi else await _fetch_okx_trade_stats(client, symbol))
        # Read the local jsonl ONCE (it grows for months over a campaign and
        # used to be parsed twice per command): it is both the fallback for
        # the headline stats and the only source of the exit-reason breakdown
        # (OKX history can't distinguish a time-stop from a market exit).
        local_stats = (
            _real_trade_stats(_read_live_trades(log_dir))
            if log_dir is not None else None
        )
        if real_stats is None:
            real_stats = local_stats
        elif local_stats is not None:
            real_stats["by_reason"] = local_stats.get("by_reason", {})
    if data == "cmd_status":
        return _build_status_text(
            session, cfg, mode_label, bot_label, start_time, real_acct, real_stats,
            live_executor=live_executor, all_sessions=sessions,
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
    if data == "cmd_plan":
        return _build_plan_text(session, cfg, bot_label, mode_label,
                                real_acct, real_stats, live_executor,
                                all_sessions=sessions)
    return None


def _build_plan_text(
    session: "VolumeFarmerSession",
    cfg: Dict[str, Any],
    bot_label: str,
    mode_label: str,
    real_acct: Optional[Dict[str, Any]],
    real_stats: Optional[Dict[str, Any]],
    live_executor: Optional[Any] = None,
    all_sessions: Optional[list] = None,
) -> str:
    """Roadmap-to-5M card: real progress, top-up needed vs the breaker, and the
    safety-net checklist — all from real numbers (or 'unavailable')."""
    _esc = TelegramNotifier.escape
    fees = cfg.get("fees", {}) or {}
    maker = float(fees.get("maker", 0.0002))
    rebate = float(fees.get("rebate_pct", 0.20))
    target = float((cfg.get("target", {}) or {}).get("volume_usd", 5_000_000))
    reward = float((cfg.get("target", {}) or {}).get("reward_usd", 1500))
    risk = cfg.get("risk", {}) or {}
    mdd = float(risk.get("live_breaker_max_drawdown_pct", 0.40))
    # net $/1M ≈ all-maker floor = maker×(1-rebate)×1e6, +5% for taker reality
    net_per_1m = maker * (1 - rebate) * 1e6 * 1.05
    gross_per_1m = maker * 1e6 * 1.05

    is_real = mode_label != "PAPER"
    sim_vol = sum(s_.total_volume_usd for s_ in (all_sessions or [session]))
    real_vol = float(real_stats.get("volume", 0.0)) if (is_real and real_stats) else sim_vol
    wallet = real_acct.get("equity") if (is_real and real_acct) else session.equity
    pct = real_vol / target * 100 if target else 0
    remaining = max(target - real_vol, 0.0)
    remaining_net = remaining / 1e6 * net_per_1m

    lines = [
        f"🗺 *Roadmap to 5M · {_esc(mode_label)}*",
        _SEP,
        f"Volume   `${real_vol:,.0f}` / `${target:,.0f}`  \\(`{pct:.1f}%`\\)",
        f"Cost so far  ≈ `${real_vol/1e6*net_per_1m:,.0f}` net",
        f"Remaining    ≈ `${remaining_net:,.0f}` net  \\(`${net_per_1m:.0f}/1M`\\)",
        _SEP,
    ]
    if wallet is not None:
        # The 40% MDD breaker halts when wallet < (1-mdd)×peak. Approximating
        # peak≈current wallet, the wallet can absorb mdd×wallet of bleed.
        absorb = mdd * float(wallet)
        topup = max(0.0, remaining_net - absorb)
        lines.append(f"Wallet   `${float(wallet):,.2f}`")
        if topup <= 0:
            lines.append(f"Top\\-up needed  `none` ✅  \\(wallet funds the rest\\)")
        else:
            lines.append(
                f"⚠️ Top\\-up needed  ≈ `${topup:,.0f}`  before the {int(mdd*100)}% "
                f"halt \\(at ≈ `${float(wallet)*(1-mdd):,.2f}`\\)"
            )
        lines.append(f"Reward at 5M  `${reward:,.0f}`  → net ≈ `${reward - remaining_net:,.0f}`")
    else:
        lines.append("Wallet   `unavailable` \\(API\\) — top\\-up calc paused")
    # pace / ETA from the session controller
    pace = (session.summary().get("pace") or {})
    if pace.get("achieved_per_day"):
        ach = float(pace["achieved_per_day"])
        if ach > 0:
            lines.append(f"Pace `${ach:,.0f}/day` → ETA ≈ `{remaining/ach:.1f}d`")
    lines += [
        _SEP,
        "*Safety nets*",
        f"• Hard halt at `{int(mdd*100)}%` drawdown \\(persisted\\)",
        f"• Daily pause at `{int(float(risk.get('live_breaker_daily_loss_pct',0.10))*100)}%`",
        "• 1\\-bar time\\-stop · maker exits · 15× \\(liq `6.2%` ≫ stop\\)",
        "• Naked\\-position flatten on restart · keep\\-alive supervisor",
        "• Top\\-up asks in active hours only · deposit alerts on",
        _SEP,
        f"\\[{_esc(bot_label)}\\]",
    ]
    return "\n".join(lines)


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
            f"Amount must be \\> 0\n"
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
    symbols: Optional[list] = None,
    sessions: Optional[list] = None,
) -> None:
    if not notifier.enabled:
        # CRITICAL: returning here used to end this task, which unblocked
        # asyncio.wait(FIRST_COMPLETED) in main_async and shut the WHOLE BOT
        # down seconds after startup whenever Telegram creds were missing or
        # --no-telegram was passed (a 15s systemd crash loop). Without
        # Telegram the bot must simply trade without a command channel.
        LOGGER.info("telegram disabled — command loop idle (bot keeps running)")
        await stop_evt.wait()
        return
    _esc = TelegramNotifier.escape

    # Drain the update backlog BEFORE acting on anything. Telegram retains
    # up to 24h of updates; starting at offset=0 used to replay commands sent
    # while the bot was down — including a stale /close tap firing a real
    # market close on a brand-new position the moment the bot restarted.
    offset = 0
    try:
        discarded = 0
        # Bounded: each iteration returns up to 100 updates, so 30 rounds
        # clears any realistic 24h backlog; the bound stops a pathological
        # spammer (or a buggy peer) from pinning startup in this loop.
        for _ in range(30):
            stale = await notifier.get_updates(offset=offset, timeout=0)
            if not stale:
                break
            for upd in stale:
                uid = upd.get("update_id") if isinstance(upd, dict) else None
                if uid is not None:
                    offset = uid + 1
                    discarded += 1
        if discarded:
            LOGGER.info("discarded %d stale Telegram update(s) from before startup", discarded)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("startup backlog drain failed (continuing): %s", exc)

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
        symbols=symbols, sessions=sessions,
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
            # Tell the operator the command failed — silence after tapping a
            # close button is indistinguishable from "it worked".
            try:
                await _reply(
                    f"⚠️ */{_esc(action)} failed* — `{_esc(str(exc)[:150])}`\n"
                    f"{_SEP}\n\\[{_esc(bot_label)}\\]",
                    None, reply_to,
                )
            except Exception:  # noqa: BLE001
                pass
        finally:
            if guarded:
                _inflight.discard("close")

    def _spawn(coro) -> None:
        t = asyncio.create_task(coro)
        _tasks.add(t)
        t.add_done_callback(_tasks.discard)

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
                # Thread the reply under the message whose button was tapped
                # (e.g. an EXIT-FAILED card), not the day-old Controls menu.
                tapped_msg_id = (cq.get("message") or {}).get("message_id") or menu_msg_id
                _spawn(_dispatch(action, "", tapped_msg_id))

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
                    "orders", "cancel", "help", "plan", "roadmap",
                ):
                    _spawn(_dispatch(cmd, raw_text, reply_to))


async def _fees_last_24h(client: Any, symbol: str) -> Optional[float]:
    """Sum of |fee| over FILLED orders in the last 24h — straight from OKX
    order history, no local bookkeeping. Returns None when the API is
    unavailable so callers can say "unavailable" instead of a fake $0."""
    inst = to_okx_inst_id(symbol)
    cutoff_ms = int((time.time() - 86_400) * 1000)
    total = 0.0
    after = ""
    ok = False
    reached_cutoff = False
    try:
        for _ in range(10):                      # ≤1,000 orders, newest first
            params = {"instType": "SWAP", "instId": inst, "limit": "100"}
            if after:
                params["after"] = after
            r = await client._request(  # noqa: SLF001
                "GET", "/api/v5/trade/orders-history", params=params, auth=True,
            )
            if str(r.get("code")) != "0":
                break
            ok = True
            data = r.get("data") or []
            if not data:
                reached_cutoff = True
                break
            page_older_than_cutoff = False
            for o in data:
                # Canceled orders with partial fills carry REAL fees — the
                # maker-exit ladder produces these every day; skipping them
                # silently under-counted the rebate ask.
                try:
                    acc = float(o.get("accFillSz") or 0.0)
                except (TypeError, ValueError):
                    acc = 0.0
                if str(o.get("state", "")) != "filled" and acc <= 0:
                    continue
                ft = int(o.get("fillTime") or 0)
                if ft and ft < cutoff_ms:
                    page_older_than_cutoff = True
                    continue
                try:
                    # OKX: fees paid are NEGATIVE; a positive fee is a rebate
                    # credit and must not be counted as cost.
                    fee = float(o.get("fee") or 0.0)
                except (TypeError, ValueError):
                    continue
                total += max(-fee, 0.0)
            after = data[-1].get("ordId") or ""
            if page_older_than_cutoff:
                reached_cutoff = True
                break
            if not after or len(data) < 100:
                reached_cutoff = True
                break
    except Exception as exc:  # noqa: BLE001
        # A mid-pagination failure would return a PARTIAL sum as fact.
        LOGGER.warning("24h fee fetch failed: %s", exc)
        return None
    if not ok:
        return None
    if not reached_cutoff:
        # Page cap exhausted before reaching the 24h boundary: the sum is a
        # lower bound, not a fact — report unavailable rather than undercount.
        LOGGER.warning("24h fee window did not reach cutoff within page cap")
        return None
    return total


async def _verify_fee_tier(client: Any, symbol: str, cfg: Dict[str, Any]) -> str:
    """Compare the config's assumed maker/taker rates against the account's
    ACTUAL tier from GET /account/trade-fee. The entire campaign cost model
    (TP fee floor, cost-per-1M, rebate estimates) rests on these two numbers —
    they must come from the exchange, not a yaml comment.

    Returns a MarkdownV2-safe note for the startup card: verified, mismatch
    warning, or unavailable.
    """
    try:
        row = await client.get_trade_fee(symbol)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("fee-tier verification unavailable: %s", exc)
        return "fee tier: `unverified` \\(API unavailable\\)"

    def _rate(*keys: str) -> Optional[float]:
        for k in keys:
            v = row.get(k)
            if v not in (None, ""):
                try:
                    return abs(float(v))      # OKX reports paid fees negative
                except (TypeError, ValueError):
                    continue
        return None

    real_maker = _rate("makerU", "maker")
    real_taker = _rate("takerU", "taker")
    cfg_m = float(cfg["fees"]["maker"])
    cfg_t = float(cfg["fees"]["taker"])
    LOGGER.info("OKX ACTUAL fee tier: maker=%s taker=%s (config assumes %s/%s)",
                real_maker, real_taker, cfg_m, cfg_t)
    if real_maker is None or real_taker is None:
        return "fee tier: `unverified` \\(no rate in response\\)"
    if abs(real_maker - cfg_m) <= cfg_m * 0.05 and abs(real_taker - cfg_t) <= cfg_t * 0.05:
        return f"fee tier: `verified` \\(`{real_maker*1e4:.1f}`/`{real_taker*1e4:.1f}bps` live API\\)"
    LOGGER.error("FEE TIER MISMATCH: config %s/%s but account pays %s/%s — "
                 "the cost model is wrong until the config matches",
                 cfg_m, cfg_t, real_maker, real_taker)
    return (f"⚠️ fee tier MISMATCH: config `{cfg_m*1e4:.1f}`/`{cfg_t*1e4:.1f}bps` "
            f"but account pays `{real_maker*1e4:.1f}`/`{real_taker*1e4:.1f}bps`")


def _in_active_hours(tg_cfg: Dict[str, Any]) -> bool:
    """True when local time is within the team's waking window — top-up asks
    sent at 3am get missed, so the lifeline reminders are queued to the next
    active hour (the ops loop re-checks hourly). Defaults 08:00-22:00 in the
    bot's timezone (BOT_TIMEZONE / Asia/Colombo)."""
    ah = (tg_cfg.get("active_hours") or {})
    start = int(ah.get("start_hour", 8))
    end = int(ah.get("end_hour", 22))
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(os.environ.get("BOT_TIMEZONE", "Asia/Colombo"))
        hr = datetime.now(tz=tz).hour
    except Exception:  # noqa: BLE001
        hr = datetime.now(tz=timezone.utc).hour
    return start <= hr < end if start <= end else (hr >= start or hr < end)


async def _live_ops_loop(
    notifier: Optional[TelegramNotifier],
    client: Any,
    symbol: str,
    cfg: Dict[str, Any],
    mode_label: str,
    bot_label: str,
    stop_evt: asyncio.Event,
    symbols: Optional[list] = None,
) -> None:
    """LIVE/DEMO operations watchdog — every number from the REAL exchange.

    Replaces the simulation-triggered rebate reminder in real modes (the sim's
    paper equity drifts away from the actual wallet, so its triggers were not
    "real facts"). Hourly:
      * fetch the actual wallet balance (anchored at startup for the floor),
      * fetch the actual fees paid in the last 24h from order history,
      * daily: ping the mention list to recycle ~24h × rebate%  back into the
        wallet (per-day ask — the old card re-asked the LIFETIME rebate every
        time),
      * urgent (6h cooldown): wallet under floor% of the anchor — at a small
        starting balance the rebate recycle is the campaign's lifeline, so
        this fires loudly with the configured @handles.
    """
    if not (notifier and notifier.enabled) or mode_label == "PAPER":
        await stop_evt.wait()
        return
    tg_cfg = (cfg.get("notifications", {}) or {}).get("telegram", {}) or {}
    rr = tg_cfg.get("rebate_reminder", {}) or {}
    if not bool(rr.get("enabled", True)):
        await stop_evt.wait()
        return
    floor_pct = float(rr.get("equity_floor_pct", 0.85))
    min_usd = float(rr.get("min_rebate_usd", 5.0))
    rebate_pct = float((cfg.get("fees", {}) or {}).get("rebate_pct", 0.40))
    handles = list(rr.get("mention_handles", []) or [])
    if "@rowneth" not in handles:
        handles.insert(0, "@rowneth")
    _esc = TelegramNotifier.escape
    # Persisted ops state: the floor ANCHOR must survive restarts — the
    # keep-alive supervisor restarts after crashes, and crashes correlate
    # with drawdowns; re-anchoring at the depleted wallet ratcheted the floor
    # down 15% per crash and muted the campaign's lifeline alert. The daily
    # dedupe timestamp also persists so each boot doesn't re-ping the team.
    ops_path = pathlib.Path("data") / f"ops_state_{bot_label}.json"
    anchor: Optional[float] = None
    daily_last_sent = 0.0
    last_wallet: Optional[float] = None
    try:
        if ops_path.exists():
            _ops = json.loads(ops_path.read_text())
            anchor = _ops.get("anchor")
            daily_last_sent = float(_ops.get("daily_last_sent", 0.0))
            last_wallet = _ops.get("last_wallet")
            LOGGER.info("ops state restored: anchor=%s last_wallet=%s", anchor, last_wallet)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("ops state load failed: %s", exc)

    def _save_ops() -> None:
        try:
            ops_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = ops_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(
                {"anchor": anchor, "daily_last_sent": daily_last_sent,
                 "last_wallet": last_wallet}))
            os.replace(tmp, ops_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("ops state save failed: %s", exc)

    low_bal_last = 0.0
    while not stop_evt.is_set():
        low_balance = False
        try:
            acct = await _fetch_real_account(client, symbol)
            wallet = acct.get("equity")
            if wallet is not None and anchor is None:
                anchor = float(wallet)
                LOGGER.info("ops: campaign wallet anchor $%.4f (floor $%.4f)",
                            anchor, anchor * floor_pct)
                _save_ops()
            elif wallet is not None and anchor is not None and wallet > anchor:
                # Top-ups / growth raise the anchor (floor follows the best
                # wallet ever seen, never ratchets down).
                anchor = float(wallet)
                _save_ops()
            # Deposit / reload detection — REAL balance jump vs the last
            # observed wallet (rebate transfer landing, manual top-up).
            # Threshold $5 keeps ordinary hourly PnL wiggle from pinging.
            if (wallet is not None and last_wallet is not None
                    and (wallet - last_wallet) >= 5.0):
                inc = wallet - last_wallet
                try:
                    await notifier.send_raw(
                        f"{_esc('@rowneth')}\n"
                        f"📥 *Wallet reloaded · {_esc(mode_label)}*\n{_SEP}\n"
                        f"Balance  `${last_wallet:,.4f}` → `${wallet:,.4f}`"
                        f"  \\(\\+`${inc:,.2f}`\\)\n"
                        f"Deposit received — campaign runway extended\\.\n"
                        f"{_SEP}\n\\[{_esc(bot_label)}\\]"
                    )
                    LOGGER.info("deposit detected: +$%.2f (%.4f -> %.4f)",
                                inc, last_wallet, wallet)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("deposit alert send failed: %s", exc)
            if wallet is not None:
                last_wallet = float(wallet)
                _save_ops()
            fees24: Optional[float] = 0.0
            for _sym in (symbols or [symbol]):
                _f = await _fees_last_24h(client, _sym)
                if _f is None:
                    fees24 = None
                    break
                fees24 += _f
            rebate24 = fees24 * rebate_pct if fees24 is not None else None
            low_balance = bool(
                wallet is not None and anchor is not None
                and wallet < anchor * floor_pct
            )
            daily_due = (time.time() - daily_last_sent >= 86_400
                         and rebate24 is not None and rebate24 >= min_usd)
            urgent_due = low_balance and (time.time() - low_bal_last) > 6 * 3600
            # Hold the team-mention alerts to active hours so they're not sent
            # while everyone's asleep (and missed). Re-checked hourly; the
            # dedupe/cooldown only advance on an actual send, so a reminder
            # that comes due overnight fires at the next active hour instead.
            if (daily_due or urgent_due) and not _in_active_hours(tg_cfg):
                LOGGER.info("rebate/topup alert due but outside active hours — holding")
                daily_due = urgent_due = False
            if daily_due or urgent_due:
                mention = " ".join(_esc(h) for h in handles)
                w_str = f"${wallet:,.4f}" if wallet is not None else "unavailable"
                f24_str = f"${fees24:,.4f}" if fees24 is not None else "unavailable"
                r24_str = f"${rebate24:,.2f}" if rebate24 is not None else "unavailable"
                if urgent_due:
                    head = f"🚨 *WALLET BELOW FLOOR · {_esc(mode_label)}*"
                    anchor_str = f"${anchor:,.4f}" if anchor is not None else "—"
                    floor_str = f"${anchor*floor_pct:,.4f}" if anchor is not None else "—"
                    body = (
                        f"Wallet now   `{w_str}`  \\(floor `{floor_str}` of start `{anchor_str}`\\)\n"
                        f"Fees 24h     `{f24_str}`  \\(real OKX history\\)\n"
                        f"Rebate 24h   `{r24_str}`  \\(est {int(rebate_pct*100)}%\\)\n"
                        f"{_SEP}\n"
                        f"*Action needed:* transfer accumulated rebates "
                        f"\\(and top up if available\\) — sizing shrinks with the wallet "
                        f"and the campaign stalls below the floor\\."
                    )
                else:
                    head = f"💸 *Daily rebate recycle · {_esc(mode_label)}*"
                    body = (
                        f"Wallet now   `{w_str}`\n"
                        f"Fees 24h     `{f24_str}`  \\(real OKX history\\)\n"
                        f"Rebate 24h   `{r24_str}`  \\(est {int(rebate_pct*100)}%\\)\n"
                        f"{_SEP}\n"
                        f"Please transfer `{r24_str}` back to this wallet to keep "
                        f"the campaign self\\-funding\\."
                    )
                await notifier.send_raw(
                    f"{mention}\n{head}\n{_SEP}\n{body}\n{_SEP}\n\\[{_esc(bot_label)}\\]"
                )
                if daily_due or urgent_due:
                    # the urgent card carries the same transfer ask — sending
                    # both within minutes double-asked for the same rebate
                    daily_last_sent = time.time()
                    _save_ops()
                if urgent_due:
                    low_bal_last = time.time()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("live ops loop error: %s", exc)
        try:
            await asyncio.wait_for(stop_evt.wait(),
                                   timeout=1800.0 if low_balance else 3600.0)
            return
        except asyncio.TimeoutError:
            pass


async def _target_watcher(
    notifier: Optional[TelegramNotifier],
    client: Any,
    symbols: list,
    cfg: Dict[str, Any],
    mode_label: str,
    bot_label: str,
    stop_evt: asyncio.Event,
) -> None:
    """Stop the campaign when the COMBINED real exchange volume crosses the
    target (+buffer). With multiple instruments no single session's counter is
    authoritative — only the summed OKX order history is."""
    if mode_label == "PAPER":
        await stop_evt.wait()
        return
    tgt = float((cfg.get("target", {}) or {}).get("volume_usd", 0) or 0)
    buf = float((cfg.get("target", {}) or {}).get("volume_buffer_pct", 0.02))
    if tgt <= 0:
        await stop_evt.wait()
        return
    _esc = TelegramNotifier.escape
    while not stop_evt.is_set():
        try:
            st = await _fetch_okx_trade_stats_multi(client, symbols)
            vol = float(st.get("volume", 0.0)) if st else None
            if vol is not None and vol >= tgt * (1.0 + buf):
                LOGGER.warning("TARGET REACHED: combined real volume $%.0f >= $%.0f — stopping", vol, tgt)
                if notifier and notifier.enabled:
                    try:
                        await notifier.send_and_get_id(
                            f"🏁 *TARGET REACHED · {_esc(mode_label)}*\n{_SEP}\n"
                            f"Combined real volume `${vol:,.0f}` ≥ `${tgt:,.0f}` "
                            f"\\(\\+{buf*100:.0f}% buffer\\)\n"
                            f"Campaign complete — stopping the bot\\. Claim the reward\\!\n"
                            f"{_SEP}\n\\[{_esc(bot_label)}\\]"
                        )
                    except Exception:  # noqa: BLE001
                        pass
                stop_evt.set()
                return
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("target watcher error: %s", exc)
        try:
            await asyncio.wait_for(stop_evt.wait(), timeout=600.0)
            return
        except asyncio.TimeoutError:
            pass


async def _daily_report_loop(
    notifier: Optional[TelegramNotifier],
    session: "VolumeFarmerSession",
    cfg: Dict[str, Any],
    mode_label: str,
    bot_label: str,
    stop_evt: asyncio.Event,
    client: Any = None,
    symbol_for_report: str = "BTC_USDT",
    symbols: Optional[list] = None,
    sessions: Optional[list] = None,
) -> None:
    """Send the daily digest at ``notifications.telegram.daily_report_hour`` UTC.

    The ``send_daily`` / ``daily_report_hour`` knobs existed in both configs
    but were never read by the runner — operators believed they had configured
    daily digests that never arrived. One digest per UTC day: session stats,
    fee/rebate totals and the campaign journey block.
    """
    tg_cfg = (cfg.get("notifications", {}) or {}).get("telegram", {}) or {}
    if not (notifier and notifier.enabled and bool(tg_cfg.get("send_daily", False))):
        await stop_evt.wait()
        return
    hour = int(tg_cfg.get("daily_report_hour", 0)) % 24
    _esc = TelegramNotifier.escape
    is_real = mode_label != "PAPER"
    sent_for = ""
    while not stop_evt.is_set():
        now = datetime.now(tz=timezone.utc)
        today_key = now.strftime("%Y-%m-%d")
        if now.hour == hour and sent_for != today_key:
            sent_for = today_key
            try:
                lines = [f"🗞 *Daily report · {_esc(mode_label)}*", _SEP]
                if is_real and client is not None:
                    # REAL numbers only: actual wallet + exchange order-history
                    # stats + real 24h fees. The sim digest under a LIVE header
                    # read as the real account's daily P&L — exactly the
                    # fake-fact class this bot must never emit.
                    _multi = bool(symbols and len(symbols) > 1)
                    acct = await _fetch_real_account(
                        client, None if _multi else symbol_for_report)
                    stats = (await _fetch_okx_trade_stats_multi(client, symbols)
                             if _multi else
                             await _fetch_okx_trade_stats(client, symbol_for_report))
                    w = acct.get("equity")
                    lines.append(f"Wallet  `${w:,.4f}`" if w is not None
                                 else "Wallet  `unavailable`")
                    if stats:
                        net = float(stats.get("net", 0.0))
                        n_str = f"+${net:.4f}" if net >= 0 else f"-${abs(net):.4f}"
                        lines += [
                            f"Real trades  `{stats.get('n', 0)}`  ·  WR `{stats.get('wr', 0.0):.1f}%`",
                            f"Real volume  `${stats.get('volume', 0.0):,.0f}`",
                            f"Net PnL  `{n_str}`  ·  fees `${stats.get('fees', 0.0):.4f}`",
                        ]
                    else:
                        lines.append("Trade stats  `unavailable` \\(OKX history error\\)")
                    fees24: Optional[float] = 0.0
                    for _sym in (symbols or [symbol_for_report]):
                        _f = await _fees_last_24h(client, _sym)
                        if _f is None:
                            fees24 = None   # partial data must not look total
                            break
                        fees24 += _f
                    if fees24 is not None:
                        reb = fees24 * float((cfg.get("fees", {}) or {}).get("rebate_pct", 0.40))
                        lines.append(f"Fees 24h  `${fees24:,.4f}`  ·  rebate est `${reb:,.2f}`")
                else:
                    s_sum = session.summary()
                    pnl = float(s_sum.get("total_pnl") or 0.0)
                    pnl_str = f"+${pnl:.4f}" if pnl >= 0 else f"-${abs(pnl):.4f}"
                    lines += [
                        f"Trades  `{s_sum['round_trips']}`  ·  WR `{s_sum['win_rate_pct']:.1f}%`",
                        f"Equity  `${s_sum['equity']:.4f}`  ·  PnL `{pnl_str}`",
                        f"Fees    `${s_sum['fees_gross']:.4f}` gross  ·  `${s_sum['fees_net']:.4f}` net",
                        f"Rebate  accrued `${s_sum['rebate_accrued']:.4f}`  ·  available `${s_sum['rebate_available']:.4f}`",
                    ]
                lines.extend(_journey_lines(sessions or session, cfg))
                lines += [_SEP, f"\\[{_esc(bot_label)}\\]"]
                await notifier.send_raw("\n".join(lines))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("daily report failed: %s", exc)
        try:
            await asyncio.wait_for(stop_evt.wait(), timeout=300)
            return
        except asyncio.TimeoutError:
            pass


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
        # Clock sanity: OKX rejects signatures past ~30s of drift (code 50102),
        # which would break every private call — including managing an OPEN
        # position. Check once at startup and warn loudly.
        try:
            srv_ms = await client.get_server_time_ms()
            drift_s = abs(srv_ms / 1000.0 - time.time())
            if drift_s > 10.0:
                LOGGER.error(
                    "LOCAL CLOCK DRIFT %.1fs vs OKX — private calls will fail "
                    "past ~30s. Fix NTP on this host NOW.", drift_s,
                )
                if notifier.enabled:
                    _e_drift = TelegramNotifier.escape
                    _bg(notifier.send_raw(
                        f"⏰ *Clock drift warning*\n{_SEP}\n"
                        f"Local clock is `{drift_s:.1f}s` off OKX server time\\. "
                        f"Past \\~30s every signed call fails — fix NTP\\.\n"
                        f"{_SEP}\n\\[{_e_drift(args.label or 'okx-paper')}\\]"
                    ))
            else:
                LOGGER.info("clock drift vs OKX: %.2fs (ok)", drift_s)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("server time check failed (continuing): %s", exc)

        live_executor: Optional[LiveVolumeExecutorOKX] = None
        if is_live_or_demo:
            farmer_cfg = cfg.get("farmer", {}) or {}
            ts_cfg = farmer_cfg.get("time_stop", {}) or {}
            mx_cfg = farmer_cfg.get("maker_exit", {}) or {}
            er_cfg = farmer_cfg.get("entry_repeg", {}) or {}
            risk_cfg = cfg.get("risk", {}) or {}
            tf_seconds = _parse_timeframe_seconds(tf)
            # DEMO realism overlay — constructed ONLY in simulated mode so it can
            # never reach --live real-money execution (passes None when live).
            rl_cfg = cfg.get("realism", {}) or {}
            rl_entry = rl_cfg.get("entry", {}) or {}
            rl_exit = rl_cfg.get("exit", {}) or {}
            demo_realism = None
            if client_simulated and bool(rl_cfg.get("enabled", True)):
                demo_realism = DemoRealismConfig(
                    enabled=True,
                    seed=rl_cfg.get("seed"),
                    entry_fill_prob=float(rl_entry.get("fill_prob", 0.85)),
                    entry_slip_bps=float(rl_entry.get("slip_bps", 0.4)),
                    entry_taker_fee_mult=float(rl_entry.get("taker_fee_mult", 2.5)),
                    sl_slip_bps=float(rl_exit.get("sl_slip_bps", 3.0)),
                    taker_exit_fee_mult=float(rl_exit.get("taker_exit_fee_mult", 2.5)),
                    tp_fill_prob=float(rl_exit.get("tp_fill_prob", 0.90)),
                    tp_miss_giveback_bps=float(rl_exit.get("tp_miss_giveback_bps", 6.0)),
                )
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
                # Cap the sizing base (e.g. run a $500 account on a larger demo wallet).
                working_capital_usd=float(farmer_cfg.get("working_capital_usd", 0.0)),
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
                    # Ladder knobs were previously NOT plumbed from yaml — the
                    # dataclass defaults always applied silently.
                    ladder=bool(mx_cfg.get("ladder", True)),
                    ladder_max_frac=float(mx_cfg.get("ladder_max_frac", 1.0)),
                    ladder_start_frac=float(mx_cfg.get("ladder_start_frac", 0.0)),
                ),
                entry_repeg_cfg=EntryRepegConfig(
                    enabled=bool(er_cfg.get("enabled", True)),
                    repeg_ms=int(er_cfg.get("repeg_ms", 8_000)),
                    max_repegs=int(er_cfg.get("max_repegs", 6)),
                    max_entry_seconds=float(er_cfg.get("max_entry_seconds", 60.0)),
                    poll_ms=int(er_cfg.get("poll_ms", 150)),
                    taker_fallback=bool(er_cfg.get("taker_fallback", False)),
                    max_chase_bps=float(er_cfg.get("max_chase_bps", 0.0)),
                    use_amend=bool(er_cfg.get("use_amend", True)),
                ),
                # REAL-wallet circuit breaker — gates live entries on the actual
                # OKX balance (not the simulation's paper equity). Reads its OWN
                # keys (live_breaker_*) so the session's paper halts can be left
                # loose (they exit the process; the breaker only pauses entries
                # and re-baselines its peak each restart, so no stale-drawdown
                # treadmill). Falls back to the legacy keys if not set.
                breaker_consec_loss_limit=int(risk_cfg.get("consecutive_losses_limit", 3)),
                # Which close reasons count toward the loss streak. Default: only
                # real stop-loss hits — time_stop scratches / manual / trend_break
                # losses are ignored so small non-SL exits don't pause entries.
                breaker_loss_streak_reasons=frozenset(
                    str(r).lower()
                    for r in (risk_cfg.get("consecutive_losses_count_reasons") or ["sl"])
                ),
                breaker_cooldown_s=float(risk_cfg.get("consecutive_losses_cooldown_bars", 12)) * tf_seconds,
                breaker_daily_loss_pct=float(risk_cfg.get(
                    "live_breaker_daily_loss_pct", risk_cfg.get("daily_loss_limit_pct", 0.05))),
                breaker_max_drawdown_pct=float(risk_cfg.get(
                    "live_breaker_max_drawdown_pct", risk_cfg.get("max_drawdown_pct", 0.10))),
                demo_realism=demo_realism,
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
            # Exchange leverage MUST equal the session's sizing leverage cap so
            # that locked margin = margin_fraction and the real liquidation price
            # matches the SL-safety analysis. Drive it from config sizing.max_leverage
            # (single source of truth); --live-leverage is only an optional upper clamp.
            _cfg_max_lev = int(cfg.get("farmer", {}).get("sizing", {}).get("max_leverage", args.live_leverage or 50))
            _exchange_lev = min(_cfg_max_lev, args.live_leverage) if args.live_leverage else _cfg_max_lev
            try:
                await live_executor.initialize(leverage_cap=_exchange_lev)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("live executor init failed: %s", exc)
                return 2

            # PRE-FLIGHT AUTH GATE: prove the keys actually authenticate against
            # the chosen endpoint BEFORE the loops start placing orders. A demo
            # key on --live (code 50119), an un-whitelisted droplet IP (50110),
            # or clock drift (50102) otherwise let the bot churn failed entries
            # forever, posting confusing "Entry missed" cards and fabricating
            # progress from stale logs. Fail loud and fast instead.
            try:
                _bal = await client.get_balance("USDT")
                _acode = str(_bal.get("code", "?"))
            except Exception as exc:  # noqa: BLE001
                _acode, _bal = "EXC", {}
                LOGGER.error("pre-flight balance call raised: %s", exc)
                _bal = {"msg": str(exc)[:160]}
            if _acode != "0":
                _causes = {
                    "50119": "API key doesn't exist — a DEMO key is set in OKX_API_KEY, "
                             "or the key was deleted. Create a real-trading key.",
                    "50110": "This server's IP is not whitelisted on the API key.",
                    "50102": "Clock drift > 30s — fix NTP on the host.",
                    "50101": "Wrong passphrase for this key.",
                }
                _why = _causes.get(_acode, _bal.get("msg", "see OKX docs"))
                LOGGER.error("PRE-FLIGHT AUTH FAILED (%s): %s — NOT starting the trade loop", _acode, _why)
                if notifier.enabled:
                    _e = TelegramNotifier.escape
                    try:
                        await notifier.send_and_get_id(
                            f"⛔ *LIVE AUTH FAILED — not trading · {_e(mode_label)}*\n{_SEP}\n"
                            f"OKX rejected the keys: code `{_e(_acode)}`\n"
                            f"`{_e(_why)}`\n{_SEP}\n"
                            f"Fix the API key in `.env` \\(real\\-trading key, IP whitelisted\\), "
                            f"then redeploy\\. Run `scripts/diag_live.py` to confirm `code=0`\\.\n"
                            f"{_SEP}\n\\[{_e(args.label or 'okx-paper')}\\]"
                        )
                        await notifier.stop()
                    except Exception:  # noqa: BLE001
                        pass
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
                client=client, session=session, cfg=cfg,
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

                # Abandoned entries are no longer silent: each miss forfeits
                # 2× notional of counted volume, and over a month a hidden
                # fill-miss rate is the campaign's main volume leak. One
                # compact card per miss with a running counter.
                _abandon_count = {"n": 0}
                _esc_ab = TelegramNotifier.escape

                async def _on_entry_abandoned(trade) -> None:
                    _abandon_count["n"] += 1
                    lost_vol = 2.0 * float(getattr(trade, "notional_usd", 0.0) or 0.0)
                    reason_ab = str(getattr(trade, "cancel_reason", "") or "no fill")
                    try:
                        await notifier.send_raw(
                            f"🪂 *Entry missed \\#{_abandon_count['n']} · {_esc_ab(_exec_mode)}*\n"
                            f"{_SEP}\n"
                            f"Reason  `{_esc_ab(reason_ab[:120])}`\n"
                            f"Volume forfeited  `${lost_vol:,.0f}`\n"
                            f"{_SEP}\n\\[{_esc_ab(_exec_lbl)}\\]"
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning("abandoned-entry telegram send failed: %s", exc)

                live_executor.on_entry_abandoned = _on_entry_abandoned

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

        # ---- SECONDARY INSTRUMENTS (e.g. DOGE alongside BTC) ----
        # Each gets its own session/executor/handler/poll loop; ONE position
        # at a time account-wide is enforced by the executors' shared peer
        # registry (primary checked first via the poll stagger).
        import copy as _copy
        from dataclasses import replace as _dc_replace
        sec_stacks: list = []
        all_sessions = [session]
        all_symbols = [symbol]
        for sec in (cfg.get("exchange", {}).get("secondary_symbols") or []):
            sym2 = str(sec.get("symbol", "")).strip()
            if not sym2 or sym2 == symbol:
                continue
            ov = sec.get("overrides", {}) or {}
            cfg2 = _copy.deepcopy(cfg)
            if "tp_mult" in ov:
                cfg2["farmer"].setdefault("atr", {})["tp_mult"] = float(ov["tp_mult"])
            if "max_bar_range_bps" in ov:
                cfg2["farmer"].setdefault("entry", {})["max_bar_range_bps"] = float(ov["max_bar_range_bps"])
            if "min_vol_ratio" in ov:
                cfg2["farmer"].setdefault("entry", {})["min_vol_ratio"] = float(ov["min_vol_ratio"])
            # The PRIMARY owns campaign pacing and the target halt; secondaries
            # trade at base size and never halt the process on their sim book.
            cfg2.setdefault("farmer", {}).setdefault("pace", {})["enabled"] = False
            cfg2.setdefault("risk", {})["stop_on_volume_target"] = False
            sess2 = VolumeFarmerSession(config=cfg2)
            state2 = PROJECT_ROOT / f"data/volume_farmer_okx_{args.label or 'paper'}_{sym2.lower()}_state.json"
            if state2.exists():
                try:
                    sess2.load_state(state2)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error("state load failed for %s (fresh): %s", sym2, exc)
            ex2 = None
            if live_executor is not None:
                ex2 = _dc_replace(
                    live_executor, symbol=sym2,
                    on_entry_filled=None, on_entry_abandoned=None,
                    on_close_confirmed=None, on_close_unverified=None,
                    on_real_fill=None, on_breaker=None,
                )
                await ex2.initialize(leverage_cap=_exchange_lev)
                ex2.on_entry_filled = _make_entry_filled_callback(
                    notifier, sym2, mode_label=_exec_mode, bot_label=_exec_lbl,
                    tf=tf, client=client,
                )
                ex2.on_close_confirmed = _make_real_fill_callback(
                    notifier, sym2, mode_label=_exec_mode, bot_label=_exec_lbl,
                    client=client, session=sess2, cfg=cfg2,
                )
                ex2.on_close_unverified = _make_close_unverified_callback(
                    notifier, sym2, mode_label=_exec_mode, bot_label=_exec_lbl,
                    client=client,
                )
                ex2.on_breaker = live_executor.on_breaker
                ex2.on_entry_abandoned = live_executor.on_entry_abandoned
            try:
                hist2 = await _seed_history(client, sym2, tf, SEED_CANDLES)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("seed failed for %s — skipping the instrument: %s", sym2, exc)
                continue
            if sess2.position is not None:
                sess2.reconcile_orphan_position(hist2[hist2["closed"]].copy())
            intrabar2: Dict[str, Any] = {}
            handler2 = _make_event_handler(log_dir, sym2, sess2, ex2,
                                           notifier=notifier, mode_label=mode_label,
                                           client=client, tf=tf,
                                           bot_label=args.label or "okx-paper",
                                           intrabar=intrabar2, state_path=state2,
                                           cfg=cfg2)
            sess2.event_callback = handler2
            sec_stacks.append(dict(symbol=sym2, session=sess2, executor=ex2,
                                   handler=handler2, intrabar=intrabar2,
                                   state_path=state2, history=hist2))
            all_sessions.append(sess2)
            all_symbols.append(sym2)
            LOGGER.info("secondary instrument armed: %s (tp_mult=%s max_range=%s)",
                        sym2, cfg2["farmer"]["atr"]["tp_mult"],
                        cfg2["farmer"]["entry"].get("max_bar_range_bps"))
        # Shared single-position gate: every executor sees every other.
        if live_executor is not None:
            peers = [live_executor] + [st["executor"] for st in sec_stacks if st["executor"] is not None]
            for e in peers:
                e.peer_executors = peers

        # STARTUP message.
        if notifier.enabled:
            _esc = TelegramNotifier.escape
            # Show the REAL OKX wallet in live/demo, not the hardcoded capital_usd.
            cap = cfg["farmer"]["capital_usd"]
            cap_label = "Capital"
            cap_str = f"${cap:,.2f}"
            if mode_label != "PAPER" and client is not None:
                _acct = await _fetch_real_account(client, symbol)
                cap_label = "Wallet "
                if _acct.get("equity") is not None:
                    cap_str = f"${float(_acct['equity']):,.2f}"
                else:
                    # Never print the yaml constant as if it were the real
                    # balance on the first card of a live run.
                    cap_str = "unavailable"
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
            # Show the leverage ACTUALLY set on the exchange (config cap clamped
            # by --live-leverage), not just the optional CLI clamp — the card
            # used to omit the applied setting entirely unless the flag was passed.
            okx_cap = _exchange_lev if is_live_or_demo else None
            okx_cap_line = f"  ·  OKX set `{okx_cap}×`" if okx_cap else ""
            vol_target = cfg.get("target", {}).get("volume_usd", 0)
            # Real-money guard: confirm the account's ACTUAL fee tier matches
            # the config the whole cost model is built on (live API, not yaml).
            fee_note = ""
            if mode_label != "PAPER" and client is not None:
                fee_note = await _verify_fee_tier(client, symbol, cfg) + "\n"
            startup_msg = (
                f"🚀 *vGen OKX · {_esc(mode_label)}*\n"
                f"{_esc(symbol)} · {_esc(tf)}\n"
                f"{_SEP}\n"
                f"{cap_label}    `{cap_str}`\n"
                f"Lev sizing `{min_lev}-{max_lev}×` \\(dynamic\\){okx_cap_line}\n"
                f"ATR filter `≥${atr_min:.0f}` \\(Wilder\\-14\\)\n"
                f"TP `{atr_tp_mult}×` ATR  ·  SL `{atr_sl_mult}×` ATR\n"
                f"Limit TP   `{'✓' if limit_tp else '✗'}` \\(maker fill\\)\n"
                f"{_SEP}\n"
                f"Maker  `{maker_bps:.0f}bps`  ·  rebate `{rebate_pct:.0f}%`\n"
                f"Taker  `{taker_bps:.0f}bps`\n"
                f"{fee_note}"
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
                    # CRITICAL SAFEGUARD: a pre-restart position may be NAKED —
                    # if the process died inside the time-stop window, its
                    # protective TP/SL algos were already cancelled, and nothing
                    # in this process watches it. Trading "around" it also lets
                    # new net-mode entries merge with it, corrupting every
                    # watcher. The strategy never wants to hold: flatten it for
                    # a clean slate (~3bps), sweep leftovers, and say so.
                    flatten_ok = False
                    try:
                        resp = await client.close_position_market(
                            symbol, mgn_mode="isolated", pos_side=None, auto_cxl=True,
                        )
                        cdata = (resp.get("data") or [{}])[0]
                        flatten_ok = str(cdata.get("sCode", "0")) == "0"
                        if live_executor is not None:
                            from execution.maker_exit import cancel_attached_algos as _caa
                            await _caa(client, symbol)
                        await asyncio.sleep(2.0)
                        acct2 = await _fetch_real_account(client, symbol)
                        flatten_ok = flatten_ok and not acct2.get("pos_sz")
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.error("orphan flatten failed: %s", exc)
                    lines = ["♻️ *Resuming · pre\\-restart position found on OKX*"]
                    lines.extend(_real_acct_lines(acct))
                    lines.append(f"{_SEP}")
                    if flatten_ok:
                        lines.append(_esc_resume(
                            "Flattened at market for a clean slate (a pre-restart "
                            "position may have lost its protective stops). "
                            "Fresh entries resume from the next signal."
                        ))
                    else:
                        lines.append(
                            "🚨 " + _esc_resume(
                                "COULD NOT FLATTEN — this position may have NO "
                                "stop attached. Close it on OKX now (/close market)."
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
        sec_poll_tasks = [
            asyncio.create_task(_poll_loop(
                client, st["session"], st["symbol"], tf, args.poll_seconds,
                st["history"], end_at, log_dir, st["handler"],
                notifier=notifier, mode_label=mode_label,
                bot_label=args.label or "okx-paper", cfg=cfg,
                intrabar=st["intrabar"], initial_delay=3.0 * (i + 1),
            ))
            for i, st in enumerate(sec_stacks)
        ]
        cmd_task = asyncio.create_task(_command_loop(
            notifier, session, client, symbol, tf, cfg,
            mode_label, bot_label=args.label or "okx-paper",
            stop_evt=stop_evt, start_time=start_time,
            state_path=state_path,
            log_dir=log_dir, live_executor=live_executor,
            pos_mode=args.pos_mode,
            symbols=all_symbols, sessions=all_sessions,
        ))
        daily_task = asyncio.create_task(_daily_report_loop(
            notifier, session, cfg, mode_label,
            args.label or "okx-paper", stop_evt,
            client=client, symbol_for_report=symbol,
            symbols=all_symbols, sessions=all_sessions,
        ))
        ops_task = asyncio.create_task(_live_ops_loop(
            notifier, client, symbol, cfg, mode_label,
            args.label or "okx-paper", stop_evt, symbols=all_symbols,
        ))
        target_task = asyncio.create_task(_target_watcher(
            notifier, client, all_symbols, cfg, mode_label,
            args.label or "okx-paper", stop_evt,
        ))
        stop_waiter = asyncio.create_task(stop_evt.wait())
        all_tasks = (poll_task, *sec_poll_tasks, cmd_task, daily_task,
                     ops_task, target_task, stop_waiter)
        crash_exc: Optional[BaseException] = None
        try:
            done, _ = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)
            # Capture a crashed task's exception WITHOUT re-raising here: the
            # old bare `await t` in finally re-raised immediately, skipping
            # BOTH the state save and any shutdown card — the bot died silent
            # and lost session state on every unhandled exception.
            for t in done:
                if t is stop_waiter:
                    continue
                if not t.cancelled() and t.exception() is not None:
                    crash_exc = t.exception()
                    break
        finally:
            for t in all_tasks:
                t.cancel()
            for t in all_tasks:
                try:
                    await t
                except asyncio.CancelledError:
                    pass
                except Exception:  # noqa: BLE001 — already captured above
                    pass

        if crash_exc is not None:
            LOGGER.error("FATAL: task crashed", exc_info=crash_exc)
            try:
                session.save_state(state_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("state save after crash failed: %s", exc)
            if notifier.enabled:
                _e = TelegramNotifier.escape
                try:
                    await notifier.send_and_get_id(
                        f"💥 *BOT CRASHED · {_e(mode_label)}*\n{_SEP}\n"
                        f"`{_e(type(crash_exc).__name__)}: {_e(str(crash_exc)[:200])}`\n"
                        f"State saved\\. Supervisor should restart the bot\\.\n"
                        f"{_SEP}\n\\[{_e(args.label or 'okx-paper')}\\]"
                    )
                    await notifier.stop()
                except Exception:  # noqa: BLE001
                    pass
            raise crash_exc

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
                exit_brk = _format_exit_breakdown(rs.get("by_reason") or {})
                exit_line = (
                    f"Exits  `{TelegramNotifier.escape(exit_brk)}`\n{_SEP}\n"
                    if exit_brk else f"{_SEP}\n"
                )
                await notifier.send_raw(
                    f"⏹ *vGen OKX stopped · {TelegramNotifier.escape(mode_label)}*\n"
                    f"{_SEP}\n"
                    f"Real trades  `{rs.get('n', 0)}`  ·  WR `{rs.get('wr', 0.0):.1f}%`\n"
                    f"{exit_line}"
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
    p.add_argument("--max-live-trades", type=int, default=100_000,
                   help="Hard cap on FILLED live/demo trades. The old default of "
                        "100 silently stopped all entries after ~1.3 days at the "
                        "observed pace — a direct blocker for the 5M campaign. "
                        "Keep it as a runaway backstop, not a working limit.")
    p.add_argument("--live-leverage", type=int, default=0,
                   help="Optional upper clamp on OKX exchange leverage. 0 (default) "
                        "= use config sizing.max_leverage (keeps locked margin = "
                        "margin_fraction and liquidation in sync with the SL-safety cap).")
    p.add_argument("--pos-mode", type=str, default="net", choices=["net", "hedge"],
                   help="OKX position mode (net = one-way, hedge = both sides)")
    p.add_argument("--no-telegram", action="store_true",
                   help="Disable Telegram notifications even if creds are set")
    return asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
