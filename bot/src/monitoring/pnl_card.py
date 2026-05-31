"""Generate a MEXC-style P&L sharing card as a PNG image.

Takes a list of historical position dicts from MEXCClient.get_history_positions()
and renders a styled dark-mode card with:
  - Total realized PnL (large, coloured)
  - Win rate badge
  - Stats grid (trades, win/loss, avg RR, best/worst)
  - Cumulative PnL equity curve
  - Time range footer

Also provides build_trade_card() for a single-trade MEXC-style share card
attached to each TP/SL/trend-break Telegram close notification.
"""

from __future__ import annotations

import io
import logging
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.image as mpimg

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

LOGGER = logging.getLogger(__name__)

# ── Colour palette (matches chart_generator.py dark theme) ──────────────────

_BG      = "#131722"
_PANEL   = "#1e222d"
_GREEN   = "#26a69a"
_RED     = "#ef5350"
_YELLOW  = "#ffd54f"
_PURPLE  = "#ce93d8"
_TEXT    = "#d1d4dc"
_SUBTEXT = "#787b86"
_BORDER  = "#2a2e39"

# Exit-type codes returned by MEXC
_EXIT_LABELS = {1: "TP", 2: "SL", 3: "Liq", 4: "Manual"}


# ── Public entry point ───────────────────────────────────────────────────────

def build_pnl_card(
    positions: List[Dict[str, Any]],
    symbol: str = "BTC_USDT",
    label: str = "",
) -> Optional[bytes]:
    """Render a P&L card PNG from a list of historical position dicts.

    Args:
        positions: Raw records from MEXCClient.get_history_positions() (the
                   ``data.resultList`` slice — already a flat list of dicts).
        symbol:    Symbol string used in the header.
        label:     Optional extra label shown in the footer (e.g. strategy name).

    Returns:
        PNG bytes, or None if positions is empty or rendering fails.
    """
    if not positions:
        LOGGER.warning("pnl_card: no positions to render")
        return None

    try:
        stats = _compute_stats(positions)
        return _render(stats, symbol, label)
    except Exception as exc:
        LOGGER.exception("pnl_card: render failed: %s", exc)
        return None


# ── Stats computation ────────────────────────────────────────────────────────

def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _compute_stats(positions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Derive all display values from the raw position list."""
    total_pnl    = 0.0
    total_fee    = 0.0
    wins         = 0
    losses       = 0
    best_pnl     = float("-inf")
    worst_pnl    = float("inf")
    cum_pnl: List[float] = []
    times: List[datetime] = []
    exit_counts: Dict[str, int] = {"TP": 0, "SL": 0, "Liq": 0, "Manual": 0}

    for pos in positions:
        realised = _as_float(pos.get("realised"))
        # MEXC returns fee as a negative number; totalFee is the absolute value
        fee      = _as_float(pos.get("totalFee") or abs(_as_float(pos.get("fee", 0))))
        net      = realised - fee

        total_pnl += net
        total_fee += fee
        cum_pnl.append(total_pnl)

        if net >= 0:
            wins += 1
        else:
            losses += 1

        best_pnl  = max(best_pnl,  net)
        worst_pnl = min(worst_pnl, net)

        # MEXC history_positions has no exitType — infer from net PnL direction
        if net >= 0:
            exit_counts["TP"] = exit_counts.get("TP", 0) + 1
        else:
            exit_counts["SL"] = exit_counts.get("SL", 0) + 1

        # MEXC uses updateTime as the close timestamp
        close_ms = _as_float(pos.get("updateTime") or pos.get("closeTime", 0))
        if close_ms > 0:
            times.append(datetime.fromtimestamp(close_ms / 1000, tz=timezone.utc))

    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    start_ts = min(times).strftime("%Y-%m-%d") if times else "—"
    end_ts   = max(times).strftime("%Y-%m-%d") if times else "—"

    # Max drawdown on cumulative PnL curve
    arr  = np.array(cum_pnl)
    peak = np.maximum.accumulate(arr)
    dd   = arr - peak
    max_dd = float(dd.min()) if len(dd) else 0.0

    return {
        "total_pnl":   total_pnl,
        "total_fee":   total_fee,
        "wins":        wins,
        "losses":      losses,
        "total_trades": total_trades,
        "win_rate":    win_rate,
        "best_pnl":    best_pnl if best_pnl != float("-inf") else 0.0,
        "worst_pnl":   worst_pnl if worst_pnl != float("inf") else 0.0,
        "cum_pnl":     cum_pnl,
        "times":       times,
        "start_ts":    start_ts,
        "end_ts":      end_ts,
        "exit_counts": exit_counts,
        "max_dd":      max_dd,
    }


# ── Renderer ─────────────────────────────────────────────────────────────────

def _render(stats: Dict[str, Any], symbol: str, extra_label: str) -> bytes:
    """Draw the card and return PNG bytes."""

    fig = plt.figure(figsize=(10, 6), facecolor=_BG)
    gs  = gridspec.GridSpec(
        2, 2,
        figure=fig,
        left=0.06, right=0.97,
        top=0.88,  bottom=0.16,
        wspace=0.35, hspace=0.55,
    )

    total_pnl   = stats["total_pnl"]
    win_rate    = stats["win_rate"]
    pnl_colour  = _GREEN if total_pnl >= 0 else _RED
    pnl_sign    = "+" if total_pnl >= 0 else ""

    # ── Header ───────────────────────────────────────────────────────
    sym_display = symbol.replace("_", "/")
    fig.text(
        0.06, 0.955, sym_display,
        color=_TEXT, fontsize=17, fontweight="bold", va="top",
    )
    fig.text(
        0.06, 0.915,
        f"{stats['start_ts']}  →  {stats['end_ts']}   ·   {stats['total_trades']} trades",
        color=_SUBTEXT, fontsize=9, va="top",
    )
    if extra_label:
        fig.text(0.97, 0.955, extra_label, color=_PURPLE, fontsize=9,
                 va="top", ha="right", fontstyle="italic")

    # MEXC-style big P&L
    fig.text(
        0.97, 0.955,
        f"{pnl_sign}{total_pnl:,.4f} USDT",
        color=pnl_colour, fontsize=20, fontweight="bold", va="top", ha="right",
    ) if not extra_label else fig.text(
        0.97, 0.915,
        f"{pnl_sign}{total_pnl:,.4f} USDT",
        color=pnl_colour, fontsize=20, fontweight="bold", va="top", ha="right",
    )

    # ── Win rate badge (top-right of left column) ─────────────────────
    ax_wr = fig.add_subplot(gs[0, 0])
    _style_ax(ax_wr)
    _draw_donut(ax_wr, win_rate, pnl_colour)

    # ── Stats grid (right column, top) ───────────────────────────────
    ax_stats = fig.add_subplot(gs[0, 1])
    _style_ax(ax_stats)
    _draw_stats(ax_stats, stats)

    # ── Cumulative PnL curve (full width, bottom) ─────────────────────
    # Merge the two bottom cells into one wide axes
    ax_curve = fig.add_subplot(gs[1, :])
    _style_ax(ax_curve)
    _draw_curve(ax_curve, stats)

    # ── Footer line ───────────────────────────────────────────────────
    fig.text(
        0.5, 0.04,
        "Generated by vGen · MEXC Futures",
        color=_SUBTEXT, fontsize=7.5, ha="center",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(_PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER)
    ax.tick_params(colors=_SUBTEXT, labelsize=7.5)


def _draw_donut(ax: plt.Axes, win_rate: float, colour: str) -> None:
    """Donut chart showing win rate."""
    sizes  = [win_rate, 100 - win_rate]
    colors = [colour, _BORDER]
    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.42, "edgecolor": _PANEL},
    )
    ax.set_aspect("equal")
    ax.text(
        0, 0.05, f"{win_rate:.1f}%",
        ha="center", va="center",
        color=colour, fontsize=15, fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0, -0.10, "Win Rate",
        ha="center", va="center",
        color=_SUBTEXT, fontsize=8,
        transform=ax.transAxes,
    )
    ax.axis("off")


def _draw_stats(ax: plt.Axes, s: Dict[str, Any]) -> None:
    """Mini stats table."""
    ax.axis("off")
    rows: List[Tuple[str, str]] = [
        ("Wins / Losses",    f"{s['wins']} / {s['losses']}"),
        ("Best trade",       f"+{s['best_pnl']:,.4f}"),
        ("Worst trade",      f"{s['worst_pnl']:,.4f}"),
        ("Total fees",       f"-{s['total_fee']:,.4f}"),
        ("Max drawdown",     f"{s['max_dd']:,.4f}"),
        ("TP / SL exits",    f"{s['exit_counts'].get('TP', 0)} / {s['exit_counts'].get('SL', 0)}"),
    ]
    y = 0.92
    for label, value in rows:
        ax.text(0.04, y, label,  color=_SUBTEXT, fontsize=8.5, transform=ax.transAxes, va="top")
        ax.text(0.96, y, value,  color=_TEXT,    fontsize=8.5, transform=ax.transAxes, va="top", ha="right")
        y -= 0.165


def _draw_curve(ax: plt.Axes, s: Dict[str, Any]) -> None:
    """Cumulative net PnL equity curve."""
    cum = s["cum_pnl"]
    if not cum:
        return

    xs = np.arange(len(cum))
    ys = np.array(cum, dtype=float)

    # Colour gradient: green above zero, red below
    positive = np.where(ys >= 0, ys, 0)
    negative = np.where(ys < 0, ys, 0)

    ax.fill_between(xs, positive, 0, alpha=0.25, color=_GREEN)
    ax.fill_between(xs, negative, 0, alpha=0.25, color=_RED)
    ax.plot(xs, ys, color=(_GREEN if ys[-1] >= 0 else _RED), linewidth=1.4)
    ax.axhline(0, color=_BORDER, linewidth=0.8, linestyle="--")

    ax.set_xlim(0, max(len(cum) - 1, 1))
    ax.set_xlabel("Trade #", color=_SUBTEXT, fontsize=8)
    ax.set_ylabel("Cum. Net PnL (USDT)", color=_SUBTEXT, fontsize=8)
    ax.tick_params(axis="both", labelsize=7.5, colors=_SUBTEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER)


# ═══════════════════════════════════════════════════════════════════════════════
# Single-trade card  (MEXC share-card style — sent on every TP/SL/trend-break)
# ═══════════════════════════════════════════════════════════════════════════════

def build_trade_card(
    symbol: str,
    side: str,              # "long" or "short"
    reason: str,            # "tp" | "sl" | "trend_break" | ...
    entry_price: float,
    exit_price: float,
    leverage: int,
    net_pnl: float,
    wins: int,
    losses: int,
    equity: float,
    timestamp: Optional[str] = None,   # pre-formatted string, e.g. "2026-05-01 11:35"
    strategy_label: str = "vGen · Filter-4h",
) -> Optional[bytes]:
    """Render a MEXC-style single-trade P&L share card.

    Returns PNG bytes, or None on failure.
    """
    try:
        return _render_trade_card(
            symbol, side, reason, entry_price, exit_price,
            leverage, net_pnl, wins, losses, equity,
            timestamp, strategy_label,
        )
    except Exception as exc:
        LOGGER.exception("build_trade_card failed: %s", exc)
        return None


def _render_trade_card(
    symbol: str,
    side: str,
    reason: str,
    entry_price: float,
    exit_price: float,
    leverage: int,
    net_pnl: float,
    wins: int,
    losses: int,
    equity: float,
    timestamp: Optional[str],
    strategy_label: str,
) -> bytes:
    is_long = side.lower() == "long"
    direction = 1 if is_long else -1

    # Leveraged % return (same as MEXC's profitRatio * 100)
    if entry_price > 0:
        roi_pct = (exit_price - entry_price) / entry_price * leverage * direction * 100
    else:
        roi_pct = 0.0

    is_win     = net_pnl >= 0
    pnl_colour = _GREEN if is_win else _RED
    roi_sign   = "+" if roi_pct >= 0 else ""
    pnl_sign   = "+" if net_pnl >= 0 else ""

    # Exit label
    reason_map = {"tp": "TP Hit", "sl": "Stop Loss", "trend_break": "Early Exit"}
    exit_label = reason_map.get(reason, reason.replace("_", " ").title())

    total = wins + losses
    wr    = wins / max(total, 1) * 100

    ts = timestamp or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Canvas ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.8, 7.2), facecolor=_BG)
    ax.set_facecolor(_BG)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Background panel with rounded feel (filled rect)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.04, 0.03), 0.92, 0.94,
        boxstyle="round,pad=0.01",
        linewidth=1.2, edgecolor=_BORDER,
        facecolor=_PANEL,
        transform=ax.transAxes, zorder=0,
    ))

    # ── MEXC logo + wordmark ──────────────────────────────────────────
    _logo_path = pathlib.Path(__file__).resolve().parents[2] / "assets" / "mexc_logo.png"
    if _logo_path.exists():
        logo_img = mpimg.imread(str(_logo_path))
        # Place logo in top-left inset axes
        logo_ax = fig.add_axes([0.055, 0.905, 0.065, 0.055])  # [left, bottom, w, h] in figure coords
        logo_ax.imshow(logo_img)
        logo_ax.axis("off")
        # "MEXC" wordmark beside logo
        ax.text(0.185, 0.950, "MEXC",
                color="#2C6BF5", fontsize=11, fontweight="bold",
                transform=ax.transAxes, va="top", zorder=2)
    else:
        ax.text(0.10, 0.950, "MEXC",
                color="#2C6BF5", fontsize=11, fontweight="bold",
                transform=ax.transAxes, va="top", zorder=2)

    # ── Header row ────────────────────────────────────────────────────
    ax.text(0.10, 0.905, strategy_label,
            color=_SUBTEXT, fontsize=8,
            transform=ax.transAxes, va="top", zorder=2)
    ax.text(0.90, 0.945, ts,
            color=_SUBTEXT, fontsize=7.5,
            transform=ax.transAxes, va="top", ha="right", zorder=2)

    # ── Symbol + side + leverage ──────────────────────────────────────
    sym_display = symbol.replace("_", "/") + " Perpetual"
    ax.text(0.10, 0.858, sym_display,
            color=_TEXT, fontsize=12, fontweight="bold",
            transform=ax.transAxes, va="top", zorder=2)

    side_label  = ("Long" if is_long else "Short") + f"  |  {leverage}X"
    side_colour = _GREEN if is_long else _RED
    ax.text(0.10, 0.815, side_label,
            color=side_colour, fontsize=10,
            transform=ax.transAxes, va="top", zorder=2)

    # ── Exit reason badge ─────────────────────────────────────────────
    ax.text(0.90, 0.815, exit_label,
            color=pnl_colour, fontsize=9, fontweight="bold",
            transform=ax.transAxes, va="top", ha="right", zorder=2)

    # ── Big ROI % ─────────────────────────────────────────────────────
    ax.text(0.10, 0.730,
            f"{roi_sign}{roi_pct:,.2f}%",
            color=pnl_colour, fontsize=36, fontweight="bold",
            transform=ax.transAxes, va="top", zorder=2)

    # ── Divider ───────────────────────────────────────────────────────
    ax.plot([0.07, 0.93], [0.50, 0.50],
            color=_BORDER, linewidth=0.8, transform=ax.transAxes, zorder=1)

    # ── Price rows ───────────────────────────────────────────────────
    _card_row(ax, 0.475, "Avg Entry Price", f"${entry_price:,.2f}")
    _card_row(ax, 0.415, "Avg Close Price", f"${exit_price:,.2f}")

    # ── Divider ───────────────────────────────────────────────────────
    ax.plot([0.07, 0.93], [0.375, 0.375],
            color=_BORDER, linewidth=0.8, transform=ax.transAxes, zorder=1)

    # ── Net PnL + stats row ───────────────────────────────────────────
    _card_row(ax, 0.350, "Net PnL",
              f"{pnl_sign}{abs(net_pnl):.4f} USDT",
              value_colour=pnl_colour)
    _card_row(ax, 0.295, "Balance", f"${equity:,.4f} USDT")

    ax.plot([0.07, 0.93], [0.255, 0.255],
            color=_BORDER, linewidth=0.8, transform=ax.transAxes, zorder=1)

    # ── Win rate footer ───────────────────────────────────────────────
    wr_colour = _GREEN if wr >= 78 else _YELLOW if wr >= 65 else _RED
    ax.text(0.10, 0.230,
            f"{wins}W  ·  {losses}L  ·  {wr:.1f}% WR",
            color=wr_colour, fontsize=9,
            transform=ax.transAxes, va="top", zorder=2)
    ax.text(0.90, 0.230,
            f"{total} trades",
            color=_SUBTEXT, fontsize=9,
            transform=ax.transAxes, va="top", ha="right", zorder=2)

    # ── Bottom brand line ─────────────────────────────────────────────
    ax.text(0.50, 0.065,
            "Generated by vGen  ·  MEXC Futures",
            color=_SUBTEXT, fontsize=7.5, ha="center",
            transform=ax.transAxes, va="top", zorder=2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _card_row(
    ax: plt.Axes,
    y: float,
    label: str,
    value: str,
    value_colour: str = _TEXT,
) -> None:
    ax.text(0.10, y, label,
            color=_SUBTEXT, fontsize=9,
            transform=ax.transAxes, va="top", zorder=2)
    ax.text(0.90, y, value,
            color=value_colour, fontsize=9, fontweight="bold",
            transform=ax.transAxes, va="top", ha="right", zorder=2)
