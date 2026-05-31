"""Generate a candlestick chart PNG for a live trade entry.

Fetches the last N 5m candles from MEXC public klines API, draws the
candlestick chart with horizontal lines for entry, TP, and SL, then
returns the rendered PNG as bytes ready for Telegram ``sendPhoto``.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, Optional

import matplotlib
matplotlib.use("Agg")  # headless — no display required on VPS
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

if TYPE_CHECKING:
    from exchange.mexc_client import MEXCClient

LOGGER = logging.getLogger(__name__)

_DARK = {
    "base_mpl_style": "dark_background",
    "marketcolors": mpf.make_marketcolors(
        up="#26a69a",    # teal green
        down="#ef5350",  # red
        edge="inherit",
        wick="inherit",
        volume="in",
    ),
    "mavcolors": ["#ffd54f", "#ce93d8"],
    "y_on_right": True,
    "gridcolor": "#333333",
    "gridstyle": "--",
    "gridaxis": "both",
    "facecolor": "#131722",
    "figcolor": "#131722",
    "rc": {
        "axes.labelcolor": "#d1d4dc",
        "xtick.color": "#d1d4dc",
        "ytick.color": "#d1d4dc",
        "figure.facecolor": "#131722",
        "axes.facecolor": "#131722",
    },
}
_STYLE = mpf.make_mpf_style(**_DARK)


async def build_entry_chart(
    client: "MEXCClient",
    symbol: str,
    side: str,
    entry_price: float,
    tp_price: float,
    sl_price: float,
    bars: int = 60,
    interval: str = "Min5",
    tf_label: str = "5m",
) -> Optional[bytes]:
    """Fetch klines and render a candlestick chart with trade levels.

    Args:
        client: Authenticated MEXCClient (klines endpoint is public).
        symbol: e.g. ``BTC_USDT``.
        side: ``"long"`` or ``"short"``.
        entry_price: Actual fill price.
        tp_price: Take-profit level.
        sl_price: Stop-loss level.
        bars: Number of candles to show (default 60).
        interval: MEXC kline interval string (e.g. ``Min1``, ``Min5``).
        tf_label: Display label for the chart title (e.g. ``1m``, ``5m``).

    Returns:
        PNG bytes, or None if fetch/render fails.
    """
    try:
        resp = await client.get_klines(symbol, interval=interval)
    except Exception as exc:
        LOGGER.error("chart: kline fetch failed: %s", exc)
        return None

    try:
        df = _parse_klines(resp)
        if df is None or df.empty:
            LOGGER.warning("chart: empty klines response for %s", symbol)
            return None
        df = df.iloc[-bars:]
        return _render(df, symbol, side, entry_price, tp_price, sl_price, tf_label)
    except Exception as exc:
        LOGGER.exception("chart: render failed: %s", exc)
        return None


def _parse_klines(resp: object) -> Optional[pd.DataFrame]:
    """Parse MEXC klines API response into an OHLCV DataFrame."""
    # MEXCClient._request unwraps payload["data"] so ``resp`` is the inner dict
    # with {"time":[], "open":[], ...} directly. But when called raw, the outer
    # shape is {"code":0, "data": {...}}. Handle both.
    if not isinstance(resp, dict):
        return None
    data = resp.get("data") if "data" in resp and isinstance(resp.get("data"), dict) else resp
    if not isinstance(data, dict):
        return None

    times  = data.get("time", [])
    opens  = data.get("open", [])
    highs  = data.get("high", [])
    lows   = data.get("low", [])
    closes = data.get("close", [])
    vols   = data.get("vol", data.get("volume", []))

    if not times:
        return None

    df = pd.DataFrame({
        "Open":   [float(x) for x in opens],
        "High":   [float(x) for x in highs],
        "Low":    [float(x) for x in lows],
        "Close":  [float(x) for x in closes],
        "Volume": [float(x) for x in vols],
    }, index=pd.to_datetime([int(t) * 1000 if int(t) < 1e12 else int(t) for t in times], unit="ms", utc=True))
    df.index.name = "Date"
    df.sort_index(inplace=True)
    return df


def _render(
    df: pd.DataFrame,
    symbol: str,
    side: str,
    entry: float,
    tp: float,
    sl: float,
    tf_label: str = "5m",
) -> bytes:
    """Render the candlestick chart and return PNG bytes."""

    arrow = "▲ LONG" if side == "long" else "▼ SHORT"
    title = f"{symbol}  {tf_label}  —  {arrow}  entry {entry:,.1f}"

    # Horizontal lines: entry (yellow), TP (green), SL (red)
    hlines = dict(
        hlines=dict(
            hlines=[entry, tp, sl],
            colors=["#ffd54f", "#26a69a", "#ef5350"],
            linestyle=["--", "-", "-"],
            linewidths=[1.2, 1.0, 1.0],
        )
    )

    fig, axes = mpf.plot(
        df,
        type="candle",
        style=_STYLE,
        title=title,
        ylabel="Price (USDT)",
        volume=True,
        figsize=(12, 6),
        returnfig=True,
        **hlines,
    )

    # Legend
    patches = [
        mpatches.Patch(color="#ffd54f", label=f"Entry  {entry:,.1f}"),
        mpatches.Patch(color="#26a69a", label=f"TP     {tp:,.1f}"),
        mpatches.Patch(color="#ef5350", label=f"SL     {sl:,.1f}"),
    ]
    axes[0].legend(handles=patches, loc="upper left", fontsize=8,
                   facecolor="#1e222d", edgecolor="#444", labelcolor="#d1d4dc")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()
