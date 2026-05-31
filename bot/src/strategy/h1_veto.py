"""1h-veto post-filter for the volume-farmer entry signal.

Single, intentionally simple rule (one knob's worth of free parameters):
        block a long if the 1h close is below 1h EMA-20 AND the EMA-20 has
        been falling for at least ``slope_bars`` 1h bars in a row;
        mirror for shorts.

Rationale: this only fires when the 1h regime is *strong and opposing*,
not on every counter-trend bar. We want a veto, not a primary signal.

Pure function, no I/O, no state.  Imported by volume_farmer when
``farmer.h1_veto.enabled`` is true.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def _resample_5m_to_1h(df_5m: pd.DataFrame) -> pd.DataFrame:
    if df_5m.empty or "open_time" not in df_5m.columns:
        return df_5m.iloc[0:0]
    t = pd.to_datetime(df_5m["open_time"], unit="ms", utc=True)
    g = (
        df_5m.assign(_ts=t)
        .set_index("_ts")
        .resample("1h", label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna()
    )
    if len(g) > 0:
        g = g.iloc[:-1]  # drop forming bucket
    return g.reset_index(drop=True)


def is_blocked(
    direction: str,
    history_5m: pd.DataFrame,
    ema_period: int = 20,
    slope_bars: int = 2,
    min_1h_bars: int = 25,
) -> bool:
    """Return True iff the proposed ``direction`` ('long'/'short') fights a
    *strong* opposing 1h trend, defined as:
        close < EMA20 AND EMA20 falling for ``slope_bars`` consecutive bars
        (block longs) — or the symmetric condition for shorts.

    Returns False (allow trade) on insufficient data or neutral 1h.
    """
    if direction not in ("long", "short"):
        return False
    if history_5m is None or len(history_5m) < min_1h_bars * 12:
        return False

    df_1h = _resample_5m_to_1h(history_5m)
    if len(df_1h) < ema_period + slope_bars + 1:
        return False

    close = df_1h["close"].astype(float)
    ema = close.ewm(span=ema_period, adjust=False).mean()

    last_close = float(close.iloc[-1])
    last_ema = float(ema.iloc[-1])

    # EMA slope: True if monotonically rising/falling for the last `slope_bars`
    tail = ema.iloc[-(slope_bars + 1):].to_numpy()
    diffs = tail[1:] - tail[:-1]
    rising = bool((diffs > 0).all())
    falling = bool((diffs < 0).all())

    if direction == "long":
        return (last_close < last_ema) and falling
    if direction == "short":
        return (last_close > last_ema) and rising
    return False
