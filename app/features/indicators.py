"""Wilder RSI / ATR -- textbook SMA-seed then recursive smoothing.

Implemented with NumPy over one security's series (not a Python row loop
over the lake). Grouping happens once per security_id.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
    if not np.isfinite(avg_gain) or not np.isfinite(avg_loss):
        return np.nan
    if avg_loss == 0.0 and avg_gain == 0.0:
        return 50.0
    if avg_loss == 0.0:
        return 100.0
    if avg_gain == 0.0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def wilder_rsi_np(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI. First value is at index ``period`` (period changes needed)."""
    close = np.asarray(close, dtype=np.float64)
    n = close.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return out

    delta = np.diff(close)
    gains = np.where(np.isfinite(delta) & (delta > 0), delta, 0.0)
    losses = np.where(np.isfinite(delta) & (delta < 0), -delta, 0.0)
    gains = np.where(np.isfinite(delta), gains, np.nan)
    losses = np.where(np.isfinite(delta), losses, np.nan)

    seed_g = gains[:period]
    seed_l = losses[:period]
    if not (np.all(np.isfinite(seed_g)) and np.all(np.isfinite(seed_l))):
        return out

    avg_gain = float(seed_g.mean())
    avg_loss = float(seed_l.mean())
    out[period] = _rsi_from_avgs(avg_gain, avg_loss)

    for i in range(period, delta.size):
        g = gains[i]
        lss = losses[i]
        if not (np.isfinite(g) and np.isfinite(lss)):
            out[i + 1] = np.nan
            continue
        avg_gain = (avg_gain * (period - 1) + float(g)) / period
        avg_loss = (avg_loss * (period - 1) + float(lss)) / period
        out[i + 1] = _rsi_from_avgs(avg_gain, avg_loss)
    return out


def wilder_atr_np(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Wilder ATR from adjusted high/low/close. First value at index ``period-1``."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = close.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out

    tr = np.full(n, np.nan, dtype=np.float64)
    tr[0] = high[0] - low[0]
    prev_close = close[:-1]
    hl = high[1:] - low[1:]
    hc = np.abs(high[1:] - prev_close)
    lc = np.abs(low[1:] - prev_close)
    tr[1:] = np.maximum(np.maximum(hl, hc), lc)

    seed = tr[:period]
    if not np.all(np.isfinite(seed)):
        return out
    atr = float(seed.mean())
    out[period - 1] = atr
    for i in range(period, n):
        if not np.isfinite(tr[i]):
            out[i] = np.nan
            continue
        atr = (atr * (period - 1) + float(tr[i])) / period
        out[i] = atr
    return out


def add_wilder_indicators(df: pl.DataFrame, rsi_period: int = 14, atr_period: int = 14) -> pl.DataFrame:
    """Append ``rsi_14``, ``atr_14``, ``atr_pct_14`` per security_id."""
    if df.height == 0:
        return df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("rsi_14"),
            pl.lit(None, dtype=pl.Float64).alias("atr_14"),
            pl.lit(None, dtype=pl.Float64).alias("atr_pct_14"),
        )

    df = df.sort(["security_id", "date"])
    rsi_chunks: list[np.ndarray] = []
    atr_chunks: list[np.ndarray] = []
    for _sid, group in df.group_by("security_id", maintain_order=True):
        rsi_chunks.append(wilder_rsi_np(group["adjusted_close"].to_numpy(), rsi_period))
        atr_chunks.append(
            wilder_atr_np(
                group["adjusted_high"].to_numpy(),
                group["adjusted_low"].to_numpy(),
                group["adjusted_close"].to_numpy(),
                atr_period,
            )
        )
    rsi = np.concatenate(rsi_chunks) if rsi_chunks else np.array([], dtype=np.float64)
    atr = np.concatenate(atr_chunks) if atr_chunks else np.array([], dtype=np.float64)
    atr_pct = np.where(
        np.isfinite(atr) & np.isfinite(df["adjusted_close"].to_numpy()) & (df["adjusted_close"].to_numpy() != 0),
        atr / df["adjusted_close"].to_numpy(),
        np.nan,
    )
    return df.with_columns(
        pl.Series("rsi_14", rsi),
        pl.Series("atr_14", atr),
        pl.Series("atr_pct_14", atr_pct),
    )
