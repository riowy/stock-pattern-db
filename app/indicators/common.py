"""Shared vectorized helpers. Per-security NumPy is allowed; lake-wide row loops are not."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

SID = "security_id"


def over_roll(expr: pl.Expr) -> pl.Expr:
    return expr.over(SID)


def sma(col: str, n: int) -> pl.Expr:
    return pl.col(col).rolling_mean(window_size=n, min_samples=n).over(SID)


def ewm(col: str, n: int) -> pl.Expr:
    return pl.col(col).ewm_mean(span=n, adjust=False, min_samples=n).over(SID)


def roll_max(col: str, n: int) -> pl.Expr:
    return pl.col(col).rolling_max(window_size=n, min_samples=n).over(SID)


def roll_min(col: str, n: int) -> pl.Expr:
    return pl.col(col).rolling_min(window_size=n, min_samples=n).over(SID)


def roll_std(col: str, n: int) -> pl.Expr:
    return pl.col(col).rolling_std(window_size=n, min_samples=n).over(SID)


def roll_sum(col: str, n: int) -> pl.Expr:
    return pl.col(col).rolling_sum(window_size=n, min_samples=n).over(SID)


def shift(col: str, n: int = 1) -> pl.Expr:
    return pl.col(col).shift(n).over(SID)


def typical_price() -> pl.Expr:
    return (pl.col("adjusted_high") + pl.col("adjusted_low") + pl.col("adjusted_close")) / 3


def median_price() -> pl.Expr:
    return (pl.col("adjusted_high") + pl.col("adjusted_low")) / 2


def true_range() -> pl.Expr:
    prev = shift("adjusted_close", 1)
    return pl.max_horizontal(
        pl.col("adjusted_high") - pl.col("adjusted_low"),
        (pl.col("adjusted_high") - prev).abs(),
        (pl.col("adjusted_low") - prev).abs(),
    )


def cross_up(fast: str, slow: str) -> pl.Expr:
    return (pl.col(fast) > pl.col(slow)) & (shift(fast) <= shift(slow))


def cross_down(fast: str, slow: str) -> pl.Expr:
    return (pl.col(fast) < pl.col(slow)) & (shift(fast) >= shift(slow))


def ema_np(values: np.ndarray, period: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.size, np.nan, dtype=np.float64)
    if x.size < period:
        return out
    seed = x[:period]
    if not np.all(np.isfinite(seed)):
        return out
    out[period - 1] = float(seed.mean())
    alpha = 2.0 / (period + 1.0)
    for i in range(period, x.size):
        if not np.isfinite(x[i]) or not np.isfinite(out[i - 1]):
            out[i] = np.nan
            continue
        out[i] = alpha * float(x[i]) + (1.0 - alpha) * out[i - 1]
    return out


def wma_np(values: np.ndarray, period: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.size, np.nan, dtype=np.float64)
    if x.size < period:
        return out
    weights = np.arange(1, period + 1, dtype=np.float64)
    denom = weights.sum()
    for i in range(period - 1, x.size):
        window = x[i - period + 1 : i + 1]
        if not np.all(np.isfinite(window)):
            continue
        out[i] = float(np.dot(window, weights) / denom)
    return out


def wilder_smooth_np(values: np.ndarray, period: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.size, np.nan, dtype=np.float64)
    if x.size < period:
        return out
    seed = x[:period]
    if not np.all(np.isfinite(seed)):
        return out
    acc = float(seed.sum())
    out[period - 1] = acc
    for i in range(period, x.size):
        if not np.isfinite(x[i]):
            out[i] = np.nan
            continue
        acc = acc - acc / period + float(x[i])
        out[i] = acc
    return out


def apply_by_security(
    df: pl.DataFrame,
    fn: Callable[..., np.ndarray | tuple[np.ndarray, ...]],
    columns: list[str],
    out_names: list[str],
) -> pl.DataFrame:
    """Run ``fn(*arrays)`` once per security. ``fn`` must be vectorized on the series."""
    if df.height == 0:
        return df.with_columns([pl.lit(None, dtype=pl.Float64).alias(n) for n in out_names])
    work = df.sort([SID, "date"])
    chunks: list[list[np.ndarray]] = [[] for _ in out_names]
    for _sid, group in work.group_by(SID, maintain_order=True):
        arrays = [group[c].to_numpy() for c in columns]
        result = fn(*arrays)
        if isinstance(result, tuple):
            series_list = result
        else:
            series_list = (result,)
        for i, arr in enumerate(series_list):
            chunks[i].append(np.asarray(arr, dtype=np.float64))
    extras = [pl.Series(name, np.concatenate(parts) if parts else np.array([], dtype=np.float64)) for name, parts in zip(out_names, chunks, strict=True)]
    return work.with_columns(extras)
