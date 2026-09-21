"""Linear regression geometry, z-scores, rolling percentiles."""

from __future__ import annotations

import numpy as np
import polars as pl

from app.indicators.common import SID, apply_by_security, roll_std, shift, sma
from app.indicators.registry import register
from app.indicators.schema import IndicatorSpec

LINREG = (20, 60, 120)
Z = (20, 60, 120)
PCTL = (20, 60, 252)


def _register() -> None:
    outs = []
    for n in LINREG:
        outs.extend([f"linreg_slope_{n}", f"linreg_slope_norm_{n}", f"linreg_r2_{n}", f"linreg_zscore_{n}", f"dist_from_linreg_{n}", f"corr_time_{n}"])
    register(IndicatorSpec("linreg", "Rolling linear regression vs time index", "trend", tuple(outs), minimum_history=120, warmup_sessions=120))
    register(
        IndicatorSpec(
            "zscore_percentile",
            "Rolling z-score and percentile ranks",
            "momentum",
            tuple(f"close_zscore_{n}" for n in Z)
            + tuple(f"return_zscore_{n}" for n in Z)
            + tuple(f"volume_zscore_{n}" for n in Z)
            + ("price_position_252", "volume_percentile_60", "volatility_percentile_60"),
            minimum_history=252,
            warmup_sessions=252,
        )
    )


def _linreg_np(close: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(close, dtype=np.float64)
    slope = np.full(y.size, np.nan)
    r2 = np.full(y.size, np.nan)
    z = np.full(y.size, np.nan)
    dist = np.full(y.size, np.nan)
    corr = np.full(y.size, np.nan)
    t = np.arange(n, dtype=np.float64)
    t = t - t.mean()
    denom_t = float(np.dot(t, t))
    if denom_t == 0:
        return slope, slope, r2, z, dist
    for i in range(n - 1, y.size):
        w = y[i - n + 1 : i + 1]
        if not np.all(np.isfinite(w)):
            continue
        yc = w - w.mean()
        b = float(np.dot(t, yc) / denom_t)
        intercept = float(w.mean())  # at mean t = 0, fitted mid
        fitted = intercept + b * t
        resid = w - fitted
        ss_res = float(np.dot(resid, resid))
        ss_tot = float(np.dot(yc, yc))
        slope[i] = b
        r2[i] = 1.0 - ss_res / ss_tot if ss_tot else np.nan
        sd = float(np.std(resid, ddof=1)) if n > 2 else np.nan
        z[i] = resid[-1] / sd if sd else np.nan
        dist[i] = resid[-1] / w[-1] if w[-1] else np.nan
        corr[i] = float(np.dot(t, yc) / (np.sqrt(denom_t) * np.sqrt(ss_tot))) if ss_tot else np.nan
    return slope, slope / np.where(y != 0, y, np.nan), r2, z, dist, corr  # type: ignore[return-value]


def apply_geometry(df: pl.DataFrame) -> pl.DataFrame:
    work = df
    for n in LINREG:
        work = apply_by_security(
            work,
            lambda c, nn=n: _linreg_pack(c, nn),
            ["adjusted_close"],
            [f"linreg_slope_{n}", f"linreg_slope_norm_{n}", f"linreg_r2_{n}", f"linreg_zscore_{n}", f"dist_from_linreg_{n}", f"corr_time_{n}"],
        )
    ret = pl.col("adjusted_close") / shift("adjusted_close") - 1
    for n in Z:
        work = work.with_columns(
            ((pl.col("adjusted_close") - sma("adjusted_close", n)) / roll_std("adjusted_close", n)).alias(f"close_zscore_{n}"),
            ((ret - ret.rolling_mean(n, min_samples=n).over(SID)) / ret.rolling_std(n, min_samples=n).over(SID)).alias(f"return_zscore_{n}"),
            ((pl.col("volume") - sma("volume", n)) / roll_std("volume", n)).alias(f"volume_zscore_{n}"),
        )
    work = work.with_columns(
        pl.col("adjusted_close").rank(method="average").over(SID).alias("_rk_all"),
    )
    # rolling percentile via rank in window
    work = apply_by_security(work, lambda c: _rolling_pct(c, 252), ["adjusted_close"], ["price_position_252"])
    work = apply_by_security(work, lambda v: _rolling_pct(v, 60), ["volume"], ["volume_percentile_60"])
    if "hist_vol_20" in work.columns:
        work = apply_by_security(work, lambda v: _rolling_pct(v, 60), ["hist_vol_20"], ["volatility_percentile_60"])
    elif "stdev_ret_20" in work.columns:
        work = apply_by_security(work, lambda v: _rolling_pct(v, 60), ["stdev_ret_20"], ["volatility_percentile_60"])
    else:
        work = work.with_columns(pl.lit(None, dtype=pl.Float64).alias("volatility_percentile_60"))
    return work.drop([c for c in work.columns if c.startswith("_rk")])


def _linreg_pack(close: np.ndarray, n: int) -> tuple[np.ndarray, ...]:
    y = np.asarray(close, dtype=np.float64)
    slope = np.full(y.size, np.nan)
    norm = np.full(y.size, np.nan)
    r2 = np.full(y.size, np.nan)
    z = np.full(y.size, np.nan)
    dist = np.full(y.size, np.nan)
    corr = np.full(y.size, np.nan)
    t = np.arange(n, dtype=np.float64)
    t = t - t.mean()
    denom_t = float(np.dot(t, t))
    for i in range(n - 1, y.size):
        w = y[i - n + 1 : i + 1]
        if not np.all(np.isfinite(w)):
            continue
        yc = w - w.mean()
        b = float(np.dot(t, yc) / denom_t)
        fitted = w.mean() + b * t
        resid = w - fitted
        ss_res = float(np.dot(resid, resid))
        ss_tot = float(np.dot(yc, yc))
        slope[i] = b
        norm[i] = b / w[-1] if w[-1] else np.nan
        r2[i] = 1.0 - ss_res / ss_tot if ss_tot else np.nan
        sd = float(np.std(resid, ddof=1)) if n > 2 else np.nan
        z[i] = resid[-1] / sd if sd else np.nan
        dist[i] = resid[-1] / w[-1] if w[-1] else np.nan
        corr[i] = float(np.dot(t, yc) / (np.sqrt(denom_t * ss_tot))) if ss_tot else np.nan
    return slope, norm, r2, z, dist, corr


def _rolling_pct(values: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.size, np.nan)
    for i in range(n - 1, x.size):
        w = x[i - n + 1 : i + 1]
        if not np.isfinite(x[i]) or not np.any(np.isfinite(w)):
            continue
        finite = w[np.isfinite(w)]
        out[i] = float((finite <= x[i]).sum() / finite.size)
    return out


_register()
