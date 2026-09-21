"""Volume indicators. Daily rolling VWAP is not intraday session VWAP."""

from __future__ import annotations

import numpy as np
import polars as pl

from app.indicators.common import SID, apply_by_security, shift, sma, typical_price
from app.indicators.registry import register
from app.indicators.schema import IndicatorSpec


def _register() -> None:
    register(IndicatorSpec("volume_sma", "Volume SMA and ratios (today excluded from denominator)", "volume", ("volume_sma_5", "volume_sma_10", "volume_sma_20", "volume_sma_60", "volume_ratio_5", "volume_ratio_20", "volume_ratio_60"), minimum_history=60, warmup_sessions=60))
    register(IndicatorSpec("obv", "On-balance volume", "volume", ("obv",), minimum_history=2, warmup_sessions=2))
    register(IndicatorSpec("ad_line", "Accumulation/Distribution line", "volume", ("ad_line",), minimum_history=2, warmup_sessions=2))
    register(IndicatorSpec("cmf", "Chaikin money flow 20", "volume", ("cmf_20",), minimum_history=20, warmup_sessions=20))
    register(IndicatorSpec("chaikin_osc", "Chaikin oscillator 3/10", "volume", ("chaikin_oscillator",), minimum_history=10, warmup_sessions=10))
    register(IndicatorSpec("mfi", "Money flow index 14", "volume", ("mfi_14",), minimum_history=14, warmup_sessions=14))
    register(IndicatorSpec("force_index", "Force index 13", "volume", ("force_index_13",), minimum_history=13, warmup_sessions=13))
    register(IndicatorSpec("eom", "Ease of movement 14", "volume", ("eom_14",), minimum_history=14, warmup_sessions=14))
    register(IndicatorSpec("pvt", "Price volume trend", "volume", ("pvt",), minimum_history=2, warmup_sessions=2))
    register(IndicatorSpec("nvi_pvi", "Negative / positive volume index", "volume", ("nvi", "pvi"), minimum_history=2, warmup_sessions=2))
    register(IndicatorSpec("volume_roc", "Volume rate of change", "volume", ("volume_roc_10", "volume_roc_20"), minimum_history=20, warmup_sessions=20))
    register(IndicatorSpec("rolling_vwap", "Daily HLC typical-price rolling VWAP (not intraday session VWAP)", "volume", ("rolling_vwap_20", "rolling_vwap_60"), minimum_history=60, warmup_sessions=60, description="sum(typical_price*volume)/sum(volume) over N daily bars."))


def _obv_np(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    out = np.full(close.size, np.nan)
    if close.size == 0:
        return out
    out[0] = 0.0
    acc = 0.0
    for i in range(1, close.size):
        if not (np.isfinite(close[i]) and np.isfinite(close[i - 1]) and np.isfinite(volume[i])):
            out[i] = acc
            continue
        if close[i] > close[i - 1]:
            acc += float(volume[i])
        elif close[i] < close[i - 1]:
            acc -= float(volume[i])
        out[i] = acc
    return out


def _ad_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    out = np.full(close.size, np.nan)
    acc = 0.0
    for i in range(close.size):
        hl = high[i] - low[i]
        if not np.isfinite(hl) or hl == 0:
            out[i] = acc
            continue
        mfm = ((close[i] - low[i]) - (high[i] - close[i])) / hl
        acc += mfm * volume[i]
        out[i] = acc
    return out


def _mfi_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, n: int = 14) -> np.ndarray:
    tp = (high + low + close) / 3.0
    mf = tp * volume
    out = np.full(close.size, np.nan)
    for i in range(n, close.size):
        pos = 0.0
        neg = 0.0
        ok = True
        for j in range(i - n + 1, i + 1):
            if not (np.isfinite(tp[j]) and np.isfinite(tp[j - 1]) and np.isfinite(mf[j])):
                ok = False
                break
            if tp[j] > tp[j - 1]:
                pos += mf[j]
            elif tp[j] < tp[j - 1]:
                neg += mf[j]
        if not ok or (pos + neg) == 0:
            continue
        out[i] = 100.0 - 100.0 / (1.0 + pos / neg) if neg else 100.0
    return out


def _pvt_np(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    out = np.full(close.size, np.nan)
    acc = 0.0
    for i in range(1, close.size):
        if close[i - 1] and np.isfinite(close[i]) and np.isfinite(close[i - 1]) and np.isfinite(volume[i]):
            acc += (close[i] / close[i - 1] - 1.0) * float(volume[i])
        out[i] = acc
    if close.size:
        out[0] = 0.0
    return out


def _nvi_pvi_np(close: np.ndarray, volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nvi = np.full(close.size, np.nan)
    pvi = np.full(close.size, np.nan)
    nvi[0] = 1000.0
    pvi[0] = 1000.0
    for i in range(1, close.size):
        ret = close[i] / close[i - 1] - 1 if close[i - 1] else 0.0
        if not np.isfinite(ret) or not np.isfinite(volume[i]) or not np.isfinite(volume[i - 1]):
            nvi[i] = nvi[i - 1]
            pvi[i] = pvi[i - 1]
            continue
        nvi[i] = nvi[i - 1] * (1 + ret) if volume[i] < volume[i - 1] else nvi[i - 1]
        pvi[i] = pvi[i - 1] * (1 + ret) if volume[i] > volume[i - 1] else pvi[i - 1]
    return nvi, pvi


def apply_volume(df: pl.DataFrame) -> pl.DataFrame:
    work = df
    for n in (5, 10, 20, 60):
        work = work.with_columns(sma("volume", n).alias(f"volume_sma_{n}"))
    for n in (5, 20, 60):
        denom = pl.col("volume").shift(1).rolling_mean(n, min_samples=n).over(SID)
        work = work.with_columns((pl.col("volume") / denom).alias(f"volume_ratio_{n}"))
    work = apply_by_security(work, _obv_np, ["adjusted_close", "volume"], ["obv"])
    work = apply_by_security(work, _ad_np, ["adjusted_high", "adjusted_low", "adjusted_close", "volume"], ["ad_line"])
    mf = ((pl.col("adjusted_close") - pl.col("adjusted_low")) - (pl.col("adjusted_high") - pl.col("adjusted_close"))) / (pl.col("adjusted_high") - pl.col("adjusted_low")) * pl.col("volume")
    work = work.with_columns((mf.rolling_sum(20, min_samples=20).over(SID) / pl.col("volume").rolling_sum(20, min_samples=20).over(SID)).alias("cmf_20"))
    work = work.with_columns(
        (
            pl.col("ad_line").ewm_mean(span=3, adjust=False, min_samples=3).over(SID)
            - pl.col("ad_line").ewm_mean(span=10, adjust=False, min_samples=10).over(SID)
        ).alias("chaikin_oscillator")
    )
    work = apply_by_security(work, lambda h, l, c, v: _mfi_np(h, l, c, v, 14), ["adjusted_high", "adjusted_low", "adjusted_close", "volume"], ["mfi_14"])
    force = (pl.col("adjusted_close") - shift("adjusted_close")) * pl.col("volume")
    work = work.with_columns(force.ewm_mean(span=13, adjust=False, min_samples=13).over(SID).alias("force_index_13"))
    dist = ((pl.col("adjusted_high") + pl.col("adjusted_low")) / 2) - ((shift("adjusted_high") + shift("adjusted_low")) / 2)
    box = pl.col("volume") / (pl.col("adjusted_high") - pl.col("adjusted_low"))
    work = work.with_columns((dist / box).rolling_mean(14, min_samples=14).over(SID).alias("eom_14"))
    work = apply_by_security(work, _pvt_np, ["adjusted_close", "volume"], ["pvt"])
    work = apply_by_security(work, _nvi_pvi_np, ["adjusted_close", "volume"], ["nvi", "pvi"])
    work = work.with_columns(
        (pl.col("volume") / shift("volume", 10) - 1).alias("volume_roc_10"),
        (pl.col("volume") / shift("volume", 20) - 1).alias("volume_roc_20"),
    )
    tp = typical_price()
    for n in (20, 60):
        work = work.with_columns(
            ((tp * pl.col("volume")).rolling_sum(n, min_samples=n).over(SID) / pl.col("volume").rolling_sum(n, min_samples=n).over(SID)).alias(f"rolling_vwap_{n}")
        )
    return work


_register()
