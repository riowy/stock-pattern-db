"""Moving averages, MACD family, trend strength, Supertrend, PSAR."""

from __future__ import annotations

import numpy as np
import polars as pl

from app.indicators.common import (
    SID,
    apply_by_security,
    cross_down,
    cross_up,
    ema_np,
    ewm,
    shift,
    sma,
    true_range,
    wma_np,
)
from app.indicators.registry import register
from app.indicators.schema import IndicatorSpec

SMA_LENS = (5, 10, 20, 30, 50, 60, 100, 120, 150, 200, 250)
EMA_LENS = (5, 10, 12, 20, 26, 50, 60, 100, 120, 200)
WMA_LENS = (10, 20, 50, 100)
DEMA_LENS = (10, 20, 50)
TEMA_LENS = (10, 20, 50)
HMA_LENS = (9, 20, 55)
KAMA_LENS = (10, 20)
VWMA_LENS = (10, 20, 50)


def _register_trend() -> None:
    if "sma" in {s.indicator_id for s in (register.__defaults__ or [])}:
        return
    register(
        IndicatorSpec(
            "sma",
            "Simple moving averages",
            "trend",
            tuple(f"sma_{n}" for n in SMA_LENS)
            + tuple(f"close_to_sma_{n}" for n in SMA_LENS)
            + tuple(f"sma_{n}_slope" for n in (20, 50, 200))
            + tuple(f"sma_{n}_direction" for n in (20, 50, 200))
            + (
                "close_above_sma20",
                "close_above_sma50",
                "close_above_sma200",
                "sma20_above_sma50",
                "sma50_above_sma200",
                "golden_cross_50_200",
                "death_cross_50_200",
            ),
            parameters={"lengths": SMA_LENS},
            minimum_history=250,
            warmup_sessions=250,
            description="SMA of adjusted close; distance = close/ma - 1; slope vs 5 sessions ago.",
        )
    )
    register(
        IndicatorSpec(
            "ema",
            "Exponential moving averages",
            "trend",
            tuple(f"ema_{n}" for n in EMA_LENS)
            + ("ema12_cross_ema26_up", "ema12_cross_ema26_down"),
            parameters={"lengths": EMA_LENS},
            minimum_history=200,
            warmup_sessions=200,
        )
    )
    register(IndicatorSpec("wma", "Weighted moving averages", "trend", tuple(f"wma_{n}" for n in WMA_LENS), minimum_history=100, warmup_sessions=100))
    register(IndicatorSpec("dema", "Double EMA", "trend", tuple(f"dema_{n}" for n in DEMA_LENS), minimum_history=50, warmup_sessions=50))
    register(IndicatorSpec("tema", "Triple EMA", "trend", tuple(f"tema_{n}" for n in TEMA_LENS), minimum_history=50, warmup_sessions=50))
    register(IndicatorSpec("hma", "Hull moving average", "trend", tuple(f"hma_{n}" for n in HMA_LENS), minimum_history=55, warmup_sessions=55))
    register(IndicatorSpec("kama", "Kaufman adaptive MA", "trend", tuple(f"kama_{n}" for n in KAMA_LENS), minimum_history=20, warmup_sessions=20))
    register(IndicatorSpec("vwma", "Volume-weighted MA", "trend", tuple(f"vwma_{n}" for n in VWMA_LENS), minimum_history=50, warmup_sessions=50))
    register(
        IndicatorSpec(
            "macd",
            "MACD 12/26/9",
            "trend",
            ("macd", "macd_signal", "macd_histogram", "macd_cross_signal_up", "macd_cross_signal_down", "macd_cross_zero_up", "macd_cross_zero_down"),
            parameters={"fast": 12, "slow": 26, "signal": 9},
            minimum_history=35,
            warmup_sessions=35,
        )
    )
    register(IndicatorSpec("ppo", "Percentage price oscillator 12/26/9", "trend", ("ppo", "ppo_signal", "ppo_histogram"), minimum_history=35, warmup_sessions=35))
    register(IndicatorSpec("apo", "Absolute price oscillator 12/26", "trend", ("apo",), minimum_history=26, warmup_sessions=26))
    register(
        IndicatorSpec(
            "adx",
            "Average directional index",
            "trend",
            ("plus_di_14", "minus_di_14", "dx_14", "adx_14", "plus_di_20", "minus_di_20", "dx_20", "adx_20", "adx14_gt_20", "adx14_gt_25", "adx14_gt_40", "plus_di_cross_minus_di_up", "plus_di_cross_minus_di_down"),
            parameters={"lengths": (14, 20)},
            minimum_history=40,
            warmup_sessions=40,
        )
    )
    register(IndicatorSpec("aroon", "Aroon", "trend", ("aroon_up_14", "aroon_down_14", "aroon_osc_14", "aroon_up_25", "aroon_down_25", "aroon_osc_25"), minimum_history=25, warmup_sessions=25))
    register(IndicatorSpec("vortex", "Vortex 14", "trend", ("vortex_pos_14", "vortex_neg_14"), minimum_history=14, warmup_sessions=14))
    register(IndicatorSpec("mass_index", "Mass Index 9/25", "trend", ("mass_index",), minimum_history=25, warmup_sessions=25))
    register(IndicatorSpec("choppiness", "Choppiness Index 14", "trend", ("choppiness_14",), minimum_history=14, warmup_sessions=14))
    register(
        IndicatorSpec(
            "supertrend",
            "Supertrend ATR",
            "trend",
            ("supertrend_10_3", "supertrend_10_3_dir", "supertrend_flip_up", "supertrend_flip_down", "supertrend_10_2", "supertrend_10_2_dir", "supertrend_10_4", "supertrend_10_4_dir"),
            parameters={"atr": 10, "multipliers": (2, 3, 4)},
            minimum_history=15,
            warmup_sessions=15,
        )
    )
    register(
        IndicatorSpec(
            "psar",
            "Parabolic SAR 0.02/0.20",
            "trend",
            ("psar", "psar_direction", "psar_flip_up", "psar_flip_down"),
            parameters={"step": 0.02, "max": 0.20},
            minimum_history=3,
            warmup_sessions=3,
        )
    )


def _hma_np(close: np.ndarray, n: int) -> np.ndarray:
    half = max(n // 2, 1)
    sqrt_n = max(int(round(n**0.5)), 1)
    wma_half = wma_np(close, half)
    wma_full = wma_np(close, n)
    raw = 2.0 * wma_half - wma_full
    return wma_np(raw, sqrt_n)


def _kama_np(close: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(close, dtype=np.float64)
    out = np.full(x.size, np.nan)
    if x.size < n + 1:
        return out
    fast = 2.0 / (2 + 1)
    slow = 2.0 / (30 + 1)
    out[n] = x[n]
    for i in range(n + 1, x.size):
        change = abs(x[i] - x[i - n])
        volatility = np.nansum(np.abs(np.diff(x[i - n : i + 1])))
        er = change / volatility if volatility and np.isfinite(volatility) else 0.0
        sc = (er * (fast - slow) + slow) ** 2
        if not np.isfinite(x[i]) or not np.isfinite(out[i - 1]):
            out[i] = np.nan
            continue
        out[i] = out[i - 1] + sc * (x[i] - out[i - 1])
    return out


def _adx_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = close.size
    plus_dm = np.full(n, np.nan)
    minus_dm = np.full(n, np.nan)
    tr = np.full(n, np.nan)
    tr[0] = high[0] - low[0] if np.isfinite(high[0]) and np.isfinite(low[0]) else np.nan
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if np.isfinite(up) and up > down and up > 0 else 0.0
        minus_dm[i] = down if np.isfinite(down) and down > up and down > 0 else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.full(n, np.nan)
    pdm = np.full(n, np.nan)
    mdm = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.nanmean(tr[1:period]) if period > 1 else tr[0]
        pdm[period - 1] = np.nansum(plus_dm[1:period])
        mdm[period - 1] = np.nansum(minus_dm[1:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
            pdm[i] = pdm[i - 1] - pdm[i - 1] / period + plus_dm[i]
            mdm[i] = mdm[i - 1] - mdm[i - 1] / period + minus_dm[i]
    with np.errstate(invalid="ignore", divide="ignore"):
        plus_di = np.where((atr > 0) & np.isfinite(atr), 100.0 * pdm / atr, np.nan)
        minus_di = np.where((atr > 0) & np.isfinite(atr), 100.0 * mdm / atr, np.nan)
        denom = plus_di + minus_di
        dx = np.where(denom > 0, 100.0 * np.abs(plus_di - minus_di) / denom, np.nan)
    adx = np.full(n, np.nan)
    start = 2 * period - 1
    if n > start:
        window = dx[period - 1 : start + 1]
        adx[start] = float(np.nanmean(window)) if np.any(np.isfinite(window)) else np.nan
        for i in range(start + 1, n):
            if np.isfinite(adx[i - 1]) and np.isfinite(dx[i]):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return plus_di, minus_di, dx, adx


def _supertrend_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, atr_n: int, mult: float) -> tuple[np.ndarray, np.ndarray]:
    from app.features.indicators import wilder_atr_np

    atr = wilder_atr_np(high, low, close, atr_n)
    n = close.size
    mid = (high + low) / 2.0
    upper = mid + mult * atr
    lower = mid - mult * atr
    st = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(atr[i]):
            continue
        if i == 0 or not np.isfinite(st[i - 1]):
            st[i] = lower[i]
            direction[i] = 1.0
            continue
        fu, fl = upper[i], lower[i]
        if np.isfinite(upper[i - 1]) and close[i - 1] <= st[i - 1]:
            fu = min(upper[i], upper[i - 1]) if np.isfinite(upper[i - 1]) else upper[i]
        if np.isfinite(lower[i - 1]) and close[i - 1] >= st[i - 1]:
            fl = max(lower[i], lower[i - 1]) if np.isfinite(lower[i - 1]) else lower[i]
        if direction[i - 1] == 1:
            if close[i] < fl:
                direction[i] = -1.0
                st[i] = fu
            else:
                direction[i] = 1.0
                st[i] = fl
        else:
            if close[i] > fu:
                direction[i] = 1.0
                st[i] = fl
            else:
                direction[i] = -1.0
                st[i] = fu
    return st, direction


def _psar_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, step: float = 0.02, max_af: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    n = close.size
    psar = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    if n < 2:
        return psar, direction
    bull = True
    af = step
    hp = high[0]
    lp = low[0]
    psar[0] = low[0]
    direction[0] = 1.0
    for i in range(1, n):
        prev = psar[i - 1]
        if bull:
            cand = prev + af * (hp - prev)
            cand = min(cand, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < cand:
                bull = False
                psar[i] = hp
                direction[i] = -1.0
                lp = low[i]
                af = step
            else:
                psar[i] = cand
                direction[i] = 1.0
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + step, max_af)
        else:
            cand = prev + af * (lp - prev)
            cand = max(cand, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > cand:
                bull = True
                psar[i] = lp
                direction[i] = 1.0
                hp = high[i]
                af = step
            else:
                psar[i] = cand
                direction[i] = -1.0
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + step, max_af)
    return psar, direction


def apply_trend(df: pl.DataFrame) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    for n in SMA_LENS:
        ma = sma("adjusted_close", n)
        exprs.append(ma.alias(f"sma_{n}"))
        exprs.append((pl.col("adjusted_close") / ma - 1).alias(f"close_to_sma_{n}"))
    for n in EMA_LENS:
        exprs.append(ewm("adjusted_close", n).alias(f"ema_{n}"))
    work = df.with_columns(exprs)
    work = work.with_columns(
        [(pl.col(f"sma_{n}") / shift(f"sma_{n}", 5) - 1).alias(f"sma_{n}_slope") for n in (20, 50, 200)]
    )
    work = work.with_columns(
        [
            pl.when(pl.col(f"sma_{n}_slope") > 0)
            .then(pl.lit(1))
            .when(pl.col(f"sma_{n}_slope") < 0)
            .then(pl.lit(-1))
            .otherwise(pl.lit(0))
            .alias(f"sma_{n}_direction")
            for n in (20, 50, 200)
        ]
    )
    work = work.with_columns(
        (pl.col("adjusted_close") > pl.col("sma_20")).alias("close_above_sma20"),
        (pl.col("adjusted_close") > pl.col("sma_50")).alias("close_above_sma50"),
        (pl.col("adjusted_close") > pl.col("sma_200")).alias("close_above_sma200"),
        (pl.col("sma_20") > pl.col("sma_50")).alias("sma20_above_sma50"),
        (pl.col("sma_50") > pl.col("sma_200")).alias("sma50_above_sma200"),
        cross_up("sma_50", "sma_200").alias("golden_cross_50_200"),
        cross_down("sma_50", "sma_200").alias("death_cross_50_200"),
        cross_up("ema_12", "ema_26").alias("ema12_cross_ema26_up"),
        cross_down("ema_12", "ema_26").alias("ema12_cross_ema26_down"),
        (pl.col("ema_12") - pl.col("ema_26")).alias("macd_line_raw"),
        (pl.col("ema_12") - pl.col("ema_26")).alias("apo"),
        ((pl.col("ema_12") - pl.col("ema_26")) / pl.col("ema_26") * 100.0).alias("ppo_raw"),
    )
    work = work.with_columns(
        pl.col("macd_line_raw").alias("macd"),
        pl.col("macd_line_raw").ewm_mean(span=9, adjust=False, min_samples=9).over(SID).alias("macd_signal"),
        pl.col("ppo_raw").alias("ppo"),
        pl.col("ppo_raw").ewm_mean(span=9, adjust=False, min_samples=9).over(SID).alias("ppo_signal"),
    )
    work = work.with_columns(
        (pl.col("macd") - pl.col("macd_signal")).alias("macd_histogram"),
        (pl.col("ppo") - pl.col("ppo_signal")).alias("ppo_histogram"),
        cross_up("macd", "macd_signal").alias("macd_cross_signal_up"),
        cross_down("macd", "macd_signal").alias("macd_cross_signal_down"),
        ((pl.col("macd") > 0) & (shift("macd") <= 0)).alias("macd_cross_zero_up"),
        ((pl.col("macd") < 0) & (shift("macd") >= 0)).alias("macd_cross_zero_down"),
    )
    for n in WMA_LENS:
        work = apply_by_security(work, lambda c, nn=n: wma_np(c, nn), ["adjusted_close"], [f"wma_{n}"])
    for n in DEMA_LENS:
        work = apply_by_security(
            work,
            lambda c, nn=n: 2 * ema_np(c, nn) - ema_np(ema_np(c, nn), nn),
            ["adjusted_close"],
            [f"dema_{n}"],
        )
    for n in TEMA_LENS:

        def _tema(c: np.ndarray, nn: int = n) -> np.ndarray:
            e1 = ema_np(c, nn)
            e2 = ema_np(e1, nn)
            e3 = ema_np(e2, nn)
            return 3 * e1 - 3 * e2 + e3

        work = apply_by_security(work, _tema, ["adjusted_close"], [f"tema_{n}"])
    for n in HMA_LENS:
        work = apply_by_security(work, lambda c, nn=n: _hma_np(c, nn), ["adjusted_close"], [f"hma_{n}"])
    for n in KAMA_LENS:
        work = apply_by_security(work, lambda c, nn=n: _kama_np(c, nn), ["adjusted_close"], [f"kama_{n}"])
    for n in VWMA_LENS:
        num = (pl.col("adjusted_close") * pl.col("volume")).rolling_sum(n, min_samples=n).over(SID)
        den = pl.col("volume").rolling_sum(n, min_samples=n).over(SID)
        work = work.with_columns((num / den).alias(f"vwma_{n}"))

    for period in (14, 20):
        work = apply_by_security(
            work,
            lambda h, l, c, p=period: _adx_np(h, l, c, p),
            ["adjusted_high", "adjusted_low", "adjusted_close"],
            [f"plus_di_{period}", f"minus_di_{period}", f"dx_{period}", f"adx_{period}"],
        )
    work = work.with_columns(
        (pl.col("adx_14") > 20).alias("adx14_gt_20"),
        (pl.col("adx_14") > 25).alias("adx14_gt_25"),
        (pl.col("adx_14") > 40).alias("adx14_gt_40"),
        cross_up("plus_di_14", "minus_di_14").alias("plus_di_cross_minus_di_up"),
        cross_down("plus_di_14", "minus_di_14").alias("plus_di_cross_minus_di_down"),
    )
    for n in (14, 25):
        work = work.with_columns(
            (100.0 * (n - (pl.col("adjusted_high").cum_count().over(SID) - pl.col("adjusted_high").arg_max().rolling_max(n, min_samples=n).over(SID))) / n).alias(f"_skip_aroon_up_{n}")
        )
        # Use days-since-high via rolling: (n-1 - argmax_in_window)/n * 100
        work = apply_by_security(
            work,
            lambda h, l, nn=n: _aroon_np(h, l, nn),
            ["adjusted_high", "adjusted_low"],
            [f"aroon_up_{n}", f"aroon_down_{n}"],
        )
        work = work.with_columns((pl.col(f"aroon_up_{n}") - pl.col(f"aroon_down_{n}")).alias(f"aroon_osc_{n}"))
    work = work.drop([c for c in work.columns if c.startswith("_skip_aroon")])

    work = apply_by_security(work, _vortex_np, ["adjusted_high", "adjusted_low", "adjusted_close"], ["vortex_pos_14", "vortex_neg_14"])
    hl = pl.col("adjusted_high") - pl.col("adjusted_low")
    ema9 = hl.ewm_mean(span=9, adjust=False, min_samples=9).over(SID)
    ema9_2 = ema9.ewm_mean(span=9, adjust=False, min_samples=9).over(SID)
    ratio = ema9 / ema9_2
    work = work.with_columns(ratio.rolling_sum(window_size=25, min_samples=25).over(SID).alias("mass_index"))
    tr = true_range()
    work = work.with_columns(tr.alias("true_range"))
    hh = pl.col("adjusted_high").rolling_max(14, min_samples=14).over(SID)
    ll = pl.col("adjusted_low").rolling_min(14, min_samples=14).over(SID)
    atr_sum = tr.rolling_sum(14, min_samples=14).over(SID)
    work = work.with_columns((100.0 * (atr_sum / (hh - ll)).log() / np.log(14)).alias("choppiness_14"))

    for mult in (2, 3, 4):
        work = apply_by_security(
            work,
            lambda h, l, c, m=mult: _supertrend_np(h, l, c, 10, m),
            ["adjusted_high", "adjusted_low", "adjusted_close"],
            [f"supertrend_10_{mult}", f"supertrend_10_{mult}_dir"],
        )
    work = work.with_columns(
        ((pl.col("supertrend_10_3_dir") == 1) & (shift("supertrend_10_3_dir") == -1)).alias("supertrend_flip_up"),
        ((pl.col("supertrend_10_3_dir") == -1) & (shift("supertrend_10_3_dir") == 1)).alias("supertrend_flip_down"),
    )
    work = apply_by_security(
        work,
        lambda h, l, c: _psar_np(h, l, c),
        ["adjusted_high", "adjusted_low", "adjusted_close"],
        ["psar", "psar_direction"],
    )
    work = work.with_columns(
        ((pl.col("psar_direction") == 1) & (shift("psar_direction") == -1)).alias("psar_flip_up"),
        ((pl.col("psar_direction") == -1) & (shift("psar_direction") == 1)).alias("psar_flip_down"),
    )
    return work


def _aroon_np(high: np.ndarray, low: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    up = np.full(high.size, np.nan)
    down = np.full(low.size, np.nan)
    for i in range(n - 1, high.size):
        hwin = high[i - n + 1 : i + 1]
        lwin = low[i - n + 1 : i + 1]
        if not (np.all(np.isfinite(hwin)) and np.all(np.isfinite(lwin))):
            continue
        up[i] = 100.0 * (np.argmax(hwin) + 1) / n
        down[i] = 100.0 * (np.argmin(lwin) + 1) / n
    return up, down


def _vortex_np(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = close.size
    vp = np.full(n, np.nan)
    vm = np.full(n, np.nan)
    tr = np.full(n, np.nan)
    plus = np.full(n, np.nan)
    minus = np.full(n, np.nan)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        plus[i] = abs(high[i] - low[i - 1])
        minus[i] = abs(low[i] - high[i - 1])
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    period = 14
    for i in range(period, n):
        trs = np.nansum(tr[i - period + 1 : i + 1])
        if trs <= 0:
            continue
        vp[i] = np.nansum(plus[i - period + 1 : i + 1]) / trs
        vm[i] = np.nansum(minus[i - period + 1 : i + 1]) / trs
    return vp, vm


def apply_trend_lite(df: pl.DataFrame) -> pl.DataFrame:
    """SMA/EMA/MACD/ADX14/Supertrend only — mining path skips WMA/HMA/KAMA."""
    exprs: list[pl.Expr] = []
    for n in (5, 10, 20, 50, 60, 120, 200):
        ma = sma("adjusted_close", n)
        exprs.append(ma.alias(f"sma_{n}"))
        exprs.append((pl.col("adjusted_close") / ma - 1).alias(f"close_to_sma_{n}"))
    for n in (12, 20, 26, 50, 200):
        exprs.append(ewm("adjusted_close", n).alias(f"ema_{n}"))
    work = df.with_columns(exprs)
    work = work.with_columns(
        (pl.col("adjusted_close") > pl.col("sma_20")).alias("close_above_sma20"),
        (pl.col("adjusted_close") > pl.col("sma_50")).alias("close_above_sma50"),
        (pl.col("adjusted_close") > pl.col("sma_200")).alias("close_above_sma200"),
        (pl.col("sma_20") > pl.col("sma_50")).alias("sma20_above_sma50"),
        (pl.col("sma_50") > pl.col("sma_200")).alias("sma50_above_sma200"),
        cross_up("sma_50", "sma_200").alias("golden_cross_50_200"),
        cross_down("sma_50", "sma_200").alias("death_cross_50_200"),
        (pl.col("ema_12") - pl.col("ema_26")).alias("macd"),
        pl.when(pl.col("sma_20") > shift("sma_20", 5)).then(pl.lit(1)).when(pl.col("sma_20") < shift("sma_20", 5)).then(pl.lit(-1)).otherwise(pl.lit(0)).alias("sma_20_direction"),
        pl.when(pl.col("sma_50") > shift("sma_50", 5)).then(pl.lit(1)).when(pl.col("sma_50") < shift("sma_50", 5)).then(pl.lit(-1)).otherwise(pl.lit(0)).alias("sma_50_direction"),
        pl.when(pl.col("sma_200") > shift("sma_200", 5)).then(pl.lit(1)).when(pl.col("sma_200") < shift("sma_200", 5)).then(pl.lit(-1)).otherwise(pl.lit(0)).alias("sma_200_direction"),
    )
    work = work.with_columns(pl.col("macd").ewm_mean(span=9, adjust=False, min_samples=9).over(SID).alias("macd_signal"))
    work = work.with_columns(
        (pl.col("macd") - pl.col("macd_signal")).alias("macd_histogram"),
        cross_up("macd", "macd_signal").alias("macd_cross_signal_up"),
        cross_down("macd", "macd_signal").alias("macd_cross_signal_down"),
    )
    work = apply_by_security(
        work,
        lambda h, l, c: _adx_np(h, l, c, 14),
        ["adjusted_high", "adjusted_low", "adjusted_close"],
        ["plus_di_14", "minus_di_14", "dx_14", "adx_14"],
    )
    work = work.with_columns((pl.col("adx_14") > 25).alias("adx14_gt_25"))
    work = apply_by_security(
        work,
        lambda h, l, c: _supertrend_np(h, l, c, 10, 3),
        ["adjusted_high", "adjusted_low", "adjusted_close"],
        ["supertrend_10_3", "supertrend_10_3_dir"],
    )
    work = work.with_columns(
        ((pl.col("supertrend_10_3_dir") == 1) & (shift("supertrend_10_3_dir") == -1)).alias("supertrend_flip_up"),
        ((pl.col("supertrend_10_3_dir") == -1) & (shift("supertrend_10_3_dir") == 1)).alias("supertrend_flip_down"),
    )
    return work


_register_trend()
