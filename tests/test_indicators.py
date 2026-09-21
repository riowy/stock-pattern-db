"""Indicator engine: bounds, causal no-leakage, persistence disabled."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from app.indicators.engine import IndicatorEngine
from app.indicators.persist import IndicatorPersistenceDisabled, persist_indicators
from app.indicators.schema import INDICATOR_PERSISTENCE_ENABLED


def _prices(n: int = 80, start: date = date(2024, 1, 2), seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    dates = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    high = close + rng.uniform(0.2, 1.5, n)
    low = close - rng.uniform(0.2, 1.5, n)
    open_ = close + rng.normal(0, 0.3, n)
    return pl.DataFrame(
        {
            "security_id": ["S1"] * n,
            "ticker_at_time": ["AAA"] * n,
            "date": dates[:n],
            "open": open_,
            "high": np.maximum(high, np.maximum(open_, close)),
            "low": np.minimum(low, np.minimum(open_, close)),
            "close": close,
            "adj_close": close,
            "volume": rng.uniform(1e5, 2e5, n),
        }
    )


def test_persistence_flag_and_writer_blocked() -> None:
    assert INDICATOR_PERSISTENCE_ENABLED is False
    with pytest.raises(IndicatorPersistenceDisabled):
        persist_indicators(pl.DataFrame())


def test_rsi_stoch_williams_mfi_bounds() -> None:
    df = IndicatorEngine().compute(_prices(120), groups=["momentum", "volume"])
    last = df.filter(pl.col("rsi_14").is_not_null())
    assert last["rsi_14"].min() >= 0
    assert last["rsi_14"].max() <= 100
    sk = df.filter(pl.col("stoch_k_14").is_not_null())
    assert sk["stoch_k_14"].min() >= -1e-6
    assert sk["stoch_k_14"].max() <= 100 + 1e-6
    wr = df.filter(pl.col("williams_r_14").is_not_null())
    assert wr["williams_r_14"].min() >= -100 - 1e-6
    assert wr["williams_r_14"].max() <= 0 + 1e-6
    mfi = df.filter(pl.col("mfi_14").is_not_null())
    assert mfi["mfi_14"].min() >= 0
    assert mfi["mfi_14"].max() <= 100


def test_atr_bb_donchian_cloud_adx_vol_invariants() -> None:
    df = IndicatorEngine().compute(_prices(200), groups=["trend", "volatility", "ichimoku"])
    assert df["atr_14"].drop_nulls().min() >= 0
    bb = df.filter(pl.col("bb_20_2_upper").is_not_null())
    assert (bb["bb_20_2_upper"] >= bb["bb_20_2_mid"] - 1e-9).all()
    assert (bb["bb_20_2_mid"] >= bb["bb_20_2_lower"] - 1e-9).all()
    dc = df.filter(pl.col("donchian_20_upper").is_not_null())
    assert (dc["donchian_20_upper"] >= dc["donchian_20_lower"] - 1e-9).all()
    cloud = df.filter(pl.col("cloud_top").is_not_null())
    assert (cloud["cloud_top"] >= cloud["cloud_bottom"] - 1e-9).all()
    assert df["adx_14"].drop_nulls().min() >= 0
    assert df["hist_vol_20"].drop_nulls().min() >= 0


def test_future_price_does_not_change_past_indicators() -> None:
    base = _prices(60, seed=2)
    engine = IndicatorEngine()
    first = engine.compute(base, start=base["date"][0], end=base["date"][40])
    mutated = base.with_columns(
        pl.when(pl.col("date") == base["date"][-1]).then(pl.col("close") * 3).otherwise(pl.col("close")).alias("close"),
        pl.when(pl.col("date") == base["date"][-1]).then(pl.col("adj_close") * 3).otherwise(pl.col("adj_close")).alias("adj_close"),
    )
    second = engine.compute(mutated, start=base["date"][0], end=base["date"][40])
    cols = [c for c in ("sma_20", "rsi_14", "macd", "tenkan", "kijun", "cloud_a_at_t", "donchian_20_breakout_up", "pivot_p", "golden_cross_50_200") if c in first.columns]
    left = first.select(["date", *cols]).sort("date")
    right = second.select(["date", *cols]).sort("date")
    for col in cols:
        a = left[col].to_list()
        b = right[col].to_list()
        for x, y in zip(a, b, strict=True):
            if x is None or y is None:
                assert x == y
            elif isinstance(x, bool):
                assert x == y
            else:
                if isinstance(x, float) and isinstance(y, float) and x != x and y != y:
                    continue
                assert x == pytest.approx(y, rel=1e-9, abs=1e-9, nan_ok=True)


def test_ichimoku_cloud_is_shifted_from_past_senkou() -> None:
    df = IndicatorEngine().compute(_prices(90), groups=["ichimoku"])
    row = df.filter(pl.col("cloud_a_at_t").is_not_null()).tail(1)
    t = row["date"][0]
    past = df.filter(pl.col("date") < t).tail(26)
    assert past.height == 26
    assert row["cloud_a_at_t"][0] == pytest.approx(past["senkou_a_raw"][0], rel=1e-9, abs=1e-9)


def test_donchian_breakout_uses_prior_channel() -> None:
    n = 25
    close = np.arange(n, dtype=float) + 10
    dates = [date(2024, 1, 2) + timedelta(days=i) for i in range(n)]
    df = pl.DataFrame(
        {
            "security_id": ["S1"] * n,
            "ticker_at_time": ["AAA"] * n,
            "date": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "adj_close": close,
            "volume": [1000.0] * n,
        }
    )
    out = IndicatorEngine().compute(df, groups=["volatility"])
    # monotonically rising: every bar is a new high vs prior 20, so breakout can be true
    # but today's high is not in the prior channel
    last = out.tail(1)
    prior_max = float(close[-21:-1].max())
    assert last["adjusted_close"][0] > prior_max
    assert bool(last["donchian_20_breakout_up"][0]) is True


def test_pivot_uses_previous_session_only() -> None:
    df = _prices(10, seed=3)
    out = IndicatorEngine().compute(df, groups=["breakout"])
    i = 5
    prev = df.row(i - 1, named=True)
    p = (prev["high"] + prev["low"] + prev["close"]) / 3
    got = out.row(i, named=True)["pivot_p"]
    assert got == pytest.approx(p, rel=1e-9, abs=1e-9)


def test_visual_chikou_not_mining_enabled() -> None:
    from app.indicators.registry import get

    spec = get("visual_chikou_shifted")
    assert spec.causal_safe is False
    assert spec.mining_enabled is False
    assert spec.plot_only is True
