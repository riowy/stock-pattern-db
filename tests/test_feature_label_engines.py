"""FeatureEngine / LabelEngine correctness, null policy, versions, leakage."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from app.features.adjustment import add_adjusted_prices
from app.features.engine import FeatureEngine
from app.features.indicators import wilder_atr_np, wilder_rsi_np
from app.labels.engine import LabelEngine


def _dates(n: int, start: date = date(2024, 1, 2)) -> list[date]:
    # Consecutive calendar weekdays; tests that need a weekend gap insert it explicitly.
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _bars(
    security_id: str,
    ticker: str,
    dates: list[date],
    close: list[float],
    adj: list[float] | None = None,
    volume: list[float] | None = None,
    high: list[float] | None = None,
    low: list[float] | None = None,
    open_: list[float] | None = None,
) -> pl.DataFrame:
    n = len(dates)
    adj = adj or close
    volume = volume or [1000.0] * n
    high = high or [c + 1 for c in close]
    low = low or [c - 1 for c in close]
    open_ = open_ or close
    return pl.DataFrame(
        {
            "security_id": [security_id] * n,
            "ticker_at_time": [ticker] * n,
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": adj,
            "volume": volume,
            "dividend": [0.0] * n,
            "stock_split": [0.0] * n,
            "currency": ["USD"] * n,
            "provider": ["test"] * n,
            "retrieved_at": [datetime.now(UTC)] * n,
        }
    )


def test_wilder_rsi_hand_calculated() -> None:
    close = np.array([10.0, 12.0, 11.0, 13.0])
    rsi = wilder_rsi_np(close, period=2)
    assert np.isnan(rsi[0]) and np.isnan(rsi[1])
    assert rsi[2] == pytest.approx(100.0 - 100.0 / 3.0)
    assert rsi[3] == pytest.approx(100.0 - 100.0 / 7.0)
    assert (rsi[2] >= 0) and (rsi[2] <= 100)


def test_wilder_atr_hand_calculated() -> None:
    high = np.array([11.0, 13.0, 12.0, 14.0])
    low = np.array([9.0, 10.0, 10.0, 12.0])
    close = np.array([10.0, 12.0, 11.0, 13.0])
    atr = wilder_atr_np(high, low, close, period=2)
    assert atr[1] == pytest.approx(2.5)
    assert atr[2] == pytest.approx(2.25)
    assert atr[3] == pytest.approx(2.625)
    assert atr[3] > 0


def test_adjusted_price_split_no_fake_crash() -> None:
    dates = _dates(3)
    # 2-for-1: raw close halves, adj_close stays flat
    prices = _bars("S1", "AAA", dates, close=[100.0, 100.0, 50.0], adj=[50.0, 50.0, 50.0])
    adj = add_adjusted_prices(prices)
    assert adj["adjusted_close"].to_list() == [50.0, 50.0, 50.0]
    feats = FeatureEngine().calculate_features(prices, dates[0], dates[-1], security_ids=["S1"])
    # ret_1d across the split uses adj_close, not raw
    row = feats.filter(pl.col("date") == dates[2]).row(0, named=True)
    assert row["ret_1d"] == pytest.approx(0.0)


def test_no_silent_raw_close_fallback() -> None:
    dates = _dates(2)
    prices = _bars("S1", "AAA", dates, close=[100.0, 0.0], adj=[100.0, 0.0])
    adj = add_adjusted_prices(prices)
    assert adj["adjustment_valid"].to_list() == [True, False]
    assert adj["adjusted_close"][1] is None


def test_ma_and_ma200_history_null() -> None:
    dates = _dates(50)
    close = [100.0 + i for i in range(50)]
    prices = _bars("S1", "AAA", dates, close)
    feats = FeatureEngine().calculate_features(prices, dates[0], dates[-1], security_ids=["S1"])
    last = feats.filter(pl.col("date") == dates[-1]).row(0, named=True)
    assert last["ma_5_distance"] is not None
    assert last["ma_200_distance"] is None
    assert last["distance_high_252d"] is None


def test_volume_ratio_excludes_today() -> None:
    dates = _dates(6)
    volumes = [10.0, 20.0, 30.0, 40.0, 50.0, 100.0]
    prices = _bars("S1", "AAA", dates, close=[10.0] * 6, volume=volumes)
    feats = FeatureEngine().calculate_features(prices, dates[0], dates[-1], security_ids=["S1"])
    last = feats.filter(pl.col("date") == dates[-1]).row(0, named=True)
    # previous 5 sessions: 10,20,30,40,50 mean=30
    assert last["volume_ratio_5d"] == pytest.approx(100.0 / 30.0)


def test_rolling_high_low() -> None:
    dates = _dates(5)
    close = [10.0, 11.0, 12.0, 9.0, 10.0]
    high = [11.0, 12.0, 15.0, 10.0, 11.0]
    low = [9.0, 10.0, 11.0, 8.0, 9.0]
    prices = _bars("S1", "AAA", dates, close, high=high, low=low)
    feats = FeatureEngine().calculate_features(prices, dates[0], dates[-1], security_ids=["S1"])
    last = feats.filter(pl.col("date") == dates[-1]).row(0, named=True)
    assert last["distance_high_20d"] is None  # only 5 sessions
    # 5-day window not requested; 20d needs 20. Just assert rolling on full history via engine internals:
    # With 5 rows, 20d high/low must be null.
    assert last["distance_low_20d"] is None


def test_forward_returns_and_horizons() -> None:
    dates = _dates(25)
    close = [float(100 + i) for i in range(25)]
    prices = _bars("S1", "AAA", dates, close, adj=close)
    labels = LabelEngine().calculate_labels(prices, dates[0], dates[-1], security_ids=["S1"])
    first = labels.filter(pl.col("date") == dates[0]).row(0, named=True)
    assert first["forward_return_1d"] == pytest.approx(close[1] / close[0] - 1)
    assert first["forward_return_3d"] == pytest.approx(close[3] / close[0] - 1)
    assert first["forward_return_5d"] == pytest.approx(close[5] / close[0] - 1)
    assert first["forward_return_10d"] == pytest.approx(close[10] / close[0] - 1)
    assert first["forward_return_20d"] == pytest.approx(close[20] / close[0] - 1)
    # last row: 1d through 20d all immature
    last = labels.filter(pl.col("date") == dates[-1]).row(0, named=True)
    assert last["forward_return_1d"] is None
    assert last["forward_return_20d"] is None
    # date with 5 future sessions but not 20
    row = labels.filter(pl.col("date") == dates[-6]).row(0, named=True)
    assert row["forward_return_5d"] is not None
    assert row["forward_return_20d"] is None


def test_weekend_is_one_session_horizon() -> None:
    # Friday then Monday as consecutive rows
    dates = [date(2024, 1, 5), date(2024, 1, 8)]  # Fri, Mon
    close = [100.0, 110.0]
    prices = _bars("S1", "AAA", dates, close, adj=close)
    labels = LabelEngine().calculate_labels(prices, dates[0], dates[0], security_ids=["S1"])
    row = labels.row(0, named=True)
    assert row["forward_return_1d"] == pytest.approx(0.10)


def test_max_gain_and_drawdown() -> None:
    dates = _dates(6)
    close = [100.0, 110.0, 90.0, 120.0, 80.0, 105.0]
    prices = _bars("S1", "AAA", dates, close, adj=close)
    labels = LabelEngine().calculate_labels(prices, dates[0], dates[0], security_ids=["S1"])
    row = labels.row(0, named=True)
    # next 5 closes vs 100: +10%, -10%, +20%, -20%, +5%
    assert row["max_gain_next_5d"] == pytest.approx(0.20)
    assert row["max_drawdown_next_5d"] == pytest.approx(-0.20)


def test_spy_excess_and_missing_spy_is_null() -> None:
    dates = _dates(5)
    stock = _bars("S1", "AAA", dates, [100, 101, 102, 103, 104], adj=[100, 101, 102, 103, 104])
    spy = _bars("SPY1", "SPY", dates, [200, 202, 204, 206, 208], adj=[200, 202, 204, 206, 208])
    labels = LabelEngine().calculate_labels(pl.concat([stock, spy]), dates[0], dates[0], security_ids=["S1"])
    row = labels.row(0, named=True)
    stock_fwd = 101 / 100 - 1
    spy_fwd = 202 / 200 - 1
    assert row["forward_excess_spy_1d"] == pytest.approx(stock_fwd - spy_fwd)

    labels_no_spy = LabelEngine().calculate_labels(stock, dates[0], dates[0], security_ids=["S1"])
    assert labels_no_spy.row(0, named=True)["forward_excess_spy_1d"] is None


def test_relative_strength_spy_qqq() -> None:
    dates = _dates(6)
    stock = _bars("S1", "AAA", dates, [100, 101, 102, 103, 104, 110])
    spy = _bars("SPY1", "SPY", dates, [100, 100, 100, 100, 100, 100])
    qqq = _bars("QQQ1", "QQQ", dates, [100, 100, 100, 100, 100, 105])
    feats = FeatureEngine().calculate_features(
        pl.concat([stock, spy, qqq]), dates[-1], dates[-1], security_ids=["S1"]
    )
    row = feats.row(0, named=True)
    assert row["rel_spy_5d"] == pytest.approx((110 / 100 - 1) - (100 / 100 - 1))
    assert row["rel_qqq_5d"] == pytest.approx((110 / 100 - 1) - (105 / 100 - 1))


def test_vix_join_no_forward_fill() -> None:
    dates = _dates(6)
    prices = _bars("S1", "AAA", dates, [10.0] * 6)
    vix = pl.DataFrame(
        {
            "date": [dates[0], dates[2], dates[5]],
            "close": [15.0, 16.0, 20.0],
        }
    )
    feats = FeatureEngine().calculate_features(prices, dates[0], dates[-1], security_ids=["S1"], vix=vix)
    by_date = {row["date"]: row for row in feats.iter_rows(named=True)}
    assert by_date[dates[0]]["vix_close"] == pytest.approx(15.0)
    assert by_date[dates[1]]["vix_close"] is None  # no forward fill
    assert by_date[dates[5]]["vix_close"] == pytest.approx(20.0)


def test_feature_and_label_version_columns() -> None:
    dates = _dates(5)
    prices = _bars("S1", "AAA", dates, [10.0] * 5)
    f = FeatureEngine().calculate_features(prices, dates[0], dates[-1], feature_version="v1", security_ids=["S1"])
    f2 = FeatureEngine().calculate_features(prices, dates[0], dates[-1], feature_version="v2", security_ids=["S1"])
    assert set(f["feature_version"].unique()) == {"v1"}
    assert set(f2["feature_version"].unique()) == {"v2"}
    l = LabelEngine().calculate_labels(prices, dates[0], dates[-1], label_version="v1", security_ids=["S1"])
    l2 = LabelEngine().calculate_labels(prices, dates[0], dates[-1], label_version="v2", security_ids=["S1"])
    assert set(l["label_version"].unique()) == {"v1"}
    assert set(l2["label_version"].unique()) == {"v2"}


def test_future_price_does_not_change_past_features() -> None:
    dates = _dates(30)
    close = [100.0 + i for i in range(30)]
    prices = _bars("S1", "AAA", dates, close, volume=[1000.0] * 30)
    engine = FeatureEngine()
    base = engine.calculate_features(prices, dates[0], dates[20], security_ids=["S1"])
    mutated = prices.with_columns(
        pl.when(pl.col("date") > dates[20]).then(pl.col("close") * 10).otherwise(pl.col("close")).alias("close"),
        pl.when(pl.col("date") > dates[20]).then(pl.col("adj_close") * 10).otherwise(pl.col("adj_close")).alias("adj_close"),
        pl.when(pl.col("date") > dates[20]).then(pl.col("volume") * 50).otherwise(pl.col("volume")).alias("volume"),
    )
    after = engine.calculate_features(mutated, dates[0], dates[20], security_ids=["S1"])
    compare_cols = [c for c in base.columns if c not in {"calculated_at", "calculation_code_version"}]
    assert base.select(compare_cols).equals(after.select(compare_cols))


def test_future_volume_does_not_change_today_volume_ratio() -> None:
    dates = _dates(10)
    prices = _bars("S1", "AAA", dates, [10.0] * 10, volume=[float(i + 1) * 100 for i in range(10)])
    t = dates[6]
    engine = FeatureEngine()
    base = engine.calculate_features(prices, t, t, security_ids=["S1"])
    mutated = prices.with_columns(
        pl.when(pl.col("date") > t).then(pl.lit(999999.0)).otherwise(pl.col("volume")).alias("volume")
    )
    after = engine.calculate_features(mutated, t, t, security_ids=["S1"])
    assert base["volume_ratio_5d"][0] == after["volume_ratio_5d"][0]


def test_forward_label_changes_when_future_changes() -> None:
    dates = _dates(10)
    prices = _bars("S1", "AAA", dates, [100.0] * 10, adj=[100.0] * 10)
    engine = LabelEngine()
    base = engine.calculate_labels(prices, dates[0], dates[0], security_ids=["S1"])
    mutated = prices.with_columns(
        pl.when(pl.col("date") == dates[1]).then(pl.lit(200.0)).otherwise(pl.col("adj_close")).alias("adj_close")
    )
    after = engine.calculate_labels(mutated, dates[0], dates[0], security_ids=["S1"])
    assert base["forward_return_1d"][0] != after["forward_return_1d"][0]
    assert after["forward_return_1d"][0] == pytest.approx(1.0)


def test_future_spy_and_vix_do_not_leak_into_features() -> None:
    dates = _dates(10)
    stock = _bars("S1", "AAA", dates, [100.0] * 10)
    spy = _bars("SPY1", "SPY", dates, [200.0] * 10)
    prices = pl.concat([stock, spy])
    vix = pl.DataFrame({"date": dates, "close": [15.0] * 10})
    t = dates[5]
    engine = FeatureEngine()
    base = engine.calculate_features(prices, t, t, security_ids=["S1"], vix=vix)
    spy_mut = spy.with_columns(
        pl.when(pl.col("date") > t).then(pl.lit(999.0)).otherwise(pl.col("adj_close")).alias("adj_close"),
        pl.when(pl.col("date") > t).then(pl.lit(999.0)).otherwise(pl.col("close")).alias("close"),
    )
    vix_mut = vix.with_columns(pl.when(pl.col("date") > t).then(pl.lit(80.0)).otherwise(pl.col("close")).alias("close"))
    after = engine.calculate_features(pl.concat([stock, spy_mut]), t, t, security_ids=["S1"], vix=vix_mut)
    assert base["rel_spy_5d"][0] == after["rel_spy_5d"][0]
    assert base["vix_close"][0] == after["vix_close"][0]
    assert base["spy_ret_5d"][0] == after["spy_ret_5d"][0]


def test_sector_relative_is_null() -> None:
    dates = _dates(6)
    prices = _bars("S1", "AAA", dates, [10.0] * 6)
    feats = FeatureEngine().calculate_features(prices, dates[-1], dates[-1], security_ids=["S1"])
    row = feats.row(0, named=True)
    assert row["rel_sector_5d"] is None
    assert row["rel_sector_20d"] is None
    assert row["rel_sector_60d"] is None
