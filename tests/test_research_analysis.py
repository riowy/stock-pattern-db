"""Research analysis: splits, CS quantiles, IC, no-leakage, sample guards."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from app.research.config import (
    DEVELOPMENT_END,
    INSUFFICIENT_SAMPLE,
    SPLIT_DEVELOPMENT,
    SPLIT_VALIDATION,
    VALIDATION_START,
)
from app.research.dataset import assign_split, filter_split
from app.research.statistics import (
    assign_cs_quintiles,
    freeze_vix_cutoffs,
    ic_by_date_table,
    ic_table,
    quantile_table,
    sample_status,
    spread_table,
)
from app.research.store import attach_provenance


def _panel() -> pl.DataFrame:
    """10 names × many dates in development and validation.

    Feature ``ret_20d`` equals name index 0..9 every day (cross-section stable).
    Development excess return increases with the feature; validation reverses.
    """
    rows: list[dict] = []
    d = date(2020, 1, 2)
    while d <= date(2020, 6, 30):
        for i in range(10):
            rows.append(
                {
                    "security_id": f"S{i}",
                    "ticker": f"T{i}",
                    "date": d,
                    "ret_20d": float(i),
                    "forward_excess_spy_5d": 0.001 * i,
                    "forward_excess_spy_10d": 0.002 * i,
                    "forward_excess_spy_20d": 0.003 * i,
                    "forward_return_5d": 0.001 * i,
                    "forward_return_10d": 0.002 * i,
                    "forward_return_20d": 0.003 * i,
                    "vix_close": 12.0 if i < 5 else 30.0,
                    "spy_ma200_distance": 0.05,
                    "raw_close": 25.0,
                    "universe_name": "research-common-equity-500",
                    "instrument_class": "COMMON_EQUITY",
                    "data_quality_valid": True,
                }
            )
        d += timedelta(days=1)
    d = date(2024, 1, 2)
    while d <= date(2024, 3, 31):
        for i in range(10):
            rows.append(
                {
                    "security_id": f"S{i}",
                    "ticker": f"T{i}",
                    "date": d,
                    "ret_20d": float(i),
                    "forward_excess_spy_5d": 0.001 * (9 - i),
                    "forward_excess_spy_10d": 0.002 * (9 - i),
                    "forward_excess_spy_20d": 0.003 * (9 - i),
                    "forward_return_5d": 0.001 * (9 - i),
                    "forward_return_10d": 0.002 * (9 - i),
                    "forward_return_20d": 0.003 * (9 - i),
                    "vix_close": 40.0,
                    "spy_ma200_distance": -0.04,
                    "raw_close": 25.0,
                    "universe_name": "research-common-equity-500",
                    "instrument_class": "COMMON_EQUITY",
                    "data_quality_valid": True,
                }
            )
        d += timedelta(days=1)
    return pl.DataFrame(rows).with_columns(
        pl.when(pl.col("date") <= DEVELOPMENT_END)
        .then(pl.lit(SPLIT_DEVELOPMENT))
        .otherwise(pl.lit(SPLIT_VALIDATION))
        .alias("split")
    )


def test_development_validation_split_bounds() -> None:
    assert assign_split(DEVELOPMENT_END) == SPLIT_DEVELOPMENT
    assert assign_split(VALIDATION_START) == SPLIT_VALIDATION
    df = _panel()
    dev = filter_split(df, SPLIT_DEVELOPMENT)
    val = filter_split(df, SPLIT_VALIDATION)
    assert dev["date"].max() <= DEVELOPMENT_END
    assert val["date"].min() >= VALIDATION_START


def test_cs_quantile_does_not_use_future_distribution() -> None:
    df = _panel()
    day = df["date"].min()
    one = df.filter(pl.col("date") == day)
    q_one = assign_cs_quintiles(one, "ret_20d").select(["security_id", "quintile"]).sort("security_id")
    q_all = (
        assign_cs_quintiles(df, "ret_20d")
        .filter(pl.col("date") == day)
        .select(["security_id", "quintile"])
        .sort("security_id")
    )
    assert q_one.equals(q_all)
    future = df.with_columns(pl.when(pl.col("date") == day).then(pl.col("ret_20d")).otherwise(pl.col("ret_20d") * 100))
    q_mut = (
        assign_cs_quintiles(future, "ret_20d")
        .filter(pl.col("date") == day)
        .select(["security_id", "quintile"])
        .sort("security_id")
    )
    assert q_one.equals(q_mut)


def test_q5_minus_q1_and_monotonic_dev_vs_validation() -> None:
    df = _panel()
    q_dev = quantile_table(df, ["ret_20d"], SPLIT_DEVELOPMENT)
    q_val = quantile_table(df, ["ret_20d"], SPLIT_VALIDATION)
    s_dev = spread_table(q_dev)
    s_val = spread_table(q_val)
    assert s_dev.height == 1
    assert s_dev["forward_excess_spy_20d_q5_minus_q1"][0] > 0
    assert s_val["forward_excess_spy_20d_q5_minus_q1"][0] < 0
    # Validation sign flip must not be used to rewrite development.
    assert s_dev["forward_excess_spy_20d_q5_minus_q1"][0] != s_val["forward_excess_spy_20d_q5_minus_q1"][0]


def test_spearman_ic_positive_in_development() -> None:
    df = _panel()
    ic_dev = ic_table(df, ["ret_20d"], SPLIT_DEVELOPMENT)
    ic_val = ic_table(df, ["ret_20d"], SPLIT_VALIDATION)
    row_dev = ic_dev.filter(pl.col("target") == "forward_excess_spy_20d").row(0, named=True)
    row_val = ic_val.filter(pl.col("target") == "forward_excess_spy_20d").row(0, named=True)
    assert row_dev["mean_ic"] > 0.2
    assert row_val["mean_ic"] < -0.2
    daily = ic_by_date_table(df, "ret_20d", "forward_excess_spy_20d", SPLIT_DEVELOPMENT)
    assert daily.height == row_dev["n_dates"]


def test_small_sample_guard() -> None:
    assert sample_status(10, 5) == INSUFFICIENT_SAMPLE
    assert sample_status(1000, 1) == "ok"
    assert sample_status(10, 100) == "ok"


def test_vix_cutoff_frozen_from_development_only() -> None:
    df = _panel()
    dev = filter_split(df, SPLIT_DEVELOPMENT)
    val = filter_split(df, SPLIT_VALIDATION)
    q33, q67 = freeze_vix_cutoffs(dev)
    val_q33, val_q67 = freeze_vix_cutoffs(val)
    assert q33 is not None and q67 is not None
    assert (val_q33, val_q67) != (q33, q67)
    frozen_again, frozen_again_hi = freeze_vix_cutoffs(dev)
    assert frozen_again == q33
    assert frozen_again_hi == q67
    from app.research.statistics import attach_regimes

    labeled = attach_regimes(val, q33, q67)
    assert set(labeled["vix_regime"].drop_nulls().unique().to_list()) == {"vix_high"}


def test_research_result_versioning(tmp_path) -> None:
    df = pl.DataFrame({"feature": ["ret_20d"], "mean_ic": [0.01]})
    out = attach_provenance(df, research_version="v1")
    assert out["research_version"][0] == "v1"
    assert out["feature_version"][0] == "v1"
    assert out["label_version"][0] == "v1"
    assert out["universe_name"][0] == "research-common-equity-500"
    assert out["development_end"][0] == DEVELOPMENT_END.isoformat()
    assert "calculation_code_version" in out.columns
    assert "calculated_at" in out.columns
