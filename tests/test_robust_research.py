"""Robust research statistics: frozen winsor, overlays, redundancy."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from app.research.config import SPLIT_DEVELOPMENT, SPLIT_VALIDATION
from app.research.concentration import low_price_concentration
from app.research.dataset import filter_split
from app.research.overlay import filter_price_floor, overlay_tables
from app.research.redundancy import feature_rank_correlation
from app.research.robust import (
    apply_winsor_columns,
    cutoff_map,
    freeze_target_cutoffs,
    robust_quantile_table,
    robust_spread_table,
)
from app.research.statistics import ic_table, overlapping_spread_table, quantile_table, spread_table
from tests.test_research_analysis import _panel


def _panel_with_outlier() -> pl.DataFrame:
    df = _panel().with_columns((pl.col("ret_20d") + 0.05).alias("rel_spy_20d"))
    extra = []
    d = date(2024, 2, 1)
    while d <= date(2024, 2, 20):
        extra.append(
            {
                "security_id": "PENNY",
                "ticker": "PENNY",
                "date": d,
                "ret_20d": 9.0,
                "rel_spy_20d": 9.05,
                "forward_excess_spy_5d": 40.0,
                "forward_excess_spy_10d": 40.0,
                "forward_excess_spy_20d": 50.0,
                "forward_return_5d": 40.0,
                "forward_return_10d": 40.0,
                "forward_return_20d": 50.0,
                "vix_close": 40.0,
                "spy_ma200_distance": -0.04,
                "raw_close": 0.4,
                "universe_name": "research-common-equity-500",
                "instrument_class": "COMMON_EQUITY",
                "data_quality_valid": True,
                "split": SPLIT_VALIDATION,
            }
        )
        d += timedelta(days=1)
    return pl.concat([df, pl.DataFrame(extra)], how="diagonal_relaxed")


def test_winsor_cutoff_uses_development_only() -> None:
    df = _panel_with_outlier()
    mixed = freeze_target_cutoffs(df)
    mapping = cutoff_map(mixed)
    lo, hi = mapping["forward_return_20d"]
    assert hi < 1.0
    val_as_dev = df.filter(pl.col("split") == SPLIT_VALIDATION).with_columns(
        pl.lit(SPLIT_DEVELOPMENT).alias("split"),
        pl.lit(date(2020, 6, 1)).alias("date"),
    )
    val_cut = cutoff_map(freeze_target_cutoffs(val_as_dev))
    assert val_cut["forward_return_20d"][1] > hi
    assert val_cut["forward_return_20d"][1] > 10


def test_validation_uses_frozen_cutoff_not_its_own() -> None:
    df = _panel_with_outlier()
    frozen = cutoff_map(freeze_target_cutoffs(df))
    val = filter_split(df, SPLIT_VALIDATION)
    raw_mean = float(val["forward_return_20d"].mean())
    winsor = apply_winsor_columns(val, frozen)
    winsor_mean = float(winsor["forward_return_20d_winsor"].mean())
    assert "forward_return_20d" in winsor.columns
    assert raw_mean > winsor_mean
    assert (winsor["forward_return_20d"] == val["forward_return_20d"]).all()


def test_median_quantile_and_winsorized_q5_q1() -> None:
    df = _panel_with_outlier()
    frozen = cutoff_map(freeze_target_cutoffs(df))
    q = quantile_table(df, ["ret_20d"], SPLIT_DEVELOPMENT, cutoffs=frozen)
    assert "forward_return_20d_mean" in q.columns
    assert "forward_return_20d_median" in q.columns
    assert "forward_return_20d_winsorized_mean_1pct" in q.columns
    s = spread_table(q)
    assert s["forward_excess_spy_20d_q5_minus_q1"][0] > 0
    assert s["forward_excess_spy_20d_q5_minus_q1_median"][0] > 0

    rq = robust_quantile_table(df, ["ret_20d"], SPLIT_VALIDATION, frozen)
    rs = robust_spread_table(rq)
    row = rs.filter(pl.col("target") == "forward_return_20d").row(0, named=True)
    assert row["raw_mean_spread"] is not None
    assert row["median_spread"] is not None
    assert row["winsorized_mean_spread"] is not None
    assert abs(row["raw_mean_spread"]) >= abs(row["winsorized_mean_spread"]) - 1e-12


def test_price_overlay_subsets_and_ic_floors() -> None:
    df = _panel_with_outlier()
    tables = overlay_tables(df, ["ret_20d"], SPLIT_VALIDATION)
    sizes = tables["sizes"]
    assert set(sizes["price_subset"].to_list()) >= {
        "research_all",
        "price_ge_1",
        "price_ge_5",
        "price_ge_20",
        "lt_1",
    }
    all_n = sizes.filter(pl.col("price_subset") == "research_all")["n"][0]
    ge1_n = sizes.filter(pl.col("price_subset") == "price_ge_1")["n"][0]
    assert ge1_n < all_n
    ge5 = filter_price_floor(df, 5.0)
    ic_all = ic_table(df, ["ret_20d"], SPLIT_VALIDATION)
    ic_ge5 = ic_table(ge5, ["ret_20d"], SPLIT_VALIDATION)
    assert ic_all.height and ic_ge5.height
    assert "mean_ic" in ic_all.columns
    assert "median_ic" in ic_all.columns
    assert "ic_ir" in ic_all.columns
    assert "positive_ic_ratio" in ic_all.columns


def test_low_price_concentration_flags_penny_tail() -> None:
    df = _panel_with_outlier()
    conc = low_price_concentration(df, SPLIT_VALIDATION, "forward_return_20d")
    top = conc.filter(pl.col("section") == "top_1pct")
    lt1 = top.filter(pl.col("price_bucket") == "lt_1")
    assert lt1.height == 1
    assert lt1["share"][0] > 0.5
    tickers = conc.filter(pl.col("section") == "ticker_concentration_top_1pct")
    assert "PENNY" in tickers["ticker"].to_list()


def test_feature_rank_correlation_flags_duplicate_ranks() -> None:
    df = _panel().with_columns((pl.col("ret_20d") + 0.05).alias("rel_spy_20d"))
    corr = feature_rank_correlation(df, ["ret_20d", "rel_spy_20d", "rsi_14"], SPLIT_DEVELOPMENT)
    pair = corr.filter((pl.col("feature_a") == "ret_20d") & (pl.col("feature_b") == "rel_spy_20d"))
    assert pair.height == 1
    assert pair["mean_rank_corr"][0] >= 0.95
    assert pair["redundant"][0] is True
    assert corr.filter(pl.col("redundant"))["feature_a"].len() >= 1


def test_overlapping_includes_median_and_extreme_dates() -> None:
    df = _panel()
    ov = overlapping_spread_table(df, ["ret_20d"], SPLIT_DEVELOPMENT)
    assert ov.height
    assert "median_spread" in ov.columns
    assert "positive_spread_ratio" in ov.columns
    assert "newey_west_tstat" in ov.columns
    from app.research.statistics import overlapping_spread_extreme_dates

    dates = overlapping_spread_extreme_dates(df, ["ret_20d"], SPLIT_DEVELOPMENT)
    assert dates.filter(pl.col("kind") == "best").height >= 1
    assert dates.filter(pl.col("kind") == "worst").height >= 1
