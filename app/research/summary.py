"""Factual per-feature research summary. No ranking, scoring, or candidate flags."""

from __future__ import annotations

import polars as pl

from app.research.config import SPLIT_DEVELOPMENT, SPLIT_VALIDATION, SUMMARY_TARGET
from app.research.overlay import filter_price_floor
from app.research.robust import robust_spread_table
from app.research.statistics import ic_table, overlapping_spread_table


def _lookup(df: pl.DataFrame, feature: str, split: str, col: str, target: str = SUMMARY_TARGET) -> float | None:
    if df.height == 0:
        return None
    filt = df
    if "feature" in df.columns:
        filt = filt.filter(pl.col("feature") == feature)
    if "split" in filt.columns:
        filt = filt.filter(pl.col("split") == split)
    if "target" in filt.columns:
        filt = filt.filter(pl.col("target") == target)
    if filt.height == 0:
        return None
    val = filt[col][0]
    return float(val) if val is not None else None


def feature_research_summary(
    df: pl.DataFrame,
    features: list[str],
    *,
    ic: pl.DataFrame,
    overlapping: pl.DataFrame,
    robust_spreads: pl.DataFrame,
    target: str = SUMMARY_TARGET,
) -> pl.DataFrame:
    ge5 = filter_price_floor(df, 5.0)
    ic_ge5 = pl.concat(
        [
            ic_table(ge5, features, SPLIT_DEVELOPMENT),
            ic_table(ge5, features, SPLIT_VALIDATION),
        ],
        how="diagonal_relaxed",
    ) if ge5.height else pl.DataFrame()
    ov_ge5 = pl.concat(
        [
            overlapping_spread_table(ge5, features, SPLIT_DEVELOPMENT),
            overlapping_spread_table(ge5, features, SPLIT_VALIDATION),
        ],
        how="diagonal_relaxed",
    ) if ge5.height else pl.DataFrame()
    from app.research.robust import robust_quantile_table

    rq_ge5 = pl.concat(
        [
            robust_quantile_table(ge5, features, SPLIT_DEVELOPMENT),
            robust_quantile_table(ge5, features, SPLIT_VALIDATION),
        ],
        how="diagonal_relaxed",
    ) if ge5.height else pl.DataFrame()
    sp_ge5 = robust_spread_table(rq_ge5) if rq_ge5.height else pl.DataFrame()

    rows: list[dict] = []
    for feature in features:
        dev = df.filter(pl.col("split") == SPLIT_DEVELOPMENT) if "split" in df.columns else df
        val = df.filter(pl.col("split") == SPLIT_VALIDATION) if "split" in df.columns else df
        rows.append(
            {
                "feature": feature,
                "target": target,
                "development_ic": _lookup(ic, feature, SPLIT_DEVELOPMENT, "mean_ic", target),
                "validation_ic": _lookup(ic, feature, SPLIT_VALIDATION, "mean_ic", target),
                "development_overlapping_nw_t": _lookup(
                    overlapping, feature, SPLIT_DEVELOPMENT, "newey_west_tstat", target
                ),
                "validation_overlapping_nw_t": _lookup(
                    overlapping, feature, SPLIT_VALIDATION, "newey_west_tstat", target
                ),
                "development_median_q5_q1": _lookup(
                    robust_spreads, feature, SPLIT_DEVELOPMENT, "median_spread", target
                ),
                "validation_median_q5_q1": _lookup(
                    robust_spreads, feature, SPLIT_VALIDATION, "median_spread", target
                ),
                "development_n": int(dev.height) if dev.height else 0,
                "validation_n": int(val.height) if val.height else 0,
                "sample_count": int(df.height) if df.height else 0,
                "price_ge_5_development_ic": _lookup(ic_ge5, feature, SPLIT_DEVELOPMENT, "mean_ic", target),
                "price_ge_5_validation_ic": _lookup(ic_ge5, feature, SPLIT_VALIDATION, "mean_ic", target),
                "price_ge_5_development_median_q5_q1": _lookup(
                    sp_ge5, feature, SPLIT_DEVELOPMENT, "median_spread", target
                ),
                "price_ge_5_validation_median_q5_q1": _lookup(
                    sp_ge5, feature, SPLIT_VALIDATION, "median_spread", target
                ),
                "price_ge_5_development_overlapping_nw_t": _lookup(
                    ov_ge5, feature, SPLIT_DEVELOPMENT, "newey_west_tstat", target
                ),
                "price_ge_5_validation_overlapping_nw_t": _lookup(
                    ov_ge5, feature, SPLIT_VALIDATION, "newey_west_tstat", target
                ),
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()
