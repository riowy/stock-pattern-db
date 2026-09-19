"""Data-quality rules for features_daily. Findings only -- never drop/winsorize."""

from __future__ import annotations

from datetime import date

import polars as pl

Issue = dict


def _issue(security_id: str | None, d: date | None, issue_type: str, severity: str, details: str) -> Issue:
    return {
        "security_id": security_id,
        "date": d,
        "issue_type": issue_type,
        "severity": severity,
        "details": details,
    }


def check_rsi_bounds(df: pl.DataFrame) -> list[Issue]:
    if "rsi_14" not in df.columns:
        return []
    bad = df.filter(pl.col("rsi_14").is_not_null() & ((pl.col("rsi_14") < 0) | (pl.col("rsi_14") > 100)))
    return [
        _issue(row["security_id"], row["date"], "RSI_OUT_OF_BOUNDS", "critical", f"rsi_14={row['rsi_14']}")
        for row in bad.iter_rows(named=True)
    ]


def check_non_negative(df: pl.DataFrame, column: str, issue_type: str) -> list[Issue]:
    if column not in df.columns:
        return []
    bad = df.filter(pl.col(column).is_not_null() & (pl.col(column) < 0))
    return [
        _issue(row["security_id"], row["date"], issue_type, "critical", f"{column}={row[column]}")
        for row in bad.iter_rows(named=True)
    ]


def check_distance_high_tolerance(df: pl.DataFrame, tolerance: float = 0.01) -> list[Issue]:
    issues: list[Issue] = []
    for col in ("distance_high_20d", "distance_high_60d", "distance_high_252d"):
        if col not in df.columns:
            continue
        bad = df.filter(pl.col(col).is_not_null() & (pl.col(col) > tolerance))
        for row in bad.iter_rows(named=True):
            issues.append(
                _issue(
                    row["security_id"],
                    row["date"],
                    "DISTANCE_HIGH_ABOVE_TOLERANCE",
                    "warning",
                    f"{col}={row[col]} exceeds {tolerance}",
                )
            )
    return issues


def check_infinite_features(df: pl.DataFrame, columns: list[str]) -> list[Issue]:
    issues: list[Issue] = []
    for col in columns:
        if col not in df.columns:
            continue
        bad = df.filter(pl.col(col).is_not_null() & pl.col(col).is_infinite())
        for row in bad.iter_rows(named=True):
            issues.append(_issue(row["security_id"], row["date"], "NON_FINITE_FEATURE", "critical", f"{col}={row[col]}"))
    return issues


def check_duplicate_feature_keys(df: pl.DataFrame) -> list[Issue]:
    keys = ["security_id", "date", "feature_version"]
    if not all(k in df.columns for k in keys):
        return []
    dups = df.group_by(keys).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    return [
        _issue(row["security_id"], row["date"], "DUPLICATE_KEY", "critical", f"{row['n']} rows for {row['feature_version']}")
        for row in dups.iter_rows(named=True)
    ]


def run_feature_rules(df: pl.DataFrame) -> list[Issue]:
    from app.features.schema import FEATURE_VALUE_COLUMNS

    issues: list[Issue] = []
    issues += check_rsi_bounds(df)
    issues += check_non_negative(df, "atr_14", "NEGATIVE_ATR")
    issues += check_non_negative(df, "atr_pct_14", "NEGATIVE_ATR")
    issues += check_non_negative(df, "volume_ratio_5d", "NEGATIVE_VOLUME_RATIO")
    issues += check_non_negative(df, "volume_ratio_20d", "NEGATIVE_VOLUME_RATIO")
    issues += check_non_negative(df, "volume_ratio_60d", "NEGATIVE_VOLUME_RATIO")
    issues += check_non_negative(df, "volatility_10d", "NEGATIVE_VOLATILITY")
    issues += check_non_negative(df, "volatility_20d", "NEGATIVE_VOLATILITY")
    issues += check_non_negative(df, "volatility_60d", "NEGATIVE_VOLATILITY")
    issues += check_distance_high_tolerance(df)
    issues += check_infinite_features(df, FEATURE_VALUE_COLUMNS)
    issues += check_duplicate_feature_keys(df)
    return issues
