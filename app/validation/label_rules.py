"""Data-quality rules for labels_forward_returns. Findings only -- never drop/winsorize."""

from __future__ import annotations

from datetime import date

import polars as pl

from app.labels.schema import FORWARD_RETURN_COLUMNS, LABEL_VALUE_COLUMNS

Issue = dict


def _issue(security_id: str | None, d: date | None, issue_type: str, severity: str, details: str) -> Issue:
    return {
        "security_id": security_id,
        "date": d,
        "issue_type": issue_type,
        "severity": severity,
        "details": details,
    }


def check_return_below_minus_one(df: pl.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    for col in FORWARD_RETURN_COLUMNS:
        if col not in df.columns:
            continue
        bad = df.filter(pl.col(col).is_not_null() & (pl.col(col) < -1))
        for row in bad.iter_rows(named=True):
            issues.append(
                _issue(row["security_id"], row["date"], "FORWARD_RETURN_BELOW_MINUS_ONE", "critical", f"{col}={row[col]}")
            )
    return issues


def check_extreme_labels(df: pl.DataFrame, threshold: float = 2.0) -> list[Issue]:
    """|label| > threshold is a warning for review -- not auto-deleted."""
    issues: list[Issue] = []
    for col in LABEL_VALUE_COLUMNS:
        if col not in df.columns:
            continue
        bad = df.filter(pl.col(col).is_not_null() & (pl.col(col).abs() > threshold))
        for row in bad.iter_rows(named=True):
            issues.append(
                _issue(row["security_id"], row["date"], "EXTREME_LABEL", "warning", f"{col}={row[col]} exceeds ±{threshold}")
            )
    return issues


def check_duplicate_label_keys(df: pl.DataFrame) -> list[Issue]:
    keys = ["security_id", "date", "label_version"]
    if not all(k in df.columns for k in keys):
        return []
    dups = df.group_by(keys).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    return [
        _issue(row["security_id"], row["date"], "DUPLICATE_KEY", "critical", f"{row['n']} rows for {row['label_version']}")
        for row in dups.iter_rows(named=True)
    ]


def run_label_rules(df: pl.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    issues += check_return_below_minus_one(df)
    issues += check_extreme_labels(df)
    issues += check_duplicate_label_keys(df)
    return issues
