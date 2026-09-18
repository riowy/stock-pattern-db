"""Validation orchestration: run rules over the price lake and persist findings.

Findings are never used to silently delete or "fix" data -- they are only
recorded in ``data_quality_issues`` for manual review (see spec section 9).
Issue ids are deterministic hashes of (dataset, security_id, date,
issue_type) so re-running validation repeatedly upserts the same rows
instead of accumulating duplicates.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import duckdb
import polars as pl

from app.config.settings import Settings
from app.db.schema import create_lake_views
from app.utils.logging import get_logger
from app.validation.rules import (
    check_invalid_dates,
    check_missing_recent_data,
    check_nan_ratio,
    check_negative_values,
    check_ohlc_consistency,
    check_sudden_price_change,
    check_weekend_dates,
)

logger = get_logger("validation")


@dataclass
class ValidationSummary:
    dataset: str
    rows_checked: int
    issues_found: int
    critical: int = 0
    warning: int = 0
    info: int = 0
    by_type: dict[str, int] = field(default_factory=dict)


def _issue_id(dataset: str, security_id: str | None, d: date | None, issue_type: str) -> str:
    key = f"{dataset}|{security_id}|{d}|{issue_type}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _persist_issues(con: duckdb.DuckDBPyConnection, dataset: str, issues: list[dict]) -> None:
    """Bulk upsert findings via a registered relation (see universe_sync for
    why this is much faster than one ``INSERT`` per row at any real scale)."""
    if not issues:
        return
    now = datetime.now(UTC)
    records = [
        {
            "issue_id": _issue_id(dataset, i.get("security_id"), i.get("date"), i["issue_type"]),
            "dataset": dataset,
            "security_id": i.get("security_id"),
            "date": i.get("date"),
            "issue_type": i["issue_type"],
            "severity": i["severity"],
            "details": i["details"],
            "detected_at": now,
        }
        for i in issues
    ]
    df = pl.DataFrame(records)
    con.register("_tmp_dq_issues", df)
    try:
        con.execute(
            """
            INSERT INTO data_quality_issues
                (issue_id, dataset, security_id, date, issue_type, severity, details, detected_at, resolved)
            SELECT issue_id, dataset, security_id, date, issue_type, severity, details, detected_at, FALSE
            FROM _tmp_dq_issues
            ON CONFLICT (issue_id) DO UPDATE SET
                severity = excluded.severity,
                details = excluded.details,
                detected_at = excluded.detected_at
            """
        )
    finally:
        con.unregister("_tmp_dq_issues")


def validate_prices(
    settings: Settings,
    con: duckdb.DuckDBPyConnection,
    symbols: list[str] | None = None,
    dry_run: bool = False,
) -> ValidationSummary:
    create_lake_views(con, settings)

    from app.utils.parquet_io import LakeDataset

    prices_dataset = LakeDataset(settings.prices_daily_dir, "date", ["security_id", "date"], ["security_id", "date"])
    if not prices_dataset.has_any_files():
        logger.warning("No price data found in the lake yet -- nothing to validate.")
        return ValidationSummary("prices_daily", 0, 0)

    if symbols:
        placeholders = ", ".join("?" for _ in symbols)
        query = f"SELECT * FROM prices_daily WHERE ticker_at_time IN ({placeholders})"
        df = con.execute(query, [s.upper() for s in symbols]).pl()
    else:
        df = con.execute("SELECT * FROM prices_daily").pl()

    if df.height == 0:
        logger.warning("Price view returned 0 rows -- nothing to validate.")
        return ValidationSummary("prices_daily", 0, 0)

    active_ids = [
        r[0] for r in con.execute("SELECT security_id FROM securities WHERE is_active = TRUE").fetchall()
    ]
    if symbols:
        active_ids = df.select("security_id").unique().to_series().to_list()

    all_issues: list[dict] = []
    all_issues += check_negative_values(df)
    all_issues += check_ohlc_consistency(df)
    all_issues += check_weekend_dates(df)
    all_issues += check_invalid_dates(df)
    all_issues += check_nan_ratio(df)
    all_issues += check_sudden_price_change(df)
    all_issues += check_missing_recent_data(df, active_ids, date.today())

    critical = sum(1 for i in all_issues if i["severity"] == "critical")
    warning = sum(1 for i in all_issues if i["severity"] == "warning")
    info = sum(1 for i in all_issues if i["severity"] == "info")
    by_type: dict[str, int] = {}
    for i in all_issues:
        by_type[i["issue_type"]] = by_type.get(i["issue_type"], 0) + 1

    if not dry_run:
        _persist_issues(con, "prices_daily", all_issues)

    logger.info(
        "Validation complete: %d rows checked, %d issues found (%d critical, %d warning, %d info)",
        df.height,
        len(all_issues),
        critical,
        warning,
        info,
    )

    return ValidationSummary(
        dataset="prices_daily",
        rows_checked=df.height,
        issues_found=len(all_issues),
        critical=critical,
        warning=warning,
        info=info,
        by_type=by_type,
    )
