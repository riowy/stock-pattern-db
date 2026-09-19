"""Validation orchestration: run rules over the price lake and persist findings.

Findings are never used to silently delete or "fix" data -- they are only
recorded in ``data_quality_issues`` for manual review (see spec section 9).
Issue ids are deterministic hashes of (dataset, security_id, date,
issue_type) so re-running validation repeatedly upserts the same rows
instead of accumulating duplicates.

Scope (see project task "1. Tracked Universe 개념 추가" / "Validation 수정"):
by default, ``MISSING_RECENT_DATA`` (and the row-level rules) only run
against the small *tracked* price universe, never the full ~10k-security
security master -- checking "is AAPL's price data current" is meaningful,
checking "does this SEC-registered company that we have never backfilled
have current price data" is not, and previously produced ~10,000 bogus
warnings. Pass ``all_universe=True`` (``--all-universe``) to explicitly
widen the check to every active security when that is actually what you
want.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime

import duckdb
import polars as pl

from app.config.settings import Settings
from app.db.schema import create_lake_views
from app.services.market_calendar import MarketCalendarService
from app.services.tracked_universe_service import get_tracked_price_security_ids
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
    scope: str = "tracked"  # tracked | all_universe | explicit_symbols
    scope_size: int = 0
    resolved_stale_issues: int = 0


def _issue_id(dataset: str, security_id: str | None, d, issue_type: str) -> str:
    key = f"{dataset}|{security_id}|{d}|{issue_type}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _persist_issues(con: duckdb.DuckDBPyConnection, dataset: str, issues: list[dict]) -> set[str]:
    """Bulk upsert findings via a registered relation (see universe_sync for
    why this is much faster than one ``INSERT`` per row at any real scale).
    Returns the set of issue_ids that are (still) open after this call."""
    issue_ids = {_issue_id(dataset, i.get("security_id"), i.get("date"), i["issue_type"]) for i in issues}
    if not issues:
        return issue_ids

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
                detected_at = excluded.detected_at,
                resolved = FALSE
            """
        )
    finally:
        con.unregister("_tmp_dq_issues")
    return issue_ids


def _resolve_stale_issues_in_scope(
    con: duckdb.DuckDBPyConnection, dataset: str, scope_security_ids: list[str], still_open_issue_ids: set[str]
) -> int:
    """For every security we just (re-)checked, close any previously-open
    issue that was not reproduced this run (e.g. a data problem got fixed,
    or a security caught back up on missing data). History is preserved --
    rows are marked ``resolved``, never deleted."""
    if not scope_security_ids:
        return 0
    ids_df = pl.DataFrame({"security_id": scope_security_ids})
    open_df = pl.DataFrame({"issue_id": list(still_open_issue_ids)}, schema={"issue_id": pl.Utf8})
    con.register("_tmp_scope_ids", ids_df)
    con.register("_tmp_still_open", open_df)
    try:
        before = con.execute(
            "SELECT count(*) FROM data_quality_issues WHERE dataset = ? AND NOT resolved", [dataset]
        ).fetchone()[0]
        con.execute(
            """
            UPDATE data_quality_issues
            SET resolved = TRUE
            WHERE dataset = ? AND NOT resolved
              AND security_id IN (SELECT security_id FROM _tmp_scope_ids)
              AND issue_id NOT IN (SELECT issue_id FROM _tmp_still_open)
            """,
            [dataset],
        )
        after = con.execute(
            "SELECT count(*) FROM data_quality_issues WHERE dataset = ? AND NOT resolved", [dataset]
        ).fetchone()[0]
        return before - after
    finally:
        con.unregister("_tmp_scope_ids")
        con.unregister("_tmp_still_open")


def _resolve_out_of_scope_missing_recent_data(
    con: duckdb.DuckDBPyConnection, dataset: str, tracked_ids: list[str]
) -> int:
    """One-time-safe cleanup: ``MISSING_RECENT_DATA`` is only meaningful for
    the tracked price universe. Any such issue recorded for a security that
    is *not* tracked (e.g. from before the tracked-universe concept existed)
    no longer applies and is marked resolved, regardless of whether this
    run's scope happens to include that security."""
    ids_df = pl.DataFrame({"security_id": tracked_ids}) if tracked_ids else pl.DataFrame({"security_id": pl.Series([], dtype=pl.Utf8)})
    con.register("_tmp_tracked_ids", ids_df)
    try:
        before = con.execute(
            "SELECT count(*) FROM data_quality_issues WHERE dataset = ? AND issue_type = 'MISSING_RECENT_DATA' AND NOT resolved",
            [dataset],
        ).fetchone()[0]
        con.execute(
            """
            UPDATE data_quality_issues
            SET resolved = TRUE
            WHERE dataset = ? AND issue_type = 'MISSING_RECENT_DATA' AND NOT resolved
              AND (security_id IS NULL OR security_id NOT IN (SELECT security_id FROM _tmp_tracked_ids))
            """,
            [dataset],
        )
        after = con.execute(
            "SELECT count(*) FROM data_quality_issues WHERE dataset = ? AND issue_type = 'MISSING_RECENT_DATA' AND NOT resolved",
            [dataset],
        ).fetchone()[0]
        return before - after
    finally:
        con.unregister("_tmp_tracked_ids")


def validate_prices(
    settings: Settings,
    con: duckdb.DuckDBPyConnection,
    symbols: list[str] | None = None,
    dry_run: bool = False,
    all_universe: bool = False,
) -> ValidationSummary:
    create_lake_views(con, settings)

    from app.config.lake_datasets import get_lake_dataset

    prices_dataset = get_lake_dataset(settings.lake_dir, "prices_daily")
    tracked_ids = get_tracked_price_security_ids(con)

    if not prices_dataset.has_any_files():
        logger.warning("No price data found in the lake yet -- nothing to validate.")
        if not dry_run and not all_universe:
            _resolve_out_of_scope_missing_recent_data(con, "prices_daily", tracked_ids)
        return ValidationSummary("prices_daily", 0, 0, scope="tracked" if not all_universe else "all_universe")

    if symbols:
        scope_label = "explicit_symbols"
        placeholders = ", ".join("?" for _ in symbols)
        query = f"SELECT * FROM prices_daily WHERE ticker_at_time IN ({placeholders})"
        df = con.execute(query, [s.upper() for s in symbols]).pl()
        scope_ids = df.select("security_id").unique().to_series().to_list()
    elif all_universe:
        scope_label = "all_universe"
        scope_ids = [r[0] for r in con.execute("SELECT security_id FROM securities WHERE is_active = TRUE").fetchall()]
        df = con.execute("SELECT * FROM prices_daily").pl()
    else:
        scope_label = "tracked"
        scope_ids = tracked_ids
        if not scope_ids:
            logger.warning(
                "Tracked price universe is empty -- nothing to validate. "
                "Use 'stockdb universe add <TICKER>...' or backfill prices first."
            )
            if not dry_run:
                _resolve_out_of_scope_missing_recent_data(con, "prices_daily", tracked_ids)
            return ValidationSummary("prices_daily", 0, 0, scope=scope_label, scope_size=0)
        placeholders = ", ".join("?" for _ in scope_ids)
        df = con.execute(f"SELECT * FROM prices_daily WHERE security_id IN ({placeholders})", scope_ids).pl()

    if df.height == 0:
        logger.warning("Price view returned 0 rows for the current scope -- nothing to validate.")
        if not dry_run and not all_universe:
            _resolve_out_of_scope_missing_recent_data(con, "prices_daily", tracked_ids)
        return ValidationSummary("prices_daily", 0, 0, scope=scope_label, scope_size=len(scope_ids))

    calendar = MarketCalendarService(settings.market_calendar, settings.market_data_grace_minutes)

    all_issues: list[dict] = []
    all_issues += check_negative_values(df)
    all_issues += check_ohlc_consistency(df)
    all_issues += check_weekend_dates(df)
    all_issues += check_invalid_dates(df)
    all_issues += check_nan_ratio(df)
    all_issues += check_sudden_price_change(df)
    all_issues += check_missing_recent_data(df, scope_ids, calendar)

    critical = sum(1 for i in all_issues if i["severity"] == "critical")
    warning = sum(1 for i in all_issues if i["severity"] == "warning")
    info = sum(1 for i in all_issues if i["severity"] == "info")
    by_type: dict[str, int] = {}
    for i in all_issues:
        by_type[i["issue_type"]] = by_type.get(i["issue_type"], 0) + 1

    resolved_count = 0
    if not dry_run:
        still_open_ids = _persist_issues(con, "prices_daily", all_issues)
        resolved_count += _resolve_stale_issues_in_scope(con, "prices_daily", scope_ids, still_open_ids)
        if not all_universe:
            # Standing cleanup: MISSING_RECENT_DATA only ever makes sense for
            # tracked securities, regardless of this run's exact scope.
            resolved_count += _resolve_out_of_scope_missing_recent_data(con, "prices_daily", tracked_ids)

    logger.info(
        "Validation complete (scope=%s, %d securities): %d rows checked, %d issues found "
        "(%d critical, %d warning, %d info), %d stale issue(s) resolved",
        scope_label,
        len(scope_ids),
        df.height,
        len(all_issues),
        critical,
        warning,
        info,
        resolved_count,
    )

    return ValidationSummary(
        dataset="prices_daily",
        rows_checked=df.height,
        issues_found=len(all_issues),
        critical=critical,
        warning=warning,
        info=info,
        by_type=by_type,
        scope=scope_label,
        scope_size=len(scope_ids),
        resolved_stale_issues=resolved_count,
    )
