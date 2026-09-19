"""Tracked (operational) universe management.

``securities`` = everything known from the security master (~10k rows).
``tracked_securities`` = the deliberately small subset we actually collect
data for and validate. Daily jobs and default validation must read from
here, never from the full security master.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import duckdb
import polars as pl

from app.config.lake_datasets import get_lake_dataset
from app.config.settings import Settings


@dataclass
class TrackedRow:
    security_id: str
    primary_ticker: str | None
    company_name: str | None
    enabled: bool
    price_tracking: bool
    filings_tracking: bool
    feature_tracking: bool
    tracking_reason: str | None
    added_at: datetime
    removed_at: datetime | None
    notes: str | None


def add_tracked(
    con: duckdb.DuckDBPyConnection,
    security_ids: list[str],
    reason: str,
    price_tracking: bool = True,
    filings_tracking: bool = True,
    feature_tracking: bool = False,
    notes: str | None = None,
) -> int:
    """Upsert: re-enables + updates flags for existing rows, inserts new ones."""
    if not security_ids:
        return 0
    now = datetime.now(UTC)
    df = pl.DataFrame(
        {
            "security_id": security_ids,
            "enabled": [True] * len(security_ids),
            "tracking_reason": [reason] * len(security_ids),
            "added_at": [now] * len(security_ids),
            "price_tracking": [price_tracking] * len(security_ids),
            "filings_tracking": [filings_tracking] * len(security_ids),
            "feature_tracking": [feature_tracking] * len(security_ids),
            "notes": [notes] * len(security_ids),
        }
    )
    con.register("_tmp_tracked_add", df)
    try:
        con.execute(
            """
            INSERT INTO tracked_securities
                (security_id, enabled, tracking_reason, added_at, removed_at,
                 price_tracking, filings_tracking, feature_tracking, notes)
            SELECT security_id, enabled, tracking_reason, added_at, NULL,
                   price_tracking, filings_tracking, feature_tracking, notes
            FROM _tmp_tracked_add
            ON CONFLICT (security_id) DO UPDATE SET
                enabled = TRUE,
                removed_at = NULL,
                tracking_reason = excluded.tracking_reason,
                price_tracking = excluded.price_tracking OR tracked_securities.price_tracking,
                filings_tracking = excluded.filings_tracking OR tracked_securities.filings_tracking,
                feature_tracking = excluded.feature_tracking OR tracked_securities.feature_tracking,
                notes = COALESCE(excluded.notes, tracked_securities.notes)
            """
        )
    finally:
        con.unregister("_tmp_tracked_add")
    return len(security_ids)


def add_tracked_if_absent(
    con: duckdb.DuckDBPyConnection,
    security_ids: list[str],
    reason: str,
    price_tracking: bool = True,
    filings_tracking: bool = True,
) -> int:
    """Insert-only variant used by migrations: never touches a row that
    already exists (so it can never silently undo a user's explicit
    ``remove``). Returns the number of genuinely new rows inserted.

    ``filings_tracking`` defaults to True here too -- the whole point of the
    tracked universe is that it stays small, so per-security SEC filing
    checks for the tracked set are cheap even though checking all ~10k
    securities in ``securities`` every day would not be (see
    ``app/ingestion/filings_sync.py``).
    """
    if not security_ids:
        return 0
    now = datetime.now(UTC)
    df = pl.DataFrame(
        {
            "security_id": security_ids,
            "enabled": [True] * len(security_ids),
            "tracking_reason": [reason] * len(security_ids),
            "added_at": [now] * len(security_ids),
            "price_tracking": [price_tracking] * len(security_ids),
            "filings_tracking": [filings_tracking] * len(security_ids),
        }
    )
    before = con.execute("SELECT count(*) FROM tracked_securities").fetchone()[0]
    con.register("_tmp_tracked_add_if_absent", df)
    try:
        con.execute(
            """
            INSERT INTO tracked_securities
                (security_id, enabled, tracking_reason, added_at, removed_at,
                 price_tracking, filings_tracking, feature_tracking, notes)
            SELECT security_id, enabled, tracking_reason, added_at, NULL,
                   price_tracking, filings_tracking, FALSE, NULL
            FROM _tmp_tracked_add_if_absent
            ON CONFLICT (security_id) DO NOTHING
            """
        )
    finally:
        con.unregister("_tmp_tracked_add_if_absent")
    after = con.execute("SELECT count(*) FROM tracked_securities").fetchone()[0]
    return after - before


def remove_tracked(con: duckdb.DuckDBPyConnection, security_ids: list[str]) -> int:
    """Soft-remove: sets enabled=FALSE + removed_at. History is preserved."""
    if not security_ids:
        return 0
    now = datetime.now(UTC)
    placeholders = ", ".join("?" for _ in security_ids)
    con.execute(
        f"""
        UPDATE tracked_securities
        SET enabled = FALSE, removed_at = ?
        WHERE security_id IN ({placeholders}) AND enabled = TRUE
        """,
        [now, *security_ids],
    )
    return con.execute(
        f"SELECT count(*) FROM tracked_securities WHERE security_id IN ({placeholders}) AND removed_at = ?",
        [*security_ids, now],
    ).fetchone()[0]


def get_tracked_price_security_ids(con: duckdb.DuckDBPyConnection) -> list[str]:
    rows = con.execute(
        "SELECT security_id FROM tracked_securities WHERE enabled = TRUE AND price_tracking = TRUE ORDER BY security_id"
    ).fetchall()
    return [r[0] for r in rows]


def get_tracked_filing_ciks(con: duckdb.DuckDBPyConnection) -> dict[str, list[str]]:
    """{cik: [security_id, ...]} for securities with filings_tracking enabled."""
    rows = con.execute(
        """
        SELECT s.cik, s.security_id
        FROM tracked_securities t
        JOIN securities s ON s.security_id = t.security_id
        WHERE t.enabled = TRUE AND t.filings_tracking = TRUE AND s.cik IS NOT NULL
        ORDER BY s.cik
        """
    ).fetchall()
    grouped: dict[str, list[str]] = {}
    for cik, security_id in rows:
        grouped.setdefault(cik, []).append(security_id)
    return grouped


def list_tracked(con: duckdb.DuckDBPyConnection, include_disabled: bool = False) -> list[TrackedRow]:
    where = "" if include_disabled else "WHERE t.enabled = TRUE"
    rows = con.execute(
        f"""
        SELECT t.security_id, s.primary_ticker, s.company_name, t.enabled, t.price_tracking,
               t.filings_tracking, t.feature_tracking, t.tracking_reason, t.added_at, t.removed_at, t.notes
        FROM tracked_securities t
        LEFT JOIN securities s ON s.security_id = t.security_id
        {where}
        ORDER BY s.primary_ticker
        """
    ).fetchall()
    return [
        TrackedRow(
            security_id=r[0], primary_ticker=r[1], company_name=r[2], enabled=r[3], price_tracking=r[4],
            filings_tracking=r[5], feature_tracking=r[6], tracking_reason=r[7], added_at=r[8],
            removed_at=r[9], notes=r[10],
        )
        for r in rows
    ]


def auto_register_from_existing_price_data(con: duckdb.DuckDBPyConnection, settings: Settings) -> int:
    """Idempotent migration: any security that already has price data in the
    lake becomes tracked (for prices) if it is not already a tracked_securities
    row (existing rows -- including deliberately removed ones -- are never
    touched). Safe to call on every startup."""
    ds = get_lake_dataset(settings.lake_dir, "prices_daily")
    if not ds.has_any_files():
        return 0
    glob_path = ds.glob_pattern().replace("'", "''")
    rows = con.execute(
        f"SELECT DISTINCT security_id FROM read_parquet('{glob_path}', hive_partitioning = true)"
    ).fetchall()
    security_ids = [r[0] for r in rows]
    return add_tracked_if_absent(con, security_ids, reason="auto_registered_existing_price_data", price_tracking=True)
