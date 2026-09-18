"""Security master ("universe") sync: SEC EDGAR + ETF seed list -> DuckDB.

Idempotent by design:
* ``securities`` rows are upserted (``ON CONFLICT security_id DO UPDATE``),
  never re-inserted; ``first_seen_at`` / ``created_at`` are preserved.
* ``security_identifiers`` history is preserved: if a ticker changes, the
  old identifier row is closed (``valid_to``) and a new one opened, instead
  of being overwritten in place.
* ``security_snapshots`` uses ``(snapshot_date, security_id)`` as primary
  key, so re-running the sync multiple times on the same day updates that
  day's snapshot in place rather than duplicating rows, while a new day
  always creates a new historical snapshot row.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import duckdb
import polars as pl

from app.config.etf_seed import SEED_ETF_TICKERS
from app.config.settings import Settings
from app.models.security import Security, SecurityIdentifier, SecuritySnapshot
from app.normalization.security_master import (
    NormalizedUniverse,
    build_etf_seed_universe,
    normalize_sec_universe_rows,
)
from app.providers.security_master.sec_provider import SecUniverseProvider
from app.services.manifest_service import finish_run, record_source_file, start_run
from app.utils.logging import get_logger

logger = get_logger("universe_sync")


@dataclass
class UniverseSyncResult:
    run_id: str | None
    securities_seen: int
    identifiers_seen: int
    snapshots_written: int
    dry_run: bool


def _bulk_upsert_securities(con: duckdb.DuckDBPyConnection, securities: list[Security]) -> None:
    """Vectorized upsert via a registered relation instead of one query per row.

    Thousands of individual ``INSERT ... ON CONFLICT`` round-trips (even via
    ``executemany``) are dominated by per-statement overhead rather than
    actual data volume. Registering the batch as a DuckDB relation and doing
    a single ``INSERT ... SELECT ... ON CONFLICT`` is column-vectorized and
    orders of magnitude faster for the ~10k-security SEC universe.
    """
    if not securities:
        return
    df = pl.DataFrame([s.model_dump() for s in securities])
    con.register("_tmp_securities", df)
    try:
        con.execute(
            """
            INSERT INTO securities
                (security_id, cik, company_name, primary_ticker, exchange, asset_type,
                 currency, is_active, first_seen_at, last_seen_at, created_at, updated_at)
            SELECT security_id, cik, company_name, primary_ticker, exchange, asset_type,
                   currency, is_active, first_seen_at, last_seen_at, created_at, updated_at
            FROM _tmp_securities
            ON CONFLICT (security_id) DO UPDATE SET
                cik = excluded.cik,
                company_name = excluded.company_name,
                primary_ticker = excluded.primary_ticker,
                exchange = excluded.exchange,
                asset_type = excluded.asset_type,
                currency = excluded.currency,
                is_active = excluded.is_active,
                last_seen_at = excluded.last_seen_at,
                updated_at = excluded.updated_at
            """
        )
    finally:
        con.unregister("_tmp_securities")


def _bulk_upsert_identifiers(con: duckdb.DuckDBPyConnection, identifiers: list[SecurityIdentifier]) -> None:
    """Close changed identifiers and insert new/unchanged ones, fully set-based."""
    if not identifiers:
        return
    df = pl.DataFrame([i.model_dump() for i in identifiers])
    con.register("_tmp_identifiers", df)
    try:
        # Close any currently-open identifier whose value differs from the newly observed one.
        con.execute(
            """
            UPDATE security_identifiers AS si
            SET valid_to = new.valid_from
            FROM _tmp_identifiers AS new
            WHERE si.security_id = new.security_id
              AND si.identifier_type = new.identifier_type
              AND si.valid_to IS NULL
              AND si.identifier_value <> new.identifier_value
            """
        )
        # Insert rows that are not already open with the exact same value (avoids
        # duplicating unchanged identifiers while still inserting genuinely new ones).
        con.execute(
            """
            INSERT INTO security_identifiers
                (security_id, identifier_type, identifier_value, valid_from, valid_to, source)
            SELECT DISTINCT new.security_id, new.identifier_type, new.identifier_value,
                   new.valid_from, new.valid_to, new.source
            FROM _tmp_identifiers AS new
            WHERE NOT EXISTS (
                SELECT 1 FROM security_identifiers si
                WHERE si.security_id = new.security_id
                  AND si.identifier_type = new.identifier_type
                  AND si.valid_to IS NULL
                  AND si.identifier_value = new.identifier_value
            )
            ON CONFLICT (security_id, identifier_type, identifier_value, valid_from) DO NOTHING
            """
        )
    finally:
        con.unregister("_tmp_identifiers")


def _bulk_upsert_snapshots(con: duckdb.DuckDBPyConnection, snapshots: list[SecuritySnapshot]) -> None:
    if not snapshots:
        return
    df = pl.DataFrame([snap.model_dump() for snap in snapshots])
    con.register("_tmp_snapshots", df)
    try:
        con.execute(
            """
            INSERT INTO security_snapshots
                (snapshot_date, security_id, ticker, company_name, exchange, source)
            SELECT snapshot_date, security_id, ticker, company_name, exchange, source
            FROM _tmp_snapshots
            ON CONFLICT (snapshot_date, security_id) DO UPDATE SET
                ticker = excluded.ticker,
                company_name = excluded.company_name,
                exchange = excluded.exchange,
                source = excluded.source
            """
        )
    finally:
        con.unregister("_tmp_snapshots")


def sync_universe(
    settings: Settings, con: duckdb.DuckDBPyConnection, dry_run: bool = False
) -> UniverseSyncResult:
    if dry_run:
        # True dry-run: no HTTP call to SEC, no ingest_runs row. We can only
        # report what's already known locally (current security count +
        # configured ETF seed list size), not what SEC would return today.
        current_count = con.execute("SELECT count(*) FROM securities").fetchone()[0]
        logger.info(
            "[dry-run] would sync SEC company_tickers_exchange.json + %d ETF seed tickers "
            "(no request made; %d securities currently known locally)",
            len(SEED_ETF_TICKERS),
            current_count,
        )
        return UniverseSyncResult(run_id=None, securities_seen=current_count, identifiers_seen=0, snapshots_written=0, dry_run=True)

    provider = SecUniverseProvider(settings)
    run_id = start_run(con, provider.capabilities.provider_name, "security_master", {"dry_run": dry_run})

    try:
        fetch = provider.fetch_universe()
        snapshot_date = fetch.retrieved_at.date()

        sec_universe = normalize_sec_universe_rows(fetch.rows, fetch.retrieved_at, snapshot_date)
        known_tickers = {s.primary_ticker for s in sec_universe.securities if s.primary_ticker}
        etf_universe = build_etf_seed_universe(SEED_ETF_TICKERS, known_tickers, fetch.retrieved_at, snapshot_date)

        combined = NormalizedUniverse(
            securities=sec_universe.securities + etf_universe.securities,
            identifiers=sec_universe.identifiers + etf_universe.identifiers,
            snapshots=sec_universe.snapshots + etf_universe.snapshots,
        )

        _bulk_upsert_securities(con, combined.securities)
        _bulk_upsert_identifiers(con, combined.identifiers)
        _bulk_upsert_snapshots(con, combined.snapshots)

        raw_path = fetch.extra.get("raw_file_path")
        if raw_path:
            record_source_file(
                con,
                run_id,
                provider.capabilities.provider_name,
                "security_master",
                raw_path,
                fetch.retrieved_at,
                row_count=len(fetch.rows),
            )

        finish_run(
            con,
            run_id,
            "success",
            requested_items=len(combined.securities),
            successful_items=len(combined.securities),
            failed_items=0,
            rows_written=len(combined.securities),
        )
        logger.info(
            "Universe sync complete: %d securities (%d from SEC, %d ETF-seed fallback)",
            len(combined.securities),
            len(sec_universe.securities),
            len(etf_universe.securities),
        )
        return UniverseSyncResult(run_id, len(combined.securities), len(combined.identifiers), len(combined.snapshots), False)

    except Exception as exc:  # noqa: BLE001
        finish_run(con, run_id, "failed", error_message=str(exc))
        raise
