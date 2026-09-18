"""Incremental daily price sync (for ``stockdb sync-prices`` / ``run-daily``).

Re-fetches a short trailing window (default 10 calendar days) for the
*tracked* price universe (not the full security master -- see
``app.ingestion.price_backfill.resolve_symbols``) rather than trying to
track a precise per-symbol last-synced-date. This is deliberately simple
and self-healing: it also picks up late data corrections/adjustments from
the provider, and the Parquet dedup-on-read (last write wins by
``retrieved_at``) makes re-fetching already-known days perfectly safe -- no
duplicates, no special-casing.
"""

from __future__ import annotations

from datetime import timedelta

import duckdb

from app.config.settings import Settings
from app.ingestion.price_backfill import BackfillResult, run_price_ingestion
from app.services.market_calendar import MarketCalendarService

DEFAULT_LOOKBACK_DAYS = 10


def sync_recent_prices(
    settings: Settings,
    con: duckdb.DuckDBPyConnection,
    symbols: list[str] | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    dry_run: bool = False,
) -> BackfillResult:
    calendar = MarketCalendarService(settings.market_calendar, settings.market_data_grace_minutes)
    # Anchor the trailing window on the latest *expected* trading session
    # (America/New_York-aware) rather than the server's own local "today",
    # so a server running in Korea doesn't compute the wrong window.
    reference_day = calendar.latest_expected_session()
    start = reference_day - timedelta(days=lookback_days)
    return run_price_ingestion(
        settings,
        con,
        symbols=symbols,
        start=start,
        end=None,
        batch_size=settings.price_batch_size,
        resume=False,
        dry_run=dry_run,
        job_name="sync_prices",
        default_scope="tracked",
    )
