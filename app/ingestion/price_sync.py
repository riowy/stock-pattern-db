"""Incremental daily price sync with a per-security fast path.

When every tracked price target already has the latest completed XNYS
session, the provider is not instantiated and no network fetch runs.
Otherwise only STALE and NO_DATA names are requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import duckdb

from app.config.settings import Settings
from app.ingestion.price_backfill import BackfillResult, run_price_ingestion
from app.ingestion.price_freshness import (
    classify_price_freshness,
    daily_fetch_window,
    provider_end_exclusive,
)
from app.services.market_calendar import MarketCalendarService
from app.utils.logging import get_logger

logger = get_logger("price_sync")


@dataclass
class PriceSyncResult:
    backfill: BackfillResult
    tracked_targets: int = 0
    current: int = 0
    stale: int = 0
    no_data: int = 0
    fetch_targets: int = 0
    provider_fetch_skipped: bool = False
    requested_session_start: date | None = None
    requested_session_end: date | None = None
    expected_latest: date | None = None
    message: str = ""
    dry_run: bool = False
    failures: list = field(default_factory=list)

    @property
    def total_symbols(self) -> int:
        return self.backfill.total_symbols

    @property
    def successful(self) -> int:
        return self.backfill.successful

    @property
    def failed(self) -> int:
        return self.backfill.failed

    @property
    def rows_written(self) -> int:
        return self.backfill.rows_written

    @property
    def provider_fetch_count(self) -> int:
        return self.backfill.provider_fetch_count

    @property
    def symbols_requested(self) -> int:
        return self.backfill.symbols_requested

    @property
    def symbols_skipped_current(self) -> int:
        return self.current if self.provider_fetch_skipped or self.fetch_targets == 0 else self.current

    @property
    def batches_requested(self) -> int:
        return self.backfill.batches_requested

    @property
    def rows_received(self) -> int:
        return self.backfill.rows_received


def _empty_backfill(*, dry_run: bool, skipped_current: int = 0) -> BackfillResult:
    return BackfillResult(
        run_id=None,
        total_symbols=0,
        successful=0,
        failed=0,
        rows_written=0,
        dry_run=dry_run,
        symbols_requested=0,
        symbols_skipped_current=skipped_current,
        batches_requested=0,
        rows_received=0,
        provider_fetch_count=0,
    )


def _format_fast_path(
    *,
    tracked: int,
    current: int,
    stale: int,
    no_data: int,
    fetch_targets: int,
    network: str | int | None,
    start: date | None = None,
    end: date | None = None,
) -> str:
    net = "SKIPPED" if network is None else str(network)
    parts = [
        f"tracked={tracked}",
        f"current={current}",
        f"stale={stale}",
        f"no_data={no_data}",
        f"fetch_targets={fetch_targets}",
        f"network={net}",
    ]
    if start is not None and end is not None and fetch_targets:
        parts.append(f"sessions={start.isoformat()}..{end.isoformat()}")
    return " ".join(parts)


def format_price_fast_path(result: PriceSyncResult) -> str:
    network: str | int
    if result.provider_fetch_skipped or result.fetch_targets == 0:
        network = "SKIPPED"
    elif result.dry_run:
        network = "SKIPPED"
    else:
        network = result.provider_fetch_count
    return _format_fast_path(
        tracked=result.tracked_targets,
        current=result.current,
        stale=result.stale,
        no_data=result.no_data,
        fetch_targets=result.fetch_targets,
        network=network,
        start=result.requested_session_start,
        end=result.requested_session_end,
    )


def sync_recent_prices(
    settings: Settings,
    con: duckdb.DuckDBPyConnection,
    symbols: list[str] | None = None,
    lookback_days: int | None = None,  # noqa: ARG001  kept for CLI compatibility; unused
    dry_run: bool = False,
    now=None,  # noqa: ANN001
) -> PriceSyncResult:
    calendar = MarketCalendarService(settings.market_calendar, settings.market_data_grace_minutes)
    freshness = classify_price_freshness(settings, con, now=now, calendar=calendar)

    if symbols:
        # Explicit tickers: still fetch only those names, using the daily window
        # relative to expected latest (overlap from their stored last session).
        tickers = [s.strip().upper() for s in symbols if s.strip()]
        stale_or_missing = [r for r in (freshness.stale + freshness.no_data) if r.ticker in tickers]
        current_explicit = [r for r in freshness.current if r.ticker in tickers]
        unknown = [t for t in tickers if t not in {r.ticker for r in freshness.current + freshness.stale + freshness.no_data}]
        fetch_tickers = [r.ticker for r in stale_or_missing if r.ticker] + unknown
        if not fetch_tickers:
            msg = (
                f"SKIPPED - all {len(tickers)} requested securities already current "
                f"through {freshness.expected_latest.isoformat()}"
            )
            return PriceSyncResult(
                backfill=_empty_backfill(dry_run=dry_run, skipped_current=len(current_explicit)),
                tracked_targets=freshness.tracked_targets,
                current=len(current_explicit),
                stale=0,
                no_data=0,
                fetch_targets=0,
                provider_fetch_skipped=True,
                expected_latest=freshness.expected_latest,
                message=msg,
                dry_run=dry_run,
            )
        window = daily_fetch_window(freshness, calendar, settings.daily_price_lookback_sessions)
        start = window[0] if window else freshness.expected_latest
        end_inclusive = freshness.expected_latest
        ingest = run_price_ingestion(
            settings,
            con,
            symbols=fetch_tickers,
            start=start,
            end=provider_end_exclusive(end_inclusive),
            batch_size=settings.price_batch_size,
            resume=False,
            dry_run=dry_run,
            job_name="sync_prices",
            default_scope="tracked",
        )
        ingest.symbols_skipped_current = len(current_explicit)
        return PriceSyncResult(
            backfill=ingest,
            tracked_targets=freshness.tracked_targets,
            current=len(current_explicit),
            stale=sum(1 for r in stale_or_missing if r.status == "STALE"),
            no_data=sum(1 for r in stale_or_missing if r.status == "NO_DATA") + len(unknown),
            fetch_targets=len(fetch_tickers),
            provider_fetch_skipped=False,
            requested_session_start=start,
            requested_session_end=end_inclusive,
            expected_latest=freshness.expected_latest,
            message=_format_fast_path(
                tracked=freshness.tracked_targets,
                current=len(current_explicit),
                stale=sum(1 for r in stale_or_missing if r.status == "STALE"),
                no_data=sum(1 for r in stale_or_missing if r.status == "NO_DATA") + len(unknown),
                fetch_targets=len(fetch_tickers),
                network="SKIPPED" if dry_run else ingest.provider_fetch_count,
                start=start,
                end=end_inclusive,
            ),
            dry_run=dry_run,
            failures=ingest.failures,
        )

    if freshness.tracked_targets == 0:
        msg = "SKIPPED - tracked universe is empty (use 'stockdb universe add')"
        return PriceSyncResult(
            backfill=_empty_backfill(dry_run=dry_run),
            message=msg,
            expected_latest=freshness.expected_latest,
            dry_run=dry_run,
        )

    if not freshness.stale and not freshness.no_data:
        msg = (
            f"SKIPPED - all {freshness.tracked_targets} tracked securities already current "
            f"through {freshness.expected_latest.isoformat()}"
        )
        logger.info(msg)
        return PriceSyncResult(
            backfill=_empty_backfill(dry_run=dry_run, skipped_current=freshness.current_count),
            tracked_targets=freshness.tracked_targets,
            current=freshness.current_count,
            stale=0,
            no_data=0,
            fetch_targets=0,
            provider_fetch_skipped=True,
            expected_latest=freshness.expected_latest,
            requested_session_end=freshness.expected_latest,
            message=msg,
            dry_run=dry_run,
        )

    window = daily_fetch_window(freshness, calendar, settings.daily_price_lookback_sessions)
    assert window is not None
    start, end_inclusive = window
    fetch_tickers = freshness.fetch_tickers
    if not fetch_tickers:
        msg = (
            f"SKIPPED - {freshness.stale_count} stale / {freshness.no_data_count} no_data "
            "tracked securities have no ticker; provider not called"
        )
        return PriceSyncResult(
            backfill=_empty_backfill(dry_run=dry_run, skipped_current=freshness.current_count),
            tracked_targets=freshness.tracked_targets,
            current=freshness.current_count,
            stale=freshness.stale_count,
            no_data=freshness.no_data_count,
            fetch_targets=0,
            provider_fetch_skipped=True,
            expected_latest=freshness.expected_latest,
            message=msg,
            dry_run=dry_run,
        )
    ingest = run_price_ingestion(
        settings,
        con,
        symbols=fetch_tickers,
        start=start,
        end=provider_end_exclusive(end_inclusive),
        batch_size=settings.price_batch_size,
        resume=False,
        dry_run=dry_run,
        job_name="sync_prices",
        default_scope="tracked",
    )
    ingest.symbols_skipped_current = freshness.current_count
    msg = _format_fast_path(
        tracked=freshness.tracked_targets,
        current=freshness.current_count,
        stale=freshness.stale_count,
        no_data=freshness.no_data_count,
        fetch_targets=len(fetch_tickers),
        network="SKIPPED" if dry_run else ingest.provider_fetch_count,
        start=start,
        end=end_inclusive,
    )
    logger.info("Price fast path: %s", msg)
    return PriceSyncResult(
        backfill=ingest,
        tracked_targets=freshness.tracked_targets,
        current=freshness.current_count,
        stale=freshness.stale_count,
        no_data=freshness.no_data_count,
        fetch_targets=len(fetch_tickers),
        provider_fetch_skipped=False,
        requested_session_start=start,
        requested_session_end=end_inclusive,
        expected_latest=freshness.expected_latest,
        message=msg,
        dry_run=dry_run,
        failures=ingest.failures,
    )
