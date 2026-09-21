"""Weekly (on-demand) price repair: re-fetch recent XNYS sessions for tracked names."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import duckdb

from app.config.settings import Settings
from app.ingestion.price_backfill import BackfillResult, run_price_ingestion
from app.ingestion.price_freshness import provider_end_exclusive
from app.services.market_calendar import MarketCalendarService
from app.services.tracked_universe_service import get_tracked_price_security_ids
from app.utils.logging import get_logger

logger = get_logger("price_repair")


@dataclass
class PriceRepairResult:
    tracked_targets: int
    lookback_sessions: int
    start_session: date
    end_session: date
    batch_size: int
    workers: int
    backfill: BackfillResult
    dry_run: bool = False

    @property
    def network_requests_planned(self) -> int:
        return self.tracked_targets

    @property
    def provider_fetch_count(self) -> int:
        return self.backfill.provider_fetch_count

    @property
    def rows_written(self) -> int:
        return self.backfill.rows_written


def repair_window(
    calendar: MarketCalendarService,
    lookback_sessions: int,
    now=None,  # noqa: ANN001
) -> tuple[date, date]:
    end = calendar.expected_latest_completed_session(now)
    start = calendar.sessions_ago(end, lookback_sessions)
    return start, end


def repair_prices(
    settings: Settings,
    con: duckdb.DuckDBPyConnection,
    *,
    lookback_sessions: int | None = None,
    dry_run: bool = False,
    now=None,  # noqa: ANN001
) -> PriceRepairResult:
    calendar = MarketCalendarService(settings.market_calendar, settings.market_data_grace_minutes)
    sessions = lookback_sessions if lookback_sessions is not None else settings.price_repair_lookback_sessions
    start, end_inclusive = repair_window(calendar, sessions, now=now)
    tracked = get_tracked_price_security_ids(con)
    if not tracked:
        ingest = BackfillResult(
            run_id=None,
            total_symbols=0,
            successful=0,
            failed=0,
            rows_written=0,
            dry_run=dry_run,
            provider_fetch_count=0,
        )
    else:
        ingest = run_price_ingestion(
            settings,
            con,
            symbols=None,
            start=start,
            end=provider_end_exclusive(end_inclusive),
            batch_size=settings.price_batch_size,
            resume=False,
            dry_run=dry_run,
            job_name="repair_prices",
            default_scope="tracked",
            workers=settings.max_workers,
        )
    return PriceRepairResult(
        tracked_targets=len(tracked),
        lookback_sessions=sessions,
        start_session=start,
        end_session=end_inclusive,
        batch_size=settings.price_batch_size,
        workers=settings.max_workers,
        backfill=ingest,
        dry_run=dry_run,
    )
