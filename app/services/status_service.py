"""Aggregate a point-in-time status report (``stockdb status``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from app.config.settings import Settings
from app.db.schema import create_lake_views
from app.ingestion.checkpoint import CheckpointStore
from app.services.dataset_metadata_service import PRICES_DAILY_DATASET, get_metadata
from app.services.market_calendar import MarketCalendarService
from app.services.storage_health_service import compaction_candidates
from app.services.tracked_universe_service import get_tracked_price_security_ids
from app.utils.parquet_io import LakeDataset


@dataclass
class UniverseStatus:
    total_known_securities: int = 0
    active_securities: int = 0
    tracked_securities: int = 0
    tracked_for_prices: int = 0


@dataclass
class PricesStatus:
    total_rows_estimate: int = 0
    earliest_date: str | None = None
    latest_date: str | None = None
    tracked_current: int = 0
    tracked_stale: int = 0
    tracked_no_data: int = 0


@dataclass
class MacroStatus:
    series_count: int = 0
    last_update: str | None = None
    fred_configured: bool = False


@dataclass
class VixStatus:
    rows: int = 0
    latest_date: str | None = None


@dataclass
class SecStatus:
    last_filing_retrieved_at: str | None = None
    filings_tracked: int = 0
    tracked_ciks: int = 0


@dataclass
class StorageStatus:
    raw_bytes: int = 0
    lake_bytes: int = 0
    state_bytes: int = 0
    compaction_candidate_count: int = 0
    compaction_candidate_labels: list[str] = field(default_factory=list)


@dataclass
class JobsStatus:
    last_success: dict | None = None
    last_failure: dict | None = None
    active_checkpoints: list[dict] = field(default_factory=list)


@dataclass
class DataQualityStatus:
    critical_open: int = 0
    warning_open: int = 0


@dataclass
class ResearchIntegrityStatus:
    historical_universe_complete: bool = False
    survivorship_safe: bool = False
    point_in_time_security_master: bool = False
    price_provider: str = ""
    commercial_use_safe: bool = False


@dataclass
class StatusReport:
    universe: UniverseStatus
    prices: PricesStatus
    macro: MacroStatus
    vix: VixStatus
    sec: SecStatus
    storage: StorageStatus
    jobs: JobsStatus
    data_quality: DataQualityStatus
    research_integrity: ResearchIntegrityStatus


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            fp = Path(dirpath) / name
            try:
                total += fp.stat().st_size
            except OSError:
                continue
    return total


def gather_status(settings: Settings, con: duckdb.DuckDBPyConnection) -> StatusReport:
    create_lake_views(con, settings)

    total_known, active_sec = con.execute(
        "SELECT count(*), sum(CASE WHEN is_active THEN 1 ELSE 0 END) FROM securities"
    ).fetchone()
    tracked_total, tracked_price_count = con.execute(
        "SELECT count(*), sum(CASE WHEN price_tracking THEN 1 ELSE 0 END) "
        "FROM tracked_securities WHERE enabled = TRUE"
    ).fetchone()
    universe = UniverseStatus(
        total_known_securities=total_known or 0,
        active_securities=active_sec or 0,
        tracked_securities=tracked_total or 0,
        tracked_for_prices=tracked_price_count or 0,
    )

    tracked_price_ids = get_tracked_price_security_ids(con)
    prices_ds = LakeDataset(settings.prices_daily_dir, "date", ["security_id", "date"], ["security_id", "date"])
    calendar = MarketCalendarService(settings.market_calendar, settings.market_data_grace_minutes)

    if prices_ds.has_any_files():
        row_count, min_d, max_d = con.execute("SELECT count(*), min(date), max(date) FROM prices_daily").fetchone()

        if tracked_price_ids:
            latest_expected = calendar.latest_expected_session()
            cutoff = calendar.sessions_ago(latest_expected, 5)
            placeholders = ", ".join("?" for _ in tracked_price_ids)
            last_date_rows = con.execute(
                f"SELECT security_id, max(date) FROM prices_daily WHERE security_id IN ({placeholders}) GROUP BY security_id",
                tracked_price_ids,
            ).fetchall()
            last_date_map = {r[0]: r[1] for r in last_date_rows}
            current = sum(1 for sid in tracked_price_ids if last_date_map.get(sid) and last_date_map[sid] >= cutoff)
            no_data = sum(1 for sid in tracked_price_ids if sid not in last_date_map)
            stale = len(tracked_price_ids) - current - no_data
        else:
            current = stale = no_data = 0

        prices = PricesStatus(
            total_rows_estimate=row_count or 0,
            earliest_date=str(min_d) if min_d else None,
            latest_date=str(max_d) if max_d else None,
            tracked_current=current,
            tracked_stale=stale,
            tracked_no_data=no_data,
        )
    else:
        prices = PricesStatus(tracked_no_data=len(tracked_price_ids))

    macro_ds = LakeDataset(settings.macro_dir, "date", ["series_id", "date"], ["series_id", "date"])
    if macro_ds.has_any_files():
        series_count, last_update = con.execute(
            "SELECT count(DISTINCT series_id), max(retrieved_at) FROM macro"
        ).fetchone()
        macro = MacroStatus(
            series_count=series_count or 0,
            last_update=str(last_update) if last_update else None,
            fred_configured=bool(settings.fred_api_key.strip()),
        )
    else:
        macro = MacroStatus(fred_configured=bool(settings.fred_api_key.strip()))

    vix_ds = LakeDataset(settings.volatility_dir, "date", ["date"], ["date"])
    if vix_ds.has_any_files():
        rows, latest = con.execute("SELECT count(*), max(date) FROM volatility").fetchone()
        vix = VixStatus(rows=rows or 0, latest_date=str(latest) if latest else None)
    else:
        vix = VixStatus()

    tracked_cik_count = con.execute(
        """
        SELECT count(DISTINCT s.cik) FROM tracked_securities t
        JOIN securities s ON s.security_id = t.security_id
        WHERE t.enabled = TRUE AND t.filings_tracking = TRUE AND s.cik IS NOT NULL
        """
    ).fetchone()[0]
    filings_ds = LakeDataset(
        settings.filings_dir, "filing_date", ["accession_number"], ["security_id", "filing_date"]
    )
    if filings_ds.has_any_files():
        count, last_retrieved = con.execute("SELECT count(*), max(retrieved_at) FROM filings").fetchone()
        sec = SecStatus(
            last_filing_retrieved_at=str(last_retrieved) if last_retrieved else None,
            filings_tracked=count or 0,
            tracked_ciks=tracked_cik_count or 0,
        )
    else:
        sec = SecStatus(tracked_ciks=tracked_cik_count or 0)

    candidates = compaction_candidates(settings)
    storage = StorageStatus(
        raw_bytes=_dir_size(settings.raw_dir),
        lake_bytes=_dir_size(settings.lake_dir),
        state_bytes=_dir_size(settings.state_dir),
        compaction_candidate_count=len(candidates),
        compaction_candidate_labels=[f"{c.dataset} {c.year:04d}-{c.month:02d} ({c.file_count} files)" for c in candidates[:5]],
    )

    last_success_row = con.execute(
        "SELECT provider, dataset, started_at, finished_at FROM ingest_runs "
        "WHERE status = 'success' ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    last_failure_row = con.execute(
        "SELECT provider, dataset, started_at, error_message FROM ingest_runs "
        "WHERE status = 'failed' ORDER BY started_at DESC LIMIT 1"
    ).fetchone()

    checkpoint_store = CheckpointStore(settings.checkpoints_dir)
    checkpoints = [
        {
            "job_key": key,
            "completed": len(state.get("completed", [])),
            "failed": len(state.get("failed", [])),
        }
        for key, state in checkpoint_store.list_all()
    ]

    jobs = JobsStatus(
        last_success=(
            {"provider": last_success_row[0], "dataset": last_success_row[1], "at": str(last_success_row[2])}
            if last_success_row
            else None
        ),
        last_failure=(
            {
                "provider": last_failure_row[0],
                "dataset": last_failure_row[1],
                "at": str(last_failure_row[2]),
                "error": last_failure_row[3],
            }
            if last_failure_row
            else None
        ),
        active_checkpoints=checkpoints,
    )

    critical_open, warning_open = con.execute(
        "SELECT "
        "  sum(CASE WHEN severity = 'critical' AND NOT resolved THEN 1 ELSE 0 END), "
        "  sum(CASE WHEN severity = 'warning' AND NOT resolved THEN 1 ELSE 0 END) "
        "FROM data_quality_issues"
    ).fetchone()
    dq = DataQualityStatus(critical_open=critical_open or 0, warning_open=warning_open or 0)

    meta = get_metadata(con, PRICES_DAILY_DATASET)
    research_integrity = ResearchIntegrityStatus(
        historical_universe_complete=meta.get("historical_universe_complete") == "true",
        survivorship_safe=meta.get("survivorship_safe") == "true",
        point_in_time_security_master=meta.get("point_in_time_security_master") == "true",
        price_provider=meta.get("price_provider", settings.price_provider),
        commercial_use_safe=meta.get("research_only_price_provider") == "false",
    )

    return StatusReport(universe, prices, macro, vix, sec, storage, jobs, dq, research_integrity)
