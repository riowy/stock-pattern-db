"""Aggregate a point-in-time status report (``stockdb status``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import duckdb

from app.config.settings import Settings
from app.db.schema import create_lake_views
from app.ingestion.checkpoint import CheckpointStore
from app.utils.parquet_io import LakeDataset


@dataclass
class UniverseStatus:
    total_securities: int = 0
    active_securities: int = 0


@dataclass
class PricesStatus:
    total_rows_estimate: int = 0
    earliest_date: str | None = None
    latest_date: str | None = None
    securities_with_recent_data: int = 0
    securities_missing_recent_data: int = 0


@dataclass
class MacroStatus:
    series_count: int = 0
    last_update: str | None = None


@dataclass
class VixStatus:
    rows: int = 0
    latest_date: str | None = None


@dataclass
class SecStatus:
    last_filing_retrieved_at: str | None = None
    filings_tracked: int = 0


@dataclass
class DiskStatus:
    raw_bytes: int = 0
    lake_bytes: int = 0
    state_bytes: int = 0


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
class StatusReport:
    universe: UniverseStatus
    prices: PricesStatus
    macro: MacroStatus
    vix: VixStatus
    sec: SecStatus
    disk: DiskStatus
    jobs: JobsStatus
    data_quality: DataQualityStatus


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

    total_sec, active_sec = con.execute(
        "SELECT count(*), sum(CASE WHEN is_active THEN 1 ELSE 0 END) FROM securities"
    ).fetchone()
    universe = UniverseStatus(total_securities=total_sec or 0, active_securities=active_sec or 0)

    prices_ds = LakeDataset(settings.prices_daily_dir, "date", ["security_id", "date"], ["security_id", "date"])
    if prices_ds.has_any_files():
        row_count, min_d, max_d = con.execute(
            "SELECT count(*), min(date), max(date) FROM prices_daily"
        ).fetchone()
        recent_cutoff = date.today() - timedelta(days=5)
        recent_count = con.execute(
            "SELECT count(DISTINCT security_id) FROM prices_daily WHERE date >= ?", [recent_cutoff]
        ).fetchone()[0]
        missing = max((active_sec or 0) - recent_count, 0)
        prices = PricesStatus(
            total_rows_estimate=row_count or 0,
            earliest_date=str(min_d) if min_d else None,
            latest_date=str(max_d) if max_d else None,
            securities_with_recent_data=recent_count or 0,
            securities_missing_recent_data=missing,
        )
    else:
        prices = PricesStatus()

    macro_ds = LakeDataset(settings.macro_dir, "date", ["series_id", "date"], ["series_id", "date"])
    if macro_ds.has_any_files():
        series_count, last_update = con.execute(
            "SELECT count(DISTINCT series_id), max(retrieved_at) FROM macro"
        ).fetchone()
        macro = MacroStatus(series_count=series_count or 0, last_update=str(last_update) if last_update else None)
    else:
        macro = MacroStatus()

    vix_ds = LakeDataset(settings.volatility_dir, "date", ["date"], ["date"])
    if vix_ds.has_any_files():
        rows, latest = con.execute("SELECT count(*), max(date) FROM volatility").fetchone()
        vix = VixStatus(rows=rows or 0, latest_date=str(latest) if latest else None)
    else:
        vix = VixStatus()

    filings_ds = LakeDataset(
        settings.filings_dir, "filing_date", ["accession_number"], ["security_id", "filing_date"]
    )
    if filings_ds.has_any_files():
        count, last_retrieved = con.execute("SELECT count(*), max(retrieved_at) FROM filings").fetchone()
        sec = SecStatus(
            last_filing_retrieved_at=str(last_retrieved) if last_retrieved else None, filings_tracked=count or 0
        )
    else:
        sec = SecStatus()

    disk = DiskStatus(
        raw_bytes=_dir_size(settings.raw_dir),
        lake_bytes=_dir_size(settings.lake_dir),
        state_bytes=_dir_size(settings.state_dir),
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

    return StatusReport(universe, prices, macro, vix, sec, disk, jobs, dq)
