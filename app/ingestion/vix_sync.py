"""Cboe VIX sync (``stockdb sync-vix``)."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from app.config.settings import Settings
from app.normalization.vix import normalize_vix_rows
from app.providers.volatility.cboe_vix_provider import CboeVixProvider
from app.services.manifest_service import finish_run, record_source_file, start_run
from app.utils.logging import get_logger
from app.utils.parquet_io import LakeDataset

logger = get_logger("vix_sync")


@dataclass
class VixSyncResult:
    run_id: str | None
    rows_written: int
    status: str
    dry_run: bool = False


def sync_vix(settings: Settings, con: duckdb.DuckDBPyConnection, dry_run: bool = False) -> VixSyncResult:
    if dry_run:
        # True dry-run: no provider instantiation, no HTTP call, no
        # ingest_runs row, no Parquet write.
        logger.info("[dry-run] would fetch the latest Cboe VIX history (no request made)")
        return VixSyncResult(run_id=None, rows_written=0, status="planned", dry_run=True)

    provider = CboeVixProvider(settings)
    lake = LakeDataset(settings.volatility_dir, "date", ["date"], ["date"])
    run_id = start_run(con, provider.capabilities.provider_name, "volatility", {})

    try:
        fetch = provider.fetch_history()
        df = normalize_vix_rows(fetch)
        rows_written = 0
        if df.height > 0:
            write_result = lake.write_increment(df, run_id)
            rows_written = write_result.rows_written
            for path in write_result.files_written:
                record_source_file(
                    con,
                    run_id,
                    provider.capabilities.provider_name,
                    "volatility",
                    path,
                    fetch.retrieved_at,
                    row_count=df.height,
                    min_date=write_result.min_date,
                    max_date=write_result.max_date,
                )

        finish_run(con, run_id, "success", requested_items=1, successful_items=1, failed_items=0, rows_written=rows_written)
        return VixSyncResult(run_id, rows_written, "success")
    except Exception as exc:  # noqa: BLE001
        finish_run(con, run_id, "failed", error_message=str(exc))
        raise
