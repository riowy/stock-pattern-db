"""FRED macro series sync (``stockdb sync-macro``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import duckdb

from app.config.fred_series import FRED_SEED_SERIES
from app.config.settings import Settings
from app.normalization.macro import normalize_macro_rows
from app.providers.macro.fred_provider import FredMacroProvider
from app.services.manifest_service import finish_run, record_source_file, start_run
from app.utils.logging import get_logger
from app.utils.parquet_io import LakeDataset

logger = get_logger("macro_sync")


@dataclass
class MacroSyncResult:
    run_id: str
    series_synced: int
    series_failed: int
    rows_written: int
    failures: list[str] = field(default_factory=list)


def sync_macro(
    settings: Settings,
    con: duckdb.DuckDBPyConnection,
    series_ids: list[str] | None = None,
    start: date | None = None,
    dry_run: bool = False,
) -> MacroSyncResult:
    targets = series_ids or list(FRED_SEED_SERIES.keys())
    provider = FredMacroProvider(settings)
    lake = LakeDataset(settings.macro_dir, "date", ["series_id", "date"], ["series_id", "date"])

    run_id = start_run(
        con, provider.capabilities.provider_name, "macro", {"series_ids": targets, "start": str(start)}
    )

    rows_written = 0
    successful = 0
    failed = 0
    failures: list[str] = []

    try:
        for series_id in targets:
            if dry_run:
                logger.info("[dry-run] would fetch FRED series %s", series_id)
                successful += 1
                continue
            try:
                fetch = provider.fetch_series(series_id, start=start)
                df = normalize_macro_rows(fetch)
                if df.height > 0:
                    write_result = lake.write_increment(df, run_id)
                    rows_written += write_result.rows_written
                    for path in write_result.files_written:
                        record_source_file(
                            con,
                            run_id,
                            provider.capabilities.provider_name,
                            "macro",
                            path,
                            fetch.retrieved_at,
                            row_count=df.height,
                            min_date=write_result.min_date,
                            max_date=write_result.max_date,
                        )
                successful += 1
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to sync FRED series %s: %s", series_id, exc)
                failed += 1
                failures.append(f"{series_id}: {exc}")

        status = "success" if failed == 0 else ("partial" if successful > 0 else "failed")
        finish_run(
            con, run_id, status, requested_items=len(targets), successful_items=successful,
            failed_items=failed, rows_written=rows_written,
        )
        return MacroSyncResult(run_id, successful, failed, rows_written, failures)
    except Exception as exc:  # noqa: BLE001
        finish_run(con, run_id, "failed", error_message=str(exc))
        raise
