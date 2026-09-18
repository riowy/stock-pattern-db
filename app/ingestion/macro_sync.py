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

FRED_KEY_MISSING_REASON = "FRED_API_KEY not configured"


@dataclass
class MacroSyncResult:
    run_id: str | None
    series_synced: int
    series_failed: int
    rows_written: int
    failures: list[str] = field(default_factory=list)
    status: str = "success"  # success | partial | failed | skipped | planned
    skip_reason: str | None = None
    dry_run: bool = False


def sync_macro(
    settings: Settings,
    con: duckdb.DuckDBPyConnection,
    series_ids: list[str] | None = None,
    start: date | None = None,
    dry_run: bool = False,
) -> MacroSyncResult:
    targets = series_ids or list(FRED_SEED_SERIES.keys())

    # A missing API key is a configuration choice, not a system failure --
    # report it as SKIPPED, never as FAILED, and never raise (the daily
    # pipeline must keep going).
    if not settings.fred_api_key.strip():
        logger.info("SKIPPED macro sync: %s", FRED_KEY_MISSING_REASON)
        return MacroSyncResult(
            run_id=None, series_synced=0, series_failed=0, rows_written=0,
            status="skipped", skip_reason=FRED_KEY_MISSING_REASON,
        )

    if dry_run:
        # True dry-run: no provider instantiation, no HTTP call, no DB write.
        logger.info("[dry-run] would fetch %d FRED series (no request made): %s", len(targets), targets)
        return MacroSyncResult(
            run_id=None, series_synced=len(targets), series_failed=0, rows_written=0,
            status="planned", dry_run=True,
        )

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
        return MacroSyncResult(run_id, successful, failed, rows_written, failures, status=status)
    except Exception as exc:  # noqa: BLE001
        finish_run(con, run_id, "failed", error_message=str(exc))
        raise
