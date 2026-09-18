"""SEC filing metadata sync (``stockdb sync-sec-filings``).

Only pulls filing metadata (form type, dates, accession number, primary
document URL) for securities that have a known CIK. Document content is
never fetched or parsed in this phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb

from app.config.settings import Settings
from app.normalization.filings import normalize_filing_rows
from app.providers.filings.sec_filings_provider import SecFilingsProvider
from app.services.manifest_service import finish_run, record_source_file, start_run
from app.utils.logging import get_logger
from app.utils.parquet_io import LakeDataset

logger = get_logger("filings_sync")


@dataclass
class FilingsSyncResult:
    run_id: str
    successful: int
    failed: int
    rows_written: int
    failures: list[str] = field(default_factory=list)


def _resolve_targets(con: duckdb.DuckDBPyConnection, ciks: list[str] | None) -> dict[str, list[str]]:
    """Return {cik: [security_id, ...]}.

    Multiple securities can share one CIK (e.g. multi-class shares like
    GOOGL/GOOG both filed under the same company). We fetch SEC filing
    metadata once per *distinct* CIK and attach the result to every
    security_id sharing it, instead of making redundant SEC API calls.
    """
    if ciks:
        placeholders = ", ".join("?" for _ in ciks)
        rows = con.execute(
            f"SELECT security_id, cik FROM securities WHERE cik IN ({placeholders})", ciks
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT security_id, cik FROM securities WHERE cik IS NOT NULL AND is_active = TRUE ORDER BY security_id"
        ).fetchall()

    grouped: dict[str, list[str]] = {}
    for security_id, cik in rows:
        grouped.setdefault(cik, []).append(security_id)
    return grouped


def sync_filings(
    settings: Settings,
    con: duckdb.DuckDBPyConnection,
    ciks: list[str] | None = None,
    dry_run: bool = False,
) -> FilingsSyncResult:
    targets = _resolve_targets(con, ciks)
    if not targets:
        raise ValueError("No securities with a CIK found. Run 'stockdb sync-universe' first.")

    provider = SecFilingsProvider(settings)
    lake = LakeDataset(settings.filings_dir, "filing_date", ["accession_number"], ["security_id", "filing_date"])
    run_id = start_run(con, provider.capabilities.provider_name, "filings", {"cik_count": len(targets)})

    successful = 0
    failed = 0
    rows_written = 0
    failures: list[str] = []

    try:
        for cik, security_ids in targets.items():
            if dry_run:
                logger.info("[dry-run] would fetch SEC filings for CIK %s (%d securities)", cik, len(security_ids))
                successful += 1
                continue
            try:
                fetch = provider.fetch_filings(cik)
                # A filing is metadata about the *filer* (CIK); attach it to every
                # security_id that currently maps to this CIK (multi-class shares).
                for security_id in security_ids:
                    df = normalize_filing_rows(fetch, security_id)
                    if df.height > 0:
                        write_result = lake.write_increment(df, run_id)
                        rows_written += write_result.rows_written
                        for path in write_result.files_written:
                            record_source_file(
                                con,
                                run_id,
                                provider.capabilities.provider_name,
                                "filings",
                                path,
                                fetch.retrieved_at,
                                row_count=df.height,
                                min_date=write_result.min_date,
                                max_date=write_result.max_date,
                            )
                successful += 1
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to sync filings for CIK %s: %s", cik, exc)
                failed += 1
                failures.append(f"{cik}: {exc}")

        status = "success" if failed == 0 else ("partial" if successful > 0 else "failed")
        finish_run(
            con, run_id, status, requested_items=len(targets), successful_items=successful,
            failed_items=failed, rows_written=rows_written,
        )
        return FilingsSyncResult(run_id, successful, failed, rows_written, failures)
    except Exception as exc:  # noqa: BLE001
        finish_run(con, run_id, "failed", error_message=str(exc))
        raise
