"""SEC filing metadata sync (``stockdb sync-sec-filings``).

Only pulls filing metadata (form type, dates, accession number, primary
document URL) for securities that have a known CIK. Document content is
never fetched or parsed in this phase.

Scope (see project task "8. run-daily 대상 수정"): the default target when
``ciks`` is omitted is the *tracked* universe's filing-tracking set, not
every security with a CIK in the security master (~8,000+ CIKs). Making
~8,000 individual ``data.sec.gov/submissions/CIK....json`` requests every
day would be both slow and needlessly hard on SEC's infrastructure for data
most of which nobody is actually watching.

TODO (future, not implemented here): once filing-metadata coverage needs to
grow beyond the tracked universe, prefer SEC's bulk submissions archive
(https://www.sec.gov/Archives/edgar/full-index/ or the bulk
`submissions.zip` dataset) over looping per-CIK API calls, and only widen
this function's default scope once that bulk path exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb

from app.config.settings import Settings
from app.normalization.filings import normalize_filing_rows
from app.providers.filings.sec_filings_provider import SecFilingsProvider
from app.services.manifest_service import finish_run, record_source_file, start_run
from app.services.tracked_universe_service import get_tracked_filing_ciks
from app.utils.logging import get_logger
from app.utils.parquet_io import LakeDataset

logger = get_logger("filings_sync")


@dataclass
class FilingsSyncResult:
    run_id: str | None
    successful: int
    failed: int
    rows_written: int
    failures: list[str] = field(default_factory=list)
    status: str = "success"  # success | partial | failed | skipped | planned
    skip_reason: str | None = None
    dry_run: bool = False


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
        grouped: dict[str, list[str]] = {}
        for security_id, cik in rows:
            grouped.setdefault(cik, []).append(security_id)
        return grouped

    return get_tracked_filing_ciks(con)


def sync_filings(
    settings: Settings,
    con: duckdb.DuckDBPyConnection,
    ciks: list[str] | None = None,
    dry_run: bool = False,
) -> FilingsSyncResult:
    targets = _resolve_targets(con, ciks)

    if not targets:
        reason = (
            "no explicit --ciks given and no securities have filings_tracking enabled "
            "(use 'stockdb universe add <TICKER>...' to enable it)"
        )
        logger.info("SKIPPED filings sync: %s", reason)
        return FilingsSyncResult(run_id=None, successful=0, failed=0, rows_written=0, status="skipped", skip_reason=reason)

    if dry_run:
        # True dry-run: no provider instantiation, no HTTP call, no DB write,
        # no per-CIK loop.
        logger.info(
            "[dry-run] would check SEC filing metadata for %d tracked CIK(s) (no request made)", len(targets)
        )
        return FilingsSyncResult(
            run_id=None, successful=len(targets), failed=0, rows_written=0, status="planned", dry_run=True
        )

    provider = SecFilingsProvider(settings)
    lake = LakeDataset(settings.filings_dir, "filing_date", ["accession_number"], ["security_id", "filing_date"])
    run_id = start_run(con, provider.capabilities.provider_name, "filings", {"cik_count": len(targets)})

    successful = 0
    failed = 0
    rows_written = 0
    failures: list[str] = []

    try:
        for cik, security_ids in targets.items():
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
        return FilingsSyncResult(run_id, successful, failed, rows_written, failures, status=status)
    except Exception as exc:  # noqa: BLE001
        finish_run(con, run_id, "failed", error_message=str(exc))
        raise
