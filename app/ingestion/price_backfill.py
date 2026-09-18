"""Batch daily-price backfill/sync with checkpoint/resume.

Core loop (see README "10. 서버 자원 사용 제한"):

    50개 종목 다운로드 -> normalization -> validation -> Parquet 저장
    -> checkpoint -> 다음 50개

If the process dies mid-run, re-invoking with ``--resume`` continues from
the first *not-yet-completed* symbol instead of starting over, and no
partition file is ever left half-written (see ``LakeDataset.write_increment``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import duckdb

from app.config.settings import Settings
from app.ingestion.checkpoint import CheckpointStore, make_job_key
from app.normalization.corporate_actions import derive_corporate_actions
from app.normalization.prices import normalize_price_rows
from app.normalization.symbols import resolve_provider_symbol
from app.providers.registry import get_price_provider
from app.services.manifest_service import finish_run, record_source_file, start_run
from app.utils.logging import get_logger
from app.utils.parquet_io import LakeDataset

logger = get_logger("price_backfill")


@dataclass
class SymbolFailure:
    symbol: str
    error: str


@dataclass
class BackfillResult:
    run_id: str
    total_symbols: int
    successful: int
    failed: int
    rows_written: int
    failures: list[SymbolFailure] = field(default_factory=list)
    resumed: bool = False
    checkpoint_job_key: str | None = None


def resolve_symbols(con: duckdb.DuckDBPyConnection, symbols: list[str] | None) -> list[tuple[str, str]]:
    """Return [(canonical_ticker, security_id), ...].

    Looks up ``securities.primary_ticker`` first, then falls back to the
    ``security_identifiers`` TICKER history (currently-valid rows only).
    """
    if symbols:
        resolved: list[tuple[str, str]] = []
        for raw in symbols:
            ticker = raw.strip().upper()
            if not ticker:
                continue
            row = con.execute(
                "SELECT security_id FROM securities WHERE upper(primary_ticker) = ?", [ticker]
            ).fetchone()
            if row is None:
                row = con.execute(
                    """
                    SELECT security_id FROM security_identifiers
                    WHERE identifier_type = 'TICKER' AND upper(identifier_value) = ? AND valid_to IS NULL
                    """,
                    [ticker],
                ).fetchone()
            if row is None:
                logger.warning(
                    "Ticker '%s' not found in security master. Run 'stockdb sync-universe' first, "
                    "or check the spelling. Skipping.",
                    ticker,
                )
                continue
            resolved.append((ticker, row[0]))
        return resolved

    rows = con.execute(
        "SELECT primary_ticker, security_id FROM securities WHERE is_active = TRUE AND primary_ticker IS NOT NULL "
        "ORDER BY primary_ticker"
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def run_price_ingestion(
    settings: Settings,
    con: duckdb.DuckDBPyConnection,
    symbols: list[str] | None,
    start: date,
    end: date | None,
    batch_size: int,
    resume: bool,
    dry_run: bool = False,
    job_name: str = "backfill_prices",
) -> BackfillResult:
    targets = resolve_symbols(con, symbols)
    if not targets:
        raise ValueError("No symbols resolved. Run 'stockdb sync-universe' first or check --symbols.")

    provider = get_price_provider(settings.price_provider, settings)

    params = {
        "symbols": sorted(t for t, _ in targets) if symbols else "ALL_ACTIVE",
        "start": start.isoformat(),
        "end": end.isoformat() if end else None,
        "batch_size": batch_size,
        "provider": settings.price_provider,
    }
    job_key = make_job_key(job_name, params)
    store = CheckpointStore(settings.checkpoints_dir)

    state = store.load(job_key) if resume else None
    resumed = state is not None
    if state is None:
        state = {"completed": [], "failed": [], "status": "in_progress"}

    completed_set = set(state.get("completed", []))
    pending = [t for t, _ in targets if t not in completed_set]
    symbol_to_security = dict(targets)

    if resumed:
        logger.info(
            "Resuming job %s: %d already completed, %d remaining",
            job_key,
            len(completed_set),
            len(pending),
        )

    lake = LakeDataset(settings.prices_daily_dir, "date", ["security_id", "date"], ["security_id", "date"])
    corp_actions_lake = LakeDataset(
        settings.corporate_actions_dir,
        "effective_date",
        ["security_id", "effective_date", "action_type"],
        ["security_id", "effective_date"],
    )
    run_id = start_run(con, settings.price_provider, "prices_daily", params)

    rows_written = 0
    successful = 0
    failed = 0
    failures: list[SymbolFailure] = []

    try:
        while pending:
            batch = pending[:batch_size]
            logger.info("Processing batch of %d symbols: %s", len(batch), batch)

            for canonical in batch:
                if dry_run:
                    logger.info("[dry-run] would fetch daily bars for %s from %s to %s", canonical, start, end)
                    successful += 1
                    state["completed"].append(canonical)
                    continue

                security_id = symbol_to_security[canonical]
                default_provider_symbol = provider.to_provider_symbol(canonical)
                provider_symbol = resolve_provider_symbol(
                    con, canonical, settings.price_provider, default_provider_symbol
                )
                try:
                    fetch = provider.fetch_daily_bars(provider_symbol, start, end)
                    df = normalize_price_rows(fetch, security_id, canonical)
                    if df.height > 0:
                        write_result = lake.write_increment(df, run_id)
                        rows_written += write_result.rows_written
                        for path in write_result.files_written:
                            record_source_file(
                                con,
                                run_id,
                                settings.price_provider,
                                "prices_daily",
                                path,
                                fetch.retrieved_at,
                                row_count=df.height,
                                min_date=write_result.min_date,
                                max_date=write_result.max_date,
                            )
                    else:
                        logger.warning("No rows returned for %s (%s)", canonical, provider_symbol)

                    # Dividends/splits reported alongside the same price fetch are
                    # written to their own dataset -- never merged into OHLC rows.
                    corp_df = derive_corporate_actions(fetch, security_id)
                    if corp_df.height > 0:
                        corp_write = corp_actions_lake.write_increment(corp_df, run_id)
                        for path in corp_write.files_written:
                            record_source_file(
                                con,
                                run_id,
                                settings.price_provider,
                                "corporate_actions",
                                path,
                                fetch.retrieved_at,
                                row_count=corp_df.height,
                                min_date=corp_write.min_date,
                                max_date=corp_write.max_date,
                            )
                    successful += 1
                    state["completed"].append(canonical)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to fetch %s (%s): %s", canonical, provider_symbol, exc)
                    failed += 1
                    failures.append(SymbolFailure(canonical, str(exc)))
                    state.setdefault("failed", []).append({"symbol": canonical, "error": str(exc)})

            pending = pending[len(batch):]
            state["completed"] = state.get("completed", [])
            store.save(job_key, state)

        status = "success" if failed == 0 else ("partial" if successful > 0 else "failed")
        finish_run(
            con,
            run_id,
            status,
            requested_items=len(targets),
            successful_items=successful,
            failed_items=failed,
            rows_written=rows_written,
        )

        if not pending and failed == 0:
            store.clear(job_key)

        return BackfillResult(run_id, len(targets), successful, failed, rows_written, failures, resumed, job_key)

    except Exception as exc:  # noqa: BLE001
        finish_run(con, run_id, "failed", error_message=str(exc))
        raise
