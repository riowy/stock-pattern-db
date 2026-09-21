"""Batch daily-price backfill/sync with checkpoint/resume.

Core loop (see README "10. 서버 자원 사용 제한"):

    25-symbol batch fetch -> concat -> one write_increment per month partition
    -> checkpoint -> next batch

If the process dies mid-run, re-invoking with ``--resume`` continues from
the first *not-yet-completed* symbol instead of starting over, and no
partition file is ever left half-written (see ``LakeDataset.write_increment``).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import duckdb
import polars as pl

from app.config.lake_datasets import get_lake_dataset
from app.config.settings import Settings
from app.ingestion.checkpoint import CheckpointStore, make_job_key
from app.normalization.corporate_actions import derive_corporate_actions
from app.normalization.prices import normalize_price_rows
from app.normalization.symbols import resolve_provider_symbol
from app.providers.registry import get_price_provider
from app.services.manifest_service import finish_run, record_source_file, start_run
from app.services.tracked_universe_service import get_tracked_price_security_ids
from app.utils.logging import get_logger

logger = get_logger("price_backfill")


@dataclass
class SymbolFailure:
    symbol: str
    error: str


@dataclass
class BackfillResult:
    run_id: str | None
    total_symbols: int
    successful: int
    failed: int
    rows_written: int
    failures: list[SymbolFailure] = field(default_factory=list)
    resumed: bool = False
    checkpoint_job_key: str | None = None
    dry_run: bool = False
    symbols_requested: int = 0
    symbols_skipped_current: int = 0
    batches_requested: int = 0
    rows_received: int = 0
    provider_fetch_count: int = 0


def resolve_symbols(
    con: duckdb.DuckDBPyConnection, symbols: list[str] | None, default_scope: str = "all_active"
) -> list[tuple[str, str]]:
    """Return [(canonical_ticker, security_id), ...].

    Looks up ``securities.primary_ticker`` first, then falls back to the
    ``security_identifiers`` TICKER history (currently-valid rows only).

    When ``symbols`` is omitted, ``default_scope`` decides the fallback:
      * "all_active" -- every active security in the (large) security
        master. This is the deliberate, explicit-full-backfill scope used
        by ``backfill-prices`` when expanding coverage (10 -> 100 -> 500 ->
        full universe), matching the README's expansion workflow.
      * "tracked" -- only the small operational ``tracked_securities`` set.
        Used by day-to-day incremental sync (``sync-prices``, ``run-daily``)
        so routine operations never implicitly touch the full ~10k-security
        universe.
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

    if default_scope == "tracked":
        tracked_ids = set(get_tracked_price_security_ids(con))
        if not tracked_ids:
            return []
        placeholders = ", ".join("?" for _ in tracked_ids)
        rows = con.execute(
            f"SELECT primary_ticker, security_id FROM securities "
            f"WHERE security_id IN ({placeholders}) AND primary_ticker IS NOT NULL ORDER BY primary_ticker",
            list(tracked_ids),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

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
    default_scope: str = "all_active",
    workers: int | None = None,
) -> BackfillResult:
    targets = resolve_symbols(con, symbols, default_scope=default_scope)
    if not targets:
        if dry_run:
            return BackfillResult(run_id=None, total_symbols=0, successful=0, failed=0, rows_written=0, dry_run=True)
        raise ValueError(
            "No symbols resolved. Run 'stockdb sync-universe' first, pass --symbols, or "
            "'stockdb universe add <TICKER>...' to populate the tracked universe."
        )

    if dry_run:
        # True dry-run: no provider instantiation, no network call, no DuckDB
        # write, no checkpoint touch, no per-symbol loop -- just report what
        # *would* happen based on locally-known metadata.
        logger.info(
            "[dry-run] would fetch daily bars for %d symbol(s) from %s to %s (provider=%s): %s",
            len(targets),
            start,
            end or "latest",
            settings.price_provider,
            [t for t, _ in targets][:10] if len(targets) <= 10 else f"{len(targets)} symbols",
        )
        n_batches = (len(targets) + batch_size - 1) // batch_size if targets else 0
        return BackfillResult(
            run_id=None,
            total_symbols=len(targets),
            successful=len(targets),
            failed=0,
            rows_written=0,
            dry_run=True,
            symbols_requested=len(targets),
            batches_requested=n_batches,
            provider_fetch_count=0,
        )

    provider = get_price_provider(settings.price_provider, settings)

    params = {
        "symbols": sorted(t for t, _ in targets) if symbols else f"DEFAULT_SCOPE:{default_scope}",
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

    workers_n = max(1, int(workers if workers is not None else settings.max_workers))
    logger.info("Price ingestion workers=%d batch_size=%d", workers_n, batch_size)

    lake = get_lake_dataset(settings.lake_dir, "prices_daily")
    corp_actions_lake = get_lake_dataset(settings.lake_dir, "corporate_actions")
    run_id = start_run(con, settings.price_provider, "prices_daily", params)

    rows_written = 0
    successful = 0
    failed = 0
    failures: list[SymbolFailure] = []
    provider_fetch_count = 0
    rows_received = 0
    batches_requested = 0

    try:
        while pending:
            batch = pending[:batch_size]
            batches_requested += 1
            logger.info("Processing batch of %d symbols: %s", len(batch), batch)
            batch_price_frames: list[pl.DataFrame] = []
            batch_corp_frames: list[pl.DataFrame] = []
            batch_retrieved_at = None
            batch_ok: list[str] = []

            resolved_batch: list[tuple[str, str, str]] = []
            for canonical in batch:
                security_id = symbol_to_security[canonical]
                default_provider_symbol = provider.to_provider_symbol(canonical)
                provider_symbol = resolve_provider_symbol(
                    con, canonical, settings.price_provider, default_provider_symbol
                )
                resolved_batch.append((canonical, security_id, provider_symbol))
            provider_fetch_count += len(resolved_batch)

            def _fetch_one(item: tuple[str, str, str]):  # noqa: ANN202
                canonical, security_id, provider_symbol = item
                fetch = provider.fetch_daily_bars(provider_symbol, start, end)
                if settings.price_request_pause_seconds:
                    time.sleep(settings.price_request_pause_seconds)
                return canonical, security_id, provider_symbol, fetch

            fetched: list[tuple[str, str, str, object]] = []
            if workers_n == 1 or len(resolved_batch) == 1:
                for item in resolved_batch:
                    try:
                        fetched.append(_fetch_one(item))
                    except Exception as exc:  # noqa: BLE001
                        canonical, _, provider_symbol = item
                        logger.error("Failed to fetch %s (%s): %s", canonical, provider_symbol, exc)
                        failed += 1
                        failures.append(SymbolFailure(canonical, str(exc)))
                        state.setdefault("failed", []).append({"symbol": canonical, "error": str(exc)})
            else:
                with ThreadPoolExecutor(max_workers=min(workers_n, len(resolved_batch))) as pool:
                    future_map = {pool.submit(_fetch_one, item): item for item in resolved_batch}
                    for fut in as_completed(future_map):
                        item = future_map[fut]
                        canonical, _, provider_symbol = item
                        try:
                            fetched.append(fut.result())
                        except Exception as exc:  # noqa: BLE001
                            logger.error("Failed to fetch %s (%s): %s", canonical, provider_symbol, exc)
                            failed += 1
                            failures.append(SymbolFailure(canonical, str(exc)))
                            state.setdefault("failed", []).append({"symbol": canonical, "error": str(exc)})

            for canonical, security_id, provider_symbol, fetch in fetched:
                try:
                    df = normalize_price_rows(fetch, security_id, canonical)
                    rows_received += len(getattr(fetch, "rows", []) or [])
                    if df.height > 0:
                        batch_price_frames.append(df)
                    else:
                        logger.warning("No rows returned for %s (%s)", canonical, provider_symbol)
                    corp_df = derive_corporate_actions(fetch, security_id)
                    if corp_df.height > 0:
                        batch_corp_frames.append(corp_df)
                    if fetch.retrieved_at is not None:
                        batch_retrieved_at = (
                            fetch.retrieved_at
                            if batch_retrieved_at is None
                            else max(batch_retrieved_at, fetch.retrieved_at)
                        )
                    batch_ok.append(canonical)
                    successful += 1
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to normalize/write %s (%s): %s", canonical, provider_symbol, exc)
                    failed += 1
                    failures.append(SymbolFailure(canonical, str(exc)))
                    state.setdefault("failed", []).append({"symbol": canonical, "error": str(exc)})

            # One write per batch, grouped by year/month inside write_increment.
            # That is ~1 file per month per batch, not 1 file per symbol per month.
            if batch_price_frames:
                combined = pl.concat(batch_price_frames, how="vertical_relaxed")
                write_result = lake.write_increment(combined, run_id)
                rows_written += write_result.rows_written
                retrieved = batch_retrieved_at or datetime.now(UTC)
                for path in write_result.files_written:
                    record_source_file(
                        con,
                        run_id,
                        settings.price_provider,
                        "prices_daily",
                        path,
                        retrieved,
                        row_count=write_result.rows_written,
                        min_date=write_result.min_date,
                        max_date=write_result.max_date,
                    )
            if batch_corp_frames:
                corp_combined = pl.concat(batch_corp_frames, how="vertical_relaxed")
                corp_write = corp_actions_lake.write_increment(corp_combined, run_id)
                retrieved = batch_retrieved_at or datetime.now(UTC)
                for path in corp_write.files_written:
                    record_source_file(
                        con,
                        run_id,
                        settings.price_provider,
                        "corporate_actions",
                        path,
                        retrieved,
                        row_count=corp_write.rows_written,
                        min_date=corp_write.min_date,
                        max_date=corp_write.max_date,
                    )
            state["completed"].extend(batch_ok)

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

        return BackfillResult(
            run_id,
            len(targets),
            successful,
            failed,
            rows_written,
            failures,
            resumed,
            job_key,
            symbols_requested=len(targets),
            batches_requested=batches_requested,
            rows_received=rows_received,
            provider_fetch_count=provider_fetch_count,
        )

    except Exception as exc:  # noqa: BLE001
        finish_run(con, run_id, "failed", error_message=str(exc))
        raise
