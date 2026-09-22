"""Incremental labels_forward_returns computation with checkpoint/resume."""

from __future__ import annotations

from datetime import date

import duckdb

from app.config.lake_datasets import get_lake_dataset
from app.config.settings import Settings
from app.ingestion.checkpoint import CheckpointStore, make_job_key
from app.ingestion.compute_common import (
    lookahead_end,
    load_prices,
    month_partition_count,
    resolve_feature_targets,
)
from app.ingestion.feature_compute import ComputePlan, ComputeResult
from app.labels.engine import LabelEngine
from app.labels.schema import LABEL_VERSION_V1
from app.services.manifest_service import finish_run, record_source_file, start_run
from app.utils.logging import get_logger
from app.utils.versioning import get_calculation_code_version

logger = get_logger("label_compute")

LABEL_LOOKAHEAD_SESSIONS = 20


def plan_label_compute(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    symbols: list[str] | None,
    start: date,
    end: date | None,
    version: str,
) -> ComputePlan:
    targets = resolve_feature_targets(con, symbols)
    end_d = end or date.today()
    return ComputePlan(
        kind="labels",
        targets=len(targets),
        tickers=[t for t, _ in targets],
        start=start,
        end=end,
        version=version,
        lookback_sessions=0,
        lookahead_sessions=LABEL_LOOKAHEAD_SESSIONS,
        estimated_partitions=month_partition_count(start, end_d),
        batch_size=settings.feature_batch_size,
    )


def compute_labels(
    settings: Settings,
    con: duckdb.DuckDBPyConnection,
    symbols: list[str] | None,
    start: date,
    end: date | None,
    version: str = LABEL_VERSION_V1,
    resume: bool = False,
    dry_run: bool = False,
) -> ComputeResult:
    if not settings.derived_data_persistence_enabled:
        logger.info("SKIPPED - derived persistence disabled (labels_forward_returns)")
        return ComputeResult(
            run_id=None,
            total_symbols=0,
            successful=0,
            failed=0,
            rows_written=0,
            dry_run=dry_run,
        )

    targets = resolve_feature_targets(con, symbols)
    plan = plan_label_compute(con, settings, symbols, start, end, version)
    if dry_run:
        logger.info(
            "[dry-run] Label computation plan: %d tracked securities, start=%s end=%s "
            "version=%s lookahead=%d sessions, estimated partitions=%d. No writes performed.",
            plan.targets,
            plan.start,
            plan.end or "latest available",
            plan.version,
            plan.lookahead_sessions,
            plan.estimated_partitions,
        )
        return ComputeResult(
            run_id=None,
            total_symbols=len(targets),
            successful=len(targets),
            failed=0,
            rows_written=0,
            dry_run=True,
            plan=plan,
        )

    if not targets:
        raise ValueError(
            "No feature-tracking securities resolved. Use 'stockdb universe add' "
            "or pass --symbols. The full security master is never selected automatically."
        )

    read_end = lookahead_end(settings, end or date.today(), LABEL_LOOKAHEAD_SESSIONS)
    params = {
        "symbols": sorted(t for t, _ in targets) if symbols else "FEATURE_TRACKED",
        "start": start.isoformat(),
        "end": end.isoformat() if end else None,
        "version": version,
    }
    job_key = make_job_key("compute_labels", params)
    store = CheckpointStore(settings.checkpoints_dir)
    state = store.load(job_key) if resume else None
    resumed = state is not None
    if state is None:
        state = {"completed": [], "failed": [], "status": "in_progress"}
    completed = set(state.get("completed", []))
    pending = [(t, sid) for t, sid in targets if sid not in completed and t not in completed]

    lake = get_lake_dataset(settings.lake_dir, "labels_forward_returns")
    run_id = start_run(con, "local_compute", "labels_forward_returns", params)
    engine = LabelEngine()
    code_version = get_calculation_code_version()
    rows_written = 0
    successful = 0
    failed = 0
    failures: list[dict] = []
    batch_size = settings.feature_batch_size

    try:
        while pending:
            batch = pending[:batch_size]
            batch_ids = [sid for _, sid in batch]
            prices = load_prices(con, settings, batch_ids, start, read_end)
            if prices.height == 0:
                for ticker, _sid in batch:
                    logger.warning("No price rows for %s", ticker)
                    failed += 1
                    failures.append({"symbol": ticker, "error": "no price data"})
                    state.setdefault("failed", []).append({"symbol": ticker, "error": "no price data"})
                pending = pending[len(batch) :]
                store.save(job_key, state)
                continue

            write_end = end or prices["date"].max()
            result_df = engine.calculate_labels(
                prices,
                start,
                write_end,
                label_version=version,
                security_ids=batch_ids,
                calculation_code_version=code_version,
            )
            if result_df.height > 0:
                write = lake.write_increment(result_df, run_id)
                rows_written += write.rows_written
                for path in write.files_written:
                    record_source_file(
                        con,
                        run_id,
                        "local_compute",
                        "labels_forward_returns",
                        path,
                        result_df["calculated_at"][0],
                        row_count=result_df.height,
                        min_date=write.min_date,
                        max_date=write.max_date,
                    )
            for _ticker, sid in batch:
                successful += 1
                state["completed"].append(sid)
            pending = pending[len(batch) :]
            store.save(job_key, state)
            logger.info("Label batch done: %s (%d rows so far)", [t for t, _ in batch], rows_written)

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
        if failed == 0:
            store.clear(job_key)
        return ComputeResult(
            run_id,
            len(targets),
            successful,
            failed,
            rows_written,
            failures,
            plan=plan,
            checkpoint_job_key=job_key,
            resumed=resumed,
        )
    except Exception as exc:  # noqa: BLE001
        finish_run(con, run_id, "failed", error_message=str(exc))
        raise
