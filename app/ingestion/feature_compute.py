"""Incremental features_daily computation with checkpoint/resume."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import duckdb

from app.config.lake_datasets import get_lake_dataset
from app.config.settings import Settings
from app.features.engine import FeatureEngine
from app.features.schema import FEATURE_VERSION_V1
from app.ingestion.checkpoint import CheckpointStore, make_job_key
from app.ingestion.compute_common import (
    load_prices,
    load_vix,
    lookback_start,
    month_partition_count,
    resolve_feature_targets,
)
from app.services.manifest_service import finish_run, record_source_file, start_run
from app.utils.logging import get_logger
from app.utils.versioning import get_calculation_code_version

logger = get_logger("feature_compute")


@dataclass
class ComputePlan:
    kind: str
    targets: int
    tickers: list[str]
    start: date
    end: date | None
    version: str
    lookback_sessions: int
    lookahead_sessions: int
    estimated_partitions: int
    batch_size: int


@dataclass
class ComputeResult:
    run_id: str | None
    total_symbols: int
    successful: int
    failed: int
    rows_written: int
    failures: list[dict] = field(default_factory=list)
    dry_run: bool = False
    plan: ComputePlan | None = None
    checkpoint_job_key: str | None = None
    resumed: bool = False


def plan_feature_compute(
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
        kind="features",
        targets=len(targets),
        tickers=[t for t, _ in targets],
        start=start,
        end=end,
        version=version,
        lookback_sessions=settings.max_feature_lookback_sessions,
        lookahead_sessions=0,
        estimated_partitions=month_partition_count(start, end_d),
        batch_size=settings.feature_batch_size,
    )


def compute_features(
    settings: Settings,
    con: duckdb.DuckDBPyConnection,
    symbols: list[str] | None,
    start: date,
    end: date | None,
    version: str = FEATURE_VERSION_V1,
    resume: bool = False,
    dry_run: bool = False,
) -> ComputeResult:
    if not settings.derived_data_persistence_enabled:
        logger.info("SKIPPED - derived persistence disabled (features_daily)")
        return ComputeResult(
            run_id=None,
            total_symbols=0,
            successful=0,
            failed=0,
            rows_written=0,
            dry_run=dry_run,
        )

    targets = resolve_feature_targets(con, symbols)
    plan = plan_feature_compute(con, settings, symbols, start, end, version)
    if dry_run:
        logger.info(
            "[dry-run] Feature computation plan: %d tracked securities, start=%s end=%s "
            "version=%s lookback=%d sessions, estimated partitions=%d. No writes performed.",
            plan.targets,
            plan.start,
            plan.end or "latest available",
            plan.version,
            plan.lookback_sessions,
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

    end_d = end
    read_start = lookback_start(settings, start)
    params = {
        "symbols": sorted(t for t, _ in targets) if symbols else "FEATURE_TRACKED",
        "start": start.isoformat(),
        "end": end_d.isoformat() if end_d else None,
        "version": version,
    }
    job_key = make_job_key("compute_features", params)
    store = CheckpointStore(settings.checkpoints_dir)
    state = store.load(job_key) if resume else None
    resumed = state is not None
    if state is None:
        state = {"completed": [], "failed": [], "status": "in_progress"}
    completed = set(state.get("completed", []))
    pending = [(t, sid) for t, sid in targets if sid not in completed and t not in completed]

    lake = get_lake_dataset(settings.lake_dir, "features_daily")
    run_id = start_run(con, "local_compute", "features_daily", params)
    engine = FeatureEngine()
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
            prices = load_prices(con, settings, batch_ids, read_start, end_d or date(2100, 1, 1))
            if prices.height == 0:
                for ticker, sid in batch:
                    logger.warning("No price rows for %s (%s)", ticker, sid)
                    failed += 1
                    failures.append({"symbol": ticker, "error": "no price data"})
                    state.setdefault("failed", []).append({"symbol": ticker, "error": "no price data"})
                pending = pending[len(batch) :]
                store.save(job_key, state)
                continue

            write_end = end_d or prices["date"].max()
            vix = load_vix(con, settings, read_start, write_end)
            result_df = engine.calculate_features(
                prices,
                start,
                write_end,
                feature_version=version,
                security_ids=batch_ids,
                vix=vix,
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
                        "features_daily",
                        path,
                        result_df["calculated_at"][0],
                        row_count=result_df.height,
                        min_date=write.min_date,
                        max_date=write.max_date,
                    )
            for ticker, sid in batch:
                successful += 1
                state["completed"].append(sid)
            pending = pending[len(batch) :]
            store.save(job_key, state)
            logger.info("Feature batch done: %s (%d rows so far)", [t for t, _ in batch], rows_written)

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
