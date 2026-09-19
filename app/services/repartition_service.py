"""One-off, safe migration of a dataset's on-disk partition granularity.

Used to convert ``corporate_actions`` and ``volatility`` from
``year=YYYY/month=MM/`` to ``year=YYYY/`` partitioning without ever deleting
existing data or requiring a re-backfill (see README "Parquet partition
전략" and the storage-health "Partition policy" panel).

Strategy (temp write -> validate -> atomic replace, same principle as every
other write path in this project, just at the directory level):

1. Read every row from the dataset's *current* on-disk layout (whatever it
   actually is, independent of the dataset's *configured* granularity --
   this lets the migration run even after the config has already switched),
   deduplicated exactly the same way normal reads are (last write wins by
   ``retrieved_at``).
2. Write one consolidated file per year into a fresh staging directory,
   validating each file's row count as it's written.
3. Before touching anything live, re-read the staging directory and confirm
   total row count and min/max date exactly match what was computed from
   the original data. Abort (and clean up the staging directory) if not --
   the original directory is never touched in this case.
4. Only then: rename the live dataset directory to a timestamped backup,
   then rename the staging directory into its place. Both renames are
   single filesystem operations on the same volume (as atomic as a
   directory swap can be). If the second rename fails, the first is
   immediately undone so the dataset directory is never left missing.
5. The backup directory is kept on disk (never auto-deleted) so a human can
   inspect it or manually rename it back if anything looks wrong -- that is
   the "rollback" path.
"""

from __future__ import annotations

import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from app.config.lake_datasets import get_spec
from app.config.settings import Settings
from app.utils.atomic_io import atomic_write_via
from app.utils.parquet_io import LakeDataset, PartitionKey, dedup_dataframe


@dataclass
class RepartitionResult:
    dataset: str
    files_before: int = 0
    files_after: int = 0
    rows_before: int = 0
    rows_after: int = 0
    min_date_before: str | None = None
    max_date_before: str | None = None
    min_date_after: str | None = None
    max_date_after: str | None = None
    years_migrated: list[int] = field(default_factory=list)
    rows_by_year: dict[int, int] = field(default_factory=dict)
    backup_dir: Path | None = None
    elapsed_seconds: float = 0.0
    dry_run: bool = False
    skipped_reason: str | None = None


def _read_year_raw(base_dir: Path, year: int) -> pl.DataFrame:
    """All rows for one year from the OLD monthly layout, undeduped."""
    files = sorted((base_dir / f"year={year:04d}").glob("month=*/*.parquet"))
    if not files:
        return pl.DataFrame()
    frames = [pl.read_parquet(f) for f in files]
    return pl.concat(frames, how="vertical_relaxed")


def repartition_to_yearly(settings: Settings, dataset_key: str, dry_run: bool = False) -> RepartitionResult:
    spec = get_spec(dataset_key)
    if spec.granularity != "year":
        raise ValueError(
            f"Dataset '{dataset_key}' is configured for '{spec.granularity}' partitions, not yearly -- "
            "nothing to migrate."
        )

    base_dir = settings.lake_dir / spec.dir_name

    # Read whatever is CURRENTLY on disk as monthly, regardless of the
    # dataset's now-yearly configured granularity.
    old_ds = LakeDataset(base_dir, spec.date_column, spec.dedup_keys, spec.sort_keys, granularity="month")
    if not old_ds.has_any_files():
        return RepartitionResult(
            dataset=dataset_key, skipped_reason="no month=-partitioned files found (already migrated, or no data yet)"
        )

    files_before = len(old_ds.all_files())
    rows_before, min_before, max_before = old_ds.aggregate_stats()
    years = sorted({key.year for key in old_ds.list_partitions()})

    if dry_run:
        return RepartitionResult(
            dataset=dataset_key,
            files_before=files_before,
            files_after=len(years),
            rows_before=rows_before,
            rows_after=rows_before,
            min_date_before=min_before,
            max_date_before=max_before,
            min_date_after=min_before,
            max_date_after=max_before,
            years_migrated=years,
            dry_run=True,
        )

    t0 = time.monotonic()
    staging_dir = base_dir.parent / f".{base_dir.name}__migrating_{uuid.uuid4().hex[:8]}"
    staging_dir.mkdir(parents=True, exist_ok=False)
    rows_by_year: dict[int, int] = {}

    try:
        for year in years:
            raw = _read_year_raw(base_dir, year)
            deduped = dedup_dataframe(raw, spec.dedup_keys, spec.sort_keys)
            if deduped.height == 0:
                continue
            rows_by_year[year] = deduped.height

            target_dir = staging_dir / f"year={year:04d}"
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / f"repartitioned-{uuid.uuid4().hex[:12]}.parquet"

            def _writer(tmp_path: Path, _df: pl.DataFrame = deduped) -> None:
                _df.write_parquet(tmp_path, compression="zstd")

            def _validator(tmp_path: Path, _expected_rows: int = deduped.height) -> None:
                meta = pq.read_metadata(tmp_path)
                if meta.num_rows != _expected_rows:
                    raise ValueError(
                        f"Repartition validation failed writing year={year}: "
                        f"expected {_expected_rows} rows, found {meta.num_rows}"
                    )

            atomic_write_via(target_path, _writer, _validator)

        # Validate the fully-written staging area BEFORE touching anything live.
        staging_ds = LakeDataset(staging_dir, spec.date_column, spec.dedup_keys, spec.sort_keys, granularity="year")
        rows_after, min_after, max_after = staging_ds.aggregate_stats()
        files_after = len(staging_ds.all_files())

        if rows_after != rows_before:
            raise RuntimeError(
                f"Repartition aborted for '{dataset_key}': row count mismatch "
                f"({rows_before} before vs {rows_after} after). Original data untouched."
            )
        if (min_after, max_after) != (min_before, max_before):
            raise RuntimeError(
                f"Repartition aborted for '{dataset_key}': date range mismatch "
                f"({min_before}..{max_before} before vs {min_after}..{max_after} after). Original data untouched."
            )
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    # Atomic-as-possible directory swap, with a backup kept for rollback.
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = base_dir.parent / f"{base_dir.name}__pre_yearly_backup_{timestamp}"
    os.rename(base_dir, backup_dir)
    try:
        os.rename(staging_dir, base_dir)
    except Exception:
        os.rename(backup_dir, base_dir)  # roll back: restore the original directory
        raise

    elapsed = time.monotonic() - t0
    return RepartitionResult(
        dataset=dataset_key,
        files_before=files_before,
        files_after=files_after,
        rows_before=rows_before,
        rows_after=rows_after,
        min_date_before=min_before,
        max_date_before=max_before,
        min_date_after=min_after,
        max_date_after=max_after,
        years_migrated=years,
        rows_by_year=rows_by_year,
        backup_dir=backup_dir,
        elapsed_seconds=elapsed,
    )


def pending_yearly_migrations(settings: Settings) -> list[str]:
    """Dataset keys configured for yearly partitions that still have
    month=-partitioned files on disk."""
    from app.config.lake_datasets import LAKE_DATASET_SPECS

    pending = []
    for key, spec in LAKE_DATASET_SPECS.items():
        if spec.granularity != "year" or not spec.implemented:
            continue
        base_dir = settings.lake_dir / spec.dir_name
        if any(base_dir.glob("year=*/month=*")):
            pending.append(key)
    return pending
