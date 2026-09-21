"""Parquet partition compaction (``stockdb compact <dataset>``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from app.config.lake_datasets import get_lake_dataset, get_spec, resolve_dataset_key
from app.config.settings import Settings
from app.db.schema import create_lake_views
from app.utils.parquet_io import CompactResult, LakeDataset, PartitionKey


class CompactVerificationError(RuntimeError):
    """Logical snapshot after compaction does not match the pre-compact snapshot."""


@dataclass(frozen=True)
class LogicalSnapshot:
    rows: int
    securities: int
    min_date: str | None
    max_date: str | None
    duplicates: int
    files: int


@dataclass
class VerifiedCompactResult:
    dataset: str
    partitions: list[CompactResult]
    before: LogicalSnapshot
    after: LogicalSnapshot
    dry_run: bool


def get_dataset(lake_dir: Path, name: str) -> LakeDataset:
    """Backwards-compatible alias -- prefer ``app.config.lake_datasets.get_lake_dataset``."""
    return get_lake_dataset(lake_dir, name)


def snapshot_dataset(settings: Settings, name: str, con: duckdb.DuckDBPyConnection | None = None) -> LogicalSnapshot:
    """Logical (last-write-wins) row stats plus physical file count."""
    key = resolve_dataset_key(name)
    ds = get_lake_dataset(settings.lake_dir, key)
    files = len(ds.all_files())
    if files == 0:
        return LogicalSnapshot(0, 0, None, None, 0, 0)

    if con is not None:
        create_lake_views(con, settings)
        date_col = ds.date_column
        keys = ", ".join(ds.dedup_keys)
        cols = [r[0] for r in con.execute(f"DESCRIBE {key}").fetchall()]
        nsec_expr = "count(DISTINCT security_id)" if "security_id" in cols else "0"
        rows, nsec, mn, mx = con.execute(
            f"SELECT count(*), {nsec_expr}, min({date_col}), max({date_col}) FROM {key}"
        ).fetchone()
        dups = con.execute(
            f"SELECT count(*) FROM (SELECT {keys}, count(*) n FROM {key} GROUP BY {keys} HAVING n > 1)"
        ).fetchone()[0]
        return LogicalSnapshot(
            int(rows),
            int(nsec),
            str(mn) if mn is not None else None,
            str(mx) if mx is not None else None,
            int(dups),
            files,
        )

    securities: set[str] = set()
    total = 0
    min_date: str | None = None
    max_date: str | None = None
    for part_key in ds.list_partitions():
        deduped = ds.read_partition_deduped(part_key)
        if deduped.height == 0:
            continue
        total += deduped.height
        if "security_id" in deduped.columns:
            securities.update(deduped["security_id"].unique().to_list())
        col = deduped[ds.date_column]
        part_min, part_max = str(col.min()), str(col.max())
        if min_date is None or part_min < min_date:
            min_date = part_min
        if max_date is None or part_max > max_date:
            max_date = part_max
    return LogicalSnapshot(total, len(securities), min_date, max_date, 0, files)


def compact_dataset(
    lake_dir: Path, name: str, year: int | None = None, month: int | None = None, dry_run: bool = False
) -> list[CompactResult]:
    key = resolve_dataset_key(name)
    spec = get_spec(key)
    ds = get_lake_dataset(lake_dir, key)

    if spec.granularity == "year" and month is not None:
        raise ValueError(
            f"Dataset '{name}' uses yearly partitions (year={{'{spec.dir_name}'}}/year=YYYY/) -- "
            f"--month is not applicable. Use --year only."
        )

    if year is not None:
        partitions = [ds.make_key(year, month)]
    else:
        partitions = ds.list_partitions()

    results: list[CompactResult] = []
    for partition_key in partitions:
        if dry_run:
            files = sorted(ds.partition_dir(partition_key).glob("*.parquet"))
            if len(files) > 1:
                results.append(CompactResult(partition_key, len(files), 1, -1, -1))
            continue
        result = ds.compact_partition(partition_key)
        if result is not None:
            results.append(result)
    return results


def compact_dataset_verified(
    settings: Settings,
    name: str,
    year: int | None = None,
    month: int | None = None,
    dry_run: bool = False,
    con: duckdb.DuckDBPyConnection | None = None,
) -> VerifiedCompactResult:
    """Compact then assert logical rows/securities/dates/dups are unchanged."""
    before = snapshot_dataset(settings, name, con)
    results = compact_dataset(settings.lake_dir, name, year, month, dry_run=dry_run)
    after = snapshot_dataset(settings, name, con)
    if not dry_run:
        if (
            before.rows != after.rows
            or before.securities != after.securities
            or before.min_date != after.min_date
            or before.max_date != after.max_date
            or before.duplicates != after.duplicates
        ):
            raise CompactVerificationError(
                f"Compaction changed logical snapshot for {name}: before={before} after={after}"
            )
    return VerifiedCompactResult(resolve_dataset_key(name), results, before, after, dry_run)


def format_partition_label(key: PartitionKey) -> str:
    return key.label()
