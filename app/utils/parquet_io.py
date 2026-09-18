"""Hive-partitioned Parquet lake writer/reader utilities.

Design (see README "Parquet partition 전략"):

* Data is partitioned by ``year=YYYY/month=MM`` (derived from a dataset's
  primary date column).
* Every ingestion run *appends* a new, uniquely named parquet file into the
  partitions it touched. We never read-modify-rewrite an entire partition on
  every ingest, because for a large universe that would mean repeatedly
  rewriting an ever-growing file (O(n^2) total I/O over a long backfill).
* Because the same date range can legitimately be re-ingested (re-runs,
  corrections, provider switch), the *same* logical row
  (``dedup_keys``) can appear in more than one file. We never delete or
  silently overwrite history for this: every row keeps its own
  ``retrieved_at`` / provider metadata. Instead, reads always go through a
  "last write wins" dedup step (keep the row with the max ``retrieved_at``
  per ``dedup_keys``), and an explicit ``compact`` operation later merges a
  partition's files into one and physically drops the superseded rows.
* Every write is temp-file -> validate -> ``os.replace`` (atomic), so a
  crash mid-write can never corrupt or truncate an existing file.
"""

from __future__ import annotations

import glob as glob_module
import uuid
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from app.utils.atomic_io import atomic_write_via


@dataclass
class WriteResult:
    rows_written: int
    partitions_written: list[tuple[int, int]]
    min_date: str | None
    max_date: str | None
    files_written: list[Path]


@dataclass
class CompactResult:
    partition: tuple[int, int]
    files_before: int
    files_after: int
    rows_before: int
    rows_after: int


class LakeDataset:
    """A single hive-partitioned Parquet dataset under ``data/lake/<name>``."""

    def __init__(
        self,
        base_dir: Path,
        date_column: str,
        dedup_keys: list[str],
        sort_keys: list[str] | None = None,
        tie_break_column: str = "retrieved_at",
    ) -> None:
        self.base_dir = base_dir
        self.date_column = date_column
        self.dedup_keys = dedup_keys
        self.sort_keys = sort_keys or dedup_keys
        self.tie_break_column = tie_break_column

    # ------------------------------------------------------------------ paths
    def partition_dir(self, year: int, month: int) -> Path:
        return self.base_dir / f"year={year:04d}" / f"month={month:02d}"

    def glob_pattern(self) -> str:
        return str(self.base_dir / "year=*" / "month=*" / "*.parquet")

    def has_any_files(self) -> bool:
        return len(glob_module.glob(self.glob_pattern())) > 0

    def all_files(self) -> list[Path]:
        return [Path(p) for p in glob_module.glob(self.glob_pattern())]

    # ------------------------------------------------------------------ write
    def write_increment(self, df: pl.DataFrame, run_id: str) -> WriteResult:
        """Append ``df`` to the lake, split across the partitions it belongs to.

        ``df`` must contain ``self.date_column`` as a ``date`` (or castable)
        column. Rows are grouped by year/month of that column.
        """
        if df.height == 0:
            return WriteResult(0, [], None, None, [])

        df = df.with_columns(pl.col(self.date_column).cast(pl.Date))
        df = df.with_columns(
            pl.col(self.date_column).dt.year().alias("__year"),
            pl.col(self.date_column).dt.month().alias("__month"),
        )

        partitions_written: list[tuple[int, int]] = []
        files_written: list[Path] = []
        total_rows = 0

        for (year, month), part_df in df.group_by(["__year", "__month"], maintain_order=True):
            part_df = part_df.drop(["__year", "__month"]).sort(self.sort_keys)
            target_dir = self.partition_dir(int(year), int(month))
            target_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"part-{run_id}-{uuid.uuid4().hex[:8]}.parquet"
            target_path = target_dir / file_name

            def _writer(tmp_path: Path, _part_df: pl.DataFrame = part_df) -> None:
                _part_df.write_parquet(tmp_path, compression="zstd")

            def _validator(tmp_path: Path, _expected_rows: int = part_df.height) -> None:
                meta = pq.read_metadata(tmp_path)
                if meta.num_rows != _expected_rows:
                    raise ValueError(
                        f"Parquet validation failed for {tmp_path}: "
                        f"expected {_expected_rows} rows, found {meta.num_rows}"
                    )

            atomic_write_via(target_path, _writer, _validator)
            partitions_written.append((int(year), int(month)))
            files_written.append(target_path)
            total_rows += part_df.height

        dates = df[self.date_column]
        return WriteResult(
            rows_written=total_rows,
            partitions_written=partitions_written,
            min_date=str(dates.min()),
            max_date=str(dates.max()),
            files_written=files_written,
        )

    # -------------------------------------------------------------- dedup/read
    def read_partition_deduped(self, year: int, month: int) -> pl.DataFrame:
        part_dir = self.partition_dir(year, month)
        files = sorted(part_dir.glob("*.parquet"))
        if not files:
            return pl.DataFrame()
        frames = [pl.read_parquet(f) for f in files]
        combined = pl.concat(frames, how="vertical_relaxed")
        return self._dedup(combined)

    def _dedup(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            return df
        sort_cols = [*self.dedup_keys, self.tie_break_column]
        df = df.sort(sort_cols)
        df = df.unique(subset=self.dedup_keys, keep="last", maintain_order=True)
        return df.sort(self.sort_keys)

    # ------------------------------------------------------------------ compact
    def compact_partition(self, year: int, month: int) -> CompactResult | None:
        part_dir = self.partition_dir(year, month)
        files = sorted(part_dir.glob("*.parquet"))
        if len(files) == 0:
            return None

        frames = [pl.read_parquet(f) for f in files]
        rows_before = sum(f.height for f in frames)
        combined = pl.concat(frames, how="vertical_relaxed")
        deduped = self._dedup(combined)

        if len(files) == 1 and rows_before == deduped.height:
            # Nothing to compact.
            return CompactResult((year, month), len(files), len(files), rows_before, deduped.height)

        new_name = f"compacted-{uuid.uuid4().hex[:12]}.parquet"
        new_path = part_dir / new_name

        def _writer(tmp_path: Path, _df: pl.DataFrame = deduped) -> None:
            _df.write_parquet(tmp_path, compression="zstd")

        def _validator(tmp_path: Path, _expected_rows: int = deduped.height) -> None:
            meta = pq.read_metadata(tmp_path)
            if meta.num_rows != _expected_rows:
                raise ValueError("Compaction validation failed: row count mismatch")

        atomic_write_via(new_path, _writer, _validator)

        # Only remove old files after the new consolidated file is safely in place.
        for f in files:
            f.unlink(missing_ok=True)

        return CompactResult((year, month), len(files), 1, rows_before, deduped.height)

    def list_partitions(self) -> list[tuple[int, int]]:
        partitions: set[tuple[int, int]] = set()
        for year_dir in sorted(self.base_dir.glob("year=*")):
            for month_dir in sorted(year_dir.glob("month=*")):
                try:
                    year = int(year_dir.name.split("=")[1])
                    month = int(month_dir.name.split("=")[1])
                except (IndexError, ValueError):
                    continue
                partitions.add((year, month))
        return sorted(partitions)
