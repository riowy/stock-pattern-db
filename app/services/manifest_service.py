"""Provenance bookkeeping: ``ingest_runs`` / ``source_files`` helpers.

Every ingestion command (universe sync, price backfill, macro sync, VIX
sync, filings sync) should wrap its work in ``start_run`` / ``finish_run``
so we always know which provider produced which rows and when.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import duckdb


def new_id() -> str:
    return uuid.uuid4().hex


def start_run(
    con: duckdb.DuckDBPyConnection, provider: str, dataset: str, params: dict | None = None
) -> str:
    run_id = new_id()
    con.execute(
        """
        INSERT INTO ingest_runs
            (run_id, provider, dataset, started_at, status,
             requested_items, successful_items, failed_items, rows_written, params_json)
        VALUES (?, ?, ?, ?, 'running', 0, 0, 0, 0, ?)
        """,
        [run_id, provider, dataset, datetime.now(UTC), json.dumps(params or {}, default=str)],
    )
    return run_id


def finish_run(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    status: str,
    requested_items: int = 0,
    successful_items: int = 0,
    failed_items: int = 0,
    rows_written: int = 0,
    error_message: str | None = None,
) -> None:
    con.execute(
        """
        UPDATE ingest_runs
        SET finished_at = ?, status = ?, requested_items = ?, successful_items = ?,
            failed_items = ?, rows_written = ?, error_message = ?
        WHERE run_id = ?
        """,
        [
            datetime.now(UTC),
            status,
            requested_items,
            successful_items,
            failed_items,
            rows_written,
            error_message,
            run_id,
        ],
    )


def checksum_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def record_source_file(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    provider: str,
    dataset: str,
    path: Path,
    retrieved_at: datetime,
    row_count: int | None = None,
    min_date: str | None = None,
    max_date: str | None = None,
    checksum: str | None = None,
) -> str:
    file_id = new_id()
    con.execute(
        """
        INSERT INTO source_files
            (file_id, run_id, provider, dataset, path, retrieved_at, checksum, row_count, min_date, max_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            file_id,
            run_id,
            provider,
            dataset,
            str(path),
            retrieved_at,
            checksum or checksum_file(path),
            row_count,
            min_date,
            max_date,
        ],
    )
    return file_id
