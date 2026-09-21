"""DuckDB connection management.

DuckDB here is used strictly as a *metadata / catalog* database (security
master, ingest run bookkeeping, data-quality findings) plus a convenient SQL
query layer over the Parquet lake (via views). Bulk time-series data is never
loaded wholesale into DuckDB tables -- see app/utils/parquet_io.py.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from app.config.settings import Settings


def _configure(con: duckdb.DuckDBPyConnection, settings: Settings) -> None:
    con.execute(f"SET threads = {int(settings.duckdb_threads)}")
    con.execute(f"SET memory_limit = '{settings.duckdb_memory_limit}'")


def get_connection(settings: Settings, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection against the on-disk catalog database."""
    db_path: Path = settings.duckdb_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=read_only)
    if not read_only:
        _configure(con, settings)
    else:
        # Still cap memory on read-only analytical connections.
        con.execute(f"SET memory_limit = '{settings.duckdb_memory_limit}'")
    return con


@contextmanager
def duckdb_connection(settings: Settings, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    con = get_connection(settings, read_only=read_only)
    try:
        yield con
    finally:
        con.close()


@contextmanager
def analytics_connection(settings: Settings) -> Iterator[duckdb.DuckDBPyConnection]:
    """In-memory DuckDB that ATTACH-es the catalog read-only.

    Lake views created on this connection stay in memory. The on-disk
    catalog file is not opened for write.
    """
    mem = duckdb.connect(":memory:")
    mem.execute(f"SET memory_limit = '{settings.duckdb_memory_limit}'")
    mem.execute(f"SET threads = {int(settings.duckdb_threads)}")
    path = str(settings.duckdb_path).replace("'", "''")
    mem.execute(f"ATTACH '{path}' AS catalog (READ_ONLY)")
    tables = mem.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_catalog = 'catalog'
          AND table_schema = 'main'
          AND table_type = 'BASE TABLE'
        """
    ).fetchall()
    for (name,) in tables:
        mem.execute(f'CREATE VIEW "{name}" AS SELECT * FROM catalog."{name}"')
    try:
        yield mem
    finally:
        mem.close()
