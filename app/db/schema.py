"""DuckDB metadata/catalog schema.

Only reference data, job bookkeeping, and data-quality findings live in
DuckDB tables. Bulk OHLCV/macro/etc. time series live in the Parquet lake
(``app/utils/parquet_io.py``) and are exposed to SQL through views created
by ``create_lake_views``.
"""

from __future__ import annotations

import duckdb

_DDL_STATEMENTS: list[str] = [
    # --- job / provenance bookkeeping -----------------------------------------
    """
    CREATE TABLE IF NOT EXISTS ingest_runs (
        run_id            TEXT PRIMARY KEY,
        provider          TEXT NOT NULL,
        dataset           TEXT NOT NULL,
        started_at        TIMESTAMP NOT NULL,
        finished_at       TIMESTAMP,
        status            TEXT NOT NULL,       -- running | success | partial | failed
        requested_items   INTEGER,
        successful_items  INTEGER,
        failed_items      INTEGER,
        rows_written      BIGINT,
        error_message     TEXT,
        params_json       TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS source_files (
        file_id       TEXT PRIMARY KEY,
        run_id        TEXT NOT NULL,
        provider      TEXT NOT NULL,
        dataset       TEXT NOT NULL,
        path          TEXT NOT NULL,
        retrieved_at  TIMESTAMP NOT NULL,
        checksum      TEXT,
        row_count     BIGINT,
        min_date      DATE,
        max_date      DATE
    )
    """,
    # --- security master -------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS securities (
        security_id     TEXT PRIMARY KEY,
        cik             TEXT,
        company_name    TEXT,
        primary_ticker  TEXT,
        exchange        TEXT,
        asset_type      TEXT,
        currency        TEXT,
        is_active       BOOLEAN NOT NULL DEFAULT TRUE,
        first_seen_at   TIMESTAMP NOT NULL,
        last_seen_at    TIMESTAMP NOT NULL,
        created_at      TIMESTAMP NOT NULL,
        updated_at      TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS security_identifiers (
        security_id       TEXT NOT NULL,
        identifier_type    TEXT NOT NULL,   -- TICKER | CIK | ISIN | ...
        identifier_value   TEXT NOT NULL,
        valid_from         DATE NOT NULL,
        valid_to           DATE,
        source             TEXT NOT NULL,
        PRIMARY KEY (security_id, identifier_type, identifier_value, valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS security_snapshots (
        snapshot_date   DATE NOT NULL,
        security_id     TEXT NOT NULL,
        ticker          TEXT,
        company_name    TEXT,
        exchange        TEXT,
        source          TEXT NOT NULL,
        PRIMARY KEY (snapshot_date, security_id)
    )
    """,
    # --- symbol mapping (canonical <-> provider ticker spelling) ----------------
    """
    CREATE TABLE IF NOT EXISTS symbol_mappings (
        canonical_symbol  TEXT NOT NULL,
        provider          TEXT NOT NULL,
        provider_symbol   TEXT NOT NULL,
        source            TEXT NOT NULL,
        updated_at        TIMESTAMP NOT NULL,
        PRIMARY KEY (canonical_symbol, provider)
    )
    """,
    # --- data quality -------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS data_quality_issues (
        issue_id      TEXT PRIMARY KEY,
        dataset       TEXT NOT NULL,
        security_id   TEXT,
        date          DATE,
        issue_type    TEXT NOT NULL,
        severity      TEXT NOT NULL,        -- critical | warning | info
        details       TEXT,
        detected_at   TIMESTAMP NOT NULL,
        resolved      BOOLEAN NOT NULL DEFAULT FALSE
    )
    """,
    # --- tracked (operational) universe --------------------------------------------
    # ``securities`` is everything known from the security master. This table is
    # the much smaller subset we actually collect data for and validate.
    """
    CREATE TABLE IF NOT EXISTS tracked_securities (
        security_id       TEXT PRIMARY KEY,
        enabled           BOOLEAN NOT NULL DEFAULT TRUE,
        tracking_reason   TEXT,
        added_at          TIMESTAMP NOT NULL,
        removed_at        TIMESTAMP,
        price_tracking    BOOLEAN NOT NULL DEFAULT TRUE,
        filings_tracking  BOOLEAN NOT NULL DEFAULT FALSE,
        feature_tracking  BOOLEAN NOT NULL DEFAULT FALSE,
        notes             TEXT
    )
    """,
    # Heuristic instrument classes. Not a complete security master
    # (instrument_class_complete=false). A security may be AMBIGUOUS.
    """
    CREATE TABLE IF NOT EXISTS instrument_classifications (
        security_id                 TEXT PRIMARY KEY,
        ticker                      TEXT,
        instrument_class            TEXT NOT NULL,
        classification_source       TEXT NOT NULL,
        classification_confidence   TEXT NOT NULL,
        exclusion_reason            TEXT,
        updated_at                  TIMESTAMP NOT NULL
    )
    """,
    # Named universes (scale-test vs research common equity vs benchmark).
    # Distinct from tracked_securities: one security can belong to several.
    """
    CREATE TABLE IF NOT EXISTS universe_memberships (
        universe_name       TEXT NOT NULL,
        security_id         TEXT NOT NULL,
        universe_type       TEXT NOT NULL,
        selection_version   TEXT NOT NULL,
        selection_rule      TEXT NOT NULL,
        added_at            TIMESTAMP NOT NULL,
        PRIMARY KEY (universe_name, security_id)
    )
    """,
    # --- research-integrity / survivorship-bias metadata ---------------------------
    # Machine-readable counterpart to the README limitations section, so a future
    # features/labels/backtest layer can programmatically check e.g.
    # "is this price data survivorship-safe?" instead of relying on humans
    # having read the docs.
    """
    CREATE TABLE IF NOT EXISTS dataset_metadata (
        dataset_name    TEXT NOT NULL,
        metadata_key    TEXT NOT NULL,
        metadata_value  TEXT NOT NULL,
        updated_at      TIMESTAMP NOT NULL,
        PRIMARY KEY (dataset_name, metadata_key)
    )
    """,
]

_INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_source_files_run_id ON source_files(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_security_identifiers_value ON security_identifiers(identifier_value)",
    "CREATE INDEX IF NOT EXISTS idx_dqi_dataset ON data_quality_issues(dataset, resolved)",
    "CREATE INDEX IF NOT EXISTS idx_universe_memberships_type ON universe_memberships(universe_type, universe_name)",
    "CREATE INDEX IF NOT EXISTS idx_instrument_class ON instrument_classifications(instrument_class)",
]


def apply_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create all metadata tables/indexes if they do not already exist."""
    for stmt in _DDL_STATEMENTS:
        con.execute(stmt)
    for stmt in _INDEXES:
        con.execute(stmt)


def create_lake_views(con: duckdb.DuckDBPyConnection, settings) -> None:  # noqa: ANN001
    """Create/refresh DuckDB views over the Parquet lake for convenient SQL.

    Each view applies a "last write wins" dedup (by the dataset's
    ``tie_break_column``) over ``dedup_keys`` so callers never have to think
    about overlapping re-ingestion / re-compute runs. Views are skipped
    (not created) for datasets that have no Parquet files yet.

    ``research_daily`` is a convenience join of features + labels. Feature
    calculation code must not read it. ``research_common_equity_daily_v1``
    is the analysis sample (RESEARCH_COMMON_EQUITY only; no benchmark ETF
    rows).
    """
    from app.config.lake_datasets import get_lake_dataset, get_spec, implemented_dataset_keys

    created: set[str] = set()
    for name in implemented_dataset_keys():
        spec = get_spec(name)
        ds = get_lake_dataset(settings.lake_dir, name)
        if not ds.has_any_files():
            continue
        glob_path = ds.glob_pattern().replace("'", "''")
        key_cols = ", ".join(ds.dedup_keys)
        tie = spec.tie_break_column
        sql = f"""
            CREATE OR REPLACE VIEW {name} AS
            SELECT * EXCLUDE (__rn)
            FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY {key_cols}
                    ORDER BY {tie} DESC
                ) AS __rn
                FROM read_parquet('{glob_path}', hive_partitioning = true)
            )
            WHERE __rn = 1
        """
        con.execute(sql)
        created.add(name)

    _create_research_daily_view(con, created)
    from app.research.dataset import create_research_common_equity_view

    create_research_common_equity_view(con, created)


def _create_research_daily_view(con: duckdb.DuckDBPyConnection, created: set[str]) -> None:
    if "features_daily" not in created or "labels_forward_returns" not in created:
        return
    from app.features.schema import FEATURE_VALUE_COLUMNS
    from app.labels.schema import LABEL_VALUE_COLUMNS

    feature_cols = ", ".join(f"f.{c}" for c in FEATURE_VALUE_COLUMNS)
    label_cols = ", ".join(f"l.{c}" for c in LABEL_VALUE_COLUMNS)
    sql = f"""
        CREATE OR REPLACE VIEW research_daily AS
        SELECT
            f.security_id,
            f.ticker_at_time AS ticker,
            f.date,
            f.feature_version,
            l.label_version,
            {feature_cols},
            {label_cols}
        FROM features_daily f
        LEFT JOIN labels_forward_returns l
          ON f.security_id = l.security_id
         AND f.date = l.date
    """
    con.execute(sql)
