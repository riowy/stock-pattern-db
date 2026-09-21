"""Compute services: dry-run, idempotency, checkpoint, research_daily join."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import duckdb
import polars as pl
import pytest

from app.db.schema import apply_schema, create_lake_views
from app.features.schema import FEATURE_VERSION_V1
from app.ingestion.feature_compute import compute_features
from app.ingestion.label_compute import compute_labels
from app.labels.schema import LABEL_VERSION_V1
from app.services.tracked_universe_service import add_tracked
from app.utils.parquet_io import LakeDataset


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    apply_schema(c)
    return c


def _insert_security(con, security_id="SEC0001", ticker="AAPL") -> None:  # noqa: ANN001
    now = datetime.now(UTC)
    con.execute(
        """
        INSERT INTO securities
            (security_id, cik, company_name, primary_ticker, exchange, asset_type,
             currency, is_active, first_seen_at, last_seen_at, created_at, updated_at)
        VALUES (?, '0001', 'Test', ?, 'NYSE', 'EQUITY', 'USD', TRUE, ?, ?, ?, ?)
        """,
        [security_id, ticker, now, now, now, now],
    )


def _write_prices(settings, security_id: str, ticker: str, n: int = 40) -> None:  # noqa: ANN001
    ds = LakeDataset(settings.prices_daily_dir, "date", ["security_id", "date"], ["security_id", "date"])
    start = date(2024, 1, 2)
    dates: list[date] = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    df = pl.DataFrame(
        {
            "security_id": [security_id] * n,
            "ticker_at_time": [ticker] * n,
            "date": dates,
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.0 + i for i in range(n)],
            "adj_close": [100.0 + i for i in range(n)],
            "volume": [1000.0] * n,
            "dividend": [0.0] * n,
            "stock_split": [0.0] * n,
            "currency": ["USD"] * n,
            "provider": ["test"] * n,
            "retrieved_at": [datetime.now(UTC)] * n,
        }
    )
    ds.write_increment(df, run_id="seed")


def test_dry_run_compute_features_no_write(con, settings, monkeypatch) -> None:
    _insert_security(con)
    add_tracked(con, ["SEC0001"], reason="test", feature_tracking=True)
    ingest_before = con.execute("SELECT count(*) FROM ingest_runs").fetchone()[0]
    result = compute_features(
        settings, con, None, date(2024, 1, 1), None, version="v1", dry_run=True
    )
    assert result.dry_run is True
    assert result.plan is not None
    assert result.plan.targets == 1
    assert result.plan.lookback_sessions == settings.max_feature_lookback_sessions
    assert con.execute("SELECT count(*) FROM ingest_runs").fetchone()[0] == ingest_before
    assert list(settings.checkpoints_dir.glob("*.json")) == []
    assert not any(settings.features_daily_dir.rglob("*.parquet"))


def test_dry_run_compute_labels_no_write(con, settings) -> None:
    _insert_security(con)
    add_tracked(con, ["SEC0001"], reason="test", feature_tracking=True)
    result = compute_labels(settings, con, None, date(2024, 1, 1), None, version="v1", dry_run=True)
    assert result.dry_run is True
    assert not any(settings.labels_forward_returns_dir.rglob("*.parquet"))
    assert list(settings.checkpoints_dir.glob("*.json")) == []


def test_compute_does_not_select_full_universe(con, settings) -> None:
    for i in range(20):
        _insert_security(con, security_id=f"SEC{i:04d}", ticker=f"T{i:04d}")
    add_tracked(con, ["SEC0001"], reason="test", feature_tracking=True)
    result = compute_features(settings, con, None, date(2024, 1, 1), None, dry_run=True)
    assert result.total_symbols == 1


def test_idempotent_feature_query_and_research_daily(con, settings) -> None:
    _insert_security(con, "SEC0001", "AAPL")
    _insert_security(con, "SPY1", "SPY")
    add_tracked(con, ["SEC0001", "SPY1"], reason="test", feature_tracking=True)
    _write_prices(settings, "SEC0001", "AAPL", n=40)
    _write_prices(settings, "SPY1", "SPY", n=40)

    r1 = compute_features(settings, con, None, date(2024, 1, 2), date(2024, 2, 15), version=FEATURE_VERSION_V1)
    create_lake_views(con, settings)
    n_before = con.execute(
        "SELECT count(*) FROM (SELECT DISTINCT security_id, date FROM features_daily)"
    ).fetchone()[0]
    r2 = compute_features(settings, con, None, date(2024, 1, 2), date(2024, 2, 15), version=FEATURE_VERSION_V1)
    create_lake_views(con, settings)
    n_after = con.execute(
        "SELECT count(*) FROM (SELECT DISTINCT security_id, date FROM features_daily)"
    ).fetchone()[0]
    assert r1.rows_written > 0
    assert r2.rows_written > 0
    assert n_after == n_before

    compute_labels(settings, con, None, date(2024, 1, 2), date(2024, 2, 15), version=LABEL_VERSION_V1)
    create_lake_views(con, settings)
    n = con.execute("SELECT count(*) FROM features_daily").fetchone()[0]
    n_keys = con.execute("SELECT count(*) FROM (SELECT DISTINCT security_id, date, feature_version FROM features_daily)").fetchone()[0]
    assert n == n_keys

    aapl_n = con.execute("SELECT count(*) FROM features_daily WHERE ticker_at_time = 'AAPL'").fetchone()[0]
    rows = con.execute(
        "SELECT ticker, date, ret_1d, forward_return_1d FROM research_daily WHERE ticker = 'AAPL' ORDER BY date"
    ).fetchall()
    assert len(rows) == aapl_n
    assert rows[0][0] == "AAPL"


def test_checkpoint_resume_features(con, settings) -> None:
    _insert_security(con, "SEC0001", "AAPL")
    _insert_security(con, "SEC0002", "MSFT")
    add_tracked(con, ["SEC0001", "SEC0002"], reason="test", feature_tracking=True)
    _write_prices(settings, "SEC0001", "AAPL", n=30)
    _write_prices(settings, "SEC0002", "MSFT", n=30)
    settings.feature_batch_size = 1

    from app.ingestion.checkpoint import CheckpointStore, make_job_key

    params = {
        "symbols": "FEATURE_TRACKED",
        "start": "2024-01-02",
        "end": "2024-02-15",
        "version": "v1",
    }
    job_key = make_job_key("compute_features", params)
    store = CheckpointStore(settings.checkpoints_dir)
    # Pretend AAPL (first by ticker? SEC0001) -- we complete by security_id
    store.save(job_key, {"completed": ["SEC0001"], "failed": [], "status": "in_progress"})

    result = compute_features(
        settings, con, None, date(2024, 1, 2), date(2024, 2, 15), version="v1", resume=True
    )
    assert result.resumed is True
    # Only the remaining symbol should have been processed this run
    assert result.successful == 1
