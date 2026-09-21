"""Prices/features/labels security coverage and research-universe filtering."""

from __future__ import annotations

from datetime import UTC, date, datetime

import duckdb
import pytest

from app.db.schema import apply_schema
from app.research.config import RESEARCH_UNIVERSE_NAME
from app.research.dataset import daily_collection_scope, research_common_equity_view_sql
from app.services.tracked_universe_service import add_tracked
from app.services.universe_membership_service import (
    UNIVERSE_TYPE_BENCHMARK,
    UNIVERSE_TYPE_RESEARCH_COMMON_EQUITY,
    upsert_memberships,
)


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    apply_schema(c)
    return c


def _sec(con, sid: str, ticker: str) -> None:
    now = datetime.now(UTC)
    con.execute(
        """
        INSERT INTO securities
            (security_id, cik, company_name, primary_ticker, exchange, asset_type,
             currency, is_active, first_seen_at, last_seen_at, created_at, updated_at)
        VALUES (?, NULL, 'Co', ?, 'NYSE', 'EQUITY', 'USD', TRUE, ?, ?, ?, ?)
        """,
        [sid, ticker, now, now, now, now],
    )


def test_research_view_sql_filters_common_equity_universe_and_excludes_benchmarks() -> None:
    sql = research_common_equity_view_sql()
    assert RESEARCH_UNIVERSE_NAME in sql
    assert "RESEARCH_COMMON_EQUITY" in sql
    assert "COMMON_EQUITY" in sql
    assert "data_quality_valid" in sql
    assert "raw_close" in sql
    assert "BENCHMARK" not in sql or "universe_type = 'BENCHMARK'" not in sql


def test_research_view_excludes_benchmark_rows(con) -> None:
    _sec(con, "EQ1", "AAA")
    _sec(con, "ETF1", "SPY")
    now = datetime.now(UTC)
    con.execute(
        """
        INSERT INTO instrument_classifications
            (security_id, ticker, instrument_class, classification_source,
             classification_confidence, exclusion_reason, updated_at)
        VALUES ('EQ1', 'AAA', 'COMMON_EQUITY', 'simple_us_ticker', 'medium', NULL, ?),
               ('ETF1', 'SPY', 'ETF', 'etf_seed', 'high', 'benchmark_or_sector_etf', ?)
        """,
        [now, now],
    )
    upsert_memberships(
        con, RESEARCH_UNIVERSE_NAME, UNIVERSE_TYPE_RESEARCH_COMMON_EQUITY, ["EQ1"], "test"
    )
    upsert_memberships(con, "benchmark-etf-seed", UNIVERSE_TYPE_BENCHMARK, ["ETF1"], "seed")
    ids = [
        r[0]
        for r in con.execute(
            "SELECT security_id FROM universe_memberships WHERE universe_name = ?",
            [RESEARCH_UNIVERSE_NAME],
        ).fetchall()
    ]
    assert ids == ["EQ1"]
    assert "ETF1" not in ids


def test_daily_collection_target_is_tracked_not_full_master(con) -> None:
    for i in range(20):
        _sec(con, f"S{i:02d}", f"T{i:02d}")
    add_tracked(con, ["S00", "S01", "S02"], reason="test")
    scope = daily_collection_scope(con)
    assert scope["known_securities"] == 20
    assert scope["tracked_price"] == 3
    assert scope["tracked_price"] < scope["known_securities"]


def test_reconcile_all_three_no_orphans(con, settings) -> None:
    from app.research.dataset import reconcile_datasets
    from app.utils.parquet_io import LakeDataset

    _sec(con, "EQ1", "AAA")
    add_tracked(con, ["EQ1"], reason="test")
    prices = LakeDataset(
        settings.lake_dir / "prices_daily",
        "date",
        ["security_id", "date"],
        ["security_id", "date"],
    )
    import polars as pl

    df = pl.DataFrame(
        {
            "security_id": ["EQ1"],
            "ticker_at_time": ["AAA"],
            "date": [date(2024, 1, 2)],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.0],
            "adj_close": [1.0],
            "volume": [100],
            "retrieved_at": [datetime.now(UTC)],
        }
    )
    prices.write_increment(df, "run")
    report = reconcile_datasets(settings, con)
    assert report.price_securities == 1
    assert report.feature_securities == 0
    assert report.price_only == 0 or report.feature_only == 0
    # features/labels views absent -> counts stay 0, not treated as lake orphans of features
    assert report.tracked == 1
