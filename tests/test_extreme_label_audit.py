from __future__ import annotations

from datetime import UTC, date, datetime

import duckdb
import polars as pl

from app.config.lake_datasets import get_lake_dataset
from app.db.schema import apply_schema
from app.services.extreme_label_audit_service import (
    CAUSE_EXTREME_MARKET,
    CAUSE_NON_COMMON_INSTRUMENT,
    audit_extreme_labels,
)
from app.services.instrument_classification_service import refresh_instrument_classifications
from app.services.universe_membership_service import (
    UNIVERSE_TYPE_RESEARCH_COMMON_EQUITY,
    upsert_memberships,
)


def _insert(con, sid: str, ticker: str) -> None:
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


def test_extreme_label_audit_aggregates_by_ticker_class_horizon(settings) -> None:
    con = duckdb.connect(":memory:")
    apply_schema(con)
    _insert(con, "S_AAPL", "AAPL")
    _insert(con, "S_WT", "GCTS-WT")
    refresh_instrument_classifications(con)
    upsert_memberships(
        con,
        "research-common-equity-100",
        UNIVERSE_TYPE_RESEARCH_COMMON_EQUITY,
        ["S_AAPL"],
        selection_rule="test",
    )

    now = datetime.now(UTC)
    labels = pl.DataFrame(
        {
            "security_id": ["S_AAPL", "S_WT"],
            "ticker_at_time": ["AAPL", "GCTS-WT"],
            "date": [date(2024, 1, 2), date(2024, 1, 3)],
            "label_version": ["v1", "v1"],
            "calculated_at": [now, now],
            "calculation_code_version": ["test", "test"],
            "forward_return_10d": [2.5, 8.0],
            "forward_return_1d": [0.1, 0.1],
        }
    )
    ds = get_lake_dataset(settings.lake_dir, "labels_forward_returns")
    ds.write_increment(labels, run_id="audit-test")

    prices = pl.DataFrame(
        {
            "security_id": ["S_AAPL", "S_AAPL", "S_WT", "S_WT"],
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 3)],
            "open": [10.0, 11.0, 1.0, 1.1],
            "high": [11.0, 12.0, 1.2, 1.3],
            "low": [9.0, 10.0, 0.9, 1.0],
            "close": [10.5, 11.5, 1.05, 1.2],
            "adj_close": [10.5, 11.5, 1.05, 1.2],
            "stock_split": [0.0, 0.0, 0.0, 0.0],
            "retrieved_at": [now] * 4,
        }
    )
    get_lake_dataset(settings.lake_dir, "prices_daily").write_increment(prices, run_id="audit-test")

    report = audit_extreme_labels(settings, con)
    assert report.total == 2
    tickers = dict(report.by_ticker)
    assert tickers["AAPL"] == 1
    assert tickers["GCTS-WT"] == 1
    assert report.by_horizon["forward_return_10d"] == 2
    assert report.by_instrument_class.get("COMMON_EQUITY", 0) == 1
    assert report.by_instrument_class.get("WARRANT", 0) == 1
    assert report.by_cause.get(CAUSE_NON_COMMON_INSTRUMENT, 0) == 1
    assert report.by_cause.get(CAUSE_EXTREME_MARKET, 0) == 1
    assert report.research_unexplained == 0
    assert report.largest_positive[0]["ticker"] == "GCTS-WT"
