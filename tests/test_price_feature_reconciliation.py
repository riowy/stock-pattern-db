"""Price vs feature identity-row reconciliation and label maturity."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import duckdb
import polars as pl
import pytest

from app.db.schema import apply_schema, create_lake_views
from app.ingestion.daily_pipeline import FEATURE_MISSING_ROW_BACKFILL_SESSIONS, _feature_daily_start
from app.research.dataset import (
    BUCKET_PRICE_NO_FEATURE_EXPECTED,
    BUCKET_PRICE_NO_FEATURE_UNEXPECTED,
    BUCKET_PRICE_WITH_FEATURE,
    label_maturity_dates,
    reconcile_price_feature_rows,
)
from app.services.tracked_universe_service import add_tracked
from app.utils.parquet_io import LakeDataset


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    apply_schema(c)
    return c


def _sec(con, sid="SEC0001", ticker="AAA") -> None:  # noqa: ANN001
    now = datetime.now(UTC)
    con.execute(
        """
        INSERT INTO securities
            (security_id, cik, company_name, primary_ticker, exchange, asset_type,
             currency, is_active, first_seen_at, last_seen_at, created_at, updated_at)
        VALUES (?, '0001', 'Co', ?, 'NYSE', 'EQUITY', 'USD', TRUE, ?, ?, ?, ?)
        """,
        [sid, ticker, now, now, now, now],
    )


def _price_rows(security_id: str, ticker: str, start: date, n: int) -> pl.DataFrame:
    dates: list[date] = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return pl.DataFrame(
        {
            "security_id": [security_id] * n,
            "ticker_at_time": [ticker] * n,
            "date": dates,
            "open": [10.0] * n,
            "high": [10.1] * n,
            "low": [9.9] * n,
            "close": [10.0] * n,
            "adj_close": [10.0] * n,
            "volume": [1000.0] * n,
            "retrieved_at": [datetime.now(UTC)] * n,
        }
    )


def test_expected_vs_unexpected_missing_feature_rows(con, settings) -> None:
    _sec(con, "FEAT1", "AAA")
    _sec(con, "PRICEONLY", "BBB")
    add_tracked(con, ["FEAT1"], reason="test", feature_tracking=True)
    add_tracked(con, ["PRICEONLY"], reason="test", price_tracking=True, feature_tracking=False)
    prices = LakeDataset(
        settings.lake_dir / "prices_daily",
        "date",
        ["security_id", "date"],
        ["security_id", "date"],
    )
    feat = LakeDataset(
        settings.lake_dir / "features_daily",
        "date",
        ["security_id", "date"],
        ["security_id", "date"],
    )
    p = pl.concat(
        [
            _price_rows("FEAT1", "AAA", date(2024, 1, 2), 5),
            _price_rows("PRICEONLY", "BBB", date(2024, 1, 2), 5),
        ]
    )
    prices.write_increment(p, "p")
    # Feature identity rows for only the first 3 AAA dates.
    aaa = _price_rows("FEAT1", "AAA", date(2024, 1, 2), 3).select(["security_id", "ticker_at_time", "date"])
    feat.write_increment(
        aaa.with_columns(
            pl.lit("v1").alias("feature_version"),
            pl.lit(datetime.now(UTC)).alias("calculated_at"),
        ),
        "f",
    )
    create_lake_views(con, settings)
    recon = reconcile_price_feature_rows(settings, con)
    assert recon.price_with_feature == 3
    assert recon.price_no_feature_expected == 5
    assert recon.price_no_feature_unexpected == 2
    assert recon.unexpected_examples
    text = (
        f"{BUCKET_PRICE_WITH_FEATURE}={recon.price_with_feature} "
        f"{BUCKET_PRICE_NO_FEATURE_EXPECTED}={recon.price_no_feature_expected} "
        f"{BUCKET_PRICE_NO_FEATURE_UNEXPECTED}={recon.price_no_feature_unexpected}"
    )
    assert "PRICE_NO_FEATURE_UNEXPECTED=2" in text


def test_feature_daily_start_includes_recent_missing_identity_rows(con, settings) -> None:
    _sec(con, "FEAT1", "AAA")
    add_tracked(con, ["FEAT1"], reason="test", feature_tracking=True)
    prices = LakeDataset(
        settings.lake_dir / "prices_daily", "date", ["security_id", "date"], ["security_id", "date"]
    )
    feat = LakeDataset(
        settings.lake_dir / "features_daily", "date", ["security_id", "date"], ["security_id", "date"]
    )
    start = date(2026, 9, 1)
    p = _price_rows("FEAT1", "AAA", start, 15)
    prices.write_increment(p, "p")
    latest = p["date"].max()
    # Features only for the last 3 sessions.
    last3 = p.sort("date").tail(3).select(["security_id", "ticker_at_time", "date"])
    feat.write_increment(
        last3.with_columns(
            pl.lit("v1").alias("feature_version"),
            pl.lit(datetime.now(UTC)).alias("calculated_at"),
        ),
        "f",
    )
    create_lake_views(con, settings)
    feat_start = _feature_daily_start(con, settings)
    assert feat_start is not None
    assert feat_start <= p.sort("date").tail(4)["date"].min()
    assert FEATURE_MISSING_ROW_BACKFILL_SESSIONS >= 10
    assert feat_start <= latest


def test_label_maturity_distinguishes_immature_nulls(con, settings) -> None:
    labels = LakeDataset(
        settings.lake_dir / "labels_forward_returns",
        "date",
        ["security_id", "date"],
        ["security_id", "date"],
    )
    dates = [date(2026, 9, 10), date(2026, 9, 11), date(2026, 9, 12), date(2026, 9, 15), date(2026, 9, 16)]
    n = len(dates)
    df = pl.DataFrame(
        {
            "security_id": ["S1"] * n,
            "ticker_at_time": ["AAA"] * n,
            "date": dates,
            "label_version": ["v1"] * n,
            "forward_return_1d": [0.01, 0.01, 0.01, 0.01, None],
            "forward_return_5d": [0.02, 0.02, None, None, None],
            "forward_return_10d": [0.03, None, None, None, None],
            "forward_return_20d": [None, None, None, None, None],
            "calculated_at": [datetime.now(UTC)] * n,
        }
    )
    labels.write_increment(df, "l")
    create_lake_views(con, settings)
    maturity = label_maturity_dates(settings, con)
    assert maturity["latest_label_row_date"] == "2026-09-16"
    assert maturity["latest_mature_1d"] == "2026-09-15"
    assert maturity["latest_mature_5d"] == "2026-09-11"
    assert maturity["latest_mature_10d"] == "2026-09-10"
    assert maturity["latest_mature_20d"] is None
