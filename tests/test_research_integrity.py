from __future__ import annotations

import duckdb
import pytest

from app.db.schema import apply_schema
from app.services.dataset_metadata_service import (
    FEATURES_DAILY_DATASET,
    LABELS_FORWARD_RETURNS_DATASET,
    PRICES_DAILY_DATASET,
    get_metadata,
    seed_research_integrity_metadata,
    set_metadata,
)
from app.services.status_service import gather_status


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    apply_schema(c)
    return c


def test_seed_research_integrity_metadata_sets_expected_flags(con, settings) -> None:
    seed_research_integrity_metadata(con, settings)
    meta = get_metadata(con, PRICES_DAILY_DATASET)

    assert meta["historical_universe_complete"] == "false"
    assert meta["survivorship_safe"] == "false"
    assert meta["point_in_time_security_master"] == "false"
    assert meta["price_provider"] == "yfinance"
    # yfinance.capabilities.commercial_use_safe is False -> research_only is True
    assert meta["research_only_price_provider"] == "true"


def test_seed_is_idempotent_and_reflects_current_settings(con, settings) -> None:
    seed_research_integrity_metadata(con, settings)
    seed_research_integrity_metadata(con, settings)
    rows = con.execute(
        "SELECT count(*) FROM dataset_metadata WHERE dataset_name = ?", [PRICES_DAILY_DATASET]
    ).fetchone()[0]
    assert rows == 7  # no duplicate rows from re-seeding


def test_set_metadata_upserts(con) -> None:
    set_metadata(con, "prices_daily", "k", "v1")
    set_metadata(con, "prices_daily", "k", "v2")
    meta = get_metadata(con, "prices_daily")
    assert meta["k"] == "v2"
    count = con.execute("SELECT count(*) FROM dataset_metadata WHERE metadata_key = 'k'").fetchone()[0]
    assert count == 1


def test_seed_feature_and_label_integrity_flags(con, settings) -> None:
    seed_research_integrity_metadata(con, settings)
    feat = get_metadata(con, FEATURES_DAILY_DATASET)
    lab = get_metadata(con, LABELS_FORWARD_RETURNS_DATASET)
    assert feat["signal_timing"] == "market_close"
    assert feat["price_adjustment_point_in_time"] == "false"
    assert feat["macro_point_in_time"] == "false"
    assert feat["survivorship_safe"] == "false"
    assert feat["point_in_time_security_master"] == "false"
    assert lab["return_basis"] == "adjusted_close"
    assert lab["benchmark"] == "SPY"
    assert lab["historical_universe_complete"] == "false"


def test_status_report_exposes_research_integrity(con, settings) -> None:
    seed_research_integrity_metadata(con, settings)
    report = gather_status(settings, con)

    assert report.research_integrity.historical_universe_complete is False
    assert report.research_integrity.survivorship_safe is False
    assert report.research_integrity.point_in_time_security_master is False
    assert report.research_integrity.price_provider == "yfinance"
    assert report.research_integrity.commercial_use_safe is False
