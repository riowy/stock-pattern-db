from __future__ import annotations

from datetime import UTC, datetime

import duckdb
import pytest

from app.db.schema import apply_schema
from app.ingestion.research_universe import build_research_scale_universe, eligible_research_securities


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    apply_schema(c)
    now = datetime.now(UTC)
    # Mix of eligible / OTC / invalid
    rows = [
        ("NYSEAAA", "AAA", "NYSE"),
        ("NASBBB", "BBB", "Nasdaq"),
        ("OTCZZZ", "ZZZ", "OTC"),
        ("NYSECCC", "CCC", "NYSE"),
        ("BAD SP", "BAD SP", "NYSE"),
        ("TCKSPY", "SPY", "NYSE"),
        ("TCKQQQ", "QQQ", None),
    ]
    for sid, ticker, exch in rows:
        c.execute(
            """
            INSERT INTO securities
                (security_id, cik, company_name, primary_ticker, exchange, asset_type,
                 currency, is_active, first_seen_at, last_seen_at, created_at, updated_at)
            VALUES (?, NULL, 'Co', ?, ?, 'EQUITY', 'USD', TRUE, ?, ?, ?, ?)
            """,
            [sid, ticker, exch, now, now, now, now],
        )
    return c


def test_eligible_excludes_otc_and_invalid_tickers(con) -> None:
    eligible = {t for t, _ in eligible_research_securities(con)}
    assert "ZZZ" not in eligible
    assert "BAD SP" not in eligible
    assert "AAA" in eligible
    assert "QQQ" in eligible  # ETF seed, even with null exchange


def test_research_scale_is_deterministic(con) -> None:
    a = build_research_scale_universe(con, 3)
    b = build_research_scale_universe(con, 3)
    assert a.tickers == b.tickers
    assert a.size == 3
    assert "SPY" in a.tickers
    assert "QQQ" in a.tickers
