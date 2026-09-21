from __future__ import annotations

from datetime import UTC, datetime

import duckdb
import pytest

from app.db.schema import apply_schema
from app.ingestion.research_universe import (
    build_research_common_equity_universe,
    build_research_scale_universe,
    eligible_common_equity_securities,
    eligible_research_securities,
)
from app.services.instrument_classification_service import refresh_instrument_classifications


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


def _add_security(con, sid: str, ticker: str, exchange: str = "NYSE", company_name: str = "Co") -> None:
    now = datetime.now(UTC)
    con.execute(
        """
        INSERT INTO securities
            (security_id, cik, company_name, primary_ticker, exchange, asset_type,
             currency, is_active, first_seen_at, last_seen_at, created_at, updated_at)
        VALUES (?, NULL, ?, ?, ?, 'EQUITY', 'USD', TRUE, ?, ?, ?, ?)
        """,
        [sid, company_name, ticker, exchange, now, now, now, now],
    )


def test_research_common_equity_excludes_non_common_and_keeps_benchmarks_out(con) -> None:
    _add_security(con, "WT1", "GCTS-WT")
    _add_security(con, "PF1", "ATH-PA")
    _add_security(con, "UN1", "FOO-U")
    _add_security(con, "RT1", "OCAC-RI")
    _add_security(
        con,
        "CEV1",
        "CEV",
        company_name="Eaton Vance California Municipal Income Trust",
    )
    _add_security(con, "KMPB1", "KMPB", company_name="Kemper Corporation")
    refresh_instrument_classifications(con)
    eligible = {t for t, _ in eligible_common_equity_securities(con)}
    assert "GCTS-WT" not in eligible
    assert "ATH-PA" not in eligible
    assert "FOO-U" not in eligible
    assert "OCAC-RI" not in eligible
    assert "SPY" not in eligible
    assert "QQQ" not in eligible
    assert "CEV" not in eligible
    assert "KMPB" not in eligible
    assert "AAA" in eligible


def test_research_common_equity_is_deterministic_and_prefix_stable(con) -> None:
    for i in range(20):
        _add_security(con, f"COM{i:02d}", f"CMA{chr(65 + i)}")
    a = build_research_common_equity_universe(con, 5)
    b = build_research_common_equity_universe(con, 5)
    assert a.tickers == b.tickers
    assert a.size == 5
    bigger = build_research_common_equity_universe(con, 10)
    assert bigger.tickers[:5] == a.tickers
    assert "SPY" not in a.tickers
    assert all(t not in {"GCTS-WT", "ATH-PA"} for t in bigger.tickers)
