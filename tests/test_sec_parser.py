from __future__ import annotations

from datetime import UTC, date, datetime

from app.normalization.security_master import (
    build_etf_seed_universe,
    normalize_sec_universe_rows,
    security_id_for_cik,
    security_id_for_ticker_fallback,
)

RETRIEVED_AT = datetime.now(UTC)
SNAPSHOT_DATE = date(2024, 1, 1)

# A small sample mimicking real rows from
# https://www.sec.gov/files/company_tickers_exchange.json
SAMPLE_SEC_ROWS = [
    {"cik": 320193, "name": "Apple Inc.", "ticker": "AAPL", "exchange": "Nasdaq"},
    {"cik": 1652044, "name": "Alphabet Inc.", "ticker": "GOOGL", "exchange": "Nasdaq"},
    {"cik": 1652044, "name": "Alphabet Inc.", "ticker": "GOOG", "exchange": "Nasdaq"},
    {"cik": 1067983, "name": "BERKSHIRE HATHAWAY INC", "ticker": "BRK-A", "exchange": "NYSE"},
    {"cik": 1067983, "name": "BERKSHIRE HATHAWAY INC", "ticker": "BRK-B", "exchange": "NYSE"},
]


def test_single_ticker_cik_gets_pure_cik_security_id() -> None:
    result = normalize_sec_universe_rows(SAMPLE_SEC_ROWS, RETRIEVED_AT, SNAPSHOT_DATE)
    aapl = next(s for s in result.securities if s.primary_ticker == "AAPL")
    assert aapl.security_id == security_id_for_cik(320193)
    assert aapl.security_id == "CIK0000320193"


def test_multi_ticker_cik_gets_disambiguated_security_ids() -> None:
    result = normalize_sec_universe_rows(SAMPLE_SEC_ROWS, RETRIEVED_AT, SNAPSHOT_DATE)
    googl = next(s for s in result.securities if s.primary_ticker == "GOOGL")
    goog = next(s for s in result.securities if s.primary_ticker == "GOOG")

    assert googl.security_id != goog.security_id
    assert googl.security_id.startswith(security_id_for_cik(1652044))
    assert goog.security_id.startswith(security_id_for_cik(1652044))
    assert googl.cik == goog.cik == f"{1652044:010d}"


def test_identifiers_include_both_cik_and_ticker() -> None:
    result = normalize_sec_universe_rows(SAMPLE_SEC_ROWS, RETRIEVED_AT, SNAPSHOT_DATE)
    aapl_id = next(s for s in result.securities if s.primary_ticker == "AAPL").security_id
    types = {i.identifier_type for i in result.identifiers if i.security_id == aapl_id}
    assert types == {"CIK", "TICKER"}


def test_rows_missing_cik_or_ticker_are_skipped() -> None:
    rows = [{"cik": None, "name": "x", "ticker": "ZZZ", "exchange": "Nasdaq"}, {"cik": 1, "name": "y", "ticker": None, "exchange": "Nasdaq"}]
    result = normalize_sec_universe_rows(rows, RETRIEVED_AT, SNAPSHOT_DATE)
    assert result.securities == []


def test_etf_seed_skips_tickers_already_known_from_sec() -> None:
    known = {"AAPL"}
    result = build_etf_seed_universe(["AAPL", "QQQ"], known, RETRIEVED_AT, SNAPSHOT_DATE)
    tickers = {s.primary_ticker for s in result.securities}
    assert tickers == {"QQQ"}
    qqq = result.securities[0]
    assert qqq.security_id == security_id_for_ticker_fallback("QQQ")
    assert qqq.asset_type == "ETF"
    assert qqq.cik is None
