"""Weekly price repair: tracked universe, recent XNYS sessions, dry-run has no network/writes."""

from __future__ import annotations

from datetime import UTC, date, datetime

import duckdb
import polars as pl
import pytest

from app.db.schema import apply_schema, create_lake_views
from app.ingestion.price_freshness import provider_end_exclusive
from app.ingestion.price_repair import repair_prices, repair_window
from app.providers.base import RawFetchResult
from app.providers.price.yfinance_provider import YFinancePriceProvider
from app.services.market_calendar import MarketCalendarService
from app.services.tracked_universe_service import add_tracked
from app.utils.parquet_io import LakeDataset

CALENDAR = MarketCalendarService()
SATURDAY = datetime(2024, 1, 6, 15, 0, tzinfo=UTC)
FRIDAY = date(2024, 1, 5)


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    apply_schema(c)
    return c


def _insert_security(con, security_id: str, ticker: str) -> None:
    now = datetime.now(UTC)
    con.execute(
        """
        INSERT INTO securities
            (security_id, cik, company_name, primary_ticker, exchange, asset_type,
             currency, is_active, first_seen_at, last_seen_at, created_at, updated_at)
        VALUES (?, NULL, 'Co', ?, 'NYSE', 'EQUITY', 'USD', TRUE, ?, ?, ?, ?)
        """,
        [security_id, ticker, now, now, now, now],
    )


def _write_price(settings, security_id: str, d: date, close: float = 10.0) -> None:  # noqa: ANN001
    ds = LakeDataset(settings.prices_daily_dir, "date", ["security_id", "date"], ["security_id", "date"])
    df = pl.DataFrame(
        {
            "security_id": [security_id],
            "date": [d.isoformat()],
            "ticker_at_time": ["X"],
            "open": [close],
            "high": [close],
            "low": [close],
            "close": [close],
            "adj_close": [close],
            "volume": [1000.0],
            "dividend": [0.0],
            "stock_split": [0.0],
            "currency": ["USD"],
            "provider": ["test"],
            "retrieved_at": [datetime.now(UTC)],
        }
    ).with_columns(pl.col("date").str.to_date())
    ds.write_increment(df, run_id="seed")


def _bar(d: date, close: float = 10.5) -> dict:
    return {
        "date": d,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "adj_close": close,
        "volume": 1000.0,
        "dividend": 0.0,
        "stock_split": 0.0,
    }


def test_repair_fetches_all_tracked_even_if_current(con, settings, monkeypatch) -> None:
    for sid, t in [("S0", "AAA"), ("S1", "BBB")]:
        _insert_security(con, sid, t)
        add_tracked(con, [sid], reason="test")
        _write_price(settings, sid, FRIDAY)
    calls: list[str] = []

    def fake_fetch(self, provider_symbol: str, start: date, end: date | None = None) -> RawFetchResult:  # noqa: ARG001
        calls.append(provider_symbol)
        return RawFetchResult(rows=[_bar(FRIDAY)], provider="yfinance", retrieved_at=datetime.now(UTC))

    monkeypatch.setattr(YFinancePriceProvider, "fetch_daily_bars", fake_fetch)
    settings.price_request_pause_seconds = 0.0

    result = repair_prices(settings, con, dry_run=False, now=SATURDAY)
    assert result.tracked_targets == 2
    assert set(calls) == {"AAA", "BBB"}
    assert result.provider_fetch_count == 2


def test_repair_window_is_twenty_trading_sessions() -> None:
    start, end = repair_window(CALENDAR, 20, now=SATURDAY)
    assert end == FRIDAY
    assert start == CALENDAR.sessions_ago(FRIDAY, 20)
    sessions = CALENDAR.trading_days_between(start, end)
    assert sessions[0] == start
    assert sessions[-1] == end
    assert len(sessions) == 21


def test_repair_passes_exclusive_end_and_lookback_start(con, settings, monkeypatch) -> None:
    _insert_security(con, "S0", "AAA")
    add_tracked(con, ["S0"], reason="test")
    captured: list[tuple[date, date | None]] = []

    def fake_fetch(self, provider_symbol: str, start: date, end: date | None = None) -> RawFetchResult:  # noqa: ARG001
        captured.append((start, end))
        return RawFetchResult(rows=[_bar(FRIDAY)], provider="yfinance", retrieved_at=datetime.now(UTC))

    monkeypatch.setattr(YFinancePriceProvider, "fetch_daily_bars", fake_fetch)
    settings.price_request_pause_seconds = 0.0
    result = repair_prices(settings, con, lookback_sessions=20, dry_run=False, now=SATURDAY)
    assert result.start_session == CALENDAR.sessions_ago(FRIDAY, 20)
    assert result.end_session == FRIDAY
    assert captured == [(result.start_session, provider_end_exclusive(FRIDAY))]


def test_repair_dry_run_no_network_no_writes(con, settings, monkeypatch) -> None:
    _insert_security(con, "S0", "AAA")
    add_tracked(con, ["S0"], reason="test")
    _write_price(settings, "S0", FRIDAY)

    def boom(*_a, **_k):  # noqa: ANN002, ANN003
        raise AssertionError("dry-run must not construct a price provider")

    monkeypatch.setattr("app.ingestion.price_backfill.get_price_provider", boom)
    monkeypatch.setattr(YFinancePriceProvider, "fetch_daily_bars", boom)
    files_before = list(settings.prices_daily_dir.rglob("*.parquet"))
    ingest_before = con.execute("SELECT count(*) FROM ingest_runs").fetchone()[0]

    result = repair_prices(settings, con, dry_run=True, now=SATURDAY)

    assert result.dry_run is True
    assert result.tracked_targets == 1
    assert result.network_requests_planned == 1
    assert result.provider_fetch_count == 0
    assert result.rows_written == 0
    assert list(settings.prices_daily_dir.rglob("*.parquet")) == files_before
    assert con.execute("SELECT count(*) FROM ingest_runs").fetchone()[0] == ingest_before
    assert list(settings.checkpoints_dir.glob("*.json")) == []


def test_repair_logical_duplicates_are_zero(con, settings, monkeypatch) -> None:
    _insert_security(con, "S0", "AAA")
    add_tracked(con, ["S0"], reason="test")
    _write_price(settings, "S0", FRIDAY, close=10.0)
    create_lake_views(con, settings)
    before = con.execute("SELECT count(*) FROM prices_daily").fetchone()[0]

    def fake_fetch(self, provider_symbol: str, start: date, end: date | None = None) -> RawFetchResult:  # noqa: ARG001
        return RawFetchResult(rows=[_bar(FRIDAY, close=10.0)], provider="yfinance", retrieved_at=datetime.now(UTC))

    monkeypatch.setattr(YFinancePriceProvider, "fetch_daily_bars", fake_fetch)
    settings.price_request_pause_seconds = 0.0
    repair_prices(settings, con, dry_run=False, now=SATURDAY)
    create_lake_views(con, settings)
    after = con.execute("SELECT count(*) FROM prices_daily").fetchone()[0]
    assert after == before == 1
    assert con.execute("SELECT close FROM prices_daily").fetchone()[0] == 10.0
