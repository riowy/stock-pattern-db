"""Daily price fast path: per-security CURRENT/STALE/NO_DATA, not a global max(date) skip."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import duckdb
import polars as pl
import pytest

from app.config.settings import Settings
from app.db.schema import apply_schema, create_lake_views
from app.ingestion.daily_pipeline import run_daily_pipeline
from app.ingestion.price_freshness import (
    NO_DATA_DAILY_CALENDAR_LOOKBACK_DAYS,
    daily_fetch_window,
    provider_end_exclusive,
    stale_fetch_start,
)
from app.ingestion.price_sync import sync_recent_prices
from app.providers.base import RawFetchResult
from app.providers.price.yfinance_provider import YFinancePriceProvider
from app.services.market_calendar import MarketCalendarService
from app.services.tracked_universe_service import add_tracked
from app.utils.parquet_io import LakeDataset

CALENDAR = MarketCalendarService()
SATURDAY = datetime(2024, 1, 6, 15, 0, tzinfo=UTC)
FRIDAY = date(2024, 1, 5)
MONDAY_BEFORE_CLOSE = datetime(2024, 1, 8, 20, 0, tzinfo=UTC)
MONDAY_AFTER_GRACE = datetime(2024, 1, 8, 23, 30, tzinfo=UTC)
MONDAY = date(2024, 1, 8)


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


def _write_prices(settings: Settings, rows: list[tuple[str, date, float]]) -> None:
    ds = LakeDataset(settings.prices_daily_dir, "date", ["security_id", "date"], ["security_id", "date"])
    now = datetime.now(UTC)
    df = pl.DataFrame(
        {
            "security_id": [r[0] for r in rows],
            "date": [r[1].isoformat() for r in rows],
            "ticker_at_time": ["X"] * len(rows),
            "open": [r[2] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[2] for r in rows],
            "adj_close": [r[2] for r in rows],
            "volume": [1000.0] * len(rows),
            "dividend": [0.0] * len(rows),
            "stock_split": [0.0] * len(rows),
            "currency": ["USD"] * len(rows),
            "provider": ["test"] * len(rows),
            "retrieved_at": [now] * len(rows),
        }
    ).with_columns(pl.col("date").str.to_date())
    ds.write_increment(df, run_id="seed")


def _write_feature_and_label_identity(settings: Settings, security_id: str, ticker: str, d: date) -> None:
    from app.features.schema import FEATURE_VALUE_COLUMNS
    from app.labels.schema import LABEL_VALUE_COLUMNS

    now = datetime.now(UTC)
    feat = {
        "security_id": [security_id],
        "ticker_at_time": [ticker],
        "date": [d.isoformat()],
        "feature_version": ["v1"],
        "calculated_at": [now],
        "calculation_code_version": ["test"],
    }
    feat.update({c: [0.0] for c in FEATURE_VALUE_COLUMNS})
    LakeDataset(
        settings.lake_dir / "features_daily", "date", ["security_id", "date"], ["security_id", "date"]
    ).write_increment(pl.DataFrame(feat).with_columns(pl.col("date").str.to_date()), run_id="seed")
    labels = {
        "security_id": [security_id],
        "ticker_at_time": [ticker],
        "date": [d.isoformat()],
        "label_version": ["v1"],
        "calculated_at": [now],
        "calculation_code_version": ["test"],
    }
    labels.update({c: [0.0] for c in LABEL_VALUE_COLUMNS})
    LakeDataset(
        settings.lake_dir / "labels_forward_returns", "date", ["security_id", "date"], ["security_id", "date"]
    ).write_increment(pl.DataFrame(labels).with_columns(pl.col("date").str.to_date()), run_id="seed")


def _bar(d: date) -> dict:
    return {
        "date": d,
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "adj_close": 10.5,
        "volume": 1000.0,
        "dividend": 0.0,
        "stock_split": 0.0,
    }


def _boom_provider(*_a, **_k):  # noqa: ANN002, ANN003
    raise AssertionError("price provider must not be constructed")


def test_settings_price_lookback_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.daily_price_lookback_sessions == 3
    assert s.price_repair_lookback_sessions == 20


def test_provider_end_exclusive_keeps_friday() -> None:
    friday = date(2026, 9, 18)
    assert provider_end_exclusive(friday) == date(2026, 9, 19)
    assert provider_end_exclusive(friday) != friday


def test_all_tracked_current_skips_provider(con, settings, monkeypatch) -> None:
    tickers = [f"T{i:02d}" for i in range(12)]
    for i, t in enumerate(tickers):
        sid = f"S{i:02d}"
        _insert_security(con, sid, t)
        add_tracked(con, [sid], reason="test")
    _write_prices(settings, [(f"S{i:02d}", FRIDAY, 100.0) for i in range(12)])
    monkeypatch.setattr("app.ingestion.price_backfill.get_price_provider", _boom_provider)
    monkeypatch.setattr(YFinancePriceProvider, "fetch_daily_bars", _boom_provider)

    result = sync_recent_prices(settings, con, dry_run=False, now=SATURDAY)

    assert result.tracked_targets == 12
    assert result.current == 12
    assert result.stale == 0
    assert result.no_data == 0
    assert result.fetch_targets == 0
    assert result.provider_fetch_skipped is True
    assert result.provider_fetch_count == 0
    assert result.expected_latest == FRIDAY
    assert "already current through 2024-01-05" in result.message


def test_one_stale_fetches_only_that_symbol(con, settings, monkeypatch) -> None:
    names = [("S0", "AAA"), ("S1", "BBB"), ("S2", "CCC")]
    for sid, t in names:
        _insert_security(con, sid, t)
        add_tracked(con, [sid], reason="test")
    _write_prices(
        settings,
        [("S0", FRIDAY, 1.0), ("S1", FRIDAY, 2.0), ("S2", date(2024, 1, 2), 3.0)],
    )
    calls: list[str] = []

    def fake_fetch(self, provider_symbol: str, start: date, end: date | None = None) -> RawFetchResult:  # noqa: ARG001
        calls.append(provider_symbol)
        return RawFetchResult(rows=[_bar(FRIDAY)], provider="yfinance", retrieved_at=datetime.now(UTC))

    monkeypatch.setattr(YFinancePriceProvider, "fetch_daily_bars", fake_fetch)
    settings.price_request_pause_seconds = 0.0

    result = sync_recent_prices(settings, con, dry_run=False, now=SATURDAY)

    assert result.current == 2
    assert result.stale == 1
    assert result.no_data == 0
    assert result.fetch_targets == 1
    assert calls == ["CCC"]
    assert result.provider_fetch_count == 1


def test_mixed_current_stale_no_data_fetches_only_gaps(con, settings, monkeypatch) -> None:
    for sid, t in [("S0", "CUR"), ("S1", "STL"), ("S2", "NEW")]:
        _insert_security(con, sid, t)
        add_tracked(con, [sid], reason="test")
    _write_prices(settings, [("S0", FRIDAY, 1.0), ("S1", date(2024, 1, 3), 2.0)])
    calls: list[str] = []

    def fake_fetch(self, provider_symbol: str, start: date, end: date | None = None) -> RawFetchResult:  # noqa: ARG001
        calls.append(provider_symbol)
        return RawFetchResult(rows=[_bar(FRIDAY)], provider="yfinance", retrieved_at=datetime.now(UTC))

    monkeypatch.setattr(YFinancePriceProvider, "fetch_daily_bars", fake_fetch)
    settings.price_request_pause_seconds = 0.0

    result = sync_recent_prices(settings, con, dry_run=False, now=SATURDAY)

    assert result.current == 1
    assert result.stale == 1
    assert result.no_data == 1
    assert result.fetch_targets == 2
    assert set(calls) == {"STL", "NEW"}


def test_daily_overlap_is_three_trading_sessions(con, settings, monkeypatch) -> None:
    _insert_security(con, "S0", "AAA")
    add_tracked(con, ["S0"], reason="test")
    _write_prices(settings, [("S0", FRIDAY, 1.0)])
    captured: list[tuple[date, date | None]] = []

    def fake_fetch(self, provider_symbol: str, start: date, end: date | None = None) -> RawFetchResult:  # noqa: ARG001
        captured.append((start, end))
        return RawFetchResult(rows=[_bar(MONDAY)], provider="yfinance", retrieved_at=datetime.now(UTC))

    monkeypatch.setattr(YFinancePriceProvider, "fetch_daily_bars", fake_fetch)
    settings.price_request_pause_seconds = 0.0
    settings.daily_price_lookback_sessions = 3

    result = sync_recent_prices(settings, con, dry_run=False, now=MONDAY_AFTER_GRACE)

    assert result.stale == 1
    assert result.requested_session_end == MONDAY
    expected_start = stale_fetch_start(CALENDAR, FRIDAY, 3)
    assert expected_start == date(2024, 1, 2)
    assert result.requested_session_start == expected_start
    assert captured == [(expected_start, provider_end_exclusive(MONDAY))]
    assert provider_end_exclusive(MONDAY) == date(2024, 1, 9)


def test_weekend_expected_latest_is_prior_friday(con, settings, monkeypatch) -> None:
    _insert_security(con, "S0", "AAA")
    add_tracked(con, ["S0"], reason="test")
    _write_prices(settings, [("S0", FRIDAY, 1.0)])
    monkeypatch.setattr("app.ingestion.price_backfill.get_price_provider", _boom_provider)

    result = sync_recent_prices(settings, con, dry_run=False, now=SATURDAY)

    assert result.expected_latest == FRIDAY
    assert result.fetch_targets == 0
    assert result.provider_fetch_count == 0


def test_before_close_does_not_fetch_incomplete_session(con, settings, monkeypatch) -> None:
    _insert_security(con, "S0", "AAA")
    add_tracked(con, ["S0"], reason="test")
    thursday = date(2024, 1, 4)
    _write_prices(settings, [("S0", thursday, 1.0)])
    captured_end: list[date | None] = []

    def fake_fetch(self, provider_symbol: str, start: date, end: date | None = None) -> RawFetchResult:  # noqa: ARG001
        captured_end.append(end)
        return RawFetchResult(rows=[_bar(FRIDAY)], provider="yfinance", retrieved_at=datetime.now(UTC))

    monkeypatch.setattr(YFinancePriceProvider, "fetch_daily_bars", fake_fetch)
    settings.price_request_pause_seconds = 0.0

    result = sync_recent_prices(settings, con, dry_run=False, now=MONDAY_BEFORE_CLOSE)

    assert result.expected_latest == FRIDAY
    assert result.requested_session_end == FRIDAY
    assert MONDAY not in {result.expected_latest, result.requested_session_end}
    assert captured_end == [provider_end_exclusive(FRIDAY)]


def test_same_session_second_run_fetches_zero(con, settings, monkeypatch) -> None:
    _insert_security(con, "S0", "AAA")
    add_tracked(con, ["S0"], reason="test")
    _write_prices(settings, [("S0", FRIDAY, 1.0)])
    calls = {"n": 0}

    def fake_fetch(self, provider_symbol: str, start: date, end: date | None = None) -> RawFetchResult:  # noqa: ARG001
        calls["n"] += 1
        return RawFetchResult(rows=[_bar(FRIDAY)], provider="yfinance", retrieved_at=datetime.now(UTC))

    monkeypatch.setattr(YFinancePriceProvider, "fetch_daily_bars", fake_fetch)

    first = sync_recent_prices(settings, con, dry_run=False, now=SATURDAY)
    second = sync_recent_prices(settings, con, dry_run=False, now=SATURDAY)

    assert first.provider_fetch_count == 0
    assert second.provider_fetch_count == 0
    assert calls["n"] == 0


def test_no_data_uses_prior_daily_calendar_bootstrap_not_three_sessions(con, settings) -> None:
    _insert_security(con, "S0", "NEW")
    add_tracked(con, ["S0"], reason="test")
    from app.ingestion.price_freshness import classify_price_freshness

    freshness = classify_price_freshness(settings, con, now=SATURDAY, calendar=CALENDAR)
    window = daily_fetch_window(freshness, CALENDAR, lookback_sessions=3)
    assert window is not None
    start, end = window
    assert end == FRIDAY
    assert start == FRIDAY - timedelta(days=NO_DATA_DAILY_CALENDAR_LOOKBACK_DAYS)
    assert start != stale_fetch_start(CALENDAR, FRIDAY, 3)


def test_existing_price_rows_not_dropped_when_other_name_is_stale(con, settings, monkeypatch) -> None:
    _insert_security(con, "S0", "AAA")
    _insert_security(con, "S1", "BBB")
    add_tracked(con, ["S0", "S1"], reason="test")
    older = date(2024, 1, 3)
    _write_prices(settings, [("S0", FRIDAY, 10.0), ("S0", older, 9.0), ("S1", older, 8.0)])
    create_lake_views(con, settings)
    before = con.execute("SELECT security_id, date, close FROM prices_daily ORDER BY 1, 2").fetchall()

    def fake_fetch(self, provider_symbol: str, start: date, end: date | None = None) -> RawFetchResult:  # noqa: ARG001
        assert provider_symbol == "BBB"
        return RawFetchResult(rows=[_bar(FRIDAY)], provider="yfinance", retrieved_at=datetime.now(UTC))

    monkeypatch.setattr(YFinancePriceProvider, "fetch_daily_bars", fake_fetch)
    settings.price_request_pause_seconds = 0.0
    sync_recent_prices(settings, con, dry_run=False, now=SATURDAY)
    create_lake_views(con, settings)
    after = con.execute("SELECT security_id, date, close FROM prices_daily ORDER BY 1, 2").fetchall()
    for row in before:
        assert row in after


def test_single_symbol_failure_does_not_abort_job(con, settings, monkeypatch) -> None:
    _insert_security(con, "S0", "AAA")
    _insert_security(con, "S1", "GCTS-WT")
    add_tracked(con, ["S0", "S1"], reason="test")

    def fake_fetch(self, provider_symbol: str, start: date, end: date | None = None) -> RawFetchResult:  # noqa: ARG001
        if provider_symbol == "GCTS-WT":
            raise RuntimeError("yfinance missing/error")
        return RawFetchResult(rows=[_bar(FRIDAY)], provider="yfinance", retrieved_at=datetime.now(UTC))

    monkeypatch.setattr(YFinancePriceProvider, "fetch_daily_bars", fake_fetch)
    monkeypatch.setattr(
        "app.services.price_audit_service.audit_prices",
        lambda *_a, **_k: SimpleNamespace(research_critical=0),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.sync_universe",
        lambda *_a, **_k: SimpleNamespace(dry_run=False, securities_seen=2),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.sync_vix",
        lambda *_a, **_k: SimpleNamespace(dry_run=False, rows_written=0),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.sync_macro",
        lambda *_a, **_k: SimpleNamespace(
            status="skipped", skip_reason="test", dry_run=False, series_synced=0, rows_written=0
        ),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.sync_filings",
        lambda *_a, **_k: SimpleNamespace(
            status="skipped", skip_reason="test", dry_run=False, successful=0, rows_written=0
        ),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.validate_prices",
        lambda *_a, **_k: SimpleNamespace(scope_size=2, issues_found=0, critical=0),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.validate_features",
        lambda *_a, **_k: SimpleNamespace(issues_found=0, critical=0),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.validate_labels",
        lambda *_a, **_k: SimpleNamespace(issues_found=0, critical=0),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.compute_features",
        lambda *_a, **_k: SimpleNamespace(dry_run=False, successful=0, total_symbols=0, rows_written=0),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.compute_labels",
        lambda *_a, **_k: SimpleNamespace(dry_run=False, successful=0, total_symbols=0, rows_written=0),
    )
    settings.price_request_pause_seconds = 0.0

    pipeline = run_daily_pipeline(settings, con, dry_run=False, now=SATURDAY)
    assert pipeline.aborted is False
    assert pipeline.price_provider_fetch_count == 2
    vix = next(s for s in pipeline.steps if s.name == "vix")
    assert vix.failed is False


def test_features_and_labels_skip_when_prices_already_current(con, settings, monkeypatch) -> None:
    _insert_security(con, "S0", "AAA")
    add_tracked(con, ["S0"], reason="test", feature_tracking=True)
    _write_prices(settings, [("S0", FRIDAY, 1.0)])
    _write_feature_and_label_identity(settings, "S0", "AAA", FRIDAY)
    create_lake_views(con, settings)

    def boom(*_a, **_k):  # noqa: ANN002, ANN003
        raise AssertionError("must not recompute features/labels")

    monkeypatch.setattr("app.ingestion.price_backfill.get_price_provider", _boom_provider)
    monkeypatch.setattr(
        "app.services.price_audit_service.audit_prices",
        lambda *_a, **_k: SimpleNamespace(research_critical=0),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.sync_universe",
        lambda *_a, **_k: SimpleNamespace(dry_run=False, securities_seen=1),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.sync_vix",
        lambda *_a, **_k: SimpleNamespace(dry_run=False, rows_written=1),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.sync_macro",
        lambda *_a, **_k: SimpleNamespace(
            status="skipped", skip_reason="FRED optional", dry_run=False, series_synced=0, rows_written=0
        ),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.sync_filings",
        lambda *_a, **_k: SimpleNamespace(
            status="planned", skip_reason=None, dry_run=False, successful=1, rows_written=0
        ),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.validate_prices",
        lambda *_a, **_k: SimpleNamespace(scope_size=1, issues_found=0, critical=0),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.validate_features",
        lambda *_a, **_k: SimpleNamespace(issues_found=0, critical=0),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.validate_labels",
        lambda *_a, **_k: SimpleNamespace(issues_found=0, critical=0),
    )
    monkeypatch.setattr("app.ingestion.daily_pipeline.compute_features", boom)
    monkeypatch.setattr("app.ingestion.daily_pipeline.compute_labels", boom)

    pipeline = run_daily_pipeline(settings, con, dry_run=False, now=SATURDAY)
    assert next(s.result for s in pipeline.steps if s.name == "prices").startswith("SKIPPED")
    assert next(s.result for s in pipeline.steps if s.name == "features") == "SKIPPED - no new price sessions"
    assert next(s.result for s in pipeline.steps if s.name == "labels") == "SKIPPED - no new price sessions"
    assert next(s for s in pipeline.steps if s.name == "vix").failed is False
    assert pipeline.price_provider_fetch_count == 0
    assert pipeline.price_symbols_requested == 0
    assert pipeline.price_symbols_skipped_current == 1
