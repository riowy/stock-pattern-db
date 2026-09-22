"""True dry-run guarantees (see project task section 6): --dry-run must never
make a network call, write Parquet, mutate DuckDB tables, or touch checkpoints.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import duckdb
import pytest

from app.db.schema import apply_schema
from app.ingestion.filings_sync import sync_filings
from app.ingestion.macro_sync import sync_macro
from app.ingestion.price_backfill import run_price_ingestion
from app.ingestion.price_sync import sync_recent_prices
from app.ingestion.universe_sync import sync_universe
from app.ingestion.vix_sync import sync_vix
from app.providers.filings.sec_filings_provider import SecFilingsProvider
from app.providers.macro.fred_provider import FredMacroProvider
from app.providers.price.yfinance_provider import YFinancePriceProvider
from app.providers.security_master.sec_provider import SecUniverseProvider
from app.providers.volatility.cboe_vix_provider import CboeVixProvider
from app.services.tracked_universe_service import add_tracked
from app.utils.parquet_io import LakeDataset


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    apply_schema(c)
    return c


def _fail_if_called(*_args, **_kwargs):  # noqa: ANN002, ANN003
    raise AssertionError("dry-run must never make a network call")


def _insert_security(con, security_id="SEC0001", ticker="AAPL", cik="0000000001") -> None:  # noqa: ANN001
    now = datetime.now(UTC)
    con.execute(
        """
        INSERT INTO securities
            (security_id, cik, company_name, primary_ticker, exchange, asset_type,
             currency, is_active, first_seen_at, last_seen_at, created_at, updated_at)
        VALUES (?, ?, 'Test Co', ?, 'NYSE', 'EQUITY', 'USD', TRUE, ?, ?, ?, ?)
        """,
        [security_id, cik, ticker, now, now, now, now],
    )


def _assert_no_mutations(con, settings, ingest_runs_before: int = 0) -> None:  # noqa: ANN001
    assert con.execute("SELECT count(*) FROM ingest_runs").fetchone()[0] == ingest_runs_before
    assert list(settings.checkpoints_dir.glob("*.json")) == []


# --- E/F/G: sync-universe ------------------------------------------------------
def test_dry_run_universe_sync_makes_no_network_call_and_no_db_write(con, settings, monkeypatch) -> None:
    monkeypatch.setattr(SecUniverseProvider, "fetch_universe", _fail_if_called)

    result = sync_universe(settings, con, dry_run=True)

    assert result.dry_run is True
    _assert_no_mutations(con, settings)
    assert not any(settings.lake_dir.rglob("*.parquet"))


# --- E/F/G: backfill-prices / sync-prices --------------------------------------
def test_dry_run_backfill_prices_makes_no_network_call_and_no_db_write(con, settings, monkeypatch) -> None:
    _insert_security(con)
    monkeypatch.setattr(YFinancePriceProvider, "fetch_daily_bars", _fail_if_called)

    result = run_price_ingestion(
        settings, con, symbols=["AAPL"], start=date(2024, 1, 1), end=None,
        batch_size=50, resume=False, dry_run=True,
    )

    assert result.dry_run is True
    assert result.total_symbols == 1
    _assert_no_mutations(con, settings)
    assert not any(settings.prices_daily_dir.rglob("*.parquet"))


def test_dry_run_backfill_prices_does_not_loop_over_full_universe(con, settings, monkeypatch) -> None:
    """Regression test for the bug where dry-run still resolved+looped over
    every active security (e.g. 10,438) before skipping each one."""
    for i in range(50):
        _insert_security(con, security_id=f"SEC{i:04d}", ticker=f"TCK{i:04d}", cik=f"{i:010d}")
    monkeypatch.setattr(YFinancePriceProvider, "fetch_daily_bars", _fail_if_called)

    result = run_price_ingestion(
        settings, con, symbols=None, start=date(2024, 1, 1), end=None,
        batch_size=50, resume=False, dry_run=True, default_scope="all_active",
    )
    assert result.dry_run is True
    assert result.total_symbols == 50
    _assert_no_mutations(con, settings)


def test_dry_run_sync_prices_uses_tracked_scope_and_no_network(con, settings, monkeypatch) -> None:
    _insert_security(con, security_id="SEC0001", ticker="AAPL")
    _insert_security(con, security_id="SEC0002", ticker="MSFT", cik="0000000002")
    add_tracked(con, ["SEC0001"], reason="test")
    monkeypatch.setattr(YFinancePriceProvider, "fetch_daily_bars", _fail_if_called)

    result = sync_recent_prices(settings, con, symbols=None, dry_run=True)

    assert result.dry_run is True
    assert result.total_symbols == 1  # only the tracked security, not both
    _assert_no_mutations(con, settings)


# --- E/F/G: sync-vix ------------------------------------------------------------
def test_dry_run_vix_sync_makes_no_network_call_and_no_db_write(con, settings, monkeypatch) -> None:
    monkeypatch.setattr(CboeVixProvider, "fetch_history", _fail_if_called)

    result = sync_vix(settings, con, dry_run=True)

    assert result.dry_run is True
    _assert_no_mutations(con, settings)
    assert not any(settings.volatility_dir.rglob("*.parquet"))


# --- E/F/G: sync-macro -----------------------------------------------------------
def test_dry_run_macro_sync_makes_no_network_call_and_no_db_write(con, settings, monkeypatch) -> None:
    monkeypatch.setattr(FredMacroProvider, "fetch_series", _fail_if_called)

    result = sync_macro(settings, con, dry_run=True)

    assert result.dry_run is True
    assert result.status == "planned"
    _assert_no_mutations(con, settings)
    assert not any(settings.macro_dir.rglob("*.parquet"))


def test_missing_fred_key_is_skipped_not_failed_even_without_dry_run(con, settings, monkeypatch) -> None:
    settings.fred_api_key = ""
    monkeypatch.setattr(FredMacroProvider, "fetch_series", _fail_if_called)

    result = sync_macro(settings, con, dry_run=False)

    assert result.status == "skipped"
    assert "FRED_API_KEY" in result.skip_reason
    _assert_no_mutations(con, settings)


# --- E/F/G: sync-sec-filings -----------------------------------------------------
def test_dry_run_filings_sync_makes_no_network_call_and_no_db_write(con, settings, monkeypatch) -> None:
    _insert_security(con, security_id="SEC0001", ticker="AAPL", cik="0000000001")
    add_tracked(con, ["SEC0001"], reason="test", filings_tracking=True)
    monkeypatch.setattr(SecFilingsProvider, "fetch_filings", _fail_if_called)

    result = sync_filings(settings, con, ciks=None, dry_run=True)

    assert result.dry_run is True
    assert result.status == "planned"
    _assert_no_mutations(con, settings)
    assert not any(settings.filings_dir.rglob("*.parquet"))


def test_filings_sync_skipped_when_nothing_tracked(con, settings, monkeypatch) -> None:
    monkeypatch.setattr(SecFilingsProvider, "fetch_filings", _fail_if_called)
    result = sync_filings(settings, con, ciks=None, dry_run=False)
    assert result.status == "skipped"
    _assert_no_mutations(con, settings)


# --- Parquet file count truly unchanged before/after (F) ------------------------
def test_dry_run_leaves_lake_file_count_unchanged(con, settings, monkeypatch) -> None:
    _insert_security(con)
    monkeypatch.setattr(YFinancePriceProvider, "fetch_daily_bars", _fail_if_called)
    monkeypatch.setattr(CboeVixProvider, "fetch_history", _fail_if_called)
    monkeypatch.setattr(SecUniverseProvider, "fetch_universe", _fail_if_called)

    ds = LakeDataset(settings.prices_daily_dir, "date", ["security_id", "date"], ["security_id", "date"])
    files_before = len(ds.all_files())

    sync_universe(settings, con, dry_run=True)
    run_price_ingestion(
        settings, con, symbols=["AAPL"], start=date(2024, 1, 1), end=None,
        batch_size=50, resume=False, dry_run=True,
    )
    sync_vix(settings, con, dry_run=True)

    files_after = len(ds.all_files())
    assert files_before == files_after == 0


def test_dry_run_daily_pipeline_includes_feature_label_steps(con, settings, monkeypatch) -> None:
    from app.ingestion.daily_pipeline import run_daily_pipeline
    from app.providers.security_master.sec_provider import SecUniverseProvider
    from app.providers.volatility.cboe_vix_provider import CboeVixProvider
    from app.providers.price.yfinance_provider import YFinancePriceProvider

    monkeypatch.setattr(SecUniverseProvider, "fetch_universe", _fail_if_called)
    monkeypatch.setattr(CboeVixProvider, "fetch_history", _fail_if_called)
    monkeypatch.setattr(YFinancePriceProvider, "fetch_daily_bars", _fail_if_called)

    # Soak default: derived persistence off — steps still present, clearly skipped.
    assert settings.derived_data_persistence_enabled is False
    result = run_daily_pipeline(settings, con, dry_run=True)
    names = [s.name for s in result.steps]
    assert "features" in names
    assert "labels" in names
    assert "feature_label_validation" in names
    by_name = {s.name: s.result for s in result.steps}
    assert by_name["features"] == "SKIPPED - derived persistence disabled"
    assert by_name["labels"] == "SKIPPED - derived persistence disabled"
    assert by_name["feature_label_validation"] == "SKIPPED - derived persistence disabled"
    assert "prices" in by_name
    _assert_no_mutations(con, settings)
    assert not any(settings.lake_dir.rglob("*.parquet"))

    again = run_daily_pipeline(settings, con, dry_run=True)
    assert [s.name for s in again.steps] == names
    _assert_no_mutations(con, settings)

    # With derived persistence enabled, dry-run still plans features/labels without mutation.
    settings.derived_data_persistence_enabled = True
    enabled = run_daily_pipeline(settings, con, dry_run=True)
    enabled_by = {s.name: s.result for s in enabled.steps}
    assert "SKIPPED - derived persistence disabled" not in enabled_by["features"]
    assert "SKIPPED - derived persistence disabled" not in enabled_by["labels"]
    _assert_no_mutations(con, settings)
    assert not any(settings.lake_dir.rglob("*.parquet"))
