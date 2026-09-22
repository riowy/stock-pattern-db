"""Source-only soak: DERIVED_DATA_PERSISTENCE_ENABLED pauses feature/label writes."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import duckdb
import polars as pl
import pytest

from app.db.schema import apply_schema, create_lake_views
from app.ingestion.daily_pipeline import run_daily_pipeline
from app.ingestion.feature_compute import compute_features
from app.ingestion.label_compute import compute_labels
from app.ingestion.price_repair import repair_prices
from app.research.dataset import reconcile_price_feature_rows
from app.services.doctor_service import STATUS_OK, run_doctor
from app.services.status_service import gather_status
from app.services.tracked_universe_service import add_tracked
from app.utils.parquet_io import LakeDataset


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    apply_schema(c)
    return c


def _insert_security(con, security_id="SEC0001", ticker="AAPL") -> None:  # noqa: ANN001
    now = datetime.now(UTC)
    con.execute(
        """
        INSERT INTO securities
            (security_id, cik, company_name, primary_ticker, exchange, asset_type,
             currency, is_active, first_seen_at, last_seen_at, created_at, updated_at)
        VALUES (?, '0001', 'Test', ?, 'NYSE', 'EQUITY', 'USD', TRUE, ?, ?, ?, ?)
        """,
        [security_id, ticker, now, now, now, now],
    )


def _write_prices(settings, security_id: str, ticker: str, n: int = 10) -> pl.DataFrame:  # noqa: ANN001
    ds = LakeDataset(settings.prices_daily_dir, "date", ["security_id", "date"], ["security_id", "date"])
    start = date(2024, 1, 2)
    dates: list[date] = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    df = pl.DataFrame(
        {
            "security_id": [security_id] * n,
            "ticker_at_time": [ticker] * n,
            "date": dates,
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.0 + i for i in range(n)],
            "adj_close": [100.0 + i for i in range(n)],
            "volume": [1000.0] * n,
            "dividend": [0.0] * n,
            "stock_split": [0.0] * n,
            "currency": ["USD"] * n,
            "provider": ["test"] * n,
            "retrieved_at": [datetime.now(UTC)] * n,
        }
    )
    ds.write_increment(df, run_id="seed")
    return df


def _write_feature_rows(settings, security_id: str, ticker: str, dates: list[date]) -> None:  # noqa: ANN001
    ds = LakeDataset(
        settings.features_daily_dir, "date", ["security_id", "date"], ["security_id", "date"]
    )
    n = len(dates)
    df = pl.DataFrame(
        {
            "security_id": [security_id] * n,
            "ticker_at_time": [ticker] * n,
            "date": dates,
            "feature_version": ["v1"] * n,
            "calculated_at": [datetime.now(UTC)] * n,
        }
    )
    ds.write_increment(df, run_id="seed-feat")


def _write_label_rows(settings, security_id: str, ticker: str, dates: list[date]) -> None:  # noqa: ANN001
    ds = LakeDataset(
        settings.labels_forward_returns_dir, "date", ["security_id", "date"], ["security_id", "date"]
    )
    n = len(dates)
    df = pl.DataFrame(
        {
            "security_id": [security_id] * n,
            "ticker_at_time": [ticker] * n,
            "date": dates,
            "label_version": ["v1"] * n,
            "forward_return_1d": [0.01] * n,
            "forward_return_5d": [None] * n,
            "forward_return_10d": [None] * n,
            "forward_return_20d": [None] * n,
            "calculated_at": [datetime.now(UTC)] * n,
        }
    )
    ds.write_increment(df, run_id="seed-lab")


def _snapshot_derived(settings) -> tuple[dict[Path, int], dict[Path, int]]:  # noqa: ANN001
    feat = {p: p.stat().st_mtime_ns for p in settings.features_daily_dir.rglob("*.parquet")}
    labels = {p: p.stat().st_mtime_ns for p in settings.labels_forward_returns_dir.rglob("*.parquet")}
    return feat, labels


def _assert_derived_unchanged(settings, before: tuple[dict[Path, int], dict[Path, int]]) -> None:  # noqa: ANN001
    feat_b, lab_b = before
    feat_a = {p: p.stat().st_mtime_ns for p in settings.features_daily_dir.rglob("*.parquet")}
    lab_a = {p: p.stat().st_mtime_ns for p in settings.labels_forward_returns_dir.rglob("*.parquet")}
    assert feat_a == feat_b
    assert lab_a == lab_b


def _stub_upstream(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.sync_universe",
        lambda *_a, **_k: SimpleNamespace(dry_run=False, securities_seen=1),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.sync_recent_prices",
        lambda *_a, **_k: SimpleNamespace(
            dry_run=False,
            tracked_targets=1,
            provider_fetch_skipped=False,
            fetch_targets=1,
            successful=1,
            total_symbols=1,
            rows_written=5,
            message="ok",
            symbols_requested=1,
            current=0,
            stale=1,
            no_data=0,
            batches_requested=1,
            rows_received=5,
            provider_fetch_count=1,
            requested_session_start=date(2024, 1, 2),
            requested_session_end=date(2024, 1, 5),
        ),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.sync_vix",
        lambda *_a, **_k: SimpleNamespace(dry_run=False, rows_written=1),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.sync_macro",
        lambda *_a, **_k: SimpleNamespace(
            status="skipped", skip_reason="optional", dry_run=False, series_synced=0, rows_written=0
        ),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.sync_filings",
        lambda *_a, **_k: SimpleNamespace(
            status="skipped", skip_reason="optional", dry_run=False, successful=0, rows_written=0
        ),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.validate_prices",
        lambda *_a, **_k: SimpleNamespace(scope_size=1, issues_found=0, critical=0),
    )
    monkeypatch.setattr(
        "app.services.price_audit_service.audit_prices",
        lambda *_a, **_k: SimpleNamespace(research_critical=0),
    )


def test_derived_disabled_prices_can_write_features_labels_zero(con, settings, monkeypatch) -> None:
    assert settings.derived_data_persistence_enabled is False
    _insert_security(con)
    add_tracked(con, ["SEC0001"], reason="test", feature_tracking=True)
    prices = _write_prices(settings, "SEC0001", "AAPL", n=5)
    create_lake_views(con, settings)
    _write_feature_rows(settings, "SEC0001", "AAPL", [prices["date"][0]])
    _write_label_rows(settings, "SEC0001", "AAPL", [prices["date"][0]])
    before = _snapshot_derived(settings)

    calls = {"features": 0, "labels": 0}

    def boom_features(*_a, **_k):  # noqa: ANN002, ANN003
        calls["features"] += 1
        raise AssertionError("feature compute must not run when derived persistence disabled")

    def boom_labels(*_a, **_k):  # noqa: ANN002, ANN003
        calls["labels"] += 1
        raise AssertionError("label compute must not run when derived persistence disabled")

    _stub_upstream(monkeypatch)
    monkeypatch.setattr("app.ingestion.daily_pipeline.compute_features", boom_features)
    monkeypatch.setattr("app.ingestion.daily_pipeline.compute_labels", boom_labels)

    pipeline = run_daily_pipeline(settings, con, dry_run=False)
    by_name = {s.name: s.result for s in pipeline.steps}
    assert by_name["features"] == "SKIPPED - derived persistence disabled"
    assert by_name["labels"] == "SKIPPED - derived persistence disabled"
    assert by_name["feature_label_validation"] == "SKIPPED - derived persistence disabled"
    assert calls["features"] == 0
    assert calls["labels"] == 0

    r_feat = compute_features(settings, con, None, date(2024, 1, 2), None, dry_run=False)
    r_lab = compute_labels(settings, con, None, date(2024, 1, 2), None, dry_run=False)
    assert r_feat.rows_written == 0
    assert r_lab.rows_written == 0
    _assert_derived_unchanged(settings, before)
    assert any(settings.prices_daily_dir.rglob("*.parquet"))


def test_repair_prices_does_not_trigger_derived_writes(con, settings, monkeypatch) -> None:
    assert settings.derived_data_persistence_enabled is False
    _insert_security(con)
    add_tracked(con, ["SEC0001"], reason="test", feature_tracking=True)
    _write_feature_rows(settings, "SEC0001", "AAPL", [date(2024, 1, 2)])
    _write_label_rows(settings, "SEC0001", "AAPL", [date(2024, 1, 2)])
    before = _snapshot_derived(settings)

    monkeypatch.setattr(
        "app.ingestion.price_repair.run_price_ingestion",
        lambda *_a, **_k: SimpleNamespace(
            run_id=None,
            total_symbols=1,
            successful=1,
            failed=0,
            rows_written=3,
            dry_run=False,
            provider_fetch_count=1,
        ),
    )

    result = repair_prices(settings, con, lookback_sessions=5, dry_run=False)
    assert result.rows_written == 3
    assert compute_features(settings, con, None, date(2024, 1, 2), None).rows_written == 0
    assert compute_labels(settings, con, None, date(2024, 1, 2), None).rows_written == 0
    _assert_derived_unchanged(settings, before)


def test_reconciliation_treats_missing_derived_as_expected_when_disabled(con, settings) -> None:
    settings.derived_data_persistence_enabled = False
    _insert_security(con, "FEAT1", "AAA")
    add_tracked(con, ["FEAT1"], reason="test", feature_tracking=True)
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
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
    n = len(dates)
    prices.write_increment(
        pl.DataFrame(
            {
                "security_id": ["FEAT1"] * n,
                "ticker_at_time": ["AAA"] * n,
                "date": dates,
                "open": [10.0] * n,
                "high": [10.1] * n,
                "low": [9.9] * n,
                "close": [10.0] * n,
                "adj_close": [10.0] * n,
                "volume": [1000.0] * n,
                "retrieved_at": [datetime.now(UTC)] * n,
            }
        ),
        "p",
    )
    feat.write_increment(
        pl.DataFrame(
            {
                "security_id": ["FEAT1", "FEAT1"],
                "ticker_at_time": ["AAA", "AAA"],
                "date": dates[:2],
                "feature_version": ["v1", "v1"],
                "calculated_at": [datetime.now(UTC)] * 2,
            }
        ),
        "f",
    )
    create_lake_views(con, settings)
    recon = reconcile_price_feature_rows(settings, con)
    assert recon.price_with_feature == 2
    assert recon.price_no_feature_unexpected == 0
    assert recon.price_no_feature_expected == 2
    assert recon.unexpected_examples == []
    assert "DERIVED_DATA_PERSISTENCE_ENABLED=false" in recon.note


def test_doctor_ok_when_derived_persistence_disabled(con, settings) -> None:
    settings.derived_data_persistence_enabled = False
    _insert_security(con, "FEAT1", "AAA")
    add_tracked(con, ["FEAT1"], reason="test", feature_tracking=True)
    prices = LakeDataset(
        settings.lake_dir / "prices_daily",
        "date",
        ["security_id", "date"],
        ["security_id", "date"],
    )
    prices.write_increment(
        pl.DataFrame(
            {
                "security_id": ["FEAT1"],
                "ticker_at_time": ["AAA"],
                "date": [date(2024, 1, 2)],
                "open": [10.0],
                "high": [10.1],
                "low": [9.9],
                "close": [10.0],
                "adj_close": [10.0],
                "volume": [1000.0],
                "retrieved_at": [datetime.now(UTC)],
            }
        ),
        "p",
    )
    create_lake_views(con, settings)
    report = run_doctor(settings, con)
    names = {c.name: c for c in report.checks}
    assert names["derived_data_persistence"].status == STATUS_OK
    assert "disabled" in names["derived_data_persistence"].detail.lower()
    assert names["price_feature_reconciliation"].status == STATUS_OK
    assert "UNEXPECTED=0" in names["price_feature_reconciliation"].detail
    assert names["price_feature_reconciliation"].status != STATUS_FAIL

    status = gather_status(settings, con)
    assert status.persistence.derived_data_persistence_enabled is False


def test_flag_true_restores_feature_label_pipeline(con, settings, monkeypatch) -> None:
    settings.derived_data_persistence_enabled = True
    _insert_security(con)
    add_tracked(con, ["SEC0001"], reason="test", feature_tracking=True)
    _write_prices(settings, "SEC0001", "AAPL", n=15)
    create_lake_views(con, settings)

    called = {"features": 0, "labels": 0}

    def fake_features(*_a, **_k):  # noqa: ANN002, ANN003
        called["features"] += 1
        return SimpleNamespace(dry_run=False, successful=1, total_symbols=1, rows_written=7)

    def fake_labels(*_a, **_k):  # noqa: ANN002, ANN003
        called["labels"] += 1
        return SimpleNamespace(dry_run=False, successful=1, total_symbols=1, rows_written=7)

    _stub_upstream(monkeypatch)
    monkeypatch.setattr("app.ingestion.daily_pipeline.compute_features", fake_features)
    monkeypatch.setattr("app.ingestion.daily_pipeline.compute_labels", fake_labels)
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.validate_features",
        lambda *_a, **_k: SimpleNamespace(issues_found=0, critical=0),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.validate_labels",
        lambda *_a, **_k: SimpleNamespace(issues_found=0, critical=0),
    )

    pipeline = run_daily_pipeline(settings, con, dry_run=False)
    by_name = {s.name: s.result for s in pipeline.steps}
    assert called["features"] == 1
    assert called["labels"] == 1
    assert "derived persistence disabled" not in by_name["features"]
    assert "ok" in by_name["features"]
    assert "ok" in by_name["labels"]


def test_dry_run_no_mutation_with_derived_disabled(con, settings, monkeypatch) -> None:
    settings.derived_data_persistence_enabled = False
    _insert_security(con)
    add_tracked(con, ["SEC0001"], reason="test", feature_tracking=True)
    _write_feature_rows(settings, "SEC0001", "AAPL", [date(2024, 1, 2)])
    before = _snapshot_derived(settings)
    ingest_before = con.execute("SELECT count(*) FROM ingest_runs").fetchone()[0]

    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.sync_universe",
        lambda *_a, **_k: SimpleNamespace(dry_run=True, securities_seen=1),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.sync_recent_prices",
        lambda *_a, **_k: SimpleNamespace(
            dry_run=True,
            tracked_targets=1,
            provider_fetch_skipped=False,
            fetch_targets=1,
            successful=0,
            total_symbols=1,
            rows_written=0,
            message="would fetch",
            symbols_requested=1,
            current=0,
            stale=1,
            no_data=0,
            batches_requested=1,
            rows_received=0,
            provider_fetch_count=0,
            requested_session_start=date(2024, 1, 2),
            requested_session_end=date(2024, 1, 5),
        ),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.sync_vix",
        lambda *_a, **_k: SimpleNamespace(dry_run=True, rows_written=0),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.sync_macro",
        lambda *_a, **_k: SimpleNamespace(
            status="planned", skip_reason=None, dry_run=True, series_synced=3, rows_written=0
        ),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.sync_filings",
        lambda *_a, **_k: SimpleNamespace(
            status="planned", skip_reason=None, dry_run=True, successful=1, rows_written=0
        ),
    )
    monkeypatch.setattr(
        "app.ingestion.daily_pipeline.validate_prices",
        lambda *_a, **_k: SimpleNamespace(scope_size=1, issues_found=0, critical=0),
    )

    pipeline = run_daily_pipeline(settings, con, dry_run=True)
    by_name = {s.name: s.result for s in pipeline.steps}
    assert "fetch" in by_name["prices"]
    assert by_name["features"] == "SKIPPED - derived persistence disabled"
    assert by_name["labels"] == "SKIPPED - derived persistence disabled"
    assert con.execute("SELECT count(*) FROM ingest_runs").fetchone()[0] == ingest_before
    _assert_derived_unchanged(settings, before)
