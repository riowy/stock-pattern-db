"""Doctor command: environment checks without leaking secrets."""

from __future__ import annotations

import duckdb
import pytest
from typer.testing import CliRunner

from app.cli.main import app
from app.db.schema import apply_schema
from app.services.doctor_service import STATUS_FAIL, STATUS_OK, run_doctor


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    apply_schema(c)
    return c


def test_doctor_reports_ok_warn_fail_and_hides_secrets(con, settings) -> None:
    settings.sec_user_agent = "SuperSecretUserAgent hidden@example.com"
    settings.fred_api_key = "FRED-SECRET-KEY-XYZ"
    report = run_doctor(settings, con)
    blob = " | ".join(f"{c.name}={c.status}:{c.detail}" for c in report.checks)
    assert "SuperSecretUserAgent" not in blob
    assert "hidden@example.com" not in blob
    assert "FRED-SECRET-KEY-XYZ" not in blob
    names = {c.name: c for c in report.checks}
    assert names["SEC_USER_AGENT"].status == STATUS_OK
    assert names["SEC_USER_AGENT"].detail == "configured"
    assert names["FRED_API_KEY"].status == STATUS_OK
    assert names["commercial_mode"].status == STATUS_OK
    assert names["derived_data_persistence"].status == STATUS_OK
    assert "disabled" in names["derived_data_persistence"].detail.lower()
    assert names["market_calendar"].status == STATUS_OK
    assert any(c.status in (STATUS_OK, STATUS_FAIL) for c in report.checks)


def test_doctor_fails_when_sec_user_agent_missing(con, settings) -> None:
    settings.sec_user_agent = "  "
    report = run_doctor(settings, con)
    sec = next(c for c in report.checks if c.name == "SEC_USER_AGENT")
    assert sec.status == STATUS_FAIL
    assert "configured" in sec.detail.lower() or "not configured" in sec.detail.lower()
    assert report.fail_count >= 1
    assert report.scheduler_ready_from_doctor is False


def test_doctor_cli_does_not_print_secret(monkeypatch, settings, con) -> None:
    secret = "CLI-SECRET-USER-AGENT-4242"
    settings.sec_user_agent = secret

    def _boot():
        return settings

    monkeypatch.setattr("app.cli.main._bootstrap", _boot)

    class _Ctx:
        def __enter__(self):
            return con

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("app.cli.main.duckdb_connection", lambda *_a, **_k: _Ctx())
    monkeypatch.setattr("app.cli.main._prepare_db", lambda *_a, **_k: None)
    runner = CliRunner()
    result = runner.invoke(app, ["doctor"])
    assert secret not in result.output
    assert "SEC_USER_AGENT" in result.output
    assert "YES" in result.output or "OK" in result.output
