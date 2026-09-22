"""Operational environment checks before Windows Task Scheduler registration."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import duckdb

from app.config.settings import Settings
from app.research.dataset import daily_collection_scope, label_maturity_dates, reconcile_price_feature_rows
from app.services.market_calendar import MarketCalendarService
from app.utils.time_utils import format_session_date

STATUS_OK = "OK"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"

DISK_FAIL_GB = 2.0
DISK_WARN_GB = 10.0


@dataclass
class DoctorCheck:
    name: str
    status: str
    detail: str


@dataclass
class DoctorReport:
    checks: list[DoctorCheck] = field(default_factory=list)
    fail_count: int = 0
    warn_count: int = 0
    ok_count: int = 0
    scheduler_ready_from_doctor: bool = False

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(DoctorCheck(name, status, detail))
        if status == STATUS_FAIL:
            self.fail_count += 1
        elif status == STATUS_WARN:
            self.warn_count += 1
        else:
            self.ok_count += 1


def _writable(path: Path) -> tuple[bool, str]:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".stockdb_doctor_write"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True, str(path.resolve())
    except OSError as exc:
        return False, str(exc)


def run_doctor(settings: Settings, con: duckdb.DuckDBPyConnection | None = None) -> DoctorReport:
    """Inspect the local environment. Never prints secret values."""
    report = DoctorReport()
    owns_con = con is None
    local_con = con

    py = sys.version.split()[0]
    report.add("python_environment", STATUS_OK, f"Python {py} ({sys.executable})")

    uv_path = shutil.which("uv")
    if uv_path is None:
        candidate = Path.home() / ".local" / "bin" / "uv.exe"
        if candidate.exists():
            uv_path = str(candidate)
    if uv_path:
        report.add("uv_environment", STATUS_OK, f"uv available at {uv_path}")
    else:
        report.add("uv_environment", STATUS_WARN, "uv executable not found on PATH")

    try:
        if local_con is None:
            local_con = duckdb.connect(str(settings.duckdb_path))
        local_con.execute("SELECT 1").fetchone()
        report.add("duckdb_open", STATUS_OK, str(settings.duckdb_path))
    except Exception as exc:  # noqa: BLE001
        report.add("duckdb_open", STATUS_FAIL, str(exc))
        local_con = None

    lake_ok, lake_detail = _writable(settings.lake_dir)
    report.add("lake_writable", STATUS_OK if lake_ok else STATUS_FAIL, lake_detail)
    state_ok, state_detail = _writable(settings.state_dir)
    report.add("state_writable", STATUS_OK if state_ok else STATUS_FAIL, state_detail)

    if settings.sec_user_agent_configured:
        report.add("SEC_USER_AGENT", STATUS_OK, "configured")
    else:
        report.add(
            "SEC_USER_AGENT",
            STATUS_FAIL,
            'not configured - set SEC_USER_AGENT="StockPatternResearch contact@example.com" in .env',
        )
    if settings.fred_api_key_configured:
        report.add("FRED_API_KEY", STATUS_OK, "configured")
    else:
        report.add("FRED_API_KEY", STATUS_WARN, "optional; macro step will SKIP")

    try:
        from app.providers.price.yfinance_provider import YFinancePriceProvider

        YFinancePriceProvider(settings)
        report.add("yfinance_provider", STATUS_OK, "import and construct succeeded (no network fetch)")
    except Exception as exc:  # noqa: BLE001
        report.add("yfinance_provider", STATUS_FAIL, str(exc))

    if settings.commercial_mode:
        report.add("commercial_mode", STATUS_FAIL, "true — research lake expects commercial_mode=false")
    else:
        report.add("commercial_mode", STATUS_OK, "false")

    if settings.derived_data_persistence_enabled:
        report.add(
            "derived_data_persistence",
            STATUS_OK,
            "enabled — features_daily / labels_forward_returns writes active",
        )
    else:
        report.add(
            "derived_data_persistence",
            STATUS_OK,
            "disabled — source-only soak; features/labels skipped (existing derived data preserved)",
        )

    try:
        calendar = MarketCalendarService(settings.market_calendar, settings.market_data_grace_minutes)
        friday = date(2026, 9, 18)
        saturday = date(2026, 9, 19)
        ok_cal = calendar.is_trading_day(friday) and not calendar.is_trading_day(saturday)
        if ok_cal:
            report.add(
                "market_calendar",
                STATUS_OK,
                f"{settings.market_calendar}: {format_session_date(friday)} trading, "
                f"{format_session_date(saturday)} not trading",
            )
        else:
            report.add("market_calendar", STATUS_FAIL, "2026-09-18/19 session map mismatch")
    except Exception as exc:  # noqa: BLE001
        report.add("market_calendar", STATUS_FAIL, str(exc))

    if local_con is not None:
        try:
            scope = daily_collection_scope(local_con)
            tracked = scope.get("total_tracked_price_targets") or scope.get("tracked_price") or 0
            status = STATUS_OK if tracked > 0 else STATUS_WARN
            report.add(
                "tracked_securities",
                status,
                (
                    f"research_common={scope.get('research_common_equity_500', 0)} "
                    f"scale_extras={scope.get('provider_scale_test_extras', 0)} "
                    f"benchmarks={scope.get('benchmarks', 0)} "
                    f"total_tracked_price={tracked} "
                    f"(known_securities={scope.get('known_securities', 0)} is not the daily target)"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            report.add("tracked_securities", STATUS_WARN, str(exc))
        try:
            from app.db.schema import create_lake_views

            create_lake_views(local_con, settings)
            names = {r[0] for r in local_con.execute("SHOW TABLES").fetchall()}
            latest_price = None
            latest_feat = None
            if "prices_daily" in names:
                row = local_con.execute("SELECT max(date) FROM prices_daily").fetchone()
                latest_price = row[0] if row else None
            if "features_daily" in names:
                row = local_con.execute("SELECT max(date) FROM features_daily").fetchone()
                latest_feat = row[0] if row else None
            report.add(
                "latest_price_session",
                STATUS_OK if latest_price else STATUS_WARN,
                format_session_date(latest_price),
            )
            report.add(
                "latest_feature_date",
                STATUS_OK if latest_feat else STATUS_WARN,
                format_session_date(latest_feat),
            )
            maturity = label_maturity_dates(settings, local_con)
            report.add(
                "labels_maturity",
                STATUS_OK if maturity.get("latest_label_row_date") else STATUS_WARN,
                (
                    f"row={format_session_date(maturity.get('latest_label_row_date'))} "
                    f"mature_1d={format_session_date(maturity.get('latest_mature_1d'))} "
                    f"mature_5d={format_session_date(maturity.get('latest_mature_5d'))} "
                    f"mature_10d={format_session_date(maturity.get('latest_mature_10d'))} "
                    f"mature_20d={format_session_date(maturity.get('latest_mature_20d'))} "
                    "(recent nulls are immature, not missing rows)"
                ),
            )
            recon = reconcile_price_feature_rows(settings, local_con)
            unexpected = recon.price_no_feature_unexpected
            report.add(
                "price_feature_reconciliation",
                STATUS_OK if unexpected == 0 else STATUS_WARN,
                _recon_detail(recon),
            )
        except Exception as exc:  # noqa: BLE001
            report.add("lake_status", STATUS_WARN, str(exc))

    try:
        usage = shutil.disk_usage(settings.data_root if settings.data_root.exists() else Path("."))
        free_gb = usage.free / (1024**3)
        if free_gb < DISK_FAIL_GB:
            disk_status = STATUS_FAIL
        elif free_gb < DISK_WARN_GB:
            disk_status = STATUS_WARN
        else:
            disk_status = STATUS_OK
        report.add("disk_free", disk_status, f"{free_gb:.1f} GB free")
    except OSError as exc:
        report.add("disk_free", STATUS_WARN, str(exc))

    report.scheduler_ready_from_doctor = report.fail_count == 0 and settings.sec_user_agent_configured
    if owns_con and local_con is not None and con is None:
        local_con.close()
    return report


def _recon_detail(recon) -> str:  # noqa: ANN001
    detail = (
        f"PRICE_WITH_FEATURE={recon.price_with_feature} "
        f"PRICE_NO_FEATURE_EXPECTED={recon.price_no_feature_expected} "
        f"PRICE_NO_FEATURE_UNEXPECTED={recon.price_no_feature_unexpected}"
    )
    if "DERIVED_DATA_PERSISTENCE_ENABLED=false" in (recon.note or ""):
        detail += " (derived persistence disabled; missing features expected)"
    return detail
