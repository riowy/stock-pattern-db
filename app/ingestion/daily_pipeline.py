"""Daily pipeline: ingest -> validate -> incremental features/labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import duckdb

from app.config.lake_datasets import get_lake_dataset
from app.config.settings import Settings
from app.db.schema import create_lake_views
from app.ingestion.feature_compute import compute_features
from app.ingestion.filings_sync import sync_filings
from app.ingestion.label_compute import compute_labels
from app.ingestion.macro_sync import sync_macro
from app.ingestion.price_sync import sync_recent_prices
from app.ingestion.universe_sync import sync_universe
from app.ingestion.vix_sync import sync_vix
from app.services.market_calendar import MarketCalendarService
from app.utils.logging import get_logger
from app.validation.runner import validate_features, validate_labels, validate_prices

logger = get_logger("daily_pipeline")

FEATURE_DAILY_OVERLAP_SESSIONS = 5


@dataclass
class StepResult:
    name: str
    result: str
    failed: bool = False
    abort: bool = False


@dataclass
class DailyPipelineResult:
    steps: list[StepResult] = field(default_factory=list)
    aborted: bool = False
    dry_run: bool = False


def _latest_price_date(con: duckdb.DuckDBPyConnection, settings: Settings) -> date | None:
    create_lake_views(con, settings)
    ds = get_lake_dataset(settings.lake_dir, "prices_daily")
    if not ds.has_any_files():
        return None
    row = con.execute("SELECT max(date) FROM prices_daily").fetchone()
    return row[0] if row and row[0] else None


def _latest_feature_date(con: duckdb.DuckDBPyConnection, settings: Settings) -> date | None:
    create_lake_views(con, settings)
    ds = get_lake_dataset(settings.lake_dir, "features_daily")
    if not ds.has_any_files():
        return None
    row = con.execute("SELECT max(date) FROM features_daily").fetchone()
    return row[0] if row and row[0] else None


def _feature_daily_start(con: duckdb.DuckDBPyConnection, settings: Settings) -> date | None:
    latest_price = _latest_price_date(con, settings)
    if latest_price is None:
        return None
    calendar = MarketCalendarService(settings.market_calendar, settings.market_data_grace_minutes)
    latest_feat = _latest_feature_date(con, settings)
    if latest_feat is None:
        row = con.execute("SELECT min(date) FROM prices_daily").fetchone()
        return row[0] if row and row[0] else latest_price
    return calendar.sessions_ago(latest_price, FEATURE_DAILY_OVERLAP_SESSIONS)


def _label_daily_start(con: duckdb.DuckDBPyConnection, settings: Settings) -> date | None:
    latest_price = _latest_price_date(con, settings)
    if latest_price is None:
        return None
    calendar = MarketCalendarService(settings.market_calendar, settings.market_data_grace_minutes)
    return calendar.sessions_ago(latest_price, settings.label_recompute_sessions)


def run_daily_pipeline(
    settings: Settings, con: duckdb.DuckDBPyConnection, dry_run: bool = False
) -> DailyPipelineResult:
    out = DailyPipelineResult(dry_run=dry_run)

    def add(name: str, result: str, failed: bool = False, abort: bool = False) -> None:
        out.steps.append(StepResult(name, result, failed, abort))
        if abort:
            out.aborted = True

    try:
        r = sync_universe(settings, con, dry_run)
        verb = "would sync" if r.dry_run else "synced"
        add("universe", f"{verb} SEC universe + ETF seed list ({r.securities_seen} securities known)")
    except Exception as exc:  # noqa: BLE001
        logger.exception("run-daily: universe sync failed")
        add("universe", f"FAILED: {exc}", failed=True)

    try:
        r = sync_recent_prices(settings, con, None, 10, dry_run)
        if r.total_symbols == 0:
            add("prices", "SKIPPED - tracked universe is empty (use 'stockdb universe add')")
        elif r.dry_run:
            add("prices", f"would check/update {r.total_symbols} tracked securities")
        else:
            add("prices", f"ok ({r.successful}/{r.total_symbols}, {r.rows_written} rows)")
    except Exception as exc:  # noqa: BLE001
        logger.exception("run-daily: price sync failed")
        add("prices", f"FAILED: {exc}", failed=True)

    try:
        r = sync_vix(settings, con, dry_run)
        add("vix", "would fetch latest VIX" if r.dry_run else f"ok ({r.rows_written} rows)")
    except Exception as exc:  # noqa: BLE001
        logger.exception("run-daily: vix sync failed")
        add("vix", f"FAILED: {exc}", failed=True)

    try:
        r = sync_macro(settings, con, None, None, dry_run)
        if r.status == "skipped":
            add("macro", f"SKIPPED - {r.skip_reason}")
        elif r.dry_run:
            add("macro", f"would fetch {r.series_synced} FRED series")
        else:
            add("macro", f"ok ({r.series_synced} series, {r.rows_written} rows)")
    except Exception as exc:  # noqa: BLE001
        logger.exception("run-daily: macro sync failed unexpectedly")
        add("macro", f"FAILED: {exc}", failed=True)

    try:
        r = sync_filings(settings, con, None, dry_run)
        if r.status == "skipped":
            add("filings", f"SKIPPED - {r.skip_reason}")
        elif r.dry_run:
            add("filings", f"would check {r.successful} tracked CIK(s)")
        else:
            add("filings", f"ok ({r.successful} CIKs, {r.rows_written} rows)")
    except Exception as exc:  # noqa: BLE001
        logger.exception("run-daily: filings sync failed unexpectedly")
        add("filings", f"FAILED: {exc}", failed=True)

    try:
        summary = validate_prices(settings, con, None, dry_run, all_universe=False)
        if summary.scope_size == 0:
            add("price_validation", "SKIPPED - tracked universe is empty")
        else:
            msg = f"{summary.issues_found} issues ({summary.critical} critical) over {summary.scope_size} securities"
            if summary.critical > 0 and not dry_run:
                add("price_validation", msg, failed=True, abort=True)
                return out
            add("price_validation", msg)
    except Exception as exc:  # noqa: BLE001
        logger.exception("run-daily: price validation failed")
        add("price_validation", f"FAILED: {exc}", failed=True, abort=True)
        return out

    feat_start = _feature_daily_start(con, settings)
    try:
        if feat_start is None:
            add("features", "SKIPPED - no price data")
        else:
            r = compute_features(settings, con, None, feat_start, None, resume=False, dry_run=dry_run)
            if r.dry_run:
                add("features", f"would compute from {feat_start} for {r.total_symbols} securities")
            else:
                add("features", f"ok ({r.successful}/{r.total_symbols}, {r.rows_written} rows)")
    except Exception as exc:  # noqa: BLE001
        logger.exception("run-daily: feature compute failed")
        add("features", f"FAILED: {exc}", failed=True)

    label_start = _label_daily_start(con, settings)
    try:
        if label_start is None:
            add("labels", "SKIPPED - no price data")
        else:
            r = compute_labels(settings, con, None, label_start, None, resume=False, dry_run=dry_run)
            if r.dry_run:
                add("labels", f"would recompute last {settings.label_recompute_sessions} sessions from {label_start}")
            else:
                add("labels", f"ok ({r.successful}/{r.total_symbols}, {r.rows_written} rows)")
    except Exception as exc:  # noqa: BLE001
        logger.exception("run-daily: label compute failed")
        add("labels", f"FAILED: {exc}", failed=True)

    try:
        fs = validate_features(settings, con, None, dry_run)
        ls = validate_labels(settings, con, None, dry_run)
        msg = (
            f"features {fs.issues_found} issues ({fs.critical} critical); "
            f"labels {ls.issues_found} issues ({ls.critical} critical)"
        )
        if (fs.critical > 0 or ls.critical > 0) and not dry_run:
            add("feature_label_validation", msg, failed=True, abort=True)
            return out
        add("feature_label_validation", msg)
    except Exception as exc:  # noqa: BLE001
        logger.exception("run-daily: feature/label validation failed")
        add("feature_label_validation", f"FAILED: {exc}", failed=True, abort=True)

    add("report", "status collected")
    return out
