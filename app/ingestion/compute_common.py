"""Shared helpers for feature/label compute jobs."""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

from app.config.settings import Settings
from app.db.schema import create_lake_views
from app.ingestion.price_backfill import resolve_symbols
from app.services.market_calendar import MarketCalendarService
from app.services.tracked_universe_service import get_tracked_feature_security_ids


def resolve_feature_targets(
    con: duckdb.DuckDBPyConnection, symbols: list[str] | None
) -> list[tuple[str, str]]:
    """[(ticker, security_id), ...]. Default: tracked + feature_tracking."""
    if symbols:
        return resolve_symbols(con, symbols, default_scope="tracked")
    ids = get_tracked_feature_security_ids(con)
    if not ids:
        return []
    placeholders = ", ".join("?" for _ in ids)
    rows = con.execute(
        f"SELECT primary_ticker, security_id FROM securities "
        f"WHERE security_id IN ({placeholders}) AND primary_ticker IS NOT NULL "
        f"ORDER BY primary_ticker",
        ids,
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def lookback_start(settings: Settings, start: date) -> date:
    calendar = MarketCalendarService(settings.market_calendar, settings.market_data_grace_minutes)
    anchor = start
    if not calendar.is_trading_day(anchor):
        anchor = calendar.next_trading_day(anchor)
    return calendar.sessions_ago(anchor, settings.max_feature_lookback_sessions)


def lookahead_end(settings: Settings, end: date, sessions: int = 20) -> date:
    calendar = MarketCalendarService(settings.market_calendar, settings.market_data_grace_minutes)
    anchor = end
    if not calendar.is_trading_day(anchor):
        anchor = calendar.previous_trading_day(anchor)
    return calendar.sessions_ahead(anchor, sessions)


def month_partition_count(start: date, end: date) -> int:
    if end < start:
        return 0
    return (end.year - start.year) * 12 + (end.month - start.month) + 1


def load_prices(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    security_ids: list[str],
    start: date,
    end: date,
) -> pl.DataFrame:
    create_lake_views(con, settings)
    from app.config.lake_datasets import get_lake_dataset

    if not get_lake_dataset(settings.lake_dir, "prices_daily").has_any_files():
        return pl.DataFrame()
    if not security_ids:
        return pl.DataFrame()
    placeholders = ", ".join("?" for _ in security_ids)
    return con.execute(
        f"""
        SELECT * FROM prices_daily
        WHERE date >= ? AND date <= ?
          AND (security_id IN ({placeholders}) OR ticker_at_time IN ('SPY', 'QQQ'))
        ORDER BY security_id, date
        """,
        [start, end, *security_ids],
    ).pl()


def load_vix(con: duckdb.DuckDBPyConnection, settings: Settings, start: date, end: date) -> pl.DataFrame:
    create_lake_views(con, settings)
    from app.config.lake_datasets import get_lake_dataset

    if not get_lake_dataset(settings.lake_dir, "volatility").has_any_files():
        return pl.DataFrame()
    return con.execute(
        "SELECT date, close FROM volatility WHERE date >= ? AND date <= ? ORDER BY date",
        [start, end],
    ).pl()
