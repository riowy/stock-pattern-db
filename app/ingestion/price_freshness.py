"""Per-security price freshness versus the latest completed XNYS session.

Daily sync must not skip the whole universe from ``max(prices.date)`` alone.
Each tracked price target is CURRENT, STALE, or NO_DATA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import duckdb

from app.config.lake_datasets import get_lake_dataset
from app.config.settings import Settings
from app.db.schema import create_lake_views
from app.services.market_calendar import MarketCalendarService
from app.services.tracked_universe_service import get_tracked_price_security_ids

STATUS_CURRENT = "CURRENT"
STATUS_STALE = "STALE"
STATUS_NO_DATA = "NO_DATA"

# Previous daily trailing window, kept for names with no price rows yet.
NO_DATA_DAILY_CALENDAR_LOOKBACK_DAYS = 10


def provider_end_exclusive(last_inclusive_session: date) -> date:
    """Convert an inclusive last session into yfinance ``history(end=)``.

    yfinance/Yahoo treat ``end`` as exclusive. Passing the session itself
    would drop that bar (off-by-one). A +1 calendar day keeps the last
    wanted session without inventing a new timezone.
    """
    return last_inclusive_session + timedelta(days=1)


def stale_fetch_start(calendar: MarketCalendarService, latest_stored: date, lookback_sessions: int) -> date:
    """Inclusive start: ``lookback_sessions`` XNYS sessions before last stored."""
    return calendar.sessions_ago(latest_stored, lookback_sessions)


@dataclass
class FreshnessRow:
    security_id: str
    ticker: str | None
    latest_session: date | None
    status: str


@dataclass
class FreshnessReport:
    expected_latest: date
    tracked_targets: int = 0
    current: list[FreshnessRow] = field(default_factory=list)
    stale: list[FreshnessRow] = field(default_factory=list)
    no_data: list[FreshnessRow] = field(default_factory=list)

    @property
    def current_count(self) -> int:
        return len(self.current)

    @property
    def stale_count(self) -> int:
        return len(self.stale)

    @property
    def no_data_count(self) -> int:
        return len(self.no_data)

    @property
    def fetch_tickers(self) -> list[str]:
        rows = self.stale + self.no_data
        return [r.ticker for r in rows if r.ticker]


def classify_price_freshness(
    settings: Settings,
    con: duckdb.DuckDBPyConnection,
    *,
    now=None,  # noqa: ANN001
    calendar: MarketCalendarService | None = None,
) -> FreshnessReport:
    cal = calendar or MarketCalendarService(settings.market_calendar, settings.market_data_grace_minutes)
    expected = cal.expected_latest_completed_session(now)
    report = FreshnessReport(expected_latest=expected)
    ids = get_tracked_price_security_ids(con)
    report.tracked_targets = len(ids)
    if not ids:
        return report

    create_lake_views(con, settings)
    latest_by_sid: dict[str, date] = {}
    ds = get_lake_dataset(settings.lake_dir, "prices_daily")
    names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if ds.has_any_files() and "prices_daily" in names:
        placeholders = ", ".join("?" for _ in ids)
        rows = con.execute(
            f"SELECT security_id, max(date) FROM prices_daily "
            f"WHERE security_id IN ({placeholders}) GROUP BY security_id",
            ids,
        ).fetchall()
        latest_by_sid = {r[0]: r[1] for r in rows if r[1] is not None}

    placeholders = ", ".join("?" for _ in ids)
    tickers = {
        r[0]: r[1]
        for r in con.execute(
            f"SELECT security_id, primary_ticker FROM securities WHERE security_id IN ({placeholders})",
            ids,
        ).fetchall()
    }
    for sid in ids:
        latest = latest_by_sid.get(sid)
        ticker = tickers.get(sid)
        if latest is None:
            status = STATUS_NO_DATA
        elif latest >= expected:
            status = STATUS_CURRENT
        else:
            status = STATUS_STALE
        row = FreshnessRow(sid, ticker, latest, status)
        if status == STATUS_CURRENT:
            report.current.append(row)
        elif status == STATUS_STALE:
            report.stale.append(row)
        else:
            report.no_data.append(row)
    return report


def daily_fetch_window(
    freshness: FreshnessReport,
    calendar: MarketCalendarService,
    lookback_sessions: int,
) -> tuple[date, date] | None:
    """Inclusive start and inclusive end for STALE+NO_DATA daily fetches."""
    if not freshness.stale and not freshness.no_data:
        return None
    starts: list[date] = []
    for row in freshness.stale:
        if row.latest_session is None:
            continue
        starts.append(stale_fetch_start(calendar, row.latest_session, lookback_sessions))
    if freshness.no_data:
        starts.append(freshness.expected_latest - timedelta(days=NO_DATA_DAILY_CALENDAR_LOOKBACK_DAYS))
    if not starts:
        return None
    return min(starts), freshness.expected_latest
