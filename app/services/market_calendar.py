"""US equity market trading-day calendar, wrapped behind our own interface.

Nothing outside this module should import ``exchange_calendars`` directly --
that keeps the rest of the codebase independent of which calendar library
we use. Uses the official XNYS (NYSE) session calendar so that Saturdays,
Sundays, *and* US market holidays / special closures are all correctly
excluded, instead of a naive ``weekday() < 5`` check.

Market-open/close-time reasoning always happens in ``America/New_York``
(the exchange's local timezone) regardless of the server's own timezone --
this service is exactly the seam that protects the rest of the app from
"what is 'today' for the US market" mistakes when running from Korea (or
anywhere else). All *storage* timestamps remain UTC per project convention;
this module only uses NY time internally to decide which trading session is
"latest" right now.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

MARKET_TIMEZONE = ZoneInfo("America/New_York")

# How far back/forward the underlying calendar is pre-computed. Wide enough
# to cover a full historical backfill (spec's 2000-01-01 starting point and
# earlier) plus a few years of runway into the future.
_CALENDAR_START = "1990-01-01"


def _calendar_end() -> str:
    return f"{date.today().year + 3}-12-31"


@lru_cache(maxsize=8)
def _get_calendar(name: str) -> xcals.ExchangeCalendar:
    return xcals.get_calendar(name, start=_CALENDAR_START, end=_calendar_end())


@dataclass
class MarketCalendarService:
    """Trading-day calendar abstraction. Default: NYSE (XNYS)."""

    calendar_name: str = "XNYS"
    grace_minutes: int = 120

    def __post_init__(self) -> None:
        self._cal = _get_calendar(self.calendar_name)

    # ------------------------------------------------------------------ basics
    def is_trading_day(self, d: date) -> bool:
        return bool(self._cal.is_session(pd.Timestamp(d)))

    def previous_trading_day(self, d: date) -> date:
        """The last trading session strictly before ``d``."""
        session = self._cal.date_to_session(pd.Timestamp(d), direction="previous")
        if session.date() == d:
            session = self._cal.previous_session(session)
        return session.date()

    def next_trading_day(self, d: date) -> date:
        """The next trading session strictly after ``d``."""
        session = self._cal.date_to_session(pd.Timestamp(d), direction="next")
        if session.date() == d:
            session = self._cal.next_session(session)
        return session.date()

    def trading_days_between(self, start: date, end: date) -> list[date]:
        """Inclusive list of trading sessions in ``[start, end]``."""
        if end < start:
            return []
        sessions = self._cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
        return [ts.date() for ts in sessions]

    def sessions_ago(self, d: date, n: int) -> date:
        """``d`` shifted back by ``n`` trading sessions (n >= 0)."""
        cursor = d
        for _ in range(n):
            cursor = self.previous_trading_day(cursor)
        return cursor

    # ------------------------------------------------------------------ "now"
    def latest_expected_session(self, now: datetime | None = None) -> date:
        """The most recent trading session whose end-of-day data should
        already be available, given the current wall-clock time.

        Converts ``now`` to America/New_York, and if "today" (NY-local) is a
        trading session that has not yet closed (plus ``grace_minutes`` to
        allow for provider EOD-data lag), falls back to the previous
        session instead of assuming today's bar already exists.
        """
        if now is None:
            now = datetime.now(UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)

        now_ny = now.astimezone(MARKET_TIMEZONE)
        today_ny = now_ny.date()

        if not self.is_trading_day(today_ny):
            return self.previous_trading_day(today_ny)

        close = self._cal.session_close(pd.Timestamp(today_ny))  # tz-aware UTC Timestamp
        ready_at = close.to_pydatetime() + timedelta(minutes=self.grace_minutes)
        if now >= ready_at:
            return today_ny
        return self.previous_trading_day(today_ny)
