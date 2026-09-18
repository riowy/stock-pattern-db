from __future__ import annotations

from datetime import UTC, date, datetime

from app.services.market_calendar import MarketCalendarService

CALENDAR = MarketCalendarService()


def test_is_trading_day_weekday() -> None:
    assert CALENDAR.is_trading_day(date(2024, 1, 2)) is True  # Tuesday


def test_is_trading_day_weekend() -> None:
    assert CALENDAR.is_trading_day(date(2024, 1, 6)) is False  # Saturday
    assert CALENDAR.is_trading_day(date(2024, 1, 7)) is False  # Sunday


def test_is_trading_day_us_holiday() -> None:
    # New Year's Day and MLK Day are NYSE holidays, not just "a Monday".
    assert CALENDAR.is_trading_day(date(2024, 1, 1)) is False
    assert CALENDAR.is_trading_day(date(2024, 1, 15)) is False  # MLK Day


def test_previous_trading_day_skips_weekend() -> None:
    assert CALENDAR.previous_trading_day(date(2024, 1, 8)) == date(2024, 1, 5)  # Mon -> Fri


def test_previous_trading_day_skips_holiday() -> None:
    assert CALENDAR.previous_trading_day(date(2024, 1, 16)) == date(2024, 1, 12)  # after MLK -> prior Fri


def test_next_trading_day_skips_weekend() -> None:
    assert CALENDAR.next_trading_day(date(2024, 1, 13)) == date(2024, 1, 16)  # Sat -> Tue (Mon is MLK)


def test_trading_days_between_excludes_weekends_and_holidays() -> None:
    days = CALENDAR.trading_days_between(date(2024, 1, 1), date(2024, 1, 8))
    assert date(2024, 1, 1) not in days  # holiday
    assert date(2024, 1, 6) not in days  # Saturday
    assert date(2024, 1, 7) not in days  # Sunday
    assert date(2024, 1, 2) in days
    assert date(2024, 1, 8) in days


def test_sessions_ago() -> None:
    # 1 session before Tuesday 2024-01-09 is Monday 2024-01-08.
    assert CALENDAR.sessions_ago(date(2024, 1, 9), 1) == date(2024, 1, 8)


def test_latest_expected_session_on_saturday_is_friday() -> None:
    now = datetime(2024, 1, 6, 10, 0, tzinfo=UTC)
    assert CALENDAR.latest_expected_session(now) == date(2024, 1, 5)


def test_latest_expected_session_korea_early_monday_is_still_friday() -> None:
    # 2024-01-08 02:00 UTC = 11:00 KST Monday, but it's still Sunday 21:00
    # in New York -- the US market's Monday session has not started yet.
    now = datetime(2024, 1, 8, 2, 0, tzinfo=UTC)
    assert CALENDAR.latest_expected_session(now) == date(2024, 1, 5)


def test_latest_expected_session_before_close_plus_grace_falls_back() -> None:
    # Market closes at 21:00 UTC on 2024-01-02; grace period is 120 min by
    # default, so 20:00 UTC (before close) must fall back to the prior session.
    now = datetime(2024, 1, 2, 20, 0, tzinfo=UTC)
    assert CALENDAR.latest_expected_session(now) == date(2023, 12, 29)


def test_latest_expected_session_after_close_plus_grace() -> None:
    now = datetime(2024, 1, 2, 23, 30, tzinfo=UTC)  # 21:00 close + 120min grace = 23:00
    assert CALENDAR.latest_expected_session(now) == date(2024, 1, 2)


def test_grace_period_is_configurable() -> None:
    short_grace = MarketCalendarService(grace_minutes=0)
    now = datetime(2024, 1, 2, 21, 5, tzinfo=UTC)  # just after close, no grace
    assert short_grace.latest_expected_session(now) == date(2024, 1, 2)
