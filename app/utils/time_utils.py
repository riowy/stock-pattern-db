"""Time helpers.

Design rule (see README "날짜와 시간"): every timestamp we persist is UTC.
Trading dates (daily bar dates) are stored as plain DATE values representing
the US-exchange local trading date, never as timestamps, so they are immune
to timezone-of-the-server confusion. We deliberately do not hardcode a
scheduling timezone anywhere in this module -- that is a deployment concern
(cron / Task Scheduler), not a library concern.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

# Fixed English names. Never use strftime("%A") / locale.strftime — a Korean
# Windows locale would print the wrong weekday label for a US session date.
ENGLISH_WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def is_weekend(d: date) -> bool:
    return d.weekday() >= 5  # 5=Sat, 6=Sun


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def english_weekday(d: date) -> str:
    """Gregorian weekday in English. Independent of OS locale and timezone."""
    return ENGLISH_WEEKDAYS[d.weekday()]


def coerce_date(value: date | datetime | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    return date.fromisoformat(text[:10])


def format_session_date(value: date | datetime | str | None) -> str:
    """US session calendar date plus English weekday, e.g. ``2026-09-18 (Friday)``.

    Trading dates are stored as naive DATE values for the US exchange session.
    This helper does not convert timezones; it only labels the stored date.
    """
    d = coerce_date(value)
    if d is None:
        return "-"
    return f"{d.isoformat()} ({english_weekday(d)})"
