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


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def is_weekend(d: date) -> bool:
    return d.weekday() >= 5  # 5=Sat, 6=Sun


def parse_date(value: str) -> date:
    return date.fromisoformat(value)
