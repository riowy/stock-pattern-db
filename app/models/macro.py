"""FRED macro series observation model.

``realtime_start``/``realtime_end`` are kept so the schema can hold ALFRED
vintage data later without a migration -- for a plain (non-vintage) pull they
simply both equal the observation date.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel


class MacroObservation(BaseModel):
    series_id: str
    date: date
    value: float | None = None
    realtime_start: date | None = None
    realtime_end: date | None = None
    retrieved_at: datetime
    source: str = "FRED"
