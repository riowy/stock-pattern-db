"""Cboe VIX daily bar model."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel


class VixBar(BaseModel):
    date: date
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    provider: str = "cboe"
    retrieved_at: datetime
