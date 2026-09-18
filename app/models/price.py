"""Daily price bar model.

Column list matches the spec exactly. Missing/invalid prices are stored as
``None`` -- never coerced to 0.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel


class PriceBar(BaseModel):
    security_id: str
    ticker_at_time: str
    date: date
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    adj_close: float | None = None
    volume: float | None = None
    dividend: float = 0.0
    stock_split: float = 0.0  # 0 = no split event on this date; else the split ratio (e.g. 4.0 = 4-for-1)
    currency: str = "USD"
    provider: str
    retrieved_at: datetime
