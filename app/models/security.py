"""Security master models.

``ticker`` is explicitly NOT treated as a permanent identifier. The permanent
internal key is ``security_id``; CIK (when available) is the primary durable
external identifier. Ticker/company-name history is tracked separately in
``SecurityIdentifier`` / ``SecuritySnapshot`` so symbol changes never
overwrite history.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class Security(BaseModel):
    security_id: str
    cik: str | None = None
    company_name: str | None = None
    primary_ticker: str | None = None
    exchange: str | None = None
    asset_type: str = Field(default="EQUITY")  # EQUITY | ETF | UNKNOWN
    currency: str = Field(default="USD")
    is_active: bool = True
    first_seen_at: datetime
    last_seen_at: datetime
    created_at: datetime
    updated_at: datetime


class SecurityIdentifier(BaseModel):
    security_id: str
    identifier_type: str  # TICKER | CIK | ISIN | ...
    identifier_value: str
    valid_from: date
    valid_to: date | None = None
    source: str


class SecuritySnapshot(BaseModel):
    snapshot_date: date
    security_id: str
    ticker: str | None = None
    company_name: str | None = None
    exchange: str | None = None
    source: str
