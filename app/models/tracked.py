"""Tracked-universe model.

``securities`` = everything we know about from the security master (SEC +
ETF seed list) -- currently ~10k rows.

``tracked_securities`` = the much smaller operational subset we actually
keep collecting data for and running validation against. This is the table
that daily jobs and validation should read by default; ``securities`` alone
must never be treated as "the set of things we track".
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class TrackedSecurity(BaseModel):
    security_id: str
    enabled: bool = True
    tracking_reason: str | None = None
    added_at: datetime
    removed_at: datetime | None = None
    price_tracking: bool = True
    filings_tracking: bool = False
    feature_tracking: bool = False
    notes: str | None = None
