"""Short-sale volume model -- schema placeholder (see spec section 4-F).

No provider currently populates this dataset (see
``app/providers/short_volume/finra_provider.py``). The schema exists now so
that when it is implemented later, the Parquet layout and DuckDB view are
already in place.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel


class ShortVolumeRecord(BaseModel):
    security_id: str
    date: date
    short_volume: float | None = None
    total_volume: float | None = None
    provider: str
    retrieved_at: datetime
