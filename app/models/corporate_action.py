"""Corporate action model. Kept strictly separate from raw OHLC price bars."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel


class ActionType(StrEnum):
    DIVIDEND = "DIVIDEND"
    SPLIT = "SPLIT"
    # Reserved for future use -- see spec section 8.
    MERGER = "MERGER"
    SPINOFF = "SPINOFF"
    SYMBOL_CHANGE = "SYMBOL_CHANGE"
    DELISTING = "DELISTING"


class CorporateAction(BaseModel):
    security_id: str
    effective_date: date
    action_type: ActionType
    value: float
    provider: str
    retrieved_at: datetime
