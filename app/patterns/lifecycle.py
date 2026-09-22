"""Pattern lifecycle statuses and status-history records.

FAILED/REJECTED patterns are archived, never deleted.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class PatternStatus(StrEnum):
    DISCOVERED = "DISCOVERED"
    VALIDATING = "VALIDATING"
    PASSED = "PASSED"
    REJECTED = "REJECTED"
    MONITORING = "MONITORING"
    DEGRADED = "DEGRADED"
    RETIRED = "RETIRED"


# Statuses that represent a durable validated identity for rediscovery matching.
ACTIVE_VALIDATED = frozenset({PatternStatus.PASSED, PatternStatus.MONITORING, PatternStatus.DEGRADED})

TERMINAL_ARCHIVE = frozenset({PatternStatus.REJECTED, PatternStatus.RETIRED})


class StatusHistoryEntry(BaseModel):
    pattern_id: str
    pattern_version: int
    from_status: PatternStatus | None
    to_status: PatternStatus
    reason: str
    changed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    changed_by: str | None = None  # human / system actor id

    model_config = {"extra": "forbid"}


ALLOWED_TRANSITIONS: dict[PatternStatus, frozenset[PatternStatus]] = {
    PatternStatus.DISCOVERED: frozenset(
        {PatternStatus.VALIDATING, PatternStatus.REJECTED, PatternStatus.RETIRED}
    ),
    PatternStatus.VALIDATING: frozenset(
        {PatternStatus.PASSED, PatternStatus.REJECTED, PatternStatus.RETIRED}
    ),
    PatternStatus.PASSED: frozenset(
        {PatternStatus.MONITORING, PatternStatus.DEGRADED, PatternStatus.RETIRED, PatternStatus.REJECTED}
    ),
    PatternStatus.MONITORING: frozenset(
        {PatternStatus.DEGRADED, PatternStatus.PASSED, PatternStatus.RETIRED, PatternStatus.REJECTED}
    ),
    PatternStatus.DEGRADED: frozenset(
        {PatternStatus.MONITORING, PatternStatus.PASSED, PatternStatus.RETIRED, PatternStatus.REJECTED}
    ),
    PatternStatus.REJECTED: frozenset({PatternStatus.RETIRED}),  # archive stays; no silent revive
    PatternStatus.RETIRED: frozenset(),
}


def assert_transition(from_status: PatternStatus | None, to_status: PatternStatus) -> None:
    if from_status is None:
        if to_status != PatternStatus.DISCOVERED:
            raise ValueError(f"New patterns must start as DISCOVERED, got {to_status}")
        return
    allowed = ALLOWED_TRANSITIONS.get(from_status, frozenset())
    if to_status not in allowed:
        raise ValueError(f"Illegal status transition {from_status} -> {to_status}")
