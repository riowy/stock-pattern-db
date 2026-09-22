"""Pattern registry domain models."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from app.patterns.definition import PatternDefinition
from app.patterns.lifecycle import PatternStatus


class PatternVersionRecord(BaseModel):
    """Immutable versioned pattern definition + current lifecycle status."""

    pattern_id: str
    version: int
    definition: PatternDefinition
    structural_fingerprint: str
    hypothesis_fingerprint: str
    canonical_structural_json: str
    canonical_hypothesis_json: str
    status: PatternStatus = PatternStatus.DISCOVERED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    rejection_reason: str | None = None
    rediscovery_count: int = 0

    model_config = {"extra": "forbid"}


class PatternSummary(BaseModel):
    pattern_id: str
    latest_version: int
    status: PatternStatus
    direction: str
    target: str
    horizon: int
    structural_fingerprint: str
    hypothesis_fingerprint: str
    rediscovery_count: int = 0
    first_discovered_by: str | None = None
    discoverer_ids: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}
