"""Pattern research package: durable pattern assets, fingerprints, registry."""

from __future__ import annotations

from app.patterns.definition import (
    BoolExpr,
    Condition,
    ConditionOp,
    Direction,
    EventMode,
    PatternDefinition,
    UniverseContract,
)
from app.patterns.fingerprint import hypothesis_fingerprint, structural_fingerprint
from app.patterns.lifecycle import PatternStatus
from app.patterns.registry import PatternRegistry
from app.patterns.store import PatternResearchStore

__all__ = [
    "BoolExpr",
    "Condition",
    "ConditionOp",
    "Direction",
    "EventMode",
    "PatternDefinition",
    "PatternRegistry",
    "PatternResearchStore",
    "PatternStatus",
    "UniverseContract",
    "hypothesis_fingerprint",
    "structural_fingerprint",
]
