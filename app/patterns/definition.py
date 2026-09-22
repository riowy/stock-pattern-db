"""Canonical pattern definition DSL.

Miner-agnostic rule representation. Definitions freeze into immutable
PatternVersions; material changes create a new version, never overwrite.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


DEFINITION_SCHEMA_VERSION = "1"


class EventMode(StrEnum):
    STATE = "STATE"
    ENTRY_EVENT = "ENTRY_EVENT"
    # Reserved for future sequence / multi-bar event modes.
    SEQUENCE = "SEQUENCE"


class Direction(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class ConditionOp(StrEnum):
    GE = "GE"
    GT = "GT"
    LE = "LE"
    LT = "LT"
    EQ = "EQ"
    BETWEEN = "BETWEEN"
    IN = "IN"
    IS_TRUE = "IS_TRUE"
    IS_FALSE = "IS_FALSE"


class Condition(BaseModel):
    """Atomic indicator/feature predicate."""

    feature: str
    op: ConditionOp
    value: float | int | str | bool | list[float | int | str] | None = None
    value_high: float | int | None = None  # for BETWEEN

    model_config = {"extra": "forbid"}


class BoolExpr(BaseModel):
    """Boolean AND/OR composition of conditions / nested exprs."""

    op: Literal["AND", "OR"]
    children: list[Condition | BoolExpr]

    model_config = {"extra": "forbid"}


class UniverseContract(BaseModel):
    """Eligibility / universe membership that is part of the frozen hypothesis."""

    universe_name: str | None = None
    universe_version: str | None = None
    eligibility_rules: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class PatternDefinition(BaseModel):
    """Frozen statistical hypothesis definition.

    Price floor / regime belong here only when they are part of the hypothesis,
    not when they are merely an audit overlay.
    """

    definition_schema_version: str = DEFINITION_SCHEMA_VERSION
    pattern_version: int = 1
    name: str | None = None
    description: str | None = None
    rules: BoolExpr | Condition
    direction: Direction
    target: str
    horizon: int = Field(ge=1)
    event_mode: EventMode = EventMode.STATE
    cooldown_sessions: int = Field(default=0, ge=0)
    universe: UniverseContract = Field(default_factory=UniverseContract)
    price_floor: float | None = None
    regime: str | None = None
    extras: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    @field_validator("target")
    @classmethod
    def _nonempty_target(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("target must be non-empty")
        return v.strip()


# Resolve self-referential BoolExpr.children union for Pydantic.
BoolExpr.model_rebuild()
PatternDefinition.model_rebuild()
