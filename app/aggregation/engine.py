"""Pluggable aggregation of daily pattern signals into research candidates.

v1 does not lock a permanent scoring formula. Future reliability weighting /
AI aggregation / ML ranking plug in via AggregationEngine.
"""

from __future__ import annotations

from collections import defaultdict
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.signals.models import DailyPatternSignal


class CandidateLabel(StrEnum):
    BUY_CANDIDATE = "BUY_CANDIDATE"
    SELL_CANDIDATE = "SELL_CANDIDATE"
    CONFLICTING = "CONFLICTING"
    WATCH = "WATCH"


class AggregatedCandidate(BaseModel):
    security_id: str
    ticker: str | None = None
    label: CandidateLabel
    bullish_pattern_count: int = 0
    bearish_pattern_count: int = 0
    conflicting: bool = False
    contributing_patterns: list[dict[str, Any]] = Field(default_factory=list)
    reliability_notes: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class AggregationEngine(Protocol):
    def aggregate(self, signals: list[DailyPatternSignal]) -> list[AggregatedCandidate]: ...


class SimpleCountAggregationEngine:
    """First implementation: aggregate evidence by ticker/direction."""

    def aggregate(self, signals: list[DailyPatternSignal]) -> list[AggregatedCandidate]:
        by_sec: dict[str, list[DailyPatternSignal]] = defaultdict(list)
        for s in signals:
            by_sec[s.security_id].append(s)

        out: list[AggregatedCandidate] = []
        for security_id, group in by_sec.items():
            bullish = [s for s in group if s.direction.upper() in ("BULLISH", "LONG", "BUY")]
            bearish = [s for s in group if s.direction.upper() in ("BEARISH", "SHORT", "SELL")]
            conflicting = bool(bullish) and bool(bearish)
            if conflicting:
                label = CandidateLabel.CONFLICTING
            elif bullish and not bearish:
                label = CandidateLabel.BUY_CANDIDATE
            elif bearish and not bullish:
                label = CandidateLabel.SELL_CANDIDATE
            else:
                label = CandidateLabel.WATCH

            ticker = next((s.ticker for s in group if s.ticker), None)
            contributing = [
                {
                    "pattern_id": s.pattern_id,
                    "pattern_version": s.pattern_version,
                    "direction": s.direction,
                    "hit_rate": s.hit_rate,
                    "hit_rate_success_rule": s.hit_rate_success_rule,
                    "median_outcome": s.median_outcome,
                }
                for s in group
            ]
            out.append(
                AggregatedCandidate(
                    security_id=security_id,
                    ticker=ticker,
                    label=label,
                    bullish_pattern_count=len(bullish),
                    bearish_pattern_count=len(bearish),
                    conflicting=conflicting,
                    contributing_patterns=contributing,
                )
            )
        return out
