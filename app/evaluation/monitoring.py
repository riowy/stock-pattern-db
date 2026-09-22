"""Continuous monitoring helpers for PASSED/MONITORING patterns."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from app.evaluation.models import EvaluationDecision, EvaluationMetric, EvaluationRunSpec, EvaluationService, SplitRole
from app.patterns.lifecycle import PatternStatus
from app.patterns.registry import PatternRegistry


class PerformanceTrend(StrEnum):
    STABLE = "STABLE"
    WEAKENING = "WEAKENING"
    CHANGED = "CHANGED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


ROLLING_WINDOWS = (20, 50, 100)


def classify_trend(*, recent_hit_rate: float | None, baseline_hit_rate: float | None, drop_threshold: float = 0.15) -> PerformanceTrend:
    if recent_hit_rate is None or baseline_hit_rate is None:
        return PerformanceTrend.INSUFFICIENT_DATA
    if recent_hit_rate < baseline_hit_rate - drop_threshold:
        return PerformanceTrend.WEAKENING
    if abs(recent_hit_rate - baseline_hit_rate) > drop_threshold:
        return PerformanceTrend.CHANGED
    return PerformanceTrend.STABLE


class MonitoringService:
    """Record rolling/cumulative evaluations without deleting patterns."""

    def __init__(self, evaluation: EvaluationService, registry: PatternRegistry) -> None:
        self.evaluation = evaluation
        self.registry = registry

    def record_rolling(
        self,
        pattern_id: str,
        *,
        pattern_version: int,
        period_start: date,
        period_end: date,
        window: int,
        metrics: list[EvaluationMetric],
        decision: EvaluationDecision | None = None,
        decision_reason: str | None = None,
    ) -> str:
        if window not in ROLLING_WINDOWS:
            # Still allow custom windows; core set is documented.
            pass
        return self.evaluation.record(
            EvaluationRunSpec(
                pattern_id=pattern_id,
                pattern_version=pattern_version,
                evaluator_id="monitoring_rolling",
                evaluator_version="1",
                period_start=period_start,
                period_end=period_end,
                split_role=SplitRole.MONITORING_ROLLING,
                decision=decision,
                decision_reason=decision_reason,
                metrics=metrics,
                thresholds={"window": window},
            )
        )

    def maybe_mark_degraded(
        self,
        pattern_id: str,
        *,
        trend: PerformanceTrend,
        reason: str,
    ) -> None:
        record = self.registry.get(pattern_id)
        if record is None:
            raise KeyError(pattern_id)
        if trend != PerformanceTrend.WEAKENING:
            return
        if record.status in (PatternStatus.PASSED, PatternStatus.MONITORING):
            self.registry.transition(pattern_id, PatternStatus.DEGRADED, reason=reason)
