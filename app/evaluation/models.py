"""Evaluation / validation registry: separate from frozen pattern definitions."""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.patterns.lifecycle import PatternStatus
from app.patterns.models import PatternVersionRecord
from app.patterns.registry import PatternRegistry
from app.patterns.store import PatternResearchStore, dumps_json, new_id


class SplitRole(StrEnum):
    ANALYSIS = "ANALYSIS"
    VALIDATION = "VALIDATION"
    FUTURE_HOLDOUT = "FUTURE_HOLDOUT"
    WALK_FORWARD = "WALK_FORWARD"
    MONITORING_ROLLING = "MONITORING_ROLLING"
    MONITORING_CUMULATIVE = "MONITORING_CUMULATIVE"


class EvaluationDecision(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    SKIPPED = "SKIPPED"


class EvaluationMetric(BaseModel):
    """Extensible metric row — new names do not require schema rewrite."""

    metric_name: str
    numeric_value: float | None = None
    unit: str | None = None
    aggregation: str | None = None
    horizon: int | None = None
    scope: str | None = None
    period: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Success rule must be explicit when metric is a hit/success rate.
    success_rule: str | None = None

    model_config = {"extra": "forbid"}


class EvaluationRunSpec(BaseModel):
    pattern_id: str
    pattern_version: int
    evaluator_id: str
    evaluator_version: str
    period_start: date
    period_end: date
    split_role: SplitRole
    universe_name: str | None = None
    universe_version: str | None = None
    data_version: str | None = None
    feature_version: str | None = None
    label_version: str | None = None
    code_version: str | None = None
    sample_count: int | None = None
    date_count: int | None = None
    security_count: int | None = None
    decision: EvaluationDecision | None = None
    decision_reason: str | None = None
    thresholds: dict[str, Any] = Field(default_factory=dict)
    metrics: list[EvaluationMetric] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class Evaluator(Protocol):
    """Modular evaluator. Must never retune a frozen PatternVersion."""

    @property
    def evaluator_id(self) -> str: ...

    @property
    def evaluator_version(self) -> str: ...

    def evaluate(
        self,
        pattern: PatternVersionRecord,
        *,
        period_start: date,
        period_end: date,
        split_role: SplitRole,
    ) -> EvaluationRunSpec: ...


class FrozenDefinitionGuard:
    """Ensure evaluation cannot mutate frozen definition / cutoffs."""

    def __init__(self, pattern: PatternVersionRecord) -> None:
        self._pattern_id = pattern.pattern_id
        self._version = pattern.version
        self._hypothesis_fp = pattern.hypothesis_fingerprint
        self._definition_json = pattern.definition.model_dump_json()

    def assert_unchanged(self, pattern: PatternVersionRecord) -> None:
        if pattern.pattern_id != self._pattern_id or pattern.version != self._version:
            raise RuntimeError("Evaluation targeted a different pattern version")
        if pattern.hypothesis_fingerprint != self._hypothesis_fp:
            raise RuntimeError("Frozen hypothesis fingerprint changed during evaluation")
        if pattern.definition.model_dump_json() != self._definition_json:
            raise RuntimeError("Frozen pattern definition mutated during evaluation")


class EvaluationService:
    def __init__(self, store: PatternResearchStore, registry: PatternRegistry) -> None:
        self.store = store
        self.registry = registry

    def record(self, spec: EvaluationRunSpec) -> str:
        pattern = self.store.get_pattern_version(spec.pattern_id, spec.pattern_version)
        if pattern is None:
            raise KeyError(f"Unknown pattern {spec.pattern_id}@{spec.pattern_version}")
        guard = FrozenDefinitionGuard(pattern)

        evaluation_id = new_id("eval_")
        self.store.insert_evaluation_run(
            {
                "evaluation_id": evaluation_id,
                "pattern_id": spec.pattern_id,
                "pattern_version": spec.pattern_version,
                "evaluator_id": spec.evaluator_id,
                "evaluator_version": spec.evaluator_version,
                "period_start": spec.period_start,
                "period_end": spec.period_end,
                "split_role": str(spec.split_role),
                "universe_name": spec.universe_name,
                "universe_version": spec.universe_version,
                "data_version": spec.data_version,
                "feature_version": spec.feature_version,
                "label_version": spec.label_version,
                "code_version": spec.code_version,
                "sample_count": spec.sample_count,
                "date_count": spec.date_count,
                "security_count": spec.security_count,
                "decision": str(spec.decision) if spec.decision else None,
                "decision_reason": spec.decision_reason,
                "thresholds_json": dumps_json(spec.thresholds) if spec.thresholds else None,
                "created_at": datetime.now(UTC),
            }
        )
        for metric in spec.metrics:
            meta = dict(metric.metadata)
            if metric.success_rule:
                meta["success_rule"] = metric.success_rule
            self.store.insert_evaluation_metric(
                {
                    "evaluation_id": evaluation_id,
                    "metric_name": metric.metric_name,
                    "numeric_value": metric.numeric_value,
                    "unit": metric.unit,
                    "aggregation": metric.aggregation,
                    "horizon": metric.horizon,
                    "scope": metric.scope,
                    "period": metric.period,
                    "metadata_json": dumps_json(meta) if meta else None,
                }
            )

        # Re-load and assert frozen definition untouched
        after = self.store.get_pattern_version(spec.pattern_id, spec.pattern_version)
        assert after is not None
        guard.assert_unchanged(after)
        return evaluation_id

    def apply_decision_to_lifecycle(
        self,
        pattern_id: str,
        decision: EvaluationDecision,
        *,
        reason: str,
        version: int | None = None,
    ) -> PatternVersionRecord:
        if decision == EvaluationDecision.PASS:
            return self.registry.transition(pattern_id, PatternStatus.PASSED, reason=reason, version=version)
        if decision == EvaluationDecision.FAIL:
            return self.registry.transition(
                pattern_id,
                PatternStatus.REJECTED,
                reason=reason,
                version=version,
                rejection_reason=reason,
            )
        return self.registry.get(pattern_id, version)  # type: ignore[return-value]
