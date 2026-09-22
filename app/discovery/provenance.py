"""Discovery provenance: record every proposal with dedupe classification."""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.patterns.definition import PatternDefinition
from app.patterns.fingerprint import fingerprints
from app.patterns.lifecycle import ACTIVE_VALIDATED, PatternStatus
from app.patterns.models import PatternVersionRecord
from app.patterns.registry import PatternRegistry
from app.patterns.store import PatternResearchStore, dumps_json, new_id


class ProposalKind(StrEnum):
    NEW = "NEW"
    KNOWN_REJECTED_DUPLICATE = "KNOWN_REJECTED_DUPLICATE"
    KNOWN_PASSED_DUPLICATE = "KNOWN_PASSED_DUPLICATE"
    OTHER_DUPLICATE = "OTHER_DUPLICATE"


class DiscoveryRunSpec(BaseModel):
    generator_id: str
    generator_version: str
    analysis_start: date | None = None
    analysis_end: date | None = None
    universe_name: str | None = None
    universe_version: str | None = None
    target: str | None = None
    horizon: int | None = None
    candidate_params: dict[str, Any] = Field(default_factory=dict)
    quantile_fdr: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class ProposalResult(BaseModel):
    event_id: str
    run_id: str
    proposal_kind: ProposalKind
    pattern: PatternVersionRecord
    validation_skipped: bool
    matched_pattern_id: str | None = None
    matched_pattern_version: int | None = None

    model_config = {"extra": "forbid"}


class DiscoveryService:
    """Many-to-many provenance: Generator -> Run -> Event -> PatternVersion."""

    def __init__(self, store: PatternResearchStore, registry: PatternRegistry) -> None:
        self.store = store
        self.registry = registry

    def start_run(self, spec: DiscoveryRunSpec) -> str:
        run_id = new_id("run_")
        self.store.insert_discovery_run(
            {
                "run_id": run_id,
                "generator_id": spec.generator_id,
                "generator_version": spec.generator_version,
                "started_at": datetime.now(UTC),
                "analysis_start": spec.analysis_start,
                "analysis_end": spec.analysis_end,
                "universe_name": spec.universe_name,
                "universe_version": spec.universe_version,
                "target": spec.target,
                "horizon": spec.horizon,
                "candidate_params_json": dumps_json(spec.candidate_params) if spec.candidate_params else None,
                "quantile_fdr_json": dumps_json(spec.quantile_fdr) if spec.quantile_fdr else None,
            }
        )
        return run_id

    def finish_run(
        self,
        run_id: str,
        *,
        runtime_seconds: float | None = None,
        cpu_seconds: float | None = None,
        peak_rss_mb: float | None = None,
        candidate_count: int | None = None,
        supported_candidate_count: int | None = None,
        proposed_pattern_count: int | None = None,
    ) -> None:
        self.store.update_discovery_run(
            run_id,
            finished_at=datetime.now(UTC),
            runtime_seconds=runtime_seconds,
            cpu_seconds=cpu_seconds,
            peak_rss_mb=peak_rss_mb,
            candidate_count=candidate_count,
            supported_candidate_count=supported_candidate_count,
            proposed_pattern_count=proposed_pattern_count,
        )

    def propose(
        self,
        *,
        run_id: str,
        generator_id: str,
        generator_version: str,
        definition: PatternDefinition,
        force_validation: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ProposalResult:
        """Record a proposal. Always creates a DiscoveryEvent.

        If a REJECTED match exists and force_validation is False, classify as
        KNOWN_REJECTED_DUPLICATE and skip expensive validation by default.
        """
        _s_fp, h_fp, _s_json, _h_json = fingerprints(definition)
        existing = self.registry.find_by_hypothesis(h_fp)
        validation_skipped = False
        matched_id: str | None = None
        matched_ver: int | None = None

        if existing is None:
            kind = ProposalKind.NEW
            pattern = self.registry.register_new(definition, reason=f"discovered by {generator_id}")
        else:
            matched_id = existing.pattern_id
            matched_ver = existing.version
            if existing.status == PatternStatus.REJECTED:
                kind = ProposalKind.KNOWN_REJECTED_DUPLICATE
                validation_skipped = not force_validation
                pattern = existing
                self.store.increment_rediscovery(existing.pattern_id, existing.version)
            elif existing.status in ACTIVE_VALIDATED or existing.status == PatternStatus.PASSED:
                kind = ProposalKind.KNOWN_PASSED_DUPLICATE
                pattern = existing
                self.store.increment_rediscovery(existing.pattern_id, existing.version)
            else:
                kind = ProposalKind.OTHER_DUPLICATE
                pattern = existing
                self.store.increment_rediscovery(existing.pattern_id, existing.version)

        # Refresh rediscovery count on returned record
        refreshed = self.store.get_pattern_version(pattern.pattern_id, pattern.version) or pattern

        event_id = new_id("evt_")
        self.store.insert_discovery_event(
            {
                "event_id": event_id,
                "run_id": run_id,
                "generator_id": generator_id,
                "generator_version": generator_version,
                "timestamp": datetime.now(UTC),
                "pattern_id": refreshed.pattern_id,
                "pattern_version": refreshed.version,
                "hypothesis_fingerprint": h_fp,
                "proposal_kind": str(kind),
                "matched_pattern_id": matched_id,
                "matched_pattern_version": matched_ver,
                "validation_skipped": validation_skipped,
                "definition_json": definition.model_dump_json(),
                "metadata_json": dumps_json(metadata) if metadata else None,
            }
        )
        return ProposalResult(
            event_id=event_id,
            run_id=run_id,
            proposal_kind=kind,
            pattern=refreshed,
            validation_skipped=validation_skipped,
            matched_pattern_id=matched_id,
            matched_pattern_version=matched_ver,
        )
