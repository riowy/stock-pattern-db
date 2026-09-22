"""Pattern registry service: register, version, transition status."""

from __future__ import annotations

from app.patterns.definition import PatternDefinition
from app.patterns.fingerprint import fingerprints
from app.patterns.lifecycle import PatternStatus, StatusHistoryEntry, assert_transition
from app.patterns.models import PatternVersionRecord
from app.patterns.store import PatternResearchStore, new_id


class PatternRegistry:
    def __init__(self, store: PatternResearchStore) -> None:
        self.store = store

    def register_new(
        self,
        definition: PatternDefinition,
        *,
        pattern_id: str | None = None,
        status: PatternStatus = PatternStatus.DISCOVERED,
        reason: str = "initial discovery",
    ) -> PatternVersionRecord:
        assert_transition(None, status)
        s_fp, h_fp, s_json, h_json = fingerprints(definition)
        pid = pattern_id or new_id("pat_")
        version = definition.pattern_version
        record = PatternVersionRecord(
            pattern_id=pid,
            version=version,
            definition=definition.model_copy(update={"pattern_version": version}),
            structural_fingerprint=s_fp,
            hypothesis_fingerprint=h_fp,
            canonical_structural_json=s_json,
            canonical_hypothesis_json=h_json,
            status=status,
        )
        self.store.insert_pattern_version(record)
        self.store.add_status_history(
            StatusHistoryEntry(
                pattern_id=pid,
                pattern_version=version,
                from_status=None,
                to_status=status,
                reason=reason,
            )
        )
        return record

    def create_new_version(
        self,
        pattern_id: str,
        definition: PatternDefinition,
        *,
        reason: str = "definition change",
    ) -> PatternVersionRecord:
        """Material definition changes create a new version; never overwrite."""
        current = self.store.get_pattern_version(pattern_id)
        if current is None:
            raise KeyError(f"Unknown pattern_id={pattern_id}")
        next_version = current.version + 1
        updated = definition.model_copy(update={"pattern_version": next_version})
        s_fp, h_fp, s_json, h_json = fingerprints(updated)
        record = PatternVersionRecord(
            pattern_id=pattern_id,
            version=next_version,
            definition=updated,
            structural_fingerprint=s_fp,
            hypothesis_fingerprint=h_fp,
            canonical_structural_json=s_json,
            canonical_hypothesis_json=h_json,
            status=PatternStatus.DISCOVERED,
        )
        self.store.insert_pattern_version(record)
        self.store.add_status_history(
            StatusHistoryEntry(
                pattern_id=pattern_id,
                pattern_version=next_version,
                from_status=None,
                to_status=PatternStatus.DISCOVERED,
                reason=reason,
            )
        )
        return record

    def transition(
        self,
        pattern_id: str,
        to_status: PatternStatus,
        *,
        reason: str,
        version: int | None = None,
        rejection_reason: str | None = None,
        changed_by: str | None = None,
    ) -> PatternVersionRecord:
        record = self.store.get_pattern_version(pattern_id, version)
        if record is None:
            raise KeyError(f"Unknown pattern_id={pattern_id}")
        assert_transition(record.status, to_status)
        self.store.update_pattern_status(
            pattern_id,
            record.version,
            to_status,
            rejection_reason=rejection_reason if to_status == PatternStatus.REJECTED else None,
        )
        self.store.add_status_history(
            StatusHistoryEntry(
                pattern_id=pattern_id,
                pattern_version=record.version,
                from_status=record.status,
                to_status=to_status,
                reason=reason,
                changed_by=changed_by,
            )
        )
        updated = self.store.get_pattern_version(pattern_id, record.version)
        assert updated is not None
        return updated

    def get(self, pattern_id: str, version: int | None = None) -> PatternVersionRecord | None:
        return self.store.get_pattern_version(pattern_id, version)

    def find_by_hypothesis(self, hypothesis_fp: str) -> PatternVersionRecord | None:
        return self.store.find_by_hypothesis_fingerprint(hypothesis_fp)

    def list(self, **kwargs):  # noqa: ANN003
        return self.store.list_patterns(**kwargs)
