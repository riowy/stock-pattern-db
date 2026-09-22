"""Generator performance and pairwise overlap metrics.

Do NOT collapse into one opaque score. Expose underlying metrics so a human
can decide whether an engine is useful (including as an early filter).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from app.patterns.store import PatternResearchStore


@dataclass(frozen=True)
class GeneratorMetrics:
    generator_id: str
    generator_version: str | None
    total_discovery_runs: int
    total_proposals: int
    unique_patterns_proposed: int
    new_pattern_count: int
    new_pattern_rate: float
    duplicate_count: int
    duplicate_rate: float
    known_rejected_rediscovery_count: int
    known_rejected_rediscovery_rate: float
    known_passed_rediscovery_count: int
    known_passed_rediscovery_rate: float
    patterns_sent_to_validation: int
    passed_count: int
    passed_rate: float
    rejected_count: int
    rejected_rate: float
    monitoring_count: int
    degraded_count: int
    unique_passed_contribution: int
    rejected_only_output_rate: float
    total_runtime_seconds: float
    total_cpu_seconds: float
    peak_rss_mb_max: float | None
    passed_per_compute_hour: float | None
    validations_avoided: int
    estimated_runtime_saved_seconds: float | None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class PairwiseOverlap:
    generator_a: str
    generator_b: str
    shared_pattern_count: int
    jaccard_proposed: float
    overlap_rejected: int
    overlap_passed: int
    unique_accepted_a: int
    unique_accepted_b: int
    co_discovery_count: int
    redundant_discovery_rate: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _safe_rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def compute_generator_metrics(
    store: PatternResearchStore,
    generator_id: str,
    *,
    generator_version: str | None = None,
    mean_validation_seconds: float | None = None,
) -> GeneratorMetrics:
    events = store.list_discovery_events(generator_id=generator_id)
    if generator_version:
        events = [e for e in events if e["generator_version"] == generator_version]

    runs = store.list_discovery_runs(generator_id=generator_id)
    if generator_version:
        runs = [r for r in runs if r["generator_version"] == generator_version]

    total_proposals = len(events)
    pattern_ids = {e["pattern_id"] for e in events if e.get("pattern_id")}
    kinds = [e["proposal_kind"] for e in events]
    new_c = sum(1 for k in kinds if k == "NEW")
    rej_dup = sum(1 for k in kinds if k == "KNOWN_REJECTED_DUPLICATE")
    pass_dup = sum(1 for k in kinds if k == "KNOWN_PASSED_DUPLICATE")
    dup_c = total_proposals - new_c
    validations_avoided = sum(1 for e in events if e.get("validation_skipped"))

    # Downstream pattern outcomes for patterns this generator proposed
    passed = rejected = monitoring = degraded = 0
    sent_to_validation = 0
    for pid in pattern_ids:
        rec = store.get_pattern_version(pid)
        if rec is None:
            continue
        st = str(rec.status)
        if st in ("PASSED", "MONITORING", "DEGRADED", "REJECTED"):
            sent_to_validation += 1
        if st == "PASSED":
            passed += 1
        elif st == "REJECTED":
            rejected += 1
        elif st == "MONITORING":
            monitoring += 1
        elif st == "DEGRADED":
            degraded += 1

    # Unique PASSED contribution: passed patterns not proposed by any other generator
    unique_passed = 0
    for pid in pattern_ids:
        rec = store.get_pattern_version(pid)
        if rec is None or str(rec.status) != "PASSED":
            continue
        proposers = {
            e["generator_id"]
            for e in store.list_discovery_events(pattern_id=pid)
            if e.get("generator_id")
        }
        if proposers == {generator_id}:
            unique_passed += 1

    rejected_only_rate = _safe_rate(rejected, len(pattern_ids)) if pattern_ids else 0.0
    # If all proposed patterns are rejected and none passed: rejected-only output
    if pattern_ids and passed == 0 and monitoring == 0 and degraded == 0 and rejected == len(pattern_ids):
        rejected_only_rate = 1.0

    total_runtime = sum(float(r["runtime_seconds"] or 0) for r in runs)
    total_cpu = sum(float(r["cpu_seconds"] or 0) for r in runs)
    rss_vals = [float(r["peak_rss_mb"]) for r in runs if r.get("peak_rss_mb") is not None]
    peak_rss = max(rss_vals) if rss_vals else None

    compute_hours = total_cpu / 3600.0 if total_cpu else (total_runtime / 3600.0 if total_runtime else None)
    passed_per_hour = (passed / compute_hours) if compute_hours and compute_hours > 0 else None

    est_saved = None
    if mean_validation_seconds is not None:
        est_saved = validations_avoided * mean_validation_seconds

    return GeneratorMetrics(
        generator_id=generator_id,
        generator_version=generator_version,
        total_discovery_runs=len(runs),
        total_proposals=total_proposals,
        unique_patterns_proposed=len(pattern_ids),
        new_pattern_count=new_c,
        new_pattern_rate=_safe_rate(new_c, total_proposals),
        duplicate_count=dup_c,
        duplicate_rate=_safe_rate(dup_c, total_proposals),
        known_rejected_rediscovery_count=rej_dup,
        known_rejected_rediscovery_rate=_safe_rate(rej_dup, total_proposals),
        known_passed_rediscovery_count=pass_dup,
        known_passed_rediscovery_rate=_safe_rate(pass_dup, total_proposals),
        patterns_sent_to_validation=sent_to_validation,
        passed_count=passed,
        passed_rate=_safe_rate(passed, len(pattern_ids)),
        rejected_count=rejected,
        rejected_rate=_safe_rate(rejected, len(pattern_ids)),
        monitoring_count=monitoring,
        degraded_count=degraded,
        unique_passed_contribution=unique_passed,
        rejected_only_output_rate=rejected_only_rate,
        total_runtime_seconds=total_runtime,
        total_cpu_seconds=total_cpu,
        peak_rss_mb_max=peak_rss,
        passed_per_compute_hour=passed_per_hour,
        validations_avoided=validations_avoided,
        estimated_runtime_saved_seconds=est_saved,
    )


def _pattern_set(store: PatternResearchStore, generator_id: str) -> set[str]:
    return {e["pattern_id"] for e in store.list_discovery_events(generator_id=generator_id) if e.get("pattern_id")}


def _status_subset(store: PatternResearchStore, pattern_ids: set[str], status: str) -> set[str]:
    out: set[str] = set()
    for pid in pattern_ids:
        rec = store.get_pattern_version(pid)
        if rec is not None and str(rec.status) == status:
            out.add(pid)
    return out


def compute_pairwise_overlap(store: PatternResearchStore, generator_a: str, generator_b: str) -> PairwiseOverlap:
    a = _pattern_set(store, generator_a)
    b = _pattern_set(store, generator_b)
    shared = a & b
    union = a | b
    jaccard = _safe_rate(len(shared), len(union))

    a_rej = _status_subset(store, a, "REJECTED")
    b_rej = _status_subset(store, b, "REJECTED")
    a_pass = _status_subset(store, a, "PASSED")
    b_pass = _status_subset(store, b, "PASSED")

    # Co-discovery: patterns that appear in discovery events from both
    co = len(shared)
    # Redundant: shared / max(|A|,|B|) style rate of rediscovery overlap
    redundant = _safe_rate(len(shared), max(len(a), len(b)))

    return PairwiseOverlap(
        generator_a=generator_a,
        generator_b=generator_b,
        shared_pattern_count=len(shared),
        jaccard_proposed=jaccard,
        overlap_rejected=len(a_rej & b_rej),
        overlap_passed=len(a_pass & b_pass),
        unique_accepted_a=len(a_pass - b_pass),
        unique_accepted_b=len(b_pass - a_pass),
        co_discovery_count=co,
        redundant_discovery_rate=redundant,
    )


def all_pairwise_overlaps(store: PatternResearchStore) -> list[PairwiseOverlap]:
    gens = [g["generator_id"] for g in store.list_generators()]
    out: list[PairwiseOverlap] = []
    for i, a in enumerate(gens):
        for b in gens[i + 1 :]:
            out.append(compute_pairwise_overlap(store, a, b))
    return out


def proposer_map(store: PatternResearchStore) -> dict[str, list[str]]:
    """pattern_id -> ordered unique generator_ids that proposed it."""
    mapping: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for e in store.list_discovery_events():
        pid = e.get("pattern_id")
        gid = e.get("generator_id")
        if not pid or not gid:
            continue
        if gid not in seen[pid]:
            seen[pid].add(gid)
            mapping[pid].append(gid)
    return dict(mapping)
