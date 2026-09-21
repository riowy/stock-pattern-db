"""Explain why a pattern was or was not selected. Does not change mining results."""

from __future__ import annotations

from typing import Any

from app.mining.config import DEFAULT_FDR_Q, INSUFFICIENT_SAMPLE
from app.mining.events import pattern_parts
from app.mining.states import StateSpec

NOT_GENERATED = "NOT_GENERATED"
INSUFFICIENT_SUPPORT = "INSUFFICIENT_SUPPORT"
PAIR_PRUNED = "PAIR_PRUNED"
PAIR_CAP_EXCLUDED = "PAIR_CAP_EXCLUDED"
FDR_FAIL = "FDR_FAIL"
SELECTED = "SELECTED"

# Numeric pair cap is not implemented. Reported so fold5/fold6 zeros are not blamed on a cap.
PAIR_CAP_LIMIT = None


def pair_generation_skip(a: StateSpec, b: StateSpec) -> str | None:
    """Mirror of discover pair filters. Must stay aligned with engine.discover; does not score."""
    if a.source == "feature" or b.source == "feature":
        return "feature_source"
    if a.source == b.source and a.name.split("_")[0] == b.name.split("_")[0] and a.name != b.name:
        return "same_source_prefix"
    return None


def _min_finite(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None and v == v]
    return min(nums) if nums else None


def _lookup_result(results_by_name: dict[str, Any], pattern: str):
    if pattern in results_by_name:
        return results_by_name[pattern], pattern
    parts = pattern_parts(pattern)
    if len(parts) == 2:
        alt = f"{parts[1]} AND {parts[0]}"
        if alt in results_by_name:
            return results_by_name[alt], alt
    return None, pattern


def _parent_status(name: str, generated_singles: set[str], passed_singles: set[str]) -> dict:
    generated = name in generated_singles
    support = name in passed_singles
    if not generated:
        reason = NOT_GENERATED
    elif not support:
        reason = INSUFFICIENT_SUPPORT
    else:
        reason = "SUPPORT_PASSED"
    return {"name": name, "generated": generated, "support_passed": support, "reason": reason}


def classify_pattern(
    pattern: str,
    *,
    results_by_name: dict,
    generated_singles: set[str],
    passed_singles: set[str],
    generated_pairs: set[str],
    insufficient_pairs: set[str],
    specs_by_name: dict[str, StateSpec],
    fdr_threshold: float,
    max_rule_size: int,
) -> dict:
    parts = pattern_parts(pattern)
    is_pair = len(parts) == 2
    row, matched = _lookup_result(results_by_name, pattern)
    generated = False
    support_passed = False
    pruned = False
    cap_excluded = False
    eligible = False
    parent_a = None
    parent_b = None
    raw_p = None
    fdr_q = None
    selected = bool(row.selected) if row is not None else False
    if row is not None:
        raw_p = row.analysis.get("p_value")
        fdr_q = row.fdr_q
        generated = True
        support_passed = row.sample_status == "ok"
        if selected:
            reason = SELECTED
        elif row.sample_status == INSUFFICIENT_SAMPLE:
            reason = INSUFFICIENT_SUPPORT
        else:
            reason = FDR_FAIL
    elif not is_pair:
        generated = pattern in generated_singles
        support_passed = pattern in passed_singles
        reason = NOT_GENERATED if not generated else INSUFFICIENT_SUPPORT
    else:
        a_name, b_name = parts[0], parts[1]
        parent_a = _parent_status(a_name, generated_singles, passed_singles)
        parent_b = _parent_status(b_name, generated_singles, passed_singles)
        alt = f"{b_name} AND {a_name}"
        in_scored = pattern in generated_pairs or alt in generated_pairs
        in_insufficient = pattern in insufficient_pairs or alt in insufficient_pairs
        spec_a = specs_by_name.get(a_name)
        spec_b = specs_by_name.get(b_name)
        skip = pair_generation_skip(spec_a, spec_b) if spec_a is not None and spec_b is not None else None
        both_passed = parent_a["support_passed"] and parent_b["support_passed"]
        eligible = bool(max_rule_size >= 2 and both_passed and skip is None)
        if in_scored or in_insufficient:
            generated = True
            support_passed = in_scored and not in_insufficient
            reason = INSUFFICIENT_SUPPORT if in_insufficient or not support_passed else FDR_FAIL
        elif skip is not None and both_passed:
            pruned = True
            eligible = False
            reason = PAIR_PRUNED
        else:
            reason = NOT_GENERATED

    out = {
        "pattern": pattern,
        "candidate_generated": generated,
        "support_passed": support_passed,
        "candidate_pruned": pruned,
        "raw_p_value": raw_p,
        "fdr_q_value": fdr_q,
        "selection_threshold": fdr_threshold,
        "selected": selected,
        "reason": reason,
        "matched_name": matched,
    }
    if is_pair:
        if parent_a is None:
            parent_a = _parent_status(parts[0], generated_singles, passed_singles)
            parent_b = _parent_status(parts[1], generated_singles, passed_singles)
            spec_a = specs_by_name.get(parts[0])
            spec_b = specs_by_name.get(parts[1])
            skip = pair_generation_skip(spec_a, spec_b) if spec_a is not None and spec_b is not None else None
            both_passed = parent_a["support_passed"] and parent_b["support_passed"]
            eligible = bool(max_rule_size >= 2 and both_passed and skip is None and not pruned)
        out["parent_a"] = parent_a
        out["parent_b"] = parent_b
        out["pair_generation_eligible"] = eligible
        out["pair_cap_excluded"] = cap_excluded
        out["pair_cap_limit"] = PAIR_CAP_LIMIT
    return out


def summarize_selection(
    *,
    n_generated_singles: int,
    n_support_passed_singles: int,
    n_generated_pairs: int,
    n_support_passed_pairs: int,
    pvalues: list[float | None],
    qvalues: list[float | None],
    fdr_threshold: float = DEFAULT_FDR_Q,
    max_rule_size: int,
    results: list,
    generated_singles: list[str],
    passed_singles: list[str],
    generated_pairs: list[str],
    insufficient_pairs: list[str],
    specs: list[StateSpec],
    tracked: tuple[str, ...] | list[str] = (),
) -> dict:
    tests = [p for p in pvalues if p is not None]
    qs = [q for q in qvalues if q is not None]
    results_by_name = {r.pattern: r for r in results}
    specs_by_name = {s.name: s for s in specs}
    tracked_rows = [
        classify_pattern(
            pattern,
            results_by_name=results_by_name,
            generated_singles=set(generated_singles),
            passed_singles=set(passed_singles),
            generated_pairs=set(generated_pairs),
            insufficient_pairs=set(insufficient_pairs),
            specs_by_name=specs_by_name,
            fdr_threshold=fdr_threshold,
            max_rule_size=max_rule_size,
        )
        for pattern in tracked
    ]
    return {
        "n_generated_singles": n_generated_singles,
        "n_support_passed_singles": n_support_passed_singles,
        "n_generated_pairs": n_generated_pairs,
        "n_support_passed_pairs": n_support_passed_pairs,
        "n_statistical_tests": len(tests),
        "min_raw_p": _min_finite(pvalues),
        "min_fdr_q": _min_finite(qvalues),
        "n_q_le_0_10": sum(1 for q in qs if q <= 0.10),
        "n_q_le_0_20": sum(1 for q in qs if q <= 0.20),
        "n_q_le_0_30": sum(1 for q in qs if q <= 0.30),
        "selection_threshold": fdr_threshold,
        "pair_cap_limit": PAIR_CAP_LIMIT,
        "tracked": tracked_rows,
    }


def format_selection_lines(diag: dict) -> list[str]:
    if not diag:
        return []
    cap = diag.get("pair_cap_limit")
    cap_txt = "none" if cap is None else str(cap)
    lines = [
        "Selection diagnostics (does not retune FDR or thresholds)",
        f"  generated singles: {diag.get('n_generated_singles', 0)}",
        f"  support-passed singles: {diag.get('n_support_passed_singles', 0)}",
        f"  generated pairs: {diag.get('n_generated_pairs', 0)}",
        f"  support-passed pairs: {diag.get('n_support_passed_pairs', 0)}",
        f"  statistical tests executed: {diag.get('n_statistical_tests', 0)}",
        f"  min raw p: {_fmt(diag.get('min_raw_p'))}",
        f"  min BH-FDR q: {_fmt(diag.get('min_fdr_q'))}",
        f"  q<=0.10: {diag.get('n_q_le_0_10', 0)}",
        f"  q<=0.20: {diag.get('n_q_le_0_20', 0)}",
        f"  q<=0.30: {diag.get('n_q_le_0_30', 0)}",
        f"  selection_threshold: {_fmt(diag.get('selection_threshold'))} pair_cap: {cap_txt}",
    ]
    for row in diag.get("tracked") or []:
        bits = [
            row.get("pattern", ""),
            f"reason={row.get('reason')}",
            f"candidate_generated={_b(row.get('candidate_generated'))}",
            f"support_passed={_b(row.get('support_passed'))}",
            f"candidate_pruned={_b(row.get('candidate_pruned'))}",
            f"raw_p={_fmt(row.get('raw_p_value'))}",
            f"fdr_q={_fmt(row.get('fdr_q_value'))}",
            f"threshold={_fmt(row.get('selection_threshold'))}",
            f"selected={_b(row.get('selected'))}",
        ]
        if "parent_a" in row:
            pa = row.get("parent_a") or {}
            pb = row.get("parent_b") or {}
            bits.extend(
                [
                    f"parent_a={pa.get('name')}:{pa.get('reason')}",
                    f"parent_b={pb.get('name')}:{pb.get('reason')}",
                    f"pair_generation_eligible={_b(row.get('pair_generation_eligible'))}",
                    f"pair_cap_excluded={_b(row.get('pair_cap_excluded'))}",
                ]
            )
        lines.append("  " + " | ".join(bits))
    return lines


def summarize_from_raw(raw: dict | None, tracked: tuple[str, ...] | list[str] = ()) -> dict | None:
    if not raw:
        return None
    return summarize_selection(tracked=tracked, **{k: v for k, v in raw.items()})


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _b(value: object) -> str:
    return "true" if value else "false"
