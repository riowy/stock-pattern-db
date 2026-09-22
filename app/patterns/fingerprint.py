"""Deterministic canonical JSON and fingerprints for PatternDefinition."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.patterns.definition import BoolExpr, Condition, PatternDefinition, UniverseContract


def _normalize_condition(c: Condition) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "feature": c.feature,
        "op": str(c.op),
        "value": c.value,
    }
    if c.value_high is not None:
        payload["value_high"] = c.value_high
    return payload


def _normalize_rules(node: BoolExpr | Condition) -> dict[str, Any]:
    if isinstance(node, Condition):
        return {"type": "condition", **_normalize_condition(node)}
    # Sort children for order-invariant structural equality.
    children = [_normalize_rules(ch) for ch in node.children]
    children.sort(key=lambda d: json.dumps(d, sort_keys=True, separators=(",", ":")))
    return {"type": "bool", "op": node.op, "children": children}


def _normalize_universe(u: UniverseContract) -> dict[str, Any]:
    rules = sorted(u.eligibility_rules)
    return {
        "universe_name": u.universe_name,
        "universe_version": u.universe_version,
        "eligibility_rules": rules,
    }


def structural_payload(definition: PatternDefinition) -> dict[str, Any]:
    """Rule structure only (conditions / boolean composition)."""
    return {
        "definition_schema_version": definition.definition_schema_version,
        "rules": _normalize_rules(definition.rules),
    }


def hypothesis_payload(definition: PatternDefinition) -> dict[str, Any]:
    """Fields that change the statistical hypothesis."""
    return {
        **structural_payload(definition),
        "direction": str(definition.direction),
        "target": definition.target,
        "horizon": definition.horizon,
        "event_mode": str(definition.event_mode),
        "cooldown_sessions": definition.cooldown_sessions,
        "universe": _normalize_universe(definition.universe),
        "price_floor": definition.price_floor,
        "regime": definition.regime,
    }


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def structural_fingerprint(definition: PatternDefinition) -> str:
    return _sha256(canonical_json(structural_payload(definition)))


def hypothesis_fingerprint(definition: PatternDefinition) -> str:
    return _sha256(canonical_json(hypothesis_payload(definition)))


def fingerprints(definition: PatternDefinition) -> tuple[str, str, str, str]:
    """Return (structural_fp, hypothesis_fp, structural_json, hypothesis_json)."""
    s_json = canonical_json(structural_payload(definition))
    h_json = canonical_json(hypothesis_payload(definition))
    return _sha256(s_json), _sha256(h_json), s_json, h_json
