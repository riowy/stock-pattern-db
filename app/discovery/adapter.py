"""Thin adapter: wrap existing MiningEngine proposals into DiscoveryService."""

from __future__ import annotations

from typing import Any

from app.discovery.generators import GeneratorType, GeneratorVersionSpec, hash_configuration
from app.discovery.provenance import DiscoveryRunSpec, DiscoveryService, ProposalResult
from app.patterns.definition import (
    BoolExpr,
    Condition,
    ConditionOp,
    Direction,
    EventMode,
    PatternDefinition,
    UniverseContract,
)


def parse_mining_pattern_string(pattern: str) -> BoolExpr | Condition:
    """Best-effort parse of mining CLI pattern strings like 'rsi14_ge_70 AND volume_ratio20_gt_2'."""
    parts = [p.strip() for p in pattern.split(" AND ") if p.strip()]
    conditions: list[Condition] = []
    for part in parts:
        cond = _parse_atom(part)
        if cond is not None:
            conditions.append(cond)
    if not conditions:
        return Condition(feature=pattern, op=ConditionOp.IS_TRUE, value=True)
    if len(conditions) == 1:
        return conditions[0]
    return BoolExpr(op="AND", children=conditions)


def _parse_atom(atom: str) -> Condition | None:
    for op_token, op in (
        ("_ge_", ConditionOp.GE),
        ("_gt_", ConditionOp.GT),
        ("_le_", ConditionOp.LE),
        ("_lt_", ConditionOp.LT),
        ("_eq_", ConditionOp.EQ),
    ):
        if op_token in atom:
            feature, _, rest = atom.partition(op_token)
            try:
                value: float | str = float(rest)
            except ValueError:
                value = rest
            return Condition(feature=feature, op=op, value=value)
    if atom.endswith("_breakout") or "above" in atom or "below" in atom:
        return Condition(feature=atom, op=ConditionOp.IS_TRUE, value=True)
    return Condition(feature=atom, op=ConditionOp.IS_TRUE, value=True)


def definition_from_mining_result(
    pattern: str,
    *,
    target: str,
    horizon: int = 20,
    direction: Direction = Direction.BULLISH,
    event_mode: str = "state",
    cooldown_sessions: int = 0,
    universe_name: str | None = "research-common-equity-500",
) -> PatternDefinition:
    mode = EventMode.STATE if event_mode.lower() == "state" else EventMode.ENTRY_EVENT
    return PatternDefinition(
        name=pattern,
        rules=parse_mining_pattern_string(pattern),
        direction=direction,
        target=target,
        horizon=horizon,
        event_mode=mode,
        cooldown_sessions=cooldown_sessions,
        universe=UniverseContract(universe_name=universe_name),
    )


def default_mining_generator_spec(*, config: dict[str, Any] | None = None) -> tuple[str, GeneratorVersionSpec]:
    cfg = config or {}
    gid = "mining_engine_v1"
    return gid, GeneratorVersionSpec(
        generator_id=gid,
        version="1",
        implementation_module_id="app.mining.engine.MiningEngine",
        configuration_hash=hash_configuration(cfg),
        settings={k: v for k, v in cfg.items() if "key" not in k.lower() and "secret" not in k.lower()},
    )


def propose_mining_patterns(
    service: DiscoveryService,
    *,
    patterns: list[str],
    target: str,
    generator_id: str,
    generator_version: str,
    event_mode: str = "state",
    cooldown_sessions: int = 0,
    run_meta: dict[str, Any] | None = None,
) -> list[ProposalResult]:
    meta = run_meta or {}
    run_id = service.start_run(
        DiscoveryRunSpec(
            generator_id=generator_id,
            generator_version=generator_version,
            target=target,
            horizon=20,
            candidate_params=meta,
        )
    )
    results: list[ProposalResult] = []
    for pat in patterns:
        definition = definition_from_mining_result(
            pat,
            target=target,
            event_mode=event_mode,
            cooldown_sessions=cooldown_sessions,
        )
        results.append(
            service.propose(
                run_id=run_id,
                generator_id=generator_id,
                generator_version=generator_version,
                definition=definition,
            )
        )
    service.finish_run(run_id, proposed_pattern_count=len(results), candidate_count=len(patterns))
    return results


# Re-export for discoverability
__all__ = [
    "GeneratorType",
    "definition_from_mining_result",
    "default_mining_generator_spec",
    "parse_mining_pattern_string",
    "propose_mining_patterns",
]
