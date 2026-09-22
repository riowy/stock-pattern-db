"""Synthetic fixtures for tests / demo dashboard. Never seed production state."""

from __future__ import annotations

from datetime import date

from app.discovery.generators import GeneratorRegistry, GeneratorType, GeneratorVersionSpec, hash_configuration
from app.discovery.provenance import DiscoveryRunSpec, DiscoveryService, ProposalKind
from app.evaluation.models import EvaluationDecision, EvaluationMetric, EvaluationRunSpec, EvaluationService, SplitRole
from app.patterns.definition import BoolExpr, Condition, ConditionOp, Direction, EventMode, PatternDefinition, UniverseContract
from app.patterns.lifecycle import PatternStatus
from app.patterns.registry import PatternRegistry
from app.patterns.store import PatternResearchStore
from app.signals.models import DailyPatternSignal, SignalService


def sample_definition(
    *,
    name: str = "rsi14_ge_70",
    target: str = "forward_excess_spy_20d",
    horizon: int = 20,
    direction: Direction = Direction.BEARISH,
) -> PatternDefinition:
    return PatternDefinition(
        name=name,
        rules=Condition(feature="rsi_14", op=ConditionOp.GE, value=70),
        direction=direction,
        target=target,
        horizon=horizon,
        event_mode=EventMode.STATE,
        cooldown_sessions=0,
        universe=UniverseContract(universe_name="research-common-equity-500", universe_version="v1"),
    )


def compound_definition_swapped_order() -> PatternDefinition:
    """Same structure as AND(a,b) with children in reverse order."""
    return PatternDefinition(
        name="pair",
        rules=BoolExpr(
            op="AND",
            children=[
                Condition(feature="volume_ratio_20", op=ConditionOp.GT, value=2),
                Condition(feature="rsi_14", op=ConditionOp.GE, value=70),
            ],
        ),
        direction=Direction.BEARISH,
        target="forward_excess_spy_20d",
        horizon=20,
        event_mode=EventMode.STATE,
        universe=UniverseContract(universe_name="research-common-equity-500"),
    )


def compound_definition_canonical_order() -> PatternDefinition:
    return PatternDefinition(
        name="pair",
        rules=BoolExpr(
            op="AND",
            children=[
                Condition(feature="rsi_14", op=ConditionOp.GE, value=70),
                Condition(feature="volume_ratio_20", op=ConditionOp.GT, value=2),
            ],
        ),
        direction=Direction.BEARISH,
        target="forward_excess_spy_20d",
        horizon=20,
        event_mode=EventMode.STATE,
        universe=UniverseContract(universe_name="research-common-equity-500"),
    )


def seed_fixture_store(store: PatternResearchStore) -> dict:
    """Populate an opened (typically in-memory or temp) store for dashboard/tests."""
    gens = GeneratorRegistry(store)
    registry = PatternRegistry(store)
    discovery = DiscoveryService(store, registry)
    evaluation = EvaluationService(store, registry)
    signals = SignalService(store)

    gens.register(
        generator_id="mine_a",
        name="Mining Engine A",
        generator_type=GeneratorType.MINING_ENGINE,
        version=GeneratorVersionSpec(
            generator_id="mine_a",
            version="1",
            implementation_module_id="app.mining.engine",
            configuration_hash=hash_configuration({"fdr_q": 0.1}),
        ),
    )
    gens.register(
        generator_id="ai_b",
        name="Research AI B",
        generator_type=GeneratorType.AI,
        version=GeneratorVersionSpec(
            generator_id="ai_b",
            version="1",
            implementation_module_id="app.discovery.adapter",
            ai_provider="fixture",
            ai_model="demo-model",
            ai_model_version="1",
            prompt_template_id="tpl_demo",
            temperature=0.0,
        ),
    )

    run_a = discovery.start_run(
        DiscoveryRunSpec(generator_id="mine_a", generator_version="1", target="forward_excess_spy_20d", horizon=20)
    )
    r1 = discovery.propose(
        run_id=run_a,
        generator_id="mine_a",
        generator_version="1",
        definition=sample_definition(),
    )
    r2 = discovery.propose(
        run_id=run_a,
        generator_id="mine_a",
        generator_version="1",
        definition=compound_definition_canonical_order(),
    )
    discovery.finish_run(run_a, runtime_seconds=12.5, cpu_seconds=10.0, proposed_pattern_count=2)

    # Reject first pattern, then rediscover from AI
    registry.transition(
        r1.pattern.pattern_id,
        PatternStatus.VALIDATING,
        reason="start validation",
    )
    registry.transition(
        r1.pattern.pattern_id,
        PatternStatus.REJECTED,
        reason="failed validation NW threshold",
        rejection_reason="failed validation NW threshold",
    )

    run_b = discovery.start_run(
        DiscoveryRunSpec(generator_id="ai_b", generator_version="1", target="forward_excess_spy_20d", horizon=20)
    )
    rediscovered = discovery.propose(
        run_id=run_b,
        generator_id="ai_b",
        generator_version="1",
        definition=sample_definition(),
    )
    assert rediscovered.proposal_kind == ProposalKind.KNOWN_REJECTED_DUPLICATE

    # Second pattern passes
    registry.transition(r2.pattern.pattern_id, PatternStatus.VALIDATING, reason="start validation")
    evaluation.record(
        EvaluationRunSpec(
            pattern_id=r2.pattern.pattern_id,
            pattern_version=r2.pattern.version,
            evaluator_id="fixture_evaluator",
            evaluator_version="1",
            period_start=date(2024, 1, 2),
            period_end=date(2024, 12, 31),
            split_role=SplitRole.VALIDATION,
            decision=EvaluationDecision.PASS,
            decision_reason="median excess above baseline",
            metrics=[
                EvaluationMetric(
                    metric_name="hit_rate",
                    numeric_value=0.58,
                    unit="ratio",
                    success_rule="forward_excess_spy_20d > 0",
                ),
                EvaluationMetric(metric_name="median_return", numeric_value=0.012, unit="return", aggregation="median"),
            ],
        )
    )
    registry.transition(r2.pattern.pattern_id, PatternStatus.PASSED, reason="validation pass")
    registry.transition(r2.pattern.pattern_id, PatternStatus.MONITORING, reason="enter monitoring")

    # AI also proposes the passed pattern (order-normalized duplicate)
    discovery.propose(
        run_id=run_b,
        generator_id="ai_b",
        generator_version="1",
        definition=compound_definition_swapped_order(),
    )
    discovery.finish_run(run_b, runtime_seconds=3.0, cpu_seconds=2.5, proposed_pattern_count=2)

    today = date(2026, 9, 22)
    demo_signals = [
        DailyPatternSignal(
            signal_date=today,
            security_id="sec_aapl",
            ticker="AAPL",
            pattern_id=r2.pattern.pattern_id,
            pattern_version=r2.pattern.version,
            direction="BULLISH",
            expected_horizon=20,
            event_mode="STATE",
            historical_sample_size=1200,
            hit_rate=0.58,
            hit_rate_success_rule="forward_excess_spy_20d > 0",
            median_outcome=0.012,
            typical_loss_when_wrong=-0.04,
            generator_provenance_summary=["mine_a", "ai_b"],
        ),
        DailyPatternSignal(
            signal_date=today,
            security_id="sec_msft",
            ticker="MSFT",
            pattern_id=r2.pattern.pattern_id,
            pattern_version=r2.pattern.version,
            direction="BEARISH",
            expected_horizon=20,
            event_mode="STATE",
            historical_sample_size=1200,
            hit_rate=0.58,
            hit_rate_success_rule="forward_excess_spy_20d > 0",
            median_outcome=0.012,
            typical_loss_when_wrong=-0.04,
            generator_provenance_summary=["mine_a", "ai_b"],
        ),
    ]
    # In-memory store allows signal inserts regardless of production flag.
    signals.persist_signals(demo_signals)

    return {
        "rejected_pattern_id": r1.pattern.pattern_id,
        "passed_pattern_id": r2.pattern.pattern_id,
        "signal_date": today,
    }
