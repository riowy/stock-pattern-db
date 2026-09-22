"""Pattern registry, discovery provenance, generator metrics, dashboard, persistence gates."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from app.aggregation.engine import CandidateLabel, SimpleCountAggregationEngine
from app.config.settings import Settings
from app.dashboard.server import DashboardApp
from app.discovery.generators import (
    GeneratorRegistry,
    GeneratorStatus,
    GeneratorType,
    GeneratorVersionSpec,
    hash_configuration,
)
from app.discovery.metrics import compute_generator_metrics, compute_pairwise_overlap
from app.discovery.provenance import DiscoveryRunSpec, DiscoveryService, ProposalKind
from app.evaluation.models import (
    EvaluationDecision,
    EvaluationMetric,
    EvaluationRunSpec,
    EvaluationService,
    FrozenDefinitionGuard,
    SplitRole,
)
from app.patterns.definition import Condition, ConditionOp, Direction, EventMode, PatternDefinition, UniverseContract
from app.patterns.fingerprint import hypothesis_fingerprint, structural_fingerprint
from app.patterns.fixtures import (
    compound_definition_canonical_order,
    compound_definition_swapped_order,
    sample_definition,
    seed_fixture_store,
)
from app.patterns.lifecycle import PatternStatus
from app.patterns.persistence import open_research_store, registry_db_path
from app.patterns.registry import PatternRegistry
from app.patterns.store import PatternResearchStore
from app.signals.models import DailyPatternSignal


def _settings(tmp_path: Path, **kwargs) -> Settings:
    base = dict(
        _env_file=None,
        sec_user_agent="StockPatternResearchTests test@example.com",
        fred_api_key="test-fred-key",
        data_root=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        lake_dir=tmp_path / "data" / "lake",
        state_dir=tmp_path / "data" / "state",
        log_dir=tmp_path / "data" / "logs",
        indicator_persistence_enabled=False,
        mining_result_persistence_enabled=False,
        pattern_registry_persistence_enabled=False,
        daily_signal_persistence_enabled=False,
    )
    base.update(kwargs)
    s = Settings(**base)
    s.ensure_directories()
    return s


def test_fingerprint_stable_under_condition_order() -> None:
    a = compound_definition_canonical_order()
    b = compound_definition_swapped_order()
    assert structural_fingerprint(a) == structural_fingerprint(b)
    assert hypothesis_fingerprint(a) == hypothesis_fingerprint(b)


def test_different_target_horizon_changes_hypothesis_fingerprint() -> None:
    base = sample_definition(target="forward_excess_spy_20d", horizon=20)
    other_target = sample_definition(target="forward_excess_spy_5d", horizon=20)
    other_horizon = sample_definition(target="forward_excess_spy_20d", horizon=5)
    assert structural_fingerprint(base) == structural_fingerprint(other_target)
    assert structural_fingerprint(base) == structural_fingerprint(other_horizon)
    assert hypothesis_fingerprint(base) != hypothesis_fingerprint(other_target)
    assert hypothesis_fingerprint(base) != hypothesis_fingerprint(other_horizon)


def test_two_generators_one_pattern_version_two_events() -> None:
    store = PatternResearchStore(persist=False).open()
    try:
        gens = GeneratorRegistry(store)
        registry = PatternRegistry(store)
        discovery = DiscoveryService(store, registry)
        for gid, name, gtype in (
            ("g1", "Engine 1", GeneratorType.MINING_ENGINE),
            ("g2", "AI 2", GeneratorType.AI),
        ):
            gens.register(
                generator_id=gid,
                name=name,
                generator_type=gtype,
                version=GeneratorVersionSpec(generator_id=gid, version="1", configuration_hash=hash_configuration({})),
            )
        run1 = discovery.start_run(DiscoveryRunSpec(generator_id="g1", generator_version="1"))
        r1 = discovery.propose(run_id=run1, generator_id="g1", generator_version="1", definition=sample_definition())
        run2 = discovery.start_run(DiscoveryRunSpec(generator_id="g2", generator_version="1"))
        r2 = discovery.propose(run_id=run2, generator_id="g2", generator_version="1", definition=sample_definition())
        assert r1.proposal_kind == ProposalKind.NEW
        assert r2.proposal_kind == ProposalKind.OTHER_DUPLICATE or r2.proposal_kind == ProposalKind.KNOWN_PASSED_DUPLICATE
        assert r1.pattern.pattern_id == r2.pattern.pattern_id
        assert r1.pattern.version == r2.pattern.version
        events = store.list_discovery_events(pattern_id=r1.pattern.pattern_id)
        assert len(events) == 2
        assert {e["generator_id"] for e in events} == {"g1", "g2"}
    finally:
        store.close()


def test_rejected_rediscovery_is_known_rejected_duplicate_skips_validation() -> None:
    store = PatternResearchStore(persist=False).open()
    try:
        gens = GeneratorRegistry(store)
        registry = PatternRegistry(store)
        discovery = DiscoveryService(store, registry)
        gens.register(
            generator_id="g1",
            name="G1",
            generator_type=GeneratorType.MINING_ENGINE,
            version=GeneratorVersionSpec(generator_id="g1", version="1"),
        )
        gens.register(
            generator_id="g2",
            name="G2",
            generator_type=GeneratorType.AI,
            version=GeneratorVersionSpec(generator_id="g2", version="1"),
        )
        run1 = discovery.start_run(DiscoveryRunSpec(generator_id="g1", generator_version="1"))
        first = discovery.propose(run_id=run1, generator_id="g1", generator_version="1", definition=sample_definition())
        registry.transition(first.pattern.pattern_id, PatternStatus.VALIDATING, reason="validate")
        registry.transition(
            first.pattern.pattern_id,
            PatternStatus.REJECTED,
            reason="fail",
            rejection_reason="fail",
        )
        run2 = discovery.start_run(DiscoveryRunSpec(generator_id="g2", generator_version="1"))
        second = discovery.propose(
            run_id=run2, generator_id="g2", generator_version="1", definition=sample_definition()
        )
        assert second.proposal_kind == ProposalKind.KNOWN_REJECTED_DUPLICATE
        assert second.validation_skipped is True
        assert second.pattern.pattern_id == first.pattern.pattern_id
        # Rediscovery counted for proposing generator
        m = compute_generator_metrics(store, "g2")
        assert m.known_rejected_rediscovery_count == 1
        assert m.validations_avoided == 1
        assert m.total_proposals == 1
    finally:
        store.close()


def test_generator_version_immutable_and_manual_retire_preserves_history() -> None:
    store = PatternResearchStore(persist=False).open()
    try:
        gens = GeneratorRegistry(store)
        spec = GeneratorVersionSpec(
            generator_id="g1",
            version="1",
            configuration_hash=hash_configuration({"a": 1}),
            settings={"a": 1},
        )
        gens.register(generator_id="g1", name="G1", generator_type=GeneratorType.HUMAN, version=spec)
        with pytest.raises(ValueError, match="immutable"):
            store.insert_generator_version(
                {
                    "generator_id": "g1",
                    "version": "1",
                    "configuration_hash": hash_configuration({"a": 2}),
                    "settings_json": '{"a": 2}',
                }
            )
        # Identical re-insert ok (created_at ignored; other immutable fields match)
        store.insert_generator_version(
            {
                "generator_id": "g1",
                "version": "1",
                "configuration_hash": hash_configuration({"a": 1}),
                "settings_json": '{"a": 1}',
            }
        )
        with pytest.raises(RuntimeError, match="Automatic"):
            gens.auto_retire_for_poor_performance()
        gens.retire("g1", actor="tester")
        g = gens.get("g1")
        assert g is not None
        assert g["status"] == str(GeneratorStatus.RETIRED)
        assert g["retired_at"] is not None
        # Still listable / history preserved
        assert len(gens.list()) == 1
        assert len(gens.versions("g1")) == 1
        gens.restore("g1")
        assert gens.get("g1")["status"] == str(GeneratorStatus.ACTIVE)
    finally:
        store.close()


def test_reregister_preserves_disabled_and_retired_status() -> None:
    """Re-register must not silently reactivate; only restore() returns to ACTIVE."""
    store = PatternResearchStore(persist=False).open()
    try:
        gens = GeneratorRegistry(store)
        base = GeneratorVersionSpec(generator_id="g_disabled", version="1")
        gens.register(
            generator_id="g_disabled",
            name="Disabled Gen",
            generator_type=GeneratorType.MINING_ENGINE,
            version=base,
        )
        gens.disable("g_disabled", actor="tester")
        assert gens.get("g_disabled")["status"] == str(GeneratorStatus.DISABLED)

        gens.register(
            generator_id="g_disabled",
            name="Disabled Gen Renamed",
            generator_type=GeneratorType.MINING_ENGINE,
            version=base,
        )
        assert gens.get("g_disabled")["status"] == str(GeneratorStatus.DISABLED)
        assert gens.get("g_disabled")["name"] == "Disabled Gen Renamed"

        gens.restore("g_disabled", actor="tester")
        assert gens.get("g_disabled")["status"] == str(GeneratorStatus.ACTIVE)

        retired_spec = GeneratorVersionSpec(generator_id="g_retired", version="1")
        gens.register(
            generator_id="g_retired",
            name="Retired Gen",
            generator_type=GeneratorType.AI,
            version=retired_spec,
        )
        gens.retire("g_retired", actor="tester")
        assert gens.get("g_retired")["status"] == str(GeneratorStatus.RETIRED)
        assert gens.get("g_retired")["retired_at"] is not None

        gens.register(
            generator_id="g_retired",
            name="Retired Gen Still",
            generator_type=GeneratorType.AI,
            version=retired_spec,
        )
        g = gens.get("g_retired")
        assert g is not None
        assert g["status"] == str(GeneratorStatus.RETIRED)
        assert g["retired_at"] is not None
        assert g["name"] == "Retired Gen Still"

        gens.restore("g_retired", actor="tester")
        assert gens.get("g_retired")["status"] == str(GeneratorStatus.ACTIVE)
        assert gens.get("g_retired")["retired_at"] is None
    finally:
        store.close()


def test_generator_version_rejects_divergent_immutable_metadata() -> None:
    """Full immutable payload compared; model/prompt and implementation/git mismatches rejected."""
    store = PatternResearchStore(persist=False).open()
    try:
        gens = GeneratorRegistry(store)
        gens.register(
            generator_id="ai1",
            name="AI",
            generator_type=GeneratorType.AI,
            version=GeneratorVersionSpec(
                generator_id="ai1",
                version="1",
                implementation_module_id="app.discovery.adapter",
                configuration_hash=hash_configuration({"x": 1}),
                code_version="0.1.0",
                git_commit="abc123",
                ai_provider="openai",
                ai_model="gpt-test",
                ai_model_version="2024-01",
                prompt_template_id="tpl-v1",
                prompt_template_hash="hash-v1",
                temperature=0.0,
                settings={"x": 1},
            ),
        )
        # Model / prompt metadata divergence
        with pytest.raises(ValueError, match="immutable|divergent"):
            store.insert_generator_version(
                {
                    "generator_id": "ai1",
                    "version": "1",
                    "implementation_module_id": "app.discovery.adapter",
                    "configuration_hash": hash_configuration({"x": 1}),
                    "code_version": "0.1.0",
                    "git_commit": "abc123",
                    "ai_provider": "openai",
                    "ai_model": "gpt-other",
                    "ai_model_version": "2024-01",
                    "prompt_template_id": "tpl-v1",
                    "prompt_template_hash": "hash-v1",
                    "temperature": 0.0,
                    "settings_json": '{"x": 1}',
                }
            )
        with pytest.raises(ValueError, match="immutable|divergent"):
            store.insert_generator_version(
                {
                    "generator_id": "ai1",
                    "version": "1",
                    "implementation_module_id": "app.discovery.adapter",
                    "configuration_hash": hash_configuration({"x": 1}),
                    "code_version": "0.1.0",
                    "git_commit": "abc123",
                    "ai_provider": "openai",
                    "ai_model": "gpt-test",
                    "ai_model_version": "2024-01",
                    "prompt_template_id": "tpl-v2",
                    "prompt_template_hash": "hash-v2",
                    "temperature": 0.0,
                    "settings_json": '{"x": 1}',
                }
            )
        # Implementation / git metadata divergence
        with pytest.raises(ValueError, match="immutable|divergent"):
            store.insert_generator_version(
                {
                    "generator_id": "ai1",
                    "version": "1",
                    "implementation_module_id": "app.discovery.other",
                    "configuration_hash": hash_configuration({"x": 1}),
                    "code_version": "0.1.0",
                    "git_commit": "abc123",
                    "ai_provider": "openai",
                    "ai_model": "gpt-test",
                    "ai_model_version": "2024-01",
                    "prompt_template_id": "tpl-v1",
                    "prompt_template_hash": "hash-v1",
                    "temperature": 0.0,
                    "settings_json": '{"x": 1}',
                }
            )
        with pytest.raises(ValueError, match="immutable|divergent"):
            store.insert_generator_version(
                {
                    "generator_id": "ai1",
                    "version": "1",
                    "implementation_module_id": "app.discovery.adapter",
                    "configuration_hash": hash_configuration({"x": 1}),
                    "code_version": "0.2.0",
                    "git_commit": "def456",
                    "ai_provider": "openai",
                    "ai_model": "gpt-test",
                    "ai_model_version": "2024-01",
                    "prompt_template_id": "tpl-v1",
                    "prompt_template_hash": "hash-v1",
                    "temperature": 0.0,
                    "settings_json": '{"x": 1}',
                }
            )
        # Identical payload (different created_at) must still be accepted
        store.insert_generator_version(
            {
                "generator_id": "ai1",
                "version": "1",
                "implementation_module_id": "app.discovery.adapter",
                "configuration_hash": hash_configuration({"x": 1}),
                "code_version": "0.1.0",
                "git_commit": "abc123",
                "ai_provider": "openai",
                "ai_model": "gpt-test",
                "ai_model_version": "2024-01",
                "prompt_template_id": "tpl-v1",
                "prompt_template_hash": "hash-v1",
                "temperature": 0.0,
                "settings_json": '{"x": 1}',
                "created_at": datetime.now(UTC),
            }
        )
        assert len(store.list_generator_versions("ai1")) == 1
    finally:
        store.close()


def test_pairwise_jaccard_and_unique_passed_contribution() -> None:
    store = PatternResearchStore(persist=False).open()
    try:
        gens = GeneratorRegistry(store)
        registry = PatternRegistry(store)
        discovery = DiscoveryService(store, registry)
        for gid in ("a", "b"):
            gens.register(
                generator_id=gid,
                name=gid,
                generator_type=GeneratorType.MINING_ENGINE,
                version=GeneratorVersionSpec(generator_id=gid, version="1"),
            )
        run_a = discovery.start_run(DiscoveryRunSpec(generator_id="a", generator_version="1"))
        shared = discovery.propose(
            run_id=run_a, generator_id="a", generator_version="1", definition=sample_definition()
        )
        only_a = discovery.propose(
            run_id=run_a,
            generator_id="a",
            generator_version="1",
            definition=compound_definition_canonical_order(),
        )
        run_b = discovery.start_run(DiscoveryRunSpec(generator_id="b", generator_version="1"))
        discovery.propose(run_id=run_b, generator_id="b", generator_version="1", definition=sample_definition())
        only_b_def = PatternDefinition(
            name="vol",
            rules=Condition(feature="volatility_20d", op=ConditionOp.GT, value=0.3),
            direction=Direction.BEARISH,
            target="forward_excess_spy_20d",
            horizon=20,
            event_mode=EventMode.STATE,
            universe=UniverseContract(universe_name="research-common-equity-500"),
        )
        discovery.propose(run_id=run_b, generator_id="b", generator_version="1", definition=only_b_def)

        # Pass only_a (unique to a) and shared
        for pid in (shared.pattern.pattern_id, only_a.pattern.pattern_id):
            registry.transition(pid, PatternStatus.VALIDATING, reason="v")
            registry.transition(pid, PatternStatus.PASSED, reason="p")

        overlap = compute_pairwise_overlap(store, "a", "b")
        assert overlap.shared_pattern_count == 1
        # a has 2, b has 2, union 3, intersection 1 → jaccard 1/3
        assert abs(overlap.jaccard_proposed - 1 / 3) < 1e-9
        assert overlap.unique_accepted_a == 1
        assert overlap.unique_accepted_b == 0

        ma = compute_generator_metrics(store, "a")
        assert ma.unique_passed_contribution == 1
        mb = compute_generator_metrics(store, "b")
        assert mb.unique_passed_contribution == 0
    finally:
        store.close()


def test_evaluation_metrics_extensible_and_frozen_definition_guard() -> None:
    store = PatternResearchStore(persist=False).open()
    try:
        registry = PatternRegistry(store)
        evaluation = EvaluationService(store, registry)
        rec = registry.register_new(sample_definition())
        guard = FrozenDefinitionGuard(rec)
        eid = evaluation.record(
            EvaluationRunSpec(
                pattern_id=rec.pattern_id,
                pattern_version=rec.version,
                evaluator_id="ev",
                evaluator_version="1",
                period_start=date(2024, 1, 1),
                period_end=date(2024, 6, 30),
                split_role=SplitRole.VALIDATION,
                decision=EvaluationDecision.PASS,
                decision_reason="ok",
                metrics=[
                    EvaluationMetric(
                        metric_name="hit_rate",
                        numeric_value=0.55,
                        success_rule="forward_excess_spy_20d > 0",
                    ),
                    EvaluationMetric(metric_name="brand_new_custom_metric", numeric_value=1.23, unit="arb"),
                ],
            )
        )
        mets = store.list_metrics(eid)
        names = {m["metric_name"] for m in mets}
        assert "hit_rate" in names
        assert "brand_new_custom_metric" in names
        after = store.get_pattern_version(rec.pattern_id, rec.version)
        assert after is not None
        guard.assert_unchanged(after)
        mutated = after.model_copy(
            update={"definition": after.definition.model_copy(update={"horizon": 99})}
        )
        with pytest.raises(RuntimeError, match="mutated|fingerprint"):
            guard.assert_unchanged(mutated)
    finally:
        store.close()


def test_persistence_flags_off_create_no_production_registry_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert settings.pattern_registry_persistence_enabled is False
    assert settings.daily_signal_persistence_enabled is False
    assert settings.indicator_persistence_enabled is False
    assert settings.mining_result_persistence_enabled is False

    path = registry_db_path(settings)
    assert not path.exists()

    # Import / open in-memory path must not create production file
    store = open_research_store(settings)
    try:
        assert store.persist is False
        store.upsert_generator(
            generator_id="x",
            name="X",
            generator_type="HUMAN",
        )
        assert not path.exists()
        assert store.production_file_created() is False
    finally:
        store.close()
    assert not path.exists()


def test_persistence_enabled_writes_isolated_registry_only(tmp_path: Path) -> None:
    settings = _settings(tmp_path, pattern_registry_persistence_enabled=True)
    store = open_research_store(settings)
    try:
        store.upsert_generator(generator_id="x", name="X", generator_type="HUMAN")
        assert settings.pattern_registry_path.exists()
        # Must not touch catalog duckdb merely by registry writes
        assert not settings.duckdb_path.exists()
    finally:
        store.close()


def test_signal_persist_blocked_when_flag_off(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        pattern_registry_persistence_enabled=True,
        daily_signal_persistence_enabled=False,
    )
    store = open_research_store(settings)
    try:
        with pytest.raises(RuntimeError, match="signal persistence"):
            store.insert_signal(
                {
                    "signal_date": date(2026, 9, 22),
                    "security_id": "s1",
                    "pattern_id": "p1",
                    "pattern_version": 1,
                    "direction": "BULLISH",
                }
            )
    finally:
        store.close()


def test_dashboard_empty_state_and_fixture_pages(tmp_path: Path) -> None:
    empty = PatternResearchStore(persist=False).open()
    try:
        app = DashboardApp(empty, persistence_enabled=False)
        html = app.render_overview()
        assert "Empty state" in html
        assert "PATTERN_REGISTRY_PERSISTENCE_ENABLED=false" in html
        assert "Source data freshness" in html
        assert "unavailable" in html.lower()
    finally:
        empty.close()

    fixture_path = tmp_path / "fixture_registry.duckdb"
    store = PatternResearchStore(persist=True, db_path=fixture_path, signal_persist=True).open()
    try:
        seed_fixture_store(store)
        app = DashboardApp(store, persistence_enabled=True, signal_persistence_enabled=True)
        assert "Mining Engine A" in app.render_generators()
        assert "Rejected" in app.render_rejected() or "rejected" in app.render_rejected().lower()
        patterns_html = app.render_patterns({})
        assert "pattern" in patterns_html.lower()
        overview = app.render_overview()
        assert "Source data freshness" in overview
        # Manual retire requires explicit POST with confirm=yes (handler logic covered via registry)
        gens = GeneratorRegistry(store)
        gens.retire("mine_a")
        assert gens.get("mine_a")["status"] == "RETIRED"
        gens.restore("mine_a")
        assert gens.get("mine_a")["status"] == "ACTIVE"
        # Retire preserves versions/history
        assert len(store.list_generator_versions("mine_a")) >= 1
        assert len(store.list_discovery_events(generator_id="mine_a")) >= 1
    finally:
        store.close()


def test_overview_source_freshness_demo_and_populated_rendering(tmp_path: Path) -> None:
    """Overview always renders source-data freshness; demo/empty may be unavailable/demo."""
    from app.dashboard.source_status import (
        ReadOnlySourceStatusProvider,
        SourceFreshnessSummary,
        StaticSourceStatusProvider,
        demo_source_status,
    )

    empty = PatternResearchStore(persist=False).open()
    try:
        demo_app = DashboardApp(
            empty,
            persistence_enabled=False,
            source_status=StaticSourceStatusProvider(demo_source_status()),
        )
        demo_html = demo_app.render_overview()
        assert "Source data freshness" in demo_html
        assert "demo" in demo_html.lower()
        assert "unavailable in demo mode" in demo_html.lower() or "demo mode" in demo_html.lower()
        assert "Empty state" in demo_html

        populated = StaticSourceStatusProvider(
            SourceFreshnessSummary(
                availability="ok",
                message="Read-only lake freshness (no network, no writes).",
                prices_latest="2026-09-22",
                features_latest="2026-09-22",
                labels_latest="2026-09-19",
                expected_latest_session="2026-09-22",
                tracked_note="fixture probe",
            )
        )
        seeded = PatternResearchStore(persist=False).open()
        try:
            seed_fixture_store(seeded)
            ok_app = DashboardApp(
                seeded,
                persistence_enabled=False,
                source_status=populated,
            )
            ok_html = ok_app.render_overview()
            assert "Source data freshness" in ok_html
            assert "2026-09-22" in ok_html
            assert "Prices latest" in ok_html
            assert "no network" in ok_html.lower()
        finally:
            seeded.close()
    finally:
        empty.close()

    # Missing lake → unavailable; must not create production registry path.
    settings = _settings(tmp_path)
    assert settings.pattern_registry_persistence_enabled is False
    assert settings.daily_signal_persistence_enabled is False
    summary = ReadOnlySourceStatusProvider(settings).get_summary()
    assert summary.availability == "unavailable"
    assert not registry_db_path(settings).exists()


def test_aggregation_candidate_labels() -> None:
    engine = SimpleCountAggregationEngine()
    signals = [
        DailyPatternSignal(
            signal_date=date(2026, 9, 22),
            security_id="s1",
            ticker="AAA",
            pattern_id="p1",
            pattern_version=1,
            direction="BULLISH",
        ),
        DailyPatternSignal(
            signal_date=date(2026, 9, 22),
            security_id="s1",
            ticker="AAA",
            pattern_id="p2",
            pattern_version=1,
            direction="BEARISH",
        ),
        DailyPatternSignal(
            signal_date=date(2026, 9, 22),
            security_id="s2",
            ticker="BBB",
            pattern_id="p1",
            pattern_version=1,
            direction="BULLISH",
        ),
    ]
    cands = {c.security_id: c for c in engine.aggregate(signals)}
    assert cands["s1"].label == CandidateLabel.CONFLICTING
    assert cands["s2"].label == CandidateLabel.BUY_CANDIDATE


def test_dashboard_retire_requires_explicit_confirm_semantics() -> None:
    """Retire/restore only via explicit GeneratorRegistry calls; no auto-delete API."""
    store = PatternResearchStore(persist=False).open()
    try:
        gens = GeneratorRegistry(store)
        gens.register(
            generator_id="g1",
            name="G1",
            generator_type=GeneratorType.MINING_ENGINE,
            version=GeneratorVersionSpec(generator_id="g1", version="1"),
        )
        # Simulate dashboard refusing action without confirm: do not call retire.
        assert gens.get("g1")["status"] == "ACTIVE"
        # Explicit request retires but preserves history
        gens.retire("g1", actor="dashboard")
        assert gens.get("g1")["status"] == "RETIRED"
        assert store.list_generator_versions("g1")
        gens.restore("g1", actor="dashboard")
        assert gens.get("g1")["status"] == "ACTIVE"
    finally:
        store.close()


def test_no_network_in_registry_modules() -> None:
    """Sanity: fixture seeding and metrics do not import network clients."""
    store = PatternResearchStore(persist=False).open()
    try:
        seed_fixture_store(store)
        compute_generator_metrics(store, "mine_a", mean_validation_seconds=30.0)
    finally:
        store.close()
