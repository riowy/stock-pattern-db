"""Mining engine: split isolation, support, no persistence."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from app.mining.config import INSUFFICIENT_SAMPLE, MINING_RESULT_PERSISTENCE_ENABLED
from app.mining.engine import MiningEngine
from app.mining.persist import MiningPersistenceDisabled, persist_mining_results
from app.mining.stats import bh_qvalues


def _frame(n_dates: int = 40, n_secs: int = 40) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    start = date(2018, 1, 2)
    dates = []
    d = start
    while len(dates) < n_dates:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    for i, dt in enumerate(dates):
        for s in range(n_secs):
            flag = (s + i) % 7 == 0
            y = 0.02 if flag else rng.normal(0, 0.01)
            rows.append(
                {
                    "security_id": f"S{s:03d}",
                    "date": dt,
                    "rsi_14": 25.0 if flag else 55.0,
                    "forward_excess_spy_20d": y,
                    "raw_close": 25.0,
                }
            )
    return pl.DataFrame(rows)


def test_persistence_disabled() -> None:
    assert MINING_RESULT_PERSISTENCE_ENABLED is False
    with pytest.raises(MiningPersistenceDisabled):
        persist_mining_results({})


def test_discover_does_not_use_test_rows_for_thresholds() -> None:
    analysis = _frame(80, 40).with_columns(pl.lit(date(2018, 6, 1)).alias("_"))
    # Build explicit analysis/test
    a_dates = sorted(analysis["date"].unique().to_list())
    mid = a_dates[len(a_dates) // 2]
    engine = MiningEngine()
    discovered, specs, frozen, cutoffs = engine.discover(
        analysis,
        target="forward_excess_spy_20d",
        analysis_start=a_dates[0],
        analysis_end=mid,
        min_rows=20,
        min_dates=5,
        min_securities=5,
        max_rule_size=1,
        fdr_q=1.0,
    )
    assert specs
    # Re-run discover with test-period rows appended that would flip RSI threshold if leaked
    test_extra = analysis.with_columns(
        pl.when(pl.col("date") > mid).then(pl.lit(90.0)).otherwise(pl.col("rsi_14")).alias("rsi_14")
    )
    again, _, frozen2, _ = engine.discover(
        test_extra,
        target="forward_excess_spy_20d",
        analysis_start=a_dates[0],
        analysis_end=mid,
        min_rows=20,
        min_dates=5,
        min_securities=5,
        max_rule_size=1,
        fdr_q=1.0,
    )
    names1 = {r.pattern for r in discovered if r.sample_status != INSUFFICIENT_SAMPLE}
    names2 = {r.pattern for r in again if r.sample_status != INSUFFICIENT_SAMPLE}
    assert "rsi14_le_30" in names1
    assert names1 == names2


def test_holdout_only_evaluates_selected() -> None:
    df = _frame(90, 35)
    dates = sorted(df["date"].unique().to_list())
    mid = dates[50]
    engine = MiningEngine()
    discovered, specs, _, cutoffs = engine.discover(
        df,
        target="forward_excess_spy_20d",
        analysis_start=dates[0],
        analysis_end=mid,
        min_rows=15,
        min_dates=5,
        min_securities=5,
        max_rule_size=1,
        fdr_q=1.0,
    )
    for row in discovered:
        if row.pattern == "rsi14_le_30":
            row.selected = True
        else:
            row.selected = False
    evaluated = engine.evaluate_validation(
        df, discovered, specs, cutoffs, target="forward_excess_spy_20d", validation_start=dates[dates.index(mid) + 1]
    )
    assert all(r.selected for r in evaluated)
    assert all(r.validation is not None for r in evaluated)
    assert {r.pattern for r in evaluated} == {"rsi14_le_30"}


def test_bh_qvalues_monotone() -> None:
    q = bh_qvalues([0.001, 0.02, 0.2, None])
    assert q[0] is not None and q[1] is not None and q[2] is not None
    assert q[0] <= q[1] <= q[2]
    assert q[3] is None
    assert q[2] == pytest.approx(0.2, abs=1e-12)
    assert q[0] < 0.05
    assert q[2] > 0.1


def test_price_floor_audit_does_not_drop_all_rows() -> None:
    from app.mining.engine import price_floor_audit
    from app.mining.states import StateSpec

    df = _frame(20, 10)
    specs = [StateSpec("rsi14_le_30", "MOMENTUM", pl.col("rsi_14") <= 30, "rsi")]
    rows = price_floor_audit(df, "rsi14_le_30", specs, "forward_excess_spy_20d")
    assert rows
    assert rows[0]["subset"] == "all"
    assert rows[0]["n"] >= rows[-1]["n"]


def test_insufficient_sample_excluded() -> None:
    df = _frame(10, 5)
    dates = sorted(df["date"].unique().to_list())
    discovered, _, _, _ = MiningEngine().discover(
        df,
        target="forward_excess_spy_20d",
        analysis_start=dates[0],
        analysis_end=dates[-1],
        min_rows=1000,
        min_dates=100,
        min_securities=30,
        max_rule_size=1,
        fdr_q=0.1,
    )
    assert all(r.sample_status == INSUFFICIENT_SAMPLE or r.selected is False for r in discovered)


def test_entry_has_fewer_rows_than_state() -> None:
    from app.mining.events import activate, frequency_counts

    df = _frame(30, 8).sort(["security_id", "date"])
    stated = df.with_columns((pl.col("rsi_14") <= 30).alias("rsi14_le_30"))
    state = activate(stated, "rsi14_le_30", event_mode="state")
    entry = activate(stated, "rsi14_le_30", event_mode="entry")
    fs = frequency_counts(state)
    fe = frequency_counts(entry)
    assert fs["rows"] >= fe["kept"]
    assert fe["episodes"] == fe["entry_events"]
    assert fs["rows"] >= fs["episodes"]


def test_cooldown_reduces_entry_count() -> None:
    from app.mining.events import activate, frequency_counts

    df = _frame(40, 6).sort(["security_id", "date"])
    stated = df.with_columns((pl.col("rsi_14") <= 30).alias("rsi14_le_30"))
    e0 = frequency_counts(activate(stated, "rsi14_le_30", event_mode="entry", cooldown_sessions=0))
    e20 = frequency_counts(activate(stated, "rsi14_le_30", event_mode="entry", cooldown_sessions=20))
    assert e20["kept"] <= e0["kept"]


def test_pair_jaccard_and_high_overlap() -> None:
    from app.mining.diagnostics import pair_self_overlap
    from app.mining.config import HIGH_OVERLAP

    df = _frame(20, 10).with_columns(
        (pl.col("rsi_14") <= 30).alias("rsi14_le_30"),
        (pl.col("rsi_14") <= 30).alias("volume_ratio20_gt_2"),
    )
    report = pair_self_overlap(df, "rsi14_le_30 AND volume_ratio20_gt_2")
    assert report["jaccard"] == pytest.approx(1.0)
    assert report["flag"] == HIGH_OVERLAP


def test_partial_pair_is_not_high_overlap() -> None:
    from app.mining.diagnostics import pair_self_overlap

    df = _frame(20, 10).with_columns(
        (pl.col("rsi_14") <= 30).alias("rsi14_le_30"),
        ((pl.arange(0, pl.len()) % 3) == 0).alias("volume_ratio20_gt_2"),
    )
    report = pair_self_overlap(df, "rsi14_le_30 AND volume_ratio20_gt_2")
    assert report["flag"] is None
    assert report["jaccard"] is not None
    assert report["jaccard"] < 0.9


def test_future_holdout_cap() -> None:
    from datetime import date as d

    from app.mining.config import FUTURE_HOLDOUT_START, cap_end_before_future_holdout

    capped = cap_end_before_future_holdout(d(2026, 12, 31))
    assert capped is not None
    assert capped < FUTURE_HOLDOUT_START


def test_walkforward_fold_does_not_use_future_eval_rows_for_cutoffs() -> None:
    from app.mining.states import freeze_analysis_quantiles

    df = _frame(80, 20).with_columns(pl.col("rsi_14").alias("hist_vol_20"))
    dates = sorted(df["date"].unique().to_list())
    mid = dates[40]
    disc = df.filter(pl.col("date") <= mid)
    frozen = freeze_analysis_quantiles(disc)
    mutated = df.with_columns(
        pl.when(pl.col("date") > mid).then(pl.col("hist_vol_20") * 50).otherwise(pl.col("hist_vol_20")).alias("hist_vol_20")
    )
    frozen2 = freeze_analysis_quantiles(mutated.filter(pl.col("date") <= mid))
    assert frozen.get("hist_vol_20") == frozen2.get("hist_vol_20")


def test_format_elapsed() -> None:
    from app.mining.progress import format_elapsed

    assert format_elapsed(0) == "0m 0s"
    assert format_elapsed(65) == "1m 5s"
    assert format_elapsed(90) == "1m 30s"


def test_discover_progress_does_not_change_results() -> None:
    df = _frame(80, 40)
    dates = sorted(df["date"].unique().to_list())
    mid = dates[len(dates) // 2]
    kwargs = {
        "target": "forward_excess_spy_20d",
        "analysis_start": dates[0],
        "analysis_end": mid,
        "min_rows": 20,
        "min_dates": 5,
        "min_securities": 5,
        "max_rule_size": 2,
        "fdr_q": 1.0,
    }
    engine = MiningEngine()
    silent, _, _, _ = engine.discover(df, **kwargs)
    events: list[dict] = []
    logged, _, _, _ = engine.discover(df, progress=events.append, **kwargs)
    assert [(r.pattern, r.selected, r.fdr_q, r.sample_status) for r in silent] == [
        (r.pattern, r.selected, r.fdr_q, r.sample_status) for r in logged
    ]
    phases = [e.get("phase") for e in events]
    assert "generating states" in phases
    assert "single candidates" in phases
    assert "supported singles" in phases
    assert "pair candidates" in phases
    assert "FDR" in phases
    assert "selected" in phases


def test_walkforward_progress_prints_fold_prefix() -> None:
    from io import StringIO

    from rich.console import Console

    from app.mining.progress import WalkForwardLogger

    buf = StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None, width=120)
    logger = WalkForwardLogger(console, n_folds=6, show_status=False)
    logger({"event": "fold_start", "fold": 1, "n_folds": 6})
    logger({"phase": "generating states"})
    logger({"phase": "single candidates", "n": 64})
    logger({"phase": "supported singles", "n": 40})
    logger({"phase": "pair candidates", "n": 3200})
    logger({"phase": "evaluating pairs", "done": 500, "total": 3200})
    logger({"phase": "FDR"})
    logger({"phase": "selected", "n": 12})
    logger({"phase": "evaluation complete"})
    logger({"event": "fold_elapsed"})
    logger.close()
    text = buf.getvalue()
    assert "[Fold 1/6] loading indicators..." in text
    assert "[Fold 1/6] generating states..." in text
    assert "[Fold 1/6] single candidates: 64" in text
    assert "[Fold 1/6] supported singles: 40" in text
    assert "[Fold 1/6] pair candidates: 3200" in text
    assert "[Fold 1/6] evaluating pairs: 500/3200" in text
    assert "[Fold 1/6] FDR..." in text
    assert "[Fold 1/6] selected: 12" in text
    assert "[Fold 1/6] evaluation complete" in text
    assert "elapsed:" in text
    assert "current fold: 1/6" in text
    assert "current phase:" in text


def _discover_kwargs(**overrides):
    base = {
        "target": "forward_excess_spy_20d",
        "min_rows": 20,
        "min_dates": 5,
        "min_securities": 5,
        "max_rule_size": 2,
        "fdr_q": 0.10,
    }
    base.update(overrides)
    return base


def test_selection_diagnostics_do_not_change_selected_set() -> None:
    from app.mining.selection_diag import summarize_from_raw

    df = _frame(80, 40)
    dates = sorted(df["date"].unique().to_list())
    mid = dates[len(dates) // 2]
    kwargs = _discover_kwargs(analysis_start=dates[0], analysis_end=mid, max_rule_size=1)
    a, _, _, _ = MiningEngine().discover(df, **kwargs)
    engine = MiningEngine()
    b, _, _, _ = engine.discover(df, **kwargs)
    assert [(r.pattern, r.selected, r.fdr_q, r.sample_status) for r in a] == [
        (r.pattern, r.selected, r.fdr_q, r.sample_status) for r in b
    ]
    diag = summarize_from_raw(engine.last_selection_raw, ["rsi14_le_30"])
    assert diag is not None
    assert diag["n_q_le_0_10"] == sum(1 for r in b if r.selected)
    assert diag["n_q_le_0_20"] >= diag["n_q_le_0_10"]
    assert diag["n_q_le_0_30"] >= diag["n_q_le_0_20"]
    assert diag["pair_cap_limit"] is None


def test_selection_reason_insufficient_support() -> None:
    from app.mining.selection_diag import INSUFFICIENT_SUPPORT, summarize_from_raw

    df = _frame(80, 40)
    dates = sorted(df["date"].unique().to_list())
    engine = MiningEngine()
    engine.discover(df, **_discover_kwargs(analysis_start=dates[0], analysis_end=dates[-1], min_rows=100000, max_rule_size=1))
    row = summarize_from_raw(engine.last_selection_raw, ["rsi14_le_30"])["tracked"][0]
    assert row["candidate_generated"] is True
    assert row["support_passed"] is False
    assert row["reason"] == INSUFFICIENT_SUPPORT
    assert row["selected"] is False


def test_selection_reason_fdr_fail_vs_selected() -> None:
    from app.mining.selection_diag import FDR_FAIL, SELECTED, summarize_from_raw

    df = _frame(80, 40)
    dates = sorted(df["date"].unique().to_list())
    kwargs = _discover_kwargs(analysis_start=dates[0], analysis_end=dates[-1], max_rule_size=1)
    fail_engine = MiningEngine()
    fail_engine.discover(df, **{**kwargs, "fdr_q": -1.0})
    fail = summarize_from_raw(fail_engine.last_selection_raw, ["rsi14_le_30"])["tracked"][0]
    assert fail["support_passed"] is True
    assert fail["reason"] == FDR_FAIL
    assert fail["selected"] is False
    assert fail["fdr_q_value"] is not None

    sel_engine = MiningEngine()
    sel_engine.discover(df, **{**kwargs, "fdr_q": 1.0})
    sel = summarize_from_raw(sel_engine.last_selection_raw, ["rsi14_le_30"])["tracked"][0]
    assert sel["reason"] == SELECTED
    assert sel["selected"] is True


def test_selection_reason_pair_pruned_not_cap() -> None:
    from app.mining.selection_diag import NOT_GENERATED, PAIR_PRUNED, summarize_from_raw

    df = _frame(80, 40).with_columns(
        pl.when(pl.arange(0, pl.len()) % 2 == 0).then(pl.lit(25.0)).otherwise(pl.lit(75.0)).alias("rsi_14")
    )
    dates = sorted(df["date"].unique().to_list())
    engine = MiningEngine()
    engine.discover(df, **_discover_kwargs(analysis_start=dates[0], analysis_end=dates[-1]))
    diag = summarize_from_raw(
        engine.last_selection_raw,
        ["rsi14_le_30 AND rsi14_ge_70", "missing_pattern"],
    )
    pruned = diag["tracked"][0]
    assert pruned["parent_a"]["support_passed"] is True
    assert pruned["parent_b"]["support_passed"] is True
    assert pruned["candidate_pruned"] is True
    assert pruned["pair_generation_eligible"] is False
    assert pruned["pair_cap_excluded"] is False
    assert pruned["reason"] == PAIR_PRUNED
    missing = diag["tracked"][1]
    assert missing["reason"] == NOT_GENERATED


def test_selection_reason_generated_pair_parent_status() -> None:
    from app.mining.selection_diag import summarize_from_raw

    df = _frame(80, 40).with_columns(
        pl.when(pl.arange(0, pl.len()) % 3 == 0).then(pl.lit(3.0)).otherwise(pl.lit(1.0)).alias("volume_ratio_20")
    )
    dates = sorted(df["date"].unique().to_list())
    engine = MiningEngine()
    discovered, _, _, _ = engine.discover(df, **_discover_kwargs(analysis_start=dates[0], analysis_end=dates[-1], fdr_q=1.0))
    names = {r.pattern for r in discovered if r.sample_status != INSUFFICIENT_SAMPLE}
    diag = summarize_from_raw(engine.last_selection_raw, ["rsi14_le_30 AND volume_ratio20_gt_2"])["tracked"][0]
    assert diag["parent_a"]["name"] == "rsi14_le_30"
    assert diag["parent_b"]["name"] == "volume_ratio20_gt_2"
    assert diag["pair_cap_excluded"] is False
    if "rsi14_le_30 AND volume_ratio20_gt_2" in names or "volume_ratio20_gt_2 AND rsi14_le_30" in names:
        assert diag["candidate_generated"] is True
        assert diag["pair_generation_eligible"] is True
        assert diag["reason"] in {"SELECTED", "FDR_FAIL", "INSUFFICIENT_SUPPORT"}
    else:
        assert diag["reason"] in {"INSUFFICIENT_SUPPORT", "NOT_GENERATED", "PAIR_PRUNED"}

