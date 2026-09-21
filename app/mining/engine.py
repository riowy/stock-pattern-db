"""Discover patterns on ANALYSIS; evaluate the frozen set on VALIDATION."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import polars as pl

from app.mining.config import (
    ANALYSIS_END,
    ANALYSIS_START,
    DEFAULT_COOLDOWN_SESSIONS,
    DEFAULT_EVENT_MODE,
    DEFAULT_FDR_Q,
    DEFAULT_MAX_RULE_SIZE,
    INSUFFICIENT_SAMPLE,
    MIN_PATTERN_DATES,
    MIN_PATTERN_ROWS,
    MIN_PATTERN_SECURITIES,
    PRICE_FLOORS,
    VALIDATION_START,
    WINSOR_P_HIGH,
    WINSOR_P_LOW,
    cap_end_before_future_holdout,
)
from app.mining.diagnostics import incremental_vs_parent, pair_self_overlap
from app.mining.events import activate, frequency_counts, pattern_parts
from app.mining.persist import assert_no_mining_persist
from app.mining.selection_diag import summarize_selection
from app.mining.states import (
    StateSpec,
    apply_states,
    base_state_specs,
    freeze_analysis_quantiles,
    quantile_state_specs,
)
from app.mining.stats import bh_qvalues, daily_baseline, horizon_of, pattern_stats

ProgressCallback = Callable[[dict[str, Any]], None]
PAIR_PROGRESS_EVERY = 500


@dataclass
class PatternResult:
    pattern: str
    family: str
    rule_size: int
    analysis: dict
    validation: dict | None = None
    fdr_q: float | None = None
    selected: bool = False
    same_direction: bool | None = None
    effect_ratio: float | None = None
    duplicate_warning: str | None = None
    sample_status: str = "ok"
    overlap_flag: str | None = None
    incremental: dict | None = None
    event_mode: str = "state"
    cooldown_sessions: int = 0

    @property
    def test(self) -> dict | None:
        """Deprecated alias: VALIDATION was previously called TEST."""
        return self.validation


@dataclass
class MiningRun:
    target: str
    analysis_start: date
    analysis_end: date
    validation_start: date
    validation_end: date | None
    singles: list[PatternResult] = field(default_factory=list)
    pairs: list[PatternResult] = field(default_factory=list)
    selected: list[PatternResult] = field(default_factory=list)
    frozen_quantiles: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _meets_support(stats: dict, min_rows: int, min_dates: int, min_secs: int) -> bool:
    n = stats.get("kept", stats.get("n", 0))
    return n >= min_rows and stats["n_dates"] >= min_dates and stats["n_securities"] >= min_secs


def _count_pair_candidates(passed: list[StateSpec]) -> int:
    """Count pairs that the discover loop will score. Scoring itself is unchanged."""
    used_pairs: set[tuple[str, str]] = set()
    n = 0
    for i, a in enumerate(passed):
        for b in passed[i + 1 :]:
            key = tuple(sorted((a.name, b.name)))
            if key in used_pairs:
                continue
            if a.source == "feature" or b.source == "feature":
                continue
            if a.source == b.source and a.name.split("_")[0] == b.name.split("_")[0] and a.name != b.name:
                continue
            used_pairs.add(key)
            n += 1
    return n


def _direction(stats: dict) -> int:
    med = stats.get("excess_median")
    if med is None:
        med = stats.get("median")
    if med is None:
        return 0
    if med > 0:
        return 1
    if med < 0:
        return -1
    return 0


def _score_pattern(
    window: pl.DataFrame,
    pattern: str,
    *,
    target: str,
    excess: str | None,
    horizon: int,
    lo: float | None,
    hi: float | None,
    baseline: pl.DataFrame | None,
    event_mode: str,
    cooldown_sessions: int,
) -> tuple[dict, pl.DataFrame]:
    act = activate(window, pattern, event_mode=event_mode, cooldown_sessions=cooldown_sessions)
    freq = frequency_counts(act)
    kept = act.filter(pl.col("_keep") == True)  # noqa: E712
    stats = pattern_stats(
        kept,
        target,
        excess=excess if excess and excess in kept.columns else None,
        horizon=horizon,
        lo=lo,
        hi=hi,
        baseline_daily=baseline,
    )
    stats.update(freq)
    return stats, kept


class MiningEngine:
    last_selection_raw: dict | None = None
    last_selection_diagnostics: dict | None = None

    def discover(
        self,
        frame: pl.DataFrame,
        *,
        target: str,
        analysis_start: date = ANALYSIS_START,
        analysis_end: date = ANALYSIS_END,
        min_rows: int = MIN_PATTERN_ROWS,
        min_dates: int = MIN_PATTERN_DATES,
        min_securities: int = MIN_PATTERN_SECURITIES,
        max_rule_size: int = DEFAULT_MAX_RULE_SIZE,
        fdr_q: float = DEFAULT_FDR_Q,
        event_mode: str = DEFAULT_EVENT_MODE,
        cooldown_sessions: int = DEFAULT_COOLDOWN_SESSIONS,
        attach_pair_diagnostics: bool = True,
        progress: ProgressCallback | None = None,
    ) -> tuple[list[PatternResult], list[StateSpec], dict[str, dict[str, float]], dict[str, float]]:
        assert_no_mining_persist()
        analysis = frame.filter((pl.col("date") >= analysis_start) & (pl.col("date") <= analysis_end))
        if analysis.height == 0:
            self.last_selection_raw = None
            self.last_selection_diagnostics = None
            return [], [], {}, {}
        frozen = freeze_analysis_quantiles(analysis)
        cols = set(analysis.columns)
        specs = base_state_specs(cols) + quantile_state_specs(cols, frozen)
        if progress is not None:
            progress({"phase": "generating states"})
        analysis = apply_states(analysis, specs)
        excess = target if target.startswith("forward_excess") else target.replace("forward_return_", "forward_excess_spy_")
        s = analysis[target].drop_nulls()
        lo = float(s.quantile(WINSOR_P_LOW)) if s.len() else None
        hi = float(s.quantile(WINSOR_P_HIGH)) if s.len() else None
        cutoffs = {"lo": lo, "hi": hi}
        horizon = horizon_of(target)
        baseline = daily_baseline(analysis, target)
        results: list[PatternResult] = []
        passed: list[StateSpec] = []
        single_kept: dict[str, pl.DataFrame] = {}
        n_single_candidates = sum(1 for spec in specs if spec.name in analysis.columns)
        if progress is not None:
            progress({"phase": "single candidates", "n": n_single_candidates})
        for spec in specs:
            if spec.name not in analysis.columns:
                continue
            stats, kept = _score_pattern(
                analysis,
                spec.name,
                target=target,
                excess=excess if excess in analysis.columns else None,
                horizon=horizon,
                lo=lo,
                hi=hi,
                baseline=baseline,
                event_mode=event_mode,
                cooldown_sessions=cooldown_sessions,
            )
            status = "ok" if _meets_support(stats, min_rows, min_dates, min_securities) else INSUFFICIENT_SAMPLE
            row = PatternResult(
                spec.name,
                spec.family,
                1,
                stats,
                sample_status=status,
                event_mode=event_mode,
                cooldown_sessions=cooldown_sessions,
            )
            results.append(row)
            if status == "ok":
                passed.append(spec)
                single_kept[spec.name] = kept

        if progress is not None:
            progress({"phase": "supported singles", "n": len(passed)})

        generated_pair_names: list[str] = []
        insufficient_pair_names: list[str] = []
        if max_rule_size >= 2:
            n_pair_total = 0
            n_pair_done = 0
            if progress is not None:
                n_pair_total = _count_pair_candidates(passed)
                progress({"phase": "pair candidates", "n": n_pair_total})
            used_pairs: set[tuple[str, str]] = set()
            for i, a in enumerate(passed):
                for b in passed[i + 1 :]:
                    key = tuple(sorted((a.name, b.name)))
                    if key in used_pairs:
                        continue
                    if a.source == "feature" or b.source == "feature":
                        continue
                    if a.source == b.source and a.name.split("_")[0] == b.name.split("_")[0] and a.name != b.name:
                        continue
                    used_pairs.add(key)
                    if progress is not None:
                        n_pair_done += 1
                        if n_pair_total and (n_pair_done % PAIR_PROGRESS_EVERY == 0 or n_pair_done == n_pair_total):
                            progress({"phase": "evaluating pairs", "done": n_pair_done, "total": n_pair_total})
                    pname = f"{a.name} AND {b.name}"
                    generated_pair_names.append(pname)
                    stats, kept = _score_pattern(
                        analysis,
                        pname,
                        target=target,
                        excess=excess if excess in analysis.columns else None,
                        horizon=horizon,
                        lo=lo,
                        hi=hi,
                        baseline=baseline,
                        event_mode=event_mode,
                        cooldown_sessions=cooldown_sessions,
                    )
                    status = "ok" if _meets_support(stats, min_rows, min_dates, min_securities) else INSUFFICIENT_SAMPLE
                    if status != "ok":
                        insufficient_pair_names.append(pname)
                        continue
                    overlap = pair_self_overlap(analysis, pname) if attach_pair_diagnostics else {}
                    incremental = None
                    if attach_pair_diagnostics:
                        inc_a = incremental_vs_parent(kept, single_kept[a.name], target, horizon=horizon, lo=lo, hi=hi)
                        inc_b = incremental_vs_parent(kept, single_kept[b.name], target, horizon=horizon, lo=lo, hi=hi)
                        incremental = {"vs_a": inc_a, "vs_b": inc_b, "parent_a": a.name, "parent_b": b.name}
                    results.append(
                        PatternResult(
                            pname,
                            "MIXED" if a.family != b.family else a.family,
                            2,
                            stats,
                            sample_status=status,
                            overlap_flag=overlap.get("flag"),
                            incremental=incremental,
                            event_mode=event_mode,
                            cooldown_sessions=cooldown_sessions,
                        )
                    )
                    if overlap.get("flag"):
                        results[-1].duplicate_warning = overlap["flag"]
        elif progress is not None:
            progress({"phase": "pair candidates", "n": 0})

        if progress is not None:
            progress({"phase": "FDR"})
        pvals = [r.analysis.get("p_value") for r in results if r.sample_status == "ok"]
        qvals = bh_qvalues(pvals)
        qi = 0
        for r in results:
            if r.sample_status != "ok":
                continue
            r.fdr_q = qvals[qi]
            r.selected = r.fdr_q is not None and r.fdr_q <= fdr_q
            qi += 1
        _mark_near_duplicates(results, analysis, specs)
        generated_single_names = [spec.name for spec in specs if spec.name in analysis.columns]
        p_ok = [p for p in pvals]
        self.last_selection_raw = {
            "n_generated_singles": n_single_candidates,
            "n_support_passed_singles": len(passed),
            "n_generated_pairs": len(generated_pair_names),
            "n_support_passed_pairs": len(generated_pair_names) - len(insufficient_pair_names),
            "pvalues": p_ok,
            "qvalues": qvals,
            "fdr_threshold": fdr_q,
            "max_rule_size": max_rule_size,
            "results": results,
            "generated_singles": generated_single_names,
            "passed_singles": [s.name for s in passed],
            "generated_pairs": generated_pair_names,
            "insufficient_pairs": insufficient_pair_names,
            "specs": specs,
        }
        self.last_selection_diagnostics = summarize_selection(
            n_generated_singles=n_single_candidates,
            n_support_passed_singles=len(passed),
            n_generated_pairs=len(generated_pair_names),
            n_support_passed_pairs=len(generated_pair_names) - len(insufficient_pair_names),
            pvalues=p_ok,
            qvalues=qvals,
            fdr_threshold=fdr_q,
            max_rule_size=max_rule_size,
            results=results,
            generated_singles=generated_single_names,
            passed_singles=[s.name for s in passed],
            generated_pairs=generated_pair_names,
            insufficient_pairs=insufficient_pair_names,
            specs=specs,
        )
        if progress is not None:
            progress({"phase": "selected", "n": sum(1 for r in results if r.selected)})
        return results, specs, frozen, cutoffs

    def evaluate_validation(
        self,
        frame: pl.DataFrame,
        discovered: list[PatternResult],
        specs: list[StateSpec],
        cutoffs: dict[str, float],
        *,
        target: str,
        validation_start: date = VALIDATION_START,
        validation_end: date | None = None,
        event_mode: str | None = None,
        cooldown_sessions: int | None = None,
    ) -> list[PatternResult]:
        assert_no_mining_persist()
        end = cap_end_before_future_holdout(validation_end)
        val = frame.filter(pl.col("date") >= validation_start)
        if end is not None:
            val = val.filter(pl.col("date") <= end)
        val = apply_states(val, specs)
        excess = target if target.startswith("forward_excess") else target.replace("forward_return_", "forward_excess_spy_")
        horizon = horizon_of(target)
        lo, hi = cutoffs.get("lo"), cutoffs.get("hi")
        baseline = daily_baseline(val, target)
        out: list[PatternResult] = []
        for row in discovered:
            if not row.selected:
                continue
            mode = event_mode if event_mode is not None else row.event_mode
            cd = cooldown_sessions if cooldown_sessions is not None else row.cooldown_sessions
            stats, kept = _score_pattern(
                val,
                row.pattern,
                target=target,
                excess=excess if excess in val.columns else None,
                horizon=horizon,
                lo=lo,
                hi=hi,
                baseline=baseline,
                event_mode=mode,
                cooldown_sessions=cd,
            )
            same = _direction(row.analysis) != 0 and _direction(row.analysis) == _direction(stats)
            ratio = None
            a = row.analysis.get("excess_median") or row.analysis.get("median")
            t = stats.get("excess_median") or stats.get("median")
            if a and t is not None and abs(a) > 1e-12:
                ratio = t / a
            incremental = None
            if row.rule_size == 2:
                parts = pattern_parts(row.pattern)
                if len(parts) == 2:
                    _, kept_a = _score_pattern(val, parts[0], target=target, excess=None, horizon=horizon, lo=lo, hi=hi, baseline=None, event_mode=mode, cooldown_sessions=cd)
                    _, kept_b = _score_pattern(val, parts[1], target=target, excess=None, horizon=horizon, lo=lo, hi=hi, baseline=None, event_mode=mode, cooldown_sessions=cd)
                    incremental = {
                        "vs_a": incremental_vs_parent(kept, kept_a, target, horizon=horizon, lo=lo, hi=hi),
                        "vs_b": incremental_vs_parent(kept, kept_b, target, horizon=horizon, lo=lo, hi=hi),
                        "parent_a": parts[0],
                        "parent_b": parts[1],
                    }
            out.append(
                PatternResult(
                    row.pattern,
                    row.family,
                    row.rule_size,
                    row.analysis,
                    validation=stats,
                    fdr_q=row.fdr_q,
                    selected=True,
                    same_direction=same,
                    effect_ratio=ratio,
                    duplicate_warning=row.duplicate_warning,
                    sample_status=row.sample_status,
                    overlap_flag=row.overlap_flag,
                    incremental=incremental or row.incremental,
                    event_mode=mode,
                    cooldown_sessions=cd,
                )
            )
        return out

    def evaluate_holdout(self, *args, **kwargs):  # noqa: ANN002, ANN003
        """Deprecated name. VALIDATION was previously called TEST."""
        if "test_start" in kwargs:
            kwargs["validation_start"] = kwargs.pop("test_start")
        if "test_end" in kwargs:
            kwargs["validation_end"] = kwargs.pop("test_end")
        return self.evaluate_validation(*args, **kwargs)


def _mark_near_duplicates(results: list[PatternResult], analysis: pl.DataFrame, specs: list[StateSpec]) -> None:
    pairs = [
        ("rsi14_le_30", "feat_rsi14_le_30"),
        ("volume_ratio20_gt_1_5", "feat_volume_ratio20_gt_1_5"),
        ("price_above_sma200", "feat_close_above_ma200"),
    ]
    warned = set()
    for a, b in pairs:
        if a in analysis.columns and b in analysis.columns:
            warned.add(a)
            warned.add(b)
    for r in results:
        parts = r.pattern.split(" AND ")
        if any(p in warned for p in parts) and not r.duplicate_warning:
            r.duplicate_warning = "near-duplicate of a feature-engine alias; interpret once"


def _pattern_mask(work: pl.DataFrame, pattern: str) -> pl.Expr | None:
    names = pattern_parts(pattern)
    mask = pl.lit(True)
    for name in names:
        if name not in work.columns:
            return None
        mask = mask & (pl.col(name) == True)  # noqa: E712
    return mask


def price_floor_audit(
    frame: pl.DataFrame,
    pattern: str,
    specs: list[StateSpec],
    target: str,
    floors: tuple = PRICE_FLOORS,
    cutoffs: dict | None = None,
    *,
    event_mode: str = "state",
    cooldown_sessions: int = 0,
) -> list[dict]:
    cutoffs = cutoffs or {}
    work = apply_states(frame, specs)
    rows = []
    horizon = horizon_of(target)
    close_col = "raw_close" if "raw_close" in work.columns else "close"
    for label, floor in floors:
        subset = work
        if floor is not None and close_col in subset.columns:
            subset = subset.filter(pl.col(close_col) >= floor)
        stats, _kept = _score_pattern(
            subset,
            pattern,
            target=target,
            excess=None,
            horizon=horizon,
            lo=cutoffs.get("lo"),
            hi=cutoffs.get("hi"),
            baseline=None,
            event_mode=event_mode,
            cooldown_sessions=cooldown_sessions,
        )
        rows.append({"subset": label, **stats})
    return rows


def regime_audit(
    frame: pl.DataFrame,
    pattern: str,
    specs: list[StateSpec],
    target: str,
    *,
    analysis_start: date,
    analysis_end: date,
    cutoffs: dict | None = None,
    event_mode: str = "state",
    cooldown_sessions: int = 0,
) -> list[dict]:
    """Slice an already-selected pattern by SPY-MA200 and ANALYSIS-frozen VIX tertiles."""
    from app.research.statistics import attach_regimes, freeze_vix_cutoffs

    cutoffs = cutoffs or {}
    analysis = frame.filter((pl.col("date") >= analysis_start) & (pl.col("date") <= analysis_end))
    freeze_src = analysis
    if "vix_close" not in analysis.columns and "vix_close_ctx" in analysis.columns:
        freeze_src = analysis.rename({"vix_close_ctx": "vix_close"})
    vix_q33, vix_q67 = freeze_vix_cutoffs(freeze_src)
    work = frame
    if "vix_close" not in work.columns and "vix_close_ctx" in work.columns:
        work = work.rename({"vix_close_ctx": "vix_close"})
    work = attach_regimes(work, vix_q33, vix_q67)
    work = apply_states(work, specs)
    horizon = horizon_of(target)
    rows = []
    act = activate(work, pattern, event_mode=event_mode, cooldown_sessions=cooldown_sessions)
    matched = act.filter(pl.col("_keep") == True)  # noqa: E712
    for col in ("spy_trend_regime", "vix_regime"):
        if col not in matched.columns:
            continue
        for label in matched[col].drop_nulls().unique().to_list():
            subset = matched.filter(pl.col(col) == label)
            stats = pattern_stats(subset, target, excess=None, horizon=horizon, lo=cutoffs.get("lo"), hi=cutoffs.get("hi"))
            rows.append({"subset": str(label), "kind": col, **stats})
    return rows
