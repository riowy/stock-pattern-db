"""Inspect a frozen named pattern. No discovery retuning. Memory-only."""

from __future__ import annotations

from datetime import date

import polars as pl

from app.mining.config import (
    ANALYSIS_END,
    ANALYSIS_START,
    VALIDATION_START,
    cap_end_before_future_holdout,
)
from app.mining.diagnostics import (
    incremental_vs_parent,
    leave_one_top_security,
    pair_self_overlap,
    security_concentration,
    year_breakdown,
)
from app.mining.engine import _score_pattern, price_floor_audit, regime_audit
from app.mining.events import pattern_parts
from app.mining.persist import assert_no_mining_persist
from app.mining.states import apply_states, base_state_specs, freeze_analysis_quantiles, quantile_state_specs
from app.mining.stats import daily_baseline, horizon_of


def _window(frame: pl.DataFrame, start: date, end: date | None) -> pl.DataFrame:
    out = frame.filter(pl.col("date") >= start)
    if end is not None:
        out = out.filter(pl.col("date") <= end)
    return out


def prepare_frozen_states(
    frame: pl.DataFrame,
    *,
    target: str,
    analysis_start: date = ANALYSIS_START,
    analysis_end: date = ANALYSIS_END,
) -> tuple[pl.DataFrame, list, dict, dict]:
    """Freeze quantiles/winsor on ANALYSIS only, then apply states to the full frame."""
    assert_no_mining_persist()
    analysis = _window(frame, analysis_start, analysis_end)
    frozen = freeze_analysis_quantiles(analysis)
    specs = base_state_specs(set(frame.columns)) + quantile_state_specs(set(frame.columns), frozen)
    stated = apply_states(frame, specs)
    from app.mining.config import WINSOR_P_HIGH, WINSOR_P_LOW

    s = analysis[target].drop_nulls() if target in analysis.columns else None
    lo = float(s.quantile(WINSOR_P_LOW)) if s is not None and s.len() else None
    hi = float(s.quantile(WINSOR_P_HIGH)) if s is not None and s.len() else None
    return stated, specs, frozen, {"lo": lo, "hi": hi}


def inspect_pattern(
    frame: pl.DataFrame,
    pattern: str,
    *,
    target: str,
    analysis_start: date = ANALYSIS_START,
    analysis_end: date = ANALYSIS_END,
    validation_start: date = VALIDATION_START,
    validation_end: date | None = None,
    specs: list | None = None,
    frozen: dict | None = None,
    cutoffs: dict | None = None,
) -> dict:
    """STATE / ENTRY / cooldown-20 diagnostics for one named pattern. Does not select patterns."""
    assert_no_mining_persist()
    if specs is None:
        stated, specs, frozen, cutoffs = prepare_frozen_states(
            frame, target=target, analysis_start=analysis_start, analysis_end=analysis_end
        )
    else:
        stated = apply_states(frame, specs)
        cutoffs = cutoffs or {}
        frozen = frozen or {}
    val_end = cap_end_before_future_holdout(validation_end)
    horizon = horizon_of(target)
    lo, hi = (cutoffs or {}).get("lo"), (cutoffs or {}).get("hi")
    analysis = _window(stated, analysis_start, analysis_end)
    validation = _window(stated, validation_start, val_end)
    a_base = daily_baseline(analysis, target)
    v_base = daily_baseline(validation, target)

    modes = [
        ("state", 0),
        ("entry", 0),
        ("entry", 20),
    ]
    blocks = {}
    for mode, cd in modes:
        a_stats, a_kept = _score_pattern(
            analysis, pattern, target=target, excess=target if target in analysis.columns else None,
            horizon=horizon, lo=lo, hi=hi, baseline=a_base, event_mode=mode, cooldown_sessions=cd,
        )
        v_stats, v_kept = _score_pattern(
            validation, pattern, target=target, excess=target if target in validation.columns else None,
            horizon=horizon, lo=lo, hi=hi, baseline=v_base, event_mode=mode, cooldown_sessions=cd,
        )
        conc = security_concentration(a_kept)
        loo = leave_one_top_security(a_kept, target, horizon=horizon, lo=lo, hi=hi)
        pieces = [x for x in (a_kept, v_kept) if x.height]
        years_src = pl.concat(pieces, how="diagonal_relaxed") if len(pieces) > 1 else (pieces[0] if pieces else a_kept)
        years = year_breakdown(years_src, target, lo=lo, hi=hi)
        key = f"{mode}_cd{cd}"
        blocks[key] = {
            "analysis": a_stats,
            "validation": v_stats,
            "concentration": conc,
            "leave_one_security": loo,
            "years": years,
        }

    parts = pattern_parts(pattern)
    incremental = None
    overlap = None
    if len(parts) == 2:
        overlap = pair_self_overlap(analysis, pattern)
        a_pair, a_kept = _score_pattern(analysis, pattern, target=target, excess=None, horizon=horizon, lo=lo, hi=hi, baseline=a_base, event_mode="state", cooldown_sessions=0)
        _, kept_a = _score_pattern(analysis, parts[0], target=target, excess=None, horizon=horizon, lo=lo, hi=hi, baseline=None, event_mode="state", cooldown_sessions=0)
        _, kept_b = _score_pattern(analysis, parts[1], target=target, excess=None, horizon=horizon, lo=lo, hi=hi, baseline=None, event_mode="state", cooldown_sessions=0)
        incremental = {
            "parent_a": parts[0],
            "parent_b": parts[1],
            "pair": a_pair,
            "vs_a": incremental_vs_parent(a_kept, kept_a, target, horizon=horizon, lo=lo, hi=hi),
            "vs_b": incremental_vs_parent(a_kept, kept_b, target, horizon=horizon, lo=lo, hi=hi),
        }

    floors = price_floor_audit(stated, pattern, specs, target, cutoffs=cutoffs)
    regimes = regime_audit(
        stated, pattern, specs, target, analysis_start=analysis_start, analysis_end=analysis_end, cutoffs=cutoffs,
    )
    return {
        "pattern": pattern,
        "frozen_quantiles": frozen,
        "modes": blocks,
        "incremental": incremental,
        "overlap": overlap,
        "price_floor": floors,
        "regimes": regimes,
        "notes": [
            "Quantile/winsor frozen on ANALYSIS.",
            "VALIDATION does not retune thresholds.",
            "FUTURE_HOLDOUT is not evaluated.",
        ],
    }
