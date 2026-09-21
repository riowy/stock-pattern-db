"""Historical expanding-window walk-forward. Fold cutoffs freeze on that fold's discover window."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from datetime import date
from typing import Any

import polars as pl

from app.mining.config import (
    DEFAULT_FDR_Q,
    DEFAULT_MAX_RULE_SIZE,
    MIN_PATTERN_DATES,
    MIN_PATTERN_ROWS,
    MIN_PATTERN_SECURITIES,
    SMOKE_PATTERNS,
    WALK_FOLDS,
    WalkFold,
    cap_end_before_future_holdout,
)
from app.mining.engine import MiningEngine, _score_pattern
from app.mining.persist import assert_no_mining_persist
from app.mining.selection_diag import summarize_from_raw
from app.mining.states import apply_states
from app.mining.stats import daily_baseline, horizon_of


def _effect(stats: dict) -> float | None:
    if stats.get("excess_median") is not None:
        return stats["excess_median"]
    return stats.get("median")


def _dir(value: float | None) -> int:
    if value is None:
        return 0
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def fold_eval_end(fold: WalkFold, mature: date | None) -> date | None:
    end = fold.eval_end
    if end is None:
        end = mature
    return cap_end_before_future_holdout(end)


def walk_forward(
    frame: pl.DataFrame,
    *,
    target: str,
    tracked: tuple[str, ...] = SMOKE_PATTERNS,
    max_rule_size: int = DEFAULT_MAX_RULE_SIZE,
    min_rows: int = MIN_PATTERN_ROWS,
    min_dates: int = MIN_PATTERN_DATES,
    min_securities: int = MIN_PATTERN_SECURITIES,
    fdr_q: float = DEFAULT_FDR_Q,
    event_mode: str = "state",
    cooldown_sessions: int = 0,
    mature: date | None = None,
    full_discover: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict:
    """Each fold freezes cutoffs on its discover window; evaluate only the next year. No FUTURE_HOLDOUT eval."""
    assert_no_mining_persist()
    engine = MiningEngine()
    horizon = horizon_of(target)
    fold_rows = []
    tracked_folds: dict[str, list[dict]] = defaultdict(list)
    last_evaluated: list = []
    n_folds = len(WALK_FOLDS)

    for fold_i, fold in enumerate(WALK_FOLDS, start=1):
        eval_end = fold_eval_end(fold, mature)
        if eval_end is not None and eval_end < fold.eval_start:
            if full_discover and progress is not None:
                progress({"event": "skipped", "fold": fold_i, "n_folds": n_folds, "reason": "eval window empty / FUTURE_HOLDOUT"})
            fold_rows.append({"fold": fold.name, "skipped": "eval window empty / FUTURE_HOLDOUT"})
            continue
        discovered, specs, frozen, cutoffs = ([], [], {}, {})
        n_selected = 0
        fold_selection = None
        if full_discover:
            if progress is not None:
                progress({"event": "fold_start", "fold": fold_i, "n_folds": n_folds, "name": fold.name})
            discovered, specs, frozen, cutoffs = engine.discover(
                frame,
                target=target,
                analysis_start=fold.discover_start,
                analysis_end=fold.discover_end,
                min_rows=min_rows,
                min_dates=min_dates,
                min_securities=min_securities,
                max_rule_size=max_rule_size,
                fdr_q=fdr_q,
                event_mode=event_mode,
                cooldown_sessions=cooldown_sessions,
                attach_pair_diagnostics=False,
                progress=progress,
            )
            n_selected = sum(1 for r in discovered if r.selected)
            evaluated = engine.evaluate_validation(
                frame,
                discovered,
                specs,
                cutoffs,
                target=target,
                validation_start=fold.eval_start,
                validation_end=eval_end,
                event_mode=event_mode,
                cooldown_sessions=cooldown_sessions,
            )
            last_evaluated = evaluated
            selected_names = {r.pattern for r in evaluated}
            fold_selection = summarize_from_raw(engine.last_selection_raw, tracked)
        else:
            from app.mining.states import base_state_specs, freeze_analysis_quantiles, quantile_state_specs

            discover_df = frame.filter((pl.col("date") >= fold.discover_start) & (pl.col("date") <= fold.discover_end))
            frozen = freeze_analysis_quantiles(discover_df)
            specs = base_state_specs(set(frame.columns)) + quantile_state_specs(set(frame.columns), frozen)
            s = discover_df[target].drop_nulls() if target in discover_df.columns else None
            cutoffs = {
                "lo": float(s.quantile(0.01)) if s is not None and s.len() else None,
                "hi": float(s.quantile(0.99)) if s is not None and s.len() else None,
            }
            selected_names = set()
            evaluated = []

        stated = apply_states(frame, specs)
        eval_df = stated.filter(pl.col("date") >= fold.eval_start)
        if eval_end is not None:
            eval_df = eval_df.filter(pl.col("date") <= eval_end)
        disc_df = stated.filter((pl.col("date") >= fold.discover_start) & (pl.col("date") <= fold.discover_end))
        eval_base = daily_baseline(eval_df, target)
        lo, hi = cutoffs.get("lo"), cutoffs.get("hi")
        for pattern in tracked:
            stats, _kept = _score_pattern(
                eval_df,
                pattern,
                target=target,
                excess=target if target in eval_df.columns else None,
                horizon=horizon,
                lo=lo,
                hi=hi,
                baseline=eval_base,
                event_mode=event_mode,
                cooldown_sessions=cooldown_sessions,
            )
            # Discover-window stats use that fold's freeze only.
            d_stats, _ = _score_pattern(
                disc_df,
                pattern,
                target=target,
                excess=target if target in disc_df.columns else None,
                horizon=horizon,
                lo=lo,
                hi=hi,
                baseline=daily_baseline(disc_df, target),
                event_mode=event_mode,
                cooldown_sessions=cooldown_sessions,
            )
            effect = _effect(stats)
            tracked_sel = None
            if fold_selection:
                tracked_sel = next((t for t in fold_selection.get("tracked") or [] if t.get("pattern") == pattern), None)
            tracked_folds[pattern].append(
                {
                    "fold": fold.name,
                    "eval_start": fold.eval_start,
                    "eval_end": eval_end,
                    "discover_selected": pattern in selected_names,
                    "selection_reason": None if tracked_sel is None else tracked_sel.get("reason"),
                    "selection": tracked_sel,
                    "eval": stats,
                    "discover": d_stats,
                    "effect": effect,
                    "nw_t": stats.get("vs_baseline_nw_t") if stats.get("vs_baseline_nw_t") is not None else stats.get("newey_west_t"),
                    "direction": _dir(effect),
                }
            )

        fold_rows.append(
            {
                "fold": fold.name,
                "discover": f"{fold.discover_start}..{fold.discover_end}",
                "eval": f"{fold.eval_start}..{eval_end}",
                "n_candidates": len(discovered),
                "n_selected": n_selected,
                "full_discover": full_discover,
                "n_tracked": len(tracked),
                "frozen_quantiles": list(frozen.keys()),
                "selection": fold_selection,
            }
        )
        if full_discover and progress is not None:
            if fold_selection is not None:
                progress({"phase": "selection_diagnostics", "diagnostics": fold_selection})
            progress({"phase": "evaluation complete"})
            progress({"event": "fold_elapsed"})

    stability = []
    for pattern, rows in tracked_folds.items():
        effects = [r["effect"] for r in rows if r["effect"] is not None]
        nw = [r["nw_t"] for r in rows if r["nw_t"] is not None]
        dirs = [r["direction"] for r in rows]
        pos = sum(1 for d in dirs if d > 0)
        neg = sum(1 for d in dirs if d < 0)
        # same_direction vs first fold with an effect
        nonzero = [d for d in dirs if d != 0]
        same = 0
        if nonzero:
            base = nonzero[0]
            same = sum(1 for d in nonzero if d == base)
        effects_sorted = sorted(effects)
        nw_sorted = sorted(nw)
        stability.append(
            {
                "pattern": pattern,
                "folds_tested": len(rows),
                "same_direction_folds": same,
                "positive_effect_folds": pos,
                "negative_effect_folds": neg,
                "median_fold_effect": effects_sorted[len(effects_sorted) // 2] if effects_sorted else None,
                "min_fold_effect": effects_sorted[0] if effects_sorted else None,
                "max_fold_effect": effects_sorted[-1] if effects_sorted else None,
                "median_fold_NW_t": nw_sorted[len(nw_sorted) // 2] if nw_sorted else None,
                "folds": rows,
            }
        )
    return {"folds": fold_rows, "stability": stability, "evaluated": last_evaluated}
