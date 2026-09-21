"""Pair incremental stats, overlap, concentration, yearly slices. Memory-only."""

from __future__ import annotations

from datetime import date

import math
import polars as pl

from app.mining.config import HIGH_OVERLAP, OVERLAP_CONDITIONAL, OVERLAP_JACCARD
from app.mining.events import activate, frequency_counts, pattern_parts
from app.mining.stats import horizon_of, pattern_stats, winsor_mean
from app.research.statistics import newey_west_tstat


def jaccard_flags(a: pl.Series, b: pl.Series) -> float | None:
    if a.len() == 0:
        return None
    both = (a.fill_null(False) & b.fill_null(False)).sum()
    union = (a.fill_null(False) | b.fill_null(False)).sum()
    if union == 0:
        return None
    return float(both) / float(union)


def overlap_report(frame: pl.DataFrame, left: str, right: str) -> dict:
    """Activation similarity of two boolean columns already on the frame."""
    if left not in frame.columns or right not in frame.columns:
        return {"jaccard": None, "p_right_given_left": None, "p_left_given_right": None, "flag": None}
    a = frame[left].fill_null(False)
    b = frame[right].fill_null(False)
    n_a = int(a.sum())
    n_b = int(b.sum())
    both = int((a & b).sum())
    jac = jaccard_flags(a, b)
    p_ba = (both / n_a) if n_a else None
    p_ab = (both / n_b) if n_b else None
    flag = None
    if (p_ba is not None and p_ba > OVERLAP_CONDITIONAL) or (p_ab is not None and p_ab > OVERLAP_CONDITIONAL):
        flag = HIGH_OVERLAP
    if jac is not None and jac > OVERLAP_JACCARD:
        flag = HIGH_OVERLAP
    return {
        "jaccard": jac,
        "p_right_given_left": p_ba,
        "p_left_given_right": p_ab,
        "intersection": both,
        "n_left": n_a,
        "n_right": n_b,
        "flag": flag,
    }


def pair_self_overlap(frame: pl.DataFrame, pattern: str) -> dict:
    parts = pattern_parts(pattern)
    if len(parts) != 2:
        return {"jaccard": None, "flag": None}
    left, right = parts
    if left not in frame.columns or right not in frame.columns:
        return {"jaccard": None, "flag": None}
    work = frame.with_columns((pl.col(left).fill_null(False) & pl.col(right).fill_null(False)).alias("_pair"))
    vs_a = overlap_report(work, left, "_pair")
    vs_b = overlap_report(work, right, "_pair")
    ab = overlap_report(work, left, right)
    # Pair is a subset of each parent, so P(parent|pair)=1 always. Overlap vs a parent
    # is P(pair|parent)=P(other|parent), i.e. Jaccard(pair, parent).
    jac_pair_a = vs_a.get("jaccard")
    jac_pair_b = vs_b.get("jaccard")
    flag = None
    p_ba = ab.get("p_right_given_left")
    p_ab = ab.get("p_left_given_right")
    jac = ab.get("jaccard")
    if (p_ba is not None and p_ba > OVERLAP_CONDITIONAL) or (p_ab is not None and p_ab > OVERLAP_CONDITIONAL):
        flag = HIGH_OVERLAP
    if jac is not None and jac > OVERLAP_JACCARD:
        flag = HIGH_OVERLAP
    if jac_pair_a is not None and jac_pair_a > OVERLAP_JACCARD:
        flag = HIGH_OVERLAP
    if jac_pair_b is not None and jac_pair_b > OVERLAP_JACCARD:
        flag = HIGH_OVERLAP
    return {
        "jaccard": jac,
        "p_b_given_a": p_ba,
        "p_a_given_b": p_ab,
        "jaccard_pair_vs_a": jac_pair_a,
        "jaccard_pair_vs_b": jac_pair_b,
        "flag": flag,
    }


def incremental_vs_parent(
    pair_rows: pl.DataFrame,
    parent_rows: pl.DataFrame,
    target: str,
    *,
    horizon: int,
    lo: float | None,
    hi: float | None,
) -> dict:
    """Pair minus parent, using date-clustered CS means on overlapping dates."""
    pair_s = pattern_stats(pair_rows, target, excess=None, horizon=horizon, lo=lo, hi=hi)
    parent_s = pattern_stats(parent_rows, target, excess=None, horizon=horizon, lo=lo, hi=hi)
    p_med = pair_s.get("excess_median") if pair_s.get("excess_median") is not None else pair_s.get("median")
    a_med = parent_s.get("excess_median") if parent_s.get("excess_median") is not None else parent_s.get("median")
    inc_med = (p_med - a_med) if p_med is not None and a_med is not None else None
    inc_win = None
    if pair_s.get("winsor_mean") is not None and parent_s.get("winsor_mean") is not None:
        inc_win = pair_s["winsor_mean"] - parent_s["winsor_mean"]
    pair_d = (
        pair_rows.filter(pl.col(target).is_not_null())
        .group_by("date")
        .agg(pl.col(target).mean().alias("pair_mean"))
    )
    parent_d = (
        parent_rows.filter(pl.col(target).is_not_null())
        .group_by("date")
        .agg(pl.col(target).mean().alias("parent_mean"))
    )
    joined = pair_d.join(parent_d, on="date", how="inner")
    diffs = [
        float(v)
        for v in (joined["pair_mean"] - joined["parent_mean"]).to_list()
        if v is not None and math.isfinite(v)
    ]
    inc_daily = sum(diffs) / len(diffs) if diffs else None
    inc_nw = newey_west_tstat(diffs, horizon) if diffs else None
    return {
        "parent_n": parent_s.get("n"),
        "parent_median": a_med,
        "parent_winsor_mean": parent_s.get("winsor_mean"),
        "incremental_median": inc_med,
        "incremental_winsor_mean": inc_win,
        "incremental_daily_mean": inc_daily,
        "incremental_nw_t": inc_nw,
    }


def security_concentration(activated_keep: pl.DataFrame) -> dict:
    if activated_keep.height == 0 or "security_id" not in activated_keep.columns:
        return {"top_1_security_share": None, "top_5_security_share": None, "top_security": None}
    n = activated_keep.height
    counts = (
        activated_keep.group_by("security_id")
        .len()
        .rename({"len": "n"})
        .sort("n", descending=True)
    )
    top1 = float(counts["n"][0]) / n if counts.height else None
    top5 = float(counts.head(5)["n"].sum()) / n if counts.height else None
    top_id = counts["security_id"][0] if counts.height else None
    ticker = None
    if top_id is not None:
        tcol = "ticker" if "ticker" in activated_keep.columns else ("ticker_at_time" if "ticker_at_time" in activated_keep.columns else None)
        if tcol:
            hit = activated_keep.filter(pl.col("security_id") == top_id)[tcol]
            if hit.len():
                ticker = hit[0]
    return {
        "top_1_security_share": top1,
        "top_5_security_share": top5,
        "top_security": ticker or top_id,
        "top_security_id": top_id,
    }


def leave_one_top_security(
    kept: pl.DataFrame,
    target: str,
    *,
    horizon: int,
    lo: float | None,
    hi: float | None,
) -> dict:
    conc = security_concentration(kept)
    top_id = conc.get("top_security_id")
    full = pattern_stats(kept, target, excess=None, horizon=horizon, lo=lo, hi=hi)
    if top_id is None:
        return {"dropped": None, "full_median": full.get("median"), "dropped_median": None, "delta_median": None}
    dropped = kept.filter(pl.col("security_id") != top_id)
    drop_s = pattern_stats(dropped, target, excess=None, horizon=horizon, lo=lo, hi=hi)
    full_m = full.get("excess_median") if full.get("excess_median") is not None else full.get("median")
    drop_m = drop_s.get("excess_median") if drop_s.get("excess_median") is not None else drop_s.get("median")
    delta = (drop_m - full_m) if drop_m is not None and full_m is not None else None
    return {
        "dropped": conc.get("top_security"),
        "full_median": full_m,
        "dropped_median": drop_m,
        "delta_median": delta,
        "dropped_n": drop_s.get("n"),
    }


def year_breakdown(
    kept: pl.DataFrame,
    target: str,
    *,
    lo: float | None,
    hi: float | None,
) -> list[dict]:
    if kept.height == 0:
        return []
    work = kept.filter(pl.col(target).is_not_null()).with_columns(pl.col("date").dt.year().alias("_year"))
    rows = []
    for year in range(2018, 2027):
        subset = work.filter(pl.col("_year") == year)
        n = subset.height
        if n == 0:
            rows.append({"year": year, "n": 0, "median_excess": None, "winsor_mean": None, "positive_ratio": None})
            continue
        s = subset[target]
        rows.append(
            {
                "year": year,
                "n": n,
                "median_excess": float(s.median()) if n else None,
                "winsor_mean": winsor_mean(s, lo, hi),
                "positive_ratio": float((s > 0).mean()),
            }
        )
    return rows


def outcome_subset(
    frame: pl.DataFrame,
    pattern: str,
    *,
    event_mode: str,
    cooldown_sessions: int,
    start: date | None = None,
    end: date | None = None,
) -> tuple[pl.DataFrame, dict]:
    work = frame
    if start is not None:
        work = work.filter(pl.col("date") >= start)
    if end is not None:
        work = work.filter(pl.col("date") <= end)
    act = activate(work, pattern, event_mode=event_mode, cooldown_sessions=cooldown_sessions)
    freq = frequency_counts(act)
    kept = act.filter(pl.col("_keep") == True)  # noqa: E712
    return kept, freq
