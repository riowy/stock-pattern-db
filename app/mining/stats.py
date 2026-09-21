"""Pattern statistics: robust means, date-clustered Newey-West, BH FDR."""

from __future__ import annotations

import math

import polars as pl

from app.research.statistics import newey_west_tstat


def bh_qvalues(pvalues: list[float | None]) -> list[float | None]:
    indexed = [(i, p) for i, p in enumerate(pvalues) if p is not None and math.isfinite(p)]
    m = len(indexed)
    q: list[float | None] = [None] * len(pvalues)
    if m == 0:
        return q
    indexed.sort(key=lambda t: t[1])
    raw: list[tuple[int, float]] = []
    for rank, (i, p) in enumerate(indexed, start=1):
        raw.append((i, min(1.0, p * m / rank)))
    running = 1.0
    for i, val in reversed(raw):
        running = min(running, val)
        q[i] = running
    return q


def winsor_mean(values: pl.Series, lo: float | None, hi: float | None) -> float | None:
    if values.len() == 0:
        return None
    if lo is None or hi is None:
        return float(values.mean()) if values.len() else None
    return float(values.clip(lower_bound=lo, upper_bound=hi).mean())


def two_sided_p_from_t(t: float | None, df: int) -> float | None:
    if t is None or df < 2 or not math.isfinite(t):
        return None
    # Normal approximation is enough for large date counts.
    z = abs(t)
    return math.erfc(z / math.sqrt(2.0))


def daily_baseline(frame: pl.DataFrame, target: str) -> pl.DataFrame:
    work = frame.filter(pl.col(target).is_not_null())
    if work.height == 0:
        return pl.DataFrame({"date": [], "all_mean": []})
    return work.group_by("date").agg(pl.col(target).mean().alias("all_mean")).sort("date")


def pattern_stats(
    frame: pl.DataFrame,
    target: str,
    *,
    excess: str | None,
    horizon: int,
    lo: float | None,
    hi: float | None,
    baseline_daily: pl.DataFrame | None = None,
) -> dict:
    work = frame.filter(pl.col(target).is_not_null())
    n = work.height
    dates = work["date"].n_unique() if n else 0
    secs = work["security_id"].n_unique() if n and "security_id" in work.columns else 0
    s = work[target]
    ex = work[excess] if excess and excess in work.columns else None
    mean = float(s.mean()) if n else None
    median = float(s.median()) if n else None
    wmean = winsor_mean(s, lo, hi)
    pos = float((s > 0).mean()) if n else None
    ex_mean = float(ex.mean()) if ex is not None and ex.len() else None
    ex_median = float(ex.median()) if ex is not None and ex.len() else None
    daily = (
        work.group_by("date")
        .agg(pl.col(target).mean().alias("cs_mean"))
        .sort("date")
    )
    daily_vals = [float(v) for v in daily["cs_mean"].to_list() if v is not None and math.isfinite(v)]
    nw = newey_west_tstat(daily_vals, horizon)
    daily_mean = sum(daily_vals) / len(daily_vals) if daily_vals else None
    daily_median = float(daily["cs_mean"].median()) if daily.height else None
    daily_pos = sum(1 for v in daily_vals if v > 0) / len(daily_vals) if daily_vals else None
    vs_mean = None
    vs_nw = None
    diffs: list[float] = []
    if baseline_daily is not None and baseline_daily.height and daily.height and "all_mean" in baseline_daily.columns:
        joined = daily.join(baseline_daily, on="date", how="inner")
        diffs = [
            float(v)
            for v in (joined["cs_mean"] - joined["all_mean"]).to_list()
            if v is not None and math.isfinite(v)
        ]
        if diffs:
            vs_mean = sum(diffs) / len(diffs)
            vs_nw = newey_west_tstat(diffs, horizon)
    p_src = vs_nw if vs_nw is not None else nw
    p_n = len(diffs) if vs_nw is not None else len(daily_vals)
    p_df = max(p_n - 1, 1)
    return {
        "n": n,
        "n_dates": dates,
        "n_securities": secs,
        "mean": mean,
        "median": median,
        "winsor_mean": wmean,
        "positive_ratio": pos,
        "excess_mean": ex_mean,
        "excess_median": ex_median,
        "daily_mean": daily_mean,
        "daily_median": daily_median,
        "daily_positive_ratio": daily_pos,
        "newey_west_t": nw,
        "vs_baseline_mean": vs_mean,
        "vs_baseline_nw_t": vs_nw,
        "p_value": two_sided_p_from_t(p_src, p_df),
    }


def horizon_of(target: str) -> int:
    token = target.rsplit("_", 1)[-1]
    if token.endswith("d") and token[:-1].isdigit():
        return int(token[:-1])
    return 20
