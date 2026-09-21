"""Development-frozen winsor cutoffs and robust univariate summaries.

Cutoffs are computed on DEVELOPMENT only and applied unchanged to validation.
Original label columns in the lake are never modified; clipping exists only
in research output frames.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from app.research.config import (
    ALL_TARGETS,
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    N_QUANTILES,
    RESEARCH_VERSION_V1,
    SPLIT_DEVELOPMENT,
    WINSOR_P_HIGH,
    WINSOR_P_LOW,
)
from app.research.dataset import filter_split
from app.research.statistics import (
    _robust_from_series,
    assign_cs_quintiles,
    excess_target_name,
    sample_status,
)
from app.utils.time_utils import utc_now


def horizon_from_target(target: str) -> int | None:
    token = target.rsplit("_", 1)[-1]
    if token.endswith("d") and token[:-1].isdigit():
        return int(token[:-1])
    return None


def freeze_target_cutoffs(
    development_df: pl.DataFrame,
    targets: tuple[str, ...] | list[str] = ALL_TARGETS,
    *,
    research_version: str = RESEARCH_VERSION_V1,
    p_low: float = WINSOR_P_LOW,
    p_high: float = WINSOR_P_HIGH,
    calculated_at: datetime | None = None,
) -> pl.DataFrame:
    """p1/p99 per target from DEVELOPMENT only. Do not pass validation rows."""
    work = filter_split(development_df, SPLIT_DEVELOPMENT) if "split" in development_df.columns else development_df
    stamp = calculated_at or utc_now()
    rows: list[dict] = []
    for target in targets:
        if target not in work.columns:
            continue
        s = work[target].drop_nulls()
        n = s.len()
        lower = float(s.quantile(p_low)) if n else None
        upper = float(s.quantile(p_high)) if n else None
        rows.append(
            {
                "research_version": research_version,
                "target": target,
                "development_start": DEVELOPMENT_START.isoformat(),
                "development_end": DEVELOPMENT_END.isoformat(),
                "lower_cutoff": lower,
                "upper_cutoff": upper,
                "percentile_low": p_low,
                "percentile_high": p_high,
                "n_development": n,
                "calculated_at": stamp,
            }
        )
    return pl.DataFrame(rows)


def cutoff_map(cutoffs: pl.DataFrame) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    if cutoffs.height == 0:
        return out
    for rec in cutoffs.iter_rows(named=True):
        lo, hi = rec.get("lower_cutoff"), rec.get("upper_cutoff")
        if lo is None or hi is None:
            continue
        out[str(rec["target"])] = (float(lo), float(hi))
    return out


def apply_winsor_columns(df: pl.DataFrame, cutoffs: pl.DataFrame | dict[str, tuple[float, float]]) -> pl.DataFrame:
    """Add ``{target}_winsor`` columns. Leaves original label columns untouched."""
    mapping = cutoff_map(cutoffs) if isinstance(cutoffs, pl.DataFrame) else cutoffs
    exprs: list[pl.Expr] = []
    for target, (lo, hi) in mapping.items():
        if target not in df.columns:
            continue
        exprs.append(pl.col(target).clip(lower_bound=lo, upper_bound=hi).alias(f"{target}_winsor"))
    if not exprs:
        return df
    return df.with_columns(exprs)


def robust_baseline_table(
    df: pl.DataFrame,
    split: str,
    cutoffs: dict[str, tuple[float, float]] | None = None,
) -> pl.DataFrame:
    work = filter_split(df, split)
    rows: list[dict] = []
    if work.height == 0:
        return pl.DataFrame()
    n_dates = work["date"].n_unique() if "date" in work.columns else 0
    for col in ALL_TARGETS:
        if col not in work.columns:
            continue
        cols = [col]
        excess_name = excess_target_name(col)
        if excess_name != col and excess_name in work.columns:
            cols.append(excess_name)
        if "date" in work.columns:
            cols.append("date")
        pair = work.select(cols).drop_nulls(subset=col)
        s = pair[col]
        excess = pair[excess_target_name(col)] if excess_target_name(col) in pair.columns else None
        lo_hi = (cutoffs or {}).get(col)
        stats = _robust_from_series(s, lo=lo_hi[0] if lo_hi else None, hi=lo_hi[1] if lo_hi else None, excess=excess)
        rows.append(
            {
                "split": split,
                "target": col,
                "horizon": horizon_from_target(col),
                "distinct_dates": n_dates,
                **stats,
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def robust_quantile_table(
    df: pl.DataFrame,
    features: list[str],
    split: str,
    cutoffs: dict[str, tuple[float, float]] | None = None,
) -> pl.DataFrame:
    """Long table: one row per feature / quintile / horizon(target) / split."""
    work = filter_split(df, split)
    rows: list[dict] = []
    for feature in features:
        qdf = assign_cs_quintiles(work, feature)
        if qdf.height == 0:
            continue
        for q in range(1, N_QUANTILES + 1):
            subset = qdf.filter(pl.col("quintile") == q)
            n_dates = subset["date"].n_unique() if subset.height and "date" in subset.columns else 0
            for col in ALL_TARGETS:
                if col not in subset.columns:
                    continue
                keep = [col]
                excess_name = excess_target_name(col)
                if excess_name != col and excess_name in subset.columns:
                    keep.append(excess_name)
                pair = subset.select(keep).drop_nulls(subset=col)
                s = pair[col]
                excess = pair[excess_name] if excess_name in pair.columns else None
                lo_hi = (cutoffs or {}).get(col)
                stats = _robust_from_series(
                    s, lo=lo_hi[0] if lo_hi else None, hi=lo_hi[1] if lo_hi else None, excess=excess
                )
                rows.append(
                    {
                        "feature": feature,
                        "split": split,
                        "quintile": q,
                        "target": col,
                        "horizon": horizon_from_target(col),
                        "distinct_dates": n_dates,
                        "sample_status": sample_status(stats["sample_count"], n_dates),
                        **stats,
                    }
                )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def robust_spread_table(robust_quantiles: pl.DataFrame) -> pl.DataFrame:
    if robust_quantiles.height == 0:
        return pl.DataFrame()
    rows: list[dict] = []
    keys = robust_quantiles.select(["feature", "split", "target"]).unique()
    for rec in keys.iter_rows(named=True):
        filt = robust_quantiles.filter(
            (pl.col("feature") == rec["feature"])
            & (pl.col("split") == rec["split"])
            & (pl.col("target") == rec["target"])
        )
        q1 = filt.filter(pl.col("quintile") == 1)
        q5 = filt.filter(pl.col("quintile") == 5)
        if q1.height == 0 or q5.height == 0:
            continue
        r1 = q1.row(0, named=True)
        r5 = q5.row(0, named=True)

        def _diff(key: str) -> float | None:
            a, b = r5.get(key), r1.get(key)
            if a is None or b is None:
                return None
            return float(a) - float(b)

        rows.append(
            {
                "feature": rec["feature"],
                "split": rec["split"],
                "target": rec["target"],
                "horizon": r5.get("horizon"),
                "q1_n": r1.get("sample_count"),
                "q5_n": r5.get("sample_count"),
                "raw_mean_spread": _diff("mean"),
                "median_spread": _diff("median"),
                "winsorized_mean_spread": _diff("winsorized_mean_1pct"),
                "q5_raw_mean": r5.get("mean"),
                "q1_raw_mean": r1.get("mean"),
                "q5_median": r5.get("median"),
                "q1_median": r1.get("median"),
                "q5_winsorized_mean": r5.get("winsorized_mean_1pct"),
                "q1_winsorized_mean": r1.get("winsorized_mean_1pct"),
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()
