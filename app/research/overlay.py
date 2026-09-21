"""Price-level sensitivity overlay. Absolute price is not an alpha feature."""

from __future__ import annotations

import polars as pl

from app.research.config import PRICE_BUCKETS, PRICE_FLOOR_SUBSETS, SUMMARY_TARGET
from app.research.dataset import filter_split
from app.research.robust import robust_quantile_table, robust_spread_table
from app.research.statistics import (
    attach_price_bucket,
    baseline_table,
    ic_table,
    overlapping_spread_table,
)


def filter_price_floor(df: pl.DataFrame, floor: float | None) -> pl.DataFrame:
    if floor is None:
        return df
    if "raw_close" not in df.columns:
        return df.clear()
    return df.filter(pl.col("raw_close") >= floor)


def _subset_frames(df: pl.DataFrame) -> list[tuple[str, pl.DataFrame]]:
    out: list[tuple[str, pl.DataFrame]] = []
    for name, floor in PRICE_FLOOR_SUBSETS:
        out.append((name, filter_price_floor(df, floor)))
    bucketed = attach_price_bucket(df)
    for name, _lo, _hi in PRICE_BUCKETS:
        out.append((name, bucketed.filter(pl.col("price_bucket") == name)))
    return out


def overlay_tables(
    df: pl.DataFrame,
    features: list[str],
    split: str,
    cutoffs: dict[str, tuple[float, float]] | None = None,
) -> dict[str, pl.DataFrame]:
    """Sample size, baseline, IC, quintile spread, overlapping spread per price subset."""
    work = filter_split(df, split)
    ic_rows: list[pl.DataFrame] = []
    spread_rows: list[pl.DataFrame] = []
    overlap_rows: list[pl.DataFrame] = []
    size_rows: list[dict] = []
    baseline_rows: list[pl.DataFrame] = []

    for name, subset in _subset_frames(work):
        n = subset.height
        n_dates = subset["date"].n_unique() if n and "date" in subset.columns else 0
        n_sec = subset["security_id"].n_unique() if n and "security_id" in subset.columns else 0
        size_rows.append(
            {
                "split": split,
                "price_subset": name,
                "n": n,
                "n_dates": n_dates,
                "n_securities": n_sec,
            }
        )
        if n == 0:
            continue
        tagged = subset.with_columns(pl.lit(split).alias("split")) if "split" not in subset.columns else subset
        base = baseline_table(tagged, split)
        if base.height:
            baseline_rows.append(base.with_columns(pl.lit(name).alias("price_subset")))
        ic = ic_table(tagged, features, split)
        if ic.height:
            ic_rows.append(ic.with_columns(pl.lit(name).alias("price_subset")))
        rq = robust_quantile_table(tagged, features, split, cutoffs=cutoffs)
        sp = robust_spread_table(rq)
        if sp.height:
            spread_rows.append(sp.with_columns(pl.lit(name).alias("price_subset")))
        ov = overlapping_spread_table(tagged, features, split)
        if ov.height:
            overlap_rows.append(ov.with_columns(pl.lit(name).alias("price_subset")))

    def _cat(parts: list[pl.DataFrame]) -> pl.DataFrame:
        kept = [p for p in parts if p.height]
        if not kept:
            return pl.DataFrame()
        if len(kept) == 1:
            return kept[0]
        return pl.concat(kept, how="diagonal_relaxed")

    return {
        "sizes": pl.DataFrame(size_rows) if size_rows else pl.DataFrame(),
        "baseline": _cat(baseline_rows),
        "ic": _cat(ic_rows),
        "spread": _cat(spread_rows),
        "overlapping": _cat(overlap_rows),
    }


def overlay_ic_for_target(overlay_ic: pl.DataFrame, target: str = SUMMARY_TARGET) -> pl.DataFrame:
    if overlay_ic.height == 0 or "target" not in overlay_ic.columns:
        return overlay_ic
    return overlay_ic.filter(pl.col("target") == target)
