"""Cross-sectional feature rank-correlation on DEVELOPMENT only.

Does not drop features. Flags pairs whose mean daily rank correlation
has absolute value >= RANK_CORR_REDUNDANT_ABS.
"""

from __future__ import annotations

import polars as pl

from app.research.config import RANK_CORR_REDUNDANT_ABS, REDUNDANCY_FEATURES, SPLIT_DEVELOPMENT
from app.research.dataset import filter_split


def feature_rank_correlation(
    df: pl.DataFrame,
    features: tuple[str, ...] | list[str] = REDUNDANCY_FEATURES,
    split: str = SPLIT_DEVELOPMENT,
    redundant_abs: float = RANK_CORR_REDUNDANT_ABS,
) -> pl.DataFrame:
    work = filter_split(df, split)
    present = [f for f in features if f in work.columns]
    if len(present) < 2 or work.height == 0:
        return pl.DataFrame()
    ranked = work.select(["date", *present])
    for feat in present:
        ranked = ranked.with_columns(pl.col(feat).rank(method="average").over("date").alias(f"_r_{feat}"))
    rows: list[dict] = []
    for i, a in enumerate(present):
        for b in present[i:]:
            if a == b:
                n_dates = ranked.filter(pl.col(a).is_not_null())["date"].n_unique()
                rows.append(
                    {
                        "feature_a": a,
                        "feature_b": b,
                        "mean_rank_corr": 1.0,
                        "n_dates": n_dates,
                        "redundant": False,
                        "split": split,
                    }
                )
                continue
            pair = ranked.filter(pl.col(a).is_not_null() & pl.col(b).is_not_null())
            if pair.height == 0:
                continue
            daily = (
                pair.group_by("date")
                .agg(pl.corr(pl.col(f"_r_{a}"), pl.col(f"_r_{b}")).alias("corr"))
                .filter(pl.col("corr").is_not_null())
            )
            n = daily.height
            mu = float(daily["corr"].mean()) if n else None
            rows.append(
                {
                    "feature_a": a,
                    "feature_b": b,
                    "mean_rank_corr": mu,
                    "n_dates": n,
                    "redundant": bool(mu is not None and abs(mu) >= redundant_abs),
                    "split": split,
                }
            )
    return pl.DataFrame(rows) if rows else pl.DataFrame()
