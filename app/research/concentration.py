"""Low-price concentration of extreme future returns. Descriptive only."""

from __future__ import annotations

import polars as pl

from app.research.config import PRICE_BUCKETS
from app.research.dataset import filter_split
from app.research.statistics import attach_price_bucket

EXTREME_ABS = 2.0
DEFAULT_TARGET = "forward_return_20d"


def _bucket_share(work: pl.DataFrame, split: str) -> pl.DataFrame:
    n = work.height
    rows: list[dict] = []
    for name, _lo, _hi in PRICE_BUCKETS:
        subset = work.filter(pl.col("price_bucket") == name)
        rows.append(
            {
                "section": "row_share",
                "split": split,
                "price_bucket": name,
                "n": subset.height,
                "share": (subset.height / n) if n else None,
                "ticker": None,
                "date": None,
                "target": None,
                "value": None,
            }
        )
    return pl.DataFrame(rows)


def _extreme_by_bucket(work: pl.DataFrame, split: str, target: str) -> pl.DataFrame:
    if target not in work.columns:
        return pl.DataFrame()
    flagged = work.filter(pl.col(target).is_not_null() & (pl.col(target).abs() > EXTREME_ABS))
    n = flagged.height
    rows: list[dict] = []
    for name, _lo, _hi in PRICE_BUCKETS:
        subset = flagged.filter(pl.col("price_bucket") == name)
        rows.append(
            {
                "section": "extreme_label",
                "split": split,
                "price_bucket": name,
                "n": subset.height,
                "share": (subset.height / n) if n else None,
                "ticker": None,
                "date": None,
                "target": target,
                "value": None,
            }
        )
    return pl.DataFrame(rows)


def _top_tail_by_bucket(
    work: pl.DataFrame, split: str, target: str, frac: float, section: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if target not in work.columns:
        return pl.DataFrame(), pl.DataFrame()
    valid = work.filter(pl.col(target).is_not_null())
    if valid.height == 0:
        return pl.DataFrame(), pl.DataFrame()
    k = max(1, int(round(valid.height * frac)))
    top = valid.sort(target, descending=True).head(k)
    n = top.height
    rows: list[dict] = []
    for name, _lo, _hi in PRICE_BUCKETS:
        subset = top.filter(pl.col("price_bucket") == name)
        rows.append(
            {
                "section": section,
                "split": split,
                "price_bucket": name,
                "n": subset.height,
                "share": (subset.height / n) if n else None,
                "ticker": None,
                "date": None,
                "target": target,
                "value": None,
            }
        )
    return pl.DataFrame(rows), top


def _concentration(top: pl.DataFrame, split: str, section: str, key: str, target: str) -> pl.DataFrame:
    if top.height == 0 or key not in top.columns:
        return pl.DataFrame()
    grouped = (
        top.group_by(key)
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(15)
    )
    total = top.height
    ticker_col = "ticker" if key in ("ticker", "security_id") else None
    date_col = "date" if key == "date" else None
    rows: list[dict] = []
    for rec in grouped.iter_rows(named=True):
        rows.append(
            {
                "section": section,
                "split": split,
                "price_bucket": None,
                "n": rec["n"],
                "share": (rec["n"] / total) if total else None,
                "ticker": rec[key] if ticker_col else None,
                "date": rec[key] if date_col else None,
                "target": target,
                "value": None,
            }
        )
    return pl.DataFrame(rows)


def low_price_concentration(
    df: pl.DataFrame,
    split: str,
    target: str = DEFAULT_TARGET,
) -> pl.DataFrame:
    work = attach_price_bucket(filter_split(df, split))
    if work.height == 0:
        return pl.DataFrame()
    parts: list[pl.DataFrame] = [_bucket_share(work, split), _extreme_by_bucket(work, split, target)]
    top1_tbl, top1 = _top_tail_by_bucket(work, split, target, 0.01, "top_1pct")
    top01_tbl, top01 = _top_tail_by_bucket(work, split, target, 0.001, "top_0_1pct")
    parts.extend([top1_tbl, top01_tbl])
    ticker_key = "ticker" if "ticker" in work.columns else "security_id"
    parts.append(_concentration(top1, split, "ticker_concentration_top_1pct", ticker_key, target))
    parts.append(_concentration(top1, split, "date_concentration_top_1pct", "date", target))
    parts.append(_concentration(top01, split, "ticker_concentration_top_0_1pct", ticker_key, target))
    parts.append(_concentration(top01, split, "date_concentration_top_0_1pct", "date", target))
    kept = [p for p in parts if p is not None and p.height]
    if not kept:
        return pl.DataFrame()
    return pl.concat(kept, how="diagonal_relaxed")
