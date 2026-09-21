"""STATE vs ENTRY_EVENT, episodes, and cooldown. Memory-only."""

from __future__ import annotations

import numpy as np
import polars as pl

from app.indicators.common import SID, apply_by_security

NATIVE_EVENTS = frozenset(
    {
        "golden_cross",
        "death_cross",
        "macd_cross_up",
        "squeeze_release",
        "tk_cross_up",
        "tk_cross_down",
        "supertrend_flip_up",
        "bb_reentry_lower",
    }
)


def pattern_parts(pattern: str) -> list[str]:
    return [p.strip() for p in pattern.split(" AND ") if p.strip()]


def conjunction(names: list[str]) -> pl.Expr:
    expr = pl.lit(True)
    for name in names:
        expr = expr & (pl.col(name) == True)  # noqa: E712
    return expr


def add_state_flag(df: pl.DataFrame, names: list[str]) -> pl.DataFrame:
    missing = [n for n in names if n not in df.columns]
    if missing:
        return df.with_columns(pl.lit(False).alias("_flag"))
    return df.with_columns(conjunction(names).fill_null(False).alias("_flag"))


def add_entry_and_episode(df: pl.DataFrame) -> pl.DataFrame:
    if "_flag" not in df.columns:
        raise ValueError("add_state_flag first")
    work = df.sort([SID, "date"])
    prev = pl.col("_flag").shift(1).over(SID).fill_null(False)
    entry = pl.col("_flag") & ~prev
    work = work.with_columns(entry.alias("_entry"))
    # episode_id increments on each entry; null when state is false
    work = work.with_columns(
        pl.when(pl.col("_flag"))
        .then(pl.col("_entry").cast(pl.Int32).cum_sum().over(SID))
        .otherwise(None)
        .alias("_episode")
    )
    return work


def _cooldown_np(flag: np.ndarray, cooldown: int) -> np.ndarray:
    keep = np.zeros(flag.size, dtype=np.float64)
    last = -10**9
    for i, raw in enumerate(flag):
        on = bool(raw) and raw == raw  # not nan
        if on and i > last + cooldown:
            keep[i] = 1.0
            last = i
    return keep


def apply_cooldown(df: pl.DataFrame, cooldown_sessions: int, source: str = "_entry") -> pl.DataFrame:
    if cooldown_sessions <= 0:
        return df.with_columns(pl.col(source).alias("_keep"))
    work = df.with_columns(pl.col(source).fill_null(False).cast(pl.Float64).alias("_cd_src"))
    work = apply_by_security(
        work,
        lambda x, cd=cooldown_sessions: _cooldown_np(x, cd),
        ["_cd_src"],
        ["_keep"],
    )
    return work.drop(["_cd_src"]).with_columns((pl.col("_keep") == 1.0).alias("_keep"))


def activate(
    df: pl.DataFrame,
    pattern: str,
    *,
    event_mode: str = "state",
    cooldown_sessions: int = 0,
) -> pl.DataFrame:
    """Return rows with _flag/_entry/_episode/_keep. Caller filters on _keep for outcomes."""
    names = pattern_parts(pattern)
    work = add_state_flag(df, names)
    work = add_entry_and_episode(work)
    source = "_entry" if event_mode == "entry" else "_flag"
    if event_mode == "entry" and cooldown_sessions > 0:
        work = apply_cooldown(work, cooldown_sessions, source="_entry")
    elif event_mode == "state":
        work = work.with_columns(pl.col("_flag").alias("_keep"))
    else:
        work = work.with_columns(pl.col("_entry").alias("_keep"))
    return work


def frequency_counts(activated: pl.DataFrame) -> dict:
    """Observation counts. Episodes always come from consecutive STATE runs."""
    n_rows = int(activated["_flag"].sum()) if activated.height else 0
    n_entry = int(activated["_entry"].sum()) if activated.height and "_entry" in activated.columns else 0
    n_keep = int(activated["_keep"].sum()) if activated.height and "_keep" in activated.columns else 0
    n_ep = 0
    if activated.height and "_episode" in activated.columns:
        tagged = activated.filter(pl.col("_episode").is_not_null())
        if tagged.height:
            n_ep = tagged.select(["security_id", "_episode"]).n_unique()
    kept = activated.filter(pl.col("_keep") == True) if "_keep" in activated.columns else activated  # noqa: E712
    n_dates = kept["date"].n_unique() if kept.height else 0
    n_secs = kept["security_id"].n_unique() if kept.height and "security_id" in kept.columns else 0
    return {
        "rows": n_rows,
        "entry_events": n_entry,
        "episodes": n_ep,
        "kept": n_keep,
        "distinct_dates": n_dates,
        "distinct_securities": n_secs,
    }
