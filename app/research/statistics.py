"""Univariate research statistics. Not ranking, sizing, or trade signals.

Development statistics may inform which features deserve more study.
Validation statistics are reported separately and must not change development
cutoffs (VIX regime bounds are frozen from development).
"""

from __future__ import annotations

import math
import polars as pl

from app.research.config import (
    ALL_TARGETS,
    COVERAGE_FEATURES,
    FORWARD_EXCESS_TARGETS,
    FORWARD_RETURN_TARGETS,
    HIGH_NULL_RATIO,
    HORIZONS,
    INSUFFICIENT_SAMPLE,
    MIN_QUANTILE_DATES,
    MIN_QUANTILE_OBSERVATIONS,
    N_QUANTILES,
    PRICE_BUCKETS,
    SPLIT_DEVELOPMENT,
)
from app.research.dataset import filter_split


def sample_status(n_obs: int, n_dates: int) -> str:
    if n_obs >= MIN_QUANTILE_OBSERVATIONS or n_dates >= MIN_QUANTILE_DATES:
        return "ok"
    return INSUFFICIENT_SAMPLE


def tstat(mean: float | None, std: float | None, n: int) -> float | None:
    if mean is None or std is None or n < 2 or std == 0 or math.isnan(std):
        return None
    se = std / math.sqrt(n)
    if se == 0:
        return None
    return mean / se


def newey_west_tstat(values: list[float], lags: int) -> float | None:
    """HAC t-stat for a mean. Lag typically equals the return horizon."""
    n = len(values)
    if n < 3:
        return None
    mu = sum(values) / n
    demean = [v - mu for v in values]
    gamma0 = sum(d * d for d in demean) / n
    nw = gamma0
    max_lag = min(max(lags, 1), n - 1)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1)
        gamma = sum(demean[i] * demean[i - lag] for i in range(lag, n)) / n
        nw += 2.0 * weight * gamma
    if nw <= 0:
        return None
    se = math.sqrt(nw / n)
    if se == 0:
        return None
    return mu / se


def _finite_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def assign_cs_quintiles(df: pl.DataFrame, feature: str) -> pl.DataFrame:
    """Date-by-date cross-sectional quintiles. Does not use other dates' distribution."""
    if feature not in df.columns:
        return df.clear()
    work = df.filter(pl.col(feature).is_not_null())
    if work.height == 0:
        return work
    return work.with_columns(
        pl.col(feature).rank(method="average").over("date").alias("_cs_rank"),
        pl.col(feature).count().over("date").alias("_cs_n"),
    ).with_columns(
        (
            ((pl.col("_cs_rank") - 1) / pl.col("_cs_n") * N_QUANTILES).floor().clip(0, N_QUANTILES - 1).cast(pl.Int32)
            + 1
        ).alias("quintile")
    ).drop(["_cs_rank", "_cs_n"])


def freeze_vix_cutoffs(development_df: pl.DataFrame) -> tuple[float | None, float | None]:
    """Tertile cutoffs from DEVELOPMENT vix_close only. Freeze before validation."""
    if "vix_close" not in development_df.columns or development_df.height == 0:
        return None, None
    vix = development_df["vix_close"].drop_nulls()
    if vix.len() == 0:
        return None, None
    return float(vix.quantile(1.0 / 3.0)), float(vix.quantile(2.0 / 3.0))


def attach_regimes(df: pl.DataFrame, vix_q33: float | None, vix_q67: float | None) -> pl.DataFrame:
    spy_col = "spy_ma200_distance" if "spy_ma200_distance" in df.columns else None
    parts: list[pl.Expr] = []
    if spy_col:
        parts.append(
            pl.when(pl.col(spy_col) > 0)
            .then(pl.lit("spy_above_ma200"))
            .otherwise(pl.lit("spy_below_ma200"))
            .alias("spy_trend_regime")
        )
    else:
        parts.append(pl.lit(None).cast(pl.Utf8).alias("spy_trend_regime"))
    if vix_q33 is None or vix_q67 is None or "vix_close" not in df.columns:
        parts.append(pl.lit(None).cast(pl.Utf8).alias("vix_regime"))
    else:
        parts.append(
            pl.when(pl.col("vix_close").is_null())
            .then(None)
            .when(pl.col("vix_close") <= vix_q33)
            .then(pl.lit("vix_low"))
            .when(pl.col("vix_close") <= vix_q67)
            .then(pl.lit("vix_medium"))
            .otherwise(pl.lit("vix_high"))
            .alias("vix_regime")
        )
    return df.with_columns(parts)


def attach_price_bucket(df: pl.DataFrame) -> pl.DataFrame:
    if "raw_close" not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Utf8).alias("price_bucket"))
    expr = pl.lit(None).cast(pl.Utf8)
    for name, lo, hi in PRICE_BUCKETS:
        cond = pl.col("raw_close").is_not_null()
        if lo is not None:
            cond = cond & (pl.col("raw_close") >= lo)
        if hi is not None:
            cond = cond & (pl.col("raw_close") < hi)
        expr = pl.when(cond).then(pl.lit(name)).otherwise(expr)
    return df.with_columns(expr.alias("price_bucket"))


def _agg_target(col: str) -> list[pl.Expr]:
    return [
        pl.col(col).count().alias(f"{col}_n"),
        pl.col(col).mean().alias(f"{col}_mean"),
        pl.col(col).median().alias(f"{col}_median"),
        pl.col(col).std().alias(f"{col}_std"),
        (pl.col(col) > 0).mean().alias(f"{col}_positive_ratio"),
    ]


def baseline_table(df: pl.DataFrame, split: str) -> pl.DataFrame:
    work = filter_split(df, split)
    rows: list[dict] = []
    if work.height == 0:
        return pl.DataFrame(rows)

    def _emit(year: int | None, subset: pl.DataFrame) -> None:
        for col in ALL_TARGETS:
            if col not in subset.columns:
                continue
            s = subset[col].drop_nulls()
            n = s.len()
            if n == 0:
                continue
            mean = float(s.mean())
            std = float(s.std()) if n > 1 else None
            rows.append(
                {
                    "split": split,
                    "year": year,
                    "target": col,
                    "n": n,
                    "mean": mean,
                    "median": float(s.median()),
                    "std": std,
                    "positive_ratio": float((s > 0).mean()),
                    "tstat": tstat(mean, std, n),
                }
            )

    _emit(None, work)
    if "date" in work.columns:
        years = work.with_columns(pl.col("date").dt.year().alias("_year"))
        for year in sorted(years["_year"].unique().to_list()):
            _emit(int(year), years.filter(pl.col("_year") == year))
    return pl.DataFrame(rows)


def coverage_table(df: pl.DataFrame, split: str, features: tuple[str, ...] | list[str] = COVERAGE_FEATURES) -> pl.DataFrame:
    work = filter_split(df, split)
    n_all = work.height
    rows: list[dict] = []
    for feat in features:
        if feat not in work.columns:
            rows.append(
                {
                    "split": split,
                    "feature": feat,
                    "n_rows": n_all,
                    "non_null": 0,
                    "null_ratio": 1.0,
                    "high_null": True,
                    "distinct": 0,
                }
            )
            continue
        s = work[feat]
        non_null = s.drop_nulls()
        nn = non_null.len()
        null_ratio = 1.0 - (nn / n_all) if n_all else 1.0
        row = {
            "split": split,
            "feature": feat,
            "n_rows": n_all,
            "non_null": nn,
            "null_ratio": null_ratio,
            "high_null": null_ratio >= HIGH_NULL_RATIO,
            "distinct": int(non_null.n_unique()) if nn else 0,
        }
        if nn:
            q = non_null.quantile
            row.update(
                {
                    "mean": float(non_null.mean()),
                    "std": float(non_null.std()) if nn > 1 else None,
                    "min": float(non_null.min()),
                    "p1": float(q(0.01)),
                    "p5": float(q(0.05)),
                    "p25": float(q(0.25)),
                    "median": float(non_null.median()),
                    "p75": float(q(0.75)),
                    "p95": float(q(0.95)),
                    "p99": float(q(0.99)),
                    "max": float(non_null.max()),
                }
            )
        rows.append(row)
    return pl.DataFrame(rows)


def excess_target_name(target: str) -> str:
    if target.startswith("forward_return_"):
        return target.replace("forward_return_", "forward_excess_spy_", 1)
    return target


def _robust_from_series(
    s: pl.Series,
    *,
    lo: float | None = None,
    hi: float | None = None,
    excess: pl.Series | None = None,
) -> dict:
    n = s.len()
    mean = float(s.mean()) if n else None
    std = float(s.std()) if n > 1 else None
    if lo is not None and hi is not None and n:
        clipped = s.clip(lower_bound=lo, upper_bound=hi)
        kept = s.filter((s >= lo) & (s <= hi))
        winsor_mean = float(clipped.mean()) if clipped.len() else None
        trimmed_mean = float(kept.mean()) if kept.len() else None
    else:
        winsor_mean = None
        trimmed_mean = None
    pos = float((s > 0).mean()) if n else None
    excess_pos = float((excess > 0).mean()) if excess is not None and excess.len() else None
    return {
        "sample_count": n,
        "mean": mean,
        "median": float(s.median()) if n else None,
        "trimmed_mean_1pct": trimmed_mean,
        "winsorized_mean_1pct": winsor_mean,
        "p25": float(s.quantile(0.25)) if n else None,
        "p75": float(s.quantile(0.75)) if n else None,
        "positive_ratio": pos,
        "excess_positive_ratio": excess_pos,
        "std": std,
        "tstat": tstat(mean, std, n),
    }


def _quantile_row(
    feature: str,
    split: str,
    quintile: int,
    subset: pl.DataFrame,
    regime_kind: str | None = None,
    regime: str | None = None,
    cutoffs: dict[str, tuple[float, float]] | None = None,
) -> dict:
    n_dates = subset["date"].n_unique() if "date" in subset.columns else 0
    n_obs = subset.height
    row: dict = {
        "feature": feature,
        "split": split,
        "quintile": quintile,
        "n": n_obs,
        "n_dates": n_dates,
        "sample_status": sample_status(n_obs, n_dates),
        "regime_kind": regime_kind,
        "regime": regime,
    }
    for col in ALL_TARGETS:
        if col not in subset.columns:
            continue
        s = subset[col].drop_nulls()
        n = s.len()
        mean = float(s.mean()) if n else None
        std = float(s.std()) if n > 1 else None
        row[f"{col}_n"] = n
        row[f"{col}_mean"] = mean
        row[f"{col}_median"] = float(s.median()) if n else None
        row[f"{col}_std"] = std
        row[f"{col}_positive_ratio"] = float((s > 0).mean()) if n else None
        row[f"{col}_tstat"] = tstat(mean, std, n)
        row[f"{col}_se"] = (std / math.sqrt(n)) if std is not None and n >= 2 else None
        excess_name = excess_target_name(col)
        excess = subset[excess_name].drop_nulls() if excess_name in subset.columns else None
        lo_hi = (cutoffs or {}).get(col)
        stats = _robust_from_series(
            s, lo=lo_hi[0] if lo_hi else None, hi=lo_hi[1] if lo_hi else None, excess=excess
        )
        row[f"{col}_trimmed_mean_1pct"] = stats["trimmed_mean_1pct"]
        row[f"{col}_winsorized_mean_1pct"] = stats["winsorized_mean_1pct"]
        row[f"{col}_p25"] = stats["p25"]
        row[f"{col}_p75"] = stats["p75"]
        row[f"{col}_excess_positive_ratio"] = stats["excess_positive_ratio"]
        row[f"{col}_sample_count"] = stats["sample_count"]
    return row


def quantile_table(
    df: pl.DataFrame,
    features: list[str],
    split: str,
    *,
    regime_kind: str | None = None,
    regime_col: str | None = None,
    cutoffs: dict[str, tuple[float, float]] | None = None,
) -> pl.DataFrame:
    work = filter_split(df, split)
    rows: list[dict] = []
    for feature in features:
        qdf = assign_cs_quintiles(work, feature)
        if qdf.height == 0:
            continue
        groups = [regime_col] if regime_col and regime_col in qdf.columns else []
        if groups:
            keys = qdf.select(["quintile", regime_col]).unique().sort(["quintile", regime_col])
            for rec in keys.iter_rows(named=True):
                if rec[regime_col] is None:
                    continue
                subset = qdf.filter((pl.col("quintile") == rec["quintile"]) & (pl.col(regime_col) == rec[regime_col]))
                rows.append(
                    _quantile_row(
                        feature,
                        split,
                        int(rec["quintile"]),
                        subset,
                        regime_kind,
                        str(rec[regime_col]),
                        cutoffs=cutoffs,
                    )
                )
        else:
            for q in range(1, N_QUANTILES + 1):
                subset = qdf.filter(pl.col("quintile") == q)
                rows.append(_quantile_row(feature, split, q, subset, cutoffs=cutoffs))
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def spread_table(quantile_df: pl.DataFrame) -> pl.DataFrame:
    if quantile_df.height == 0:
        return pl.DataFrame()
    rows: list[dict] = []
    group_cols = ["feature", "split"]
    if "regime" in quantile_df.columns:
        group_cols.extend(["regime_kind", "regime"])
    keys = quantile_df.select(group_cols).unique()
    for rec in keys.iter_rows(named=True):
        filt = quantile_df
        for col, val in rec.items():
            if val is None:
                filt = filt.filter(pl.col(col).is_null())
            else:
                filt = filt.filter(pl.col(col) == val)
        q1 = filt.filter(pl.col("quintile") == 1)
        q5 = filt.filter(pl.col("quintile") == 5)
        if q1.height == 0 or q5.height == 0:
            continue
        r1 = q1.row(0, named=True)
        r5 = q5.row(0, named=True)
        out = dict(rec)
        out["q1_n"] = r1.get("n")
        out["q5_n"] = r5.get("n")
        out["sample_status"] = (
            INSUFFICIENT_SAMPLE
            if r1.get("sample_status") == INSUFFICIENT_SAMPLE or r5.get("sample_status") == INSUFFICIENT_SAMPLE
            else "ok"
        )
        for col in ALL_TARGETS:
            m1 = r1.get(f"{col}_mean")
            m5 = r5.get(f"{col}_mean")
            out[f"{col}_q5_minus_q1"] = (m5 - m1) if m1 is not None and m5 is not None else None
            med1 = r1.get(f"{col}_median")
            med5 = r5.get(f"{col}_median")
            out[f"{col}_q5_minus_q1_median"] = (
                (med5 - med1) if med1 is not None and med5 is not None else None
            )
            w1 = r1.get(f"{col}_winsorized_mean_1pct")
            w5 = r5.get(f"{col}_winsorized_mean_1pct")
            out[f"{col}_q5_minus_q1_winsor_mean"] = (
                (w5 - w1) if w1 is not None and w5 is not None else None
            )
        means = [r.get(f"forward_excess_spy_20d_mean") for r in filt.sort("quintile").iter_rows(named=True)]
        qs = [r["quintile"] for r in filt.sort("quintile").iter_rows(named=True)]
        if all(m is not None for m in means) and len(means) == 5:
            out["monotonicity_excess_20d"] = pearson([float(q) for q in qs], [float(m) for m in means])
        rows.append(out)
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def overlapping_spread_table(df: pl.DataFrame, features: list[str], split: str) -> pl.DataFrame:
    """Date-clustered Q5-Q1 portfolio means (equal-weight within date)."""
    work = filter_split(df, split)
    rows: list[dict] = []
    for feature in features:
        qdf = assign_cs_quintiles(work, feature)
        if qdf.height == 0:
            continue
        for col, horizon in zip(
            FORWARD_RETURN_TARGETS + FORWARD_EXCESS_TARGETS,
            HORIZONS + HORIZONS,
            strict=True,
        ):
            if col not in qdf.columns:
                continue
            daily = (
                qdf.filter(pl.col("quintile").is_in([1, 5]) & pl.col(col).is_not_null())
                .group_by(["date", "quintile"])
                .agg(pl.col(col).mean().alias("port"))
            )
            if daily.height == 0:
                continue
            q1 = daily.filter(pl.col("quintile") == 1).select(["date", pl.col("port").alias("q1")])
            q5 = daily.filter(pl.col("quintile") == 5).select(["date", pl.col("port").alias("q5")])
            wide = q1.join(q5, on="date", how="inner").sort("date")
            if wide.height == 0:
                continue
            spread = (wide["q5"] - wide["q1"]).to_list()
            spread_f = [float(x) for x in spread]
            n = len(spread_f)
            mu = _finite_mean(spread_f)
            std = _std(spread_f)
            median_spread = float(pl.Series(spread_f).median()) if n else None
            pos_ratio = (sum(1 for x in spread_f if x > 0) / n) if n else None
            rows.append(
                {
                    "feature": feature,
                    "split": split,
                    "target": col,
                    "n_dates": n,
                    "mean_spread": mu,
                    "median_spread": median_spread,
                    "positive_spread_ratio": pos_ratio,
                    "std_spread": std,
                    "tstat": tstat(mu, std, n),
                    "newey_west_tstat": newey_west_tstat(spread_f, int(horizon)),
                    "sample_status": sample_status(n, n),
                    "method": "daily_cs_portfolio",
                }
            )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def overlapping_spread_extreme_dates(
    df: pl.DataFrame, features: list[str], split: str, *, top_n: int = 5
) -> pl.DataFrame:
    """Best/worst daily Q5-Q1 spread dates. Checks whether one day dominates t-stats."""
    work = filter_split(df, split)
    rows: list[dict] = []
    for feature in features:
        qdf = assign_cs_quintiles(work, feature)
        if qdf.height == 0:
            continue
        for col, horizon in zip(
            FORWARD_RETURN_TARGETS + FORWARD_EXCESS_TARGETS,
            HORIZONS + HORIZONS,
            strict=True,
        ):
            if col not in qdf.columns:
                continue
            daily = (
                qdf.filter(pl.col("quintile").is_in([1, 5]) & pl.col(col).is_not_null())
                .group_by(["date", "quintile"])
                .agg(pl.col(col).mean().alias("port"))
            )
            if daily.height == 0:
                continue
            q1 = daily.filter(pl.col("quintile") == 1).select(["date", pl.col("port").alias("q1")])
            q5 = daily.filter(pl.col("quintile") == 5).select(["date", pl.col("port").alias("q5")])
            wide = q1.join(q5, on="date", how="inner").sort("date")
            if wide.height == 0:
                continue
            pairs = [
                (d, float(s))
                for d, s in zip(wide["date"].to_list(), (wide["q5"] - wide["q1"]).to_list(), strict=True)
                if s is not None
            ]
            worst = sorted(pairs, key=lambda x: x[1])[:top_n]
            best = sorted(pairs, key=lambda x: x[1], reverse=True)[:top_n]
            for kind, ranked in (("worst", worst), ("best", best)):
                for rank, (d, spr) in enumerate(ranked, start=1):
                    rows.append(
                        {
                            "feature": feature,
                            "split": split,
                            "target": col,
                            "horizon": int(horizon),
                            "kind": kind,
                            "rank": rank,
                            "date": d,
                            "daily_spread": spr,
                        }
                    )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def ic_table(df: pl.DataFrame, features: list[str], split: str) -> pl.DataFrame:
    work = filter_split(df, split)
    rows: list[dict] = []
    for feature in features:
        if feature not in work.columns:
            continue
        for target in ALL_TARGETS:
            if target not in work.columns:
                continue
            pair = work.filter(pl.col(feature).is_not_null() & pl.col(target).is_not_null())
            if pair.height == 0:
                continue
            daily = (
                pair.with_columns(
                    pl.col(feature).rank(method="average").over("date").alias("_fx"),
                    pl.col(target).rank(method="average").over("date").alias("_fy"),
                )
                .group_by("date")
                .agg(pl.corr(pl.col("_fx"), pl.col("_fy")).alias("ic"), pl.len().alias("n"))
                .filter(pl.col("ic").is_not_null())
                .sort("date")
            )
            ics = [float(x) for x in daily["ic"].to_list()]
            n = len(ics)
            mu = _finite_mean(ics)
            std = _std(ics)
            ir = (mu / std) if mu is not None and std not in (None, 0) else None
            rows.append(
                {
                    "feature": feature,
                    "split": split,
                    "target": target,
                    "n_dates": n,
                    "mean_ic": mu,
                    "median_ic": float(daily["ic"].median()) if n else None,
                    "std_ic": std,
                    "ic_ir": ir,
                    "positive_ic_ratio": (sum(1 for x in ics if x > 0) / n) if n else None,
                    "sample_status": sample_status(n, n),
                }
            )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def sensitivity_table(df: pl.DataFrame, split: str) -> pl.DataFrame:
    """Price-level audit. Not an alpha feature."""
    work = attach_price_bucket(filter_split(df, split))
    if work.height == 0 or "price_bucket" not in work.columns:
        return pl.DataFrame()
    rows: list[dict] = []
    for bucket in [b[0] for b in PRICE_BUCKETS]:
        subset = work.filter(pl.col("price_bucket") == bucket)
        row: dict = {
            "split": split,
            "price_bucket": bucket,
            "n": subset.height,
            "n_dates": subset["date"].n_unique() if subset.height else 0,
            "n_securities": subset["security_id"].n_unique() if subset.height else 0,
        }
        for col in ("forward_return_20d", "forward_excess_spy_20d"):
            if col not in subset.columns:
                continue
            s = subset[col].drop_nulls()
            n = s.len()
            row[f"{col}_mean"] = float(s.mean()) if n else None
            row[f"{col}_std"] = float(s.std()) if n > 1 else None
            row[f"{col}_positive_ratio"] = float((s > 0).mean()) if n else None
            row[f"{col}_abs_gt_0_5_ratio"] = float((s.abs() > 0.5).mean()) if n else None
        rows.append(row)
    return pl.DataFrame(rows)


def ic_by_date_table(df: pl.DataFrame, feature: str, target: str, split: str) -> pl.DataFrame:
    work = filter_split(df, split)
    if feature not in work.columns or target not in work.columns:
        return pl.DataFrame()
    pair = work.filter(pl.col(feature).is_not_null() & pl.col(target).is_not_null())
    return (
        pair.with_columns(
            pl.col(feature).rank(method="average").over("date").alias("_fx"),
            pl.col(target).rank(method="average").over("date").alias("_fy"),
        )
        .group_by("date")
        .agg(pl.corr(pl.col("_fx"), pl.col("_fy")).alias("ic"), pl.len().alias("n"))
        .sort("date")
        .with_columns(pl.lit(feature).alias("feature"), pl.lit(target).alias("target"), pl.lit(split).alias("split"))
    )
