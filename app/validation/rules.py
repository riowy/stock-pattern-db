"""Data quality rules for daily price bars.

Every rule is a pure function: ``polars.DataFrame -> list[dict]``. Rules
never mutate or drop data -- they only report findings. It is the caller's
job (``app.validation.runner``) to persist findings into
``data_quality_issues``.

Expected input schema (superset of app.normalization.prices.PRICE_BAR_SCHEMA):
    security_id, date, open, high, low, close, adj_close, volume, dividend, stock_split
"""

from __future__ import annotations

from datetime import date, datetime

import polars as pl

from app.services.market_calendar import MarketCalendarService

Issue = dict


def _issue(security_id: str | None, d: date | None, issue_type: str, severity: str, details: str) -> Issue:
    return {
        "security_id": security_id,
        "date": d,
        "issue_type": issue_type,
        "severity": severity,
        "details": details,
    }


def check_negative_values(df: pl.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    for col in ("open", "high", "low", "close", "adj_close"):
        bad = df.filter(pl.col(col).is_not_null() & (pl.col(col) < 0))
        for row in bad.iter_rows(named=True):
            issues.append(_issue(row["security_id"], row["date"], "NEGATIVE_PRICE", "critical", f"{col}={row[col]}"))

    bad_vol = df.filter(pl.col("volume").is_not_null() & (pl.col("volume") < 0))
    for row in bad_vol.iter_rows(named=True):
        issues.append(_issue(row["security_id"], row["date"], "NEGATIVE_VOLUME", "critical", f"volume={row['volume']}"))
    return issues


def check_ohlc_consistency(df: pl.DataFrame) -> list[Issue]:
    """high must be >= low/open/close; low must be <= open/close."""
    complete = df.filter(
        pl.col("open").is_not_null()
        & pl.col("high").is_not_null()
        & pl.col("low").is_not_null()
        & pl.col("close").is_not_null()
    )
    issues: list[Issue] = []

    checks = [
        (pl.col("high") < pl.col("low"), "high < low"),
        (pl.col("high") < pl.col("open"), "high < open"),
        (pl.col("high") < pl.col("close"), "high < close"),
        (pl.col("low") > pl.col("open"), "low > open"),
        (pl.col("low") > pl.col("close"), "low > close"),
    ]
    for cond, label in checks:
        bad = complete.filter(cond)
        for row in bad.iter_rows(named=True):
            issues.append(
                _issue(
                    row["security_id"],
                    row["date"],
                    "OHLC_INCONSISTENT",
                    "critical",
                    f"{label} (O={row['open']} H={row['high']} L={row['low']} C={row['close']})",
                )
            )
    return issues


def check_duplicate_keys(df: pl.DataFrame) -> list[Issue]:
    counts = df.group_by(["security_id", "date"]).agg(pl.len().alias("n"))
    dups = counts.filter(pl.col("n") > 1)
    return [
        _issue(row["security_id"], row["date"], "DUPLICATE_KEY", "critical", f"{row['n']} rows for same key")
        for row in dups.iter_rows(named=True)
    ]


def check_weekend_dates(df: pl.DataFrame) -> list[Issue]:
    weekend = df.filter(pl.col("date").dt.weekday() >= 6)  # polars: 1=Mon..7=Sun -> weekend is 6,7
    return [
        _issue(row["security_id"], row["date"], "WEEKEND_DATE", "warning", "Trading date falls on a weekend")
        for row in weekend.iter_rows(named=True)
    ]


def check_invalid_dates(df: pl.DataFrame, min_date: date = date(1990, 1, 1)) -> list[Issue]:
    today = date.today()
    bad = df.filter((pl.col("date") > today) | (pl.col("date") < min_date))
    return [
        _issue(row["security_id"], row["date"], "INVALID_DATE", "critical", f"date out of plausible range")
        for row in bad.iter_rows(named=True)
    ]


def check_nan_ratio(df: pl.DataFrame, threshold: float = 0.05) -> list[Issue]:
    issues: list[Issue] = []
    if df.height == 0:
        return issues
    grouped = df.group_by("security_id").agg(
        pl.len().alias("n"),
        pl.col("close").is_null().sum().alias("null_close"),
    )
    for row in grouped.iter_rows(named=True):
        ratio = row["null_close"] / row["n"] if row["n"] else 0
        if ratio > threshold:
            issues.append(
                _issue(
                    row["security_id"],
                    None,
                    "HIGH_NULL_RATIO",
                    "warning",
                    f"{ratio:.1%} of close prices are null ({row['null_close']}/{row['n']})",
                )
            )
    return issues


def check_sudden_price_change(
    df: pl.DataFrame, warn_threshold: float = 0.80, split_hint_threshold: float = 0.45
) -> list[Issue]:
    """Flag single-day close-to-close moves >= ``warn_threshold`` in magnitude.

    A move in the 45%-90% range that coincides with a recorded
    ``stock_split`` on the same date is reported as ``severity=info``
    (expected, adjustment-related); everything else at or above the warn
    threshold is reported as ``severity=warning`` for manual review. We
    never auto-correct or drop these rows.
    """
    issues: list[Issue] = []
    if df.height == 0:
        return issues

    sorted_df = df.sort(["security_id", "date"])
    with_change = sorted_df.with_columns(
        pl.col("close").pct_change().over("security_id").alias("__pct_change")
    )
    flagged = with_change.filter(
        pl.col("__pct_change").is_not_null() & (pl.col("__pct_change").abs() >= split_hint_threshold)
    )
    for row in flagged.iter_rows(named=True):
        pct = row["__pct_change"]
        has_split = bool(row.get("stock_split") and row["stock_split"] != 0)
        if has_split:
            severity = "info"
            issue_type = "SUSPECTED_SPLIT_ADJUSTMENT"
            details = f"close changed {pct:.1%} with a recorded stock_split={row['stock_split']} on this date"
        elif abs(pct) >= warn_threshold:
            severity = "warning"
            issue_type = "SUDDEN_PRICE_CHANGE"
            details = f"close changed {pct:.1%} day-over-day with no recorded corporate action"
        else:
            continue
        issues.append(_issue(row["security_id"], row["date"], issue_type, severity, details))
    return issues


def check_missing_recent_data(
    df: pl.DataFrame,
    expected_security_ids: list[str],
    calendar: MarketCalendarService,
    now: datetime | None = None,
    max_gap_sessions: int = 5,
) -> list[Issue]:
    """Flag *tracked* securities (see caller -- ``expected_security_ids``
    should already be scoped to the tracked price universe, not the full
    security master) whose latest known bar is more than
    ``max_gap_sessions`` US-market **trading sessions** behind the latest
    session that should currently have data available.

    Uses the real NYSE trading calendar (``MarketCalendarService``), not
    calendar weekdays: weekends and US market holidays never count as
    "missing" sessions, and a session that has not closed yet (plus a
    configurable grace period) is not expected to have data yet either.
    """
    issues: list[Issue] = []
    if not expected_security_ids:
        return issues

    latest_expected = calendar.latest_expected_session(now)
    cutoff = calendar.sessions_ago(latest_expected, max_gap_sessions)

    if df.height == 0:
        for sid in expected_security_ids:
            issues.append(_issue(sid, None, "MISSING_RECENT_DATA", "warning", "No price data at all"))
        return issues

    last_dates = df.group_by("security_id").agg(pl.col("date").max().alias("last_date"))
    last_date_map = {row["security_id"]: row["last_date"] for row in last_dates.iter_rows(named=True)}

    for sid in expected_security_ids:
        last_date = last_date_map.get(sid)
        if last_date is None:
            issues.append(_issue(sid, None, "MISSING_RECENT_DATA", "warning", "No price data at all"))
        elif last_date < cutoff:
            issues.append(
                _issue(
                    sid,
                    last_date,
                    "MISSING_RECENT_DATA",
                    "warning",
                    f"Last available date is {last_date}, more than {max_gap_sessions} trading "
                    f"sessions behind the latest expected session ({latest_expected})",
                )
            )
    return issues


ALL_ROW_LEVEL_RULES = [
    check_negative_values,
    check_ohlc_consistency,
    check_duplicate_keys,
    check_weekend_dates,
    check_invalid_dates,
    check_nan_ratio,
    check_sudden_price_change,
]
