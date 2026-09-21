"""Audit EXTREME_LABEL warnings without changing the |x|>2 threshold."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date

import duckdb
import polars as pl

from app.config.lake_datasets import get_lake_dataset
from app.config.settings import Settings
from app.db.schema import create_lake_views
from app.labels.schema import LABEL_VALUE_COLUMNS
from app.services.instrument_classification_service import refresh_instrument_classifications
from app.services.universe_membership_service import (
    UNIVERSE_TYPE_RESEARCH_COMMON_EQUITY,
    membership_security_ids_by_type,
)
from app.validation.label_rules import check_extreme_labels
from app.validation.numeric_tolerance import approximately_ge

CAUSE_EXTREME_MARKET = "A_EXTREME_MARKET"
CAUSE_NON_COMMON_INSTRUMENT = "B_NON_COMMON_INSTRUMENT"
CAUSE_ADJUSTMENT_ARTIFACT = "C_ADJUSTMENT_ARTIFACT"
CAUSE_PROVIDER_BAD_DATA = "D_PROVIDER_BAD_DATA"
CAUSE_UNEXPLAINED = "E_UNEXPLAINED"

NEAR_ACTION_DAYS = 5
ADJ_JUMP = 2.0
CLOSE_STABLE = 0.5


@dataclass
class ExtremeLabelAuditReport:
    total: int = 0
    by_ticker: list[tuple[str, int]] = field(default_factory=list)
    by_instrument_class: dict[str, int] = field(default_factory=dict)
    by_horizon: dict[str, int] = field(default_factory=dict)
    by_cause: dict[str, int] = field(default_factory=dict)
    largest_positive: list[dict] = field(default_factory=list)
    largest_negative: list[dict] = field(default_factory=list)
    by_date: list[tuple[str, int]] = field(default_factory=list)
    near_corporate_action: int = 0
    adj_close_jump: int = 0
    research_unexplained: int = 0
    research_unexplained_rows: list[dict] = field(default_factory=list)


def _parse_horizon(details: str) -> str:
    # "forward_return_10d=3.2 exceeds ±2.0"
    token = (details or "").split("=", 1)[0].strip()
    return token or "unknown"


def audit_extreme_labels(settings: Settings, con: duckdb.DuckDBPyConnection) -> ExtremeLabelAuditReport:
    refresh_instrument_classifications(con)
    create_lake_views(con, settings)
    labels_ds = get_lake_dataset(settings.lake_dir, "labels_forward_returns")
    if not labels_ds.has_any_files():
        return ExtremeLabelAuditReport()

    labels = con.execute(
        """
        SELECT l.*, s.primary_ticker AS ticker,
               COALESCE(c.instrument_class, 'AMBIGUOUS') AS instrument_class
        FROM labels_forward_returns l
        LEFT JOIN securities s ON s.security_id = l.security_id
        LEFT JOIN instrument_classifications c ON c.security_id = l.security_id
        """
    ).pl()
    issues = check_extreme_labels(labels)
    if not issues:
        return ExtremeLabelAuditReport()

    prices = None
    if get_lake_dataset(settings.lake_dir, "prices_daily").has_any_files():
        prices = con.execute(
            "SELECT security_id, date, open, high, low, close, adj_close, stock_split FROM prices_daily"
        ).pl()
        prices = prices.sort(["security_id", "date"]).with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1).over("security_id") - 1).alias("adj_ret"),
            (pl.col("close") / pl.col("close").shift(1).over("security_id") - 1).alias("raw_ret"),
        )

    actions: pl.DataFrame | None = None
    if get_lake_dataset(settings.lake_dir, "corporate_actions").has_any_files():
        actions = con.execute(
            "SELECT security_id, effective_date, action_type FROM corporate_actions"
        ).pl()

    research_ids = set(membership_security_ids_by_type(con, UNIVERSE_TYPE_RESEARCH_COMMON_EQUITY))
    class_by_id = {
        row["security_id"]: row["instrument_class"]
        for row in labels.select(["security_id", "instrument_class"]).unique().iter_rows(named=True)
    }
    ticker_by_id = {
        row["security_id"]: row["ticker"]
        for row in labels.select(["security_id", "ticker"]).unique().iter_rows(named=True)
    }

    price_lookup: dict[tuple[str, date], dict] = {}
    if prices is not None and prices.height:
        for row in prices.iter_rows(named=True):
            price_lookup[(row["security_id"], row["date"])] = row

    action_dates: dict[str, list[date]] = {}
    if actions is not None and actions.height:
        for row in actions.iter_rows(named=True):
            action_dates.setdefault(row["security_id"], []).append(row["effective_date"])

    report = ExtremeLabelAuditReport(total=len(issues))
    ticker_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    horizon_counts: Counter[str] = Counter()
    cause_counts: Counter[str] = Counter()
    date_counts: Counter[str] = Counter()
    valued: list[dict] = []

    for issue in issues:
        sid = issue["security_id"]
        d = issue["date"]
        ticker = ticker_by_id.get(sid) or sid
        iclass = class_by_id.get(sid, "AMBIGUOUS")
        horizon = _parse_horizon(issue.get("details") or "")
        ticker_counts[str(ticker)] += 1
        class_counts[str(iclass)] += 1
        horizon_counts[horizon] += 1
        date_counts[str(d)] += 1

        value = None
        if "=" in (issue.get("details") or ""):
            try:
                value = float((issue["details"].split("=", 1)[1].split()[0]))
            except (TypeError, ValueError, IndexError):
                value = None

        near_action = False
        if sid in action_dates and d is not None:
            for ad in action_dates[sid]:
                if ad is None:
                    continue
                if abs((ad - d).days) <= NEAR_ACTION_DAYS:
                    near_action = True
                    break
        if near_action:
            report.near_corporate_action += 1

        prow = price_lookup.get((sid, d))
        adj_jump = False
        if prow is not None:
            adj_ret = prow.get("adj_ret")
            raw_ret = prow.get("raw_ret")
            if adj_ret is not None and abs(adj_ret) > ADJ_JUMP:
                report.adj_close_jump += 1
                adj_jump = True
                if raw_ret is not None and abs(raw_ret) < CLOSE_STABLE:
                    pass  # adjustment artifact signal
            split = prow.get("stock_split") or 0.0
            if split and split not in (0.0, 1.0):
                near_action = True

        if iclass not in {"COMMON_EQUITY", "ETF"}:
            cause = CAUSE_NON_COMMON_INSTRUMENT
        elif adj_jump and prow is not None and prow.get("raw_ret") is not None and abs(prow["raw_ret"]) < CLOSE_STABLE:
            cause = CAUSE_PROVIDER_BAD_DATA if not near_action else CAUSE_ADJUSTMENT_ARTIFACT
        elif prow is not None:
            o, h, l, c = prow.get("open"), prow.get("high"), prow.get("low"), prow.get("close")
            ohlc_ok = (
                o is not None
                and h is not None
                and l is not None
                and c is not None
                and approximately_ge(h, l)
                and approximately_ge(h, o)
                and approximately_ge(h, c)
                and approximately_ge(o, l)
                and approximately_ge(c, l)
            )
            if not ohlc_ok:
                cause = CAUSE_PROVIDER_BAD_DATA
            else:
                cause = CAUSE_EXTREME_MARKET
        else:
            cause = CAUSE_UNEXPLAINED

        cause_counts[cause] += 1
        rec = {
            "security_id": sid,
            "ticker": ticker,
            "date": str(d),
            "instrument_class": iclass,
            "horizon": horizon,
            "value": value,
            "cause": cause,
            "near_corporate_action": near_action,
        }
        valued.append(rec)
        if (
            sid in research_ids
            and iclass == "COMMON_EQUITY"
            and cause in {CAUSE_ADJUSTMENT_ARTIFACT, CAUSE_PROVIDER_BAD_DATA, CAUSE_UNEXPLAINED}
        ):
            report.research_unexplained += 1
            report.research_unexplained_rows.append(rec)

    report.by_ticker = ticker_counts.most_common()
    report.by_instrument_class = dict(class_counts)
    report.by_horizon = dict(horizon_counts)
    report.by_cause = dict(cause_counts)
    report.by_date = date_counts.most_common(30)
    with_val = [r for r in valued if r["value"] is not None]
    report.largest_positive = sorted(with_val, key=lambda r: r["value"], reverse=True)[:30]
    negatives = [r for r in with_val if r["value"] < 0]
    report.largest_negative = sorted(negatives, key=lambda r: r["value"])[:30]
    return report
