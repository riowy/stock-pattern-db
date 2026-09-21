"""Price quality audit: split research-common vs non-common OHLC issues."""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb

from app.config.instrument_patterns import CLASS_COMMON_EQUITY, CLASS_ETF
from app.config.lake_datasets import get_lake_dataset
from app.config.settings import Settings
from app.db.schema import create_lake_views
from app.services.instrument_classification_service import refresh_instrument_classifications
from app.services.universe_membership_service import (
    UNIVERSE_TYPE_RESEARCH_COMMON_EQUITY,
    membership_security_ids_by_type,
)
from app.validation.rules import check_ohlc_consistency


@dataclass
class PriceAuditReport:
    research_critical: int = 0
    research_warning: int = 0
    non_common_critical: int = 0
    non_common_warning: int = 0
    rounding_only: int = 0
    true_ohlc_violations: int = 0
    research_true_critical_tickers: list[str] = field(default_factory=list)
    all_provider_critical: int = 0
    rows_checked: int = 0


def audit_prices(settings: Settings, con: duckdb.DuckDBPyConnection) -> PriceAuditReport:
    refresh_instrument_classifications(con)
    create_lake_views(con, settings)
    ds = get_lake_dataset(settings.lake_dir, "prices_daily")
    if not ds.has_any_files():
        return PriceAuditReport()

    df = con.execute(
        """
        SELECT p.security_id, p.date, p.open, p.high, p.low, p.close, p.adj_close, p.volume,
               p.dividend, p.stock_split, s.primary_ticker AS ticker,
               COALESCE(c.instrument_class, 'AMBIGUOUS') AS instrument_class
        FROM prices_daily p
        LEFT JOIN securities s ON s.security_id = p.security_id
        LEFT JOIN instrument_classifications c ON c.security_id = p.security_id
        """
    ).pl()
    issues = check_ohlc_consistency(df)
    research_ids = set(membership_security_ids_by_type(con, UNIVERSE_TYPE_RESEARCH_COMMON_EQUITY))
    class_by_id = {
        row["security_id"]: row["instrument_class"]
        for row in df.select(["security_id", "instrument_class"]).unique().iter_rows(named=True)
    }
    ticker_by_id = {
        row["security_id"]: row["ticker"]
        for row in df.select(["security_id", "ticker"]).unique().iter_rows(named=True)
    }

    report = PriceAuditReport(rows_checked=df.height)
    research_crit_tickers: set[str] = set()
    for issue in issues:
        sid = issue.get("security_id")
        itype = issue["issue_type"]
        sev = issue["severity"]
        iclass = class_by_id.get(sid, "AMBIGUOUS")
        in_research = sid in research_ids and iclass == CLASS_COMMON_EQUITY
        is_commonish = iclass in {CLASS_COMMON_EQUITY, CLASS_ETF}

        if itype == "OHLC_ROUNDING_TOLERANCE":
            report.rounding_only += 1
            continue
        if itype == "OHLC_INCONSISTENT" and sev == "critical":
            report.true_ohlc_violations += 1
            report.all_provider_critical += 1
            if in_research:
                report.research_critical += 1
                ticker = ticker_by_id.get(sid) or sid
                research_crit_tickers.add(str(ticker))
            else:
                report.non_common_critical += 1
            continue
        if sev == "warning":
            if in_research or is_commonish:
                report.research_warning += 1
            else:
                report.non_common_warning += 1

    report.research_true_critical_tickers = sorted(research_crit_tickers)
    return report
