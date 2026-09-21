"""Deterministic research-scale and research-common-equity universes.

``research-scale-N`` is a PROVIDER_SCALE_TEST hash sample (may include
warrants/preferred). ``research-common-equity-N`` is the v1 research
target: US-listed names classified as COMMON_EQUITY. Neither uses
price momentum, forward return, or market-cap performance.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import duckdb

from app.config.etf_seed import SEED_ETF_TICKERS
from app.config.instrument_patterns import CLASS_COMMON_EQUITY
from app.ingestion.price_backfill import resolve_symbols
from app.services.dataset_metadata_service import set_metadata
from app.services.instrument_classification_service import refresh_instrument_classifications
from app.services.market_calendar import MarketCalendarService
from app.services.tracked_universe_service import add_tracked, ensure_tracked, get_tracked_price_security_ids
from app.services.universe_membership_service import (
    SELECTION_VERSION_V1,
    UNIVERSE_TYPE_BENCHMARK,
    UNIVERSE_TYPE_PROVIDER_SCALE_TEST,
    UNIVERSE_TYPE_RESEARCH_COMMON_EQUITY,
    record_universe_metadata,
    replace_memberships,
    upsert_memberships,
)

MAJOR_US_EXCHANGES = frozenset({"NYSE", "Nasdaq", "CBOE"})
VALID_TICKER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,7}$")
BENCHMARK_TICKERS = ("SPY", "QQQ")


def _rank_key(security_id: str) -> str:
    return hashlib.sha256(security_id.encode("utf-8")).hexdigest()


def _valid_ticker(ticker: str | None) -> bool:
    if not ticker:
        return False
    return bool(VALID_TICKER.match(ticker.strip().upper()))


@dataclass
class UniversePlan:
    name: str
    size: int
    tickers: list[str]
    security_ids: list[str]
    generated_at: str
    criteria: str
    pinned: list[str] = field(default_factory=list)


def eligible_research_securities(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    """Return [(ticker, security_id), ...] that pass the scale-test filters."""
    seed = {t.upper() for t in SEED_ETF_TICKERS}
    rows = con.execute(
        """
        SELECT primary_ticker, security_id, exchange
        FROM securities
        WHERE is_active = TRUE AND primary_ticker IS NOT NULL
        """
    ).fetchall()
    out: list[tuple[str, str]] = []
    for ticker, sid, exchange in rows:
        t = (ticker or "").upper()
        if not _valid_ticker(t):
            continue
        if exchange in MAJOR_US_EXCHANGES or t in seed:
            out.append((t, sid))
    out.sort(key=lambda pair: _rank_key(pair[1]))
    return out


def build_research_scale_universe(con: duckdb.DuckDBPyConnection, size: int) -> UniversePlan:
    if size < 1:
        raise ValueError("research scale size must be >= 1")
    eligible = eligible_research_securities(con)
    by_ticker = {t: sid for t, sid in eligible}
    # Also resolve pinned tickers that might have been filtered (e.g. QQQ exchange=None is in ETF seed)
    pinned: list[tuple[str, str]] = []
    for ticker in BENCHMARK_TICKERS:
        if ticker in by_ticker:
            pinned.append((ticker, by_ticker[ticker]))
        else:
            resolved = resolve_symbols(con, [ticker], default_scope="tracked")
            pinned.extend(resolved)

    tracked_ids = set(get_tracked_price_security_ids(con))
    if tracked_ids:
        placeholders = ", ".join("?" for _ in tracked_ids)
        extra = con.execute(
            f"SELECT primary_ticker, security_id FROM securities "
            f"WHERE security_id IN ({placeholders}) AND primary_ticker IS NOT NULL",
            list(tracked_ids),
        ).fetchall()
        for t, sid in extra:
            pair = (t.upper(), sid)
            if pair not in pinned:
                pinned.append(pair)

    selected: list[tuple[str, str]] = []
    seen: set[str] = set()
    for t, sid in pinned:
        if sid not in seen:
            selected.append((t, sid))
            seen.add(sid)
    for t, sid in eligible:
        if sid in seen:
            continue
        selected.append((t, sid))
        seen.add(sid)
        if len(selected) >= size:
            break

    selected = selected[:size]
    generated = datetime.now(UTC).isoformat()
    criteria = (
        "scale-test only (NOT an investment universe); active; exchange in "
        "NYSE/Nasdaq/CBOE or ETF seed list; valid ticker; provider symbol "
        "mappable; deterministic sha256(security_id) order; pinned existing "
        "tracked + SPY/QQQ; no performance-based selection"
    )
    return UniversePlan(
        name=f"research-scale-{size}",
        size=len(selected),
        tickers=[t for t, _ in selected],
        security_ids=[sid for _, sid in selected],
        generated_at=generated,
        criteria=criteria,
        pinned=[t for t, _ in pinned],
    )


def apply_research_scale_universe(
    con: duckdb.DuckDBPyConnection, plan: UniversePlan, dry_run: bool = False
) -> UniversePlan:
    if dry_run:
        return plan
    add_tracked(
        con,
        plan.security_ids,
        reason=plan.name,
        price_tracking=True,
        filings_tracking=False,
        feature_tracking=True,
        notes=f"generated_at={plan.generated_at}",
    )
    upsert_memberships(
        con,
        plan.name,
        UNIVERSE_TYPE_PROVIDER_SCALE_TEST,
        plan.security_ids,
        selection_rule=plan.criteria,
    )
    record_universe_metadata(
        con,
        plan.name,
        UNIVERSE_TYPE_PROVIDER_SCALE_TEST,
        plan.size,
        plan.criteria,
        created_at=plan.generated_at,
    )
    set_metadata(con, "research_collection_universe", "name", plan.name)
    set_metadata(con, "research_collection_universe", "size", str(plan.size))
    set_metadata(con, "research_collection_universe", "generated_at", plan.generated_at)
    set_metadata(con, "research_collection_universe", "criteria", plan.criteria)
    set_metadata(con, "research_collection_universe", "is_investment_universe", "false")
    set_metadata(con, "research_collection_universe", "historical_universe_complete", "false")
    set_metadata(con, "research_collection_universe", "survivorship_safe", "false")
    return plan


def estimate_price_backfill(
    n_symbols: int,
    start: date,
    end: date | None = None,
    calendar_name: str = "XNYS",
) -> dict:
    calendar = MarketCalendarService(calendar_name)
    end_d = end or date.today()
    sessions = calendar.trading_days_between(start, end_d)
    n_sessions = len(sessions)
    est_rows = n_symbols * n_sessions
    # Observed ~80-150 bytes/row ZSTD for daily OHLCV; use a conservative 150.
    est_bytes = est_rows * 150
    return {
        "symbols": n_symbols,
        "estimated_requests": n_symbols,
        "estimated_sessions": n_sessions,
        "estimated_rows": est_rows,
        "estimated_disk_bytes": est_bytes,
        "start": start.isoformat(),
        "end": end_d.isoformat(),
    }


def all_active_tickers(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    rows = con.execute(
        "SELECT primary_ticker, security_id FROM securities "
        "WHERE is_active = TRUE AND primary_ticker IS NOT NULL ORDER BY primary_ticker"
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


COMMON_EQUITY_SELECTION_RULE = (
    "v1 research common equity: active; exchange in NYSE/Nasdaq/CBOE; "
    "instrument_class=COMMON_EQUITY from SEC registrant/title signals then "
    "ticker-suffix heuristic; excludes CLOSED_END_FUND/FUND/DEBT/warrant/"
    "preferred/unit/right/ETF; valid ticker; provider-mappable; deterministic "
    "sha256(security_id) order; no performance/momentum/market-cap selection; "
    "instrument_class_complete=false"
)


def eligible_common_equity_securities(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    """Return [(ticker, security_id), ...] classified as COMMON_EQUITY on major US exchanges."""
    refresh_instrument_classifications(con)
    seed = {t.upper() for t in SEED_ETF_TICKERS}
    rows = con.execute(
        """
        SELECT s.primary_ticker, s.security_id, s.exchange, c.instrument_class
        FROM securities s
        JOIN instrument_classifications c ON c.security_id = s.security_id
        WHERE s.is_active = TRUE AND s.primary_ticker IS NOT NULL
        """
    ).fetchall()
    out: list[tuple[str, str]] = []
    for ticker, sid, exchange, instrument_class in rows:
        t = (ticker or "").upper()
        if t in seed:
            continue
        if not _valid_ticker(t):
            continue
        if exchange not in MAJOR_US_EXCHANGES:
            continue
        if instrument_class != CLASS_COMMON_EQUITY:
            continue
        out.append((t, sid))
    out.sort(key=lambda pair: _rank_key(pair[1]))
    return out


def build_research_common_equity_universe(con: duckdb.DuckDBPyConnection, size: int) -> UniversePlan:
    if size < 1:
        raise ValueError("research common-equity size must be >= 1")
    eligible = eligible_common_equity_securities(con)
    selected = eligible[:size]
    generated = datetime.now(UTC).isoformat()
    return UniversePlan(
        name=f"research-common-equity-{size}",
        size=len(selected),
        tickers=[t for t, _ in selected],
        security_ids=[sid for _, sid in selected],
        generated_at=generated,
        criteria=COMMON_EQUITY_SELECTION_RULE,
        pinned=[],
    )


def apply_research_common_equity_universe(
    con: duckdb.DuckDBPyConnection, plan: UniversePlan, dry_run: bool = False, track: bool = True
) -> UniversePlan:
    if dry_run:
        return plan
    if track:
        ensure_tracked(
            con,
            plan.security_ids,
            reason=plan.name,
            price_tracking=True,
            filings_tracking=False,
            feature_tracking=True,
            notes=f"generated_at={plan.generated_at}",
        )
    replace_memberships(
        con,
        plan.name,
        UNIVERSE_TYPE_RESEARCH_COMMON_EQUITY,
        plan.security_ids,
        selection_rule=plan.criteria,
        selection_version=SELECTION_VERSION_V1,
    )
    record_universe_metadata(
        con,
        plan.name,
        UNIVERSE_TYPE_RESEARCH_COMMON_EQUITY,
        plan.size,
        plan.criteria,
        created_at=plan.generated_at,
    )
    apply_benchmark_universe(con)
    return plan


def apply_benchmark_universe(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Ensure seed ETFs are tracked as BENCHMARK, not as research common equity."""
    seed = [t.upper() for t in SEED_ETF_TICKERS]
    resolved = resolve_symbols(con, seed, default_scope="tracked")
    if not resolved:
        return []
    sids = [sid for _, sid in resolved]
    tickers = [t for t, _ in resolved]
    rule = "curated ETF seed list; BENCHMARK not research common equity"
    ensure_tracked(
        con,
        sids,
        reason="benchmark-etf-seed",
        price_tracking=True,
        filings_tracking=False,
        feature_tracking=True,
        notes="SEED_ETF_TICKERS",
    )
    upsert_memberships(con, "benchmark-etf-seed", UNIVERSE_TYPE_BENCHMARK, sids, selection_rule=rule)
    record_universe_metadata(con, "benchmark-etf-seed", UNIVERSE_TYPE_BENCHMARK, len(sids), rule)
    return tickers


def estimate_price_backfill_from_observed(
    n_symbols: int,
    start: date,
    end: date | None = None,
    *,
    observed_seconds_per_symbol: float | None = None,
    observed_bytes_per_row: float | None = None,
    calendar_name: str = "XNYS",
) -> dict:
    """Dry-run estimate using observed 100/500 benchmarks when available."""
    base = estimate_price_backfill(n_symbols, start, end, calendar_name=calendar_name)
    bytes_per_row = observed_bytes_per_row if observed_bytes_per_row and observed_bytes_per_row > 0 else 150.0
    base["estimated_disk_bytes"] = int(base["estimated_rows"] * bytes_per_row)
    base["observed_bytes_per_row"] = bytes_per_row
    if observed_seconds_per_symbol and observed_seconds_per_symbol > 0:
        base["estimated_runtime_seconds"] = n_symbols * observed_seconds_per_symbol
        base["observed_seconds_per_symbol"] = observed_seconds_per_symbol
    return base

