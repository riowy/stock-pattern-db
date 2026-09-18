"""Normalize raw security-master rows (SEC + ETF seed list) into internal models.

Identifier strategy (see README "Security Master"):

* ``security_id`` is a permanent internal key, NEVER the ticker.
* When a CIK is available we use it to build a stable id: ``CIK0000320193``.
* Seed-list ETFs that SEC's ``company_tickers_exchange.json`` does not cover
  (most sector SPDRs, QQQ, IWM, SMH, HYG, LQD, ...) fall back to a
  ticker-derived id (``TCKQQQ``) -- this is a documented limitation, not a
  design ideal. As soon as a better identifier (e.g. a licensed security
  master with LEI/FIGI) is available, these can be re-keyed by adding a new
  ``identifier_type`` row without disrupting ``security_id`` for CIK-backed
  securities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from app.models.security import Security, SecurityIdentifier, SecuritySnapshot


def security_id_for_cik(cik: int | str) -> str:
    return f"CIK{int(cik):010d}"


def security_id_for_ticker_fallback(ticker: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9]", "", ticker.upper())
    return f"TCK{cleaned}"


@dataclass
class NormalizedUniverse:
    securities: list[Security] = field(default_factory=list)
    identifiers: list[SecurityIdentifier] = field(default_factory=list)
    snapshots: list[SecuritySnapshot] = field(default_factory=list)


def normalize_sec_universe_rows(
    rows: list[dict[str, Any]],
    retrieved_at: datetime,
    snapshot_date: date,
    source: str = "sec_company_tickers_exchange",
) -> NormalizedUniverse:
    """Build securities/identifiers/snapshots from raw SEC universe rows.

    IMPORTANT (documented limitation): SEC's ``company_tickers_exchange.json``
    can list *multiple tickers under the same CIK* -- most commonly a common
    stock plus several preferred-share classes/ADR duplicates (e.g. ``JPM``,
    ``JPM-PC``, ``JPM-PD``, ...), or genuine multiple share classes
    (``GOOGL``/``GOOG``, ``BRK-A``/``BRK-B``). Each such ticker is a
    distinct tradable instrument with its own price series, so it must get
    its own ``security_id``. When a CIK maps to exactly one ticker (the
    overwhelming majority of cases) we use the pure CIK-based id, which
    stays stable across ticker renames. When a CIK maps to multiple
    tickers, we disambiguate with a ``-<TICKER>`` suffix; a rename of one
    of those specific share classes would then be treated as a new
    security rather than preserving continuity -- a real gap that a
    licensed security master (LEI/FIGI-based) would close later.
    """
    cik_ticker_counts: dict[int, int] = {}
    for row in rows:
        if row.get("ticker") and row.get("cik") is not None:
            cik_int = int(row["cik"])
            cik_ticker_counts[cik_int] = cik_ticker_counts.get(cik_int, 0) + 1

    result = NormalizedUniverse()
    for row in rows:
        cik = row.get("cik")
        ticker = row.get("ticker")
        if not ticker or cik is None:
            continue
        ticker = str(ticker).upper()
        cik_int = int(cik)
        if cik_ticker_counts.get(cik_int, 0) > 1:
            security_id = f"{security_id_for_cik(cik_int)}-{re.sub(r'[^A-Z0-9]', '', ticker)}"
        else:
            security_id = security_id_for_cik(cik_int)
        cik_str = f"{cik_int:010d}"
        company_name = row.get("name")
        exchange = row.get("exchange")

        result.securities.append(
            Security(
                security_id=security_id,
                cik=cik_str,
                company_name=company_name,
                primary_ticker=ticker,
                exchange=exchange,
                asset_type="EQUITY",
                currency="USD",
                is_active=True,
                first_seen_at=retrieved_at,
                last_seen_at=retrieved_at,
                created_at=retrieved_at,
                updated_at=retrieved_at,
            )
        )
        result.identifiers.append(
            SecurityIdentifier(
                security_id=security_id,
                identifier_type="CIK",
                identifier_value=cik_str,
                valid_from=snapshot_date,
                valid_to=None,
                source=source,
            )
        )
        result.identifiers.append(
            SecurityIdentifier(
                security_id=security_id,
                identifier_type="TICKER",
                identifier_value=ticker,
                valid_from=snapshot_date,
                valid_to=None,
                source=source,
            )
        )
        result.snapshots.append(
            SecuritySnapshot(
                snapshot_date=snapshot_date,
                security_id=security_id,
                ticker=ticker,
                company_name=company_name,
                exchange=exchange,
                source=source,
            )
        )
    return result


def build_etf_seed_universe(
    seed_tickers: list[str],
    known_tickers: set[str],
    retrieved_at: datetime,
    snapshot_date: date,
    source: str = "etf_seed_list",
) -> NormalizedUniverse:
    """Build fallback entries for seed ETFs not already covered by SEC data.

    ``known_tickers`` should be the set of tickers already resolved via
    ``normalize_sec_universe_rows`` (case-insensitive, upper-cased) so we do
    not create a duplicate ticker-fallback security for something that
    already has a proper CIK-backed record.
    """
    result = NormalizedUniverse()
    for ticker in seed_tickers:
        ticker_upper = ticker.upper()
        if ticker_upper in known_tickers:
            continue
        security_id = security_id_for_ticker_fallback(ticker_upper)
        result.securities.append(
            Security(
                security_id=security_id,
                cik=None,
                company_name=None,
                primary_ticker=ticker_upper,
                exchange=None,
                asset_type="ETF",
                currency="USD",
                is_active=True,
                first_seen_at=retrieved_at,
                last_seen_at=retrieved_at,
                created_at=retrieved_at,
                updated_at=retrieved_at,
            )
        )
        result.identifiers.append(
            SecurityIdentifier(
                security_id=security_id,
                identifier_type="TICKER",
                identifier_value=ticker_upper,
                valid_from=snapshot_date,
                valid_to=None,
                source=source,
            )
        )
        result.snapshots.append(
            SecuritySnapshot(
                snapshot_date=snapshot_date,
                security_id=security_id,
                ticker=ticker_upper,
                company_name=None,
                exchange=None,
                source=source,
            )
        )
    return result
