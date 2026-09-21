"""SEC registrant / security-title signals for instrument classification.

The free ``company_tickers_exchange.json`` feed gives a registrant *name*
and ticker, not a complete security master. It does not include per-ticker
security titles. Classification therefore uses, in order:

1. Official security titles when known (``SEC_SECURITY_TITLE``).
2. Investment-company form types / entity type (``SEC_REGISTRANT_TYPE``).
3. SEC registrant-name patterns that identify registered investment
   companies / closed-end funds (also ``SEC_REGISTRANT_TYPE``).
4. Ticker-suffix heuristics (last resort).

``instrument_class_complete`` remains false. Ambiguous names are never
forced to COMMON_EQUITY.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config.instrument_patterns import (
    CLASS_CLOSED_END_FUND,
    CLASS_DEBT,
    CLASS_FUND,
)

SOURCE_SEC_SECURITY_TITLE = "SEC_SECURITY_TITLE"
SOURCE_SEC_REGISTRANT_TYPE = "SEC_REGISTRANT_TYPE"

# Forms that identify an investment-company registrant (not an operating 10-K issuer).
INVESTMENT_COMPANY_FORMS = frozenset(
    {
        "N-CSR",
        "N-CSRS",
        "N-CEN",
        "N-PORT",
        "N-PORT/A",
        "N-2",
        "N-2/A",
        "N-1A",
        "N-1A/A",
        "485BPOS",
        "24F-2NT",
        "N-PX",
        "N-MFP",
        "N-MFP2",
    }
)
CLOSED_END_FORMS = frozenset({"N-2", "N-2/A"})
OPEN_END_FORMS = frozenset({"N-1A", "N-1A/A"})


@dataclass(frozen=True)
class OfficialSecurityTitle:
    """Join key is ticker because the SEC ticker feed has no title column.

    ``security_title`` is the official listing/prospectus title, not a
    ticker heuristic. Matching is on that title text.
    """

    ticker: str
    security_title: str
    instrument_class: str
    exclusion_reason: str


# Verified from official listing / prospectus text. Extend as titles are confirmed.
# Not a performance screen and not a ticker-suffix rule.
OFFICIAL_SECURITY_TITLES: tuple[OfficialSecurityTitle, ...] = (
    OfficialSecurityTitle(
        ticker="KMPB",
        security_title="5.875% Fixed-Rate Reset Junior Subordinated Debentures due 2062",
        instrument_class=CLASS_DEBT,
        exclusion_reason="junior_subordinated_debenture",
    ),
)

# SEC registrant *name* patterns that indicate a registered investment company / CEF.
# Applied to company_tickers_exchange.json ``name``, not to the ticker.
REGISTRANT_CLOSED_END_NAME_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"MUNICIPAL INCOME TRUST",
        r"MUNICIPAL BOND (FUND|TRUST)",
        r"MUNICIPAL OPPORTUNIT",
        r"CLOSED[-\s]END",
        r"INVESTMENT TRUST",
        r"REGISTERED INVESTMENT",
        r"INCOME TRUST\s*$",
        r"BOND TRUST\s*$",
    )
)

DEBT_TITLE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"JUNIOR SUBORDINATED",
        r"DEBENTURE",
        r"NOTES DUE",
        r"SENIOR NOTES",
        r"SUBORDINATED NOTES",
        r"FIXED-RATE RESET",
    )
)


def official_title_for_ticker(ticker: str | None) -> OfficialSecurityTitle | None:
    if not ticker:
        return None
    key = ticker.strip().upper()
    for row in OFFICIAL_SECURITY_TITLES:
        if row.ticker == key:
            return row
    return None


def match_closed_end_registrant_name(company_name: str | None) -> str | None:
    if not company_name:
        return None
    for pat in REGISTRANT_CLOSED_END_NAME_PATTERNS:
        if pat.search(company_name):
            return pat.pattern
    return None


def match_debt_security_title(security_title: str | None) -> str | None:
    if not security_title:
        return None
    for pat in DEBT_TITLE_PATTERNS:
        if pat.search(security_title):
            return pat.pattern
    return None


def classify_from_investment_company_forms(form_types: tuple[str, ...] | list[str] | None) -> str | None:
    if not form_types:
        return None
    forms = {str(f).upper() for f in form_types}
    if forms & INVESTMENT_COMPANY_FORMS:
        if forms & CLOSED_END_FORMS and not (forms & OPEN_END_FORMS):
            return CLASS_CLOSED_END_FUND
        if forms & OPEN_END_FORMS and not (forms & CLOSED_END_FORMS):
            return CLASS_FUND
        return CLASS_CLOSED_END_FUND
    return None
