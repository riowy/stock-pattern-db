"""Configurable ticker-suffix heuristics for instrument classification.

This is not a complete security master. The free SEC company-tickers feed
does not reliably distinguish warrants, rights, units, or preferred shares.
Patterns below are the only v1 signals we trust enough to record, and even
those can false-positive (e.g. a 5-letter NASDAQ common ending in W).

``instrument_class_complete=false`` is the corresponding metadata flag.
Ambiguous names stay AMBIGUOUS -- they are never forced to COMMON_EQUITY.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config.etf_seed import SEED_ETF_TICKERS

CLASS_COMMON_EQUITY = "COMMON_EQUITY"
CLASS_WARRANT = "WARRANT"
CLASS_RIGHT = "RIGHT"
CLASS_UNIT = "UNIT"
CLASS_PREFERRED = "PREFERRED"
CLASS_ETF = "ETF"
CLASS_CLOSED_END_FUND = "CLOSED_END_FUND"
CLASS_FUND = "FUND"
CLASS_DEBT = "DEBT"
CLASS_AMBIGUOUS = "AMBIGUOUS"
CLASS_OTHER = "OTHER"

CONF_HIGH = "high"
CONF_MEDIUM = "medium"
CONF_LOW = "low"

SOURCE_ETF_SEED = "etf_seed"
SOURCE_TICKER_SUFFIX = "ticker_suffix_heuristic"
SOURCE_SIMPLE_TICKER = "simple_us_ticker"

NON_COMMON_CLASSES = frozenset(
    {
        CLASS_WARRANT,
        CLASS_RIGHT,
        CLASS_UNIT,
        CLASS_PREFERRED,
        CLASS_ETF,
        CLASS_CLOSED_END_FUND,
        CLASS_FUND,
        CLASS_DEBT,
        CLASS_AMBIGUOUS,
        CLASS_OTHER,
    }
)
RESEARCH_ELIGIBLE_CLASSES = frozenset({CLASS_COMMON_EQUITY})

MAJOR_US_EXCHANGES = frozenset({"NYSE", "Nasdaq", "CBOE", "NYSE American", "NYSEArca", "NYSE ARCA"})


@dataclass(frozen=True)
class SuffixRule:
    """First matching rule wins. ``pattern`` is matched against the upper ticker."""

    name: str
    pattern: str
    instrument_class: str
    confidence: str
    exclusion_reason: str | None

    def compiled(self) -> re.Pattern[str]:
        return re.compile(self.pattern)


# Order matters. Hyphenated share-class common (-B) must not be swallowed by
# preferred (-P[A-Z]) -- preferred is listed first. 5-letter *W is AMBIGUOUS
# rather than WARRANT because FINW-style commons exist.
DEFAULT_SUFFIX_RULES: tuple[SuffixRule, ...] = (
    SuffixRule("preferred_hyphen", r".*-P[A-Z]?$", CLASS_PREFERRED, CONF_HIGH, "preferred_share_suffix"),
    SuffixRule("preferred_pr", r".*-PR[A-Z]?$", CLASS_PREFERRED, CONF_HIGH, "preferred_share_suffix"),
    SuffixRule("warrant_hyphen_wt", r".*-WT[A-Z]?$", CLASS_WARRANT, CONF_HIGH, "warrant_suffix"),
    SuffixRule("warrant_hyphen_ws", r".*-WS[A-Z]?$", CLASS_WARRANT, CONF_HIGH, "warrant_suffix"),
    SuffixRule("warrant_hyphen_w", r".*-W$", CLASS_WARRANT, CONF_HIGH, "warrant_suffix"),
    SuffixRule("right_hyphen_ri", r".*-RI$", CLASS_RIGHT, CONF_HIGH, "right_suffix"),
    SuffixRule("right_hyphen_rt", r".*-RT$", CLASS_RIGHT, CONF_HIGH, "right_suffix"),
    SuffixRule("right_hyphen_r", r".*-R$", CLASS_RIGHT, CONF_HIGH, "right_suffix"),
    SuffixRule("unit_hyphen_un", r".*-UN$", CLASS_UNIT, CONF_HIGH, "unit_suffix"),
    SuffixRule("unit_hyphen_u", r".*-U$", CLASS_UNIT, CONF_HIGH, "unit_suffix"),
    SuffixRule("share_class_common", r"^[A-Z]{1,5}-[A-C]$", CLASS_COMMON_EQUITY, CONF_MEDIUM, None),
    SuffixRule("possible_warrant_fifth_w", r"^[A-Z]{4}W$", CLASS_AMBIGUOUS, CONF_LOW, "possible_warrant_fifth_letter_w"),
    SuffixRule("possible_unit_fifth_u", r"^[A-Z]{4}U$", CLASS_AMBIGUOUS, CONF_LOW, "possible_unit_fifth_letter_u"),
    SuffixRule("possible_right_fifth_r", r"^[A-Z]{4}R$", CLASS_AMBIGUOUS, CONF_LOW, "possible_right_fifth_letter_r"),
)


@dataclass(frozen=True)
class InstrumentClassification:
    ticker: str
    instrument_class: str
    classification_source: str
    classification_confidence: str
    exclusion_reason: str | None


@dataclass(frozen=True)
class InstrumentSignals:
    """Inputs the classifier is allowed to use. Missing fields stay None."""

    ticker: str | None
    exchange: str | None = None
    company_name: str | None = None
    cik: str | None = None
    security_title: str | None = None
    entity_type: str | None = None
    form_types: tuple[str, ...] = ()


def classify_ticker(
    ticker: str | None,
    *,
    exchange: str | None = None,
    etf_tickers: frozenset[str] | None = None,
    suffix_rules: tuple[SuffixRule, ...] = DEFAULT_SUFFIX_RULES,
    company_name: str | None = None,
    security_title: str | None = None,
    form_types: tuple[str, ...] = (),
    entity_type: str | None = None,
) -> InstrumentClassification:
    """Back-compat wrapper. Prefer ``classify_instrument`` for new callers."""
    return classify_instrument(
        InstrumentSignals(
            ticker=ticker,
            exchange=exchange,
            company_name=company_name,
            security_title=security_title,
            entity_type=entity_type,
            form_types=form_types,
        ),
        etf_tickers=etf_tickers,
        suffix_rules=suffix_rules,
    )


def classify_instrument(
    signals: InstrumentSignals,
    *,
    etf_tickers: frozenset[str] | None = None,
    suffix_rules: tuple[SuffixRule, ...] = DEFAULT_SUFFIX_RULES,
) -> InstrumentClassification:
    """Classify using SEC metadata first, ticker suffix last.

    Order: ETF seed → official/parsed security title → registrant type
    (forms / entity / SEC name) → ticker suffix → simple US ticker.
    """
    from app.config.sec_instrument_signals import (
        SOURCE_SEC_REGISTRANT_TYPE,
        SOURCE_SEC_SECURITY_TITLE,
        classify_from_investment_company_forms,
        match_closed_end_registrant_name,
        match_debt_security_title,
        official_title_for_ticker,
    )

    raw = (signals.ticker or "").strip().upper()
    seeds = etf_tickers if etf_tickers is not None else frozenset(t.upper() for t in SEED_ETF_TICKERS)
    if not raw:
        return InstrumentClassification("", CLASS_AMBIGUOUS, SOURCE_SIMPLE_TICKER, CONF_LOW, "missing_ticker")
    if raw in seeds:
        return InstrumentClassification(raw, CLASS_ETF, SOURCE_ETF_SEED, CONF_HIGH, "benchmark_or_sector_etf")

    official = official_title_for_ticker(raw)
    title = signals.security_title or (official.security_title if official else None)
    if official is not None:
        return InstrumentClassification(
            raw,
            official.instrument_class,
            SOURCE_SEC_SECURITY_TITLE,
            CONF_HIGH,
            official.exclusion_reason,
        )
    debt_hit = match_debt_security_title(title)
    if debt_hit:
        return InstrumentClassification(
            raw,
            CLASS_DEBT,
            SOURCE_SEC_SECURITY_TITLE,
            CONF_HIGH,
            "debt_security_title",
        )

    form_class = classify_from_investment_company_forms(signals.form_types)
    if form_class:
        return InstrumentClassification(
            raw,
            form_class,
            SOURCE_SEC_REGISTRANT_TYPE,
            CONF_HIGH,
            "investment_company_sec_forms",
        )
    entity = (signals.entity_type or "").strip().lower()
    if entity in {"investment company", "investment-company", "other"} and match_closed_end_registrant_name(
        signals.company_name
    ):
        return InstrumentClassification(
            raw,
            CLASS_CLOSED_END_FUND,
            SOURCE_SEC_REGISTRANT_TYPE,
            CONF_HIGH,
            "sec_entity_type_investment_company",
        )
    name_hit = match_closed_end_registrant_name(signals.company_name)
    if name_hit:
        return InstrumentClassification(
            raw,
            CLASS_CLOSED_END_FUND,
            SOURCE_SEC_REGISTRANT_TYPE,
            CONF_HIGH,
            "registered_investment_company_registrant_name",
        )

    for rule in suffix_rules:
        if rule.compiled().match(raw):
            return InstrumentClassification(
                raw,
                rule.instrument_class,
                SOURCE_TICKER_SUFFIX,
                rule.confidence,
                rule.exclusion_reason,
            )

    if re.fullmatch(r"[A-Z][A-Z0-9]{0,4}", raw) and (signals.exchange in MAJOR_US_EXCHANGES or signals.exchange is None):
        return InstrumentClassification(raw, CLASS_COMMON_EQUITY, SOURCE_SIMPLE_TICKER, CONF_MEDIUM, None)

    return InstrumentClassification(
        raw,
        CLASS_AMBIGUOUS,
        SOURCE_SIMPLE_TICKER,
        CONF_LOW,
        "unrecognized_ticker_pattern",
    )
