from __future__ import annotations

from datetime import date

from app.config.instrument_patterns import (
    CLASS_AMBIGUOUS,
    CLASS_CLOSED_END_FUND,
    CLASS_COMMON_EQUITY,
    CLASS_DEBT,
    CLASS_ETF,
    CLASS_PREFERRED,
    CLASS_RIGHT,
    CLASS_UNIT,
    CLASS_WARRANT,
    classify_ticker,
)
from app.config.sec_instrument_signals import SOURCE_SEC_REGISTRANT_TYPE, SOURCE_SEC_SECURITY_TITLE
from app.validation.numeric_tolerance import OHLC_ABS_TOL, approximately_ge, approximately_le
from app.validation.rules import check_ohlc_consistency
from tests.test_validation_rules import _price_df


def test_rounding_0_0001_on_24_dollar_bar_is_info_not_critical() -> None:
    df = _price_df([{"date": "2024-01-02", "open": 24.2000, "high": 24.2000, "low": 24.2001, "close": 24.2000}])
    issues = check_ohlc_consistency(df)
    assert any(i["issue_type"] == "OHLC_ROUNDING_TOLERANCE" and i["severity"] == "info" for i in issues)
    assert not any(i["issue_type"] == "OHLC_INCONSISTENT" for i in issues)


def test_true_ohlc_violation_remains_critical() -> None:
    df = _price_df([{"date": "2024-01-02", "open": 10.15, "high": 10.08, "low": 10.00, "close": 10.10}])
    issues = check_ohlc_consistency(df)
    assert any(i["issue_type"] == "OHLC_INCONSISTENT" and i["severity"] == "critical" for i in issues)


def test_tolerance_boundary_helpers() -> None:
    assert approximately_ge(10.0, 10.0)
    assert approximately_ge(10.0, 10.0 + OHLC_ABS_TOL)
    assert not approximately_ge(10.0, 10.05)
    assert approximately_le(24.2000, 24.2001)
    assert approximately_ge(24.2001, 24.2000)


def test_classify_warrant_hyphen() -> None:
    cls = classify_ticker("GCTS-WT")
    assert cls.instrument_class == CLASS_WARRANT
    assert cls.exclusion_reason == "warrant_suffix"
    assert cls.classification_confidence == "high"


def test_classify_preferred_hyphen() -> None:
    cls = classify_ticker("ATH-PA")
    assert cls.instrument_class == CLASS_PREFERRED


def test_classify_unit_and_right() -> None:
    assert classify_ticker("FOO-U").instrument_class == CLASS_UNIT
    assert classify_ticker("OCAC-RI").instrument_class == CLASS_RIGHT


def test_classify_fifth_letter_w_is_ambiguous_not_forced_common() -> None:
    cls = classify_ticker("GFAIW")
    assert cls.instrument_class == CLASS_AMBIGUOUS
    assert cls.exclusion_reason == "possible_warrant_fifth_letter_w"
    bblgw = classify_ticker("BBLGW")
    assert bblgw.instrument_class == CLASS_AMBIGUOUS
    finw = classify_ticker("FINW", exchange="Nasdaq")
    assert finw.instrument_class == CLASS_COMMON_EQUITY


def test_classify_share_class_common_and_simple_ticker() -> None:
    assert classify_ticker("LEN-B").instrument_class == CLASS_COMMON_EQUITY
    aapl = classify_ticker("AAPL", exchange="Nasdaq")
    assert aapl.instrument_class == CLASS_COMMON_EQUITY
    assert aapl.classification_confidence == "medium"


def test_classify_benchmark_etf() -> None:
    spy = classify_ticker("SPY")
    assert spy.instrument_class == CLASS_ETF
    assert spy.classification_source == "etf_seed"


def test_classify_cev_as_closed_end_fund_from_sec_registrant_name() -> None:
    cls = classify_ticker(
        "CEV",
        exchange="NYSE",
        company_name="Eaton Vance California Municipal Income Trust",
    )
    assert cls.instrument_class == CLASS_CLOSED_END_FUND
    assert cls.classification_source == SOURCE_SEC_REGISTRANT_TYPE
    assert cls.classification_confidence == "high"
    assert cls.instrument_class != CLASS_COMMON_EQUITY


def test_classify_kmpb_as_debt_from_sec_security_title() -> None:
    cls = classify_ticker("KMPB", exchange="NYSE", company_name="Kemper Corporation")
    assert cls.instrument_class == CLASS_DEBT
    assert cls.classification_source == SOURCE_SEC_SECURITY_TITLE
    assert cls.exclusion_reason == "junior_subordinated_debenture"
    assert cls.instrument_class != CLASS_COMMON_EQUITY


def test_classify_n2_forms_as_closed_end_fund() -> None:
    cls = classify_ticker("ABCD", exchange="NYSE", company_name="Some Operating Co", form_types=("N-2", "N-CSR"))
    assert cls.instrument_class == CLASS_CLOSED_END_FUND
    assert cls.classification_source == SOURCE_SEC_REGISTRANT_TYPE


def test_operating_company_name_stays_common() -> None:
    cls = classify_ticker("AAPL", exchange="Nasdaq", company_name="Apple Inc.")
    assert cls.instrument_class == CLASS_COMMON_EQUITY
    assert cls.classification_source == "simple_us_ticker"
