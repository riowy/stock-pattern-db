from __future__ import annotations

from datetime import date

import polars as pl

from app.validation.rules import (
    check_invalid_dates,
    check_missing_recent_data,
    check_negative_values,
    check_ohlc_consistency,
    check_sudden_price_change,
    check_weekend_dates,
)


def _price_df(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "security_id": "S1",
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "adj_close": 10.5,
        "volume": 1000.0,
        "dividend": 0.0,
        "stock_split": 0.0,
    }
    full_rows = [{**defaults, **r} for r in rows]
    df = pl.DataFrame(full_rows)
    return df.with_columns(pl.col("date").str.to_date())


def test_check_negative_values_flags_negative_price() -> None:
    df = _price_df([{"date": "2024-01-02", "close": -5.0}])
    issues = check_negative_values(df)
    assert any(i["issue_type"] == "NEGATIVE_PRICE" for i in issues)


def test_check_negative_values_flags_negative_volume() -> None:
    df = _price_df([{"date": "2024-01-02", "volume": -100.0}])
    issues = check_negative_values(df)
    assert any(i["issue_type"] == "NEGATIVE_VOLUME" for i in issues)


def test_check_negative_values_no_issues_for_clean_data() -> None:
    df = _price_df([{"date": "2024-01-02"}])
    assert check_negative_values(df) == []


def test_check_ohlc_consistency_flags_high_below_low() -> None:
    df = _price_df([{"date": "2024-01-02", "high": 5.0, "low": 9.0, "open": 6.0, "close": 6.0}])
    issues = check_ohlc_consistency(df)
    assert any(i["issue_type"] == "OHLC_INCONSISTENT" for i in issues)


def test_check_ohlc_consistency_passes_valid_bar() -> None:
    df = _price_df([{"date": "2024-01-02", "open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0}])
    assert check_ohlc_consistency(df) == []


def test_check_weekend_dates() -> None:
    # 2024-01-06 is a Saturday, 2024-01-08 is a Monday
    df = _price_df([{"date": "2024-01-06"}, {"date": "2024-01-08"}])
    issues = check_weekend_dates(df)
    assert len(issues) == 1
    assert issues[0]["date"] == date(2024, 1, 6)


def test_check_invalid_dates_flags_future_date() -> None:
    df = _price_df([{"date": "2999-01-01"}])
    issues = check_invalid_dates(df)
    assert any(i["issue_type"] == "INVALID_DATE" for i in issues)


def test_check_sudden_price_change_flags_unexplained_move() -> None:
    df = _price_df(
        [
            {"date": "2024-01-02", "close": 100.0, "stock_split": 0.0},
            {"date": "2024-01-03", "close": 15.0, "stock_split": 0.0},  # -85% with no split
        ]
    )
    issues = check_sudden_price_change(df)
    assert any(i["issue_type"] == "SUDDEN_PRICE_CHANGE" and i["severity"] == "warning" for i in issues)


def test_check_sudden_price_change_recorded_split_is_info_not_warning() -> None:
    df = _price_df(
        [
            {"date": "2024-01-02", "close": 100.0, "stock_split": 0.0},
            {"date": "2024-01-03", "close": 25.0, "stock_split": 4.0},  # 4-for-1 split, -75%
        ]
    )
    issues = check_sudden_price_change(df)
    assert any(i["issue_type"] == "SUSPECTED_SPLIT_ADJUSTMENT" and i["severity"] == "info" for i in issues)
    assert not any(i["issue_type"] == "SUDDEN_PRICE_CHANGE" for i in issues)


def test_check_sudden_price_change_ignores_small_moves() -> None:
    df = _price_df(
        [
            {"date": "2024-01-02", "close": 100.0},
            {"date": "2024-01-03", "close": 102.0},
        ]
    )
    assert check_sudden_price_change(df) == []


def test_check_missing_recent_data_flags_stale_security() -> None:
    df = _price_df([{"date": "2020-01-02"}])
    issues = check_missing_recent_data(df, ["S1"], as_of=date(2024, 1, 1), max_gap_days=5)
    assert any(i["issue_type"] == "MISSING_RECENT_DATA" for i in issues)


def test_check_missing_recent_data_ok_for_fresh_security() -> None:
    df = _price_df([{"date": "2024-01-01"}])
    issues = check_missing_recent_data(df, ["S1"], as_of=date(2024, 1, 2), max_gap_days=5)
    assert issues == []


def test_check_missing_recent_data_flags_security_with_no_data_at_all() -> None:
    df = _price_df([{"date": "2024-01-01"}])
    issues = check_missing_recent_data(df, ["S1", "S2"], as_of=date(2024, 1, 2), max_gap_days=5)
    assert any(i["security_id"] == "S2" and i["issue_type"] == "MISSING_RECENT_DATA" for i in issues)
