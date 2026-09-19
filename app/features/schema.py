"""v1 features_daily column lists (identity + calculated features)."""

from __future__ import annotations

FEATURE_VERSION_V1 = "v1"

IDENTITY_COLUMNS = [
    "security_id",
    "ticker_at_time",
    "date",
    "feature_version",
    "calculated_at",
    "calculation_code_version",
]

RETURN_COLUMNS = ["ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d", "ret_60d"]
MA_DISTANCE_COLUMNS = [
    "ma_5_distance",
    "ma_10_distance",
    "ma_20_distance",
    "ma_50_distance",
    "ma_120_distance",
    "ma_200_distance",
]
MA_TREND_COLUMNS = ["ma_20_slope", "ma_50_slope", "ma_200_slope"]
RSI_COLUMNS = ["rsi_14"]
VOLATILITY_COLUMNS = ["volatility_10d", "volatility_20d", "volatility_60d"]
ATR_COLUMNS = ["atr_14", "atr_pct_14"]
VOLUME_COLUMNS = ["volume_ratio_5d", "volume_ratio_20d", "volume_ratio_60d"]
HIGH_LOW_COLUMNS = [
    "distance_high_20d",
    "distance_high_60d",
    "distance_high_252d",
    "distance_low_20d",
    "distance_low_60d",
    "distance_low_252d",
]
CANDLE_COLUMNS = [
    "gap_return",
    "intraday_return",
    "daily_range_pct",
    "upper_wick_pct",
    "lower_wick_pct",
]
RELATIVE_COLUMNS = [
    "rel_spy_5d",
    "rel_spy_20d",
    "rel_spy_60d",
    "rel_qqq_5d",
    "rel_qqq_20d",
    "rel_qqq_60d",
    "rel_sector_5d",
    "rel_sector_20d",
    "rel_sector_60d",
]
MARKET_CONTEXT_COLUMNS = [
    "spy_ret_5d",
    "spy_ret_20d",
    "spy_ret_60d",
    "spy_ma200_distance",
    "qqq_ret_20d",
    "vix_close",
    "vix_change_5d",
    "vix_change_20d",
]

FEATURE_VALUE_COLUMNS = (
    RETURN_COLUMNS
    + MA_DISTANCE_COLUMNS
    + MA_TREND_COLUMNS
    + RSI_COLUMNS
    + VOLATILITY_COLUMNS
    + ATR_COLUMNS
    + VOLUME_COLUMNS
    + HIGH_LOW_COLUMNS
    + CANDLE_COLUMNS
    + RELATIVE_COLUMNS
    + MARKET_CONTEXT_COLUMNS
)

FEATURE_COLUMNS = IDENTITY_COLUMNS + FEATURE_VALUE_COLUMNS

# Internal derived columns that must never be persisted as features.
INTERNAL_COLUMNS = [
    "adjustment_factor",
    "adjusted_open",
    "adjusted_high",
    "adjusted_low",
    "adjusted_close",
    "adjustment_valid",
]
