"""Research-analysis constants.

Feature exploration and threshold discovery happen on DEVELOPMENT only.
VALIDATION is a frozen holdout. Validation statistics must never be used
to retune development rules, regime cutoffs, or feature thresholds.

These outputs are research statistics, not buy/sell recommendations.
"""

from __future__ import annotations

from datetime import date

RESEARCH_VERSION_V1 = "v1"
RESEARCH_UNIVERSE_NAME = "research-common-equity-500"
RESEARCH_VIEW_NAME = "research_common_equity_daily_v1"

# Inclusive calendar bounds; actual rows are trading dates inside the bounds.
DEVELOPMENT_START = date(2018, 1, 1)
DEVELOPMENT_END = date(2023, 12, 31)
VALIDATION_START = date(2024, 1, 1)

SPLIT_DEVELOPMENT = "development"
SPLIT_VALIDATION = "validation"

HORIZONS = (5, 10, 20)
FORWARD_RETURN_TARGETS = tuple(f"forward_return_{h}d" for h in HORIZONS)
FORWARD_EXCESS_TARGETS = tuple(f"forward_excess_spy_{h}d" for h in HORIZONS)
ALL_TARGETS = FORWARD_RETURN_TARGETS + FORWARD_EXCESS_TARGETS

MIN_QUANTILE_OBSERVATIONS = 1000
MIN_QUANTILE_DATES = 100
INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"

N_QUANTILES = 5

COVERAGE_FEATURES: tuple[str, ...] = (
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "ma_20_distance",
    "ma_50_distance",
    "ma_200_distance",
    "ma_20_slope",
    "ma_50_slope",
    "ma_200_slope",
    "rsi_14",
    "volatility_20d",
    "volatility_60d",
    "atr_pct_14",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "volume_ratio_60d",
    "distance_high_20d",
    "distance_high_60d",
    "distance_high_252d",
    "distance_low_20d",
    "distance_low_60d",
    "distance_low_252d",
    "gap_return",
    "intraday_return",
    "daily_range_pct",
    "rel_spy_5d",
    "rel_spy_20d",
    "rel_spy_60d",
    "rel_qqq_5d",
    "rel_qqq_20d",
    "rel_qqq_60d",
    "spy_ret_20d",
    "spy_ma200_distance",
    "vix_close",
    "vix_change_5d",
    "vix_change_20d",
)

SMOKE_FEATURES: tuple[str, ...] = (
    "ret_20d",
    "rel_spy_20d",
    "volume_ratio_20d",
    "rsi_14",
    "distance_high_252d",
    "volatility_20d",
)

HIGH_NULL_RATIO = 0.30

PRICE_BUCKETS: tuple[tuple[str, float | None, float | None], ...] = (
    ("lt_1", None, 1.0),
    ("1_to_5", 1.0, 5.0),
    ("5_to_20", 5.0, 20.0),
    ("ge_20", 20.0, None),
)

# Sensitivity audit floors. Not optimized against validation performance.
PRICE_FLOOR_SUBSETS: tuple[tuple[str, float | None], ...] = (
    ("research_all", None),
    ("price_ge_1", 1.0),
    ("price_ge_5", 5.0),
    ("price_ge_20", 20.0),
)

REDUNDANCY_FEATURES: tuple[str, ...] = (
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "rel_spy_5d",
    "rel_spy_20d",
    "rel_spy_60d",
    "rsi_14",
    "distance_high_252d",
    "volume_ratio_20d",
    "volatility_20d",
    "atr_pct_14",
)

RANK_CORR_REDUNDANT_ABS = 0.95

WINSOR_P_LOW = 0.01
WINSOR_P_HIGH = 0.99

SUMMARY_TARGET = "forward_excess_spy_20d"
