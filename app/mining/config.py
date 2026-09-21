"""Data-mining defaults. Not trade rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

MINING_RESULT_PERSISTENCE_ENABLED = False
MINING_VERSION = "v1"

ANALYSIS_START = date(2018, 1, 2)
ANALYSIS_END = date(2023, 12, 29)
VALIDATION_START = date(2024, 1, 2)
# Backward-compatible alias: the 2024+ window was previously called TEST.
TEST_START = VALIDATION_START

FUTURE_HOLDOUT_START = date(2026, 9, 21)
FUTURE_HOLDOUT_EVALUATION_ENABLED = False

PATTERN_FREEZE_POLICY = (
    "Quantile cutoffs, winsor bounds, candidate pairs, and FDR selection freeze "
    "on the discover window. VALIDATION and FUTURE_HOLDOUT must not retune them."
)

DEFAULT_TARGET = "forward_excess_spy_20d"
AUX_TARGETS = (
    "forward_excess_spy_5d",
    "forward_excess_spy_10d",
    "forward_excess_spy_20d",
    "forward_return_5d",
    "forward_return_10d",
    "forward_return_20d",
)

MIN_PATTERN_ROWS = 1000
MIN_PATTERN_DATES = 100
MIN_PATTERN_SECURITIES = 30
INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
HIGH_OVERLAP = "HIGH_OVERLAP"

DEFAULT_FDR_Q = 0.10
DEFAULT_MAX_RULE_SIZE = 2
DEFAULT_TOP = 30
DEFAULT_EVENT_MODE = "state"
DEFAULT_COOLDOWN_SESSIONS = 0
COOLDOWN_GRID = (0, 5, 10, 20)
OVERLAP_CONDITIONAL = 0.95
OVERLAP_JACCARD = 0.90

WINSOR_P_LOW = 0.01
WINSOR_P_HIGH = 0.99

PRICE_FLOORS = (
    ("all", None),
    ("raw_close_ge_1", 1.0),
    ("raw_close_ge_5", 5.0),
    ("raw_close_ge_20", 20.0),
)

FAMILIES = (
    "TREND",
    "MOMENTUM",
    "VOLATILITY",
    "VOLUME",
    "BAND_CHANNEL",
    "ICHIMOKU",
    "BREAKOUT",
    "CANDLE",
    "MIXED",
)

SMOKE_PATTERNS = (
    "rsi14_ge_70",
    "rsi14_ge_70 AND volume_ratio20_gt_2",
    "adx_gt_25 AND bb_above_upper",
    "donchian_20_breakout AND volume_ratio20_gt_2",
)


@dataclass(frozen=True)
class WalkFold:
    name: str
    discover_start: date
    discover_end: date
    eval_start: date
    eval_end: date | None


WALK_FOLDS = (
    WalkFold("fold1_eval_2021", date(2018, 1, 2), date(2020, 12, 31), date(2021, 1, 4), date(2021, 12, 31)),
    WalkFold("fold2_eval_2022", date(2018, 1, 2), date(2021, 12, 31), date(2022, 1, 3), date(2022, 12, 30)),
    WalkFold("fold3_eval_2023", date(2018, 1, 2), date(2022, 12, 30), date(2023, 1, 3), date(2023, 12, 29)),
    WalkFold("fold4_eval_2024", date(2018, 1, 2), date(2023, 12, 29), date(2024, 1, 2), date(2024, 12, 31)),
    WalkFold("fold5_eval_2025", date(2018, 1, 2), date(2024, 12, 31), date(2025, 1, 2), date(2025, 12, 31)),
    WalkFold("fold6_eval_2026", date(2018, 1, 2), date(2025, 12, 31), date(2026, 1, 2), None),
)


from datetime import timedelta


def last_date_before_future_holdout() -> date:
    return FUTURE_HOLDOUT_START - timedelta(days=1)


def cap_end_before_future_holdout(end: date | None) -> date | None:
    """VALIDATION/walk-forward must not consume FUTURE_HOLDOUT dates while eval is disabled."""
    if FUTURE_HOLDOUT_EVALUATION_ENABLED:
        return end
    cap = last_date_before_future_holdout()
    if end is None:
        return cap
    return min(end, cap)
