"""v1 labels_forward_returns column lists."""

from __future__ import annotations

LABEL_VERSION_V1 = "v1"

FORWARD_HORIZONS = (1, 3, 5, 10, 20)
PATH_HORIZONS = (5, 10, 20)

IDENTITY_COLUMNS = [
    "security_id",
    "ticker_at_time",
    "date",
    "label_version",
    "calculated_at",
    "calculation_code_version",
]

FORWARD_RETURN_COLUMNS = [f"forward_return_{h}d" for h in FORWARD_HORIZONS]
FORWARD_EXCESS_COLUMNS = [f"forward_excess_spy_{h}d" for h in FORWARD_HORIZONS]
MAX_GAIN_COLUMNS = [f"max_gain_next_{h}d" for h in PATH_HORIZONS]
MAX_DRAWDOWN_COLUMNS = [f"max_drawdown_next_{h}d" for h in PATH_HORIZONS]

LABEL_VALUE_COLUMNS = (
    FORWARD_RETURN_COLUMNS + FORWARD_EXCESS_COLUMNS + MAX_GAIN_COLUMNS + MAX_DRAWDOWN_COLUMNS
)
LABEL_COLUMNS = IDENTITY_COLUMNS + LABEL_VALUE_COLUMNS
