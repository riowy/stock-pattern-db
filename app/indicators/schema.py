"""Indicator metadata and persistence policy.

v1 computes indicators in memory only. Persistence flags exist so a later
writer can reuse the same registry without changing formulas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

INDICATOR_PERSISTENCE_ENABLED = False
FORMULA_VERSION = "v1"


@dataclass(frozen=True)
class IndicatorSpec:
    indicator_id: str
    display_name: str
    category: str
    output_columns: tuple[str, ...]
    required_columns: tuple[str, ...] = ("adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close")
    parameters: dict = field(default_factory=dict)
    minimum_history: int = 2
    warmup_sessions: int = 1
    causal_safe: bool = True
    mining_enabled: bool = True
    description: str = ""
    formula_version: str = FORMULA_VERSION
    plot_only: bool = False
