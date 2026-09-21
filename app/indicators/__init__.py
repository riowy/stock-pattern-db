"""On-demand technical indicator engine (memory-only)."""

from app.indicators.engine import IndicatorEngine
from app.indicators.registry import all_specs, specs_for
from app.indicators.schema import INDICATOR_PERSISTENCE_ENABLED, IndicatorSpec

__all__ = [
    "INDICATOR_PERSISTENCE_ENABLED",
    "IndicatorEngine",
    "IndicatorSpec",
    "all_specs",
    "specs_for",
]
