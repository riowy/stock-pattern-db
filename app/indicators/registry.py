"""Indicator registry. Plot-only / causal_safe=false specs are excluded from mining."""

from __future__ import annotations

from app.indicators.schema import IndicatorSpec

_SPECS: dict[str, IndicatorSpec] = {}


def register(spec: IndicatorSpec) -> IndicatorSpec:
    _SPECS[spec.indicator_id] = spec
    return spec


def get(indicator_id: str) -> IndicatorSpec:
    return _SPECS[indicator_id]


def all_specs() -> list[IndicatorSpec]:
    return list(_SPECS.values())


def specs_for(*, category: str | None = None, indicator_id: str | None = None, mining_only: bool = False) -> list[IndicatorSpec]:
    items = all_specs()
    if category:
        items = [s for s in items if s.category == category]
    if indicator_id:
        items = [s for s in items if s.indicator_id == indicator_id or indicator_id in s.output_columns]
    if mining_only:
        items = [s for s in items if s.mining_enabled and s.causal_safe and not s.plot_only]
    return items


def categories() -> list[str]:
    return sorted({s.category for s in _SPECS.values()})
