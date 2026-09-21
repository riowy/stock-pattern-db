"""Persistence seam. v1 has no writer."""

from __future__ import annotations

from app.indicators.schema import INDICATOR_PERSISTENCE_ENABLED


class IndicatorPersistenceDisabled(RuntimeError):
    """Raised if any caller tries to persist indicator output in v1."""


def assert_no_indicator_persist() -> None:
    if INDICATOR_PERSISTENCE_ENABLED:
        raise IndicatorPersistenceDisabled(
            "INDICATOR_PERSISTENCE_ENABLED is true but no writer is implemented."
        )


def persist_indicators(*_args, **_kwargs):  # noqa: ANN002, ANN003
    raise IndicatorPersistenceDisabled(
        "Indicator persistence is disabled. Results are memory-only."
    )
