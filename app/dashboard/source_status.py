"""Read-only source-data freshness for the research dashboard Overview.

No network calls. No catalog/lake writes. Demo and empty modes may report
availability as ``demo`` or ``unavailable`` without probing production data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import duckdb

from app.config.lake_datasets import get_lake_dataset
from app.config.settings import Settings
from app.services.market_calendar import MarketCalendarService


@dataclass(frozen=True)
class SourceFreshnessSummary:
    """Compact source-data freshness snapshot for Overview rendering."""

    availability: str  # "ok" | "unavailable" | "demo"
    message: str
    prices_latest: str | None = None
    features_latest: str | None = None
    labels_latest: str | None = None
    expected_latest_session: str | None = None
    tracked_note: str | None = None
    details: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "availability": self.availability,
            "message": self.message,
            "prices_latest": self.prices_latest,
            "features_latest": self.features_latest,
            "labels_latest": self.labels_latest,
            "expected_latest_session": self.expected_latest_session,
            "tracked_note": self.tracked_note,
            "details": list(self.details),
        }


class SourceStatusProvider(Protocol):
    def get_summary(self) -> SourceFreshnessSummary: ...


@dataclass(frozen=True)
class StaticSourceStatusProvider:
    """Explicit demo / empty / test summary — never touches disk or network."""

    summary: SourceFreshnessSummary

    def get_summary(self) -> SourceFreshnessSummary:
        return self.summary


def demo_source_status() -> SourceFreshnessSummary:
    return SourceFreshnessSummary(
        availability="demo",
        message="Source data freshness unavailable in demo mode (synthetic fixtures only).",
        details=("no-network", "no-write", "demo"),
    )


def unavailable_source_status(reason: str = "Source catalog/lake not available for read-only probe.") -> SourceFreshnessSummary:
    return SourceFreshnessSummary(
        availability="unavailable",
        message=reason,
        details=("no-network", "no-write"),
    )


def _max_date_from_lake(settings: Settings, dataset_key: str, date_column: str = "date") -> str | None:
    """Read max(date) from lake Parquet only. Never opens catalog.duckdb."""
    ds = get_lake_dataset(settings.lake_dir, dataset_key)
    if not ds.has_any_files():
        return None
    mem = duckdb.connect(":memory:")
    try:
        glob_path = ds.glob_pattern().replace("'", "''")
        row = mem.execute(
            f"SELECT max({date_column}) FROM read_parquet('{glob_path}', hive_partitioning = true)"
        ).fetchone()
        return str(row[0]) if row and row[0] is not None else None
    except Exception:  # noqa: BLE001
        return None
    finally:
        mem.close()


class ReadOnlySourceStatusProvider:
    """Probe lake Parquet for latest dates. Read-only; no network; no writes."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def get_summary(self) -> SourceFreshnessSummary:
        settings = self._settings
        if not settings.lake_dir.exists():
            return unavailable_source_status(
                "Source data freshness unavailable: lake directory does not exist "
                "(read-only probe; no directories created)."
            )

        calendar = MarketCalendarService(settings.market_calendar, settings.market_data_grace_minutes)
        expected = str(calendar.latest_expected_session())

        prices_latest = _max_date_from_lake(settings, "prices_daily")
        features_latest = _max_date_from_lake(settings, "features_daily")
        labels_latest = _max_date_from_lake(settings, "labels_forward_returns")

        if prices_latest is None and features_latest is None and labels_latest is None:
            return SourceFreshnessSummary(
                availability="unavailable",
                message="Source data freshness unavailable: no prices/features/labels lake files found.",
                expected_latest_session=expected,
                details=("no-network", "no-write", "empty-lake"),
            )

        return SourceFreshnessSummary(
            availability="ok",
            message="Read-only lake freshness (no network, no writes).",
            prices_latest=prices_latest,
            features_latest=features_latest,
            labels_latest=labels_latest,
            expected_latest_session=expected,
            tracked_note="Dates from lake Parquet max(date); not a live provider sync.",
            details=("no-network", "no-write", "read-only-lake"),
        )


def default_dashboard_source_status(*, demo: bool = False, settings: Settings | None = None) -> SourceStatusProvider:
    """Provider used by CLI: demo → static demo; else read-only lake probe when possible."""
    if demo:
        return StaticSourceStatusProvider(demo_source_status())
    if settings is None:
        return StaticSourceStatusProvider(unavailable_source_status())
    return ReadOnlySourceStatusProvider(settings)
