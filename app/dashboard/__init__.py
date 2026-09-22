"""Local research dashboard package."""

from __future__ import annotations

from app.dashboard.server import DashboardApp, run_dashboard
from app.dashboard.source_status import (
    ReadOnlySourceStatusProvider,
    SourceFreshnessSummary,
    StaticSourceStatusProvider,
    default_dashboard_source_status,
    demo_source_status,
    unavailable_source_status,
)

__all__ = [
    "DashboardApp",
    "ReadOnlySourceStatusProvider",
    "SourceFreshnessSummary",
    "StaticSourceStatusProvider",
    "default_dashboard_source_status",
    "demo_source_status",
    "run_dashboard",
    "unavailable_source_status",
]
