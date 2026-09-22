"""Persistence gates for pattern-research state.

Defaults OFF. Importing this module never creates production files.
"""

from __future__ import annotations

from pathlib import Path

from app.config.settings import Settings, get_settings
from app.patterns.store import PatternResearchStore


# Module-level defaults mirroring settings (also used by tests without Settings).
PATTERN_REGISTRY_PERSISTENCE_ENABLED = False
DAILY_SIGNAL_PERSISTENCE_ENABLED = False


def registry_db_path(settings: Settings | None = None) -> Path:
    s = settings or get_settings()
    return s.pattern_registry_path


def is_registry_persistence_enabled(settings: Settings | None = None) -> bool:
    s = settings or get_settings()
    return bool(s.pattern_registry_persistence_enabled)


def is_signal_persistence_enabled(settings: Settings | None = None) -> bool:
    s = settings or get_settings()
    return bool(s.daily_signal_persistence_enabled)


def open_research_store(
    settings: Settings | None = None,
    *,
    force_memory: bool = False,
    db_path: Path | None = None,
) -> PatternResearchStore:
    """Open a store. Production DuckDB is used only when persistence is enabled.

    ``force_memory`` always uses in-memory (tests / demo). When persistence is
    disabled, never creates ``data/state/pattern_registry.duckdb``.
    """
    s = settings or get_settings()
    persist = (not force_memory) and bool(s.pattern_registry_persistence_enabled)
    signal_persist = bool(s.daily_signal_persistence_enabled)
    path = db_path if db_path is not None else registry_db_path(s)
    store = PatternResearchStore(
        persist=persist,
        db_path=path if persist else None,
        signal_persist=signal_persist if persist else True,  # in-memory may always hold signals
    )
    return store.open()


def assert_no_production_registry_file(settings: Settings | None = None) -> None:
    """Raise if a production registry file exists while persistence is disabled."""
    s = settings or get_settings()
    if s.pattern_registry_persistence_enabled:
        return
    path = registry_db_path(s)
    if path.exists():
        raise RuntimeError(
            f"Pattern registry file exists at {path} but PATTERN_REGISTRY_PERSISTENCE_ENABLED=false. "
            "Remove it manually or enable persistence explicitly."
        )
