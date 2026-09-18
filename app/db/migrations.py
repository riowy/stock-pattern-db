"""Idempotent data migrations, run on every CLI invocation after schema DDL.

These are deliberately cheap and side-effect-free when there is nothing to
do, so calling this on every command (not just ``stockdb init``) is safe --
existing installations pick up new tables/behavior automatically without
ever deleting or rebuilding the DuckDB catalog.
"""

from __future__ import annotations

import duckdb

from app.config.settings import Settings
from app.services.dataset_metadata_service import seed_research_integrity_metadata
from app.services.tracked_universe_service import auto_register_from_existing_price_data
from app.utils.logging import get_logger

logger = get_logger("migrations")


def run_migrations(con: duckdb.DuckDBPyConnection, settings: Settings) -> None:
    newly_tracked = auto_register_from_existing_price_data(con, settings)
    if newly_tracked:
        logger.info(
            "Migration: auto-registered %d security(ies) with existing price data into tracked_securities",
            newly_tracked,
        )
    seed_research_integrity_metadata(con, settings)
