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
from app.services.instrument_classification_service import refresh_instrument_classifications
from app.services.tracked_universe_service import (
    auto_register_from_existing_price_data,
    enable_feature_tracking_for_price_tracked,
)
from app.services.universe_membership_service import seed_scale_test_memberships_from_tracked
from app.utils.logging import get_logger

logger = get_logger("migrations")


def run_migrations(con: duckdb.DuckDBPyConnection, settings: Settings) -> None:
    newly_tracked = auto_register_from_existing_price_data(con, settings)
    if newly_tracked:
        logger.info(
            "Migration: auto-registered %d security(ies) with existing price data into tracked_securities",
            newly_tracked,
        )
    newly_featured = enable_feature_tracking_for_price_tracked(con)
    if newly_featured:
        logger.info(
            "Migration: enabled feature_tracking on %d price-tracked security(ies)",
            newly_featured,
        )
    seed_research_integrity_metadata(con, settings)
    classified = refresh_instrument_classifications(con, settings)
    if classified:
        logger.info("Migration: classified %d securities (instrument_class_complete=false)", classified)
    seeded = seed_scale_test_memberships_from_tracked(con)
    if seeded:
        logger.info("Migration: seeded %d PROVIDER_SCALE_TEST universe memberships", seeded)
