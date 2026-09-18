"""Machine-readable dataset metadata (research-integrity flags, etc.).

This is the machine-readable counterpart to the README's "known
limitations" section -- a future features/labels/backtest layer should be
able to read e.g. ``survivorship_safe`` programmatically instead of relying
on someone having read the docs.
"""

from __future__ import annotations

from datetime import UTC, datetime

import duckdb

from app.config.settings import Settings
from app.providers.registry import get_price_provider_class

PRICES_DAILY_DATASET = "prices_daily"


def set_metadata(con: duckdb.DuckDBPyConnection, dataset_name: str, key: str, value: str) -> None:
    con.execute(
        """
        INSERT INTO dataset_metadata (dataset_name, metadata_key, metadata_value, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (dataset_name, metadata_key) DO UPDATE SET
            metadata_value = excluded.metadata_value,
            updated_at = excluded.updated_at
        """,
        [dataset_name, key, value, datetime.now(UTC)],
    )


def get_metadata(con: duckdb.DuckDBPyConnection, dataset_name: str) -> dict[str, str]:
    rows = con.execute(
        "SELECT metadata_key, metadata_value FROM dataset_metadata WHERE dataset_name = ?", [dataset_name]
    ).fetchall()
    return {k: v for k, v in rows}


def seed_research_integrity_metadata(con: duckdb.DuckDBPyConnection, settings: Settings) -> None:
    """Idempotent -- safe to call on every startup. Reflects the *current*
    ``PRICE_PROVIDER`` setting, so switching providers automatically updates
    ``research_only_price_provider`` next time any command runs."""
    try:
        provider_cls = get_price_provider_class(settings.price_provider)
        research_only = not provider_cls.capabilities.commercial_use_safe
    except ValueError:
        research_only = True  # unknown provider: assume the conservative default

    set_metadata(con, PRICES_DAILY_DATASET, "historical_universe_complete", "false")
    set_metadata(con, PRICES_DAILY_DATASET, "survivorship_safe", "false")
    set_metadata(con, PRICES_DAILY_DATASET, "point_in_time_security_master", "false")
    set_metadata(con, PRICES_DAILY_DATASET, "research_only_price_provider", str(research_only).lower())
    set_metadata(con, PRICES_DAILY_DATASET, "price_provider", settings.price_provider)
