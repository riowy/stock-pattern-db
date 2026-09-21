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
FEATURES_DAILY_DATASET = "features_daily"
LABELS_FORWARD_RETURNS_DATASET = "labels_forward_returns"


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
    set_metadata(con, PRICES_DAILY_DATASET, "commercial_use_safe", "false")
    set_metadata(con, PRICES_DAILY_DATASET, "usage", "research/prototype only")

    set_metadata(con, FEATURES_DAILY_DATASET, "feature_version", "v1")
    set_metadata(con, FEATURES_DAILY_DATASET, "signal_timing", "market_close")
    set_metadata(con, FEATURES_DAILY_DATASET, "historical_universe_complete", "false")
    set_metadata(con, FEATURES_DAILY_DATASET, "survivorship_safe", "false")
    set_metadata(con, FEATURES_DAILY_DATASET, "point_in_time_security_master", "false")
    set_metadata(con, FEATURES_DAILY_DATASET, "price_adjustment_point_in_time", "false")
    set_metadata(con, FEATURES_DAILY_DATASET, "macro_point_in_time", "false")
    set_metadata(
        con,
        FEATURES_DAILY_DATASET,
        "macro_excluded_reason",
        "FRED observation_date is not a verified publication/revision timestamp; v1 does not join macro.",
    )
    set_metadata(con, FEATURES_DAILY_DATASET, "sector_relative_strength", "null_until_trusted_mapping")
    set_metadata(con, FEATURES_DAILY_DATASET, "usage", "research/prototype only")
    set_metadata(con, FEATURES_DAILY_DATASET, "commercial_use_safe", "false")
    set_metadata(con, FEATURES_DAILY_DATASET, "instrument_class_complete", "false")

    set_metadata(con, PRICES_DAILY_DATASET, "price_adjustment_point_in_time", "false")
    set_metadata(con, PRICES_DAILY_DATASET, "instrument_class_complete", "false")

    set_metadata(con, LABELS_FORWARD_RETURNS_DATASET, "label_version", "v1")
    set_metadata(con, LABELS_FORWARD_RETURNS_DATASET, "return_basis", "adjusted_close")
    set_metadata(con, LABELS_FORWARD_RETURNS_DATASET, "benchmark", "SPY")
    set_metadata(con, LABELS_FORWARD_RETURNS_DATASET, "historical_universe_complete", "false")
    set_metadata(con, LABELS_FORWARD_RETURNS_DATASET, "survivorship_safe", "false")
    set_metadata(
        con,
        LABELS_FORWARD_RETURNS_DATASET,
        "max_drawdown_definition",
        "min(adj_close(t+1..t+h)/adj_close(t)-1); close-to-close vs entry close, not path-dependent peak-to-trough",
    )
    set_metadata(con, LABELS_FORWARD_RETURNS_DATASET, "horizon_unit", "trading_sessions")
    set_metadata(con, LABELS_FORWARD_RETURNS_DATASET, "usage", "research/prototype only")
    set_metadata(con, LABELS_FORWARD_RETURNS_DATASET, "commercial_use_safe", "false")
    set_metadata(con, LABELS_FORWARD_RETURNS_DATASET, "instrument_class_complete", "false")
    set_metadata(con, LABELS_FORWARD_RETURNS_DATASET, "price_adjustment_point_in_time", "false")
