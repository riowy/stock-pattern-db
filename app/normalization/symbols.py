"""Canonical <-> provider symbol mapping.

Analysis code and the ingestion layer should only ever deal with canonical
tickers (or better, ``security_id``). Provider adapters own their own
default translation rule (``PriceProvider.to_provider_symbol``), but the
``symbol_mappings`` DuckDB table lets you override/extend that without
touching provider code -- e.g. if a provider changes its spelling
convention for one specific ticker.
"""

from __future__ import annotations

from datetime import UTC, datetime

import duckdb

from app.config.symbol_overrides import SYMBOL_OVERRIDE_SEED


def seed_symbol_mappings(con: duckdb.DuckDBPyConnection) -> int:
    """Load the config seed overrides into the DB (idempotent upsert)."""
    now = datetime.now(UTC)
    count = 0
    for canonical, provider_map in SYMBOL_OVERRIDE_SEED.items():
        for provider, provider_symbol in provider_map.items():
            con.execute(
                """
                INSERT INTO symbol_mappings (canonical_symbol, provider, provider_symbol, source, updated_at)
                VALUES (?, ?, ?, 'config_seed', ?)
                ON CONFLICT (canonical_symbol, provider)
                DO UPDATE SET provider_symbol = excluded.provider_symbol,
                              source = excluded.source,
                              updated_at = excluded.updated_at
                """,
                [canonical, provider, provider_symbol, now],
            )
            count += 1
    return count


def resolve_provider_symbol(
    con: duckdb.DuckDBPyConnection | None,
    canonical_symbol: str,
    provider: str,
    default_symbol: str,
) -> str:
    """DB override (if any) wins; otherwise fall back to the provider's default."""
    if con is None:
        return default_symbol
    row = con.execute(
        "SELECT provider_symbol FROM symbol_mappings WHERE canonical_symbol = ? AND provider = ?",
        [canonical_symbol, provider],
    ).fetchone()
    return row[0] if row else default_symbol
