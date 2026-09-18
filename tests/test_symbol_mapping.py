from __future__ import annotations

import duckdb

from app.db.schema import apply_schema
from app.normalization.symbols import resolve_provider_symbol, seed_symbol_mappings
from app.providers.price.yfinance_provider import YFinancePriceProvider


def test_default_symbol_translation_replaces_dot_with_dash() -> None:
    p = YFinancePriceProvider.__new__(YFinancePriceProvider)  # bypass __init__ (no network/settings needed)
    assert p.to_provider_symbol("BRK.B") == "BRK-B"
    assert p.to_provider_symbol("AAPL") == "AAPL"


def test_resolve_provider_symbol_without_db_falls_back_to_default() -> None:
    result = resolve_provider_symbol(None, "BRK.B", "yfinance", "BRK-B-DEFAULT")
    assert result == "BRK-B-DEFAULT"


def test_resolve_provider_symbol_db_override_wins() -> None:
    con = duckdb.connect(":memory:")
    apply_schema(con)
    seed_symbol_mappings(con)

    # Seeded override should be returned regardless of the "default" passed in.
    result = resolve_provider_symbol(con, "BRK.B", "yfinance", "some-other-default")
    assert result == "BRK-B"


def test_resolve_provider_symbol_no_override_uses_default() -> None:
    con = duckdb.connect(":memory:")
    apply_schema(con)
    seed_symbol_mappings(con)

    result = resolve_provider_symbol(con, "AAPL", "yfinance", "AAPL")
    assert result == "AAPL"


def test_seed_symbol_mappings_is_idempotent() -> None:
    con = duckdb.connect(":memory:")
    apply_schema(con)
    n1 = seed_symbol_mappings(con)
    n2 = seed_symbol_mappings(con)
    assert n1 == n2
    count = con.execute("SELECT count(*) FROM symbol_mappings").fetchone()[0]
    assert count == n1  # no duplicate rows from re-seeding
