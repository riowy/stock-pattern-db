"""Named research / scale-test / benchmark universes.

``tracked_securities`` stays the operational collection set. Universe
memberships are an overlay: one security can be in PROVIDER_SCALE_TEST and
RESEARCH_COMMON_EQUITY at the same time. Existing tracked rows are never
disabled when a new named universe is applied.
"""

from __future__ import annotations

from datetime import UTC, datetime

import duckdb
import polars as pl

from app.services.dataset_metadata_service import set_metadata

UNIVERSE_TYPE_KNOWN = "KNOWN"
UNIVERSE_TYPE_PROVIDER_SCALE_TEST = "PROVIDER_SCALE_TEST"
UNIVERSE_TYPE_RESEARCH_COMMON_EQUITY = "RESEARCH_COMMON_EQUITY"
UNIVERSE_TYPE_BENCHMARK = "BENCHMARK"

SELECTION_VERSION_V1 = "v1"


def upsert_memberships(
    con: duckdb.DuckDBPyConnection,
    universe_name: str,
    universe_type: str,
    security_ids: list[str],
    selection_rule: str,
    selection_version: str = SELECTION_VERSION_V1,
) -> int:
    if not security_ids:
        return 0
    now = datetime.now(UTC)
    df = pl.DataFrame(
        {
            "universe_name": [universe_name] * len(security_ids),
            "security_id": security_ids,
            "universe_type": [universe_type] * len(security_ids),
            "selection_version": [selection_version] * len(security_ids),
            "selection_rule": [selection_rule] * len(security_ids),
            "added_at": [now] * len(security_ids),
        }
    )
    con.register("_tmp_universe_mem", df)
    try:
        con.execute(
            """
            INSERT INTO universe_memberships
                (universe_name, security_id, universe_type, selection_version, selection_rule, added_at)
            SELECT universe_name, security_id, universe_type, selection_version, selection_rule, added_at
            FROM _tmp_universe_mem
            ON CONFLICT (universe_name, security_id) DO UPDATE SET
                universe_type = excluded.universe_type,
                selection_version = excluded.selection_version,
                selection_rule = excluded.selection_rule
            """
        )
    finally:
        con.unregister("_tmp_universe_mem")
    return len(security_ids)


def replace_memberships(
    con: duckdb.DuckDBPyConnection,
    universe_name: str,
    universe_type: str,
    security_ids: list[str],
    selection_rule: str,
    selection_version: str = SELECTION_VERSION_V1,
) -> int:
    """Drop previous members of this named universe, then insert the new set.

    Used when classification changes so CEV/KMPB-style names actually leave
    RESEARCH_COMMON_EQUITY instead of lingering via upsert-only.
    """
    con.execute("DELETE FROM universe_memberships WHERE universe_name = ?", [universe_name])
    if not security_ids:
        return 0
    return upsert_memberships(
        con, universe_name, universe_type, security_ids, selection_rule, selection_version
    )


def record_universe_metadata(
    con: duckdb.DuckDBPyConnection,
    universe_name: str,
    universe_type: str,
    security_count: int,
    selection_rule: str,
    selection_version: str = SELECTION_VERSION_V1,
    created_at: str | None = None,
) -> None:
    created = created_at or datetime.now(UTC).isoformat()
    set_metadata(con, universe_name, "universe_name", universe_name)
    set_metadata(con, universe_name, "universe_type", universe_type)
    set_metadata(con, universe_name, "created_at", created)
    set_metadata(con, universe_name, "selection_version", selection_version)
    set_metadata(con, universe_name, "selection_rule", selection_rule)
    set_metadata(con, universe_name, "security_count", str(security_count))
    set_metadata(con, universe_name, "is_investment_universe", "false")
    set_metadata(con, universe_name, "survivorship_safe", "false")
    set_metadata(con, universe_name, "point_in_time_security_master", "false")
    set_metadata(con, universe_name, "instrument_class_complete", "false")


def membership_security_ids(con: duckdb.DuckDBPyConnection, universe_name: str) -> list[str]:
    rows = con.execute(
        "SELECT security_id FROM universe_memberships WHERE universe_name = ? ORDER BY security_id",
        [universe_name],
    ).fetchall()
    return [r[0] for r in rows]


def membership_security_ids_by_type(con: duckdb.DuckDBPyConnection, universe_type: str) -> list[str]:
    rows = con.execute(
        "SELECT DISTINCT security_id FROM universe_memberships WHERE universe_type = ? ORDER BY security_id",
        [universe_type],
    ).fetchall()
    return [r[0] for r in rows]


def seed_scale_test_memberships_from_tracked(con: duckdb.DuckDBPyConnection) -> int:
    """Preserve existing research-scale-* tracked rows as PROVIDER_SCALE_TEST."""
    rows = con.execute(
        """
        SELECT security_id, tracking_reason FROM tracked_securities
        WHERE enabled = TRUE AND tracking_reason LIKE 'research-scale-%'
        """
    ).fetchall()
    if not rows:
        return 0
    by_name: dict[str, list[str]] = {}
    for sid, reason in rows:
        by_name.setdefault(reason or "research-scale", []).append(sid)
    total = 0
    rule = (
        "preserved from tracked_securities.tracking_reason; hash-sample scale-test; "
        "may include warrants/preferred/units; NOT research common equity"
    )
    for name, sids in by_name.items():
        total += upsert_memberships(
            con, name, UNIVERSE_TYPE_PROVIDER_SCALE_TEST, sids, selection_rule=rule
        )
        record_universe_metadata(
            con, name, UNIVERSE_TYPE_PROVIDER_SCALE_TEST, len(sids), selection_rule=rule
        )
    return total
