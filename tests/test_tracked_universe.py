from __future__ import annotations

from datetime import UTC, datetime

import duckdb
import polars as pl
import pytest

from app.config.settings import Settings
from app.db.schema import apply_schema
from app.services.market_calendar import MarketCalendarService
from app.services.tracked_universe_service import (
    add_tracked,
    add_tracked_if_absent,
    auto_register_from_existing_price_data,
    get_tracked_filing_ciks,
    get_tracked_price_security_ids,
    list_tracked,
    remove_tracked,
)
from app.utils.parquet_io import LakeDataset
from app.validation.runner import validate_prices

CALENDAR = MarketCalendarService()


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    apply_schema(c)
    return c


def _insert_fake_securities(con: duckdb.DuckDBPyConnection, n: int) -> list[str]:
    now = datetime.now(UTC)
    ids = [f"SEC{i:04d}" for i in range(n)]
    df = pl.DataFrame(
        {
            "security_id": ids,
            "cik": [None] * n,
            "company_name": [f"Fake Co {i}" for i in range(n)],
            "primary_ticker": [f"TCK{i:04d}" for i in range(n)],
            "exchange": ["NYSE"] * n,
            "asset_type": ["EQUITY"] * n,
            "currency": ["USD"] * n,
            "is_active": [True] * n,
            "first_seen_at": [now] * n,
            "last_seen_at": [now] * n,
            "created_at": [now] * n,
            "updated_at": [now] * n,
        }
    )
    con.register("_tmp_sec", df)
    con.execute("INSERT INTO securities SELECT * FROM _tmp_sec")
    con.unregister("_tmp_sec")
    return ids


def _write_price_row(settings: Settings, security_id: str, d, close: float = 100.0) -> None:
    ds = LakeDataset(settings.prices_daily_dir, "date", ["security_id", "date"], ["security_id", "date"])
    df = pl.DataFrame(
        {
            "security_id": [security_id],
            "date": [d.isoformat()],
            "ticker_at_time": ["X"],
            "open": [close],
            "high": [close],
            "low": [close],
            "close": [close],
            "adj_close": [close],
            "volume": [1000.0],
            "dividend": [0.0],
            "stock_split": [0.0],
            "currency": ["USD"],
            "provider": ["test"],
            "retrieved_at": [datetime.now(UTC)],
        }
    ).with_columns(pl.col("date").str.to_date())
    ds.write_increment(df, run_id="test-run")


# --- Test A: 100 known securities, 5 tracked -> MISSING_RECENT_DATA only for the 5 ------
def test_validation_scoped_to_tracked_universe_not_full_security_master(con, settings) -> None:
    ids = _insert_fake_securities(con, 100)
    tracked_ids = ids[:5]
    add_tracked(con, tracked_ids, reason="test")

    fresh_day = CALENDAR.latest_expected_session()
    # 4 fresh, 1 deliberately stale, and NO data at all for the other 95 (untracked).
    for sid in tracked_ids[:4]:
        _write_price_row(settings, sid, fresh_day)
    stale_day = CALENDAR.sessions_ago(fresh_day, 30)
    _write_price_row(settings, tracked_ids[4], stale_day)

    summary = validate_prices(settings, con, symbols=None, dry_run=False, all_universe=False)

    assert summary.scope == "tracked"
    assert summary.scope_size == 5

    flagged = con.execute(
        "SELECT DISTINCT security_id FROM data_quality_issues WHERE issue_type = 'MISSING_RECENT_DATA' AND NOT resolved"
    ).fetchall()
    flagged_ids = {r[0] for r in flagged}
    # Only the deliberately-stale tracked security is flagged.
    assert flagged_ids == {tracked_ids[4]}
    # None of the 95 untracked (but SEC-known) securities are ever flagged.
    assert flagged_ids.isdisjoint(set(ids[5:]))


def test_all_universe_flag_widens_scope(con, settings) -> None:
    ids = _insert_fake_securities(con, 10)
    tracked_ids = ids[:2]
    add_tracked(con, tracked_ids, reason="test")
    fresh_day = CALENDAR.latest_expected_session()
    for sid in tracked_ids:
        _write_price_row(settings, sid, fresh_day)

    summary = validate_prices(settings, con, symbols=None, dry_run=False, all_universe=True)
    assert summary.scope == "all_universe"
    assert summary.scope_size == 10  # all active securities, not just the 2 tracked


def test_stale_missing_recent_data_issues_are_resolved_after_rescope(con, settings) -> None:
    """Simulates the historical bug: MISSING_RECENT_DATA recorded for an
    untracked security should be resolved once tracked-scoped validation runs."""
    ids = _insert_fake_securities(con, 5)
    # Manually insert a stale bogus issue for an untracked security, as if it
    # had been created by the old "check every active security" behavior.
    con.execute(
        """
        INSERT INTO data_quality_issues (issue_id, dataset, security_id, date, issue_type, severity, details, detected_at, resolved)
        VALUES ('legacy1', 'prices_daily', ?, NULL, 'MISSING_RECENT_DATA', 'warning', 'bogus', ?, FALSE)
        """,
        [ids[4], datetime.now(UTC)],
    )
    tracked_ids = ids[:1]
    add_tracked(con, tracked_ids, reason="test")
    fresh_day = CALENDAR.latest_expected_session()
    _write_price_row(settings, tracked_ids[0], fresh_day)

    validate_prices(settings, con, symbols=None, dry_run=False, all_universe=False)

    row = con.execute("SELECT resolved FROM data_quality_issues WHERE issue_id = 'legacy1'").fetchone()
    assert row[0] is True


# --- add / remove -------------------------------------------------------------
def test_add_tracked_upserts_and_reenables(con) -> None:
    ids = _insert_fake_securities(con, 3)
    add_tracked(con, ids, reason="first")
    remove_tracked(con, [ids[0]])
    assert set(get_tracked_price_security_ids(con)) == set(ids[1:])

    add_tracked(con, [ids[0]], reason="re-add")
    assert set(get_tracked_price_security_ids(con)) == set(ids)


def test_remove_tracked_is_soft_delete(con) -> None:
    ids = _insert_fake_securities(con, 2)
    add_tracked(con, ids, reason="test")
    remove_tracked(con, [ids[0]])

    all_rows = list_tracked(con, include_disabled=True)
    assert len(all_rows) == 2
    removed = next(r for r in all_rows if r.security_id == ids[0])
    assert removed.enabled is False
    assert removed.removed_at is not None


def test_get_tracked_filing_ciks_groups_by_cik(con) -> None:
    now = datetime.now(UTC)
    df = pl.DataFrame(
        {
            "security_id": ["A1", "A2"],
            "cik": ["0000000001", "0000000001"],  # same CIK, two share classes
            "company_name": ["X", "X"],
            "primary_ticker": ["AAA", "AAB"],
            "exchange": ["NYSE", "NYSE"],
            "asset_type": ["EQUITY", "EQUITY"],
            "currency": ["USD", "USD"],
            "is_active": [True, True],
            "first_seen_at": [now, now],
            "last_seen_at": [now, now],
            "created_at": [now, now],
            "updated_at": [now, now],
        }
    )
    con.register("_tmp", df)
    con.execute("INSERT INTO securities SELECT * FROM _tmp")
    con.unregister("_tmp")

    add_tracked(con, ["A1", "A2"], reason="test", filings_tracking=True)
    grouped = get_tracked_filing_ciks(con)
    assert grouped == {"0000000001": ["A1", "A2"]}


# --- Test J: migration is idempotent ------------------------------------------
def test_migration_auto_register_is_idempotent(con, settings) -> None:
    ids = _insert_fake_securities(con, 3)
    for sid in ids:
        _write_price_row(settings, sid, CALENDAR.latest_expected_session())

    added_first = auto_register_from_existing_price_data(con, settings)
    assert added_first == 3
    count_after_first = con.execute("SELECT count(*) FROM tracked_securities").fetchone()[0]

    added_second = auto_register_from_existing_price_data(con, settings)
    assert added_second == 0  # nothing new -- already registered
    count_after_second = con.execute("SELECT count(*) FROM tracked_securities").fetchone()[0]
    assert count_after_first == count_after_second == 3


def test_migration_never_resurrects_explicitly_removed_security(con, settings) -> None:
    ids = _insert_fake_securities(con, 2)
    for sid in ids:
        _write_price_row(settings, sid, CALENDAR.latest_expected_session())

    auto_register_from_existing_price_data(con, settings)
    remove_tracked(con, [ids[0]])

    # Re-running the migration must NOT silently re-enable the removed security.
    added = auto_register_from_existing_price_data(con, settings)
    assert added == 0
    tracked_ids = set(get_tracked_price_security_ids(con))
    assert ids[0] not in tracked_ids
    assert ids[1] in tracked_ids


def test_add_tracked_if_absent_never_overwrites_existing_row(con) -> None:
    ids = _insert_fake_securities(con, 1)
    add_tracked(con, ids, reason="manual", price_tracking=True, filings_tracking=False)
    remove_tracked(con, ids)  # disable it

    added = add_tracked_if_absent(con, ids, reason="migration")
    assert added == 0
    row = list_tracked(con, include_disabled=True)[0]
    assert row.enabled is False  # migration did not resurrect it
