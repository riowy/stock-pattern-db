"""Research sample view, load, split, and lake reconciliation.

``research_common_equity_daily_v1`` is an analysis view. It does not mutate
the Parquet lake. Benchmark ETFs are joinable as market-context columns
already stored on features; they are not rows in the stock sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import duckdb
import polars as pl

from app.config.instrument_patterns import CLASS_COMMON_EQUITY
from app.config.settings import Settings
from app.db.schema import create_lake_views
from app.features.schema import FEATURE_VALUE_COLUMNS
from app.labels.schema import LABEL_VALUE_COLUMNS
from app.research.config import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    RESEARCH_UNIVERSE_NAME,
    RESEARCH_VIEW_NAME,
    SPLIT_DEVELOPMENT,
    SPLIT_VALIDATION,
    VALIDATION_START,
)
from app.services.tracked_universe_service import (
    get_tracked_feature_security_ids,
    get_tracked_price_security_ids,
)
from app.services.universe_membership_service import (
    UNIVERSE_TYPE_BENCHMARK,
    UNIVERSE_TYPE_PROVIDER_SCALE_TEST,
    UNIVERSE_TYPE_RESEARCH_COMMON_EQUITY,
)


def assign_split(value: date) -> str:
    if value <= DEVELOPMENT_END:
        return SPLIT_DEVELOPMENT
    return SPLIT_VALIDATION


def split_expr() -> pl.Expr:
    return (
        pl.when(pl.col("date") <= DEVELOPMENT_END)
        .then(pl.lit(SPLIT_DEVELOPMENT))
        .otherwise(pl.lit(SPLIT_VALIDATION))
        .alias("split")
    )


def filter_split(df: pl.DataFrame, split: str) -> pl.DataFrame:
    if split == SPLIT_DEVELOPMENT:
        return df.filter(
            (pl.col("date") >= DEVELOPMENT_START) & (pl.col("date") <= DEVELOPMENT_END)
        )
    if split == SPLIT_VALIDATION:
        return df.filter(pl.col("date") >= VALIDATION_START)
    raise ValueError(f"Unknown split {split!r}")


def research_common_equity_view_sql(universe_name: str = RESEARCH_UNIVERSE_NAME) -> str:
    feature_cols = ", ".join(f"f.{c}" for c in FEATURE_VALUE_COLUMNS)
    label_cols = ", ".join(f"l.{c}" for c in LABEL_VALUE_COLUMNS)
    # universe_name is a project constant, never user SQL.
    return f"""
        CREATE OR REPLACE VIEW {RESEARCH_VIEW_NAME} AS
        SELECT
            f.security_id,
            f.ticker_at_time AS ticker,
            f.date,
            f.feature_version,
            l.label_version,
            {feature_cols},
            {label_cols},
            COALESCE(ic.instrument_class, 'AMBIGUOUS') AS instrument_class,
            um.universe_name,
            p.close AS raw_close,
            (crit.security_id IS NULL) AS data_quality_valid
        FROM features_daily f
        INNER JOIN labels_forward_returns l
          ON f.security_id = l.security_id AND f.date = l.date
        INNER JOIN (
            SELECT DISTINCT security_id, universe_name
            FROM universe_memberships
            WHERE universe_name = '{universe_name}'
              AND universe_type = '{UNIVERSE_TYPE_RESEARCH_COMMON_EQUITY}'
        ) um ON um.security_id = f.security_id
        LEFT JOIN instrument_classifications ic ON ic.security_id = f.security_id
        LEFT JOIN prices_daily p
          ON p.security_id = f.security_id AND p.date = f.date
        LEFT JOIN (
            SELECT DISTINCT security_id, date
            FROM data_quality_issues
            WHERE dataset = 'prices_daily'
              AND severity = 'critical'
              AND resolved = FALSE
        ) crit ON crit.security_id = f.security_id AND crit.date = f.date
        WHERE COALESCE(ic.instrument_class, 'AMBIGUOUS') = '{CLASS_COMMON_EQUITY}'
    """


def create_research_common_equity_view(con: duckdb.DuckDBPyConnection, created: set[str]) -> None:
    if "features_daily" not in created or "labels_forward_returns" not in created:
        return
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if "universe_memberships" not in tables:
        return
    con.execute(research_common_equity_view_sql())


def load_research_frame(
    settings: Settings,
    con: duckdb.DuckDBPyConnection,
    *,
    require_valid: bool = True,
) -> pl.DataFrame:
    create_lake_views(con, settings)
    names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if RESEARCH_VIEW_NAME not in names:
        return pl.DataFrame()
    df = con.execute(f"SELECT * FROM {RESEARCH_VIEW_NAME}").pl()
    if df.height == 0:
        return df
    df = df.with_columns(split_expr())
    if require_valid and "data_quality_valid" in df.columns:
        df = df.filter(pl.col("data_quality_valid"))
    return df


@dataclass
class CoverageBucket:
    security_id: str
    ticker: str | None
    has_price: bool
    has_feature: bool
    has_label: bool
    price_rows: int
    feature_rows: int
    label_rows: int
    universes: list[str]


@dataclass
class ReconciliationReport:
    price_securities: int = 0
    feature_securities: int = 0
    label_securities: int = 0
    price_rows: int = 0
    feature_rows: int = 0
    label_rows: int = 0
    all_three: int = 0
    price_only: int = 0
    feature_only: int = 0
    label_only: int = 0
    partial: int = 0
    row_count_mismatches: int = 0
    tracked: int = 0
    tracked_price: int = 0
    tracked_feature: int = 0
    tracked_without_prices: int = 0
    research_common_equity_500: int = 0
    research_common_equity_500_with_prices: int = 0
    benchmark: int = 0
    provider_scale_test: int = 0
    orphan_rows: bool = False
    cause: str = ""
    tracked_without_price_tickers: list[str] = field(default_factory=list)


def _count_distinct(con: duckdb.DuckDBPyConnection, table: str) -> tuple[int, int]:
    names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if table not in names:
        return 0, 0
    return con.execute(f"SELECT count(*), count(DISTINCT security_id) FROM {table}").fetchone()


def _membership_count(con: duckdb.DuckDBPyConnection, universe_type: str, name: str | None = None) -> int:
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if "universe_memberships" not in tables:
        return 0
    if name:
        row = con.execute(
            "SELECT count(DISTINCT security_id) FROM universe_memberships "
            "WHERE universe_type = ? AND universe_name = ?",
            [universe_type, name],
        ).fetchone()
    else:
        row = con.execute(
            "SELECT count(DISTINCT security_id) FROM universe_memberships WHERE universe_type = ?",
            [universe_type],
        ).fetchone()
    return int(row[0] or 0)


def reconcile_datasets(settings: Settings, con: duckdb.DuckDBPyConnection) -> ReconciliationReport:
    create_lake_views(con, settings)
    price_rows, price_sec = _count_distinct(con, "prices_daily")
    feat_rows, feat_sec = _count_distinct(con, "features_daily")
    lab_rows, lab_sec = _count_distinct(con, "labels_forward_returns")
    report = ReconciliationReport(
        price_securities=price_sec,
        feature_securities=feat_sec,
        label_securities=lab_sec,
        price_rows=price_rows,
        feature_rows=feat_rows,
        label_rows=lab_rows,
        tracked=con.execute("SELECT count(*) FROM tracked_securities WHERE enabled = TRUE").fetchone()[0] or 0,
        tracked_price=len(get_tracked_price_security_ids(con)),
        tracked_feature=len(get_tracked_feature_security_ids(con)),
        research_common_equity_500=_membership_count(
            con, UNIVERSE_TYPE_RESEARCH_COMMON_EQUITY, RESEARCH_UNIVERSE_NAME
        ),
        benchmark=_membership_count(con, UNIVERSE_TYPE_BENCHMARK),
        provider_scale_test=_membership_count(con, UNIVERSE_TYPE_PROVIDER_SCALE_TEST),
    )

    names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if "prices_daily" in names:
        report.research_common_equity_500_with_prices = con.execute(
            """
            SELECT count(DISTINCT um.security_id)
            FROM universe_memberships um
            JOIN prices_daily p ON p.security_id = um.security_id
            WHERE um.universe_name = ?
            """,
            [RESEARCH_UNIVERSE_NAME],
        ).fetchone()[0] or 0
        missing = con.execute(
            """
            SELECT COALESCE(ic.ticker, t.security_id)
            FROM tracked_securities t
            LEFT JOIN prices_daily p ON p.security_id = t.security_id
            LEFT JOIN instrument_classifications ic ON ic.security_id = t.security_id
            WHERE t.enabled = TRUE AND p.security_id IS NULL
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchall()
        report.tracked_without_price_tickers = [r[0] for r in missing]
        report.tracked_without_prices = len(report.tracked_without_price_tickers)

    if "prices_daily" in names and "features_daily" in names and "labels_forward_returns" in names:
        rows = con.execute(
            """
            WITH p AS (SELECT security_id, count(*) n FROM prices_daily GROUP BY 1),
                 f AS (SELECT security_id, count(*) n FROM features_daily GROUP BY 1),
                 l AS (SELECT security_id, count(*) n FROM labels_forward_returns GROUP BY 1)
            SELECT
              p.n IS NOT NULL, f.n IS NOT NULL, l.n IS NOT NULL,
              COALESCE(p.n, 0), COALESCE(f.n, 0), COALESCE(l.n, 0)
            FROM p
            FULL OUTER JOIN f ON p.security_id = f.security_id
            FULL OUTER JOIN l ON COALESCE(p.security_id, f.security_id) = l.security_id
            """
        ).fetchall()
        for hp, hf, hl, pn, fn, ln in rows:
            if hp and hf and hl:
                report.all_three += 1
                if pn != fn or pn != ln:
                    report.row_count_mismatches += 1
            elif hp and not hf and not hl:
                report.price_only += 1
            elif hf and not hp and not hl:
                report.feature_only += 1
            elif hl and not hp and not hf:
                report.label_only += 1
            else:
                report.partial += 1
        report.orphan_rows = (
            report.price_only + report.feature_only + report.label_only + report.partial + report.row_count_mismatches
        ) > 0

    if not report.orphan_rows and report.price_securities == report.feature_securities == report.label_securities:
        report.cause = (
            "Lake views share the same security_id set. The larger tracked count "
            f"({report.tracked}) includes memberships without price rows "
            f"({report.tracked_without_prices} tickers, typically unused benchmark ETFs). "
            "Status must report lake distinct securities separately from tracked universe size."
        )
    elif report.orphan_rows:
        report.cause = (
            "Orphan or mismatched security coverage across prices/features/labels."
        )
    else:
        report.cause = "Security counts differ across lake datasets."
    return report


def _membership_ids(con: duckdb.DuckDBPyConnection, universe_type: str, name: str | None = None) -> set[str]:
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if "universe_memberships" not in tables:
        return set()
    if name:
        rows = con.execute(
            "SELECT DISTINCT security_id FROM universe_memberships "
            "WHERE universe_type = ? AND universe_name = ?",
            [universe_type, name],
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT DISTINCT security_id FROM universe_memberships WHERE universe_type = ?",
            [universe_type],
        ).fetchall()
    return {r[0] for r in rows}


def daily_collection_scope(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Operational daily targets: tracked universe, never the full master."""
    known = con.execute("SELECT count(*) FROM securities").fetchone()[0] or 0
    tracked_price = set(get_tracked_price_security_ids(con))
    research = _membership_ids(con, UNIVERSE_TYPE_RESEARCH_COMMON_EQUITY, RESEARCH_UNIVERSE_NAME)
    scale = _membership_ids(con, UNIVERSE_TYPE_PROVIDER_SCALE_TEST)
    benches = _membership_ids(con, UNIVERSE_TYPE_BENCHMARK)
    extras = tracked_price & scale - research
    tracked_benchmarks = tracked_price & benches
    other = tracked_price - research - extras - tracked_benchmarks
    return {
        "known_securities": int(known),
        "tracked_price": len(tracked_price),
        "tracked_feature": len(get_tracked_feature_security_ids(con)),
        "research_common_equities": len(research),
        "research_common_equity_500": len(research),
        "provider_scale_test_extras": len(extras),
        "benchmarks": len(tracked_benchmarks),
        "other_tracked_price": len(other),
        "total_tracked_price_targets": len(tracked_price),
        "benchmark": _membership_count(con, UNIVERSE_TYPE_BENCHMARK),
    }


BUCKET_PRICE_WITH_FEATURE = "PRICE_WITH_FEATURE"
BUCKET_PRICE_NO_FEATURE_EXPECTED = "PRICE_NO_FEATURE_EXPECTED"
BUCKET_PRICE_NO_FEATURE_UNEXPECTED = "PRICE_NO_FEATURE_UNEXPECTED"


@dataclass
class PriceFeatureRowRecon:
    price_rows: int = 0
    feature_rows: int = 0
    price_with_feature: int = 0
    price_no_feature_expected: int = 0
    price_no_feature_unexpected: int = 0
    expected_not_feature_tracked: int = 0
    unexpected_examples: list[dict] = field(default_factory=list)
    note: str = (
        "A price row for a feature-tracked security should have a feature identity row "
        "even when long-lookback columns are null. Missing identity rows for "
        "feature-tracked names are unexpected."
    )


def reconcile_price_feature_rows(settings: Settings, con: duckdb.DuckDBPyConnection) -> PriceFeatureRowRecon:
    """Classify each price row as having a feature row, expected-missing, or unexpected-missing."""
    create_lake_views(con, settings)
    names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    out = PriceFeatureRowRecon()
    if "prices_daily" not in names:
        return out
    out.price_rows = con.execute("SELECT count(*) FROM prices_daily").fetchone()[0] or 0
    if "features_daily" in names:
        out.feature_rows = con.execute("SELECT count(*) FROM features_daily").fetchone()[0] or 0
    if "features_daily" not in names:
        # No feature lake yet: every feature-tracked price row is unexpected once tracking exists.
        rows = con.execute(
            """
            SELECT
              count(*) FILTER (WHERE COALESCE(t.feature_tracking, FALSE) = FALSE),
              count(*) FILTER (WHERE COALESCE(t.feature_tracking, FALSE) = TRUE)
            FROM prices_daily p
            LEFT JOIN tracked_securities t ON t.security_id = p.security_id AND t.enabled = TRUE
            """
        ).fetchone()
        out.price_no_feature_expected = int(rows[0] or 0)
        out.price_no_feature_unexpected = int(rows[1] or 0)
        out.expected_not_feature_tracked = out.price_no_feature_expected
        return out

    counts = con.execute(
        """
        SELECT
          count(*) FILTER (WHERE f.security_id IS NOT NULL),
          count(*) FILTER (
            WHERE f.security_id IS NULL AND COALESCE(t.feature_tracking, FALSE) = FALSE
          ),
          count(*) FILTER (
            WHERE f.security_id IS NULL AND COALESCE(t.feature_tracking, FALSE) = TRUE
          )
        FROM prices_daily p
        LEFT JOIN features_daily f
          ON f.security_id = p.security_id AND f.date = p.date
        LEFT JOIN tracked_securities t
          ON t.security_id = p.security_id AND t.enabled = TRUE
        """
    ).fetchone()
    out.price_with_feature = int(counts[0] or 0)
    out.price_no_feature_expected = int(counts[1] or 0)
    out.price_no_feature_unexpected = int(counts[2] or 0)
    out.expected_not_feature_tracked = out.price_no_feature_expected
    examples = con.execute(
        """
        SELECT p.security_id, p.ticker_at_time, p.date
        FROM prices_daily p
        LEFT JOIN features_daily f
          ON f.security_id = p.security_id AND f.date = p.date
        JOIN tracked_securities t
          ON t.security_id = p.security_id AND t.enabled = TRUE AND t.feature_tracking = TRUE
        WHERE f.security_id IS NULL
        ORDER BY p.date, p.ticker_at_time
        LIMIT 20
        """
    ).fetchall()
    out.unexpected_examples = [
        {"security_id": r[0], "ticker": r[1], "date": str(r[2])} for r in examples
    ]
    return out


def label_maturity_dates(settings: Settings, con: duckdb.DuckDBPyConnection) -> dict[str, str | None]:
    """Latest label row vs latest date with a non-null forward return for each horizon.

    Recent nulls are immature (horizon not yet complete), not missing rows.
    """
    create_lake_views(con, settings)
    names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    out: dict[str, str | None] = {
        "latest_label_row_date": None,
        "latest_mature_1d": None,
        "latest_mature_5d": None,
        "latest_mature_10d": None,
        "latest_mature_20d": None,
    }
    if "labels_forward_returns" not in names:
        return out
    latest = con.execute("SELECT max(date) FROM labels_forward_returns").fetchone()
    out["latest_label_row_date"] = str(latest[0]) if latest and latest[0] else None
    for horizon in (1, 5, 10, 20):
        col = f"forward_return_{horizon}d"
        try:
            row = con.execute(
                f"SELECT max(date) FROM labels_forward_returns WHERE {col} IS NOT NULL"
            ).fetchone()
        except Exception:  # noqa: BLE001
            row = None
        out[f"latest_mature_{horizon}d"] = str(row[0]) if row and row[0] else None
    return out

