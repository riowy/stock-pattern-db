"""Persist heuristic instrument classifications into DuckDB."""

from __future__ import annotations

from datetime import UTC, datetime

import duckdb
import polars as pl

from app.config.instrument_patterns import (
    CLASS_COMMON_EQUITY,
    RESEARCH_ELIGIBLE_CLASSES,
    InstrumentClassification,
    InstrumentSignals,
    classify_instrument,
)
from app.config.lake_datasets import get_lake_dataset
from app.config.sec_instrument_signals import official_title_for_ticker
from app.services.dataset_metadata_service import set_metadata
from app.utils.logging import get_logger

logger = get_logger("instrument_classification")

INSTRUMENT_METADATA_DATASET = "instrument_classification"


def classify_security_row(
    ticker: str | None,
    exchange: str | None,
    *,
    company_name: str | None = None,
    cik: str | None = None,
    form_types: tuple[str, ...] = (),
    security_title: str | None = None,
) -> InstrumentClassification:
    official = official_title_for_ticker(ticker)
    title = security_title or (official.security_title if official else None)
    return classify_instrument(
        InstrumentSignals(
            ticker=ticker,
            exchange=exchange,
            company_name=company_name,
            cik=cik,
            security_title=title,
            form_types=form_types,
        )
    )


def _load_form_types_by_cik(con: duckdb.DuckDBPyConnection, settings) -> dict[str, tuple[str, ...]]:  # noqa: ANN001
    """Use filings lake when present. Does not fetch the full SEC universe."""
    try:
        ds = get_lake_dataset(settings.lake_dir, "filings")
    except Exception:  # noqa: BLE001
        return {}
    if not ds.has_any_files():
        return {}
    glob_path = ds.glob_pattern().replace("'", "''")
    try:
        rows = con.execute(
            f"""
            SELECT CAST(cik AS VARCHAR), list(DISTINCT form_type)
            FROM read_parquet('{glob_path}', hive_partitioning = true)
            WHERE cik IS NOT NULL
            GROUP BY 1
            """
        ).fetchall()
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for cik, forms in rows:
        if cik is None:
            continue
        padded = str(cik).zfill(10)
        if isinstance(forms, list):
            out[padded] = tuple(str(f) for f in forms if f)
        else:
            out[padded] = tuple(str(forms).split(",")) if forms else ()
    return out


def refresh_instrument_classifications(con: duckdb.DuckDBPyConnection, settings=None) -> int:  # noqa: ANN001
    """Reclassify every security. Idempotent upsert; never deletes rows."""
    rows = con.execute(
        "SELECT security_id, primary_ticker, exchange, company_name, cik FROM securities"
    ).fetchall()
    if not rows:
        set_metadata(con, INSTRUMENT_METADATA_DATASET, "instrument_class_complete", "false")
        return 0

    form_by_cik: dict[str, tuple[str, ...]] = {}
    if settings is not None:
        form_by_cik = _load_form_types_by_cik(con, settings)

    now = datetime.now(UTC)
    records = []
    for security_id, ticker, exchange, company_name, cik in rows:
        forms = ()
        if cik:
            forms = form_by_cik.get(str(cik).zfill(10), ())
        cls = classify_security_row(
            ticker,
            exchange,
            company_name=company_name,
            cik=cik,
            form_types=forms,
        )
        records.append(
            {
                "security_id": security_id,
                "ticker": cls.ticker or ticker,
                "instrument_class": cls.instrument_class,
                "classification_source": cls.classification_source,
                "classification_confidence": cls.classification_confidence,
                "exclusion_reason": cls.exclusion_reason,
                "updated_at": now,
            }
        )
    df = pl.DataFrame(records)
    con.register("_tmp_instrument_class", df)
    try:
        con.execute(
            """
            INSERT INTO instrument_classifications
                (security_id, ticker, instrument_class, classification_source,
                 classification_confidence, exclusion_reason, updated_at)
            SELECT security_id, ticker, instrument_class, classification_source,
                   classification_confidence, exclusion_reason, updated_at
            FROM _tmp_instrument_class
            ON CONFLICT (security_id) DO UPDATE SET
                ticker = excluded.ticker,
                instrument_class = excluded.instrument_class,
                classification_source = excluded.classification_source,
                classification_confidence = excluded.classification_confidence,
                exclusion_reason = excluded.exclusion_reason,
                updated_at = excluded.updated_at
            """
        )
    finally:
        con.unregister("_tmp_instrument_class")

    set_metadata(con, INSTRUMENT_METADATA_DATASET, "instrument_class_complete", "false")
    set_metadata(
        con,
        INSTRUMENT_METADATA_DATASET,
        "classification_method",
        "sec_security_title+sec_registrant_type+ticker_suffix_heuristic",
    )
    logger.info("Refreshed instrument classifications for %d securities", len(records))
    return len(records)


def get_classification_map(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    rows = con.execute(
        "SELECT security_id, instrument_class FROM instrument_classifications"
    ).fetchall()
    return {sid: cls for sid, cls in rows}


def is_research_common_equity_class(instrument_class: str) -> bool:
    return instrument_class in RESEARCH_ELIGIBLE_CLASSES


def common_equity_security_ids(con: duckdb.DuckDBPyConnection) -> list[str]:
    rows = con.execute(
        """
        SELECT security_id FROM instrument_classifications
        WHERE instrument_class = ?
        ORDER BY security_id
        """,
        [CLASS_COMMON_EQUITY],
    ).fetchall()
    return [r[0] for r in rows]
