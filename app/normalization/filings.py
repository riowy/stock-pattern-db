"""Normalize raw SEC EDGAR filing rows into the internal schema."""

from __future__ import annotations

import polars as pl

from app.models.filing import FilingMetadata
from app.providers.base import RawFetchResult

FILINGS_SCHEMA = {
    "security_id": pl.Utf8,
    "cik": pl.Utf8,
    "accession_number": pl.Utf8,
    "form_type": pl.Utf8,
    "filing_date": pl.Date,
    "accepted_at": pl.Datetime(time_zone="UTC"),
    "report_date": pl.Date,
    "primary_document": pl.Utf8,
    "source_url": pl.Utf8,
    "retrieved_at": pl.Datetime(time_zone="UTC"),
}


def normalize_filing_rows(fetch: RawFetchResult, security_id: str | None) -> pl.DataFrame:
    if not fetch.rows:
        return pl.DataFrame(schema=FILINGS_SCHEMA)

    records = []
    for row in fetch.rows:
        try:
            filing = FilingMetadata(
                security_id=security_id,
                cik=row["cik"],
                accession_number=row["accession_number"],
                form_type=row["form_type"],
                filing_date=row["filing_date"],
                accepted_at=row.get("accepted_at"),
                report_date=row.get("report_date"),
                primary_document=row.get("primary_document"),
                source_url=row.get("source_url") or "",
                retrieved_at=fetch.retrieved_at,
            )
        except Exception:  # noqa: BLE001
            continue
        records.append(filing.model_dump())

    df = pl.DataFrame(records, schema=FILINGS_SCHEMA)
    return df.sort(["accession_number"]).unique(subset=["accession_number"], keep="last")
