"""SEC EDGAR filing metadata model (metadata only -- no document text/analysis)."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel


class FilingMetadata(BaseModel):
    security_id: str | None = None
    cik: str
    accession_number: str
    form_type: str
    filing_date: date
    accepted_at: datetime | None = None
    report_date: date | None = None
    primary_document: str | None = None
    source_url: str
    retrieved_at: datetime
