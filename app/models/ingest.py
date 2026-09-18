"""Job-provenance / data-quality bookkeeping models.

These mirror the DuckDB tables in ``app/db/schema.py`` one-to-one and are
used by ``app/services/manifest_service.py`` to write/read that bookkeeping.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class IngestRun(BaseModel):
    run_id: str
    provider: str
    dataset: str
    started_at: datetime
    finished_at: datetime | None = None
    status: RunStatus = RunStatus.RUNNING
    requested_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    rows_written: int = 0
    error_message: str | None = None
    params_json: str | None = None


class SourceFile(BaseModel):
    file_id: str
    run_id: str
    provider: str
    dataset: str
    path: str
    retrieved_at: datetime
    checksum: str | None = None
    row_count: int | None = None
    min_date: str | None = None
    max_date: str | None = None


class IssueSeverity(StrEnum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class DataQualityIssue(BaseModel):
    issue_id: str
    dataset: str
    security_id: str | None = None
    date: str | None = None
    issue_type: str
    severity: IssueSeverity
    details: str | None = None
    detected_at: datetime
    resolved: bool = False
