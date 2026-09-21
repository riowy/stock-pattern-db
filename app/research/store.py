"""Persist research tables under data/research/{version}/. Never mix with the lake."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from app.config.settings import Settings
from app.research.config import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    RESEARCH_UNIVERSE_NAME,
    RESEARCH_VERSION_V1,
    VALIDATION_START,
)
from app.utils.atomic_io import atomic_write_via
from app.utils.versioning import get_calculation_code_version
from app.features.schema import FEATURE_VERSION_V1
from app.labels.schema import LABEL_VERSION_V1


def research_output_dir(settings: Settings, version: str = RESEARCH_VERSION_V1) -> Path:
    return settings.research_dir / version


def provenance_columns(
    *,
    research_version: str = RESEARCH_VERSION_V1,
    universe_name: str = RESEARCH_UNIVERSE_NAME,
    calculated_at: datetime | None = None,
) -> dict:
    return {
        "research_version": research_version,
        "feature_version": FEATURE_VERSION_V1,
        "label_version": LABEL_VERSION_V1,
        "universe_name": universe_name,
        "development_start": DEVELOPMENT_START.isoformat(),
        "development_end": DEVELOPMENT_END.isoformat(),
        "validation_start": VALIDATION_START.isoformat(),
        "validation_end": None,
        "calculated_at": calculated_at or datetime.now(UTC),
        "calculation_code_version": get_calculation_code_version(),
    }


def attach_provenance(df: pl.DataFrame, **kwargs) -> pl.DataFrame:  # noqa: ANN003
    if df.height == 0:
        return df
    meta = provenance_columns(**kwargs)
    exprs = []
    for key, value in meta.items():
        if value is None:
            exprs.append(pl.lit(None, dtype=pl.Utf8).alias(key))
        else:
            exprs.append(pl.lit(value).alias(key))
    return df.with_columns(exprs)


def write_research_parquet(path: Path, df: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _writer(tmp: Path) -> None:
        df.write_parquet(tmp, compression="zstd")

    def _validate(tmp: Path) -> None:
        check = pl.read_parquet(tmp)
        if check.height != df.height:
            raise ValueError(f"research parquet row mismatch: {check.height} != {df.height}")

    atomic_write_via(path, _writer, _validate)
