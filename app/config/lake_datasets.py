"""Single source of truth for every Parquet lake dataset's layout.

Previously each dataset's directory/date-column/dedup-keys/sort-keys were
hardcoded independently in half a dozen places (``create_lake_views``,
``status_service``, ``validation/runner``, each ``ingestion/*_sync.py``,
``compact_service``...), which made it easy for two call sites to disagree
about a dataset's shape. Everything now goes through
``get_lake_dataset(lake_dir, key)`` below.

Partition granularity policy (see README "Parquet partition 전략"):

* ``"month"`` (``year=YYYY/month=MM/``) -- for datasets with enough daily
  row volume that a whole year in one file would still be reasonable to
  read/append incrementally: daily prices, and the future daily
  features/labels datasets.
* ``"year"`` (``year=YYYY/``) -- for low-row-volume datasets where monthly
  partitioning produces hundreds of near-empty files for no benefit: VIX
  (1 row/day but only 1 series) and corporate actions (only a handful of
  dividend/split events per security per year).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.utils.parquet_io import Granularity, LakeDataset


@dataclass(frozen=True)
class LakeDatasetSpec:
    key: str
    dir_name: str
    date_column: str
    dedup_keys: list[str]
    sort_keys: list[str]
    granularity: Granularity
    implemented: bool = True  # False = documented policy only, no data yet
    tie_break_column: str = "retrieved_at"


LAKE_DATASET_SPECS: dict[str, LakeDatasetSpec] = {
    "prices_daily": LakeDatasetSpec(
        "prices_daily", "prices_daily", "date", ["security_id", "date"], ["security_id", "date"], "month"
    ),
    "corporate_actions": LakeDatasetSpec(
        "corporate_actions",
        "corporate_actions",
        "effective_date",
        ["security_id", "effective_date", "action_type"],
        ["security_id", "effective_date"],
        "year",
    ),
    "macro": LakeDatasetSpec(
        "macro", "macro", "date", ["series_id", "date"], ["series_id", "date"], "month"
    ),
    "volatility": LakeDatasetSpec("volatility", "volatility", "date", ["date"], ["date"], "year"),
    "filings": LakeDatasetSpec(
        "filings", "filings", "filing_date", ["accession_number"], ["security_id", "filing_date"], "month"
    ),
    "short_volume": LakeDatasetSpec(
        "short_volume", "short_volume", "date", ["security_id", "date"], ["security_id", "date"], "month"
    ),
    "features_daily": LakeDatasetSpec(
        "features_daily",
        "features_daily",
        "date",
        ["security_id", "date", "feature_version"],
        ["security_id", "date"],
        "month",
        implemented=True,
        tie_break_column="calculated_at",
    ),
    "labels_forward_returns": LakeDatasetSpec(
        "labels_forward_returns",
        "labels_forward_returns",
        "date",
        ["security_id", "date", "label_version"],
        ["security_id", "date"],
        "month",
        implemented=True,
        tie_break_column="calculated_at",
    ),
}

# Short, CLI-facing aliases for backwards compatibility (e.g. `stockdb compact prices`).
DATASET_ALIASES: dict[str, str] = {
    "prices": "prices_daily",
    "features": "features_daily",
    "labels": "labels_forward_returns",
}


def resolve_dataset_key(name: str) -> str:
    key = DATASET_ALIASES.get(name, name)
    if key not in LAKE_DATASET_SPECS:
        available = ", ".join(sorted({*LAKE_DATASET_SPECS, *DATASET_ALIASES}))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return key


def get_spec(name: str) -> LakeDatasetSpec:
    return LAKE_DATASET_SPECS[resolve_dataset_key(name)]


def get_lake_dataset(lake_dir: Path, name: str) -> LakeDataset:
    spec = get_spec(name)
    return LakeDataset(
        lake_dir / spec.dir_name,
        spec.date_column,
        spec.dedup_keys,
        spec.sort_keys,
        tie_break_column=spec.tie_break_column,
        granularity=spec.granularity,
    )


def implemented_dataset_keys() -> list[str]:
    return [k for k, spec in LAKE_DATASET_SPECS.items() if spec.implemented]
