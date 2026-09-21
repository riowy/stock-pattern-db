"""Run the v1 research report and write versioned parquet outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import duckdb
import polars as pl

from app.config.settings import Settings
from app.research.concentration import low_price_concentration
from app.research.config import (
    COVERAGE_FEATURES,
    REDUNDANCY_FEATURES,
    SMOKE_FEATURES,
    SPLIT_DEVELOPMENT,
    SPLIT_VALIDATION,
)
from app.research.dataset import load_research_frame
from app.research.overlay import overlay_tables
from app.research.redundancy import feature_rank_correlation
from app.research.robust import (
    cutoff_map,
    freeze_target_cutoffs,
    robust_baseline_table,
    robust_quantile_table,
    robust_spread_table,
)
from app.research.statistics import (
    attach_regimes,
    baseline_table,
    coverage_table,
    freeze_vix_cutoffs,
    ic_table,
    overlapping_spread_extreme_dates,
    overlapping_spread_table,
    quantile_table,
    sensitivity_table,
    spread_table,
)
from app.research.store import attach_provenance, research_output_dir, write_research_parquet
from app.research.summary import feature_research_summary
from app.utils.memory import peak_rss_mb, rss_mb


@dataclass
class ResearchReportResult:
    securities: int = 0
    rows: int = 0
    development_rows: int = 0
    validation_rows: int = 0
    vix_q33: float | None = None
    vix_q67: float | None = None
    output_dir: str = ""
    runtime_sec: float = 0.0
    rss_mb: float | None = None
    peak_rss_mb: float | None = None
    files: list[str] = field(default_factory=list)


def _concat(parts: list[pl.DataFrame]) -> pl.DataFrame:
    kept = [p for p in parts if p.height]
    if not kept:
        return pl.DataFrame()
    if len(kept) == 1:
        return kept[0]
    return pl.concat(kept, how="diagonal_relaxed")


def run_research_report(
    settings: Settings,
    con: duckdb.DuckDBPyConnection,
    *,
    features: list[str] | None = None,
    version: str = "v1",
) -> ResearchReportResult:
    started = perf_counter()
    feat_list = list(features or SMOKE_FEATURES)
    frame = load_research_frame(settings, con, require_valid=True)
    dev = frame.filter(pl.col("split") == SPLIT_DEVELOPMENT) if frame.height else frame
    vix_q33, vix_q67 = freeze_vix_cutoffs(dev)
    frame = attach_regimes(frame, vix_q33, vix_q67)
    winsor_cutoffs = freeze_target_cutoffs(dev, research_version=version)
    frozen = cutoff_map(winsor_cutoffs)

    out_dir = research_output_dir(settings, version)
    files: list[str] = []

    def _save(name: str, df: pl.DataFrame) -> pl.DataFrame:
        stamped = attach_provenance(df, research_version=version)
        path = out_dir / name
        if stamped.height:
            write_research_parquet(path, stamped)
            files.append(str(path))
        return stamped

    baseline = _concat([baseline_table(frame, SPLIT_DEVELOPMENT), baseline_table(frame, SPLIT_VALIDATION)])
    coverage = _concat(
        [
            coverage_table(frame, SPLIT_DEVELOPMENT, COVERAGE_FEATURES),
            coverage_table(frame, SPLIT_VALIDATION, COVERAGE_FEATURES),
        ]
    )
    quantiles = _concat(
        [
            quantile_table(frame, feat_list, SPLIT_DEVELOPMENT, cutoffs=frozen),
            quantile_table(frame, feat_list, SPLIT_VALIDATION, cutoffs=frozen),
        ]
    )
    regimes = _concat(
        [
            quantile_table(frame, feat_list, SPLIT_DEVELOPMENT, regime_kind="spy_trend", regime_col="spy_trend_regime"),
            quantile_table(frame, feat_list, SPLIT_VALIDATION, regime_kind="spy_trend", regime_col="spy_trend_regime"),
            quantile_table(frame, feat_list, SPLIT_DEVELOPMENT, regime_kind="vix", regime_col="vix_regime"),
            quantile_table(frame, feat_list, SPLIT_VALIDATION, regime_kind="vix", regime_col="vix_regime"),
        ]
    )
    spreads = spread_table(quantiles)
    overlapping = _concat(
        [
            overlapping_spread_table(frame, feat_list, SPLIT_DEVELOPMENT),
            overlapping_spread_table(frame, feat_list, SPLIT_VALIDATION),
        ]
    )
    overlapping_dates = _concat(
        [
            overlapping_spread_extreme_dates(frame, feat_list, SPLIT_DEVELOPMENT),
            overlapping_spread_extreme_dates(frame, feat_list, SPLIT_VALIDATION),
        ]
    )
    ic = _concat([ic_table(frame, feat_list, SPLIT_DEVELOPMENT), ic_table(frame, feat_list, SPLIT_VALIDATION)])
    sensitivity = _concat([sensitivity_table(frame, SPLIT_DEVELOPMENT), sensitivity_table(frame, SPLIT_VALIDATION)])

    robust_baseline = _concat(
        [
            robust_baseline_table(frame, SPLIT_DEVELOPMENT, frozen),
            robust_baseline_table(frame, SPLIT_VALIDATION, frozen),
        ]
    )
    robust_quantiles = _concat(
        [
            robust_quantile_table(frame, feat_list, SPLIT_DEVELOPMENT, frozen),
            robust_quantile_table(frame, feat_list, SPLIT_VALIDATION, frozen),
        ]
    )
    robust_spreads = robust_spread_table(robust_quantiles)

    overlay_dev = overlay_tables(frame, feat_list, SPLIT_DEVELOPMENT, frozen)
    overlay_val = overlay_tables(frame, feat_list, SPLIT_VALIDATION, frozen)
    overlay_ic = _concat([overlay_dev["ic"], overlay_val["ic"]])
    overlay_spread = _concat([overlay_dev["spread"], overlay_val["spread"]])
    overlay_overlapping = _concat([overlay_dev["overlapping"], overlay_val["overlapping"]])
    overlay_sizes = _concat([overlay_dev["sizes"], overlay_val["sizes"]])
    overlay_baseline = _concat([overlay_dev["baseline"], overlay_val["baseline"]])

    concentration = _concat(
        [
            low_price_concentration(frame, SPLIT_DEVELOPMENT),
            low_price_concentration(frame, SPLIT_VALIDATION),
        ]
    )
    rank_corr = feature_rank_correlation(frame, REDUNDANCY_FEATURES, SPLIT_DEVELOPMENT)
    summary = feature_research_summary(
        frame,
        feat_list,
        ic=ic,
        overlapping=overlapping,
        robust_spreads=robust_spreads,
    )

    _save("baseline.parquet", baseline)
    _save("feature_coverage.parquet", coverage)
    _save("quantile_results.parquet", quantiles)
    _save("spread_results.parquet", spreads)
    _save("overlapping_spread.parquet", overlapping)
    _save("overlapping_spread_dates.parquet", overlapping_dates)
    _save("ic_results.parquet", ic)
    _save("regime_results.parquet", regimes)
    _save("price_sensitivity.parquet", sensitivity)
    _save("robust_cutoffs.parquet", winsor_cutoffs)
    _save("robust_baseline.parquet", robust_baseline)
    _save("quantile_results_robust.parquet", robust_quantiles)
    _save("spread_results_robust.parquet", robust_spreads)
    _save("overlay_sizes.parquet", overlay_sizes)
    _save("overlay_baseline.parquet", overlay_baseline)
    _save("overlay_ic.parquet", overlay_ic)
    _save("overlay_spread.parquet", overlay_spread)
    _save("overlay_overlapping.parquet", overlay_overlapping)
    _save("low_price_concentration.parquet", concentration)
    _save("feature_rank_correlation.parquet", rank_corr)
    _save("feature_research_summary.parquet", summary)
    if vix_q33 is not None:
        freeze = pl.DataFrame(
            {
                "parameter": ["vix_q33", "vix_q67"],
                "value": [vix_q33, vix_q67],
                "source_split": [SPLIT_DEVELOPMENT, SPLIT_DEVELOPMENT],
            }
        )
        _save("regime_cutoffs.parquet", freeze)

    return ResearchReportResult(
        securities=int(frame["security_id"].n_unique()) if frame.height else 0,
        rows=frame.height,
        development_rows=int(dev.height) if frame.height else 0,
        validation_rows=int(frame.filter(pl.col("split") == SPLIT_VALIDATION).height) if frame.height else 0,
        vix_q33=vix_q33,
        vix_q67=vix_q67,
        output_dir=str(out_dir),
        runtime_sec=perf_counter() - started,
        rss_mb=rss_mb(),
        peak_rss_mb=peak_rss_mb(),
        files=files,
    )
