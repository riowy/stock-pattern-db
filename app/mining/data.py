"""Read-only frame assembly for mining. Nothing is written."""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

from app.config.settings import Settings
from app.db.schema import create_lake_views
from app.indicators.engine import IndicatorEngine
from app.ingestion.compute_common import load_prices, load_vix, lookback_start
from app.mining.config import cap_end_before_future_holdout
from app.research.config import RESEARCH_UNIVERSE_NAME
from app.research.dataset import load_research_frame
from app.services.market_calendar import MarketCalendarService
from app.services.universe_membership_service import membership_security_ids


def latest_mature_date(df: pl.DataFrame, target: str) -> date | None:
    if target not in df.columns or df.height == 0:
        return None
    s = df.filter(pl.col(target).is_not_null())["date"]
    if s.len() == 0:
        return None
    return s.max()


def load_mining_frame(
    settings: Settings,
    con: duckdb.DuckDBPyConnection,
    *,
    analysis_start: date,
    test_end: date | None = None,
    validation_end: date | None = None,
    target: str,
) -> tuple[pl.DataFrame, date | None]:
    """Join research features/labels with on-demand indicators. Memory only."""
    create_lake_views(con, settings)
    research = load_research_frame(settings, con, require_valid=True)
    if research.height == 0:
        return research, None
    mature = latest_mature_date(research, target)
    end = cap_end_before_future_holdout(validation_end if validation_end is not None else test_end or mature)
    ids = membership_security_ids(con, RESEARCH_UNIVERSE_NAME)
    prices = load_prices(con, settings, ids, lookback_start(settings, analysis_start), end or analysis_start)
    vix = load_vix(con, settings, lookback_start(settings, analysis_start), end or analysis_start)
    engine = IndicatorEngine()
    # lite path: skip WMA/HMA/KAMA/geometry so 500 names stay in-memory
    indicators = engine.compute(
        prices,
        groups=["trend", "momentum", "volatility", "volume", "ichimoku", "breakout", "candles"],
        start=analysis_start,
        end=end,
        include_geometry=False,
        lite=True,
    )
    keep = [c for c in indicators.columns if c not in {"open", "high", "low", "close", "adj_close", "volume", "provider", "retrieved_at", "dividend", "stock_split", "currency"}]
    # Avoid exploding duplicate feature columns: prefer indicator names, keep feature extras
    ind = indicators.select([c for c in keep if c in indicators.columns])
    overlap = [c for c in ind.columns if c in research.columns and c not in {"security_id", "date", "ticker_at_time"}]
    ind = ind.drop(overlap)
    joined = research.join(ind, on=["security_id", "date"], how="left")
    if vix.height and "date" in vix.columns:
        v = vix.rename({"close": "vix_close_ctx"}) if "close" in vix.columns else vix
        if "vix_close_ctx" in v.columns and "vix_close" not in joined.columns:
            joined = joined.join(v.select(["date", "vix_close_ctx"]), on="date", how="left")
    return joined, mature
