"""On-demand indicator engine. Memory-only; never writes lake/catalog/research files."""

from __future__ import annotations

from datetime import date

import polars as pl

from app.features.adjustment import add_adjusted_prices
from app.indicators.candles import apply_candles
from app.indicators.geometry import apply_geometry
from app.indicators.ichimoku import apply_ichimoku
from app.indicators.momentum import apply_momentum
from app.indicators.persist import assert_no_indicator_persist
from app.indicators.registry import all_specs, specs_for
from app.indicators.support_resistance import apply_support_resistance
from app.indicators.trend import apply_trend
from app.indicators.volatility import apply_volatility
from app.indicators.volume import apply_volume

APPLY = {
    "trend": apply_trend,
    "momentum": apply_momentum,
    "volatility": apply_volatility,
    "volume": apply_volume,
    "ichimoku": apply_ichimoku,
    "breakout": apply_support_resistance,
    "candles": apply_candles,
}

# geometry is extra trend/momentum; always after volatility so hist_vol exists
GROUP_ORDER = ("trend", "momentum", "volatility", "volume", "ichimoku", "breakout", "candles")


class IndicatorEngine:
    """Read-only calculator. Returns a Polars DataFrame in memory."""

    def compute(
        self,
        prices: pl.DataFrame,
        *,
        groups: list[str] | None = None,
        indicator_ids: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
        include_geometry: bool = True,
        lite: bool = False,
    ) -> pl.DataFrame:
        assert_no_indicator_persist()
        if prices.height == 0:
            return prices
        needed = set(groups or [])
        if indicator_ids:
            for spec in specs_for():
                if spec.indicator_id in indicator_ids or any(c in indicator_ids for c in spec.output_columns):
                    needed.add(spec.category)
        if not needed and not indicator_ids:
            needed = set(GROUP_ORDER)
        df = add_adjusted_prices(prices)
        df = df.sort(["security_id", "date"])
        for group in GROUP_ORDER:
            if group not in needed:
                continue
            fn = APPLY[group]
            if lite and group == "trend":
                from app.indicators.trend import apply_trend_lite

                df = apply_trend_lite(df)
            else:
                df = fn(df)
        if include_geometry and not lite and (not needed or "trend" in needed or "momentum" in needed or "volatility" in needed):
            df = apply_geometry(df)
        if start is not None:
            df = df.filter(pl.col("date") >= start)
        if end is not None:
            df = df.filter(pl.col("date") <= end)
        return df

    def available(self) -> list[str]:
        return [s.indicator_id for s in all_specs()]
