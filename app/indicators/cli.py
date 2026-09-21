"""CLI helpers for on-demand indicators. Terminal output only."""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl
from rich.console import Console
from rich.table import Table

from app.config.settings import Settings
from app.db.schema import create_lake_views
from app.indicators.engine import GROUP_ORDER, IndicatorEngine
from app.indicators.registry import all_specs, specs_for
from app.ingestion.compute_common import lookback_start
from app.services.market_calendar import MarketCalendarService


def load_symbol_prices(
    settings: Settings,
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    start: date,
    end: date,
) -> pl.DataFrame:
    create_lake_views(con, settings)
    lookback = lookback_start(settings, start)
    ticker = symbol.strip().upper()
    return con.execute(
        """
        SELECT * FROM prices_daily
        WHERE date >= ? AND date <= ?
          AND (upper(ticker_at_time) = ? OR security_id IN (
              SELECT security_id FROM securities WHERE upper(primary_ticker) = ?
          ))
        ORDER BY security_id, date
        """,
        [lookback, end, ticker, ticker],
    ).pl()


def print_list(console: Console, category: str | None = None) -> None:
    import app.indicators.engine as _eng  # noqa: F401  populate registry

    table = Table(title="Indicator registry (memory-only; not persisted)")
    table.add_column("id")
    table.add_column("category")
    table.add_column("causal")
    table.add_column("mining")
    table.add_column("warmup")
    table.add_column("outputs")
    for spec in specs_for(category=category):
        table.add_row(
            spec.indicator_id,
            spec.category,
            "yes" if spec.causal_safe else "NO",
            "yes" if spec.mining_enabled else "no",
            str(spec.warmup_sessions),
            str(len(spec.output_columns)),
        )
    console.print(table)
    console.print("[dim]INDICATOR_PERSISTENCE_ENABLED=false. No Parquet/CSV/catalog writes.[/dim]")


def print_show(
    console: Console,
    df: pl.DataFrame,
    *,
    group: str | None,
    indicator: str | None,
    show_all: bool,
    tail: int,
) -> None:
    if df.height == 0:
        console.print("[yellow]No price rows for that symbol/date range.[/yellow]")
        return
    df = df.sort("date")
    if tail:
        df = df.tail(tail)
    if show_all:
        for cat in GROUP_ORDER:
            cols = _category_columns(df, cat)
            if not cols:
                continue
            _print_preview(console, df, f"{cat} summary", ["date", "ticker_at_time"] + cols[:8])
        return
    if indicator:
        spec = next((s for s in all_specs() if s.indicator_id == indicator), None)
        cols = list(spec.output_columns) if spec else [c for c in df.columns if indicator in c]
        cols = [c for c in cols if c in df.columns]
        _print_split(console, df, indicator, cols)
        return
    if group:
        cols = _category_columns(df, group)
        _print_split(console, df, group, cols)
        return
    core = [c for c in ("date", "adjusted_close", "sma_20", "sma_50", "sma_200", "rsi_14", "macd", "atr_14") if c in df.columns]
    _print_preview(console, df, "core", core)


def _category_columns(df: pl.DataFrame, category: str) -> list[str]:
    names: list[str] = []
    for spec in specs_for(category=category):
        names.extend(c for c in spec.output_columns if c in df.columns)
    # unique preserve order
    seen: set[str] = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _print_split(console: Console, df: pl.DataFrame, title: str, cols: list[str]) -> None:
    numeric, flags = [], []
    for c in cols:
        if c not in df.columns:
            continue
        dtype = df.schema[c]
        if dtype == pl.Boolean:
            flags.append(c)
        else:
            numeric.append(c)
    if numeric:
        for i in range(0, len(numeric), 5):
            chunk = numeric[i : i + 5]
            suffix = "" if i == 0 else f" ({i // 5 + 1})"
            _print_preview(console, df, f"{title}{suffix}", ["date"] + chunk)
    if flags:
        for i in range(0, len(flags), 6):
            chunk = flags[i : i + 6]
            suffix = "" if i == 0 else f" ({i // 6 + 1})"
            _print_preview(console, df, f"{title} states{suffix}", ["date"] + chunk)


def _print_preview(console: Console, df: pl.DataFrame, title: str, cols: list[str]) -> None:
    cols = [c for c in cols if c in df.columns]
    table = Table(title=title, expand=False)
    for c in cols:
        width = 12 if c == "date" else None
        table.add_column(c, overflow="fold", no_wrap=True, min_width=width)
    for row in df.select(cols).iter_rows():
        table.add_row(*[_fmt(v) for v in row])
    console.print(table)


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # noqa: PLR0124
            return ""
        return f"{value:.4f}"
    if isinstance(value, bool):
        return "Y" if value else "n"
    if hasattr(value, "isoformat") and not hasattr(value, "hour"):
        return value.isoformat()
    return str(value)
