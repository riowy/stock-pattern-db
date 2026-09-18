"""stockdb command-line interface.

This file only wires Typer commands to the ``app.ingestion`` / ``app.services``
layer -- it must not contain data-fetching, parsing, or business logic
itself.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from app.config.settings import get_settings
from app.db.connection import duckdb_connection
from app.db.schema import apply_schema, create_lake_views
from app.normalization.symbols import seed_symbol_mappings
from app.providers.base import ProviderNotAllowedError
from app.services.compact_service import compact_dataset
from app.services.status_service import gather_status
from app.utils.logging import get_logger, setup_logging

app = typer.Typer(
    name="stockdb",
    help="Personal research data infrastructure for US equity pattern research. "
    "Not a trading/recommendation system.",
    no_args_is_help=True,
)
validate_app = typer.Typer(help="Run data-quality validation against a dataset.")
compact_app = typer.Typer(help="Compact small Parquet partition files into one file per partition.")
app.add_typer(validate_app, name="validate")
app.add_typer(compact_app, name="compact")

console = Console()
logger = get_logger("cli")


def _parse_symbols(symbols: str | None) -> list[str] | None:
    if not symbols:
        return None
    return [s.strip() for s in symbols.split(",") if s.strip()]


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def _bootstrap():
    settings = get_settings()
    settings.ensure_directories()
    setup_logging(settings.log_dir, settings.log_level)
    return settings


@app.command()
def init() -> None:
    """Initialize directories, DuckDB metadata schema, and seed config data."""
    settings = _bootstrap()
    with duckdb_connection(settings) as con:
        apply_schema(con)
        n = seed_symbol_mappings(con)
    console.print("[bold green]stockdb initialized.[/bold green]")
    console.print(f"  Data root:     {settings.data_root.resolve()}")
    console.print(f"  DuckDB file:   {settings.duckdb_path.resolve()}")
    console.print(f"  Symbol seed mappings loaded: {n}")
    env_file = Path(".env")
    if not env_file.exists():
        console.print(
            "[yellow]No .env file found. Copy .env.example to .env and fill in "
            "SEC_USER_AGENT / FRED_API_KEY before running sync commands.[/yellow]"
        )


@app.command("sync-universe")
def sync_universe_cmd(dry_run: bool = typer.Option(False, "--dry-run", help="Preview without writing.")) -> None:
    """Sync the security master from SEC EDGAR + the ETF seed list."""
    settings = _bootstrap()
    from app.ingestion.universe_sync import sync_universe

    with duckdb_connection(settings) as con:
        apply_schema(con)
        try:
            result = sync_universe(settings, con, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[bold red]sync-universe failed:[/bold red] {exc}")
            raise typer.Exit(1) from exc
    console.print(
        f"[bold green]Universe sync complete[/bold green] "
        f"(securities={result.securities_seen}, identifiers={result.identifiers_seen}, "
        f"snapshots={result.snapshots_written}, dry_run={result.dry_run})"
    )


@app.command("backfill-prices")
def backfill_prices_cmd(
    symbols: str | None = typer.Option(None, "--symbols", help="Comma-separated canonical tickers. Omit for ALL active securities (explicit full backfill)."),
    start: str = typer.Option(..., "--start", help="Start date YYYY-MM-DD."),
    end: str | None = typer.Option(None, "--end", help="End date YYYY-MM-DD (default: latest available)."),
    batch_size: int | None = typer.Option(None, "--batch-size", help="Symbols per batch (default from settings)."),
    resume: bool = typer.Option(False, "--resume", help="Resume from the last checkpoint for this exact job."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without fetching/writing."),
) -> None:
    """Backfill daily price history for one or many symbols, with checkpoint/resume."""
    settings = _bootstrap()
    from app.ingestion.price_backfill import run_price_ingestion

    try:
        with duckdb_connection(settings) as con:
            apply_schema(con)
            try:
                result = run_price_ingestion(
                    settings,
                    con,
                    symbols=_parse_symbols(symbols),
                    start=_parse_date(start),
                    end=_parse_date(end),
                    batch_size=batch_size or settings.price_batch_size,
                    resume=resume,
                    dry_run=dry_run,
                )
            except ProviderNotAllowedError as exc:
                console.print(f"[bold red]{exc}[/bold red]")
                raise typer.Exit(1) from exc
    except ValueError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(1) from exc

    console.print(
        f"[bold green]Backfill complete[/bold green]: {result.successful}/{result.total_symbols} succeeded, "
        f"{result.failed} failed, {result.rows_written} rows written."
    )
    if result.failures:
        for f in result.failures:
            console.print(f"  [red]FAILED[/red] {f.symbol}: {f.error}")
        console.print(
            f"[yellow]Re-run the same command with --resume to retry only the incomplete symbols "
            f"(checkpoint: {result.checkpoint_job_key}).[/yellow]"
        )


@app.command("sync-prices")
def sync_prices_cmd(
    symbols: str | None = typer.Option(None, "--symbols", help="Comma-separated canonical tickers. Omit for all active securities."),
    lookback_days: int = typer.Option(10, "--lookback-days", help="Trailing window to re-fetch."),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Incrementally refresh recent daily prices for the active universe."""
    settings = _bootstrap()
    from app.ingestion.price_sync import sync_recent_prices

    with duckdb_connection(settings) as con:
        apply_schema(con)
        result = sync_recent_prices(settings, con, _parse_symbols(symbols), lookback_days, dry_run)
    console.print(
        f"[bold green]Price sync complete[/bold green]: {result.successful}/{result.total_symbols} succeeded, "
        f"{result.failed} failed, {result.rows_written} rows written."
    )


@app.command("sync-macro")
def sync_macro_cmd(
    series: str | None = typer.Option(None, "--series", help="Comma-separated FRED series ids. Omit for the seed list."),
    start: str | None = typer.Option(None, "--start", help="Observation start date YYYY-MM-DD."),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Sync macro series from FRED."""
    settings = _bootstrap()
    from app.ingestion.macro_sync import sync_macro

    with duckdb_connection(settings) as con:
        apply_schema(con)
        try:
            result = sync_macro(settings, con, _parse_symbols(series), _parse_date(start), dry_run)
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            raise typer.Exit(1) from exc
    console.print(
        f"[bold green]Macro sync complete[/bold green]: {result.series_synced} series ok, "
        f"{result.series_failed} failed, {result.rows_written} rows written."
    )


@app.command("sync-vix")
def sync_vix_cmd(dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Sync the official Cboe VIX historical series."""
    settings = _bootstrap()
    from app.ingestion.vix_sync import sync_vix

    with duckdb_connection(settings) as con:
        apply_schema(con)
        result = sync_vix(settings, con, dry_run)
    console.print(f"[bold green]VIX sync complete[/bold green]: {result.rows_written} rows written.")


@app.command("sync-sec-filings")
def sync_sec_filings_cmd(
    ciks: str | None = typer.Option(None, "--ciks", help="Comma-separated CIKs. Omit for all active securities with a known CIK."),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Sync SEC filing metadata (10-K/10-Q/8-K/20-F/6-K) for known securities."""
    settings = _bootstrap()
    from app.ingestion.filings_sync import sync_filings

    with duckdb_connection(settings) as con:
        apply_schema(con)
        try:
            result = sync_filings(settings, con, _parse_symbols(ciks), dry_run)
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            raise typer.Exit(1) from exc
    console.print(
        f"[bold green]SEC filings sync complete[/bold green]: {result.successful} CIKs ok, "
        f"{result.failed} failed, {result.rows_written} filing rows written."
    )


@validate_app.command("prices")
def validate_prices_cmd(
    symbols: str | None = typer.Option(None, "--symbols"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report findings without persisting them."),
) -> None:
    """Run data-quality checks over the daily price lake."""
    settings = _bootstrap()
    from app.validation.runner import validate_prices

    with duckdb_connection(settings) as con:
        apply_schema(con)
        summary = validate_prices(settings, con, _parse_symbols(symbols), dry_run)

    console.print(
        f"[bold]Validation[/bold] ({summary.dataset}): {summary.rows_checked} rows checked, "
        f"{summary.issues_found} issues found "
        f"([red]{summary.critical} critical[/red], [yellow]{summary.warning} warning[/yellow], {summary.info} info)"
    )
    if summary.by_type:
        table = Table(title="Issues by type")
        table.add_column("Issue type")
        table.add_column("Count", justify="right")
        for k, v in sorted(summary.by_type.items(), key=lambda kv: -kv[1]):
            table.add_row(k, str(v))
        console.print(table)


@compact_app.command("prices")
def compact_prices_cmd(
    year: int | None = typer.Option(None, "--year"),
    month: int | None = typer.Option(None, "--month"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Merge small partition files in data/lake/prices_daily into one file per partition."""
    _compact_generic("prices", year, month, dry_run)


@compact_app.command("macro")
def compact_macro_cmd(
    year: int | None = typer.Option(None, "--year"),
    month: int | None = typer.Option(None, "--month"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    _compact_generic("macro", year, month, dry_run)


@compact_app.command("volatility")
def compact_volatility_cmd(
    year: int | None = typer.Option(None, "--year"),
    month: int | None = typer.Option(None, "--month"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    _compact_generic("volatility", year, month, dry_run)


@compact_app.command("filings")
def compact_filings_cmd(
    year: int | None = typer.Option(None, "--year"),
    month: int | None = typer.Option(None, "--month"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    _compact_generic("filings", year, month, dry_run)


def _compact_generic(name: str, year: int | None, month: int | None, dry_run: bool) -> None:
    settings = _bootstrap()
    results = compact_dataset(settings.lake_dir, name, year, month, dry_run)
    if not results:
        console.print("[yellow]Nothing to compact.[/yellow]")
        return
    table = Table(title=f"Compaction results: {name}")
    table.add_column("Partition")
    table.add_column("Files before", justify="right")
    table.add_column("Files after", justify="right")
    table.add_column("Rows before", justify="right")
    table.add_column("Rows after", justify="right")
    for r in results:
        table.add_row(
            f"{r.partition[0]:04d}-{r.partition[1]:02d}",
            str(r.files_before),
            str(r.files_after),
            str(r.rows_before) if r.rows_before >= 0 else "?",
            str(r.rows_after) if r.rows_after >= 0 else "?",
        )
    console.print(table)


@app.command()
def status() -> None:
    """Show database/lake/job status."""
    settings = _bootstrap()
    with duckdb_connection(settings, read_only=False) as con:
        apply_schema(con)
        report = gather_status(settings, con)

    universe_table = Table(title="Universe")
    universe_table.add_column("Metric")
    universe_table.add_column("Value", justify="right")
    universe_table.add_row("Total securities", str(report.universe.total_securities))
    universe_table.add_row("Active securities", str(report.universe.active_securities))
    console.print(universe_table)

    prices_table = Table(title="Prices (daily)")
    prices_table.add_column("Metric")
    prices_table.add_column("Value", justify="right")
    prices_table.add_row("Row count (estimate)", str(report.prices.total_rows_estimate))
    prices_table.add_row("Earliest date", report.prices.earliest_date or "-")
    prices_table.add_row("Latest date", report.prices.latest_date or "-")
    prices_table.add_row("Securities with recent data (5d)", str(report.prices.securities_with_recent_data))
    prices_table.add_row("Securities missing recent data", str(report.prices.securities_missing_recent_data))
    console.print(prices_table)

    macro_table = Table(title="Macro (FRED)")
    macro_table.add_column("Metric")
    macro_table.add_column("Value", justify="right")
    macro_table.add_row("Series tracked", str(report.macro.series_count))
    macro_table.add_row("Last update", report.macro.last_update or "-")
    console.print(macro_table)

    vix_table = Table(title="VIX")
    vix_table.add_column("Metric")
    vix_table.add_column("Value", justify="right")
    vix_table.add_row("Rows", str(report.vix.rows))
    vix_table.add_row("Latest date", report.vix.latest_date or "-")
    console.print(vix_table)

    sec_table = Table(title="SEC Filings")
    sec_table.add_column("Metric")
    sec_table.add_column("Value", justify="right")
    sec_table.add_row("Filings tracked", str(report.sec.filings_tracked))
    sec_table.add_row("Last retrieved at", report.sec.last_filing_retrieved_at or "-")
    console.print(sec_table)

    disk_table = Table(title="Disk usage")
    disk_table.add_column("Location")
    disk_table.add_column("Size", justify="right")
    disk_table.add_row("raw/", _human_bytes(report.disk.raw_bytes))
    disk_table.add_row("lake/", _human_bytes(report.disk.lake_bytes))
    disk_table.add_row("state/", _human_bytes(report.disk.state_bytes))
    console.print(disk_table)

    jobs_table = Table(title="Jobs")
    jobs_table.add_column("Metric")
    jobs_table.add_column("Value")
    jobs_table.add_row(
        "Last success",
        f"{report.jobs.last_success['dataset']} @ {report.jobs.last_success['at']}" if report.jobs.last_success else "-",
    )
    jobs_table.add_row(
        "Last failure",
        f"{report.jobs.last_failure['dataset']} @ {report.jobs.last_failure['at']}: {report.jobs.last_failure['error']}"
        if report.jobs.last_failure
        else "-",
    )
    jobs_table.add_row(
        "Active checkpoints",
        ", ".join(f"{c['job_key']} ({c['completed']} done)" for c in report.jobs.active_checkpoints) or "-",
    )
    console.print(jobs_table)

    dq_table = Table(title="Data quality (open issues)")
    dq_table.add_column("Severity")
    dq_table.add_column("Count", justify="right")
    dq_table.add_row("critical", str(report.data_quality.critical_open))
    dq_table.add_row("warning", str(report.data_quality.warning_open))
    console.print(dq_table)


def _human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


@app.command("run-daily")
def run_daily_cmd(dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Run the full daily pipeline: universe -> prices -> vix -> macro -> filings -> validate -> report."""
    settings = _bootstrap()
    from app.ingestion.filings_sync import sync_filings
    from app.ingestion.macro_sync import sync_macro
    from app.ingestion.price_sync import sync_recent_prices
    from app.ingestion.universe_sync import sync_universe
    from app.ingestion.vix_sync import sync_vix
    from app.validation.runner import validate_prices

    steps: list[tuple[str, str]] = []

    with duckdb_connection(settings) as con:
        apply_schema(con)

        console.rule("1/7 Universe sync")
        try:
            r = sync_universe(settings, con, dry_run)
            steps.append(("universe", f"ok ({r.securities_seen} securities)"))
        except Exception as exc:  # noqa: BLE001
            steps.append(("universe", f"FAILED: {exc}"))
            logger.exception("run-daily: universe sync failed")

        console.rule("2/7 Prices sync")
        try:
            r = sync_recent_prices(settings, con, None, 10, dry_run)
            steps.append(("prices", f"ok ({r.successful}/{r.total_symbols}, {r.rows_written} rows)"))
        except Exception as exc:  # noqa: BLE001
            steps.append(("prices", f"FAILED: {exc}"))
            logger.exception("run-daily: price sync failed")

        console.rule("3/7 VIX sync")
        try:
            r = sync_vix(settings, con, dry_run)
            steps.append(("vix", f"ok ({r.rows_written} rows)"))
        except Exception as exc:  # noqa: BLE001
            steps.append(("vix", f"FAILED: {exc}"))
            logger.exception("run-daily: vix sync failed")

        console.rule("4/7 Macro sync (FRED)")
        try:
            r = sync_macro(settings, con, None, None, dry_run)
            steps.append(("macro", f"ok ({r.series_synced} series, {r.rows_written} rows)"))
        except Exception as exc:  # noqa: BLE001
            steps.append(("macro", f"skipped/FAILED: {exc}"))
            logger.warning("run-daily: macro sync skipped/failed: %s", exc)

        console.rule("5/7 SEC filings sync")
        try:
            r = sync_filings(settings, con, None, dry_run)
            steps.append(("filings", f"ok ({r.successful} CIKs, {r.rows_written} rows)"))
        except Exception as exc:  # noqa: BLE001
            steps.append(("filings", f"skipped/FAILED: {exc}"))
            logger.warning("run-daily: filings sync skipped/failed: %s", exc)

        console.rule("6/7 Validation")
        try:
            summary = validate_prices(settings, con, None, dry_run)
            steps.append(("validation", f"{summary.issues_found} issues ({summary.critical} critical)"))
        except Exception as exc:  # noqa: BLE001
            steps.append(("validation", f"FAILED: {exc}"))
            logger.exception("run-daily: validation failed")

        console.rule("7/7 Status report")
        report = gather_status(settings, con)

    table = Table(title="run-daily summary")
    table.add_column("Step")
    table.add_column("Result")
    for name, result in steps:
        table.add_row(name, result)
    console.print(table)
    console.print(
        f"[bold]Universe:[/bold] {report.universe.active_securities} active | "
        f"[bold]Prices latest date:[/bold] {report.prices.latest_date} | "
        f"[bold]Open critical issues:[/bold] {report.data_quality.critical_open}"
    )


if __name__ == "__main__":
    app()
