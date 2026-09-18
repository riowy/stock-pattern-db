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

from app.config.settings import Settings, get_settings
from app.db.connection import duckdb_connection
from app.db.migrations import run_migrations
from app.db.schema import apply_schema
from app.normalization.symbols import seed_symbol_mappings
from app.providers.base import ProviderNotAllowedError
from app.services.compact_service import compact_dataset
from app.services.status_service import gather_status
from app.services.storage_health_service import get_storage_health
from app.utils.logging import get_logger, setup_logging

app = typer.Typer(
    name="stockdb",
    help="Personal research data infrastructure for US equity pattern research. "
    "Not a trading/recommendation system.",
    no_args_is_help=True,
)
validate_app = typer.Typer(help="Run data-quality validation against a dataset.")
compact_app = typer.Typer(help="Compact small Parquet partition files into one file per partition.")
universe_app = typer.Typer(help="Manage the tracked (operational) universe -- distinct from the full security master.")
app.add_typer(validate_app, name="validate")
app.add_typer(compact_app, name="compact")
app.add_typer(universe_app, name="universe")

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


def _bootstrap() -> Settings:
    settings = get_settings()
    settings.ensure_directories()
    setup_logging(settings.log_dir, settings.log_level)
    return settings


def _prepare_db(con, settings: Settings, dry_run: bool = False) -> None:  # noqa: ANN001
    """Apply schema DDL + idempotent data migrations.

    ``apply_schema`` is pure ``CREATE TABLE IF NOT EXISTS`` DDL -- a no-op
    once tables exist, safe unconditionally. ``run_migrations`` performs
    actual (idempotent, but non-trivial) writes -- e.g. bumping
    ``dataset_metadata.updated_at`` -- so it is skipped entirely in
    --dry-run mode to keep the "no DuckDB state change" guarantee airtight
    at the byte level, not just at the row-count level. Migrations still
    run the very next time a non-dry-run command executes.
    """
    apply_schema(con)
    if not dry_run:
        run_migrations(con, settings)


@app.command()
def init() -> None:
    """Initialize directories, DuckDB metadata schema, and seed config data."""
    settings = _bootstrap()
    with duckdb_connection(settings) as con:
        _prepare_db(con, settings)
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
        _prepare_db(con, settings, dry_run=dry_run)
        try:
            result = sync_universe(settings, con, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[bold red]sync-universe failed:[/bold red] {exc}")
            raise typer.Exit(1) from exc
    label = "[dry-run] would sync" if result.dry_run else "Universe sync complete"
    console.print(
        f"[bold green]{label}[/bold green] "
        f"(securities={result.securities_seen}, identifiers={result.identifiers_seen}, "
        f"snapshots={result.snapshots_written})"
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
    """Backfill daily price history for one or many symbols, with checkpoint/resume.

    Omitting --symbols targets every *active* security in the security
    master (the deliberate 10 -> 100 -> 500 -> full universe expansion
    path) -- not the small tracked universe. Use 'stockdb universe add' to
    grow the tracked universe once you're happy with a symbol's data.
    """
    settings = _bootstrap()
    from app.ingestion.price_backfill import run_price_ingestion

    try:
        with duckdb_connection(settings) as con:
            _prepare_db(con, settings, dry_run=dry_run)
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
                    default_scope="all_active",
                )
            except ProviderNotAllowedError as exc:
                console.print(f"[bold red]{exc}[/bold red]")
                raise typer.Exit(1) from exc
    except ValueError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(1) from exc

    if result.dry_run:
        console.print(f"[bold]\\[dry-run][/bold] would backfill {result.total_symbols} symbol(s). No requests made.")
        return

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
    symbols: str | None = typer.Option(None, "--symbols", help="Comma-separated canonical tickers. Omit for the tracked price universe."),
    lookback_days: int = typer.Option(10, "--lookback-days", help="Trailing window to re-fetch."),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Incrementally refresh recent daily prices for the tracked price universe."""
    settings = _bootstrap()
    from app.ingestion.price_sync import sync_recent_prices

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings, dry_run=dry_run)
        result = sync_recent_prices(settings, con, _parse_symbols(symbols), lookback_days, dry_run)

    if result.dry_run or result.total_symbols == 0:
        console.print(
            f"[bold]\\[dry-run][/bold] would refresh {result.total_symbols} tracked symbol(s)."
            if result.dry_run
            else "[yellow]No tracked price securities -- nothing to sync. Use 'stockdb universe add'.[/yellow]"
        )
        return
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
        _prepare_db(con, settings, dry_run=dry_run)
        result = sync_macro(settings, con, _parse_symbols(series), _parse_date(start), dry_run)

    if result.status == "skipped":
        console.print(f"[yellow]SKIPPED[/yellow] macro sync -- {result.skip_reason}")
        return
    if result.dry_run:
        console.print(f"[bold]\\[dry-run][/bold] would sync {result.series_synced} FRED series. No requests made.")
        return
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
        _prepare_db(con, settings, dry_run=dry_run)
        result = sync_vix(settings, con, dry_run)

    if result.dry_run:
        console.print("[bold]\\[dry-run][/bold] would fetch the latest Cboe VIX history. No request made.")
        return
    console.print(f"[bold green]VIX sync complete[/bold green]: {result.rows_written} rows written.")


@app.command("sync-sec-filings")
def sync_sec_filings_cmd(
    ciks: str | None = typer.Option(None, "--ciks", help="Comma-separated CIKs. Omit for the tracked filings-enabled CIKs."),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Sync SEC filing metadata (10-K/10-Q/8-K/20-F/6-K) for tracked securities."""
    settings = _bootstrap()
    from app.ingestion.filings_sync import sync_filings

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings, dry_run=dry_run)
        result = sync_filings(settings, con, _parse_symbols(ciks), dry_run)

    if result.status == "skipped":
        console.print(f"[yellow]SKIPPED[/yellow] filings sync -- {result.skip_reason}")
        return
    if result.dry_run:
        console.print(f"[bold]\\[dry-run][/bold] would check filings for {result.successful} CIK(s). No requests made.")
        return
    console.print(
        f"[bold green]SEC filings sync complete[/bold green]: {result.successful} CIKs ok, "
        f"{result.failed} failed, {result.rows_written} filing rows written."
    )


@validate_app.command("prices")
def validate_prices_cmd(
    symbols: str | None = typer.Option(None, "--symbols"),
    all_universe: bool = typer.Option(
        False, "--all-universe", help="Check every active security instead of just the tracked universe."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report findings without persisting them."),
) -> None:
    """Run data-quality checks over the daily price lake (tracked universe by default)."""
    settings = _bootstrap()
    from app.validation.runner import validate_prices

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings, dry_run=dry_run)
        summary = validate_prices(settings, con, _parse_symbols(symbols), dry_run, all_universe)

    console.print(
        f"[bold]Validation[/bold] ({summary.dataset}, scope={summary.scope}, {summary.scope_size} securities): "
        f"{summary.rows_checked} rows checked, {summary.issues_found} issues found "
        f"([red]{summary.critical} critical[/red], [yellow]{summary.warning} warning[/yellow], {summary.info} info)"
    )
    if summary.resolved_stale_issues:
        console.print(f"[dim]Resolved {summary.resolved_stale_issues} stale issue(s) that no longer apply.[/dim]")
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


# --------------------------------------------------------------------------- universe
def _resolve_tickers_to_security_ids(con, tickers: list[str]) -> list[tuple[str, str]]:  # noqa: ANN001
    from app.ingestion.price_backfill import resolve_symbols

    return resolve_symbols(con, tickers)


@universe_app.command("tracked")
def universe_tracked_cmd(
    include_disabled: bool = typer.Option(False, "--include-disabled", help="Also show removed/disabled entries."),
) -> None:
    """List the tracked (operational) universe."""
    settings = _bootstrap()
    from app.services.tracked_universe_service import list_tracked

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings)
        rows = list_tracked(con, include_disabled=include_disabled)

    if not rows:
        console.print(
            "[yellow]Tracked universe is empty.[/yellow] Use 'stockdb universe add AAPL MSFT ...' to add symbols, "
            "or run 'stockdb backfill-prices' -- securities with existing price data are auto-tracked."
        )
        return

    table = Table(title=f"Tracked universe ({len(rows)})")
    table.add_column("Ticker")
    table.add_column("Company")
    table.add_column("Enabled")
    table.add_column("Price")
    table.add_column("Filings")
    table.add_column("Reason")
    table.add_column("Added at")
    for r in rows:
        table.add_row(
            r.primary_ticker or r.security_id,
            (r.company_name or "-")[:40],
            "yes" if r.enabled else "no",
            "yes" if r.price_tracking else "no",
            "yes" if r.filings_tracking else "no",
            r.tracking_reason or "-",
            str(r.added_at)[:19],
        )
    console.print(table)


@universe_app.command("add")
def universe_add_cmd(
    symbols: list[str] = typer.Argument(..., help="Tickers to add, e.g. AAPL MSFT NVDA"),
    reason: str = typer.Option("manual_add", "--reason", help="Free-text reason recorded for audit."),
    no_filings: bool = typer.Option(False, "--no-filings", help="Do not enable SEC filing tracking for these."),
) -> None:
    """Add tickers to the tracked (operational) universe."""
    settings = _bootstrap()
    from app.services.tracked_universe_service import add_tracked

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings)
        resolved = _resolve_tickers_to_security_ids(con, symbols)
        if not resolved:
            console.print("[red]None of the given tickers were found in the security master.[/red]")
            raise typer.Exit(1)
        security_ids = [sid for _, sid in resolved]
        add_tracked(con, security_ids, reason=reason, filings_tracking=not no_filings)

    console.print(f"[bold green]Added {len(resolved)} security(ies) to the tracked universe:[/bold green] "
                  f"{', '.join(t for t, _ in resolved)}")


@universe_app.command("remove")
def universe_remove_cmd(symbols: list[str] = typer.Argument(..., help="Tickers to remove, e.g. AAPL")) -> None:
    """Remove tickers from the tracked universe (soft-remove -- history is kept)."""
    settings = _bootstrap()
    from app.services.tracked_universe_service import remove_tracked

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings)
        resolved = _resolve_tickers_to_security_ids(con, symbols)
        if not resolved:
            console.print("[red]None of the given tickers were found in the security master.[/red]")
            raise typer.Exit(1)
        security_ids = [sid for _, sid in resolved]
        removed = remove_tracked(con, security_ids)

    console.print(f"[bold green]Removed {removed} security(ies) from the tracked universe.[/bold green]")


# --------------------------------------------------------------------------- status
@app.command("storage-health")
def storage_health_cmd(
    dataset: str | None = typer.Option(None, "--dataset", help="Limit to one dataset (prices, macro, volatility, filings, ...)."),
) -> None:
    """Show Parquet partition file counts/sizes and flag compaction candidates."""
    settings = _bootstrap()
    partitions = get_storage_health(settings, [dataset] if dataset else None)

    if not partitions:
        console.print("[yellow]No Parquet partitions found yet.[/yellow]")
        return

    table = Table(title="Storage health")
    table.add_column("Dataset")
    table.add_column("Partition")
    table.add_column("Files", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Rows (est.)", justify="right")
    table.add_column("Avg file size", justify="right")
    table.add_column("Newest file")
    table.add_column("Compact?")
    for p in sorted(partitions, key=lambda x: (x.dataset, x.year, x.month)):
        table.add_row(
            p.dataset,
            f"{p.year:04d}-{p.month:02d}",
            str(p.file_count),
            _human_bytes(p.total_size_bytes),
            str(p.row_count_estimate),
            _human_bytes(p.avg_file_size_bytes),
            str(p.newest_file_time)[:19] if p.newest_file_time else "-",
            "[yellow]YES[/yellow]" if p.needs_compaction else "no",
        )
    console.print(table)
    candidates = [p for p in partitions if p.needs_compaction]
    if candidates:
        console.print(
            f"[yellow]{len(candidates)} partition(s) look like good compaction candidates.[/yellow] "
            f"Run e.g. 'stockdb compact prices --year YYYY --month MM' to merge them "
            f"(compaction is never automatic in the daily pipeline)."
        )


@app.command()
def status() -> None:
    """Show database/lake/job status."""
    settings = _bootstrap()
    with duckdb_connection(settings, read_only=False) as con:
        _prepare_db(con, settings)
        report = gather_status(settings, con)

    universe_table = Table(title="Universe")
    universe_table.add_column("Metric")
    universe_table.add_column("Value", justify="right")
    universe_table.add_row("Total known securities", str(report.universe.total_known_securities))
    universe_table.add_row("Active securities", str(report.universe.active_securities))
    universe_table.add_row("Tracked securities", str(report.universe.tracked_securities))
    universe_table.add_row("Tracked for prices", str(report.universe.tracked_for_prices))
    console.print(universe_table)

    prices_table = Table(title="Prices (daily)")
    prices_table.add_column("Metric")
    prices_table.add_column("Value", justify="right")
    prices_table.add_row("Row count (estimate)", str(report.prices.total_rows_estimate))
    prices_table.add_row("Earliest date", report.prices.earliest_date or "-")
    prices_table.add_row("Latest date", report.prices.latest_date or "-")
    prices_table.add_row("Tracked securities current", str(report.prices.tracked_current))
    prices_table.add_row("Tracked securities stale", str(report.prices.tracked_stale))
    prices_table.add_row("Tracked securities with no data", str(report.prices.tracked_no_data))
    console.print(prices_table)

    macro_table = Table(title="Macro (FRED)")
    macro_table.add_column("Metric")
    macro_table.add_column("Value", justify="right")
    macro_table.add_row("FRED_API_KEY configured", "yes" if report.macro.fred_configured else "no")
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
    sec_table.add_row("Tracked CIKs", str(report.sec.tracked_ciks))
    sec_table.add_row("Filings tracked", str(report.sec.filings_tracked))
    sec_table.add_row("Last retrieved at", report.sec.last_filing_retrieved_at or "-")
    console.print(sec_table)

    storage_table = Table(title="Storage")
    storage_table.add_column("Metric")
    storage_table.add_column("Value", justify="right")
    storage_table.add_row("raw/", _human_bytes(report.storage.raw_bytes))
    storage_table.add_row("lake/", _human_bytes(report.storage.lake_bytes))
    storage_table.add_row("state/", _human_bytes(report.storage.state_bytes))
    storage_table.add_row("Compaction candidates", str(report.storage.compaction_candidate_count))
    console.print(storage_table)
    if report.storage.compaction_candidate_labels:
        console.print("  " + ", ".join(report.storage.compaction_candidate_labels))

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

    ri_table = Table(title="Research integrity")
    ri_table.add_column("Metric")
    ri_table.add_column("Value", justify="right")
    ri_table.add_row("Historical universe complete", "YES" if report.research_integrity.historical_universe_complete else "NO")
    ri_table.add_row("Survivorship-safe universe", "YES" if report.research_integrity.survivorship_safe else "NO")
    ri_table.add_row("Point-in-time security master", "YES" if report.research_integrity.point_in_time_security_master else "NO")
    ri_table.add_row("Price provider", report.research_integrity.price_provider)
    ri_table.add_row("Commercial use safe", "YES" if report.research_integrity.commercial_use_safe else "NO")
    console.print(ri_table)


def _human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


@app.command("run-daily")
def run_daily_cmd(dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Run the full daily pipeline: universe -> prices -> vix -> macro -> filings -> validate -> report.

    Prices/filings operate on the *tracked* universe, not the full security
    master -- see 'stockdb universe'. In --dry-run mode this makes NO
    network requests and NO database/checkpoint/Parquet writes; it only
    reports, from local metadata, what each step would do.
    """
    settings = _bootstrap()
    from app.ingestion.filings_sync import sync_filings
    from app.ingestion.macro_sync import sync_macro
    from app.ingestion.price_sync import sync_recent_prices
    from app.ingestion.universe_sync import sync_universe
    from app.ingestion.vix_sync import sync_vix
    from app.validation.runner import validate_prices

    steps: list[tuple[str, str]] = []
    title = "Daily pipeline dry run" if dry_run else "Daily pipeline run"
    console.rule(f"[bold]{title}[/bold]")

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings, dry_run=dry_run)

        console.rule("1/6 Universe")
        try:
            r = sync_universe(settings, con, dry_run)
            verb = "would sync" if r.dry_run else "synced"
            steps.append(("universe", f"{verb} SEC universe + ETF seed list ({r.securities_seen} securities known)"))
        except Exception as exc:  # noqa: BLE001
            steps.append(("universe", f"FAILED: {exc}"))
            logger.exception("run-daily: universe sync failed")

        console.rule("2/6 Prices (tracked universe)")
        try:
            r = sync_recent_prices(settings, con, None, 10, dry_run)
            if r.total_symbols == 0:
                steps.append(("prices", "SKIPPED - tracked universe is empty (use 'stockdb universe add')"))
            elif r.dry_run:
                steps.append(("prices", f"would check/update {r.total_symbols} tracked securities"))
            else:
                steps.append(("prices", f"ok ({r.successful}/{r.total_symbols}, {r.rows_written} rows)"))
        except Exception as exc:  # noqa: BLE001
            steps.append(("prices", f"FAILED: {exc}"))
            logger.exception("run-daily: price sync failed")

        console.rule("3/6 VIX")
        try:
            r = sync_vix(settings, con, dry_run)
            steps.append(("vix", "would fetch latest VIX" if r.dry_run else f"ok ({r.rows_written} rows)"))
        except Exception as exc:  # noqa: BLE001
            steps.append(("vix", f"FAILED: {exc}"))
            logger.exception("run-daily: vix sync failed")

        console.rule("4/6 Macro (FRED)")
        try:
            r = sync_macro(settings, con, None, None, dry_run)
            if r.status == "skipped":
                steps.append(("macro", f"SKIPPED - {r.skip_reason}"))
            elif r.dry_run:
                steps.append(("macro", f"would fetch {r.series_synced} FRED series"))
            else:
                steps.append(("macro", f"ok ({r.series_synced} series, {r.rows_written} rows)"))
        except Exception as exc:  # noqa: BLE001
            steps.append(("macro", f"FAILED: {exc}"))
            logger.exception("run-daily: macro sync failed unexpectedly")

        console.rule("5/6 SEC filings (tracked CIKs)")
        try:
            r = sync_filings(settings, con, None, dry_run)
            if r.status == "skipped":
                steps.append(("filings", f"SKIPPED - {r.skip_reason}"))
            elif r.dry_run:
                steps.append(("filings", f"would check {r.successful} tracked CIK(s)"))
            else:
                steps.append(("filings", f"ok ({r.successful} CIKs, {r.rows_written} rows)"))
        except Exception as exc:  # noqa: BLE001
            steps.append(("filings", f"FAILED: {exc}"))
            logger.exception("run-daily: filings sync failed unexpectedly")

        console.rule("6/6 Validation (tracked universe)")
        try:
            summary = validate_prices(settings, con, None, dry_run, all_universe=False)
            if summary.scope_size == 0:
                steps.append(("validation", "SKIPPED - tracked universe is empty"))
            else:
                steps.append(("validation", f"{summary.issues_found} issues ({summary.critical} critical) over {summary.scope_size} securities"))
        except Exception as exc:  # noqa: BLE001
            steps.append(("validation", f"FAILED: {exc}"))
            logger.exception("run-daily: validation failed")

        report = gather_status(settings, con)

    table = Table(title=title + " summary")
    table.add_column("Step")
    table.add_column("Result")
    for name, result in steps:
        table.add_row(name, result)
    console.print(table)
    console.print(
        f"[bold]Tracked price securities:[/bold] {report.universe.tracked_for_prices} | "
        f"[bold]Prices latest date:[/bold] {report.prices.latest_date} | "
        f"[bold]Open critical issues:[/bold] {report.data_quality.critical_open}"
    )


if __name__ == "__main__":
    app()
