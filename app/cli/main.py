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
    symbols: str | None = typer.Option(None, "--symbols", help="Comma-separated canonical tickers."),
    start: str = typer.Option(..., "--start", help="Start date YYYY-MM-DD."),
    end: str | None = typer.Option(None, "--end", help="End date YYYY-MM-DD (default: latest available)."),
    batch_size: int | None = typer.Option(None, "--batch-size", help="Symbols per batch (default from settings)."),
    resume: bool = typer.Option(False, "--resume", help="Resume from the last checkpoint for this exact job."),
    tracked: bool = typer.Option(False, "--tracked", help="Target the tracked price universe."),
    all_active: bool = typer.Option(
        False, "--all-active", help="Target every active security (explicit full-universe backfill)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without fetching/writing."),
) -> None:
    """Backfill daily price history for one or many symbols, with checkpoint/resume.

    Pass --symbols, --tracked, or --all-active. Omitting all three still
    defaults to --all-active for backwards compatibility, but that path is
    never started automatically -- use --dry-run first.
    """
    settings = _bootstrap()
    from app.ingestion.price_backfill import run_price_ingestion
    from app.ingestion.research_universe import estimate_price_backfill

    if tracked and all_active:
        console.print("[red]Pass only one of --tracked or --all-active.[/red]")
        raise typer.Exit(1)
    if symbols:
        default_scope = "tracked"
    elif tracked:
        default_scope = "tracked"
    else:
        default_scope = "all_active"

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
                    default_scope=default_scope,
                )
            except ProviderNotAllowedError as exc:
                console.print(f"[bold red]{exc}[/bold red]")
                raise typer.Exit(1) from exc
    except ValueError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(1) from exc

    if result.dry_run:
        est = estimate_price_backfill(result.total_symbols, _parse_date(start), _parse_date(end))
        console.print(f"[bold]\\[dry-run][/bold] would backfill {result.total_symbols} symbol(s). No requests made.")
        console.print(
            f"  Estimated requests: {est['estimated_requests']}  "
            f"Estimated rows (upper bound): {est['estimated_rows']}  "
            f"Estimated disk: {_human_bytes(est['estimated_disk_bytes'])}"
        )
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


def _print_compute_plan(title: str, result) -> None:  # noqa: ANN001
    plan = result.plan
    console.print(f"[bold]{title}[/bold]")
    if plan is None:
        console.print(f"  Symbols: {result.total_symbols}")
        console.print("  No writes performed.")
        return
    console.print(f"  Tracked securities: {plan.targets}")
    console.print(f"  Start: {plan.start}")
    console.print(f"  End: {plan.end or 'latest available'}")
    console.print(f"  Version: {plan.version}")
    if plan.lookback_sessions:
        console.print(f"  Lookback: {plan.lookback_sessions} sessions")
    if plan.lookahead_sessions:
        console.print(f"  Lookahead: {plan.lookahead_sessions} sessions")
    console.print(f"  Estimated partitions: {plan.estimated_partitions}")
    console.print("  No writes performed.")


@app.command("compute-features")
def compute_features_cmd(
    symbols: str | None = typer.Option(None, "--symbols", help="Comma-separated tickers. Omit for feature_tracking universe."),
    start: str = typer.Option(..., "--start", help="Start date YYYY-MM-DD."),
    end: str | None = typer.Option(None, "--end", help="End date YYYY-MM-DD."),
    version: str = typer.Option("v1", "--version", help="feature_version."),
    resume: bool = typer.Option(False, "--resume"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Compute features_daily for the feature-tracking universe (never the full 10k master)."""
    settings = _bootstrap()
    from app.ingestion.feature_compute import compute_features

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings, dry_run=dry_run)
        try:
            result = compute_features(
                settings,
                con,
                _parse_symbols(symbols),
                _parse_date(start),
                _parse_date(end),
                version=version,
                resume=resume,
                dry_run=dry_run,
            )
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            raise typer.Exit(1) from exc
    if result.dry_run:
        _print_compute_plan("Feature computation plan", result)
        return
    console.print(
        f"[bold green]Feature compute complete[/bold green]: {result.successful}/{result.total_symbols} "
        f"succeeded, {result.rows_written} rows written."
    )


@app.command("compute-labels")
def compute_labels_cmd(
    symbols: str | None = typer.Option(None, "--symbols", help="Comma-separated tickers. Omit for feature_tracking universe."),
    start: str = typer.Option(..., "--start", help="Start date YYYY-MM-DD."),
    end: str | None = typer.Option(None, "--end", help="End date YYYY-MM-DD."),
    version: str = typer.Option("v1", "--version", help="label_version."),
    resume: bool = typer.Option(False, "--resume"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Compute labels_forward_returns for the feature-tracking universe."""
    settings = _bootstrap()
    from app.ingestion.label_compute import compute_labels

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings, dry_run=dry_run)
        try:
            result = compute_labels(
                settings,
                con,
                _parse_symbols(symbols),
                _parse_date(start),
                _parse_date(end),
                version=version,
                resume=resume,
                dry_run=dry_run,
            )
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            raise typer.Exit(1) from exc
    if result.dry_run:
        _print_compute_plan("Label computation plan", result)
        return
    console.print(
        f"[bold green]Label compute complete[/bold green]: {result.successful}/{result.total_symbols} "
        f"succeeded, {result.rows_written} rows written."
    )


@app.command("expand-universe")
def expand_universe_cmd(
    research_scale: int | None = typer.Option(None, "--research-scale", help="Deterministic scale-test size, e.g. 100 or 500."),
    all_active: bool = typer.Option(False, "--all-active", help="Register every active security as tracked."),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Grow the tracked/research-scale universe. Never starts a full-history backfill."""
    settings = _bootstrap()
    from app.ingestion.research_universe import (
        all_active_tickers,
        apply_research_scale_universe,
        build_research_scale_universe,
        estimate_price_backfill,
    )
    from app.services.tracked_universe_service import add_tracked

    if research_scale and all_active:
        console.print("[red]Pass only one of --research-scale or --all-active.[/red]")
        raise typer.Exit(1)
    if not research_scale and not all_active:
        console.print("[red]Pass --research-scale N or --all-active.[/red]")
        raise typer.Exit(1)

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings, dry_run=dry_run)
        if research_scale:
            plan = build_research_scale_universe(con, research_scale)
            apply_research_scale_universe(con, plan, dry_run=dry_run)
            label = "[dry-run] would register" if dry_run else "Registered"
            console.print(
                f"[bold green]{label}[/bold green] {plan.name}: {plan.size} securities "
                f"(scale-test universe, not an investment universe)."
            )
            console.print(f"  Criteria: {plan.criteria}")
            console.print(f"  Sample: {', '.join(plan.tickers[:15])}{'...' if plan.size > 15 else ''}")
            return

        pairs = all_active_tickers(con)
        est = estimate_price_backfill(len(pairs), date(2000, 1, 1))
        console.print(f"Active universe: {len(pairs)} symbols")
        console.print(
            f"  Full-history backfill estimate from 2000-01-01: "
            f"{est['estimated_requests']} requests, {est['estimated_rows']} rows (upper bound), "
            f"{_human_bytes(est['estimated_disk_bytes'])} disk"
        )
        if dry_run:
            console.print("[bold]\\[dry-run][/bold] would add them to tracked_securities. No writes performed.")
            return
        add_tracked(
            con,
            [sid for _, sid in pairs],
            reason="all-active",
            price_tracking=True,
            filings_tracking=False,
            feature_tracking=True,
            notes="explicit expand-universe --all-active",
        )
        console.print(f"[bold green]Added {len(pairs)} active securities to the tracked universe.[/bold green]")
        console.print("[yellow]This does NOT start a price backfill. Run backfill-prices --tracked --dry-run first.[/yellow]")


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


@validate_app.command("features")
def validate_features_cmd(
    symbols: str | None = typer.Option(None, "--symbols"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run data-quality checks over features_daily."""
    settings = _bootstrap()
    from app.validation.runner import validate_features

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings, dry_run=dry_run)
        summary = validate_features(settings, con, _parse_symbols(symbols), dry_run)
    console.print(
        f"[bold]Validation[/bold] ({summary.dataset}, scope={summary.scope}, {summary.scope_size} securities): "
        f"{summary.rows_checked} rows checked, {summary.issues_found} issues found "
        f"([red]{summary.critical} critical[/red], [yellow]{summary.warning} warning[/yellow], {summary.info} info)"
    )


@validate_app.command("labels")
def validate_labels_cmd(
    symbols: str | None = typer.Option(None, "--symbols"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run data-quality checks over labels_forward_returns."""
    settings = _bootstrap()
    from app.validation.runner import validate_labels

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings, dry_run=dry_run)
        summary = validate_labels(settings, con, _parse_symbols(symbols), dry_run)
    console.print(
        f"[bold]Validation[/bold] ({summary.dataset}, scope={summary.scope}, {summary.scope_size} securities): "
        f"{summary.rows_checked} rows checked, {summary.issues_found} issues found "
        f"([red]{summary.critical} critical[/red], [yellow]{summary.warning} warning[/yellow], {summary.info} info)"
    )


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
    year: int | None = typer.Option(None, "--year", help="Yearly-partitioned dataset -- no --month."),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    _compact_generic("volatility", year, None, dry_run)


@compact_app.command("corporate-actions")
def compact_corporate_actions_cmd(
    year: int | None = typer.Option(None, "--year", help="Yearly-partitioned dataset -- no --month."),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    _compact_generic("corporate_actions", year, None, dry_run)


@compact_app.command("filings")
def compact_filings_cmd(
    year: int | None = typer.Option(None, "--year"),
    month: int | None = typer.Option(None, "--month"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    _compact_generic("filings", year, month, dry_run)


@compact_app.command("features")
def compact_features_cmd(
    year: int | None = typer.Option(None, "--year"),
    month: int | None = typer.Option(None, "--month"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    _compact_generic("features_daily", year, month, dry_run)


@compact_app.command("labels")
def compact_labels_cmd(
    year: int | None = typer.Option(None, "--year"),
    month: int | None = typer.Option(None, "--month"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    _compact_generic("labels_forward_returns", year, month, dry_run)


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
            r.partition.label(),
            str(r.files_before),
            str(r.files_after),
            str(r.rows_before) if r.rows_before >= 0 else "?",
            str(r.rows_after) if r.rows_after >= 0 else "?",
        )
    console.print(table)


@app.command("migrate-partitions")
def migrate_partitions_cmd(
    dataset: str | None = typer.Option(
        None, "--dataset", help="Limit to one dataset (e.g. corporate_actions, volatility). Omit for all pending."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report what would change without writing anything."),
) -> None:
    """Migrate a dataset's on-disk partitioning to match its configured granularity.

    Currently relevant for 'corporate_actions' and 'volatility', which moved
    from monthly (year=YYYY/month=MM/) to yearly (year=YYYY/) partitioning.
    Existing data is never deleted: rows are read, deduplicated, and
    rewritten into new yearly files in a staging area, validated (row count
    + date range must exactly match), and only then atomically swapped in --
    the pre-migration directory is kept as a timestamped backup for manual
    rollback.
    """
    from app.config.lake_datasets import LAKE_DATASET_SPECS, resolve_dataset_key
    from app.services.repartition_service import pending_yearly_migrations, repartition_to_yearly

    settings = _bootstrap()

    if dataset:
        targets = [resolve_dataset_key(dataset)]
    else:
        targets = pending_yearly_migrations(settings)

    if not targets:
        console.print("[green]Nothing to migrate -- every dataset's on-disk layout already matches its configured partition policy.[/green]")
        return

    table = Table(title="Partition migration" + (" (dry-run)" if dry_run else ""))
    table.add_column("Dataset")
    table.add_column("Files before", justify="right")
    table.add_column("Files after", justify="right")
    table.add_column("Rows before", justify="right")
    table.add_column("Rows after", justify="right")
    table.add_column("Date range")
    table.add_column("Time (s)", justify="right")
    table.add_column("Backup / note")

    for key in targets:
        spec = LAKE_DATASET_SPECS[key]
        if spec.granularity != "year":
            console.print(f"[yellow]Skipping '{key}': not configured for yearly partitions.[/yellow]")
            continue
        try:
            result = repartition_to_yearly(settings, key, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[bold red]Migration FAILED for '{key}':[/bold red] {exc}")
            logger.exception("migrate-partitions failed for %s", key)
            continue

        if result.skipped_reason:
            table.add_row(key, "-", "-", "-", "-", "-", "-", result.skipped_reason)
            continue

        date_range = f"{result.min_date_after}..{result.max_date_after}"
        note = "[dim](dry-run, nothing written)[/dim]" if result.dry_run else str(result.backup_dir)
        table.add_row(
            key,
            str(result.files_before),
            str(result.files_after),
            str(result.rows_before),
            str(result.rows_after),
            date_range,
            f"{result.elapsed_seconds:.2f}",
            note,
        )

    console.print(table)
    if not dry_run:
        console.print(
            "[dim]Pre-migration data is kept in the '*__pre_yearly_backup_*' directories above -- "
            "verify with 'stockdb storage-health' / 'stockdb status', then delete them manually once satisfied.[/dim]"
        )


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
    """Show Parquet partition file counts/sizes, per-dataset partition policy, and compaction candidates."""
    from app.config.lake_datasets import resolve_dataset_key
    from app.services.storage_health_service import get_dataset_policies

    settings = _bootstrap()
    dataset_key = resolve_dataset_key(dataset) if dataset else None

    policy_table = Table(title="Partition policy")
    policy_table.add_column("Dataset")
    policy_table.add_column("Granularity")
    policy_table.add_column("File-count threshold", justify="right")
    policy_table.add_column("Avg-size threshold", justify="right")
    policy_table.add_column("On disk")
    policy_table.add_column("Status")
    for pol in get_dataset_policies(settings):
        if dataset_key and pol.dataset != dataset_key:
            continue
        if not pol.implemented:
            status = "[dim]not implemented yet[/dim]"
        elif pol.on_disk_granularity is None:
            status = "[dim]no data yet[/dim]"
        elif pol.migration_pending:
            status = f"[red]MIGRATION PENDING[/red] (on disk: {pol.on_disk_granularity})"
        else:
            status = "[green]OK[/green]"
        policy_table.add_row(
            pol.dataset,
            pol.granularity,
            str(pol.file_count_threshold),
            f"{pol.avg_file_size_mb_threshold:.1f}MB",
            pol.on_disk_granularity or "-",
            status,
        )
    console.print(policy_table)

    partitions = get_storage_health(settings, [dataset_key] if dataset_key else None)
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
    for p in sorted(partitions, key=lambda x: (x.dataset, x.partition.year, x.partition.month or 0)):
        table.add_row(
            p.dataset,
            p.partition.label(),
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
            f"Run e.g. 'stockdb compact prices --year YYYY --month MM' (monthly datasets) or "
            f"'stockdb compact volatility --year YYYY' (yearly datasets) to merge them "
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

    feat_table = Table(title="Features")
    feat_table.add_column("Metric")
    feat_table.add_column("Value", justify="right")
    feat_table.add_row("Rows", str(report.features.rows))
    feat_table.add_row("Latest feature date", report.features.latest_date or "-")
    feat_table.add_row("Tracked current", str(report.features.tracked_current))
    feat_table.add_row("Tracked stale", str(report.features.tracked_stale))
    console.print(feat_table)

    lab_table = Table(title="Labels")
    lab_table.add_column("Metric")
    lab_table.add_column("Value", justify="right")
    lab_table.add_row("Rows", str(report.labels.rows))
    lab_table.add_row("Latest date", report.labels.latest_date or "-")
    lab_table.add_row("Latest mature 20d horizon", report.labels.latest_mature_horizon or "-")
    lab_table.add_row("Recent null expected", "yes" if report.labels.recent_null_expected else "no")
    console.print(lab_table)

    daily_table = Table(title="Daily")
    daily_table.add_column("Metric")
    daily_table.add_column("Value")
    daily_table.add_row("Last successful run", report.daily.last_success or "-")
    daily_table.add_row("Last failed run", report.daily.last_failure or "-")
    daily_table.add_row("Failed symbols", ", ".join(report.daily.failed_symbols) or "-")
    console.print(daily_table)

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
    """Run the daily pipeline through incremental features/labels.

    Order: universe -> prices -> VIX -> optional FRED -> filings ->
    price validation -> incremental features -> recent label recompute ->
    feature/label validation -> status. A single symbol failure does not
    abort the job; catalog/schema/critical-validation failures do.
    """
    settings = _bootstrap()
    from app.ingestion.daily_pipeline import run_daily_pipeline

    title = "Daily pipeline dry run" if dry_run else "Daily pipeline run"
    console.rule(f"[bold]{title}[/bold]")

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings, dry_run=dry_run)
        pipeline = run_daily_pipeline(settings, con, dry_run)
        report = gather_status(settings, con)

    table = Table(title=title + " summary")
    table.add_column("Step")
    table.add_column("Result")
    for step in pipeline.steps:
        table.add_row(step.name, step.result)
    console.print(table)
    console.print(
        f"[bold]Tracked price securities:[/bold] {report.universe.tracked_for_prices} | "
        f"[bold]Prices latest date:[/bold] {report.prices.latest_date} | "
        f"[bold]Open critical issues:[/bold] {report.data_quality.critical_open}"
    )
    if pipeline.aborted:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
