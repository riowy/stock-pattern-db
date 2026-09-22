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
from app.db.connection import analytics_connection, duckdb_connection
from app.db.migrations import run_migrations
from app.db.schema import apply_schema
from app.normalization.symbols import seed_symbol_mappings
from app.providers.base import ProviderNotAllowedError
from app.services.compact_service import compact_dataset
from app.services.status_service import gather_status
from app.services.storage_health_service import get_storage_health
from app.utils.logging import get_logger, setup_logging
from app.utils.time_utils import format_session_date

app = typer.Typer(
    name="stockdb",
    help="Personal research data infrastructure for US equity pattern research. "
    "Not a trading/recommendation system.",
    no_args_is_help=True,
)
validate_app = typer.Typer(help="Run data-quality validation against a dataset.")
compact_app = typer.Typer(help="Compact small Parquet partition files into one file per partition.")
universe_app = typer.Typer(help="Manage the tracked (operational) universe -- distinct from the full security master.")
audit_app = typer.Typer(help="Audit data quality without mutating raw prices.")
research_app = typer.Typer(help="Univariate feature research. Not recommendations or trading.")
indicators_app = typer.Typer(help="On-demand technical indicators. Memory-only; never persisted.")
mine_app = typer.Typer(help="Ephemeral pattern mining. Analysis discovers; validation only evaluates. Not trading.")
registry_app = typer.Typer(
    help="Pattern research registry (generators, patterns, metrics, local dashboard). "
    "Persistence OFF by default; no live mining."
)
app.add_typer(validate_app, name="validate")
app.add_typer(compact_app, name="compact")
app.add_typer(universe_app, name="universe")
app.add_typer(audit_app, name="audit")
app.add_typer(research_app, name="research")
app.add_typer(indicators_app, name="indicators")
app.add_typer(mine_app, name="mine")
app.add_typer(registry_app, name="registry")

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
    universe: str | None = typer.Option(
        None, "--universe", help="Named universe membership (e.g. research-common-equity-500)."
    ),
    workers: int | None = typer.Option(None, "--workers", help="Parallel fetch workers (default: MAX_WORKERS)."),
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
    if universe and (tracked or all_active or symbols):
        console.print("[red]--universe cannot be combined with --symbols/--tracked/--all-active.[/red]")
        raise typer.Exit(1)
    if symbols:
        default_scope = "tracked"
        symbol_list = _parse_symbols(symbols)
    elif tracked:
        default_scope = "tracked"
        symbol_list = None
    elif universe:
        default_scope = "tracked"
        symbol_list = None
    else:
        default_scope = "all_active"
        symbol_list = None

    try:
        with duckdb_connection(settings) as con:
            _prepare_db(con, settings, dry_run=dry_run)
            if universe:
                from app.services.universe_membership_service import membership_security_ids

                sids = membership_security_ids(con, universe)
                if not sids:
                    console.print(f"[red]Universe '{universe}' has no members. Create it with expand-universe first.[/red]")
                    raise typer.Exit(1)
                placeholders = ", ".join("?" for _ in sids)
                rows = con.execute(
                    f"SELECT primary_ticker FROM securities WHERE security_id IN ({placeholders}) "
                    f"AND primary_ticker IS NOT NULL ORDER BY primary_ticker",
                    sids,
                ).fetchall()
                symbol_list = [r[0] for r in rows]
            try:
                result = run_price_ingestion(
                    settings,
                    con,
                    symbols=symbol_list,
                    start=_parse_date(start),
                    end=_parse_date(end),
                    batch_size=batch_size or settings.price_batch_size,
                    resume=resume,
                    dry_run=dry_run,
                    default_scope=default_scope,
                    workers=workers,
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
    lookback_days: int = typer.Option(
        10,
        "--lookback-days",
        help="Deprecated. Daily overlap uses DAILY_PRICE_LOOKBACK_SESSIONS (XNYS sessions).",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Incrementally refresh recent daily prices for the tracked price universe.

    Already-current names are not requested. Only STALE and NO_DATA targets
    hit the price provider. Overlap is DAILY_PRICE_LOOKBACK_SESSIONS.
    """
    settings = _bootstrap()
    from app.ingestion.price_sync import format_price_fast_path, sync_recent_prices

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings, dry_run=dry_run)
        result = sync_recent_prices(settings, con, _parse_symbols(symbols), lookback_days, dry_run)

    console.print(f"Price fast path: {format_price_fast_path(result)}")
    if result.tracked_targets == 0:
        console.print("[yellow]No tracked price securities -- nothing to sync. Use 'stockdb universe add'.[/yellow]")
        return
    if result.provider_fetch_skipped:
        console.print(f"[bold green]{result.message}[/bold green]")
        console.print(f"provider call count = {result.provider_fetch_count}")
        return
    if result.dry_run:
        console.print(
            f"[bold]\\[dry-run][/bold] would fetch {result.fetch_targets} of "
            f"{result.tracked_targets} tracked symbol(s). No network requests."
        )
        return
    console.print(
        f"[bold green]Price sync complete[/bold green]: {result.successful}/{result.fetch_targets} succeeded, "
        f"{result.failed} failed, {result.rows_written} rows written, "
        f"provider_fetch_count={result.provider_fetch_count}."
    )


@app.command("repair-prices")
def repair_prices_cmd(
    lookback_sessions: int | None = typer.Option(
        None,
        "--lookback-sessions",
        help="XNYS sessions to re-fetch ending at the expected latest completed session "
        "(default: PRICE_REPAIR_LOOKBACK_SESSIONS).",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan the repair window without network or writes."),
) -> None:
    """Re-fetch recent sessions for every tracked price target (weekly repair).

    Ignores CURRENT/STALE classification. Does not expand to the full security master.
    """
    settings = _bootstrap()
    from app.ingestion.price_repair import repair_prices

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings, dry_run=dry_run)
        result = repair_prices(settings, con, lookback_sessions=lookback_sessions, dry_run=dry_run)

    console.print("[bold]Weekly price repair plan[/bold]" if result.dry_run else "[bold]Weekly price repair[/bold]")
    console.print(f"Tracked targets: {result.tracked_targets}")
    console.print(f"Lookback sessions: {result.lookback_sessions}")
    console.print(f"Start session: {result.start_session.isoformat()}")
    console.print(f"End session: {result.end_session.isoformat()}")
    console.print(f"Batch size: {result.batch_size}")
    console.print(f"Workers: {result.workers}")
    if result.dry_run:
        console.print(f"Network requests planned: {result.network_requests_planned}")
        console.print("No writes performed.")
        return
    console.print(
        f"[bold green]Price repair complete[/bold green]: "
        f"{result.backfill.successful}/{result.tracked_targets} succeeded, "
        f"{result.backfill.failed} failed, {result.rows_written} rows written, "
        f"provider_fetch_count={result.provider_fetch_count}."
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
    research_scale: int | None = typer.Option(None, "--research-scale", help="Provider scale-test size, e.g. 100."),
    research_common_equity: int | None = typer.Option(
        None, "--research-common-equity", help="Research common-equity size, e.g. 100 or 500."
    ),
    all_active: bool = typer.Option(False, "--all-active", help="Register every active security as tracked."),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Grow tracked/research universes. Never starts a full-history backfill."""
    settings = _bootstrap()
    from app.ingestion.research_universe import (
        all_active_tickers,
        apply_research_common_equity_universe,
        apply_research_scale_universe,
        build_research_common_equity_universe,
        build_research_scale_universe,
        estimate_price_backfill,
    )
    from app.services.tracked_universe_service import add_tracked

    chosen = [x for x in (research_scale, research_common_equity, all_active) if x]
    if len(chosen) > 1:
        console.print("[red]Pass only one of --research-scale, --research-common-equity, or --all-active.[/red]")
        raise typer.Exit(1)
    if not chosen:
        console.print("[red]Pass --research-scale N, --research-common-equity N, or --all-active.[/red]")
        raise typer.Exit(1)

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings, dry_run=dry_run)
        if research_scale:
            plan = build_research_scale_universe(con, research_scale)
            apply_research_scale_universe(con, plan, dry_run=dry_run)
            label = "[dry-run] would register" if dry_run else "Registered"
            console.print(
                f"[bold green]{label}[/bold green] {plan.name}: {plan.size} securities "
                f"(PROVIDER_SCALE_TEST, not an investment universe)."
            )
            console.print(f"  Criteria: {plan.criteria}")
            console.print(f"  Sample: {', '.join(plan.tickers[:15])}{'...' if plan.size > 15 else ''}")
            return
        if research_common_equity:
            plan = build_research_common_equity_universe(con, research_common_equity)
            apply_research_common_equity_universe(con, plan, dry_run=dry_run)
            label = "[dry-run] would register" if dry_run else "Registered"
            console.print(
                f"[bold green]{label}[/bold green] {plan.name}: {plan.size} securities "
                f"(RESEARCH_COMMON_EQUITY; benchmarks kept separate)."
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


@audit_app.command("prices")
def audit_prices_cmd() -> None:
    """OHLC quality split: research common equity vs non-common instruments."""
    settings = _bootstrap()
    from app.services.price_audit_service import audit_prices

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings)
        report = audit_prices(settings, con)
    console.print("[bold]Price quality audit[/bold]")
    console.print(f"  Rows checked: {report.rows_checked}")
    console.print("  Research common equities")
    console.print(f"    critical: {report.research_critical}")
    console.print(f"    warning: {report.research_warning}")
    if report.research_true_critical_tickers:
        console.print(f"    unexplained tickers: {', '.join(report.research_true_critical_tickers)}")
    console.print("  Non-common instruments")
    console.print(f"    critical: {report.non_common_critical}")
    console.print(f"    warning: {report.non_common_warning}")
    console.print(f"  Rounding-only: {report.rounding_only}")
    console.print(f"  True OHLC violation: {report.true_ohlc_violations}")
    console.print(f"  All-provider critical: {report.all_provider_critical}")


@audit_app.command("extreme-labels")
def audit_extreme_labels_cmd() -> None:
    """Classify EXTREME_LABEL warnings; does not raise the |x|>2 threshold."""
    settings = _bootstrap()
    from app.services.extreme_label_audit_service import audit_extreme_labels

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings)
        report = audit_extreme_labels(settings, con)
    console.print(f"[bold]Extreme-label audit[/bold] total={report.total}")
    console.print(f"  Causes: {report.by_cause}")
    console.print(f"  Instrument class: {report.by_instrument_class}")
    console.print(f"  Horizon: {report.by_horizon}")
    console.print(f"  Near corporate action: {report.near_corporate_action}")
    console.print(f"  adj_close jump: {report.adj_close_jump}")
    console.print(f"  RESEARCH_COMMON_EQUITY unexplained (C/D/E): {report.research_unexplained}")
    table = Table(title="Top tickers")
    table.add_column("Ticker")
    table.add_column("Count", justify="right")
    for ticker, n in report.by_ticker[:20]:
        table.add_row(str(ticker), str(n))
    console.print(table)
    pos = Table(title="Largest positive 30")
    pos.add_column("Ticker")
    pos.add_column("Date")
    pos.add_column("Horizon")
    pos.add_column("Value")
    pos.add_column("Cause")
    for row in report.largest_positive:
        pos.add_row(str(row["ticker"]), str(row["date"]), str(row["horizon"]), f"{row['value']:.4f}", row["cause"])
    console.print(pos)
    neg = Table(title="Largest negative 30")
    neg.add_column("Ticker")
    neg.add_column("Date")
    neg.add_column("Horizon")
    neg.add_column("Value")
    neg.add_column("Cause")
    for row in report.largest_negative:
        neg.add_row(str(row["ticker"]), str(row["date"]), str(row["horizon"]), f"{row['value']:.4f}", row["cause"])
    console.print(neg)


@research_app.command("reconcile")
def research_reconcile_cmd() -> None:
    """Explain prices/features/labels vs tracked/universe security counts."""
    settings = _bootstrap()
    from app.research.dataset import (
        daily_collection_scope,
        label_maturity_dates,
        reconcile_datasets,
        reconcile_price_feature_rows,
    )

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings)
        report = reconcile_datasets(settings, con)
        scope = daily_collection_scope(con)
        pf = reconcile_price_feature_rows(settings, con)
        maturity = label_maturity_dates(settings, con)
    console.print("[bold]Dataset reconciliation[/bold]")
    console.print(f"  prices {report.price_rows} rows / {report.price_securities} securities")
    console.print(f"  features {report.feature_rows} rows / {report.feature_securities} securities")
    console.print(f"  labels {report.label_rows} rows / {report.label_securities} securities")
    console.print(
        f"  ALL_THREE={report.all_three} PRICE_ONLY={report.price_only} "
        f"FEATURE_ONLY={report.feature_only} LABEL_ONLY={report.label_only} "
        f"row_mismatches={report.row_count_mismatches}"
    )
    console.print(
        f"  PRICE_WITH_FEATURE={pf.price_with_feature} "
        f"PRICE_NO_FEATURE_EXPECTED={pf.price_no_feature_expected} "
        f"PRICE_NO_FEATURE_UNEXPECTED={pf.price_no_feature_unexpected}"
    )
    console.print(f"  {pf.note}")
    console.print(
        f"  Research common equities: {scope.get('research_common_equity_500', 0)}"
    )
    console.print(f"  Provider scale-test extras: {scope.get('provider_scale_test_extras', 0)}")
    console.print(f"  Benchmarks: {scope.get('benchmarks', 0)}")
    console.print(f"  Total tracked price targets: {scope.get('total_tracked_price_targets', 0)}")
    console.print(
        f"  Known securities (not daily target): {scope.get('known_securities', 0)}"
    )
    console.print(
        f"  labels maturity: row={maturity.get('latest_label_row_date')} "
        f"1d={maturity.get('latest_mature_1d')} 5d={maturity.get('latest_mature_5d')} "
        f"10d={maturity.get('latest_mature_10d')} 20d={maturity.get('latest_mature_20d')}"
    )
    console.print(f"  cause: {report.cause}")
    if pf.unexpected_examples:
        console.print("  unexpected missing feature examples: " + str(pf.unexpected_examples[:10]))
    if report.tracked_without_price_tickers:
        console.print("  tracked without prices: " + ", ".join(report.tracked_without_price_tickers))


@research_app.command("baseline")
def research_baseline_cmd(
    split: str = typer.Option("development", "--split"),
) -> None:
    """Unconditional forward-return baseline. Not a strategy."""
    settings = _bootstrap()
    from app.research.dataset import load_research_frame
    from app.research.statistics import baseline_table

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings)
        frame = load_research_frame(settings, con)
        table = baseline_table(frame, split)
    _print_df("Baseline", table)


@research_app.command("feature-coverage")
def research_feature_coverage_cmd(
    split: str = typer.Option("development", "--split"),
) -> None:
    settings = _bootstrap()
    from app.research.dataset import load_research_frame
    from app.research.statistics import coverage_table

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings)
        frame = load_research_frame(settings, con)
        table = coverage_table(frame, split)
    _print_df("Feature coverage", table.select(
        ["feature", "non_null", "null_ratio", "high_null", "mean", "median", "p5", "p95"]
        if table.height and "mean" in table.columns
        else table.columns
    ))


@research_app.command("quantiles")
def research_quantiles_cmd(
    feature: str = typer.Option(..., "--feature"),
    split: str = typer.Option("development", "--split"),
) -> None:
    """Cross-sectional quintiles. Research statistic, not a recommendation."""
    settings = _bootstrap()
    from app.research.dataset import load_research_frame
    from app.research.statistics import quantile_table, spread_table

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings)
        frame = load_research_frame(settings, con)
        q = quantile_table(frame, [feature], split)
        s = spread_table(q)
    _print_df(f"Quantiles {feature} {split}", q)
    _print_df("Q5-Q1", s)


@research_app.command("ic")
def research_ic_cmd(
    feature: str | None = typer.Option(None, "--feature"),
    split: str = typer.Option("development", "--split"),
) -> None:
    settings = _bootstrap()
    from app.research.config import SMOKE_FEATURES
    from app.research.dataset import load_research_frame
    from app.research.statistics import ic_table

    feats = [feature] if feature else list(SMOKE_FEATURES)
    with duckdb_connection(settings) as con:
        _prepare_db(con, settings)
        frame = load_research_frame(settings, con)
        table = ic_table(frame, feats, split)
    _print_df("Rank IC", table)


@research_app.command("regimes")
def research_regimes_cmd(
    feature: str = typer.Option(..., "--feature"),
    split: str = typer.Option("development", "--split"),
) -> None:
    settings = _bootstrap()
    from app.research.dataset import filter_split, load_research_frame
    from app.research.statistics import attach_regimes, freeze_vix_cutoffs, quantile_table
    from app.research.config import SPLIT_DEVELOPMENT

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings)
        frame = load_research_frame(settings, con)
        vix_q33, vix_q67 = freeze_vix_cutoffs(filter_split(frame, SPLIT_DEVELOPMENT))
        frame = attach_regimes(frame, vix_q33, vix_q67)
        spy = quantile_table(frame, [feature], split, regime_kind="spy_trend", regime_col="spy_trend_regime")
        vix = quantile_table(frame, [feature], split, regime_kind="vix", regime_col="vix_regime")
    console.print(f"VIX cutoffs frozen from development: q33={vix_q33} q67={vix_q67}")
    _print_df("SPY trend regimes", spy)
    _print_df("VIX regimes", vix)


@research_app.command("report")
def research_report_cmd(
    version: str = typer.Option("v1", "--version"),
    features: str | None = typer.Option(None, "--features", help="Comma-separated. Default: smoke 6."),
) -> None:
    """Write data/research/{version}/*.parquet. Does not retune from validation."""
    settings = _bootstrap()
    from app.research.report import run_research_report

    feat_list = [s.strip() for s in features.split(",")] if features else None
    with duckdb_connection(settings) as con:
        _prepare_db(con, settings)
        result = run_research_report(settings, con, features=feat_list, version=version)
    console.print("[bold]Research report[/bold] (statistics only; not recommendations)")
    console.print(f"  sample securities={result.securities} rows={result.rows}")
    console.print(f"  development_rows={result.development_rows} validation_rows={result.validation_rows}")
    console.print(f"  vix_cutoffs (dev freeze) q33={result.vix_q33} q67={result.vix_q67}")
    console.print(f"  output={result.output_dir}")
    console.print(f"  runtime_sec={result.runtime_sec:.1f} rss_mb={result.rss_mb} peak_rss_mb={result.peak_rss_mb}")
    for path in result.files:
        console.print(f"  wrote {path}")
    summary_path = Path(result.output_dir) / "feature_research_summary.parquet"
    if summary_path.exists():
        import polars as pl

        _print_df(
            "Feature research summary (factual statistics; not a ranking or recommendation)",
            pl.read_parquet(summary_path),
        )


def _print_df(title: str, df) -> None:  # noqa: ANN001
    console.print(f"[bold]{title}[/bold] rows={0 if df is None else df.height}")
    if df is None or df.height == 0:
        return
    table = Table(title=title)
    cols = list(df.columns)[:14]
    for col in cols:
        table.add_column(str(col))
    for rec in df.head(40).iter_rows(named=True):
        table.add_row(*[_fmt_cell(rec.get(c)) for c in cols])
    console.print(table)


def _fmt_cell(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


@compact_app.command("prices")
def compact_prices_cmd(
    year: int | None = typer.Option(None, "--year"),
    month: int | None = typer.Option(None, "--month"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Compare logical rows/securities/dates/duplicates before and after compaction.",
    ),
) -> None:
    """Merge small partition files in data/lake/prices_daily. Omit year/month to compact all months."""
    if verify:
        _compact_verified("prices", year, month, dry_run)
        return
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
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Compare logical rows/securities/dates/duplicates before and after compaction.",
    ),
) -> None:
    if verify:
        _compact_verified("corporate_actions", year, None, dry_run)
        return
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
    _print_compact_table(name, results)


def _compact_verified(name: str, year: int | None, month: int | None, dry_run: bool) -> None:
    from app.services.compact_service import CompactVerificationError, compact_dataset_verified

    settings = _bootstrap()
    with duckdb_connection(settings) as con:
        _prepare_db(con, settings, dry_run=False)
        try:
            verified = compact_dataset_verified(settings, name, year, month, dry_run=dry_run, con=con)
        except CompactVerificationError as exc:
            console.print(f"[bold red]Compaction verification FAILED:[/bold red] {exc}")
            raise typer.Exit(1) from exc
    b, a = verified.before, verified.after
    console.print(
        f"[bold]Logical snapshot[/bold] rows {b.rows}->{a.rows}, securities {b.securities}->{a.securities}, "
        f"dates {b.min_date}..{b.max_date} -> {a.min_date}..{a.max_date}, "
        f"dups {b.duplicates}->{a.duplicates}, files {b.files}->{a.files}"
    )
    if not verified.partitions:
        console.print("[yellow]Nothing to compact (logical snapshot unchanged).[/yellow]")
        return
    _print_compact_table(name, verified.partitions)


def _print_compact_table(name: str, results) -> None:  # noqa: ANN001
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
    universe_table.add_row("Research common equities", str(report.universe.research_common_equities))
    universe_table.add_row("Provider scale-test extras", str(report.universe.provider_scale_test_extras))
    universe_table.add_row("Benchmarks", str(report.universe.benchmarks))
    universe_table.add_row("Total tracked price targets", str(report.universe.total_tracked_price_targets))
    universe_table.add_row("Tracked securities (enabled)", str(report.universe.tracked_securities))
    universe_table.add_row("Tracked for prices", str(report.universe.tracked_for_prices))
    console.print(universe_table)

    prices_table = Table(title="Prices (daily)")
    prices_table.add_column("Metric")
    prices_table.add_column("Value", justify="right")
    prices_table.add_row("Row count (estimate)", str(report.prices.total_rows_estimate))
    prices_table.add_row("Lake distinct securities", str(report.prices.lake_distinct_securities))
    prices_table.add_row("Earliest date", format_session_date(report.prices.earliest_date))
    prices_table.add_row("Latest date", format_session_date(report.prices.latest_date))
    prices_table.add_row("Tracked securities current", str(report.prices.tracked_current))
    prices_table.add_row("Tracked securities stale", str(report.prices.tracked_stale))
    prices_table.add_row("Tracked securities with no data", str(report.prices.tracked_no_data))
    console.print(prices_table)

    feat_table = Table(title="Features")
    feat_table.add_column("Metric")
    feat_table.add_column("Value", justify="right")
    feat_table.add_row("Rows", str(report.features.rows))
    feat_table.add_row("Lake distinct securities", str(report.features.lake_distinct_securities))
    feat_table.add_row("Latest feature date", format_session_date(report.features.latest_date))
    feat_table.add_row("Tracked current", str(report.features.tracked_current))
    feat_table.add_row("Tracked stale", str(report.features.tracked_stale))
    feat_table.add_row("Tracked with no data", str(report.features.tracked_no_data))
    console.print(feat_table)

    lab_table = Table(title="Labels")
    lab_table.add_column("Metric")
    lab_table.add_column("Value", justify="right")
    lab_table.add_row("Rows", str(report.labels.rows))
    lab_table.add_row("Lake distinct securities", str(report.labels.lake_distinct_securities))
    lab_table.add_row("Latest date", format_session_date(report.labels.latest_date))
    lab_table.add_row("Latest mature 1d date", format_session_date(report.labels.latest_mature_1d))
    lab_table.add_row("Latest mature 5d date", format_session_date(report.labels.latest_mature_5d))
    lab_table.add_row("Latest mature 10d date", format_session_date(report.labels.latest_mature_10d))
    lab_table.add_row("Latest mature 20d date", format_session_date(report.labels.latest_mature_20d))
    lab_table.add_row("Recent null expected (immature, not missing)", "yes" if report.labels.recent_null_expected else "no")
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
    macro_table.add_row("FRED_API_KEY configured", "YES" if report.macro.fred_configured else "NO")
    macro_table.add_row("Series tracked", str(report.macro.series_count))
    macro_table.add_row("Last update", report.macro.last_update or "-")
    console.print(macro_table)

    vix_table = Table(title="VIX")
    vix_table.add_column("Metric")
    vix_table.add_column("Value", justify="right")
    vix_table.add_row("Rows", str(report.vix.rows))
    vix_table.add_row("Latest date", format_session_date(report.vix.latest_date))
    console.print(vix_table)

    sec_table = Table(title="SEC Filings")
    sec_table.add_column("Metric")
    sec_table.add_column("Value", justify="right")
    sec_table.add_row("SEC_USER_AGENT configured", "YES" if report.sec.user_agent_configured else "NO")
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


@app.command()
def doctor() -> None:
    """Operational environment check before scheduler registration. Does not print secrets."""
    settings = _bootstrap()
    from app.services.doctor_service import run_doctor

    with duckdb_connection(settings) as con:
        _prepare_db(con, settings)
        report = run_doctor(settings, con)

    table = Table(title="stockdb doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for check in report.checks:
        color = {"OK": "green", "WARN": "yellow", "FAIL": "red"}.get(check.status, "white")
        table.add_row(check.name, f"[{color}]{check.status}[/{color}]", check.detail)
    console.print(table)
    console.print(
        f"OK={report.ok_count} WARN={report.warn_count} FAIL={report.fail_count} | "
        f"doctor_gate={('PASS' if report.fail_count == 0 else 'FAIL')} | "
        f"scheduler_ready_from_doctor={str(report.scheduler_ready_from_doctor).lower()}"
    )
    console.print(
        "[dim]Full scheduler_ready also requires a successful dry-run, a real run-daily, "
        "and a same-session second run that is logically idempotent. This command does not "
        "register Task Scheduler.[/dim]"
    )
    secret_needles = [settings.sec_user_agent, settings.fred_api_key]
    rendered = str(table) + "".join(check.detail for check in report.checks)
    for needle in secret_needles:
        if needle and needle.strip() and needle.strip() in rendered:
            raise typer.Exit(code=2)
    if report.fail_count:
        raise typer.Exit(code=1)


def _human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


@indicators_app.command("list")
def indicators_list_cmd(group: str | None = typer.Option(None, "--group", help="Filter by category.")) -> None:
    """List registered indicators. Does not compute or write anything."""
    from app.indicators.cli import print_list

    print_list(console, category=group)


@indicators_app.command("show")
def indicators_show_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Canonical ticker, e.g. AAPL."),
    start: str = typer.Option("2025-01-01", "--start"),
    end: str | None = typer.Option(None, "--end"),
    group: str | None = typer.Option(None, "--group"),
    indicator: str | None = typer.Option(None, "--indicator"),
    show_all: bool = typer.Option(False, "--all"),
    tail: int = typer.Option(30, "--tail"),
) -> None:
    """Compute indicators in memory and print a terminal table. No files are written."""
    settings = _bootstrap()
    from app.indicators.cli import load_symbol_prices, print_show
    from app.indicators.engine import IndicatorEngine
    from app.indicators.persist import assert_no_indicator_persist
    from app.services.market_calendar import MarketCalendarService

    assert_no_indicator_persist()
    start_d = _parse_date(start)
    end_d = _parse_date(end) if end else MarketCalendarService(settings.market_calendar).expected_latest_completed_session()
    if start_d is None:
        raise typer.Exit(1)
    with analytics_connection(settings) as con:
        prices = load_symbol_prices(settings, con, symbol, start_d, end_d)
    groups = [group] if group else None
    ids = [indicator] if indicator else None
    df = IndicatorEngine().compute(prices, groups=groups, indicator_ids=ids, start=start_d, end=end_d, include_geometry=not show_all)
    print_show(console, df, group=group, indicator=indicator, show_all=show_all, tail=tail)


@mine_app.command("run")
def mine_run_cmd(
    target: str = typer.Option("forward_excess_spy_20d", "--target"),
    analysis_start: str = typer.Option("2018-01-02", "--analysis-start"),
    analysis_end: str = typer.Option("2023-12-29", "--analysis-end"),
    validation_start: str = typer.Option("2024-01-02", "--validation-start"),
    validation_end: str | None = typer.Option(None, "--validation-end"),
    test_start: str | None = typer.Option(None, "--test-start", help="Deprecated alias for --validation-start."),
    test_end: str | None = typer.Option(None, "--test-end", help="Deprecated alias for --validation-end."),
    max_rule_size: int = typer.Option(2, "--max-rule-size"),
    top: int = typer.Option(30, "--top"),
    min_rows: int = typer.Option(1000, "--min-rows"),
    min_dates: int = typer.Option(100, "--min-dates"),
    min_securities: int = typer.Option(30, "--min-securities"),
    fdr_q: float = typer.Option(0.10, "--fdr-q"),
    event_mode: str = typer.Option("state", "--event-mode", help="state or entry"),
    cooldown_sessions: int = typer.Option(0, "--cooldown-sessions"),
) -> None:
    """Discover on ANALYSIS; evaluate the frozen set on VALIDATION. No files written."""
    settings = _bootstrap()
    import polars as pl

    from app.mining.cli import print_audit_table, print_mining_table
    from app.mining.config import FUTURE_HOLDOUT_EVALUATION_ENABLED, FUTURE_HOLDOUT_START, PATTERN_FREEZE_POLICY
    from app.mining.data import load_mining_frame
    from app.mining.engine import MiningEngine, price_floor_audit, regime_audit
    from app.mining.persist import assert_no_mining_persist

    assert_no_mining_persist()
    a0, a1 = _parse_date(analysis_start), _parse_date(analysis_end)
    v0 = _parse_date(test_start or validation_start)
    v1 = _parse_date(test_end or validation_end) if (test_end or validation_end) else None
    if a0 is None or a1 is None or v0 is None:
        raise typer.Exit(1)
    if event_mode not in {"state", "entry"}:
        console.print("[red]--event-mode must be state or entry[/red]")
        raise typer.Exit(1)
    if a0 > a1 or (v1 is not None and v0 > v1) or a1 >= v0:
        console.print("[red]analysis and validation ranges must be disjoint, with validation after analysis.[/red]")
        raise typer.Exit(1)
    with analytics_connection(settings) as con:
        console.print("[dim]Loading research-common-equity frame and on-demand indicators (memory only)...[/dim]")
        frame, mature = load_mining_frame(settings, con, analysis_start=a0, validation_end=v1, target=target)
    if frame.height == 0:
        console.print("[yellow]No research-common-equity rows available.[/yellow]")
        raise typer.Exit(1)
    holdout_end = v1 or mature
    engine = MiningEngine()
    uncond = frame.filter((pl.col("date") >= a0) & (pl.col("date") <= a1) & pl.col(target).is_not_null())
    if uncond.height:
        console.print(
            f"[dim]ANALYSIS unconditional {target}: n={uncond.height} "
            f"median={float(uncond[target].median()):.4f} mean={float(uncond[target].mean()):.4f}[/dim]"
        )
    console.print(f"[dim]Discovering on ANALYSIS {a0}..{a1} mode={event_mode} cooldown={cooldown_sessions}...[/dim]")
    discovered, specs, frozen, cutoffs = engine.discover(
        frame,
        target=target,
        analysis_start=a0,
        analysis_end=a1,
        min_rows=min_rows,
        min_dates=min_dates,
        min_securities=min_securities,
        max_rule_size=max_rule_size,
        fdr_q=fdr_q,
        event_mode=event_mode,
        cooldown_sessions=cooldown_sessions,
    )
    selected = [r for r in discovered if r.selected]
    console.print(f"[dim]Evaluating {len(selected)} FDR-selected patterns on VALIDATION {v0}..{holdout_end}...[/dim]")
    evaluated = engine.evaluate_validation(
        frame,
        discovered,
        specs,
        cutoffs,
        target=target,
        validation_start=v0,
        validation_end=holdout_end,
        event_mode=event_mode,
        cooldown_sessions=cooldown_sessions,
    )
    evaluated.sort(
        key=lambda r: (
            r.fdr_q is None,
            r.fdr_q if r.fdr_q is not None else 1.0,
            -abs(r.analysis.get("vs_baseline_nw_t") or 0.0),
            -(r.analysis.get("kept") or r.analysis.get("n") or 0),
        )
    )
    console.print(
        f"Mining v1 | target={target} | analysis={a0}..{a1} | validation={v0}..{holdout_end} | "
        f"mode={event_mode} cooldown={cooldown_sessions} | rows={frame.height} | "
        f"singles+pairs={len(discovered)} | selected q<={fdr_q}: {len(selected)}"
    )
    console.print(f"[dim]FUTURE_HOLDOUT starts {FUTURE_HOLDOUT_START}; evaluation_enabled={FUTURE_HOLDOUT_EVALUATION_ENABLED}[/dim]")
    console.print(f"[dim]{PATTERN_FREEZE_POLICY}[/dim]")
    if frozen:
        console.print(f"Analysis-frozen quantiles: {list(frozen.keys())}")
    print_mining_table(console, evaluated or selected, title="VALIDATION-evaluated analysis candidates", top=top)
    if evaluated:
        for row in evaluated[: min(3, len(evaluated))]:
            floors = price_floor_audit(frame, row.pattern, specs, target, cutoffs=cutoffs, event_mode=event_mode, cooldown_sessions=cooldown_sessions)
            print_audit_table(console, floors, title=f"Price-floor audit | {row.pattern}")
            regimes = regime_audit(
                frame,
                row.pattern,
                specs,
                target,
                analysis_start=a0,
                analysis_end=a1,
                cutoffs=cutoffs,
                event_mode=event_mode,
                cooldown_sessions=cooldown_sessions,
            )
            print_audit_table(console, regimes, title=f"Regime audit (ANALYSIS-frozen VIX) | {row.pattern}")
    console.print("[dim]MINING_RESULT_PERSISTENCE_ENABLED=false. Thresholds were not retuned on VALIDATION.[/dim]")


@mine_app.command("walk-forward")
def mine_walkforward_cmd(
    target: str = typer.Option("forward_excess_spy_20d", "--target"),
    full_discover: bool = typer.Option(False, "--full-discover", help="Re-run pair discovery on each fold (slow)."),
    event_mode: str = typer.Option("state", "--event-mode"),
    cooldown_sessions: int = typer.Option(0, "--cooldown-sessions"),
    max_rule_size: int = typer.Option(2, "--max-rule-size"),
) -> None:
    """Expanding-window walk-forward. Fold cutoffs freeze on that fold only. No FUTURE_HOLDOUT eval."""
    settings = _bootstrap()
    from app.mining.cli import print_walkforward
    from app.mining.config import ANALYSIS_START, FUTURE_HOLDOUT_START, PATTERN_FREEZE_POLICY, SMOKE_PATTERNS, WALK_FOLDS
    from app.mining.data import load_mining_frame
    from app.mining.persist import assert_no_mining_persist
    from app.mining.walkforward import walk_forward

    assert_no_mining_persist()
    with analytics_connection(settings) as con:
        console.print("[dim]Loading research-common-equity frame (memory only)...[/dim]")
        frame, mature = load_mining_frame(settings, con, analysis_start=ANALYSIS_START, target=target)
    if frame.height == 0:
        raise typer.Exit(1)
    console.print(f"[dim]Walk-forward target={target} mature={mature} FUTURE_HOLDOUT={FUTURE_HOLDOUT_START}[/dim]")
    console.print(f"[dim]{PATTERN_FREEZE_POLICY}[/dim]")
    report = None
    logger = None
    try:
        if full_discover:
            from app.mining.progress import WalkForwardLogger

            logger = WalkForwardLogger(console, n_folds=len(WALK_FOLDS))
        report = walk_forward(
            frame,
            target=target,
            tracked=SMOKE_PATTERNS,
            max_rule_size=max_rule_size,
            event_mode=event_mode,
            cooldown_sessions=cooldown_sessions,
            mature=mature,
            full_discover=full_discover,
            progress=logger,
        )
    finally:
        if logger is not None:
            logger.close()
    print_walkforward(console, report)
    console.print("[dim]MINING_RESULT_PERSISTENCE_ENABLED=false.[/dim]")


@mine_app.command("inspect")
def mine_inspect_cmd(
    pattern: list[str] = typer.Option(..., "--pattern", help="Repeatable. Example: rsi14_ge_70 AND volume_ratio20_gt_2"),
    target: str = typer.Option("forward_excess_spy_20d", "--target"),
    analysis_start: str = typer.Option("2018-01-02", "--analysis-start"),
    analysis_end: str = typer.Option("2023-12-29", "--analysis-end"),
    validation_start: str = typer.Option("2024-01-02", "--validation-start"),
    validation_end: str | None = typer.Option(None, "--validation-end"),
) -> None:
    """Inspect named patterns: STATE vs ENTRY vs cooldown, parents, years, price/regime. Terminal only."""
    settings = _bootstrap()
    from app.mining.cli import print_inspect
    from app.mining.config import PATTERN_FREEZE_POLICY
    from app.mining.data import load_mining_frame
    from app.mining.inspect import inspect_pattern, prepare_frozen_states
    from app.mining.persist import assert_no_mining_persist

    assert_no_mining_persist()
    a0, a1 = _parse_date(analysis_start), _parse_date(analysis_end)
    v0 = _parse_date(validation_start)
    v1 = _parse_date(validation_end) if validation_end else None
    if a0 is None or a1 is None or v0 is None:
        raise typer.Exit(1)
    with analytics_connection(settings) as con:
        console.print("[dim]Loading research-common-equity frame (memory only)...[/dim]")
        frame, mature = load_mining_frame(settings, con, analysis_start=a0, validation_end=v1, target=target)
    if frame.height == 0:
        raise typer.Exit(1)
    console.print(f"[dim]{PATTERN_FREEZE_POLICY} mature={mature}[/dim]")
    stated, specs, frozen, cutoffs = prepare_frozen_states(frame, target=target, analysis_start=a0, analysis_end=a1)
    for pat in pattern:
        report = inspect_pattern(
            stated,
            pat,
            target=target,
            analysis_start=a0,
            analysis_end=a1,
            validation_start=v0,
            validation_end=v1 or mature,
            specs=specs,
            frozen=frozen,
            cutoffs=cutoffs,
        )
        print_inspect(console, report)
    console.print("[dim]MINING_RESULT_PERSISTENCE_ENABLED=false. Inspect does not retune thresholds.[/dim]")


@registry_app.command("generators")
def registry_generators_cmd() -> None:
    """List generator registry (in-memory empty unless persistence/fixture enabled)."""
    settings = _bootstrap()
    from app.patterns.persistence import open_research_store

    store = open_research_store(settings, force_memory=not settings.pattern_registry_persistence_enabled)
    try:
        gens = store.list_generators()
        table = Table(title="Generator registry")
        table.add_column("id")
        table.add_column("name")
        table.add_column("type")
        table.add_column("status")
        for g in gens:
            table.add_row(g["generator_id"], g["name"], g["generator_type"], g["status"])
        if not gens:
            console.print(
                "[yellow]No generators. Persistence is "
                f"{'ON' if settings.pattern_registry_persistence_enabled else 'OFF'}.[/yellow]"
            )
        console.print(table)
        console.print(
            f"[dim]PATTERN_REGISTRY_PERSISTENCE_ENABLED="
            f"{str(settings.pattern_registry_persistence_enabled).lower()}[/dim]"
        )
    finally:
        store.close()


@registry_app.command("patterns")
def registry_patterns_cmd(
    status: str | None = typer.Option(None, "--status"),
    search: str | None = typer.Option(None, "--search"),
) -> None:
    """List pattern registry."""
    settings = _bootstrap()
    from app.patterns.persistence import open_research_store

    store = open_research_store(settings, force_memory=not settings.pattern_registry_persistence_enabled)
    try:
        rows = store.list_patterns(status=status, search=search)
        table = Table(title="Pattern registry")
        table.add_column("pattern_id")
        table.add_column("status")
        table.add_column("direction")
        table.add_column("target")
        table.add_column("horizon")
        table.add_column("hyp_fp")
        for r in rows:
            table.add_row(
                r["pattern_id"],
                r["status"],
                r["direction"],
                r["target"],
                str(r["horizon"]),
                r["hypothesis_fingerprint"][:12] + "…",
            )
        if not rows:
            console.print("[yellow]No patterns in registry.[/yellow]")
        console.print(table)
    finally:
        store.close()


@registry_app.command("inspect-pattern")
def registry_inspect_pattern_cmd(pattern_id: str = typer.Argument(...)) -> None:
    """Inspect one pattern version + provenance."""
    settings = _bootstrap()
    from app.patterns.persistence import open_research_store

    store = open_research_store(settings, force_memory=not settings.pattern_registry_persistence_enabled)
    try:
        rec = store.get_pattern_version(pattern_id)
        if rec is None:
            console.print(f"[red]Unknown pattern_id={pattern_id}[/red]")
            raise typer.Exit(1)
        console.print(f"[bold]{rec.pattern_id}[/bold] v{rec.version} status={rec.status}")
        console.print(f"structural={rec.structural_fingerprint}")
        console.print(f"hypothesis={rec.hypothesis_fingerprint}")
        console.print(rec.definition.model_dump_json(indent=2))
        events = store.list_discovery_events(pattern_id=pattern_id)
        console.print(f"[dim]discovery events: {len(events)}[/dim]")
        for e in events:
            console.print(f"  {e['timestamp']} {e['generator_id']} {e['proposal_kind']}")
    finally:
        store.close()


@registry_app.command("generator-metrics")
def registry_generator_metrics_cmd(
    generator_id: str | None = typer.Option(None, "--generator-id"),
) -> None:
    """Show generator performance / overlap metrics (no opaque single score)."""
    settings = _bootstrap()
    from app.discovery.metrics import all_pairwise_overlaps, compute_generator_metrics
    from app.patterns.persistence import open_research_store

    store = open_research_store(settings, force_memory=not settings.pattern_registry_persistence_enabled)
    try:
        ids = [generator_id] if generator_id else [g["generator_id"] for g in store.list_generators()]
        if not ids:
            console.print("[yellow]No generators.[/yellow]")
            return
        for gid in ids:
            m = compute_generator_metrics(store, gid)
            console.print(f"[bold]{gid}[/bold]")
            for k, v in m.as_dict().items():
                console.print(f"  {k}: {v}")
        for o in all_pairwise_overlaps(store):
            console.print(
                f"overlap {o.generator_a}×{o.generator_b}: shared={o.shared_pattern_count} "
                f"jaccard={o.jaccard_proposed:.3f} rej∩={o.overlap_rejected} pass∩={o.overlap_passed}"
            )
    finally:
        store.close()


@registry_app.command("dashboard")
def registry_dashboard_cmd(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    demo: bool = typer.Option(
        False,
        "--demo",
        help="Load synthetic fixtures into an in-memory store (never writes production registry).",
    ),
) -> None:
    """Start the local research dashboard (127.0.0.1 by default)."""
    settings = _bootstrap()
    from app.dashboard.server import run_dashboard
    from app.dashboard.source_status import default_dashboard_source_status
    from app.patterns.fixtures import seed_fixture_store
    from app.patterns.persistence import open_research_store
    from app.patterns.store import PatternResearchStore

    source_status = default_dashboard_source_status(demo=demo, settings=settings)
    if demo:
        store = PatternResearchStore(persist=False).open()
        seed_fixture_store(store)
        console.print("[dim]Demo fixtures loaded in-memory. No production registry writes.[/dim]")
        persist_flag = False
        signal_flag = False
    else:
        store = open_research_store(settings, force_memory=not settings.pattern_registry_persistence_enabled)
        persist_flag = settings.pattern_registry_persistence_enabled
        signal_flag = settings.daily_signal_persistence_enabled
        if not persist_flag:
            console.print(
                "[yellow]PATTERN_REGISTRY_PERSISTENCE_ENABLED=false — "
                "dashboard will show empty state unless --demo is used.[/yellow]"
            )
    try:
        run_dashboard(
            store,
            host=host,
            port=port,
            persistence_enabled=persist_flag,
            signal_persistence_enabled=signal_flag,
            source_status=source_status,
        )
    finally:
        store.close()


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
        f"[bold]Prices latest date:[/bold] {format_session_date(report.prices.latest_date)} | "
        f"[bold]Open critical issues:[/bold] {report.data_quality.critical_open}"
    )
    console.print(
        f"[bold]Price network:[/bold] symbols_requested={pipeline.price_symbols_requested} "
        f"symbols_skipped_current={pipeline.price_symbols_skipped_current} "
        f"batches_requested={pipeline.price_batches_requested} "
        f"rows_received={pipeline.price_rows_received} "
        f"provider_fetch_count={pipeline.price_provider_fetch_count}"
    )
    if pipeline.aborted:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
