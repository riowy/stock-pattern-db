# stock-pattern-db

A personal research **data infrastructure** for US equities: a stable, provider-agnostic
foundation for accumulating daily (and later, intraday) price history, macro data, VIX,
and SEC filing metadata over multiple years, so that pattern-research code can be built
on top of it later.

> **This is NOT a trading system.** There is no recommendation model, no signal
> generation, and no automated trading anywhere in this repository. Investment
> decisions are made by a human, using data this project collects and validates.

---

## 0. Design at a glance

```
external data sources
    -> raw storage (data/raw)            <- original responses, kept as-is
    -> normalization (app/normalization)  <- provider-specific -> internal schema
    -> Parquet data lake (data/lake)      <- bulk time series, hive-partitioned, ZSTD
    -> DuckDB catalog (data/state)        <- security master, job provenance, DQ issues
    -> features_daily / labels_forward_returns
    -> research_daily convenience view
```

* **Large time series live in Parquet, not in DuckDB tables.** DuckDB is used as
  metadata/catalog storage (security master, job bookkeeping, data-quality findings)
  and as a convenient SQL query engine *over* the Parquet lake (via views).
* **No Kafka / Spark / Hadoop / Airflow / Elasticsearch / Kubernetes / Redis / cloud
  DB / paid API dependency.** Everything runs as a single Python process, on one
  always-on but modest server (1x Xeon E5-2695 v4, 64GB RAM), in small resumable
  batches.
* **Providers are swappable.** Every external data source sits behind an abstract
  interface (`PriceProvider`, `SecurityMasterProvider`, `MacroProvider`,
  `VolatilityProvider`, `FilingsProvider`, `ShortVolumeProvider` in
  `app/providers/base.py`). Swapping `yfinance` for a licensed vendor later should
  only require a new adapter class + a `.env` change, not a schema rewrite.
* **Security master vs. tracked universe.** `securities` is *everything known*
  from the SEC + ETF seed list (~10k rows). `tracked_securities` is the much
  smaller *operational* subset that daily jobs and validation actually run
  against (see section 5.1). Confusing the two was the source of ~10,000 bogus
  `MISSING_RECENT_DATA` warnings in an earlier version of this project -- keeping
  them separate is a deliberate, permanent design rule now.
* **`--dry-run` means it, everywhere.** Every sync command's dry-run path makes
  zero network requests, zero Parquet writes, zero DuckDB data mutations, and
  zero checkpoint writes -- it only reports, from already-known local metadata,
  what it *would* do. See section 8.1.

---

## 1. Requirements

* Python 3.12+
* [`uv`](https://docs.astral.sh/uv/) (recommended) or plain `pip`
* Linux or Windows (scripts are provided for both)
* No paid services required for the default setup. A **free** FRED API key is
  needed only for the macro-sync command.

### Installing Python / uv

**Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

**Windows (PowerShell):**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

If you'd rather not use `uv`, a normal `python -m venv .venv` + `pip install -e ".[dev]"`
works identically.

---

## 2. Installation

```bash
git clone <this-repo>
cd stock-pattern-db
uv venv --python 3.12 .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"
```

---

## 3. Environment configuration

```bash
cp .env.example .env
```

Then edit `.env`:

| Variable | Required for | Notes |
|---|---|---|
| `SEC_USER_AGENT` | `sync-universe`, `sync-sec-filings` | SEC requires a descriptive User-Agent with a contact email on every request, e.g. `StockPatternResearch you@example.com`. Requests **without one are rejected**. |
| `FRED_API_KEY` | `sync-macro` | Free. See below. |
| `COMMERCIAL_MODE` | all providers | Keep `false` for personal research. See "License safety" below. |
| `PRICE_PROVIDER` | price backfill/sync | `yfinance` by default; swap to a licensed provider name once one is implemented. |
| `MAX_WORKERS`, `PRICE_BATCH_SIZE`, `DUCKDB_THREADS`, `DUCKDB_MEMORY_LIMIT` | resource limits | Tuned conservatively for a shared, modest server. |
| `DATA_ROOT`, `RAW_DIR`, `LAKE_DIR`, `STATE_DIR`, `LOG_DIR` | storage location | See "Moving the data directory" below. |
| `MARKET_CALENDAR`, `MARKET_DATA_GRACE_MINUTES` | missing-data checks | Which `exchange_calendars` calendar to use (default `XNYS`) and how long after close to wait before flagging today's session as missing. See section 8.2. |
| `COMPACT_FILE_COUNT_THRESHOLD`, `COMPACT_AVG_FILE_SIZE_MB` | storage health | Thresholds used by `stockdb storage-health` to flag compaction candidates. See section 16. |

**Never commit your real `.env`** -- it's already gitignored. Only `.env.example`
is committed.

### Getting a free FRED API key

1. Create a free account at <https://fred.stlouisfed.org/>.
2. Request an API key at <https://fred.stlouisfed.org/docs/api/api_key.html>.
3. Put it in `.env` as `FRED_API_KEY=...`.

### Setting the SEC User-Agent

SEC's fair-use policy requires a descriptive User-Agent identifying your
application and a real contact email, e.g.:

```
SEC_USER_AGENT="StockPatternResearch your-email@example.com"
```

Requests are also rate-limited in code to a conservative default of 2-4
requests/second (`SEC_REQUESTS_PER_SECOND`, hard-capped at 10) -- please do not
raise this beyond SEC's documented fair-use guidance.

---

## 4. Initialize the database

```bash
stockdb init
```

This creates the `data/` directory skeleton, the DuckDB metadata schema
(`data/state/catalog.duckdb`), and seeds the symbol-mapping table. Safe to run
multiple times (idempotent).

Upgrading an **existing** installation never requires deleting or rebuilding
`catalog.duckdb`: every command applies new table DDL (`CREATE TABLE IF NOT
EXISTS`) and any accompanying data migration (`app/db/migrations.py`) on
startup. For example, upgrading to the tracked-universe feature
automatically (and idempotently) registers every security that already has
price data in the lake as tracked, without you having to do anything.

---

## 5. First run: 10-ticker smoke test

Sync the security master (SEC EDGAR + the ETF seed list):

```bash
stockdb sync-universe
```

Backfill 2 years of daily prices for the 10-ticker smoke universe:

```bash
stockdb backfill-prices --symbols AAPL,MSFT,NVDA,AMZN,GOOGL,META,BRK-B,JPM,XOM,SPY --start 2024-01-01
```

Pull VIX and (if you set `FRED_API_KEY`) macro data:

```bash
stockdb sync-vix
stockdb sync-macro
```

Securities that already have price data are **automatically** added to the
tracked universe (see 5.1), so at this point the 10 smoke-test tickers are
already tracked -- no extra step needed.

Run data-quality validation and check status:

```bash
stockdb validate prices
stockdb status
```

### 5.1 Tracked universe vs. security master

`stockdb sync-universe` populates `securities` with **everything** SEC/the ETF
seed list knows about (~10,438 rows as of this writing) -- that is the full
*security master*, not something you operate on day to day.

`tracked_securities` is the small, deliberate, *operational* subset that
`sync-prices`, `sync-sec-filings`, and `stockdb validate prices` actually run
against by default. A security becomes tracked in one of two ways:

1. **Automatically**, the first time it already has price data in the lake
   (an idempotent migration that runs on every command -- see section 12).
2. **Explicitly**, via the `universe` command group:

```bash
stockdb universe tracked                 # list the tracked universe
stockdb universe add AAPL MSFT NVDA      # add tickers (enables price + filings tracking)
stockdb universe add AAPL --no-filings   # add without enabling SEC filing checks
stockdb universe remove AAPL             # soft-remove (history is kept, never deleted)
```

`stockdb backfill-prices` (no `--symbols`) still targets **every active**
security in the security master -- that is the deliberate, explicit
full-universe-expansion tool (section 7). Day-to-day commands
(`sync-prices`, `sync-sec-filings`, `validate prices`, `run-daily`) default to
the tracked universe instead, specifically so that adding 10,000 SEC-known
tickers to the security master never implicitly makes them part of daily
operations. Pass `stockdb validate prices --all-universe` if you deliberately
want to check every active security instead.

## 6. Inspecting the data

Everything is queryable through DuckDB, either via the CLI's `status`
command or directly:

```bash
python -c "
from app.config.settings import get_settings
from app.db.connection import get_connection
from app.db.schema import create_lake_views

settings = get_settings()
con = get_connection(settings)
create_lake_views(con, settings)
print(con.execute('''
    SELECT ticker_at_time, date, close, volume
    FROM prices_daily
    WHERE ticker_at_time = \'AAPL\'
    ORDER BY date DESC
    LIMIT 20
''').fetchdf())
"
```

`prices_daily` (and `macro`, `volatility`, `filings`, `corporate_actions`,
`short_volume` once populated) are DuckDB views over the Parquet lake with
hive partitioning + a "last write wins" dedup already applied, so you never
see duplicate rows even if the same date range was re-ingested multiple
times.

---

## 7. Expanding the universe (10 -> 100 -> 500 -> full)

Backfills are **never** started implicitly. Each expansion step is an
explicit command:

```bash
# Step 1: already done above (10 tickers x 2 years)

# Step 2: same 10 tickers, full history since 2000
stockdb backfill-prices --symbols AAPL,MSFT,NVDA,AMZN,GOOGL,META,BRK-B,JPM,XOM,SPY --start 2000-01-01

# Step 3: pick your own 100-ticker list
stockdb backfill-prices --symbols <100 comma-separated tickers> --start 2015-01-01

# Step 4: 500 tickers
stockdb backfill-prices --symbols <500 comma-separated tickers> --start 2010-01-01

# Step 5: the full active universe from the security master (omit --symbols).
# This is an explicit, deliberate command -- it is never triggered automatically.
stockdb backfill-prices --start 2000-01-01 --batch-size 50
```

Backfills process symbols in small batches (`--batch-size`, default from
`PRICE_BATCH_SIZE`) and checkpoint after every batch under
`data/state/checkpoints/`. If the process is killed (power loss, OOM,
Ctrl-C, ...), re-run the **exact same command** with `--resume` to continue
from the first incomplete symbol instead of starting over:

```bash
stockdb backfill-prices --start 2000-01-01 --batch-size 50 --resume
```

---

## 8. Daily incremental updates

```bash
stockdb sync-prices          # daily fast path: only STALE/NO_DATA tracked names; skip if already current
stockdb repair-prices        # weekly repair: re-fetch last PRICE_REPAIR_LOOKBACK_SESSIONS for tracked names
stockdb run-daily            # universe -> prices -> vix -> macro -> filings -> validate -> report
```

### 8.1 What `--dry-run` actually guarantees

```bash
stockdb run-daily --dry-run
```

`--dry-run` (on every command that has it) makes **no** HTTP request to SEC,
Yahoo/yfinance, FRED, or Cboe; writes **no** Parquet file; makes **no** data
mutation to any DuckDB table (`ingest_runs`, `tracked_securities`,
`data_quality_issues`, ...); and touches **no** checkpoint file. Each
ingestion function checks `dry_run` *before* resolving/looping over its
targets or instantiating a provider, so `run-daily --dry-run` finishes in
about a second even against a 10,000+ security master -- it never loops over
more than the small tracked universe, and it never "resolves 10,438 symbols
and then skips each one in a loop" (an earlier bug). Example output:

```
Universe    would sync SEC universe + ETF seed list (10438 securities known)
Prices      SKIPPED - all 9 tracked securities already current through YYYY-MM-DD
            (or: would fetch N of 9 if any are STALE/NO_DATA)
VIX         would fetch latest VIX
Macro       SKIPPED - FRED_API_KEY not configured
Filings     would check 9 tracked CIK(s)
Validation  0 issues (0 critical) over 9 securities
```

A missing `FRED_API_KEY` is reported as **SKIPPED**, never as a failure --
it's a configuration choice, not a system error, and the rest of the
pipeline keeps going either way (with or without `--dry-run`).

### 8.2 US market trading-day calendar

"Is today's price data missing yet?" is answered using the real NYSE (XNYS)
trading-session calendar (`app/services/market_calendar.py`, backed by the
`exchange-calendars` library) -- never a naive `weekday() < 5` check. That
means weekends *and* US market holidays are never flagged as missing, and a
session that hasn't closed yet (plus a configurable grace period,
`MARKET_DATA_GRACE_MINUTES`, default 120) isn't expected to have data yet
either. This calculation always reasons in `America/New_York` time
internally regardless of the server's own timezone, so a server running in
Korea still correctly knows whether "today" (US market time) has even
started yet. Stored timestamps remain UTC as always; only this one piece of
"what does 'today' mean for the US market" logic uses NY time.

### Scheduling

**Linux (cron):**
```bash
crontab -e
# Run once a day; pick whatever local time zone your crontab uses -- the
# scripts and code never hardcode a time zone.
0 9 * * * /path/to/stock-pattern-db/scripts/run_daily.sh >> /path/to/stock-pattern-db/data/logs/cron.log 2>&1
```

**Linux (systemd timer):** create a `run-daily.service` that runs
`scripts/run_daily.sh`, and a `run-daily.timer` with `OnCalendar=*-*-* 09:00:00`,
then `systemctl enable --now run-daily.timer`.

**Windows (Task Scheduler):** do **not** auto-register. Run `uv run stockdb doctor`
first; register only when doctor has zero FAIL items and a same-session second
`run-daily` is logically idempotent. From an elevated PowerShell session,
after reviewing `STOCKDB_DAILY_TIME` (default `09:00`, **machine local time**
— Windows Task Scheduler does not use a Python timezone. On a Seoul box 09:00
is well after the US cash close plus provider EOD lag):

```powershell
.\scripts\register_daily_task.ps1
.\scripts\unregister_daily_task.ps1
```

`scripts/run_daily.ps1` sets `WorkingDirectory` to the project root via
`-LiteralPath` (Korean characters and spaces in the path are supported) and
invokes:

```
uv run python -m app.cli.main run-daily
```

not `stockdb.exe` (App Control has blocked that shim on some Windows
installs). Administrator rights are typically required. Timezone is never
hardcoded in Python; the Task Scheduler uses the machine's local clock.

---

## 8.3 Features and labels (v1)

Signal timing is **US regular-session close on trading day t**. `date=t`
features may use information observable at that close (OHLCV of t, VIX of t,
past prices). They must not use t+1 prices, future volume, future news, or
FRED values whose publication time is unverified.

```bash
uv run stockdb compute-features --start 2024-01-01 --dry-run
uv run stockdb compute-labels --start 2024-01-01 --dry-run
uv run stockdb compute-features --start 2024-01-01
uv run stockdb compute-labels --start 2024-01-01
```

Omitting `--symbols` uses `tracked_securities` with `feature_tracking=true`
-- never the full ~10k security master.

**Price adjustment (derived, never written back to `prices_daily`):**

`adjustment_factor = adj_close / close`

`adjusted_open/high/low = raw * factor`, `adjusted_close = adj_close`.

If `close` or `adj_close` is null/0, adjusted series are null. There is **no**
silent fallback to raw close. Absolute adjusted prices are never features --
only ratios / returns / distances. `price_adjustment_point_in_time=false`
(Yahoo back-adjusts history).

**Labels** use trading-session horizons 1/3/5/10/20, not calendar days.
`forward_return_h = adj_close(t+h)/adj_close(t)-1`. Missing future sessions
are null, never 0. `max_drawdown_next_h` is the minimum close-to-close return
versus date-t close over the next h sessions -- not a path-dependent
peak-to-trough drawdown.

**Incremental:** features read ~300 lookback sessions but write only the
requested range. Daily label updates recompute the last ~30 sessions so
newly mature `forward_return_20d` values get filled.

**FRED is not joined into v1 features** (`macro_point_in_time=false`).
Sector relative strength columns exist but stay null until a trusted mapping
exists.

Query convenience view (do not use it inside the feature engine):

```sql
SELECT ticker, date, ret_20d, rsi_14, volume_ratio_20d, rel_spy_20d,
       forward_return_10d, forward_excess_spy_10d
FROM research_daily
WHERE ticker = 'AAPL'
ORDER BY date DESC
LIMIT 30;
```

The **research sample** is `research_common_equity_daily_v1`: RESEARCH_COMMON_EQUITY
membership only (default `research-common-equity-500`). Benchmark ETFs are market
context columns, not stock-sample rows. Critical unresolved price-quality rows
have `data_quality_valid = false` and are excluded from analysis by default.
Raw lake files are never rewritten by research commands.

**Time split (no snooping):** feature exploration and any freezeable cutoff
(e.g. VIX tertiles) use **development** `2018-01-01`–`2023-12-31` only.
**Validation** `2024-01-01`–latest is a holdout. Validation numbers are reported
and must not be used to retune development rules. Cross-sectional quintiles are
assigned **within each trading date**, so they do not use the future distribution.

```bash
uv run stockdb research reconcile
uv run stockdb research baseline --split development
uv run stockdb research feature-coverage --split development
uv run stockdb research quantiles --feature rel_spy_20d --split development
uv run stockdb research ic --feature rel_spy_20d --split development
uv run stockdb research regimes --feature rel_spy_20d --split development
uv run stockdb research report --version v1
```

Outputs go to `data/research/v1/` (parquet + provenance columns). These are
research statistics (quintile means, medians, development-frozen 1% winsor
means, rank IC, Q5–Q1 spreads, price-bucket overlays). Winsor cutoffs are
frozen from development (`robust_cutoffs.parquet`) and applied unchanged to
validation; original lake labels are never clipped. These outputs are **not**
buy/sell recommendations, rankings, or portfolio weights.

Lake distinct security counts and tracked-universe size are different metrics.
`stockdb status` prints both. Tracked can be larger than the lake when some
tracked names (typically unused sector ETFs) have no price files.

### Research-scale expansion (not an investment universe)

```bash
uv run stockdb expand-universe --research-scale 100
uv run stockdb backfill-prices --tracked --start 2018-01-01 --resume
uv run stockdb compute-features --start 2018-01-01 --resume
uv run stockdb compute-labels --start 2018-01-01 --resume
```

Full active-universe long-term backfill is **never** started automatically:

```bash
uv run stockdb expand-universe --all-active --dry-run
uv run stockdb backfill-prices --all-active --start 2000-01-01 --dry-run
```

---

## 9. Backups

```bash
scripts/backup.sh /path/to/backup/destination      # Linux/macOS
scripts\backup.ps1 -Destination D:\backups\spdb     # Windows
```

* The Parquet lake is backed up **incrementally** (only new/changed files are
  copied).
* The DuckDB catalog is small; a full timestamped copy is taken every run (the
  last 10 snapshots are kept).
* `.env` (secrets) is never included in the backup.
* Raw downloads (`data/raw`) are skipped by default (pass `--with-raw` /
  `-WithRaw` to include them) since they can be re-downloaded.

---

## 10. Moving the data directory

Everything under `data/` is controlled by four environment variables:

```
DATA_ROOT=/mnt/bigdisk/stock-pattern-db-data
RAW_DIR=/mnt/bigdisk/stock-pattern-db-data/raw
LAKE_DIR=/mnt/bigdisk/stock-pattern-db-data/lake
STATE_DIR=/mnt/bigdisk/stock-pattern-db-data/state
LOG_DIR=/mnt/bigdisk/stock-pattern-db-data/logs
```

Set these in `.env`, then run `stockdb init` again to (re)create the
directory skeleton at the new location. Nothing else needs to change.

---

## 11. CLI reference

| Command | Purpose |
|---|---|
| `stockdb init` | Create directories + DuckDB schema + seed config data. |
| `stockdb sync-universe [--dry-run]` | Sync security master from SEC + ETF seed list. |
| `stockdb backfill-prices --start DATE [--symbols A,B,C] [--tracked] [--all-active] [--end DATE] [--batch-size N] [--resume] [--dry-run]` | Backfill daily price history. |
| `stockdb sync-prices [--symbols ...] [--dry-run]` | Daily price fast path for the tracked universe. Skips names already current through the expected XNYS session. |
| `stockdb repair-prices [--lookback-sessions N] [--dry-run]` | Weekly repair: re-fetch recent XNYS sessions for every tracked price target (not the full master). |
| `stockdb sync-macro [--series ...] [--start DATE] [--dry-run]` | Sync FRED macro series. |
| `stockdb sync-vix [--dry-run]` | Sync official Cboe VIX history. |
| `stockdb sync-sec-filings [--ciks ...] [--dry-run]` | Sync SEC filing metadata (10-K/10-Q/8-K/20-F/6-K) for the tracked (filings-enabled) CIKs. |
| `stockdb compute-features --start DATE [--symbols ...] [--end DATE] [--version v1] [--resume] [--dry-run]` | Compute `features_daily` for the feature-tracking universe. |
| `stockdb compute-labels --start DATE [--symbols ...] [--end DATE] [--version v1] [--resume] [--dry-run]` | Compute `labels_forward_returns` (session horizons; immature = null). |
| `stockdb expand-universe --research-scale 100\|500 [--dry-run]` | Deterministic PROVIDER_SCALE_TEST universe (not an investment universe). |
| `stockdb expand-universe --research-common-equity 100\|500 [--dry-run]` | Deterministic RESEARCH_COMMON_EQUITY universe. Benchmarks stay separate. |
| `stockdb expand-universe --all-active [--dry-run]` | Register every active security as tracked. Does **not** start a backfill. |
| `stockdb validate prices [--symbols ...] [--all-universe] [--dry-run]` | Run data-quality checks (tracked universe by default). |
| `stockdb validate features [--symbols ...] [--dry-run]` | Feature validation (RSI bounds, negative ATR/vol, inf). |
| `stockdb validate labels [--symbols ...] [--dry-run]` | Label validation (return < -1, extremes). Never winsorizes. |
| `stockdb audit prices` | Split OHLC critical/warning into research-common vs non-common; count rounding vs true violations. |
| `stockdb audit extreme-labels` | Classify EXTREME_LABEL warnings (does not raise the threshold). |
| `stockdb research reconcile` | Lake vs tracked vs named-universe security coverage. |
| `stockdb research baseline [--split development\|validation]` | Unconditional forward-return / SPY-excess baseline. |
| `stockdb research feature-coverage [--split ...]` | Null ratios and distribution of v1 features. |
| `stockdb research quantiles --feature NAME [--split ...]` | Date-by-date cross-sectional quintiles. |
| `stockdb research ic [--feature NAME] [--split ...]` | Spearman rank IC by date, then mean/IR. |
| `stockdb research regimes --feature NAME [--split ...]` | Quintiles sliced by SPY-vs-MA200 and frozen VIX tertiles. |
| `stockdb research report --version v1 [--features a,b]` | Write `data/research/v1/*.parquet` including robust/winsor (dev-frozen cutoffs), price overlays, and feature rank correlation. Validation is not used to retune. |
| `stockdb compact prices [--year Y --month M] [--dry-run] [--verify/--no-verify]` | Merge partition files (omit year/month for the whole prices dataset). `--verify` is the default. |
| `stockdb storage-health [--dataset NAME]` | Show per-partition file/size stats and flag compaction candidates. |
| `stockdb universe tracked [--include-disabled]` | List the tracked (operational) universe. |
| `stockdb universe add TICKER... [--reason TEXT] [--no-filings]` | Add tickers to the tracked universe. |
| `stockdb universe remove TICKER...` | Soft-remove tickers from the tracked universe (history kept). |
| `stockdb status` | Show DB/lake/job status as Rich tables (dates include English weekday; daily target vs known master). |
| `stockdb doctor` | Pre-scheduler environment check (OK/WARN/FAIL). Does not print secrets. Does not register Task Scheduler. |
| `stockdb run-daily [--dry-run]` | Daily pipeline: universe → prices → VIX → FRED → filings → validate → incremental features → recent labels → validate. |
| `stockdb indicators list [--group ...]` | List on-demand technical indicators (memory-only). |
| `stockdb indicators show --symbol TICKER [--start] [--end] [--group] [--indicator] [--all] [--tail N]` | Compute indicators in memory and print a table. Never writes files. |
| `stockdb mine run [--target] [--analysis-start/end] [--validation-start/end] [--event-mode state\|entry] [--cooldown-sessions N] [--max-rule-size 2] [--top 30]` | Discover on ANALYSIS; evaluate the frozen set on VALIDATION. No files written. Not recommendations. |
| `stockdb mine walk-forward [--target] [--full-discover] [--event-mode] [--cooldown-sessions]` | Expanding-window walk-forward. Fold cutoffs freeze on that fold. FUTURE_HOLDOUT is not evaluated. |
| `stockdb mine inspect --pattern NAME [--pattern NAME]` | STATE vs ENTRY vs cooldown, parent incremental, episodes, years, price/regime. Terminal only. |

---

## 12. Architecture details

### Directory layout

```
stock-pattern-db/
  app/
    config/        # settings, seed lists (ETFs, FRED series, symbol overrides, smoke universe)
    db/             # DuckDB connection + metadata schema + lake views
    models/         # pydantic row models (Security, PriceBar, MacroObservation, ...)
    providers/       # abstract interfaces + concrete adapters (SEC, yfinance, FRED, Cboe, ...)
    ingestion/       # orchestration: universe/price/macro/vix/filings sync, checkpoint/resume
    normalization/   # raw provider rows -> internal schema (+ symbol mapping)
    validation/       # data-quality rules + runner
    services/         # status report, compaction, provenance/manifest bookkeeping
    cli/              # Typer CLI wiring only -- no business logic
    research/         # Univariate research stats (not recommendations)
    indicators/       # On-demand TA engine (memory-only; no lake writes)
    mining/           # Ephemeral pattern discovery + holdout eval (no result files)
    utils/            # logging, rate limiting, atomic I/O, Parquet lake read/write

  data/
    raw/            # original provider responses, preserved as-is (per dataset)
    lake/            # normalized Parquet, hive-partitioned by year/month
      prices_daily/
      corporate_actions/
      macro/
      volatility/
      filings/
      short_volume/
      features_daily/
      labels_forward_returns/
    research/        # analysis outputs (never mixed into the lake)
      v1/
    state/           # DuckDB catalog + checkpoints
    logs/            # rotating log files

  scripts/          # run_daily.sh/.ps1, register/unregister_daily_task.ps1, backup.sh/.ps1
  tests/            # pytest suite (no live network calls)
```

### DuckDB metadata schema

* `ingest_runs` -- one row per ingestion run (provider, dataset, status, counts, timing, error).
* `source_files` -- one row per raw/Parquet file produced by a run (path, checksum, row count, date range).
* `securities` -- security master (`security_id`, `cik`, `company_name`, `primary_ticker`, `exchange`, `asset_type`, `is_active`, ...).
* `security_identifiers` -- identifier history (`identifier_type` in `CIK`/`TICKER`, `valid_from`/`valid_to`), so ticker renames never destroy history.
* `security_snapshots` -- point-in-time universe snapshots (`snapshot_date`, ticker/company/exchange as of that date) to help reduce survivorship bias later.
* `symbol_mappings` -- canonical ticker <-> provider-specific spelling overrides.
* `data_quality_issues` -- validation findings (dataset, security_id, date, issue_type, severity, resolved). Rows are marked `resolved`, never deleted, once a finding no longer reproduces on a subsequent validation run.
* `tracked_securities` -- the operational subset of `securities` that daily jobs/validation run against (`enabled`, `price_tracking`, `filings_tracking`, `feature_tracking`, `tracking_reason`, `added_at`/`removed_at`). See section 5.1.
* `instrument_classifications` -- heuristic instrument class per security (`instrument_class_complete=false`).
* `universe_memberships` -- named universes (`research-scale-N`, `research-common-equity-N`, `benchmark-etf-seed`) with `universe_type`.
* `dataset_metadata` -- machine-readable research-integrity flags per dataset (`dataset_name`, `metadata_key`, `metadata_value`). See section 17.

### Parquet partitioning strategy

* Hive-style `year=YYYY/month=MM/` partitions per dataset.
* Files within a partition are sorted by `(security_id, date)` (or the
  dataset's natural key) and written with ZSTD compression.
* Price (and daily-sync) writes are **batch-then-partition**: a 25-symbol
  batch is concatenated, then `write_increment` emits **one file per
  month partition per batch**, not one file per symbol. Atomic write,
  provenance, append, and last-write-wins dedup are unchanged.
* Each ingestion run **appends** a new file to the partitions it touched --
  it never rewrites an entire partition on every ingest (which would make a
  long backfill O(n^2) in total I/O). This means the same logical row can
  briefly exist in more than one file after re-ingestion; every DuckDB view
  over the lake applies a "last write wins" dedup (`ORDER BY retrieved_at DESC`)
  so queries are always correct.
* `stockdb compact prices` (omit `--year/--month` to compact every monthly
  partition) physically merges a partition's files into one, dropping
  superseded duplicate rows. Compaction is temp-write → validate → atomic
  replace → delete old files. `--verify` (default) checks that logical
  rows, security count, min/max date, and duplicate count are unchanged.
* All writes are temp-file -> validate -> atomic `os.replace()`, so a crash
  mid-write can never corrupt or truncate existing data.

### OHLC numeric tolerance

Price bars are floats. v1 treats

```
tol(a, b) = max(0.001, 1e-5 * max(|a|, |b|, 1.0))
approximately_ge(a, b)  iff  a + tol(a,b) >= b
```

A $24 preferred with `low = open + 0.0001` is `OHLC_ROUNDING_TOLERANCE`
(info), not critical. A warrant bar with `high=10.08` vs `open=10.15` stays
`OHLC_INCONSISTENT` (critical). Raw prices are never auto-corrected.
True-invalid OHLC rows keep the price bar; ATR / gap / range / wick /
high-low-distance features on that row are set to null.

### Universes

* `KNOWN` -- the SEC security master (`securities`).
* `PROVIDER_SCALE_TEST` -- `research-scale-N` hash sample (may include
  warrants/preferred). Engineering scale test, not the research target.
* `RESEARCH_COMMON_EQUITY` -- `research-common-equity-N`. Active NYSE/Nasdaq/CBOE
  names classified as common equity. No performance-based selection.
* `BENCHMARK` -- curated ETF seed (SPY, QQQ, IWM, DIA, sector ETFs, ...).

Instrument classification (`instrument_class_complete=false`) uses SEC
metadata first, ticker suffix last:

1. `SEC_SECURITY_TITLE` -- official listing/prospectus title when known
   (e.g. junior subordinated debentures → `DEBT`).
2. `SEC_REGISTRANT_TYPE` -- investment-company SEC forms (N-2/N-CSR/...)
   and registrant-name patterns (municipal income trust, closed-end, ...)
   → `CLOSED_END_FUND` / `FUND`.
3. Ticker-suffix heuristics (warrant/preferred/unit/right).
4. Simple US ticker → `COMMON_EQUITY` at medium confidence.

Ambiguous 5-letter `*W` names are not forced to COMMON_EQUITY. Raw prices
are never rewritten when a class changes.

### Providers implemented

| Interface | Adapter | Notes |
|---|---|---|
| `SecurityMasterProvider` | `SecUniverseProvider` (SEC `company_tickers_exchange.json`) | Free, official. Combined with a curated ETF seed list (`app/config/etf_seed.py`) since SEC's feed does not classify most sector/ index ETFs. |
| `PriceProvider` | `YFinancePriceProvider` (`yfinance`) | **Personal research/prototyping only** -- see licensing note below. `auto_adjust=False` by default; raw OHLC + Yahoo-computed `Adj Close` + dividends/splits are all preserved. |
| `MacroProvider` | `FredMacroProvider` | Free, requires API key. Schema already carries `realtime_start`/`realtime_end` for future ALFRED vintage support. |
| `VolatilityProvider` | `CboeVixProvider` | Free, official Cboe historical VIX CSV. |
| `FilingsProvider` | `SecFilingsProvider` | SEC EDGAR submissions API, metadata only (no document parsing). |
| `ShortVolumeProvider` | *interface + schema only* | Not implemented in v1 -- see spec section 4-F. Verify FINRA's terms before activating, and keep it disabled unless `COMMERCIAL_MODE=false`. |

### License safety net

Every provider declares `ProviderCapabilities.commercial_use_safe`. If
`COMMERCIAL_MODE=true` and a provider is not marked safe, it refuses to run
with:

```
This provider is configured for research/personal use only.
Choose a licensed commercial provider before running in commercial mode.
```

`yfinance` is marked `commercial_use_safe=False` / `redistribution_safe=False`
for exactly this reason.

---

## 13. Known limitations (please read before relying on this data)

* **Yahoo/yfinance price data is for personal research and prototyping
  only.** It is unofficial and undocumented. Before any commercial use or
  data redistribution, review licensing and switch `PRICE_PROVIDER` to a
  properly licensed vendor (Massive, Tiingo, FMP, Sharadar, etc.) by
  implementing `app.providers.base.PriceProvider`.
* **The security master is not a complete historical delisting database.**
  SEC's `company_tickers_exchange.json` reflects *currently listed* tickers;
  it does not give you a point-in-time universe for arbitrary past dates.
  **Do not assume survivorship bias has been eliminated** -- `security_snapshots`
  helps going forward (each day's universe is preserved once you start
  running `sync-universe` regularly), but past history before you started
  running this pipeline is not reconstructed.
* **Security IDs for multi-share-class companies are ticker-suffixed.** SEC's
  feed can list several tickers under one CIK (multi-class shares, e.g.
  GOOGL/GOOG/BRK-A/BRK-B, or numerous preferred-share classes). When a CIK
  maps to exactly one ticker, `security_id` is purely CIK-based and survives
  ticker renames. When a CIK maps to multiple tickers, `security_id` is
  `CIK...-TICKER`, so a rename of *that specific share class* would be
  treated as a new security -- a real gap that a licensed identifier system
  (LEI/FIGI) would close.
* **This is not an investment recommendation or automated-trading system.**
  It only collects, normalizes, and validates data. All investment decisions
  are made by a human.
* **Data errors can and do exist in free sources.** Always run
  `stockdb validate prices` and review `data_quality_issues` before drawing
  conclusions from the data; validation records findings, it never silently
  deletes or "fixes" data.
* **`survivorship_safe=false`.** The current universe is today's listed names
  plus a small ETF seed. Delisted history is incomplete.
* **`point_in_time_security_master=false`.** Ticker/exchange as-of-date is
  not reconstructed for dates before this pipeline started.
* **`price_adjustment_point_in_time=false`.** Adjustment uses the provider's
  current `adj_close/close` factor, which is back-filled after later splits
  and dividends. A stricter PIT adjustment vendor can replace this later.

These same limitations are also written as **machine-readable metadata**
(`dataset_metadata` table / `stockdb status`'s "Research integrity" panel) --
see section 17 -- so a future features/labels/backtest layer can check them
programmatically instead of relying on someone having read this file.

---

## 14. Testing

```bash
source .venv/bin/activate
pytest
```

The test suite (90 tests) never calls live external APIs -- it uses in-memory
DuckDB connections, `tmp_path`-scoped Settings, `monkeypatch`ed provider
methods, and hand-built fixture rows that mimic real provider responses (SEC
universe rows, price bars, etc.). Covered areas: SEC universe
parsing/disambiguation, symbol mapping, price normalization (nulls
preserved, never zero-filled), Parquet partition writing + dedup/compaction,
atomic file writes, checkpoint save/load/resume, validation rules, the
commercial-mode provider guard, the NYSE trading calendar (weekends/US
holidays/session-close+grace logic), the tracked-vs-full-universe validation
scoping (including stale-issue resolution), true dry-run guarantees (zero
network calls, zero Parquet writes, zero DuckDB row-count changes), storage
health/compaction-candidate thresholds, research-integrity metadata, and
migration idempotency (including "never resurrects an explicitly removed
security").

---

## 15. Resource usage principles

* Defaults (`MAX_WORKERS=2`, `PRICE_BATCH_SIZE=25`, `FEATURE_BATCH_SIZE=50`,
  `DUCKDB_THREADS=6`, `DUCKDB_MEMORY_LIMIT=24GB`) are tuned for a shared,
  single-CPU 64GB server that may also be used for other things.
* Backfills process fixed-size batches with a checkpoint after every batch,
  so a long-running job never needs to hold the whole universe in memory,
  and never needs to restart from scratch after an interruption.
* Bulk metadata writes (security master upserts, validation findings) go
  through a single vectorized `INSERT ... SELECT` against a registered
  DuckDB relation rather than thousands of individual `INSERT` statements --
  this keeps a full ~10k-security universe sync under 2 seconds instead of
  minutes.
* Polars/DuckDB are used for all bulk transformations; there are no
  Python `for` loops over large row sets in the hot path.

---

## 16. Storage health / compaction candidates

```bash
stockdb storage-health                 # every dataset
stockdb storage-health --dataset prices
```

Shows, per `dataset/year/month` partition: file count, total size, an
estimated row count (from Parquet metadata, no full read), average file
size, and the newest file's timestamp. A partition is flagged as a
**compaction candidate** when either:

* `file_count >= COMPACT_FILE_COUNT_THRESHOLD` (default 25), or
* it has more than one file and `avg_file_size_mb < COMPACT_AVG_FILE_SIZE_MB`
  (default 8).

This is detection only -- **nothing is compacted automatically**, in the
daily pipeline or anywhere else. Run the existing explicit command for a
flagged partition:

```bash
stockdb compact prices --year 2024 --month 01
```

Delta Lake / Iceberg-style automatic compaction is deliberately out of
scope; the append + view-dedup + explicit-compact architecture (section 12)
is unchanged, this just adds visibility into when running `compact` is
actually worth it.

---

## 17. Research integrity / survivorship-bias metadata

`stockdb status` ends with a small "Research integrity" panel:

```
Historical universe complete    NO
Survivorship-safe universe      NO
Point-in-time security master   NO
Price provider                  yfinance
Commercial use safe             NO
```

This is backed by the `dataset_metadata` table (`dataset_name`,
`metadata_key`, `metadata_value`), seeded/refreshed idempotently on every
command (`app/services/dataset_metadata_service.py`). It exists so a future
features/labels/backtest layer can check e.g. `survivorship_safe`
programmatically before drawing conclusions from a backtest, instead of
relying on someone having read section 13. `research_only_price_provider`
(shown inverted as "Commercial use safe") is derived automatically from the
active `PRICE_PROVIDER`'s `ProviderCapabilities.commercial_use_safe` --
switching providers updates this the next time any command runs.

---

## 18. On-demand indicators and ephemeral mining

These engines **read** `prices_daily` / `features_daily` / `labels_forward_returns`.
They **do not write** Parquet, CSV, DuckDB tables, `data/research`, manifests, or checkpoints.

```
INDICATOR_PERSISTENCE_ENABLED=false
MINING_RESULT_PERSISTENCE_ENABLED=false
PATTERN_REGISTRY_PERSISTENCE_ENABLED=false
DAILY_SIGNAL_PERSISTENCE_ENABLED=false
```

Pattern-research registry / dashboard (modular; no live mining):

```
stockdb registry generators
stockdb registry patterns
stockdb registry generator-metrics
stockdb registry dashboard --demo
```

`--demo` loads synthetic fixtures in memory and never creates `data/state/pattern_registry.duckdb`. Bind defaults to `127.0.0.1:8765`.

Adjustment matches the feature engine: `factor = adj_close / close`. Invalid factor → indicator nulls (no silent raw-close fallback).

### Indicator notes

- **Ichimoku:** `cloud_a_at_t` / `cloud_b_at_t` are Senkou A/B computed 26 sessions earlier. Visual Chikou (`visual_chikou_shifted`) is `causal_safe=false` and excluded from mining. Use `close_vs_close_26` instead.
- **Donchian / prior high-low breakouts:** threshold is the previous N-session high/low (`shift(1)`). The current bar is not in the breakout channel.
- **Pivots:** Classic / Fibonacci / Camarilla / Woodie from the **prior** session OHLC only.
- **Bollinger/Keltner squeeze:** `squeeze_on` when BB(20,2) lies entirely inside KC(EMA20, ATR10×2). `squeeze_release` is the first session `squeeze_on` becomes false.
- **Rolling VWAP:** daily typical price × volume over N bars. This is **not** an intraday session VWAP.
- **Rolling Fibonacci 60:** levels of the last 60-session high/low range. Not a manual swing retracement.
- **Candlesticks:** project rules, not TA-Lib clones. Doji body ≤ 10% of range. Hammer lower wick ≥ 2× body and upper wick ≤ 0.3× body. Engulfing uses close/open containment. Morning/evening star: two-session-ago long bar, small middle, reversal close through the midpoint.

### Mining

Default ANALYSIS `2018-01-02`–`2023-12-29`. The later window is **VALIDATION** (`2024-01-02`–latest mature date). It was previously labeled TEST; the dates did not change. Ranges must not overlap; VALIDATION is later.

`FUTURE_HOLDOUT` starts `2026-09-21`. v1 does not evaluate it. Quantile cutoffs, candidate selection, and winsor p1/p99 freeze on the discover window (`PATTERN_FREEZE_POLICY`). VALIDATION / walk-forward eval years / FUTURE_HOLDOUT must not retune them.

`--event-mode state` uses every true day. `--event-mode entry` uses false→true only. `--cooldown-sessions N` is an event robustness filter (same security is not counted again for N sessions). Episodes are consecutive true runs in memory (not stored).

Walk-forward expanding windows freeze cutoffs on each fold's discover period, then evaluate the next calendar year only.

Pair rules also report incremental effect vs each parent and Jaccard overlap (`HIGH_OVERLAP` when P(B\|A)>0.95 or Jaccard>0.90).

Support defaults: 1000 rows, 100 dates, 30 securities. FDR is Benjamini-Hochberg on ANALYSIS same-date CS contrast p-values (`q ≤ 0.10` by default). Date-clustered means use lag = horizon.

Price-floor and SPY/VIX regime slices are robustness audits of already-selected patterns. They do not retune thresholds.

This is not a trading system.
