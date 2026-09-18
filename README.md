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
    -> (future) features / pattern analysis
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

Run data-quality validation and check status:

```bash
stockdb validate prices
stockdb status
```

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
stockdb sync-prices          # refreshes a trailing 10-day window for all active securities
stockdb run-daily            # universe -> prices -> vix -> macro -> filings -> validate -> report
```

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

**Windows (Task Scheduler):** create a daily task that runs
`powershell.exe -ExecutionPolicy Bypass -File C:\path\to\stock-pattern-db\scripts\run_daily.ps1`.

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
| `stockdb backfill-prices --start DATE [--symbols A,B,C] [--end DATE] [--batch-size N] [--resume] [--dry-run]` | Backfill daily price history. |
| `stockdb sync-prices [--symbols ...] [--lookback-days N] [--dry-run]` | Incremental daily price refresh. |
| `stockdb sync-macro [--series ...] [--start DATE] [--dry-run]` | Sync FRED macro series. |
| `stockdb sync-vix [--dry-run]` | Sync official Cboe VIX history. |
| `stockdb sync-sec-filings [--ciks ...] [--dry-run]` | Sync SEC filing metadata (10-K/10-Q/8-K/20-F/6-K). |
| `stockdb validate prices [--symbols ...] [--dry-run]` | Run data-quality checks. |
| `stockdb compact prices [--year Y --month M] [--dry-run]` | Merge small Parquet files into one per partition (also: `macro`, `volatility`, `filings`). |
| `stockdb status` | Show DB/lake/job status as Rich tables. |
| `stockdb run-daily [--dry-run]` | Run the full daily pipeline end to end. |

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
    state/           # DuckDB catalog + checkpoints
    logs/            # rotating log files

  scripts/          # run_daily.sh/.ps1, backup.sh/.ps1
  tests/            # pytest suite (no live network calls)
```

### DuckDB metadata schema

* `ingest_runs` -- one row per ingestion run (provider, dataset, status, counts, timing, error).
* `source_files` -- one row per raw/Parquet file produced by a run (path, checksum, row count, date range).
* `securities` -- security master (`security_id`, `cik`, `company_name`, `primary_ticker`, `exchange`, `asset_type`, `is_active`, ...).
* `security_identifiers` -- identifier history (`identifier_type` in `CIK`/`TICKER`, `valid_from`/`valid_to`), so ticker renames never destroy history.
* `security_snapshots` -- point-in-time universe snapshots (`snapshot_date`, ticker/company/exchange as of that date) to help reduce survivorship bias later.
* `symbol_mappings` -- canonical ticker <-> provider-specific spelling overrides.
* `data_quality_issues` -- validation findings (dataset, security_id, date, issue_type, severity, resolved).

### Parquet partitioning strategy

* Hive-style `year=YYYY/month=MM/` partitions per dataset.
* Files within a partition are sorted by `(security_id, date)` (or the
  dataset's natural key) and written with ZSTD compression.
* Each ingestion run **appends** a new file to the partitions it touched --
  it never rewrites an entire partition on every ingest (which would make a
  long backfill O(n^2) in total I/O). This means the same logical row can
  briefly exist in more than one file after re-ingestion; every DuckDB view
  over the lake applies a "last write wins" dedup (`ORDER BY retrieved_at DESC`)
  so queries are always correct.
* `stockdb compact <dataset> [--year Y --month M]` physically merges a
  partition's files into one, dropping the superseded duplicate rows. Run
  this periodically (e.g. weekly, or as part of a maintenance script) once a
  partition has accumulated many small files.
* All writes are temp-file -> validate -> atomic `os.replace()`, so a crash
  mid-write can never corrupt or truncate existing data.

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

---

## 14. Testing

```bash
source .venv/bin/activate
pytest
```

The test suite never calls live external APIs -- it uses in-memory DuckDB
connections, `tmp_path`-scoped Settings, and hand-built fixture rows that
mimic real provider responses (SEC universe rows, price bars, etc.).
Covered areas: SEC universe parsing/disambiguation, symbol mapping,
price normalization (nulls preserved, never zero-filled), Parquet
partition writing + dedup/compaction, atomic file writes, checkpoint
save/load/resume, validation rules, and the commercial-mode provider guard.

---

## 15. Resource usage principles

* Defaults (`MAX_WORKERS=4`, `PRICE_BATCH_SIZE=50`, `DUCKDB_THREADS=6`,
  `DUCKDB_MEMORY_LIMIT=24GB`) are tuned for a shared, single-CPU 64GB server
  that may also be used for other things.
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
