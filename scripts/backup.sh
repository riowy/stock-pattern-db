#!/usr/bin/env bash
# Incremental backup for stock-pattern-db.
#
# Usage:
#   scripts/backup.sh [destination_dir]
#
# Default destination is ../stock-pattern-db-backups (a sibling directory,
# so backups don't live on the exact same disk region as the working data
# by default -- point this at an external drive / mounted network share /
# cloud-synced folder for real safety).
#
# Strategy (see spec section 23):
#   * The Parquet lake (data/lake) is backed up incrementally with rsync
#     (`-a --update`), since files are only appended or replaced atomically
#     during compaction -- unchanged files are skipped, so re-running this
#     script often is cheap.
#   * The DuckDB catalog (data/state) is small metadata; it is copied in
#     full every time (DuckDB files should not be rsynced while a writer
#     might be active, so we take a clean, momentary copy instead).
#   * Raw downloads (data/raw) are optional and off by default -- pass
#     --with-raw to include them (they can be re-downloaded, so they are
#     lower priority for backup).
#   * .env is NEVER copied into the backup (it contains secrets).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

WITH_RAW=false
DEST_DIR=""
for arg in "$@"; do
    if [ "$arg" = "--with-raw" ]; then
        WITH_RAW=true
    else
        DEST_DIR="$arg"
    fi
done
DEST_DIR="${DEST_DIR:-${PROJECT_DIR}/../stock-pattern-db-backups}"

mkdir -p "${DEST_DIR}/lake" "${DEST_DIR}/state" "${DEST_DIR}/config"

copy_incremental() {
    local src="$1" dst="$2"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --update "$src" "$dst"
    else
        echo "(rsync not found; falling back to 'cp -au' -- install rsync for true incremental sync: apt-get install rsync)"
        mkdir -p "$dst"
        cp -au "$src." "$dst" 2>/dev/null || cp -au "$src"* "$dst" 2>/dev/null || true
    fi
}

echo "=== Backing up Parquet lake (incremental) -> ${DEST_DIR}/lake ==="
copy_incremental "${PROJECT_DIR}/data/lake/" "${DEST_DIR}/lake/"

echo "=== Backing up DuckDB state -> ${DEST_DIR}/state ==="
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
cp "${PROJECT_DIR}/data/state/catalog.duckdb" "${DEST_DIR}/state/catalog_${STAMP}.duckdb"
# Keep only the 10 most recent DuckDB snapshots.
ls -1t "${DEST_DIR}/state"/catalog_*.duckdb 2>/dev/null | tail -n +11 | xargs -r rm --

echo "=== Backing up config -> ${DEST_DIR}/config ==="
cp "${PROJECT_DIR}/.env.example" "${DEST_DIR}/config/"
cp "${PROJECT_DIR}/pyproject.toml" "${DEST_DIR}/config/"

if [ "$WITH_RAW" = true ]; then
    echo "=== Backing up raw downloads (incremental) -> ${DEST_DIR}/raw ==="
    mkdir -p "${DEST_DIR}/raw"
    copy_incremental "${PROJECT_DIR}/data/raw/" "${DEST_DIR}/raw/"
fi

echo "=== Backup complete: ${DEST_DIR} ==="
