#!/usr/bin/env bash
# Daily pipeline runner for cron / systemd timer.
#
# Example crontab entry (adjust the time to whatever your OS's local time
# zone happens to be -- this script does not hardcode any time zone):
#   0 9 * * *  /path/to/stock-pattern-db/scripts/run_daily.sh >> /path/to/stock-pattern-db/data/logs/cron.log 2>&1
#
# Example systemd timer: create run-daily.service (ExecStart=this script)
# and run-daily.timer (OnCalendar=*-*-* 09:00:00), then
# `systemctl enable --now run-daily.timer`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

echo "=== stockdb run-daily started at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
stockdb run-daily
echo "=== stockdb run-daily finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
