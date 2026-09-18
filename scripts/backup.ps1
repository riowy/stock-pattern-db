# Incremental backup for stock-pattern-db (Windows).
#
# Usage:
#   powershell.exe -File scripts\backup.ps1 [-Destination <path>] [-WithRaw]
#
# See scripts/backup.sh for the strategy explanation (Parquet lake copied
# incrementally, DuckDB catalog copied as a full timestamped snapshot,
# .env never included).

param(
    [string]$Destination = "",
    [switch]$WithRaw
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

if ([string]::IsNullOrEmpty($Destination)) {
    $Destination = Join-Path (Split-Path -Parent $ProjectDir) "stock-pattern-db-backups"
}

New-Item -ItemType Directory -Force -Path (Join-Path $Destination "lake") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Destination "state") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Destination "config") | Out-Null

Write-Output "=== Backing up Parquet lake (incremental) -> $Destination\lake ==="
robocopy "$ProjectDir\data\lake" "$Destination\lake" /E /XO /NFL /NDL /NJH /NJS | Out-Null

Write-Output "=== Backing up DuckDB state -> $Destination\state ==="
$Stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
Copy-Item "$ProjectDir\data\state\catalog.duckdb" "$Destination\state\catalog_$Stamp.duckdb"
Get-ChildItem "$Destination\state\catalog_*.duckdb" | Sort-Object LastWriteTime -Descending | Select-Object -Skip 10 | Remove-Item

Write-Output "=== Backing up config -> $Destination\config ==="
Copy-Item "$ProjectDir\.env.example" "$Destination\config\" -Force
Copy-Item "$ProjectDir\pyproject.toml" "$Destination\config\" -Force

if ($WithRaw) {
    Write-Output "=== Backing up raw downloads (incremental) -> $Destination\raw ==="
    New-Item -ItemType Directory -Force -Path (Join-Path $Destination "raw") | Out-Null
    robocopy "$ProjectDir\data\raw" "$Destination\raw" /E /XO /NFL /NDL /NJH /NJS | Out-Null
}

Write-Output "=== Backup complete: $Destination ==="
