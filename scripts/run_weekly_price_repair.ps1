# Weekly (on-demand) price repair runner for Windows.
#
# Re-fetches the last PRICE_REPAIR_LOOKBACK_SESSIONS XNYS sessions for the
# tracked price universe (~539 names). Does not expand to the full security
# master. This script is NOT registered with Task Scheduler by default.
#
# Example (manual):
#   powershell.exe -ExecutionPolicy Bypass -File "C:\path\to\stock-pattern-db\scripts\run_weekly_price_repair.ps1"
#
# Invocation uses `uv run python -m app.cli.main repair-prices` rather than
# `stockdb.exe`. On this Windows install stockdb.exe has been blocked by
# App Control; python -m is the stable entrypoint.
#
# WorkingDirectory / Set-Location use -LiteralPath so project roots with
# spaces or non-ASCII characters (e.g. C:\Users\cheye\주식 분석\stock-pattern-db)
# resolve correctly.

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = (Resolve-Path -LiteralPath (Split-Path -Parent $ScriptDir)).Path

Set-Location -LiteralPath $ProjectDir

function Get-UvExecutable {
    $local = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (Test-Path -LiteralPath $local) {
        return $local
    }
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    throw "uv not found. Install uv (https://docs.astral.sh/uv/) and retry."
}

$Uv = Get-UvExecutable
Write-Output ("=== stockdb repair-prices started at " + (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ") + " ===")
Write-Output ("Project: " + $ProjectDir)
& $Uv run python -m app.cli.main repair-prices
if ($LASTEXITCODE -ne 0) {
    throw "repair-prices exited with code $LASTEXITCODE"
}
Write-Output ("=== stockdb repair-prices finished at " + (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ") + " ===")
