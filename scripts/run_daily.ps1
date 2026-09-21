# Daily pipeline runner for Windows Task Scheduler.
#
# Example: create a Task Scheduler task that runs
#   powershell.exe -ExecutionPolicy Bypass -File "C:\path\to\stock-pattern-db\scripts\run_daily.ps1"
# on a daily trigger (pick whatever local time your machine uses -- this
# script does not hardcode any time zone).
#
# Invocation uses `uv run python -m app.cli.main run-daily` rather than
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
Write-Output ("=== stockdb run-daily started at " + (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ") + " ===")
Write-Output ("Project: " + $ProjectDir)
& $Uv run python -m app.cli.main run-daily
if ($LASTEXITCODE -ne 0) {
    throw "run-daily exited with code $LASTEXITCODE"
}
Write-Output ("=== stockdb run-daily finished at " + (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ") + " ===")
