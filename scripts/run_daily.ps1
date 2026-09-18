# Daily pipeline runner for Windows Task Scheduler.
#
# Example: create a Task Scheduler task that runs
#   powershell.exe -ExecutionPolicy Bypass -File "C:\path\to\stock-pattern-db\scripts\run_daily.ps1"
# on a daily trigger (pick whatever local time your machine uses -- this
# script does not hardcode any time zone).

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

Set-Location $ProjectDir

$VenvActivate = Join-Path $ProjectDir ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    & $VenvActivate
}

Write-Output ("=== stockdb run-daily started at " + (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ") + " ===")
stockdb run-daily
Write-Output ("=== stockdb run-daily finished at " + (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ") + " ===")
