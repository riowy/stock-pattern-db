# Register the stockdb daily pipeline with Windows Task Scheduler.
#
# This script does NOT run automatically. Invoke it explicitly:
#   powershell.exe -ExecutionPolicy Bypass -File .\scripts\register_daily_task.ps1
#
# Creating a scheduled task typically requires an elevated (Administrator)
# PowerShell session.
#
# Schedule time comes from STOCKDB_DAILY_TIME (HH:mm, default 09:00) in
# whatever local timezone the machine uses. This script does not hardcode
# a timezone. For a machine in Asia/Seoul, 09:00 is well after the US
# cash-session close (16:00 America/New_York) plus provider EOD lag.

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$Runner = Join-Path $ScriptDir "run_daily.ps1"
$TaskName = "StockPatternDbDaily"

$Time = $env:STOCKDB_DAILY_TIME
if (-not $Time) {
    $Time = "09:00"
}

if (-not (Test-Path $Runner)) {
    throw "Daily runner not found: $Runner"
}

$Arg = "-NoProfile -ExecutionPolicy Bypass -File `"$Runner`""
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $Arg -WorkingDirectory $ProjectDir
$Trigger = New-ScheduledTaskTrigger -Daily -At $Time
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Force | Out-Null
Write-Output "Registered scheduled task '$TaskName' daily at $Time (machine local time)."
Write-Output "Project: $ProjectDir"
Write-Output "Unregister with: powershell.exe -ExecutionPolicy Bypass -File `"$ScriptDir\unregister_daily_task.ps1`""
