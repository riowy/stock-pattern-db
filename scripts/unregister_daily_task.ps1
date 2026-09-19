# Remove the stockdb daily Task Scheduler entry.
# Typically requires an elevated PowerShell session.

$ErrorActionPreference = "Stop"
$TaskName = "StockPatternDbDaily"

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Output "Unregistered scheduled task '$TaskName'."
