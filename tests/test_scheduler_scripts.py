"""Windows Task Scheduler scripts: Korean/space paths and invocation."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_run_daily_ps1_uses_uv_python_module_not_stockdb_exe() -> None:
    text = (ROOT / "scripts" / "run_daily.ps1").read_text(encoding="utf-8")
    assert "Set-Location -LiteralPath $ProjectDir" in text
    assert "Resolve-Path -LiteralPath" in text
    invoke_lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#") and not ln.lstrip().startswith("Write-Output")
    ]
    invoke = "\n".join(invoke_lines)
    assert "python -m app.cli.main run-daily" in invoke
    assert "stockdb.exe" not in invoke
    assert not any(ln.strip() == "stockdb run-daily" or ln.strip().endswith(" stockdb run-daily") for ln in invoke_lines)


def test_register_daily_task_ps1_quotes_runner_and_sets_working_directory() -> None:
    text = (ROOT / "scripts" / "register_daily_task.ps1").read_text(encoding="utf-8")
    assert "Resolve-Path -LiteralPath" in text
    assert "WorkingDirectory $ProjectDir" in text
    assert '-File `"$Runner`"' in text
    assert "Test-Path -LiteralPath $Runner" in text
    example = r"C:\Users\cheye\주식 분석\stock-pattern-db"
    # Script must not assume an ASCII-only path; LiteralPath is the contract.
    assert "LiteralPath" in text
    assert " " in example and any(ord(ch) > 127 for ch in example)


def test_run_weekly_price_repair_ps1_uses_uv_python_module_not_stockdb_exe() -> None:
    text = (ROOT / "scripts" / "run_weekly_price_repair.ps1").read_text(encoding="utf-8")
    assert "Set-Location -LiteralPath $ProjectDir" in text
    assert "Resolve-Path -LiteralPath" in text
    invoke_lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#") and not ln.lstrip().startswith("Write-Output")
    ]
    invoke = "\n".join(invoke_lines)
    assert "python -m app.cli.main repair-prices" in invoke
    assert "stockdb.exe" not in invoke
    assert "schtasks" not in text.lower()
    assert not any(ln.strip() == "stockdb repair-prices" or ln.strip().endswith(" stockdb repair-prices") for ln in invoke_lines)


def test_register_script_does_not_hardcode_timezone() -> None:
    text = (ROOT / "scripts" / "register_daily_task.ps1").read_text(encoding="utf-8")
    assert "Asia/Seoul" in text or "does not hardcode" in text.lower()
    assert "STOCKDB_DAILY_TIME" in text
    assert "machine local" in text.lower() or "local timezone" in text.lower()
