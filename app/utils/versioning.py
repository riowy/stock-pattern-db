"""calculation_code_version helper.

Git is optional: missing git / missing repo must never fail a calculation.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GIT_CANDIDATES = (
    "git",
    r"C:\Program Files\Git\cmd\git.exe",
    r"C:\Program Files\Git\bin\git.exe",
)


def get_calculation_code_version() -> str:
    """Return a short git commit hash, or ``unknown`` if git is unavailable."""
    for git_exe in _GIT_CANDIDATES:
        try:
            result = subprocess.run(
                [git_exe, "rev-parse", "HEAD"],
                cwd=_REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0:
            digest = result.stdout.strip()
            if digest:
                return digest[:12]
    return "unknown"
