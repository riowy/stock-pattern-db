"""Terminal progress for full-discover walk-forward. Does not affect mining results."""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console


def format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60}m {total % 60}s"


class WalkForwardLogger:
    """Plain-text fold logs plus a live overall elapsed/fold/phase line."""

    def __init__(self, console: Console, n_folds: int, *, show_status: bool = True) -> None:
        self.console = console
        self.n_folds = n_folds
        self.t0 = time.monotonic()
        self.fold_t0 = self.t0
        self.fold_i = 0
        self.phase = "starting"
        self._status = None
        if show_status:
            self._status = console.status(self._status_text(), spinner="line")
            self._status.start()

    def close(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def _elapsed(self) -> str:
        return format_elapsed(time.monotonic() - self.t0)

    def _status_text(self) -> str:
        return (
            f"elapsed: {self._elapsed()} | "
            f"current fold: {self.fold_i}/{self.n_folds} | "
            f"current phase: {self.phase}"
        )

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        if self._status is not None:
            self._status.update(self._status_text())

    def _fold_line(self, message: str) -> None:
        self.console.print(f"[Fold {self.fold_i}/{self.n_folds}] {message}", markup=False)

    def __call__(self, event: dict[str, Any]) -> None:
        etype = event.get("event")
        if etype == "fold_start":
            self.fold_i = int(event.get("fold") or 0)
            self.n_folds = int(event.get("n_folds") or self.n_folds)
            self.fold_t0 = time.monotonic()
            self._set_phase("loading indicators")
            self._fold_line("loading indicators...")
            return
        if etype == "fold_elapsed":
            self._set_phase("fold complete")
            self.console.print(f"elapsed: {format_elapsed(time.monotonic() - self.fold_t0)}", markup=False)
            self.console.print(f"elapsed: {self._elapsed()}", markup=False)
            self.console.print(f"current fold: {self.fold_i}/{self.n_folds}", markup=False)
            self.console.print(f"current phase: {self.phase}", markup=False)
            return
        if etype == "skipped":
            self.fold_i = int(event.get("fold") or self.fold_i)
            self._set_phase("skipped")
            self._fold_line(f"skipped ({event.get('reason', '')})")
            return

        phase = str(event.get("phase") or "")
        if phase == "generating states":
            self._set_phase(phase)
            self._fold_line("generating states...")
        elif phase == "single candidates":
            self._set_phase(phase)
            self._fold_line(f"single candidates: {event.get('n', 0)}")
        elif phase == "supported singles":
            self._set_phase(phase)
            self._fold_line(f"supported singles: {event.get('n', 0)}")
        elif phase == "pair candidates":
            self._set_phase(phase)
            self._fold_line(f"pair candidates: {event.get('n', 0)}")
        elif phase == "evaluating pairs":
            done = event.get("done", 0)
            total = event.get("total", 0)
            self._set_phase("evaluating pairs")
            self._fold_line(f"evaluating pairs: {done}/{total}")
        elif phase == "FDR":
            self._set_phase("FDR")
            self._fold_line("FDR...")
        elif phase == "selected":
            self._set_phase("selected")
            self._fold_line(f"selected: {event.get('n', 0)}")
        elif phase == "selection_diagnostics":
            self._set_phase("selection diagnostics")
            from app.mining.selection_diag import format_selection_lines

            for line in format_selection_lines(event.get("diagnostics") or {}):
                self.console.print(line, markup=False)
        elif phase == "evaluation complete":
            self._set_phase("evaluation complete")
            self._fold_line("evaluation complete")
        else:
            self._set_phase(phase or self.phase)
