"""Logging setup: Rich console handler + rotating file handler."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

_CONFIGURED = False


def setup_logging(log_dir: Path, level: str = "INFO") -> logging.Logger:
    global _CONFIGURED
    root = logging.getLogger("stockdb")
    if _CONFIGURED:
        return root

    root.setLevel(level.upper())
    root.propagate = False

    console_handler = RichHandler(rich_tracebacks=True, show_path=False)
    console_handler.setLevel(level.upper())

    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "stockdb.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    file_handler.setLevel(level.upper())

    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)
    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"stockdb.{name}")
