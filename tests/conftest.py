from __future__ import annotations

from pathlib import Path

import pytest

from app.config.settings import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(
        _env_file=None,
        sec_user_agent="StockPatternResearchTests test@example.com",
        fred_api_key="test-fred-key",
        commercial_mode=False,
        price_provider="yfinance",
        data_root=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        lake_dir=tmp_path / "data" / "lake",
        state_dir=tmp_path / "data" / "state",
        log_dir=tmp_path / "data" / "logs",
    )
    s.ensure_directories()
    return s
