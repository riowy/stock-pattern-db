"""Central application configuration.

All tunables come from environment variables (see ``.env.example``). Nothing
here should hardcode API keys, absolute machine-specific paths, or timezone
assumptions -- see AGENTS-level requirements in the project README.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- external API credentials -------------------------------------------------
    sec_user_agent: str = Field(
        default="",
        description="Required contact string for SEC EDGAR requests, e.g. "
        "'StockPatternResearch you@example.com'.",
    )
    fred_api_key: str = Field(default="", description="FRED API key (free).")

    # --- licensing / commercial-use safety -----------------------------------------
    commercial_mode: bool = Field(
        default=False,
        description="When True, providers not marked commercial_use_safe are refused.",
    )

    # --- provider selection ----------------------------------------------------------
    price_provider: str = Field(default="yfinance")

    # --- resource limits ---------------------------------------------------------
    max_workers: int = Field(default=2, ge=1, le=32)
    price_batch_size: int = Field(default=25, ge=1, le=1000)
    feature_batch_size: int = Field(default=50, ge=1, le=1000)
    max_feature_lookback_sessions: int = Field(default=300, ge=50, le=2000)
    label_recompute_sessions: int = Field(default=30, ge=20, le=120)
    price_request_pause_seconds: float = Field(default=0.0, ge=0, le=10)
    daily_price_lookback_sessions: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Trading sessions before a stale name's last stored session to re-fetch on daily sync.",
    )
    price_repair_lookback_sessions: int = Field(
        default=20,
        ge=5,
        le=60,
        description="Trading sessions re-fetched by weekly price repair, ending at expected latest.",
    )
    indicator_persistence_enabled: bool = Field(
        default=False,
        description="Must stay false in v1. Indicator results are memory-only.",
    )
    mining_result_persistence_enabled: bool = Field(
        default=False,
        description="Must stay false in v1. Mining results are memory-only / CLI output.",
    )
    pattern_registry_persistence_enabled: bool = Field(
        default=False,
        description="Independent gate for pattern-research registry DuckDB. OFF by default.",
    )
    daily_signal_persistence_enabled: bool = Field(
        default=False,
        description="Independent gate for daily research signal writes. OFF by default.",
    )
    duckdb_threads: int = Field(default=6, ge=1, le=64)
    duckdb_memory_limit: str = Field(default="24GB")

    # --- SEC rate limiting ---------------------------------------------------------
    sec_requests_per_second: float = Field(default=3.0, gt=0, le=10.0)

    # --- market calendar -------------------------------------------------------------
    market_calendar: str = Field(default="XNYS", description="exchange_calendars calendar code.")
    market_data_grace_minutes: int = Field(
        default=120,
        ge=0,
        description="Minutes after market close before today's session is assumed to have "
        "provider EOD data available.",
    )

    # --- Parquet partition health / compaction thresholds --------------------------
    # Applied to monthly-partitioned datasets (prices_daily, features_daily,
    # labels_forward_returns, filings, short_volume): a monthly partition
    # naturally accumulates more files at a faster clip.
    compact_file_count_threshold: int = Field(default=25, ge=1)
    compact_avg_file_size_mb: float = Field(default=8.0, gt=0)

    # Applied to yearly-partitioned datasets (volatility, corporate_actions):
    # a lower file-count bar, since a whole year's worth of small appends
    # sitting in one partition is worse than the monthly-dataset equivalent.
    compact_yearly_file_count_threshold: int = Field(default=12, ge=1)
    compact_yearly_avg_file_size_mb: float = Field(default=2.0, gt=0)

    # --- data directories -----------------------------------------------------------
    data_root: Path = Field(default=Path("./data"))
    raw_dir: Path = Field(default=Path("./data/raw"))
    lake_dir: Path = Field(default=Path("./data/lake"))
    state_dir: Path = Field(default=Path("./data/state"))
    log_dir: Path = Field(default=Path("./data/logs"))

    # --- logging ---------------------------------------------------------------------
    log_level: str = Field(default="INFO")

    @field_validator("sec_requests_per_second")
    @classmethod
    def _cap_sec_rate(cls, v: float) -> float:
        # Hard safety ceiling regardless of what a user puts in .env.
        return min(v, 10.0)

    # --- derived paths -----------------------------------------------------------------
    @property
    def duckdb_path(self) -> Path:
        return self.state_dir / "catalog.duckdb"

    @property
    def pattern_registry_path(self) -> Path:
        """Isolated research-registry store. Not the price lake / catalog."""
        return self.state_dir / "pattern_registry.duckdb"

    @property
    def checkpoints_dir(self) -> Path:
        return self.state_dir / "checkpoints"

    @property
    def prices_daily_dir(self) -> Path:
        return self.lake_dir / "prices_daily"

    @property
    def corporate_actions_dir(self) -> Path:
        return self.lake_dir / "corporate_actions"

    @property
    def macro_dir(self) -> Path:
        return self.lake_dir / "macro"

    @property
    def volatility_dir(self) -> Path:
        return self.lake_dir / "volatility"

    @property
    def filings_dir(self) -> Path:
        return self.lake_dir / "filings"

    @property
    def short_volume_dir(self) -> Path:
        return self.lake_dir / "short_volume"

    @property
    def features_daily_dir(self) -> Path:
        return self.lake_dir / "features_daily"

    @property
    def labels_forward_returns_dir(self) -> Path:
        return self.lake_dir / "labels_forward_returns"

    @property
    def research_dir(self) -> Path:
        return self.data_root / "research"

    @property
    def sec_user_agent_configured(self) -> bool:
        return bool(self.sec_user_agent.strip())

    @property
    def fred_api_key_configured(self) -> bool:
        return bool(self.fred_api_key.strip())

    def ensure_directories(self) -> None:
        """Create the standard directory skeleton if it does not exist yet."""
        for path in (
            self.raw_dir / "sec",
            self.raw_dir / "fred",
            self.raw_dir / "cboe",
            self.raw_dir / "yfinance",
            self.raw_dir / "sec_filings",
            self.prices_daily_dir,
            self.corporate_actions_dir,
            self.macro_dir,
            self.volatility_dir,
            self.filings_dir,
            self.short_volume_dir,
            self.features_daily_dir,
            self.labels_forward_returns_dir,
            self.research_dir,
            self.state_dir,
            self.checkpoints_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Use ``get_settings.cache_clear()`` in tests."""
    return Settings()
