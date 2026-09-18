"""Provider abstraction layer.

This module is the single most important design boundary in the project:
business/analysis code must never talk to Yahoo Finance, SEC, FRED, or Cboe
directly. It talks to a ``PriceProvider`` / ``SecurityMasterProvider`` / ...
interface. Swapping the free bootstrap provider (yfinance) for a licensed
commercial provider (Massive, Tiingo, FMP, Sharadar, ...) later should only
require writing one new adapter class and flipping ``PRICE_PROVIDER`` in
``.env`` -- no changes to the database schema or downstream analysis code.

Every provider declares ``ProviderCapabilities`` metadata, most importantly
``commercial_use_safe``. ``guard_commercial_mode`` enforces the licensing
safety net described in the spec ("데이터 라이선스 안전장치"): if the app is
running with ``COMMERCIAL_MODE=true``, any provider not explicitly marked
safe for commercial use refuses to run with a clear error instead of
silently returning data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

from app.config.settings import Settings


class ProviderCapabilities(BaseModel):
    provider_name: str
    requires_api_key: bool = False
    commercial_use_safe: bool = False
    redistribution_safe: bool = False
    notes: str = ""


class ProviderNotAllowedError(RuntimeError):
    """Raised when a provider is not allowed to run under the current mode."""


def guard_commercial_mode(capabilities: ProviderCapabilities, settings: Settings) -> None:
    if settings.commercial_mode and not capabilities.commercial_use_safe:
        raise ProviderNotAllowedError(
            "This provider is configured for research/personal use only.\n"
            "Choose a licensed commercial provider before running in commercial mode.\n"
            f"(provider={capabilities.provider_name}, "
            f"commercial_use_safe={capabilities.commercial_use_safe})"
        )


@dataclass
class RawFetchResult:
    """Generic wrapper for "whatever the provider's API returned", tagged
    with the provenance fields every downstream normalizer needs."""

    rows: list[dict[str, Any]]
    provider: str
    retrieved_at: datetime
    source_url: str | None = None
    request_key: str | None = None  # e.g. the ticker/series/cik this fetch was for
    extra: dict[str, Any] = field(default_factory=dict)


class BaseProvider(ABC):
    capabilities: ProviderCapabilities

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        guard_commercial_mode(self.capabilities, settings)


class SecurityMasterProvider(BaseProvider):
    """Provides the universe of tradable securities + identifiers."""

    @abstractmethod
    def fetch_universe(self) -> RawFetchResult:
        """Return raw security-master rows (provider-native column names)."""


class PriceProvider(BaseProvider):
    """Provides daily (and, in the future, intraday) OHLCV bars."""

    @abstractmethod
    def fetch_daily_bars(self, provider_symbol: str, start: date, end: date | None = None) -> RawFetchResult:
        """Return raw daily bar rows for one provider-native symbol."""

    def to_provider_symbol(self, canonical_symbol: str) -> str:
        """Default: no translation. Override for providers with quirky spelling."""
        return canonical_symbol


class MacroProvider(BaseProvider):
    """Provides macroeconomic time series (e.g. FRED)."""

    @abstractmethod
    def fetch_series(self, series_id: str, start: date | None = None) -> RawFetchResult:
        """Return raw observations for one series."""


class VolatilityProvider(BaseProvider):
    """Provides volatility index data (e.g. Cboe VIX)."""

    @abstractmethod
    def fetch_history(self) -> RawFetchResult:
        """Return the full (or incremental) historical volatility series."""


class FilingsProvider(BaseProvider):
    """Provides regulatory filing metadata (e.g. SEC EDGAR submissions)."""

    @abstractmethod
    def fetch_filings(self, cik: str) -> RawFetchResult:
        """Return raw filing metadata rows for one CIK."""


class ShortVolumeProvider(BaseProvider):
    """Provides short-sale volume data (e.g. FINRA).

    Not required for v1. This interface exists so the schema/architecture
    already has a slot for it; a concrete implementation should only be
    activated after confirming FINRA's terms of use permit the intended use,
    and only while ``COMMERCIAL_MODE=false`` unless a properly licensed
    provider is substituted.
    """

    @abstractmethod
    def fetch_short_volume(self, provider_symbol: str, on_date: date) -> RawFetchResult:
        """Return raw short-volume rows for one symbol/date."""
