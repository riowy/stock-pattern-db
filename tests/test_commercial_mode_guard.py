from __future__ import annotations

import pytest

from app.providers.base import ProviderCapabilities, ProviderNotAllowedError, guard_commercial_mode
from app.providers.price.yfinance_provider import YFinancePriceProvider


def test_guard_allows_commercial_safe_provider_in_commercial_mode(settings) -> None:
    caps = ProviderCapabilities(provider_name="safe_provider", commercial_use_safe=True)
    settings.commercial_mode = True
    guard_commercial_mode(caps, settings)  # should not raise


def test_guard_blocks_unsafe_provider_in_commercial_mode(settings) -> None:
    caps = ProviderCapabilities(provider_name="unsafe_provider", commercial_use_safe=False)
    settings.commercial_mode = True
    with pytest.raises(ProviderNotAllowedError):
        guard_commercial_mode(caps, settings)


def test_guard_allows_unsafe_provider_in_research_mode(settings) -> None:
    caps = ProviderCapabilities(provider_name="unsafe_provider", commercial_use_safe=False)
    settings.commercial_mode = False
    guard_commercial_mode(caps, settings)  # should not raise


def test_yfinance_provider_refuses_to_construct_in_commercial_mode(settings) -> None:
    settings.commercial_mode = True
    with pytest.raises(ProviderNotAllowedError):
        YFinancePriceProvider(settings)


def test_yfinance_provider_constructs_fine_in_research_mode(settings) -> None:
    settings.commercial_mode = False
    provider = YFinancePriceProvider(settings)
    assert provider.capabilities.commercial_use_safe is False
