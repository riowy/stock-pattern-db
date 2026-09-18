"""Provider factory / registry.

This is the ONE place that knows how a ``PRICE_PROVIDER`` string maps to a
concrete adapter class. Everything else in the codebase depends only on the
``PriceProvider`` interface. Adding a licensed provider later (Massive,
Tiingo, FMP, Sharadar, ...) means: write ``app/providers/price/xxx.py``
implementing ``PriceProvider``, register it here, then set
``PRICE_PROVIDER=xxx`` in ``.env``. No other code changes required.
"""

from __future__ import annotations

from app.config.settings import Settings
from app.providers.base import PriceProvider
from app.providers.price.yfinance_provider import YFinancePriceProvider

_PRICE_PROVIDERS: dict[str, type[PriceProvider]] = {
    "yfinance": YFinancePriceProvider,
    # Future: "massive": MassiveProvider, "tiingo": TiingoProvider, ...
}


def get_price_provider(name: str, settings: Settings) -> PriceProvider:
    try:
        cls = _PRICE_PROVIDERS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_PRICE_PROVIDERS))
        raise ValueError(
            f"Unknown PRICE_PROVIDER '{name}'. Available: {available}. "
            "Implement a new app.providers.base.PriceProvider subclass and "
            "register it in app/providers/registry.py to add another one."
        ) from exc
    return cls(settings)
