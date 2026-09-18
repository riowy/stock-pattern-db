"""Seed list of broad-market / sector ETFs.

The SEC ``company_tickers_exchange.json`` feed does not reliably classify
every ETF (asset_type, exchange linkage, etc.), so we track a small curated
seed list here. This list is intentionally editable -- add/remove tickers as
your research needs change. It does NOT need to be exhaustive; the security
master can always be extended with additional seed lists or providers later.
"""

from __future__ import annotations

# Broad market index ETFs
BROAD_MARKET_ETFS: list[str] = ["SPY", "QQQ", "IWM", "DIA"]

# S&P 500 sector SPDR ETFs
SECTOR_ETFS: list[str] = [
    "XLK",  # Technology
    "XLF",  # Financials
    "XLE",  # Energy
    "XLV",  # Health Care
    "XLI",  # Industrials
    "XLY",  # Consumer Discretionary
    "XLP",  # Consumer Staples
    "XLB",  # Materials
    "XLRE",  # Real Estate
    "XLU",  # Utilities
    "XLC",  # Communication Services
]

# Other frequently used research ETFs
OTHER_ETFS: list[str] = [
    "SMH",  # Semiconductors
    "HYG",  # High yield corporate bonds
    "LQD",  # Investment grade corporate bonds
]

SEED_ETF_TICKERS: list[str] = [*BROAD_MARKET_ETFS, *SECTOR_ETFS, *OTHER_ETFS]
