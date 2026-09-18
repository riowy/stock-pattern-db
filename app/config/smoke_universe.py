"""Small fixed universe used for smoke-testing the pipeline end to end.

Per project requirements, full-market backfills must never happen implicitly.
This list is only a convenience default for manual smoke testing; real runs
should pass explicit ``--symbols`` or go through the deliberate universe
expansion steps described in the README (10 -> 100 -> 500 -> full).
"""

from __future__ import annotations

SMOKE_TEST_TICKERS: list[str] = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "BRK-B",
    "JPM",
    "XOM",
    "SPY",
]
