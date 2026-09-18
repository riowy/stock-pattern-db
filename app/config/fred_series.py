"""Seed list of FRED macro series to track.

Extend this dict freely -- ``series_id`` is looked up directly against the
FRED API, so adding a new line here is enough to start collecting a new
series on the next ``stockdb sync-macro`` run.
"""

from __future__ import annotations

# series_id -> human readable description (for logs / status reporting only)
FRED_SEED_SERIES: dict[str, str] = {
    "DFF": "Federal Funds Effective Rate (daily)",
    "FEDFUNDS": "Federal Funds Effective Rate (monthly)",
    "DGS2": "2-Year Treasury Constant Maturity Rate",
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "T10Y2Y": "10Y-2Y Treasury Yield Spread",
    "BAMLH0A0HYM2": "ICE BofA US High Yield Index Option-Adjusted Spread",
    "DCOILWTICO": "WTI Crude Oil Price",
    "DTWEXBGS": "Trade Weighted US Dollar Index (Broad, Goods & Services)",
}
