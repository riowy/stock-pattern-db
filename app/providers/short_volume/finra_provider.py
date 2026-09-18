"""FINRA short-sale volume provider -- INTERFACE/SCHEMA ONLY (spec section 4-F).

This is intentionally NOT a working implementation in v1. Per the project
spec, FINRA short-volume data is not required for the initial data
foundation, and its terms of use need to be reviewed before any automated
collection begins. Wiring this up later should mean: implement
``fetch_short_volume`` here (respecting FINRA's rate limits and terms), then
register it wherever short-volume ingestion is added -- no changes to
``ShortVolumeProvider`` or the rest of the pipeline.

Safety notes:
* Keep ``COMMERCIAL_MODE=false`` while using this data for personal
  research. ``capabilities.commercial_use_safe=False`` below means this
  provider (once implemented) will refuse to run at all if
  ``COMMERCIAL_MODE=true`` (see ``app.providers.base.guard_commercial_mode``),
  forcing an explicit, informed decision before any commercial use.
* Verify FINRA's current data-use terms at
  https://www.finra.org/finra-data before enabling real fetches.
"""

from __future__ import annotations

from datetime import date

from app.providers.base import ProviderCapabilities, RawFetchResult, ShortVolumeProvider


class FinraShortVolumeProvider(ShortVolumeProvider):
    capabilities = ProviderCapabilities(
        provider_name="finra_short_volume",
        requires_api_key=False,
        commercial_use_safe=False,
        redistribution_safe=False,
        notes=(
            "NOT IMPLEMENTED in v1. Review FINRA's terms of use before "
            "implementing fetch_short_volume(); keep COMMERCIAL_MODE=false "
            "until a properly licensed data path is confirmed."
        ),
    )

    def fetch_short_volume(self, provider_symbol: str, on_date: date) -> RawFetchResult:
        raise NotImplementedError(
            "FinraShortVolumeProvider is a schema/interface placeholder only. "
            "Review FINRA's terms of use, then implement this method before use."
        )
