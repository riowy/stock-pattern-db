"""FRED (Federal Reserve Economic Data) macro series provider.

Free API, requires a free API key (``FRED_API_KEY``). FRED series are
public-domain US government statistics; attribution is requested by FRED's
terms but the data itself is safe for both research and commercial use.

The response schema already carries ``realtime_start``/``realtime_end`` so
this same provider can be extended to pull ALFRED vintage data later without
a schema change (see ``app.models.macro.MacroObservation``).
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.providers.base import MacroProvider, ProviderCapabilities, RawFetchResult
from app.utils.atomic_io import atomic_write_bytes
from app.utils.ratelimit import RateLimiter

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"


class FredMacroProvider(MacroProvider):
    capabilities = ProviderCapabilities(
        provider_name="fred",
        requires_api_key=True,
        commercial_use_safe=True,
        redistribution_safe=True,
        notes="Public-domain US government economic data. Attribution to FRED/ALFRED is requested.",
    )

    def __init__(self, settings) -> None:  # noqa: ANN001
        super().__init__(settings)
        if not settings.fred_api_key.strip():
            raise ValueError(
                "FRED_API_KEY must be set. Get a free key at "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        self._rate_limiter = RateLimiter(requests_per_second=2.0)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    )
    def _get(self, params: dict) -> httpx.Response:
        self._rate_limiter.wait()
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(FRED_OBSERVATIONS_URL, params=params)
            resp.raise_for_status()
            return resp

    def fetch_series(self, series_id: str, start: date | None = None) -> RawFetchResult:
        params = {
            "series_id": series_id,
            "api_key": self.settings.fred_api_key,
            "file_type": "json",
        }
        if start is not None:
            params["observation_start"] = start.isoformat()

        resp = self._get(params)
        retrieved_at = datetime.now(UTC)

        raw_dir = self.settings.raw_dir / "fred"
        raw_dir.mkdir(parents=True, exist_ok=True)
        stamp = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
        raw_path = raw_dir / f"{series_id}_{stamp}.json"
        atomic_write_bytes(raw_path, resp.content)

        payload = resp.json()
        rows = []
        for obs in payload.get("observations", []):
            raw_value = obs.get("value")
            value = None if raw_value in (None, ".", "") else float(raw_value)
            rows.append(
                {
                    "series_id": series_id,
                    "date": obs.get("date"),
                    "value": value,
                    "realtime_start": obs.get("realtime_start"),
                    "realtime_end": obs.get("realtime_end"),
                }
            )

        return RawFetchResult(
            rows=rows,
            provider=self.capabilities.provider_name,
            retrieved_at=retrieved_at,
            source_url=FRED_OBSERVATIONS_URL,
            request_key=series_id,
            extra={"raw_file_path": str(raw_path)},
        )
