"""Cboe official historical VIX data provider.

Source: https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv

This URL is Cboe-operated infrastructure, not a stable, versioned public API,
so it is kept as its own small, easily-swappable module: if Cboe changes the
path (it has moved before), only this one file needs to change.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.providers.base import ProviderCapabilities, RawFetchResult, VolatilityProvider
from app.utils.atomic_io import atomic_write_bytes

VIX_HISTORY_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


class CboeVixProvider(VolatilityProvider):
    capabilities = ProviderCapabilities(
        provider_name="cboe_vix",
        requires_api_key=False,
        commercial_use_safe=True,
        redistribution_safe=True,
        notes="Official Cboe historical VIX data, freely published.",
    )

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    )
    def _get(self) -> httpx.Response:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            resp = client.get(VIX_HISTORY_URL)
            resp.raise_for_status()
            return resp

    def fetch_history(self) -> RawFetchResult:
        resp = self._get()
        retrieved_at = datetime.now(UTC)

        raw_dir = self.settings.raw_dir / "cboe"
        raw_dir.mkdir(parents=True, exist_ok=True)
        stamp = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
        raw_path = raw_dir / f"VIX_History_{stamp}.csv"
        atomic_write_bytes(raw_path, resp.content)

        lines = resp.text.splitlines()
        if not lines:
            return RawFetchResult(rows=[], provider=self.capabilities.provider_name, retrieved_at=retrieved_at)

        header = [h.strip().upper() for h in lines[0].split(",")]
        rows = []
        for line in lines[1:]:
            if not line.strip():
                continue
            values = line.split(",")
            record = dict(zip(header, values, strict=False))
            month, day, year = record["DATE"].split("/")
            iso_date = f"{year}-{int(month):02d}-{int(day):02d}"
            rows.append(
                {
                    "date": iso_date,
                    "open": _to_float(record.get("OPEN")),
                    "high": _to_float(record.get("HIGH")),
                    "low": _to_float(record.get("LOW")),
                    "close": _to_float(record.get("CLOSE")),
                }
            )

        return RawFetchResult(
            rows=rows,
            provider=self.capabilities.provider_name,
            retrieved_at=retrieved_at,
            source_url=VIX_HISTORY_URL,
            extra={"raw_file_path": str(raw_path)},
        )


def _to_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None
