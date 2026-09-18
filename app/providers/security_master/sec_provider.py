"""SEC EDGAR security master provider.

Source: https://www.sec.gov/files/company_tickers_exchange.json

This is a free, official, no-API-key data source, but SEC requires a
descriptive ``User-Agent`` header on every request (``SEC_USER_AGENT`` env
var) and asks for conservative request rates. We never exceed
``SEC_REQUESTS_PER_SECOND`` (hard-capped at 10 req/s, default 3).

Known limitation (documented again in README): this feed does not perfectly
classify every ETF/asset type and does not contain historical delisted
tickers, so it is combined with a curated ETF seed list
(``app.config.etf_seed``) and is explicitly NOT a complete
survivorship-bias-free universe.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.providers.base import ProviderCapabilities, RawFetchResult, SecurityMasterProvider
from app.utils.atomic_io import atomic_write_bytes
from app.utils.ratelimit import RateLimiter

SEC_UNIVERSE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"


class SecUniverseProvider(SecurityMasterProvider):
    capabilities = ProviderCapabilities(
        provider_name="sec_company_tickers_exchange",
        requires_api_key=False,
        commercial_use_safe=True,
        redistribution_safe=True,
        notes=(
            "Official SEC data, free to use. Does not fully classify ETFs "
            "and has no historical delisting information -- see README "
            "'Security Master' limitations section."
        ),
    )

    def __init__(self, settings) -> None:  # noqa: ANN001
        super().__init__(settings)
        if not settings.sec_user_agent.strip():
            raise ValueError(
                "SEC_USER_AGENT must be set in the environment before calling the "
                "SEC EDGAR API, e.g. SEC_USER_AGENT='StockPatternResearch you@example.com'"
            )
        self._rate_limiter = RateLimiter(settings.sec_requests_per_second)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    )
    def _get(self, url: str) -> httpx.Response:
        self._rate_limiter.wait()
        headers = {"User-Agent": self.settings.sec_user_agent, "Accept-Encoding": "gzip, deflate"}
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            return resp

    def fetch_universe(self) -> RawFetchResult:
        resp = self._get(SEC_UNIVERSE_URL)
        retrieved_at = datetime.now(UTC)

        raw_dir = self.settings.raw_dir / "sec"
        raw_dir.mkdir(parents=True, exist_ok=True)
        stamp = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
        raw_path = raw_dir / f"company_tickers_exchange_{stamp}.json"
        atomic_write_bytes(raw_path, resp.content)

        payload = resp.json()
        fields = payload.get("fields", [])
        data = payload.get("data", [])
        rows = [dict(zip(fields, row, strict=False)) for row in data]

        return RawFetchResult(
            rows=rows,
            provider=self.capabilities.provider_name,
            retrieved_at=retrieved_at,
            source_url=SEC_UNIVERSE_URL,
            extra={"raw_file_path": str(raw_path)},
        )
