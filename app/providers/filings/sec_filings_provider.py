"""SEC EDGAR filing metadata provider.

Source: https://data.sec.gov/submissions/CIK{10-digit}.json

Only *metadata* about filings is collected (form type, dates, accession
number, primary document path) -- no document text parsing/analysis in this
phase. Only the "recent" filings window is read for now; SEC also exposes a
paginated ``files`` array for older filings and a bulk daily-index archive
under https://www.sec.gov/Archives/edgar/full-index/ -- both are natural
extension points if/when a full historical backfill of filing metadata is
needed, without changing this provider's interface or the downstream
schema.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config.sec_filing_types import TRACKED_FORM_TYPES
from app.providers.base import FilingsProvider, ProviderCapabilities, RawFetchResult
from app.utils.atomic_io import atomic_write_bytes
from app.utils.ratelimit import RateLimiter

SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"


class SecFilingsProvider(FilingsProvider):
    capabilities = ProviderCapabilities(
        provider_name="sec_edgar_submissions",
        requires_api_key=False,
        commercial_use_safe=True,
        redistribution_safe=True,
        notes="Official SEC filing metadata (no document content).",
    )

    def __init__(self, settings) -> None:  # noqa: ANN001
        super().__init__(settings)
        if not settings.sec_user_agent.strip():
            raise ValueError(
                "SEC_USER_AGENT must be set in the environment before calling the SEC EDGAR API."
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

    def fetch_filings(self, cik: str) -> RawFetchResult:
        cik_padded = f"{int(cik):010d}"
        url = SUBMISSIONS_URL_TEMPLATE.format(cik=cik_padded)
        resp = self._get(url)
        retrieved_at = datetime.now(UTC)

        raw_dir = self.settings.raw_dir / "sec_filings"
        raw_dir.mkdir(parents=True, exist_ok=True)
        stamp = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
        raw_path = raw_dir / f"CIK{cik_padded}_{stamp}.json"
        atomic_write_bytes(raw_path, resp.content)

        payload = resp.json()
        recent = payload.get("filings", {}).get("recent", {})
        n = len(recent.get("accessionNumber", []))

        rows = []
        for i in range(n):
            form_type = recent.get("form", [None] * n)[i]
            if form_type not in TRACKED_FORM_TYPES:
                continue
            accession = recent.get("accessionNumber", [None] * n)[i]
            primary_doc = recent.get("primaryDocument", [None] * n)[i]
            accession_no_dashes = accession.replace("-", "") if accession else None
            source_url = None
            if accession_no_dashes and primary_doc:
                source_url = (
                    f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                    f"{accession_no_dashes}/{primary_doc}"
                )
            rows.append(
                {
                    "cik": cik_padded,
                    "accession_number": accession,
                    "form_type": form_type,
                    "filing_date": recent.get("filingDate", [None] * n)[i],
                    "accepted_at": recent.get("acceptanceDateTime", [None] * n)[i],
                    "report_date": recent.get("reportDate", [None] * n)[i] or None,
                    "primary_document": primary_doc,
                    "source_url": source_url,
                }
            )

        return RawFetchResult(
            rows=rows,
            provider=self.capabilities.provider_name,
            retrieved_at=retrieved_at,
            source_url=url,
            request_key=cik_padded,
            extra={"raw_file_path": str(raw_path)},
        )
