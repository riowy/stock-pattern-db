"""Yahoo Finance (via the ``yfinance`` package) daily price provider.

IMPORTANT LICENSING NOTE
-------------------------
Yahoo/yfinance data is unofficial, undocumented, and intended here strictly
for **personal research and prototyping**. It must NOT be used as the basis
of a commercial product or redistributed as a data product. That is why
``capabilities.commercial_use_safe = False`` and ``redistribution_safe =
False`` below -- if ``COMMERCIAL_MODE=true`` is ever set, this provider will
refuse to run (see ``app.providers.base.guard_commercial_mode``) and you
must switch ``PRICE_PROVIDER`` to a properly licensed vendor (Massive,
Tiingo, FMP, Sharadar, etc.) implementing the same ``PriceProvider``
interface.

We deliberately request ``auto_adjust=False`` (per spec) so raw OHLC and the
Yahoo-computed ``Adj Close`` are both preserved, and corporate actions
(dividends / splits) are captured as returned by the API rather than baked
silently into the price series.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config.symbol_overrides import SYMBOL_OVERRIDE_SEED
from app.providers.base import PriceProvider, ProviderCapabilities, RawFetchResult
from app.utils.atomic_io import atomic_write_bytes


class YFinancePriceProvider(PriceProvider):
    capabilities = ProviderCapabilities(
        provider_name="yfinance",
        requires_api_key=False,
        commercial_use_safe=False,
        redistribution_safe=False,
        notes=(
            "Yahoo/yfinance data is for personal research and prototyping only. "
            "Review licensing and switch to a licensed market-data provider before "
            "any commercial use or data redistribution."
        ),
    )

    def to_provider_symbol(self, canonical_symbol: str) -> str:
        override = SYMBOL_OVERRIDE_SEED.get(canonical_symbol, {}).get("yfinance")
        if override:
            return override
        # yfinance convention: share classes use '-' instead of '.'
        return canonical_symbol.replace(".", "-")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
    def _download(self, provider_symbol: str, start: date, end: date | None) -> pd.DataFrame:
        ticker = yf.Ticker(provider_symbol)
        return ticker.history(
            start=start.isoformat(),
            end=end.isoformat() if end else None,
            auto_adjust=False,
            actions=True,
            raise_errors=True,
        )

    def fetch_daily_bars(self, provider_symbol: str, start: date, end: date | None = None) -> RawFetchResult:
        retrieved_at = datetime.now(UTC)
        df = self._download(provider_symbol, start, end)

        raw_dir = self.settings.raw_dir / "yfinance"
        raw_dir.mkdir(parents=True, exist_ok=True)
        stamp = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
        raw_path = raw_dir / f"{provider_symbol}_{stamp}.csv"

        if df.empty:
            atomic_write_bytes(raw_path, df.to_csv().encode("utf-8"))
            return RawFetchResult(
                rows=[],
                provider=self.capabilities.provider_name,
                retrieved_at=retrieved_at,
                source_url=f"yfinance://{provider_symbol}",
                request_key=provider_symbol,
                extra={"raw_file_path": str(raw_path), "empty": True},
            )

        raw_df = df.reset_index()
        # yfinance names the index column "Date" (or "Datetime" for intraday).
        date_col = raw_df.columns[0]
        raw_df = raw_df.rename(columns={date_col: "Date"})
        atomic_write_bytes(raw_path, raw_df.to_csv(index=False).encode("utf-8"))

        rows: list[dict] = []
        for record in raw_df.to_dict(orient="records"):
            trading_date = pd.Timestamp(record["Date"]).date().isoformat()
            rows.append(
                {
                    "date": trading_date,
                    "open": record.get("Open"),
                    "high": record.get("High"),
                    "low": record.get("Low"),
                    "close": record.get("Close"),
                    "adj_close": record.get("Adj Close"),
                    "volume": record.get("Volume"),
                    "dividend": record.get("Dividends", 0.0) or 0.0,
                    "stock_split": record.get("Stock Splits", 0.0) or 0.0,
                }
            )

        return RawFetchResult(
            rows=rows,
            provider=self.capabilities.provider_name,
            retrieved_at=retrieved_at,
            source_url=f"yfinance://{provider_symbol}",
            request_key=provider_symbol,
            extra={"raw_file_path": str(raw_path)},
        )
