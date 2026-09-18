"""Derive corporate action rows from a raw price fetch.

Per spec section 8, corporate actions must be kept in their own dataset and
never implicitly merged into OHLC price bars. ``yfinance`` happens to return
dividend/split information alongside daily bars, so we derive
``corporate_actions`` rows from the same fetch as a convenience -- the raw
OHLC and the action records are still written to two completely separate
Parquet datasets, never blended into one row.
"""

from __future__ import annotations

import polars as pl

from app.models.corporate_action import ActionType, CorporateAction
from app.providers.base import RawFetchResult

CORPORATE_ACTIONS_SCHEMA = {
    "security_id": pl.Utf8,
    "effective_date": pl.Date,
    "action_type": pl.Utf8,
    "value": pl.Float64,
    "provider": pl.Utf8,
    "retrieved_at": pl.Datetime(time_zone="UTC"),
}


def derive_corporate_actions(fetch: RawFetchResult, security_id: str) -> pl.DataFrame:
    if not fetch.rows:
        return pl.DataFrame(schema=CORPORATE_ACTIONS_SCHEMA)

    actions: list[dict] = []
    for row in fetch.rows:
        dividend = row.get("dividend") or 0.0
        split = row.get("stock_split") or 0.0
        if dividend:
            actions.append(
                CorporateAction(
                    security_id=security_id,
                    effective_date=row["date"],
                    action_type=ActionType.DIVIDEND,
                    value=dividend,
                    provider=fetch.provider,
                    retrieved_at=fetch.retrieved_at,
                ).model_dump()
            )
        if split:
            actions.append(
                CorporateAction(
                    security_id=security_id,
                    effective_date=row["date"],
                    action_type=ActionType.SPLIT,
                    value=split,
                    provider=fetch.provider,
                    retrieved_at=fetch.retrieved_at,
                ).model_dump()
            )

    if not actions:
        return pl.DataFrame(schema=CORPORATE_ACTIONS_SCHEMA)

    df = pl.DataFrame(actions, schema=CORPORATE_ACTIONS_SCHEMA)
    return df.sort(["security_id", "effective_date", "action_type"]).unique(
        subset=["security_id", "effective_date", "action_type"], keep="last"
    )
