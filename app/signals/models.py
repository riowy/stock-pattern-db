"""Daily research signals from PASSED/MONITORING patterns. No broker / orders."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.patterns.lifecycle import PatternStatus
from app.patterns.store import PatternResearchStore, dumps_json, new_id


class DailyPatternSignal(BaseModel):
    signal_date: date
    security_id: str
    ticker: str | None = None
    pattern_id: str
    pattern_version: int
    direction: str
    expected_horizon: int | None = None
    event_mode: str | None = None
    historical_sample_size: int | None = None
    hit_rate: float | None = None
    hit_rate_success_rule: str | None = None
    median_outcome: float | None = None
    mean_outcome: float | None = None
    typical_loss_when_wrong: float | None = None
    recent_rolling: dict[str, Any] = Field(default_factory=dict)
    generator_provenance_summary: list[str] = Field(default_factory=list)
    supporting_evaluation_ids: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class PatternApplicator(Protocol):
    """Apply PASSED/MONITORING patterns to the latest completed session."""

    def apply(self, session_date: date) -> list[DailyPatternSignal]: ...


class SignalService:
    def __init__(self, store: PatternResearchStore) -> None:
        self.store = store

    def active_patterns(self) -> list:
        patterns = self.store.list_patterns()
        return [p for p in patterns if p["status"] in (str(PatternStatus.PASSED), str(PatternStatus.MONITORING))]

    def persist_signals(self, signals: list[DailyPatternSignal]) -> int:
        """Persist only when store allows signal persistence."""
        for sig in signals:
            self.store.insert_signal(
                {
                    "signal_id": new_id("sig_"),
                    "signal_date": sig.signal_date,
                    "security_id": sig.security_id,
                    "ticker": sig.ticker,
                    "pattern_id": sig.pattern_id,
                    "pattern_version": sig.pattern_version,
                    "direction": sig.direction,
                    "expected_horizon": sig.expected_horizon,
                    "event_mode": sig.event_mode,
                    "historical_sample_size": sig.historical_sample_size,
                    "hit_rate": sig.hit_rate,
                    "hit_rate_success_rule": sig.hit_rate_success_rule,
                    "median_outcome": sig.median_outcome,
                    "mean_outcome": sig.mean_outcome,
                    "typical_loss_when_wrong": sig.typical_loss_when_wrong,
                    "recent_rolling_json": dumps_json(sig.recent_rolling) if sig.recent_rolling else None,
                    "generator_provenance_json": dumps_json(sig.generator_provenance_summary),
                    "evaluation_ids_json": dumps_json(sig.supporting_evaluation_ids),
                    "created_at": datetime.now(UTC),
                }
            )
        return len(signals)

    def list_for_date(self, session_date: date) -> list[dict[str, Any]]:
        return self.store.list_signals(session_date)


class NullApplicator:
    """Placeholder applicator — no live session scan in v1."""

    def apply(self, session_date: date) -> list[DailyPatternSignal]:
        _ = session_date
        return []
