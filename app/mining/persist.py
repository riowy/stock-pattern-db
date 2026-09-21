"""Mining results are ephemeral. No lake / research / catalog writes."""

from __future__ import annotations

from app.mining.config import MINING_RESULT_PERSISTENCE_ENABLED


class MiningPersistenceDisabled(RuntimeError):
    pass


def assert_no_mining_persist() -> None:
    if MINING_RESULT_PERSISTENCE_ENABLED:
        raise MiningPersistenceDisabled("MINING_RESULT_PERSISTENCE_ENABLED is true but no writer exists.")


def persist_mining_results(*_a, **_k):  # noqa: ANN002, ANN003
    raise MiningPersistenceDisabled("Mining result persistence is disabled. CLI output only.")
