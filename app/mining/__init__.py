"""Ephemeral data-mining engine. Analysis discovers; validation only evaluates."""

from app.mining.config import MINING_RESULT_PERSISTENCE_ENABLED
from app.mining.engine import MiningEngine, MiningRun

__all__ = ["MINING_RESULT_PERSISTENCE_ENABLED", "MiningEngine", "MiningRun"]
