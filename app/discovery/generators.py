"""Generator registry: mining engines / AI / human proposers.

No automatic deletion, disable, or retire based on performance.
Soft-delete/retire only via explicit human action.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.patterns.store import PatternResearchStore


class GeneratorType(StrEnum):
    MINING_ENGINE = "MINING_ENGINE"
    AI = "AI"
    HUMAN = "HUMAN"


class GeneratorStatus(StrEnum):
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"
    RETIRED = "RETIRED"


class GeneratorVersionSpec(BaseModel):
    """Immutable metadata for one generator version. Never store secrets."""

    generator_id: str
    version: str
    implementation_module_id: str | None = None
    configuration_hash: str | None = None
    code_version: str | None = None
    git_commit: str | None = None
    # AI-specific (optional)
    ai_provider: str | None = None
    ai_model: str | None = None
    ai_model_version: str | None = None
    prompt_template_id: str | None = None
    prompt_template_hash: str | None = None
    temperature: float | None = None
    settings: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class Generator(Protocol):
    """Thin interface future mining engines / AI proposers implement."""

    @property
    def generator_id(self) -> str: ...

    @property
    def generator_type(self) -> GeneratorType: ...

    @property
    def name(self) -> str: ...


def hash_configuration(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class GeneratorRegistry:
    def __init__(self, store: PatternResearchStore) -> None:
        self.store = store

    def register(
        self,
        *,
        generator_id: str,
        name: str,
        generator_type: GeneratorType,
        version: GeneratorVersionSpec,
        notes: str | None = None,
    ) -> None:
        self.store.upsert_generator(
            generator_id=generator_id,
            name=name,
            generator_type=str(generator_type),
            status=str(GeneratorStatus.ACTIVE),
            notes=notes,
        )
        settings_json = json.dumps(version.settings, sort_keys=True, default=str) if version.settings else None
        self.store.insert_generator_version(
            {
                "generator_id": version.generator_id,
                "version": version.version,
                "implementation_module_id": version.implementation_module_id,
                "configuration_hash": version.configuration_hash,
                "code_version": version.code_version,
                "git_commit": version.git_commit,
                "ai_provider": version.ai_provider,
                "ai_model": version.ai_model,
                "ai_model_version": version.ai_model_version,
                "prompt_template_id": version.prompt_template_id,
                "prompt_template_hash": version.prompt_template_hash,
                "temperature": version.temperature,
                "settings_json": settings_json,
                "created_at": datetime.now(UTC),
            }
        )

    def disable(self, generator_id: str, *, actor: str | None = None) -> None:
        """Manual soft-disable. History preserved."""
        _ = actor
        gen = self.store.get_generator(generator_id)
        if gen is None:
            raise KeyError(generator_id)
        self.store.set_generator_status(generator_id, str(GeneratorStatus.DISABLED))

    def retire(self, generator_id: str, *, actor: str | None = None) -> None:
        """Manual soft-retire. History preserved. Not hard delete."""
        _ = actor
        gen = self.store.get_generator(generator_id)
        if gen is None:
            raise KeyError(generator_id)
        self.store.set_generator_status(generator_id, str(GeneratorStatus.RETIRED))

    def restore(self, generator_id: str, *, actor: str | None = None) -> None:
        _ = actor
        gen = self.store.get_generator(generator_id)
        if gen is None:
            raise KeyError(generator_id)
        self.store.set_generator_status(generator_id, str(GeneratorStatus.ACTIVE))

    def list(self) -> list[dict]:
        return self.store.list_generators()

    def get(self, generator_id: str) -> dict | None:
        return self.store.get_generator(generator_id)

    def versions(self, generator_id: str) -> list[dict]:
        return self.store.list_generator_versions(generator_id)

    def auto_retire_for_poor_performance(self, *_a, **_k) -> None:  # noqa: ANN002, ANN003
        """Explicitly unsupported. Humans decide disable/retire."""
        raise RuntimeError(
            "Automatic generator deletion/retirement based on performance is not allowed. "
            "Use GeneratorRegistry.disable/retire manually."
        )
