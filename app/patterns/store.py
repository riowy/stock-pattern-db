"""In-memory + optional DuckDB research-registry store.

Production path: data/state/pattern_registry.duckdb
Created only when PATTERN_REGISTRY_PERSISTENCE_ENABLED is true AND a write
is requested. Importing modules or unrelated CLI never creates the file.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb

from app.patterns.definition import PatternDefinition
from app.patterns.lifecycle import PatternStatus, StatusHistoryEntry
from app.patterns.models import PatternVersionRecord


SCHEMA_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS generators (
        generator_id VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        generator_type VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        retired_at TIMESTAMPTZ,
        notes VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS generator_versions (
        generator_id VARCHAR NOT NULL,
        version VARCHAR NOT NULL,
        implementation_module_id VARCHAR,
        configuration_hash VARCHAR,
        code_version VARCHAR,
        git_commit VARCHAR,
        ai_provider VARCHAR,
        ai_model VARCHAR,
        ai_model_version VARCHAR,
        prompt_template_id VARCHAR,
        prompt_template_hash VARCHAR,
        temperature DOUBLE,
        settings_json VARCHAR,
        created_at TIMESTAMPTZ NOT NULL,
        retired_at TIMESTAMPTZ,
        PRIMARY KEY (generator_id, version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pattern_versions (
        pattern_id VARCHAR NOT NULL,
        version INTEGER NOT NULL,
        definition_json VARCHAR NOT NULL,
        structural_fingerprint VARCHAR NOT NULL,
        hypothesis_fingerprint VARCHAR NOT NULL,
        canonical_structural_json VARCHAR NOT NULL,
        canonical_hypothesis_json VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        rejection_reason VARCHAR,
        rediscovery_count INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (pattern_id, version)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_pattern_hyp_fp ON pattern_versions(hypothesis_fingerprint)",
    "CREATE INDEX IF NOT EXISTS idx_pattern_status ON pattern_versions(status)",
    """
    CREATE TABLE IF NOT EXISTS pattern_status_history (
        id VARCHAR PRIMARY KEY,
        pattern_id VARCHAR NOT NULL,
        pattern_version INTEGER NOT NULL,
        from_status VARCHAR,
        to_status VARCHAR NOT NULL,
        reason VARCHAR NOT NULL,
        changed_at TIMESTAMPTZ NOT NULL,
        changed_by VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS discovery_runs (
        run_id VARCHAR PRIMARY KEY,
        generator_id VARCHAR NOT NULL,
        generator_version VARCHAR NOT NULL,
        started_at TIMESTAMPTZ NOT NULL,
        finished_at TIMESTAMPTZ,
        analysis_start DATE,
        analysis_end DATE,
        universe_name VARCHAR,
        universe_version VARCHAR,
        target VARCHAR,
        horizon INTEGER,
        candidate_params_json VARCHAR,
        quantile_fdr_json VARCHAR,
        runtime_seconds DOUBLE,
        cpu_seconds DOUBLE,
        peak_rss_mb DOUBLE,
        candidate_count INTEGER,
        supported_candidate_count INTEGER,
        proposed_pattern_count INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS discovery_events (
        event_id VARCHAR PRIMARY KEY,
        run_id VARCHAR NOT NULL,
        generator_id VARCHAR NOT NULL,
        generator_version VARCHAR NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        pattern_id VARCHAR,
        pattern_version INTEGER,
        hypothesis_fingerprint VARCHAR NOT NULL,
        proposal_kind VARCHAR NOT NULL,
        matched_pattern_id VARCHAR,
        matched_pattern_version INTEGER,
        validation_skipped BOOLEAN NOT NULL DEFAULT FALSE,
        definition_json VARCHAR,
        metadata_json VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evaluation_runs (
        evaluation_id VARCHAR PRIMARY KEY,
        pattern_id VARCHAR NOT NULL,
        pattern_version INTEGER NOT NULL,
        evaluator_id VARCHAR NOT NULL,
        evaluator_version VARCHAR NOT NULL,
        period_start DATE NOT NULL,
        period_end DATE NOT NULL,
        split_role VARCHAR NOT NULL,
        universe_name VARCHAR,
        universe_version VARCHAR,
        data_version VARCHAR,
        feature_version VARCHAR,
        label_version VARCHAR,
        code_version VARCHAR,
        sample_count INTEGER,
        date_count INTEGER,
        security_count INTEGER,
        decision VARCHAR,
        decision_reason VARCHAR,
        thresholds_json VARCHAR,
        created_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evaluation_metrics (
        metric_id VARCHAR PRIMARY KEY,
        evaluation_id VARCHAR NOT NULL,
        metric_name VARCHAR NOT NULL,
        numeric_value DOUBLE,
        unit VARCHAR,
        aggregation VARCHAR,
        horizon INTEGER,
        scope VARCHAR,
        period VARCHAR,
        metadata_json VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_pattern_signals (
        signal_id VARCHAR PRIMARY KEY,
        signal_date DATE NOT NULL,
        security_id VARCHAR NOT NULL,
        ticker VARCHAR,
        pattern_id VARCHAR NOT NULL,
        pattern_version INTEGER NOT NULL,
        direction VARCHAR NOT NULL,
        expected_horizon INTEGER,
        event_mode VARCHAR,
        historical_sample_size INTEGER,
        hit_rate DOUBLE,
        hit_rate_success_rule VARCHAR,
        median_outcome DOUBLE,
        mean_outcome DOUBLE,
        typical_loss_when_wrong DOUBLE,
        recent_rolling_json VARCHAR,
        generator_provenance_json VARCHAR,
        evaluation_ids_json VARCHAR,
        created_at TIMESTAMPTZ NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_signals_date ON daily_pattern_signals(signal_date)",
]


def new_id(prefix: str = "") -> str:
    uid = uuid.uuid4().hex
    return f"{prefix}{uid}" if prefix else uid


class PatternResearchStore:
    """Unified in-memory / DuckDB store for pattern research state.

    When ``persist`` is False, uses an in-memory DuckDB and never touches disk.
    When ``persist`` is True, opens ``db_path`` (creating parent dirs / file only
    on first connect for an explicit write session).
    """

    def __init__(
        self,
        *,
        persist: bool = False,
        db_path: Path | None = None,
        signal_persist: bool = False,
    ) -> None:
        self.persist = persist
        self.signal_persist = signal_persist
        self.db_path = db_path
        self._con: duckdb.DuckDBPyConnection | None = None
        self._opened_path: Path | None = None

    @property
    def is_open(self) -> bool:
        return self._con is not None

    def open(self) -> PatternResearchStore:
        if self._con is not None:
            return self
        if self.persist:
            if self.db_path is None:
                raise ValueError("db_path required when persist=True")
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._con = duckdb.connect(str(self.db_path))
            self._opened_path = self.db_path
        else:
            self._con = duckdb.connect(":memory:")
            self._opened_path = None
        for stmt in SCHEMA_STATEMENTS:
            self._con.execute(stmt)
        return self

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> PatternResearchStore:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            raise RuntimeError("Store is not open")
        return self._con

    def production_file_created(self) -> bool:
        return self._opened_path is not None and self._opened_path.exists()

    # --- helpers ------------------------------------------------------------------

    def _exec(self, sql: str, params: list[Any] | None = None) -> None:
        if params is None:
            self.con.execute(sql)
        else:
            self.con.execute(sql, params)

    def _fetchall(self, sql: str, params: list[Any] | None = None) -> list[tuple]:
        if params is None:
            return self.con.execute(sql).fetchall()
        return self.con.execute(sql, params).fetchall()

    def _fetchone(self, sql: str, params: list[Any] | None = None) -> tuple | None:
        rows = self._fetchall(sql, params)
        return rows[0] if rows else None

    # --- generators ---------------------------------------------------------------

    def upsert_generator(
        self,
        *,
        generator_id: str,
        name: str,
        generator_type: str,
        status: str = "ACTIVE",
        notes: str | None = None,
        created_at: datetime | None = None,
    ) -> None:
        created = created_at or datetime.now(UTC)
        existing = self._fetchone("SELECT generator_id FROM generators WHERE generator_id = ?", [generator_id])
        if existing:
            self._exec(
                "UPDATE generators SET name = ?, generator_type = ?, status = ?, notes = ? WHERE generator_id = ?",
                [name, generator_type, status, notes, generator_id],
            )
        else:
            self._exec(
                """
                INSERT INTO generators (generator_id, name, generator_type, status, created_at, retired_at, notes)
                VALUES (?, ?, ?, ?, ?, NULL, ?)
                """,
                [generator_id, name, generator_type, status, created, notes],
            )

    # Immutable generator_version fields compared on re-insert.
    # Timestamp policy: created_at is assigned at first insert and ignored on
    # idempotent re-insert (callers often pass datetime.now()). retired_at is
    # part of the frozen version payload and is compared.
    _GENERATOR_VERSION_IMMUTABLE_KEYS: tuple[str, ...] = (
        "implementation_module_id",
        "configuration_hash",
        "code_version",
        "git_commit",
        "ai_provider",
        "ai_model",
        "ai_model_version",
        "prompt_template_id",
        "prompt_template_hash",
        "temperature",
        "settings_json",
        "retired_at",
    )

    def insert_generator_version(self, row: dict[str, Any]) -> None:
        """Insert an immutable generator version. Raises if (id, version) exists with different payload.

        Identical re-inserts are no-ops. Divergent immutable metadata under the
        same generator_id/version is rejected — bump the version string instead.

        Timestamp policy: ``created_at`` is not compared (first-insert value is
        kept). ``retired_at`` is compared as immutable version metadata.
        """
        existing = self._fetchone(
            """
            SELECT implementation_module_id, configuration_hash, code_version, git_commit,
                   ai_provider, ai_model, ai_model_version, prompt_template_id,
                   prompt_template_hash, temperature, settings_json, retired_at
            FROM generator_versions WHERE generator_id = ? AND version = ?
            """,
            [row["generator_id"], row["version"]],
        )
        if existing:
            stored = dict(zip(self._GENERATOR_VERSION_IMMUTABLE_KEYS, existing, strict=True))
            mismatches = [
                key
                for key in self._GENERATOR_VERSION_IMMUTABLE_KEYS
                if not self._generator_version_values_equal(stored[key], row.get(key))
            ]
            if mismatches:
                raise ValueError(
                    f"Generator version {row['generator_id']}@{row['version']} is immutable; "
                    f"divergent fields: {', '.join(mismatches)}. "
                    "Create a new version instead of overwriting."
                )
            return
        self._exec(
            """
            INSERT INTO generator_versions (
                generator_id, version, implementation_module_id, configuration_hash,
                code_version, git_commit, ai_provider, ai_model, ai_model_version,
                prompt_template_id, prompt_template_hash, temperature, settings_json,
                created_at, retired_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                row["generator_id"],
                row["version"],
                row.get("implementation_module_id"),
                row.get("configuration_hash"),
                row.get("code_version"),
                row.get("git_commit"),
                row.get("ai_provider"),
                row.get("ai_model"),
                row.get("ai_model_version"),
                row.get("prompt_template_id"),
                row.get("prompt_template_hash"),
                row.get("temperature"),
                row.get("settings_json"),
                row.get("created_at") or datetime.now(UTC),
                row.get("retired_at"),
            ],
        )

    @staticmethod
    def _generator_version_values_equal(stored: Any, incoming: Any) -> bool:
        """Compare immutable version field values with light normalization."""
        if stored is None and incoming is None:
            return True
        if stored is None or incoming is None:
            return False
        if isinstance(stored, float) or isinstance(incoming, float):
            try:
                return float(stored) == float(incoming)
            except (TypeError, ValueError):
                return False
        if isinstance(stored, datetime) or isinstance(incoming, datetime):
            # Normalize aware/naive UTC for DuckDB round-trips.
            def _as_utc(v: Any) -> datetime | None:
                if not isinstance(v, datetime):
                    return None
                if v.tzinfo is None:
                    return v.replace(tzinfo=UTC)
                return v.astimezone(UTC)

            a, b = _as_utc(stored), _as_utc(incoming)
            if a is None or b is None:
                return False
            return a == b
        return stored == incoming

    def set_generator_status(self, generator_id: str, status: str, *, retired_at: datetime | None = None) -> None:
        if status == "ACTIVE":
            self._exec(
                "UPDATE generators SET status = ?, retired_at = NULL WHERE generator_id = ?",
                [status, generator_id],
            )
        else:
            ts = retired_at or datetime.now(UTC)
            self._exec(
                "UPDATE generators SET status = ?, retired_at = ? WHERE generator_id = ?",
                [status, ts, generator_id],
            )

    def list_generators(self) -> list[dict[str, Any]]:
        rows = self._fetchall(
            """
            SELECT generator_id, name, generator_type, status, created_at, retired_at, notes
            FROM generators ORDER BY name
            """
        )
        return [
            {
                "generator_id": r[0],
                "name": r[1],
                "generator_type": r[2],
                "status": r[3],
                "created_at": r[4],
                "retired_at": r[5],
                "notes": r[6],
            }
            for r in rows
        ]

    def get_generator(self, generator_id: str) -> dict[str, Any] | None:
        r = self._fetchone(
            "SELECT generator_id, name, generator_type, status, created_at, retired_at, notes FROM generators WHERE generator_id = ?",
            [generator_id],
        )
        if not r:
            return None
        return {
            "generator_id": r[0],
            "name": r[1],
            "generator_type": r[2],
            "status": r[3],
            "created_at": r[4],
            "retired_at": r[5],
            "notes": r[6],
        }

    def list_generator_versions(self, generator_id: str) -> list[dict[str, Any]]:
        rows = self._fetchall(
            """
            SELECT generator_id, version, implementation_module_id, configuration_hash,
                   code_version, git_commit, ai_provider, ai_model, ai_model_version,
                   prompt_template_id, prompt_template_hash, temperature, settings_json,
                   created_at, retired_at
            FROM generator_versions WHERE generator_id = ? ORDER BY created_at
            """,
            [generator_id],
        )
        keys = [
            "generator_id",
            "version",
            "implementation_module_id",
            "configuration_hash",
            "code_version",
            "git_commit",
            "ai_provider",
            "ai_model",
            "ai_model_version",
            "prompt_template_id",
            "prompt_template_hash",
            "temperature",
            "settings_json",
            "created_at",
            "retired_at",
        ]
        return [dict(zip(keys, r, strict=True)) for r in rows]

    # --- patterns -----------------------------------------------------------------

    def insert_pattern_version(self, record: PatternVersionRecord) -> None:
        self._exec(
            """
            INSERT INTO pattern_versions (
                pattern_id, version, definition_json, structural_fingerprint,
                hypothesis_fingerprint, canonical_structural_json, canonical_hypothesis_json,
                status, created_at, rejection_reason, rediscovery_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record.pattern_id,
                record.version,
                record.definition.model_dump_json(),
                record.structural_fingerprint,
                record.hypothesis_fingerprint,
                record.canonical_structural_json,
                record.canonical_hypothesis_json,
                str(record.status),
                record.created_at,
                record.rejection_reason,
                record.rediscovery_count,
            ],
        )

    def find_by_hypothesis_fingerprint(self, fp: str) -> PatternVersionRecord | None:
        r = self._fetchone(
            """
            SELECT pattern_id, version, definition_json, structural_fingerprint,
                   hypothesis_fingerprint, canonical_structural_json, canonical_hypothesis_json,
                   status, created_at, rejection_reason, rediscovery_count
            FROM pattern_versions
            WHERE hypothesis_fingerprint = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            [fp],
        )
        if not r:
            return None
        return self._row_to_pattern(r)

    def get_pattern_version(self, pattern_id: str, version: int | None = None) -> PatternVersionRecord | None:
        if version is None:
            r = self._fetchone(
                """
                SELECT pattern_id, version, definition_json, structural_fingerprint,
                       hypothesis_fingerprint, canonical_structural_json, canonical_hypothesis_json,
                       status, created_at, rejection_reason, rediscovery_count
                FROM pattern_versions WHERE pattern_id = ?
                ORDER BY version DESC LIMIT 1
                """,
                [pattern_id],
            )
        else:
            r = self._fetchone(
                """
                SELECT pattern_id, version, definition_json, structural_fingerprint,
                       hypothesis_fingerprint, canonical_structural_json, canonical_hypothesis_json,
                       status, created_at, rejection_reason, rediscovery_count
                FROM pattern_versions WHERE pattern_id = ? AND version = ?
                """,
                [pattern_id, version],
            )
        return self._row_to_pattern(r) if r else None

    def _row_to_pattern(self, r: tuple) -> PatternVersionRecord:
        return PatternVersionRecord(
            pattern_id=r[0],
            version=r[1],
            definition=PatternDefinition.model_validate_json(r[2]),
            structural_fingerprint=r[3],
            hypothesis_fingerprint=r[4],
            canonical_structural_json=r[5],
            canonical_hypothesis_json=r[6],
            status=PatternStatus(r[7]),
            created_at=r[8],
            rejection_reason=r[9],
            rediscovery_count=r[10] or 0,
        )

    def update_pattern_status(
        self,
        pattern_id: str,
        version: int,
        status: PatternStatus,
        *,
        rejection_reason: str | None = None,
    ) -> None:
        if rejection_reason is not None:
            self._exec(
                "UPDATE pattern_versions SET status = ?, rejection_reason = ? WHERE pattern_id = ? AND version = ?",
                [str(status), rejection_reason, pattern_id, version],
            )
        else:
            self._exec(
                "UPDATE pattern_versions SET status = ? WHERE pattern_id = ? AND version = ?",
                [str(status), pattern_id, version],
            )

    def increment_rediscovery(self, pattern_id: str, version: int) -> None:
        self._exec(
            "UPDATE pattern_versions SET rediscovery_count = rediscovery_count + 1 WHERE pattern_id = ? AND version = ?",
            [pattern_id, version],
        )

    def add_status_history(self, entry: StatusHistoryEntry) -> None:
        self._exec(
            """
            INSERT INTO pattern_status_history
            (id, pattern_id, pattern_version, from_status, to_status, reason, changed_at, changed_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                new_id("sh_"),
                entry.pattern_id,
                entry.pattern_version,
                str(entry.from_status) if entry.from_status else None,
                str(entry.to_status),
                entry.reason,
                entry.changed_at,
                entry.changed_by,
            ],
        )

    def list_status_history(self, pattern_id: str) -> list[dict[str, Any]]:
        rows = self._fetchall(
            """
            SELECT pattern_id, pattern_version, from_status, to_status, reason, changed_at, changed_by
            FROM pattern_status_history WHERE pattern_id = ? ORDER BY changed_at
            """,
            [pattern_id],
        )
        return [
            {
                "pattern_id": r[0],
                "pattern_version": r[1],
                "from_status": r[2],
                "to_status": r[3],
                "reason": r[4],
                "changed_at": r[5],
                "changed_by": r[6],
            }
            for r in rows
        ]

    def list_patterns(
        self,
        *,
        status: str | None = None,
        search: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = """
            SELECT pattern_id, version, status, definition_json, structural_fingerprint,
                   hypothesis_fingerprint, rediscovery_count, rejection_reason, created_at
            FROM pattern_versions
            WHERE 1=1
        """
        params: list[Any] = []
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC"
        rows = self._fetchall(sql, params or None)
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for r in rows:
            pid = r[0]
            if pid in seen:
                continue
            # Keep latest version only (rows ordered by created_at desc; versions may share created_at)
            latest = self.get_pattern_version(pid)
            if latest is None:
                continue
            if status and str(latest.status) != status:
                continue
            definition = latest.definition
            blob = f"{pid} {definition.name or ''} {definition.target} {latest.hypothesis_fingerprint}"
            if search and search.lower() not in blob.lower():
                continue
            seen.add(pid)
            out.append(
                {
                    "pattern_id": latest.pattern_id,
                    "version": latest.version,
                    "status": str(latest.status),
                    "direction": str(definition.direction),
                    "target": definition.target,
                    "horizon": definition.horizon,
                    "structural_fingerprint": latest.structural_fingerprint,
                    "hypothesis_fingerprint": latest.hypothesis_fingerprint,
                    "rediscovery_count": latest.rediscovery_count,
                    "rejection_reason": latest.rejection_reason,
                    "name": definition.name,
                    "created_at": latest.created_at,
                }
            )
        return out

    def status_counts(self) -> dict[str, int]:
        rows = self._fetchall(
            """
            SELECT status, COUNT(*) FROM (
                SELECT pattern_id, status, ROW_NUMBER() OVER (PARTITION BY pattern_id ORDER BY version DESC) AS rn
                FROM pattern_versions
            ) t WHERE rn = 1
            GROUP BY status
            """
        )
        return {r[0]: int(r[1]) for r in rows}

    # --- discovery ----------------------------------------------------------------

    def insert_discovery_run(self, row: dict[str, Any]) -> None:
        self._exec(
            """
            INSERT INTO discovery_runs (
                run_id, generator_id, generator_version, started_at, finished_at,
                analysis_start, analysis_end, universe_name, universe_version,
                target, horizon, candidate_params_json, quantile_fdr_json,
                runtime_seconds, cpu_seconds, peak_rss_mb,
                candidate_count, supported_candidate_count, proposed_pattern_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                row["run_id"],
                row["generator_id"],
                row["generator_version"],
                row["started_at"],
                row.get("finished_at"),
                row.get("analysis_start"),
                row.get("analysis_end"),
                row.get("universe_name"),
                row.get("universe_version"),
                row.get("target"),
                row.get("horizon"),
                row.get("candidate_params_json"),
                row.get("quantile_fdr_json"),
                row.get("runtime_seconds"),
                row.get("cpu_seconds"),
                row.get("peak_rss_mb"),
                row.get("candidate_count"),
                row.get("supported_candidate_count"),
                row.get("proposed_pattern_count"),
            ],
        )

    def update_discovery_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        self._exec(f"UPDATE discovery_runs SET {cols} WHERE run_id = ?", [*fields.values(), run_id])

    def insert_discovery_event(self, row: dict[str, Any]) -> None:
        self._exec(
            """
            INSERT INTO discovery_events (
                event_id, run_id, generator_id, generator_version, timestamp,
                pattern_id, pattern_version, hypothesis_fingerprint, proposal_kind,
                matched_pattern_id, matched_pattern_version, validation_skipped,
                definition_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                row["event_id"],
                row["run_id"],
                row["generator_id"],
                row["generator_version"],
                row["timestamp"],
                row.get("pattern_id"),
                row.get("pattern_version"),
                row["hypothesis_fingerprint"],
                row["proposal_kind"],
                row.get("matched_pattern_id"),
                row.get("matched_pattern_version"),
                bool(row.get("validation_skipped", False)),
                row.get("definition_json"),
                row.get("metadata_json"),
            ],
        )

    def list_discovery_runs(self, generator_id: str | None = None) -> list[dict[str, Any]]:
        if generator_id:
            rows = self._fetchall(
                "SELECT * FROM discovery_runs WHERE generator_id = ? ORDER BY started_at DESC",
                [generator_id],
            )
        else:
            rows = self._fetchall("SELECT * FROM discovery_runs ORDER BY started_at DESC")
        cols = [d[0] for d in self.con.description]
        return [dict(zip(cols, r, strict=True)) for r in rows]

    def list_discovery_events(
        self,
        *,
        run_id: str | None = None,
        generator_id: str | None = None,
        pattern_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if generator_id:
            clauses.append("generator_id = ?")
            params.append(generator_id)
        if pattern_id:
            clauses.append("(pattern_id = ? OR matched_pattern_id = ?)")
            params.extend([pattern_id, pattern_id])
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._fetchall(f"SELECT * FROM discovery_events{where} ORDER BY timestamp", params or None)
        cols = [d[0] for d in self.con.description]
        return [dict(zip(cols, r, strict=True)) for r in rows]

    # --- evaluation ---------------------------------------------------------------

    def insert_evaluation_run(self, row: dict[str, Any]) -> None:
        self._exec(
            """
            INSERT INTO evaluation_runs (
                evaluation_id, pattern_id, pattern_version, evaluator_id, evaluator_version,
                period_start, period_end, split_role, universe_name, universe_version,
                data_version, feature_version, label_version, code_version,
                sample_count, date_count, security_count, decision, decision_reason,
                thresholds_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                row["evaluation_id"],
                row["pattern_id"],
                row["pattern_version"],
                row["evaluator_id"],
                row["evaluator_version"],
                row["period_start"],
                row["period_end"],
                row["split_role"],
                row.get("universe_name"),
                row.get("universe_version"),
                row.get("data_version"),
                row.get("feature_version"),
                row.get("label_version"),
                row.get("code_version"),
                row.get("sample_count"),
                row.get("date_count"),
                row.get("security_count"),
                row.get("decision"),
                row.get("decision_reason"),
                row.get("thresholds_json"),
                row.get("created_at") or datetime.now(UTC),
            ],
        )

    def insert_evaluation_metric(self, row: dict[str, Any]) -> None:
        self._exec(
            """
            INSERT INTO evaluation_metrics (
                metric_id, evaluation_id, metric_name, numeric_value, unit,
                aggregation, horizon, scope, period, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                row.get("metric_id") or new_id("m_"),
                row["evaluation_id"],
                row["metric_name"],
                row.get("numeric_value"),
                row.get("unit"),
                row.get("aggregation"),
                row.get("horizon"),
                row.get("scope"),
                row.get("period"),
                row.get("metadata_json"),
            ],
        )

    def list_evaluations(self, pattern_id: str, version: int | None = None) -> list[dict[str, Any]]:
        if version is None:
            rows = self._fetchall(
                "SELECT * FROM evaluation_runs WHERE pattern_id = ? ORDER BY period_start",
                [pattern_id],
            )
        else:
            rows = self._fetchall(
                "SELECT * FROM evaluation_runs WHERE pattern_id = ? AND pattern_version = ? ORDER BY period_start",
                [pattern_id, version],
            )
        cols = [d[0] for d in self.con.description]
        return [dict(zip(cols, r, strict=True)) for r in rows]

    def list_metrics(self, evaluation_id: str) -> list[dict[str, Any]]:
        rows = self._fetchall(
            "SELECT * FROM evaluation_metrics WHERE evaluation_id = ? ORDER BY metric_name",
            [evaluation_id],
        )
        cols = [d[0] for d in self.con.description]
        return [dict(zip(cols, r, strict=True)) for r in rows]

    # --- signals ------------------------------------------------------------------

    def insert_signal(self, row: dict[str, Any]) -> None:
        if self.persist and not self.signal_persist:
            raise RuntimeError(
                "Daily signal persistence is disabled (DAILY_SIGNAL_PERSISTENCE_ENABLED=false). "
                "Use an in-memory store or enable the flag explicitly."
            )
        self._exec(
            """
            INSERT INTO daily_pattern_signals (
                signal_id, signal_date, security_id, ticker, pattern_id, pattern_version,
                direction, expected_horizon, event_mode, historical_sample_size,
                hit_rate, hit_rate_success_rule, median_outcome, mean_outcome,
                typical_loss_when_wrong, recent_rolling_json, generator_provenance_json,
                evaluation_ids_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                row.get("signal_id") or new_id("sig_"),
                row["signal_date"],
                row["security_id"],
                row.get("ticker"),
                row["pattern_id"],
                row["pattern_version"],
                row["direction"],
                row.get("expected_horizon"),
                row.get("event_mode"),
                row.get("historical_sample_size"),
                row.get("hit_rate"),
                row.get("hit_rate_success_rule"),
                row.get("median_outcome"),
                row.get("mean_outcome"),
                row.get("typical_loss_when_wrong"),
                row.get("recent_rolling_json"),
                row.get("generator_provenance_json"),
                row.get("evaluation_ids_json"),
                row.get("created_at") or datetime.now(UTC),
            ],
        )

    def list_signals(self, signal_date: date | None = None) -> list[dict[str, Any]]:
        if signal_date is None:
            rows = self._fetchall("SELECT * FROM daily_pattern_signals ORDER BY signal_date DESC, ticker")
        else:
            rows = self._fetchall(
                "SELECT * FROM daily_pattern_signals WHERE signal_date = ? ORDER BY ticker",
                [signal_date],
            )
        cols = [d[0] for d in self.con.description]
        return [dict(zip(cols, r, strict=True)) for r in rows]

    def signal_counts_for_date(self, signal_date: date) -> int:
        r = self._fetchone(
            "SELECT COUNT(*) FROM daily_pattern_signals WHERE signal_date = ?",
            [signal_date],
        )
        return int(r[0]) if r else 0


def dumps_json(obj: Any) -> str:
    return json.dumps(obj, default=str, sort_keys=True)
