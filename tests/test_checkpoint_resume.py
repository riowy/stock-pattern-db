from __future__ import annotations

from pathlib import Path

from app.ingestion.checkpoint import CheckpointStore, make_job_key


def test_make_job_key_is_deterministic() -> None:
    params = {"symbols": ["AAPL", "MSFT"], "start": "2024-01-01"}
    key1 = make_job_key("backfill_prices", params)
    key2 = make_job_key("backfill_prices", {"start": "2024-01-01", "symbols": ["AAPL", "MSFT"]})
    assert key1 == key2


def test_make_job_key_differs_for_different_params() -> None:
    key1 = make_job_key("backfill_prices", {"symbols": ["AAPL"]})
    key2 = make_job_key("backfill_prices", {"symbols": ["MSFT"]})
    assert key1 != key2


def test_checkpoint_save_load_roundtrip(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "checkpoints")
    key = "job1"
    store.save(key, {"completed": ["AAPL"], "pending": ["MSFT"]})

    loaded = store.load(key)
    assert loaded is not None
    assert loaded["completed"] == ["AAPL"]
    assert loaded["pending"] == ["MSFT"]
    assert "updated_at" in loaded


def test_checkpoint_load_missing_returns_none(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "checkpoints")
    assert store.load("does-not-exist") is None


def test_checkpoint_clear_removes_file(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "checkpoints")
    store.save("job1", {"completed": []})
    store.clear("job1")
    assert store.load("job1") is None


def test_checkpoint_resume_simulates_crash_recovery(tmp_path: Path) -> None:
    """Simulate: batch 1 succeeds and checkpoints, process 'crashes', a fresh
    run with --resume should only see the remaining work."""
    store = CheckpointStore(tmp_path / "checkpoints")
    key = "backfill_job"
    all_symbols = ["AAPL", "MSFT", "NVDA", "AMZN"]

    # Simulate first batch of 2 completing before a crash.
    store.save(key, {"completed": all_symbols[:2], "failed": []})

    # "Resume": a new process loads the checkpoint and computes pending work.
    state = store.load(key)
    assert state is not None
    completed = set(state["completed"])
    pending = [s for s in all_symbols if s not in completed]
    assert pending == ["NVDA", "AMZN"]
