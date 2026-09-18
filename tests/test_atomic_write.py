from __future__ import annotations

from pathlib import Path

import pytest

from app.utils.atomic_io import atomic_write_bytes, atomic_write_via


def test_atomic_write_bytes_creates_file(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "file.txt"
    atomic_write_bytes(target, b"hello")
    assert target.read_bytes() == b"hello"
    # no leftover temp files
    assert list(target.parent.glob(".*")) == []


def test_atomic_write_bytes_replaces_existing(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    atomic_write_bytes(target, b"v1")
    atomic_write_bytes(target, b"v2")
    assert target.read_bytes() == b"v2"


def test_atomic_write_via_failed_validator_leaves_original_untouched(tmp_path: Path) -> None:
    target = tmp_path / "data.bin"
    atomic_write_bytes(target, b"original")

    def writer(path: Path) -> None:
        path.write_bytes(b"corrupt")

    def failing_validator(path: Path) -> None:
        raise ValueError("validation failed")

    with pytest.raises(ValueError):
        atomic_write_via(target, writer, failing_validator)

    assert target.read_bytes() == b"original"
    # no leftover temp files
    assert list(target.parent.glob(".*")) == []


def test_atomic_write_via_success(tmp_path: Path) -> None:
    target = tmp_path / "ok.bin"

    def writer(path: Path) -> None:
        path.write_bytes(b"new-content")

    atomic_write_via(target, writer)
    assert target.read_bytes() == b"new-content"
