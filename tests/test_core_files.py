"""Tests for atomic file writing."""

from __future__ import annotations

import os
import stat

import pytest

from llm.core.files import write_atomic


class TestWriteAtomic:
    def test_writes_content(self, tmp_path):
        target = tmp_path / "out.txt"
        write_atomic(target, "hello\n")
        assert target.read_text() == "hello\n"

    def test_replaces_an_existing_file(self, tmp_path):
        target = tmp_path / "out.txt"
        target.write_text("old")
        write_atomic(target, "new")
        assert target.read_text() == "new"

    def test_creates_missing_parent_directories(self, tmp_path):
        target = tmp_path / "a" / "b" / "out.txt"
        write_atomic(target, "hi")
        assert target.read_text() == "hi"

    def test_leaves_the_previous_file_intact_on_failure(self, tmp_path, monkeypatch):
        target = tmp_path / "config.toml"
        target.write_text("original = true\n")

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", _boom)
        with pytest.raises(OSError, match="disk full"):
            write_atomic(target, "truncated")

        assert target.read_text() == "original = true\n"

    def test_leaves_no_temporary_file_behind_on_failure(self, tmp_path, monkeypatch):
        target = tmp_path / "config.toml"
        target.write_text("original = true\n")

        monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
        with pytest.raises(OSError):
            write_atomic(target, "x")

        assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]

    def test_leaves_no_temporary_file_behind_on_success(self, tmp_path):
        write_atomic(tmp_path / "out.txt", "hi")
        assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]

    def test_honours_an_explicit_mode(self, tmp_path):
        target = tmp_path / "secret"
        write_atomic(target, "api-key", mode=0o600)
        assert stat.S_IMODE(target.stat().st_mode) == 0o600

    def test_defaults_to_a_readable_mode(self, tmp_path):
        """mkstemp creates 0600; a plain write would not, so the default is widened."""
        target = tmp_path / "plain"
        write_atomic(target, "public")
        assert stat.S_IMODE(target.stat().st_mode) & stat.S_IRUSR

    def test_the_temporary_file_shares_the_destination_directory(self, tmp_path, monkeypatch):
        """os.replace is only atomic within one filesystem."""
        target = tmp_path / "nested" / "out.txt"
        seen: list[str] = []

        real_mkstemp = __import__("tempfile").mkstemp

        def _spy(*args, **kwargs):
            seen.append(str(kwargs.get("dir")))
            return real_mkstemp(*args, **kwargs)

        monkeypatch.setattr("llm.core.files.tempfile.mkstemp", _spy)
        write_atomic(target, "x")
        assert seen == [str(target.parent)]
