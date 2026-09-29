"""Tests for the host shell-profile configuration.

These files are rewritten on every `llm server setup`, so the important
properties are that re-running changes nothing and that the file holding the
API key is not world-readable.
"""

from __future__ import annotations

import stat
from pathlib import Path

from llm.services.shell import _ensure_line_in_file, configure_shell_env_host


class TestEnsureLineInFile:
    def test_creates_the_file_and_reports_a_change(self, tmp_path):
        target = tmp_path / "nested" / ".bashrc"
        assert _ensure_line_in_file(target, "source me") is True
        assert "source me" in target.read_text()

    def test_is_idempotent(self, tmp_path):
        target = tmp_path / ".bashrc"
        _ensure_line_in_file(target, "source me")
        before = target.read_text()
        assert _ensure_line_in_file(target, "source me") is False
        assert target.read_text() == before

    def test_preserves_existing_content(self, tmp_path):
        target = tmp_path / ".bashrc"
        target.write_text("export EDITOR=vim\n")
        _ensure_line_in_file(target, "source me")
        content = target.read_text()
        assert "export EDITOR=vim" in content
        assert "source me" in content


class TestConfigureShellEnvHost:
    def _run(self, tmp_path, monkeypatch, **kwargs):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        defaults = {
            "base_url": "https://local-llm:8443/v1",
            "api_key": "sk-secret",
            "cert_path": "/etc/ssl/local-llm/cert.pem",
        }
        return configure_shell_env_host(**{**defaults, **kwargs})

    def test_writes_both_shell_profiles(self, tmp_path, monkeypatch):
        self._run(tmp_path, monkeypatch)
        env_file = tmp_path / ".config" / "local-llm" / "env"
        fish_conf = tmp_path / ".config" / "fish" / "conf.d" / "local-llm.fish"
        assert "sk-secret" in env_file.read_text()
        assert "sk-secret" in fish_conf.read_text()

    def test_exports_the_configured_endpoint(self, tmp_path, monkeypatch):
        self._run(tmp_path, monkeypatch, base_url="http://127.0.0.1:8080/v1")
        env_file = tmp_path / ".config" / "local-llm" / "env"
        assert 'export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"' in env_file.read_text()

    def test_the_api_key_file_is_not_world_readable(self, tmp_path, monkeypatch):
        self._run(tmp_path, monkeypatch)
        for name in (
            tmp_path / ".config" / "local-llm" / "env",
            tmp_path / ".config" / "fish" / "conf.d" / "local-llm.fish",
        ):
            assert stat.S_IMODE(name.stat().st_mode) == 0o600, name

    def test_sources_the_env_file_from_bashrc_once(self, tmp_path, monkeypatch):
        self._run(tmp_path, monkeypatch)
        self._run(tmp_path, monkeypatch)
        bashrc = (tmp_path / ".bashrc").read_text()
        source_lines = [ln for ln in bashrc.splitlines() if "/.config/local-llm/env" in ln]
        assert len(source_lines) == 1

    def test_rerunning_replaces_rather_than_appends(self, tmp_path, monkeypatch):
        self._run(tmp_path, monkeypatch, api_key="old-key")
        self._run(tmp_path, monkeypatch, api_key="new-key")
        content = (tmp_path / ".config" / "local-llm" / "env").read_text()
        assert "old-key" not in content
        assert content.count("OPENAI_API_KEY") == 1

    def test_reports_what_it_did(self, tmp_path, monkeypatch):
        actions = self._run(tmp_path, monkeypatch)
        assert any("local-llm/env" in a for a in actions)
        assert any("local-llm.fish" in a for a in actions)
