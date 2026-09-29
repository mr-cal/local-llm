"""Tests for the shared subprocess primitives."""

from __future__ import annotations

import subprocess

import pytest

from llm.core import proc


@pytest.fixture(autouse=True)
def _clean_secret_registry():
    """Keep registered secrets from leaking between tests."""
    proc._SECRET_VALUES.clear()
    yield
    proc._SECRET_VALUES.clear()


class TestRun:
    def test_captures_stdout_by_default(self):
        assert proc.run(["echo", "hi"]).stdout == "hi\n"

    def test_reports_the_return_code(self):
        assert proc.run(["false"]).returncode == 1

    def test_does_not_raise_on_failure_by_default(self):
        proc.run(["false"])

    def test_raises_on_failure_when_checked(self):
        with pytest.raises(subprocess.CalledProcessError):
            proc.run(["false"], check=True)

    def test_restores_the_terminal_when_interrupted(self, monkeypatch):
        """A Ctrl+C at a sudo prompt must not leave the shell without echo."""
        restored = []

        def _run(cmd, **kwargs):
            if cmd == ["stty", "sane"]:
                restored.append(cmd)
                return None
            raise KeyboardInterrupt

        monkeypatch.setattr(subprocess, "run", _run)
        with pytest.raises(KeyboardInterrupt):
            proc.run(["sudo", "-v"])
        assert restored == [["stty", "sane"]]


class TestSucceeded:
    def test_true_on_zero_exit(self):
        assert proc.succeeded(["true"]) is True

    def test_false_on_non_zero_exit(self):
        assert proc.succeeded(["false"]) is False


class TestSudo:
    def test_prepends_sudo(self, monkeypatch):
        seen = []
        monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: seen.append(cmd))
        proc.sudo(["systemctl", "start", "nginx"])
        assert seen == [["sudo", "systemctl", "start", "nginx"]]

    def test_ensure_sudo_authenticates_up_front(self, monkeypatch):
        seen = []
        monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: seen.append(cmd))
        proc.ensure_sudo()
        assert seen == [["sudo", "-v"]]


class TestSudoStep:
    def test_reports_success(self, monkeypatch, fake_console, _make_proc):
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(0, ""))
        assert proc.sudo_step(["echo", "hello"], desc="test cmd") is True
        assert any("test cmd" in line for line in fake_console)

    def test_reports_failure(self, monkeypatch, fake_console, _make_proc):
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(1, "permission denied"))
        assert proc.sudo_step(["echo", "hello"], desc="test cmd") is False

    def test_includes_the_error_output(self, monkeypatch, fake_console, _make_proc):
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(1, "permission denied"))
        proc.sudo_step(["echo", "hello"], desc="test cmd")
        assert any("permission denied" in line for line in fake_console)


class TestUnitIsActive:
    def test_active(self, monkeypatch, _make_proc):
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(0, "active"))
        assert proc.unit_is_active("nginx") is True

    def test_inactive(self, monkeypatch, _make_proc):
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(1, "inactive"))
        assert proc.unit_is_active("nginx") is False

    def test_activating_is_not_active(self, monkeypatch, _make_proc):
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(0, "activating"))
        assert proc.unit_is_active("nginx") is False


class TestRedaction:
    def test_registered_secret_is_replaced(self):
        proc.register_secrets("hunter2-is-long-enough")
        assert proc.redact("token=hunter2-is-long-enough") == "token=***"

    def test_short_values_are_not_registered(self):
        """Short strings collide with ordinary command text."""
        proc.register_secrets("abc")
        assert proc.redact("abc def") == "abc def"

    def test_empty_and_none_values_are_ignored(self):
        proc.register_secrets("", None)
        assert set() == proc._SECRET_VALUES

    def test_multiple_secrets_are_all_replaced(self):
        proc.register_secrets("first-secret-value", "second-secret-value")
        assert proc.redact("first-secret-value/second-secret-value") == "***/***"

    def test_text_without_secrets_is_unchanged(self):
        proc.register_secrets("first-secret-value")
        assert proc.redact("nothing to see") == "nothing to see"
