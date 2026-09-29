"""Tests for the CLI entry point."""

from __future__ import annotations

import pytest

from llm import cli
from llm.core.errors import LlmError

EXPECTED_GROUPS = {"server", "model", "benchmark", "build", "client", "config", "hermes"}


class TestAppAssembly:
    def test_every_command_group_is_registered(self):
        registered = {g.name for g in cli.app.registered_groups}
        assert registered == EXPECTED_GROUPS


class TestErrorBoundary:
    """Expected failures are reported as messages; bugs still surface as tracebacks."""

    def _failing_app(self, monkeypatch, exc):
        def _raise():
            raise exc

        monkeypatch.setattr(cli, "app", _raise)

    def test_expected_failures_exit_non_zero(self, monkeypatch, fake_console):
        self._failing_app(monkeypatch, LlmError("bad host"))
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1

    def test_the_message_is_shown_to_the_user(self, monkeypatch, fake_console):
        self._failing_app(monkeypatch, LlmError("bad host"))
        with pytest.raises(SystemExit):
            cli.main()
        assert any("bad host" in line for line in fake_console)

    def test_unexpected_errors_still_propagate(self, monkeypatch):
        self._failing_app(monkeypatch, RuntimeError("a real bug"))
        with pytest.raises(RuntimeError, match="a real bug"):
            cli.main()
