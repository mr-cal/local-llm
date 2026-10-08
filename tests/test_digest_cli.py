"""Unit tests for digest and jotter CLI commands."""

from __future__ import annotations

from unittest.mock import MagicMock

from typer.testing import CliRunner

from llm.cli import app
from llm.digest.schedule import Slot

runner = CliRunner()


def test_digest_cli_help():
    result = runner.invoke(app, ["digest", "--help"])
    assert result.exit_code == 0
    assert "Digest generation commands" in result.output


def test_jotter_cli_help():
    result = runner.invoke(app, ["jotter", "--help"])
    assert result.exit_code == 0
    assert "Jotter container provisioning" in result.output


def test_digest_run_invalid_slot():
    result = runner.invoke(app, ["digest", "run", "--slot", "midnight"])
    assert result.exit_code == 1
    assert "Invalid slot" in result.output


def test_digest_run_mocked(monkeypatch):
    mock_engine = MagicMock()
    mock_engine.run_digest.return_value = "# Daily Digest\nMock output"

    monkeypatch.setattr("llm.digest.cli.DigestEngine", lambda cfg: mock_engine)
    result = runner.invoke(app, ["digest", "run", "--slot", "morning"])
    assert result.exit_code == 0
    assert "Digest completed successfully" in result.output
    mock_engine.run_digest.assert_called_once_with(slot=Slot.MORNING)
