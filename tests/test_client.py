"""Tests for the client module."""

from __future__ import annotations

import pytest
import typer

import llm.client as client

# ── setup command (host) ─────────────────────────────────────────────────────


class TestHostSetup:
    def test_setup_no_config_exits(self, fake_no_config, fake_console):
        """Without config.toml, host setup should exit with an error."""
        with pytest.raises(typer.Exit):
            client.setup()

    def test_setup_with_config_applies_configs(
        self, fake_find_config, fake_console, monkeypatch, mocker
    ):
        """With config.toml, host setup should apply client configs."""
        mocker.patch("llm.config.apply_client_configs")
        mocker.patch("llm.config.configure_shell_env_host", return_value=["action1"])
        client.setup()


# ── show command ─────────────────────────────────────────────────────────────


class TestShowCommand:
    def test_show_no_config_exits(self, fake_no_config, fake_console):
        with pytest.raises(typer.Exit):
            client.show()

    def test_show_with_config(self, fake_find_config, fake_console):
        client.show()


# ── App definition ───────────────────────────────────────────────────────────


class TestAppDefinition:
    def test_app_is_typer(self):
        assert hasattr(client, "app")
        assert isinstance(client.app, typer.Typer)

    def test_app_has_commands(self):
        assert hasattr(client.app, "command")
