"""Tests for the client module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import typer

import llm.client as client

# ── setup command (host) ─────────────────────────────────────────────────────


class TestHostSetup:
    def test_setup_no_config_exits(self, fake_no_config, fake_console):
        """Without config.toml, host setup should exit with an error."""
        with pytest.raises(typer.Exit):
            client.setup()

    def test_setup_with_config_applies_configs(self, fake_find_config, fake_console, monkeypatch, mocker):
        """With config.toml, host setup should apply client configs."""
        mocker.patch("llm.client.apply_client_configs")
        mocker.patch("llm.client.configure_shell_env_host", return_value=["action1"])
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


class TestContainerSetup:
    def test_setup_container_invokes_setup_container_client(self, mocker):
        mock_setup_vm = mocker.patch("llm.client._setup_container_client")
        client.setup(container="craft-llm-1", recreate=True, sandbox=True)
        mock_setup_vm.assert_called_once_with("craft-llm-1", recreate=True, sandbox=True)

    def test_setup_container_client_sandbox_mode(self, mocker, tmp_path):
        from llm.settings.models import GitHubSandboxSettings, GitHubSettings, HermesSettings, Settings

        cfg = Settings(
            github=GitHubSettings(
                token="personal-token",
                git_pat="personal-pat",
                sandbox=GitHubSandboxSettings(
                    token="bot-token",
                    git_pat="bot-pat",
                    git_username="mr-cal-bot",
                    git_email="bot@example.com",
                ),
            ),
            hermes=HermesSettings(openrouter_key="sk-or-v1-sandbox-key"),
        )
        mocker.patch("llm.client.find_config", return_value=tmp_path / "config.toml")
        mocker.patch("pathlib.Path.exists", return_value=True)
        mocker.patch("llm.client.load_config", return_value=cfg)
        mock_load_lxd = mocker.patch(
            "llm.provision.client_vm.load_lxd_settings",
            return_value=([("chiptune", "/h", "/c")], []),
        )
        mock_create = mocker.patch("llm.provision.client_vm.create_and_setup")
        mock_gh_auth = mocker.patch("llm.provision.client_vm.setup_gh_auth_in_container")
        mock_git_config = mocker.patch("llm.provision.client_vm.setup_git_config_in_container")
        mocker.patch("subprocess.run", return_value=MagicMock(returncode=0))

        client._setup_container_client("craft-llm-2", recreate=False, sandbox=True)

        mock_load_lxd.assert_called_once_with(sandbox=True)
        mock_create.assert_called_once()
        assert mock_create.call_args.kwargs["sandbox"] is True
        assert mock_create.call_args.kwargs["cert_pem"] is None

        # Verify bot credentials are used, NOT personal credentials
        mock_gh_auth.assert_called_once_with(
            "craft-llm-2",
            "bot-token",
            effective_uid=mocker.ANY,
            effective_gid=mocker.ANY,
        )
        mock_git_config.assert_called_once_with(
            "craft-llm-2",
            "mr-cal-bot",
            "bot@example.com",
            "bot-pat",
            uid=mocker.ANY,
            gid=mocker.ANY,
        )


class TestListContainers:
    def test_list_containers_sandbox_badge(self, mocker, fake_console):
        mocker.patch(
            "llm.provision.exec._list_managed_containers",
            return_value=["craft-1", "craft-sandbox"],
        )
        mocker.patch(
            "subprocess.run",
            return_value=MagicMock(stdout=json.dumps([{"status": "Running"}])),
        )
        mocker.patch("llm.provision.exec.get_container_kind", return_value="client")
        mocker.patch(
            "llm.provision.exec.is_container_sandbox",
            side_effect=lambda name: name == "craft-sandbox",
        )

        client.list_containers()
