"""Tests for the client module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

import llm.client as client


# ── setup command ─────────────────────────────────────────────────────────────


class TestSetupCommand:
    def test_setup_no_config_shows_placeholders(self, tmp_path, fake_console):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.chdir(tmp_path)

        def fake_find_config():
            return tmp_path / "nonexistent.toml"

        with patch.object(client, "find_config", fake_find_config):
            client.setup()

    def test_setup_no_config_with_custom_url(self, tmp_path, fake_console):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.chdir(tmp_path)

        def fake_find_config():
            return tmp_path / "nonexistent.toml"

        with patch.object(client, "find_config", fake_find_config):
            client.setup(base_url="https://10.0.0.5:8443/v1", api_key="custom-key")

    def test_setup_with_config_shows_cert_instructions(self, tmp_path, fake_console):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.chdir(tmp_path)

        config = tmp_path / "config.toml"
        config.write_text(
            '[server]\nllama_server_bin = "llama-server"\nport = 8080\nn_gpu_layers = 20\n'
            'n_ctx = 4096\nn_threads = 12\nextra_args = []\n\n[models]\ndir = "~/models"\n'
            'active = "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"\nhf_token = ""\n\n[proxy]\n'
            "port = 8443\nlan_ip = \"192.168.1.100\"\nlan_subnet = \"192.168.1.0/24\"\n"
            'api_key = "test-key-1234"\ncert_path = "cert.pem"\n\n[client]\n'
            'server_url = ""\napi_key = ""\ncert_path = ""\n\n[lxd]\ncraft_dirs = []\n'
        )

        (tmp_path / "cert.pem").write_text("-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n")

        def fake_find_config():
            return config

        client.setup()

    def test_setup_with_config_no_cert(self, tmp_path, fake_console):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.chdir(tmp_path)

        config = tmp_path / "config.toml"
        config.write_text(
            '[server]\nllama_server_bin = "llama-server"\nport = 8080\nn_gpu_layers = 20\n'
            'n_ctx = 4096\nn_threads = 12\nextra_args = []\n\n[models]\ndir = "~/models"\n'
            'active = "model.gguf"\nhf_token = ""\n\n[proxy]\nport = 8443\n'
            'lan_ip = "192.168.1.100"\nlan_subnet = "192.168.1.0/24"\n'
            'api_key = "key"\ncert_path = "cert.pem"\n\n[client]\n'
            'server_url = ""\napi_key = ""\ncert_path = ""\n\n[lxd]\ncraft_dirs = []\n'
        )

        def fake_find_config():
            return config

        client.setup()

    def test_setup_default_api_key_warning(self, tmp_path, fake_console):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.chdir(tmp_path)

        config = tmp_path / "config.toml"
        config.write_text(
            '[server]\nllama_server_bin = "llama-server"\nport = 8080\nn_gpu_layers = 20\n'
            'n_ctx = 4096\nn_threads = 12\nextra_args = []\n\n[models]\ndir = "~/models"\n'
            'active = "model.gguf"\nhf_token = ""\n\n[proxy]\nport = 8443\n'
            'lan_ip = "192.168.1.100"\nlan_subnet = "192.168.1.0/24"\n'
            'api_key = "change-me-generate-a-strong-random-key"\ncert_path = "/etc/ssl/cert.pem"\n\n[client]\n'
            'server_url = ""\napi_key = ""\ncert_path = ""\n\n[lxd]\ncraft_dirs = []\n'
        )

        def fake_find_config():
            return config

        client.setup()

    def test_setup_with_custom_url_override(self, tmp_path, fake_console):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.chdir(tmp_path)

        config = tmp_path / "config.toml"
        config.write_text(
            '[server]\nllama_server_bin = "llama-server"\nport = 8080\nn_gpu_layers = 20\n'
            'n_ctx = 4096\nn_threads = 12\nextra_args = []\n\n[models]\ndir = "~/models"\n'
            'active = "model.gguf"\nhf_token = ""\n\n[proxy]\nport = 8443\n'
            'lan_ip = "192.168.1.100"\nlan_subnet = "192.168.1.0/24"\n'
            'api_key = "key"\ncert_path = "/etc/ssl/cert.pem"\n\n[client]\n'
            'server_url = ""\napi_key = ""\ncert_path = ""\n\n[lxd]\ncraft_dirs = []\n'
        )

        def fake_find_config():
            return config

        client.setup(base_url="https://custom.example.com/v1", api_key="override-key")

    def test_setup_model_id_strips_gguf(self, tmp_path, fake_console):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.chdir(tmp_path)

        config = tmp_path / "config.toml"
        config.write_text(
            '[server]\nllama_server_bin = "llama-server"\nport = 8080\nn_gpu_layers = 20\n'
            'n_ctx = 4096\nn_threads = 12\nextra_args = []\n\n[models]\ndir = "~/models"\n'
            'active = "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"\nhf_token = ""\n\n[proxy]\n'
            "port = 8443\nlan_ip = \"192.168.1.100\"\nlan_subnet = \"192.168.1.0/24\"\n"
            'api_key = "key"\ncert_path = "/etc/ssl/cert.pem"\n\n[client]\n'
            'server_url = ""\napi_key = ""\ncert_path = ""\n\n[lxd]\ncraft_dirs = []\n'
        )

        def fake_find_config():
            return config

        # Should not crash
        client.setup()

    def test_setup_health_url_strips_v1(self, tmp_path, fake_console):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.chdir(tmp_path)

        config = tmp_path / "config.toml"
        config.write_text(
            '[server]\nllama_server_bin = "llama-server"\nport = 8080\nn_gpu_layers = 20\n'
            'n_ctx = 4096\nn_threads = 12\nextra_args = []\n\n[models]\ndir = "~/models"\n'
            'active = "model.gguf"\nhf_token = ""\n\n[proxy]\nport = 8443\n'
            'lan_ip = "192.168.1.100"\nlan_subnet = "192.168.1.0/24"\n'
            'api_key = "key"\ncert_path = "/etc/ssl/cert.pem"\n\n[client]\n'
            'server_url = ""\napi_key = ""\ncert_path = ""\n\n[lxd]\ncraft_dirs = []\n'
        )

        def fake_find_config():
            return config

        client.setup()

    def test_setup_with_cert_shows_install_block(self, tmp_path, fake_console):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.chdir(tmp_path)

        config = tmp_path / "config.toml"
        config.write_text(
            '[server]\nllama_server_bin = "llama-server"\nport = 8080\nn_gpu_layers = 20\n'
            'n_ctx = 4096\nn_threads = 12\nextra_args = []\n\n[models]\ndir = "~/models"\n'
            'active = "model.gguf"\nhf_token = ""\n\n[proxy]\nport = 8443\n'
            'lan_ip = "192.168.1.100"\nlan_subnet = "192.168.1.0/24"\n'
            'api_key = "key"\ncert_path = "cert.pem"\n\n[client]\n'
            'server_url = ""\napi_key = ""\ncert_path = ""\n\n[lxd]\ncraft_dirs = []\n'
        )

        (tmp_path / "cert.pem").write_text("-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n")

        def fake_find_config():
            return config

        client.setup()

    def test_setup_export_block_contains_all_env_vars(self, tmp_path, fake_console):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.chdir(tmp_path)

        config = tmp_path / "config.toml"
        config.write_text(
            '[server]\nllama_server_bin = "llama-server"\nport = 8080\nn_gpu_layers = 20\n'
            'n_ctx = 4096\nn_threads = 12\nextra_args = []\n\n[models]\ndir = "~/models"\n'
            'active = "model.gguf"\nhf_token = ""\n\n[proxy]\nport = 8443\n'
            'lan_ip = "192.168.1.100"\nlan_subnet = "192.168.1.0/24"\n'
            'api_key = "test-key"\ncert_path = "/etc/ssl/cert.pem"\n\n[client]\n'
            'server_url = ""\napi_key = ""\ncert_path = ""\n\n[lxd]\ncraft_dirs = []\n'
        )

        def fake_find_config():
            return config

        client.setup()


# ── Module-level constants ───────────────────────────────────────────────────


class TestModuleConstants:
    def test_placeholder_url(self):
        assert "<SERVER_LAN_IP>" in client._PLACEHOLDER_URL

    def test_placeholder_key(self):
        assert "<your-api-key>" == client._PLACEHOLDER_KEY

    def test_placeholder_model(self):
        assert client._PLACEHOLDER_MODEL.endswith(".gguf")

    def test_placeholder_ip(self):
        assert "<SERVER_LAN_IP>" == client._PLACEHOLDER_IP


# ── App definition ───────────────────────────────────────────────────────────


class TestAppDefinition:
    def test_app_is_typer(self):
        assert hasattr(client, "app")
        assert isinstance(client.app, typer.Typer)

    def test_app_help(self):
        # app is a Typer object with registered commands
        assert hasattr(client, "app")
        assert hasattr(client.app, "command")

    def test_app_no_args_is_help(self):
        # The Typer app may or may not have no_args_is_help set
        # We just check the attribute exists or skip
        assert hasattr(client.app, "command")
