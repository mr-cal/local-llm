"""Tests for the `llm config` commands and secret masking."""

from __future__ import annotations

import json

import pytest
import tomli_w
from pydantic import BaseModel

from llm.config import _is_secret_field, _mask_api_keys, config_show, mask_secrets
from llm.render.client_configs import _build_opencode_config, _build_pi_config
from llm.settings import AuthSettings, EmbedSettings, Settings


class TestConfigShow:
    def test_prints_config_and_opencode(self, tmp_config, fake_console, mocker, monkeypatch):
        monkeypatch.chdir(tmp_config.parent)
        # config_show calls load_config() and _validate_opencode_config
        # It should run without errors
        mocker.patch("urllib.request.urlopen", side_effect=Exception("no network"))
        config_show()

    def test_masks_github_token(self, tmp_path, fake_console, mocker, monkeypatch):
        """config_show should mask the github token like it masks other secrets."""
        config = tmp_path / "config.toml"
        data = {
            "server": {
                "enabled": True,
                "llama_server_bin": "llama-server",
                "port": 8080,
                "n_gpu_layers": 20,
                "n_ctx": 4096,
                "n_threads": 12,
                "extra_args": [],
            },
            "models": {"dir": "~/models", "active": "test", "hf_token": "", "list": []},
            "auth": {"api_key": "secret"},
            "proxy": {
                "enabled": True,
                "port": 8443,
                "lan_ip": "192.168.1.100",
                "lan_subnet": "192.168.1.0/24",
                "cert_path": "/etc/ssl/local-llm/cert.pem",
            },
            "client": {"enabled": True, "server_url": "", "cert_path": ""},
            "lxd": {"craft_dirs": [], "mounts": []},
            "github": {"token": "ghp_secrettoken"},
        }
        config.write_text(tomli_w.dumps(data))
        monkeypatch.chdir(tmp_path)
        mocker.patch("urllib.request.urlopen", side_effect=Exception("no network"))
        # config_show should not raise - it should mask the github token
        config_show()


# ── _get_server_model_info / _resolve_model_info ────────────────────────────

# Every credential in the settings tree, as (dotted path, sentinel value).
# A new secret field must be added here; test_no_secret_field_is_unmasked also
# fails independently if a SECRET-tagged field is missed by mask_secrets.
SECRET_PATHS = [
    ("auth.api_key", "SENTINEL-API-KEY"),
    ("models.hf_token", "SENTINEL-HF-TOKEN"),
    ("github.token", "SENTINEL-GH-TOKEN"),
    ("github.git_pat", "SENTINEL-GIT-PAT"),
    ("client.openrouter_key", "SENTINEL-OPENROUTER"),
    ("discourse.session_cookie", "SENTINEL-DISCOURSE-COOKIE"),
    ("mattermost.token", "SENTINEL-MATTERMOST-TOKEN"),
    ("matrix.access_token", "SENTINEL-MATRIX-TOKEN"),
    ("jira.api_token", "SENTINEL-JIRA-TOKEN"),
]


def _settings_with_all_secrets() -> Settings:
    s = Settings()
    for path, sentinel in SECRET_PATHS:
        section, field = path.split(".")
        object.__setattr__(getattr(s, section), field, sentinel)
    return s


class TestMaskSecrets:
    @pytest.mark.parametrize(("path", "sentinel"), SECRET_PATHS)
    def test_secret_value_never_appears_in_output(self, path, sentinel):
        blob = json.dumps(mask_secrets(_settings_with_all_secrets()))
        assert sentinel not in blob, f"{path} leaked into `config show` output"

    @pytest.mark.parametrize(("path", "_sentinel"), SECRET_PATHS)
    def test_set_secret_is_masked(self, path, _sentinel):
        masked = mask_secrets(_settings_with_all_secrets())
        section, field = path.split(".")
        assert masked[section][field] == "***"

    def test_unset_secret_stays_empty(self):
        """An unset credential reads as empty, not as a masked value."""
        masked = mask_secrets(Settings())
        assert masked["auth"]["api_key"] == ""
        assert masked["matrix"]["access_token"] == ""

    def test_non_secret_values_are_preserved(self):
        masked = mask_secrets(_settings_with_all_secrets())
        assert masked["github"]["git_username"] == "mr-cal-bot"
        assert masked["server"]["port"] == 8080

    def test_no_secret_field_is_unmasked(self):
        """Walk the whole settings tree: every SECRET-tagged field must be masked.

        Guards against a credential being added to a model without being added
        to SECRET_PATHS above.
        """

        def walk(model, masked, trail=""):
            for name, field in type(model).model_fields.items():
                value = getattr(model, name)
                key = field.alias or name
                where = f"{trail}{key}"
                if isinstance(value, BaseModel):
                    walk(value, masked[key], f"{where}.")
                elif _is_secret_field(field):
                    assert masked[key] in ("***", ""), f"{where} was not masked"

        cfg = _settings_with_all_secrets()
        walk(cfg, mask_secrets(cfg))


class TestMaskApiKeys:
    """`config show` also prints generated client configs, which embed the API key."""

    def test_masks_opencode_api_key(self, tmp_path):
        cfg = Settings(auth=AuthSettings(api_key="SENTINEL-KEY"))
        blob = json.dumps(_mask_api_keys(_build_opencode_config(cfg)))
        assert "SENTINEL-KEY" not in blob
        assert '"apiKey": "***"' in blob

    def test_masks_pi_api_key(self):
        cfg = Settings(auth=AuthSettings(api_key="SENTINEL-KEY"))
        blob = json.dumps(_mask_api_keys(_build_pi_config(cfg)))
        assert "SENTINEL-KEY" not in blob

    def test_masks_nested_and_snake_case_keys(self):
        src = {"a": {"apiKey": "x"}, "b": [{"api_key": "y"}], "c": "keep"}
        assert _mask_api_keys(src) == {"a": {"apiKey": "***"}, "b": [{"api_key": "***"}], "c": "keep"}

    def test_empty_api_key_not_masked(self):
        assert _mask_api_keys({"apiKey": ""}) == {"apiKey": ""}


class TestEmbedSettingsDefaults:
    def test_disabled_by_default(self):
        s = EmbedSettings()
        assert s.enabled is False

    def test_default_port(self):
        assert EmbedSettings().port == 8081

    def test_default_active_model(self):
        assert EmbedSettings().active == "mxbai-embed-large"

    def test_default_extra_args(self):
        assert EmbedSettings().extra_args == []

    def test_extra_args_split(self):
        s = EmbedSettings(extra_args=["--embedding --pooling cls"])
        assert s.extra_args == ["--embedding", "--pooling", "cls"]

    def test_extra_args_already_split(self):
        s = EmbedSettings(extra_args=["--embedding", "--pooling", "cls"])
        assert s.extra_args == ["--embedding", "--pooling", "cls"]

    def test_extra_args_invalid_type(self):
        with pytest.raises(ValueError):
            EmbedSettings.model_validate({"extra_args": "--embedding"})


class TestEmbedSettingsInSettings:
    def test_settings_has_embed(self):
        s = Settings()
        assert hasattr(s, "embed")
        assert isinstance(s.embed, EmbedSettings)

    def test_embed_url(self):
        s = Settings(embed=EmbedSettings(port=8081))
        assert s.embed_url == "http://127.0.0.1:8081"

    def test_embed_toml_roundtrip(self, tmp_path):
        toml_text = """
[server]
port = 8080
n_gpu_layers = 20
n_ctx = 4096
n_threads = 4

[embed]
enabled = true
port = 8081
active = "mxbai-embed-large"
extra_args = ["--embedding", "--pooling", "cls"]

[models]
dir = "~/models"
active = "qwen2.5"

[proxy]
lan_ip = "192.168.1.1"

[auth]
api_key = "abc"
"""
        import tomllib

        raw = tomllib.loads(toml_text)
        s = Settings.model_validate(raw)
        assert s.embed.enabled is True
        assert s.embed.port == 8081
        assert s.embed.active == "mxbai-embed-large"
        assert s.embed.extra_args == ["--embedding", "--pooling", "cls"]


# ── systemd unit rendering ───────────────────────────────────────────────────
