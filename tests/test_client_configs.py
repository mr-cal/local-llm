"""Tests for the generated opencode, pi and omp client configurations."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import httpx

from llm.render.client_configs import (
    _build_omp_config_for_container,
    _build_opencode_config,
    _build_pi_config,
    _build_pi_config_for_container,
    _get_lxd_bridge_info,
    _get_server_model_info,
    _resolve_model_info,
    _validate_opencode_config,
)
from llm.settings import (
    AuthSettings,
    ModelsSettings,
    ProxySettings,
    ServerSettings,
    Settings,
    load_config,
)


class TestBuildOpencodeConfig:
    def test_basic_structure(self, tmp_config):
        cfg = load_config_from_path(tmp_config)
        result = _build_opencode_config(cfg)
        assert result["$schema"] == "https://opencode.ai/config.json"
        assert result["snapshot"] is True
        assert "permission" in result
        assert "provider" in result
        assert "model" in result

    def test_includes_local_llm_provider(self, tmp_config):
        cfg = load_config_from_path(tmp_config)
        result = _build_opencode_config(cfg)
        provider = result["provider"]["local-llm"]
        assert provider["name"] == "Local LLM"
        assert provider["npm"] == "@ai-sdk/openai-compatible"

    def test_includes_model_config(self, tmp_config):
        cfg = load_config_from_path(tmp_config)
        result = _build_opencode_config(cfg)
        models = provider_config(result)["models"]
        assert "local" in models
        assert models["local"]["tool_call"] is True

    def test_uses_n_ctx_as_limit(self, tmp_config):
        cfg = load_config_from_path(tmp_config)
        result = _build_opencode_config(cfg)
        model_cfg = provider_config(result)["models"]["local"]
        assert model_cfg["limit"]["context"] == cfg.server.n_ctx
        assert model_cfg["limit"]["input"] == cfg.server.n_ctx
        assert model_cfg["limit"]["output"] == 8192  # default

    def test_uses_custom_max_output(self, tmp_path, monkeypatch):
        config = tmp_path / "config.toml"
        config.write_text(
            '[server]\nllama_server_bin = "llama-server"\nport = 8080\nn_gpu_layers = 20\n'
            'n_ctx = 8192\nn_threads = 12\nextra_args = []\n\n[models]\ndir = "~/models"\n'
            'active = "big-model.gguf"\nhf_token = ""\n\n[auth]\napi_key = "test-key"\n'
            '\n[proxy]\nport = 8443\nlan_ip = "192.168.1.100"\nlan_subnet = "192.168.1.0/24"\n'
            'cert_path = "/etc/ssl/local-llm/cert.pem"\n\n[client]\n'
            'server_url = ""\ncert_path = ""\n\n[lxd]\ncraft_dirs = []\n'
        )
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        result = _build_opencode_config(cfg)
        # Since "big-model.gguf" isn't in KNOWN_MODELS, default max_output is 8192
        model_cfg = provider_config(result)["models"]["local"]
        assert model_cfg["limit"]["output"] == 8192

    def test_compaction_config_present(self, tmp_config):
        cfg = load_config_from_path(tmp_config)
        result = _build_opencode_config(cfg)
        compaction = result.get("compaction", {})
        assert "reserved" in compaction
        assert compaction["reserved"] == 8192
        assert "tail_turns" in compaction
        assert "preserve_recent_tokens" in compaction

    def test_agent_config_present(self, tmp_config):
        cfg = load_config_from_path(tmp_config)
        result = _build_opencode_config(cfg)
        agent = result.get("agent", {})
        assert "build" in agent
        assert "plan" in agent
        assert agent["build"]["temperature"] == 0.3
        assert agent["plan"]["temperature"] == 0.1


class TestBuildPiConfig:
    def test_basic_structure(self, tmp_config):
        cfg = load_config_from_path(tmp_config)
        result = _build_pi_config(cfg)
        assert "providers" in result
        assert "local-llm" in result["providers"]

    def test_includes_base_url_and_api_key(self, tmp_config):
        cfg = load_config_from_path(tmp_config)
        result = _build_pi_config(cfg)
        provider = result["providers"]["local-llm"]
        assert provider["baseUrl"] == cfg.client_url
        assert provider["api"] == "openai-completions"
        assert provider["apiKey"] == "key"  # auth.api_key from [auth] section

    def test_includes_model_entry(self, tmp_config):
        cfg = load_config_from_path(tmp_config)
        result = _build_pi_config(cfg)
        models = result["providers"]["local-llm"]["models"]
        assert len(models) == 1
        assert models[0]["id"] == "local"
        assert models[0]["contextWindow"] == cfg.server.n_ctx

    def test_uses_remote_url(self, tmp_config_client_only):
        cfg = load_config_from_path(tmp_config_client_only)
        result = _build_pi_config(cfg)
        assert result["providers"]["local-llm"]["baseUrl"] == "https://10.0.0.5:8443/v1"
        assert result["providers"]["local-llm"]["apiKey"] == "remote-key"

    def test_compat_flags_present(self, tmp_config):
        cfg = load_config_from_path(tmp_config)
        result = _build_pi_config(cfg)
        compat = result["providers"]["local-llm"]["compat"]
        assert compat["supportsDeveloperRole"] is False
        assert compat["supportsReasoningEffort"] is False
        assert compat["maxTokensField"] == "max_tokens"

    def test_cost_defaults_to_zero(self, tmp_config):
        cfg = load_config_from_path(tmp_config)
        result = _build_pi_config(cfg)
        model_entry = result["providers"]["local-llm"]["models"][0]
        assert model_entry["cost"]["input"] == 0.0
        assert model_entry["cost"]["output"] == 0.0
        assert model_entry["cost"]["cacheWrite"] == 0.0
        assert model_entry["cost"]["cacheRead"] == 0.0

    def test_cost_all_zero_when_not_configured(self):
        cfg = Settings()
        result = _build_pi_config(cfg)
        model_entry = result["providers"]["local-llm"]["models"][0]
        assert model_entry["cost"] == {"input": 0.0, "output": 0.0, "cacheWrite": 0.0, "cacheRead": 0.0}


# ── _build_pi_config_for_container ────────────────────────────────────────────


class TestBuildPiConfigForContainer:
    def test_uses_proxy_url_not_local(self):
        """Container config should use the proxy URL, not 127.0.0.1."""
        cfg = Settings(
            server=ServerSettings(port=8080, n_ctx=8192),
            proxy=ProxySettings(lan_ip="192.168.1.100", port=8443),
            models=ModelsSettings(active="test-model"),
        )
        result = _build_pi_config_for_container(cfg, "192.168.1.100")
        assert result["providers"]["local-llm"]["baseUrl"] == "https://192.168.1.100:8443/v1"

    def test_uses_https_scheme(self):
        """Container config should use HTTPS (via nginx proxy)."""
        cfg = Settings(
            server=ServerSettings(port=8080),
            proxy=ProxySettings(lan_ip="10.0.0.1", port=443),
            models=ModelsSettings(active="test-model"),
        )
        result = _build_pi_config_for_container(cfg, "10.0.0.1")
        assert result["providers"]["local-llm"]["baseUrl"].startswith("https://")

    def test_includes_api_key(self):
        """Container config should include the auth API key."""
        cfg = Settings(
            server=ServerSettings(port=8080),
            auth=AuthSettings(api_key="my-secret-key"),
            proxy=ProxySettings(lan_ip="192.168.1.1"),
            models=ModelsSettings(active="test-model"),
        )
        result = _build_pi_config_for_container(cfg, "192.168.1.1")
        assert result["providers"]["local-llm"]["apiKey"] == "my-secret-key"

    def test_fallback_api_key(self):
        """Container config should fallback to 'local' when no api_key is set."""
        cfg = Settings(
            server=ServerSettings(port=8080),
            auth=AuthSettings(api_key=""),
            proxy=ProxySettings(lan_ip="192.168.1.1"),
            models=ModelsSettings(active="test-model"),
        )
        result = _build_pi_config_for_container(cfg, "192.168.1.1")
        assert result["providers"]["local-llm"]["apiKey"] == "local"

    def test_includes_compatibility_settings(self):
        """Container config should include the same compat settings as the host config."""
        cfg = Settings(
            server=ServerSettings(port=8080),
            proxy=ProxySettings(lan_ip="192.168.1.1"),
            models=ModelsSettings(active="test-model"),
        )
        result = _build_pi_config_for_container(cfg, "192.168.1.1")
        compat = result["providers"]["local-llm"]["compat"]
        assert compat["supportsDeveloperRole"] is False
        assert compat["supportsReasoningEffort"] is False
        assert compat["maxTokensField"] == "max_tokens"

    def test_includes_model_entry(self):
        """Container config should include a model entry with context window and max tokens."""
        cfg = Settings(
            server=ServerSettings(port=8080, n_ctx=32768),
            proxy=ProxySettings(lan_ip="192.168.1.1"),
            models=ModelsSettings(active="test-model"),
        )
        result = _build_pi_config_for_container(cfg, "192.168.1.1")
        model_entry = result["providers"]["local-llm"]["models"][0]
        assert model_entry["id"] == "local"
        assert model_entry["contextWindow"] == 32768
        assert model_entry["maxTokens"] == 8192

    def test_uses_local_llm_hostname(self):
        """Container config should accept a hostname (e.g. 'local-llm') not just IPs."""
        cfg = Settings(
            server=ServerSettings(port=8080, n_ctx=8192),
            proxy=ProxySettings(lan_ip="192.168.1.100", port=8443),
            models=ModelsSettings(active="test-model"),
        )
        result = _build_pi_config_for_container(cfg, "local-llm")
        assert result["providers"]["local-llm"]["baseUrl"] == "https://local-llm:8443/v1"

    def test_local_llm_hostname_uses_https(self):
        """Hostname-based container URL should use HTTPS scheme."""
        cfg = Settings(
            server=ServerSettings(port=8080),
            proxy=ProxySettings(lan_ip="192.168.1.100", port=8443),
            models=ModelsSettings(active="test-model"),
        )
        result = _build_pi_config_for_container(cfg, "local-llm")
        assert result["providers"]["local-llm"]["baseUrl"].startswith("https://local-llm:")


# ── _validate_opencode_config ─────────────────────────────────────────────────


class TestValidateOpencodeConfig:
    def test_returns_warning_when_cannot_fetch_schema(self, mocker):
        mocker.patch(
            "urllib.request.urlopen",
            MagicMock(side_effect=Exception("no network")),
        )
        errors = _validate_opencode_config({"test": True})
        assert len(errors) == 1
        assert errors[0].startswith("⚠")

    def test_skips_model_enum_errors(self, monkeypatch):
        """Model field should not produce errors since we strip the $ref."""
        # If we could mock jsonschema validator, we'd test this more precisely.
        # The key behavior is that the model field is set to {"type": "string"}.
        errors = _validate_opencode_config({"test": True})
        # At minimum, it shouldn't crash
        assert isinstance(errors, list)


# ── config_show ────────────────────────────────────────────────────────────────


class TestGetServerModelInfo:
    """Tests for the server model_info endpoint query."""

    def test_returns_parsed_json_on_success(self, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "model_name": "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
            "ctx_size": 131072,
            "n_embd": 5120,
            "n_layers": 40,
        }
        mocker.patch("httpx.get", return_value=mock_resp)

        cfg = Settings(server=ServerSettings(port=8080))

        result = _get_server_model_info(cfg)
        assert result is not None
        assert result["model_name"] == "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"
        assert result["ctx_size"] == 131072

    def test_returns_none_on_http_error(self, mocker):

        mocker.patch(
            "httpx.get",
            side_effect=httpx.HTTPStatusError(
                "404 Not Found",
                request=MagicMock(),
                response=MagicMock(status_code=404),
            ),
        )

        cfg = Settings(server=ServerSettings(port=8080))

        result = _get_server_model_info(cfg)
        assert result is None

    def test_returns_none_on_connection_error(self, mocker):

        mocker.patch("httpx.get", side_effect=httpx.ConnectError("connection refused"))

        cfg = Settings(server=ServerSettings(port=8080))

        result = _get_server_model_info(cfg)
        assert result is None

    def test_returns_none_on_timeout(self, mocker):

        mocker.patch("httpx.get", side_effect=httpx.TimeoutException("timeout"))

        cfg = Settings(server=ServerSettings(port=8080))

        result = _get_server_model_info(cfg)
        assert result is None

    def test_uses_internal_url(self, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {}
        mock_get = mocker.patch("httpx.get", return_value=mock_resp)

        cfg = Settings(server=ServerSettings(port=9999))

        _get_server_model_info(cfg)
        mock_get.assert_called_once()
        call_url = mock_get.call_args[0][0]
        assert call_url == "http://127.0.0.1:9999/model_info"

    def test_uses_2s_timeout(self, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {}
        mock_get = mocker.patch("httpx.get", return_value=mock_resp)

        cfg = Settings(server=ServerSettings(port=8080))

        _get_server_model_info(cfg)
        mock_get.assert_called_once()
        assert mock_get.call_args[1].get("timeout") == 2


class TestResolveModelInfo:
    """Tests for the model info resolution with fallback chain."""

    def test_server_reported_values(self, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "model_name": "my-model.gguf",
            "ctx_size": 65536,
        }
        mocker.patch("httpx.get", return_value=mock_resp)

        cfg = Settings(
            server=ServerSettings(port=8080, n_ctx=4096),
            models=ModelsSettings(active="my-model.gguf"),
        )

        name, ctx, max_out = _resolve_model_info(cfg)
        assert name == "my-model.gguf"
        assert ctx == 65536
        assert max_out == 8192  # server n_ctx // 8 = 8192

    def test_server_ctx_size_zero_uses_cfg_default(self, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "model_name": "test.gguf",
            "ctx_size": 0,
        }
        mocker.patch("httpx.get", return_value=mock_resp)

        cfg = Settings(
            server=ServerSettings(port=8080, n_ctx=32768),
            models=ModelsSettings(active="test.gguf"),
        )

        name, ctx, max_out = _resolve_model_info(cfg)
        assert ctx == 32768  # falls back to cfg.server.n_ctx

    def test_server_without_ctx_size_uses_cfg_default(self, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "model_name": "test.gguf",
            # no ctx_size key
        }
        mocker.patch("httpx.get", return_value=mock_resp)

        cfg = Settings(
            server=ServerSettings(port=8080, n_ctx=8192),
            models=ModelsSettings(active="test.gguf"),
        )

        name, ctx, max_out = _resolve_model_info(cfg)
        assert ctx == 8192

    def test_server_small_max_output_defaults_to_8192(self, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "model_name": "tiny.gguf",
            "ctx_size": 2048,
        }
        mocker.patch("httpx.get", return_value=mock_resp)

        cfg = Settings(
            server=ServerSettings(port=8080),
            models=ModelsSettings(active="tiny.gguf"),
        )

        name, ctx, max_out = _resolve_model_info(cfg)
        assert max_out == 8192  # server n_ctx // 8 = 256 < 512 → default

    def test_server_not_reachable_uses_catalog(self, mocker):

        mocker.patch("httpx.get", side_effect=httpx.ConnectError("no server"))

        cfg = Settings(
            server=ServerSettings(port=8080, n_ctx=16384),
            models=ModelsSettings.model_validate(
                {
                    "active": "qwen2.5-coder-14b-q4",
                    "list": [
                        {
                            "alias": "qwen2.5-coder-14b-q4",
                            "repo": "x/y",
                            "filename": "x.gguf",
                            "max_output": 16384,
                        }
                    ],
                }
            ),
        )

        name, ctx, max_out = _resolve_model_info(cfg)
        assert name == "qwen2.5-coder-14b-q4"
        assert ctx == 16384
        assert max_out == 16384

    def test_server_not_reachable_uses_known_models(self, mocker):

        mocker.patch("httpx.get", side_effect=httpx.ConnectError("no server"))

        cfg = Settings(
            server=ServerSettings(port=8080, n_ctx=65536),
            models=ModelsSettings(
                active="qwen2.5-coder-14b-q4",
                # has_catalog is a computed property; no entries → has_catalog == False
            ),
        )

        name, ctx, max_out = _resolve_model_info(cfg)
        assert name == "qwen2.5-coder-14b-q4"  # from KNOWN_MODELS
        assert ctx == 65536  # from cfg
        assert max_out == 8192  # from KNOWN_MODELS default

    def test_server_not_reachable_uses_default_max_output(self, mocker):

        mocker.patch("httpx.get", side_effect=httpx.ConnectError("no server"))

        cfg = Settings(
            server=ServerSettings(port=8080, n_ctx=4096),
            models=ModelsSettings(active="custom.gguf"),  # not in any catalog
        )

        name, ctx, max_out = _resolve_model_info(cfg)
        assert name == "custom.gguf"
        assert ctx == 4096
        assert max_out == 8192  # hard default

    def test_server_n_ctx_used_for_max_output_heuristic(self, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "model_name": "big.gguf",
            "ctx_size": 131072,
        }
        mocker.patch("httpx.get", return_value=mock_resp)

        cfg = Settings(
            server=ServerSettings(port=8080, n_ctx=131072),
            models=ModelsSettings(active="big.gguf"),
        )

        name, ctx, max_out = _resolve_model_info(cfg)
        assert ctx == 131072
        assert max_out == 16384  # 131072 // 8


class TestBuildOpencodeConfigUsesServerInfo:
    """Verify that _build_opencode_config uses server-reported model info."""

    def test_uses_server_reported_context(self, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "model_name": "server-model.gguf",
            "ctx_size": 131072,
        }
        mocker.patch("httpx.get", return_value=mock_resp)

        cfg = Settings(
            server=ServerSettings(port=8080, n_ctx=4096),
            models=ModelsSettings(active="server-model.gguf"),
        )
        result = _build_opencode_config(cfg)
        model_cfg = provider_config(result)["models"]["local"]
        assert model_cfg["limit"]["context"] == 131072
        assert model_cfg["limit"]["input"] == 131072

    def test_uses_server_reported_max_output(self, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "model_name": "big-model.gguf",
            "ctx_size": 262144,
        }
        mocker.patch("httpx.get", return_value=mock_resp)

        cfg = Settings(
            server=ServerSettings(port=8080),
            models=ModelsSettings(active="big-model.gguf"),
        )
        result = _build_opencode_config(cfg)
        model_cfg = provider_config(result)["models"]["local"]
        assert model_cfg["limit"]["output"] == 32768  # 262144 // 8

    def test_uses_server_model_name(self, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "model_name": "my-custom-model.gguf",
            "ctx_size": 8192,
        }
        mocker.patch("httpx.get", return_value=mock_resp)

        cfg = Settings(
            server=ServerSettings(port=8080),
            models=ModelsSettings(active="my-custom-model.gguf"),
        )
        result = _build_opencode_config(cfg)
        model_cfg = provider_config(result)["models"]["local"]
        assert model_cfg["name"] == "my-custom-model.gguf"


class TestBuildPiConfigUsesServerInfo:
    """Verify that _build_pi_config uses server-reported model info."""

    def test_uses_server_reported_context(self, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "model_name": "server-model.gguf",
            "ctx_size": 131072,
        }
        mocker.patch("httpx.get", return_value=mock_resp)

        cfg = Settings(
            server=ServerSettings(port=8080, n_ctx=4096),
            models=ModelsSettings(active="server-model.gguf"),
        )
        result = _build_pi_config(cfg)
        model_entry = result["providers"]["local-llm"]["models"][0]
        assert model_entry["contextWindow"] == 131072

    def test_uses_server_reported_max_output(self, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "model_name": "big-model.gguf",
            "ctx_size": 262144,
        }
        mocker.patch("httpx.get", return_value=mock_resp)

        cfg = Settings(
            server=ServerSettings(port=8080),
            models=ModelsSettings(active="big-model.gguf"),
        )
        result = _build_pi_config(cfg)
        model_entry = result["providers"]["local-llm"]["models"][0]
        assert model_entry["maxTokens"] == 32768  # 262144 // 8

    def test_uses_server_model_name(self, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "model_name": "custom-model.gguf",
            "ctx_size": 8192,
        }
        mocker.patch("httpx.get", return_value=mock_resp)

        cfg = Settings(
            server=ServerSettings(port=8080),
            models=ModelsSettings(active="custom-model.gguf"),
        )
        result = _build_pi_config(cfg)
        model_entry = result["providers"]["local-llm"]["models"][0]
        assert model_entry["name"] == "custom-model.gguf"


class TestBuildPiConfigForContainerUsesServerInfo:
    """Verify that _build_pi_config_for_container uses server-reported model info."""

    def test_uses_server_reported_context(self, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "model_name": "server-model.gguf",
            "ctx_size": 131072,
        }
        mocker.patch("httpx.get", return_value=mock_resp)

        cfg = Settings(
            server=ServerSettings(port=8080, n_ctx=4096),
            proxy=ProxySettings(lan_ip="192.168.1.100", port=8443),
            models=ModelsSettings(active="server-model.gguf"),
        )
        result = _build_pi_config_for_container(cfg, "local-llm")
        model_entry = result["providers"]["local-llm"]["models"][0]
        assert model_entry["contextWindow"] == 131072

    def test_uses_server_reported_max_output(self, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "model_name": "big-model.gguf",
            "ctx_size": 262144,
        }
        mocker.patch("httpx.get", return_value=mock_resp)

        cfg = Settings(
            server=ServerSettings(port=8080),
            proxy=ProxySettings(lan_ip="192.168.1.100", port=8443),
            models=ModelsSettings(active="big-model.gguf"),
        )
        result = _build_pi_config_for_container(cfg, "local-llm")
        model_entry = result["providers"]["local-llm"]["models"][0]
        assert model_entry["maxTokens"] == 32768


class TestBuildOmpConfigForContainerUsesServerInfo:
    """Verify that _build_omp_config_for_container uses server-reported model info."""

    def test_uses_server_reported_context(self, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "model_name": "server-model.gguf",
            "ctx_size": 131072,
        }
        mocker.patch("httpx.get", return_value=mock_resp)

        cfg = Settings(
            server=ServerSettings(port=8080, n_ctx=4096),
            proxy=ProxySettings(lan_ip="192.168.1.100", port=8443),
            models=ModelsSettings(active="server-model.gguf"),
        )
        result = _build_omp_config_for_container(cfg, "local-llm")
        model_entry = result["providers"]["local-llm"]["models"][0]
        assert model_entry["contextWindow"] == 131072

    def test_uses_server_reported_max_output(self, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "model_name": "big-model.gguf",
            "ctx_size": 262144,
        }
        mocker.patch("httpx.get", return_value=mock_resp)

        cfg = Settings(
            server=ServerSettings(port=8080),
            proxy=ProxySettings(lan_ip="192.168.1.100", port=8443),
            models=ModelsSettings(active="big-model.gguf"),
        )
        result = _build_omp_config_for_container(cfg, "local-llm")
        model_entry = result["providers"]["local-llm"]["models"][0]
        assert model_entry["maxTokens"] == 32768


# ── Helpers ────────────────────────────────────────────────────────────────────


def load_config_from_path(config_path: Path):
    """Load settings from a config.toml at a specific path."""
    with config_path.open("rb") as f:
        raw = __import__("tomllib").load(f)
    return Settings.model_validate(raw)


def provider_config(opencode_cfg: dict) -> dict:
    return opencode_cfg["provider"]["local-llm"]


# ── secret masking ────────────────────────────────────────────────────────────


# Every credential in the settings tree, as (dotted path, sentinel value).
# A new secret field must be added here; test_no_secret_field_is_unmasked also
# fails independently if a SECRET-tagged field is missed by mask_secrets.
SECRET_PATHS = [
    ("auth.api_key", "SENTINEL-API-KEY"),
    ("models.hf_token", "SENTINEL-HF-TOKEN"),
    ("github.token", "SENTINEL-GH-TOKEN"),
    ("github.git_pat", "SENTINEL-GIT-PAT"),
    ("hermes.openrouter_key", "SENTINEL-OPENROUTER"),
    ("hermes.telegram_token", "SENTINEL-TELEGRAM"),
    ("hermes.github_token", "SENTINEL-HERMES-GH"),
]


def _settings_with_all_secrets() -> Settings:
    s = Settings()
    for path, sentinel in SECRET_PATHS:
        section, field = path.split(".")
        object.__setattr__(getattr(s, section), field, sentinel)
    return s


class TestGetLxdBridgeInfo:
    """Tests for _get_lxd_bridge_info which underpins both config and lxd modules."""

    _IP_OUTPUT = (
        "3: lxdbr0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP\n"
        "    link/ether 00:16:3e:xx:xx:xx brd ff:ff:ff:ff:ff:ff\n"
        "    inet 10.113.167.1/24 scope global lxdbr0\n"
        "       valid_lft forever preferred_lft forever\n"
    )

    def test_parses_bridge_ip(self, monkeypatch):
        def _run(cmd, **kw):
            p = MagicMock()
            p.returncode = 0
            p.stdout = self._IP_OUTPUT
            return p

        monkeypatch.setattr(subprocess, "run", _run)
        ip, subnet = _get_lxd_bridge_info()
        assert ip == "10.113.167.1"

    def test_parses_bridge_subnet(self, monkeypatch):
        def _run(cmd, **kw):
            p = MagicMock()
            p.returncode = 0
            p.stdout = self._IP_OUTPUT
            return p

        monkeypatch.setattr(subprocess, "run", _run)
        ip, subnet = _get_lxd_bridge_info()
        assert subnet == "10.113.167.0/24"

    def test_returns_empty_when_no_bridge(self, monkeypatch):
        def _run(cmd, **kw):
            p = MagicMock()
            p.returncode = 1
            p.stdout = ""
            return p

        monkeypatch.setattr(subprocess, "run", _run)
        ip, subnet = _get_lxd_bridge_info()
        assert ip == ""
        assert subnet == ""

    def test_returns_empty_when_no_inet_line(self, monkeypatch):
        def _run(cmd, **kw):
            p = MagicMock()
            p.returncode = 0
            p.stdout = "3: lxdbr0: <BROADCAST,MULTICAST,UP>\n"
            return p

        monkeypatch.setattr(subprocess, "run", _run)
        ip, subnet = _get_lxd_bridge_info()
        assert ip == ""
        assert subnet == ""

    def test_handles_different_subnet_size(self, monkeypatch):
        """/16 subnets should also parse correctly."""

        def _run(cmd, **kw):
            p = MagicMock()
            p.returncode = 0
            p.stdout = "    inet 172.16.0.1/16 scope global lxdbr0\n"
            return p

        monkeypatch.setattr(subprocess, "run", _run)
        ip, subnet = _get_lxd_bridge_info()
        assert ip == "172.16.0.1"
        assert subnet == "172.16.0.0/16"


# ── setup_pi_in_container ─────────────────────────────────────────────────────
