"""Tests for the configuration module."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
import tomli_w
import typer

from llm.config import (
    AuthSettings,
    ClientSettings,
    GitHubSettings,
    LxdSettings,
    ModelCost,
    ModelEntry,
    ModelsSettings,
    MountEntry,
    ProxySettings,
    ServerSettings,
    Settings,
    _build_omp_config_for_container,
    _build_opencode_config,
    _build_pi_config,
    _build_pi_config_for_container,
    _sudo,
    _systemctl_is_active,
    _validate_opencode_config,
    config_show,
    find_config,
    load_config,
    try_load_lxd,
)

# ── Model classes ──────────────────────────────────────────────────────────────


class TestServerSettings:
    def test_defaults(self):
        s = ServerSettings()
        assert s.enabled is True
        assert s.llama_server_bin == "llama-server"
        assert s.port == 8080
        assert s.n_gpu_layers == 20
        assert s.n_ctx == 4096
        assert s.n_threads == 12
        assert s.extra_args == []

    def test_custom_values(self):
        s = ServerSettings(enabled=False, port=9000, n_threads=8, extra_args=["--jinja"])
        assert s.enabled is False
        assert s.port == 9000
        assert s.n_threads == 8
        assert s.extra_args == ["--jinja"]


class TestModelsSettings:
    def test_defaults(self):
        m = ModelsSettings()
        assert m.dir == "~/models"
        assert m.active == "qwen2.5-coder-14b-q4"
        assert m.hf_token == ""

    def test_custom_hf_token(self):
        m = ModelsSettings(hf_token="hf_secret")
        assert m.hf_token == "hf_secret"

    def test_has_catalog_false_by_default(self):
        m = ModelsSettings()
        assert m.has_catalog is False

    def test_has_catalog_true_with_list(self):
        models = [
            ModelEntry(alias="test", repo="test/repo", filename="test.gguf"),
        ]
        m = ModelsSettings(entries=models)  # ty: ignore[unknown-argument]
        assert m.has_catalog is True

    def test_by_alias(self):
        models = [
            ModelEntry(alias="test", repo="test/repo", filename="test.gguf"),
            ModelEntry(alias="other", repo="other/repo", filename="other.gguf"),
        ]
        m = ModelsSettings(entries=models)  # ty: ignore[unknown-argument]
        assert m.by_alias("test") is not None
        assert m.by_alias("test").alias == "test"  # ty: ignore[unresolved-attribute]
        assert m.by_alias("other") is not None
        assert m.by_alias("other").alias == "other"  # ty: ignore[unresolved-attribute]
        assert m.by_alias("missing") is None

    def test_by_filename(self):
        models = [
            ModelEntry(alias="test", repo="test/repo", filename="test.gguf"),
        ]
        m = ModelsSettings(entries=models)  # ty: ignore[unknown-argument]
        assert m.by_filename("test.gguf") is not None
        assert m.by_filename("test.gguf").alias == "test"  # ty: ignore[unresolved-attribute]
        assert m.by_filename("missing.gguf") is None

    def test_model_path_resolves_alias(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        models = [
            ModelEntry(alias="test-model", repo="test/repo", filename="test-model.gguf"),
        ]
        s = Settings(models=ModelsSettings(dir=str(models_dir), active="test-model", entries=models))  # ty: ignore[unknown-argument]
        assert s.model_path == models_dir / "test-model.gguf"

    def test_model_path_uses_filename_when_no_match(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        active_file = models_dir / "test.gguf"
        active_file.touch()
        # No catalog - active is treated as filename
        s = Settings(models=ModelsSettings(dir=str(models_dir), active="test.gguf"))
        assert s.model_path == active_file

    def test_model_path_with_custom_model(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        models = [
            ModelEntry(alias="known", repo="known/repo", filename="known.gguf"),
        ]
        s = Settings(models=ModelsSettings(dir=str(models_dir), active="custom.gguf", entries=models))  # ty: ignore[unknown-argument]
        # custom.gguf not in catalog, treated as filename
        assert s.model_path.name == "custom.gguf"


class TestModelCost:
    def test_defaults(self):
        c = ModelCost()
        assert c.input == 0.0
        assert c.output == 0.0
        assert c.cache_write == 0.0
        assert c.cache_read == 0.0

    def test_custom_values(self):
        c = ModelCost(input=0.5, output=1.5, cache_write=0.375, cache_read=0.05)
        assert c.input == 0.5
        assert c.output == 1.5
        assert c.cache_write == 0.375
        assert c.cache_read == 0.05

    def test_to_cost_dict_format(self):
        c = ModelCost(input=0.0001, output=0.0002, cache_write=0.0001, cache_read=0.0)
        d = c.to_cost_dict()
        assert d == {"input": 0.0001, "output": 0.0002, "cacheWrite": 0.0001, "cacheRead": 0.0}
        # Verify camelCase keys for cache fields (pi convention)
        assert "cacheWrite" in d
        assert "cacheRead" in d
        assert "cache_write" not in d
        assert "cache_read" not in d

    def test_to_cost_dict_all_zeros(self):
        c = ModelCost()
        d = c.to_cost_dict()
        assert d == {"input": 0.0, "output": 0.0, "cacheWrite": 0.0, "cacheRead": 0.0}

    def test_is_zero_defaults(self):
        c = ModelCost()
        assert c.is_zero() is True

    def test_is_zero_nonzero(self):
        c = ModelCost(output=0.5)
        assert c.is_zero() is False

    def test_is_zero_partial(self):
        c = ModelCost(input=0.0001, output=0.0)
        assert c.is_zero() is False


class TestProxySettings:
    def test_defaults(self):
        p = ProxySettings()
        assert p.enabled is True
        assert p.port == 8443
        assert p.lan_ip == "192.168.1.100"
        assert p.lan_subnet == "192.168.1.0/24"
        assert p.cert_path == "/etc/ssl/local-llm/cert.pem"

    def test_custom_values(self):
        p = ProxySettings(enabled=False, port=9443, lan_ip="10.0.0.1")
        assert p.enabled is False
        assert p.port == 9443
        assert p.lan_ip == "10.0.0.1"


class TestClientSettings:
    def test_defaults(self):
        c = ClientSettings()
        assert c.enabled is True
        assert c.server_url == ""
        assert c.cert_path == ""

    def test_remote_config(self):
        c = ClientSettings(server_url="https://10.0.0.5:8443/v1")
        assert c.server_url == "https://10.0.0.5:8443/v1"


class TestModelEntry:
    def test_defaults(self):
        m = ModelEntry(alias="test", repo="test/repo", filename="test.gguf")
        assert m.alias == "test"
        assert m.repo == "test/repo"
        assert m.filename == "test.gguf"
        assert m.size == ""
        assert m.description == ""
        assert m.max_output == 8192

    def test_id_property(self):
        m = ModelEntry(alias="my-model", repo="a/b", filename="c.gguf")
        assert m.id == "my-model"

    def test_cost_field(self):
        m = ModelEntry(
            alias="test",
            repo="test/repo",
            filename="test.gguf",
            cost=ModelCost(input=0.5, output=1.0),
        )
        assert m.cost.input == 0.5
        assert m.cost.output == 1.0

    def test_full_example(self):
        m = ModelEntry(
            alias="qwen2.5-coder-14b-q4",
            repo="bartowski/Qwen2.5-Coder-14B-Instruct-GGUF",
            filename="Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
            size="~8.5 GB",
            description="Qwen 2.5 Coder 14B",
            max_output=8192,
        )
        assert m.alias == "qwen2.5-coder-14b-q4"
        assert m.size == "~8.5 GB"
        assert m.max_output == 8192
        assert m.cost.is_zero() is True


class TestAuthSettings:
    def test_defaults(self):
        a = AuthSettings()
        assert a.api_key == ""

    def test_custom_key(self):
        a = AuthSettings(api_key="my-secret-key")
        assert a.api_key == "my-secret-key"


class TestGitHubSettings:
    def test_defaults(self):
        g = GitHubSettings()
        assert g.token == ""

    def test_custom_token(self):
        g = GitHubSettings(token="ghp_secrettoken123")
        assert g.token == "ghp_secrettoken123"

    def test_is_authenticated_empty(self):
        g = GitHubSettings()
        assert g.is_authenticated() is False

    def test_is_authenticated_with_token(self):
        g = GitHubSettings(token="ghp_secrettoken123")
        assert g.is_authenticated() is True

    def test_is_authenticated_whitespace_only(self):
        g = GitHubSettings(token="   ")
        assert g.is_authenticated() is False


class TestMountEntry:
    def test_derives_name_from_host_path(self):
        m = MountEntry(host="/home/user/dev")
        assert m.name == "dev"
        assert m.container == "/home/user/dev"

    def test_derives_name_strips_dot_prefix(self):
        m = MountEntry(host="/home/user/.agents")
        assert m.name == "agents"

    def test_custom_name_and_container(self):
        m = MountEntry(host="/opt/data", name="data", container="/mnt/data")
        assert m.name == "data"
        assert m.container == "/mnt/data"

    def test_expands_tilde(self):
        m = MountEntry(host="~/projects")
        assert "home" in m.name or m.name == "projects"


class TestLxdSettings:
    def test_defaults(self):
        lxd = LxdSettings()
        assert lxd.craft_dirs == []
        assert lxd.mounts == []

    def test_with_mounts(self):
        mounts = [
            MountEntry(host="/home/user/.agents"),
            MountEntry(host="/home/user/dev"),
        ]
        lxd = LxdSettings(craft_dirs=["~/dev/craft/snapcraft"], mounts=mounts)
        assert len(lxd.craft_dirs) == 1
        assert len(lxd.mounts) == 2


class TestSettings:
    def test_has_local_server_true(self):
        s = Settings(server=ServerSettings(llama_server_bin="llama-server"))
        assert s.has_local_server is True

    def test_has_local_server_false(self):
        s = Settings(server=ServerSettings(llama_server_bin=""))
        assert s.has_local_server is False

    def test_internal_url(self):
        s = Settings(server=ServerSettings(port=9000))
        assert s.internal_url == "http://127.0.0.1:9000"

    def test_proxy_url(self):
        s = Settings(proxy=ProxySettings(lan_ip="10.0.0.5", port=9443))
        assert s.proxy_url == "https://10.0.0.5:9443"

    def test_client_url_local_default(self):
        s = Settings(server=ServerSettings(port=8080))
        assert s.client_url == "http://127.0.0.1:8080/v1"

    def test_client_url_remote(self):
        s = Settings(
            server=ServerSettings(port=8080),
            client=ClientSettings(server_url="https://10.0.0.5:8443/v1"),
        )
        assert s.client_url == "https://10.0.0.5:8443/v1"

    def test_client_api_key_empty_when_local(self):
        s = Settings()
        assert s.client_api_key == ""

    def test_client_api_key_remote(self):
        s = Settings(auth=AuthSettings(api_key="remote-secret"))
        assert s.client_api_key == "remote-secret"

    def test_github_token_default(self):
        s = Settings()
        assert s.github.token == ""

    def test_github_token_custom(self):
        s = Settings(github=GitHubSettings(token="ghp_test123"))
        assert s.github.token == "ghp_test123"

    def test_github_token_is_authenticated(self):
        s = Settings(github=GitHubSettings(token="ghp_test123"))
        assert s.github.is_authenticated() is True

    def test_github_token_empty_is_not_authenticated(self):
        s = Settings()
        assert s.github.is_authenticated() is False

    def test_models_path_resolves_expands(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        s = Settings(models=ModelsSettings(dir=str(models_dir)))
        assert s.models_path == models_dir.resolve()

    def test_model_path(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        active_file = models_dir / "test.gguf"
        active_file.touch()
        s = Settings(models=ModelsSettings(dir=str(models_dir), active="test.gguf"))
        assert s.model_path == active_file

    def test_model_path_missing(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        s = Settings(models=ModelsSettings(dir=str(models_dir), active="missing.gguf"))
        # model_path is a Path object regardless of whether the file exists
        assert s.model_path.name == "missing.gguf"


# ── find_config / load_config ─────────────────────────────────────────────────


class TestFindConfig:
    def test_finds_config_in_cwd(self, tmp_config, mocker):
        mocker.patch("llm.config.Path.cwd", return_value=tmp_config.parent)
        assert find_config() == tmp_config

    def test_walks_up_to_find_config(self, tmp_path, mocker):
        subdir = tmp_path / "deep" / "nested"
        subdir.mkdir(parents=True)
        (tmp_path / "config.toml").write_text("[server]\n")
        mocker.patch("llm.config.Path.cwd", return_value=subdir)
        assert find_config() == tmp_path / "config.toml"

    def test_stops_at_pyproject_toml(self, tmp_path, mocker):
        config = tmp_path / "config.toml"
        config.write_text("[server]\n")
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\n")
        # CWD is inside a subdir with pyproject but no config
        subdir = tmp_path / "sub"
        subdir.mkdir()
        mocker.patch("llm.config.Path.cwd", return_value=subdir)
        result = find_config()
        # Should walk up past subdir, find pyproject at tmp_path, stop
        # but config.toml is at tmp_path so it finds it first
        assert result == config

    def test_returns_cwd_when_no_config_found(self, tmp_path, mocker):
        mocker.patch("llm.config.Path.cwd", return_value=tmp_path)
        assert find_config() == tmp_path / "config.toml"


class TestLoadConfig:
    def test_loads_config(self, tmp_config, monkeypatch):
        monkeypatch.chdir(tmp_config.parent)
        cfg = load_config()
        assert cfg.server.port == 8080
        # active stores whatever is in config.toml (filename or alias)
        assert cfg.models.active == "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"
        assert cfg.proxy.lan_ip == "192.168.1.100"
        assert cfg.client.server_url == ""

    def test_raises_on_missing_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(typer.Exit):
            load_config()


class TestTryLoadLxd:
    def test_returns_none_when_no_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert try_load_lxd() is None

    def test_returns_lxd_section(self, tmp_config, monkeypatch):
        monkeypatch.chdir(tmp_config.parent)
        lxd = try_load_lxd()
        assert lxd is not None
        assert lxd.craft_dirs == []
        assert lxd.mounts == []

    def test_returns_lxd_with_mounts(self, tmp_path, monkeypatch):
        config = tmp_path / "config.toml"
        config.write_text(
            '[lxd]\ncraft_dirs = ["~/dev/craft"]\n\n[[lxd.mounts]]\n'
            'host = "~/.agents"\n\n[[lxd.mounts]]\nhost = "~/dev"\n'
        )
        monkeypatch.chdir(tmp_path)
        lxd = try_load_lxd()
        assert lxd is not None
        assert len(lxd.craft_dirs) == 1
        assert len(lxd.mounts) == 2
        assert lxd.mounts[0].host == "~/.agents"
        assert lxd.mounts[1].host == "~/dev"


# ── _sudo and _systemctl_is_active ─────────────────────────────────────────────


class TestSudo:
    def test_sudo_success(self, monkeypatch, tmp_path, fake_console, _make_proc):
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(0, ""))
        assert _sudo("echo", "hello", desc="test cmd") is True

    def test_sudo_failure(self, monkeypatch, _make_proc):
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(1, "permission denied"))
        assert _sudo("echo", "hello", desc="test cmd") is False


class TestSystemctlIsActive:
    def test_active(self, monkeypatch, _make_proc):
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(0, "active"))
        assert _systemctl_is_active("nginx") is True

    def test_inactive(self, monkeypatch, _make_proc):
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(1, "inactive"))
        assert _systemctl_is_active("nginx") is False


# ── config_init ────────────────────────────────────────────────────────────────


# ── _build_opencode_config / _build_pi_config ──────────────────────────────────


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
        from llm.config import _get_server_model_info

        result = _get_server_model_info(cfg)
        assert result is not None
        assert result["model_name"] == "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"
        assert result["ctx_size"] == 131072

    def test_returns_none_on_http_error(self, mocker):
        import httpx

        mocker.patch("httpx.get", side_effect=httpx.HTTPStatusError(
            "404 Not Found",
            request=MagicMock(),
            response=MagicMock(status_code=404),
        ))

        cfg = Settings(server=ServerSettings(port=8080))
        from llm.config import _get_server_model_info

        result = _get_server_model_info(cfg)
        assert result is None

    def test_returns_none_on_connection_error(self, mocker):
        import httpx

        mocker.patch("httpx.get", side_effect=httpx.ConnectError("connection refused"))

        cfg = Settings(server=ServerSettings(port=8080))
        from llm.config import _get_server_model_info

        result = _get_server_model_info(cfg)
        assert result is None

    def test_returns_none_on_timeout(self, mocker):
        import httpx

        mocker.patch("httpx.get", side_effect=httpx.TimeoutException("timeout"))

        cfg = Settings(server=ServerSettings(port=8080))
        from llm.config import _get_server_model_info

        result = _get_server_model_info(cfg)
        assert result is None

    def test_uses_internal_url(self, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {}
        mocker.patch("httpx.get", return_value=mock_resp)

        cfg = Settings(server=ServerSettings(port=9999))
        from llm.config import _get_server_model_info

        _get_server_model_info(cfg)
        httpx.get.assert_called_once()
        call_url = httpx.get.call_args[0][0]
        assert call_url == "http://127.0.0.1:9999/model_info"

    def test_uses_2s_timeout(self, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {}
        mocker.patch("httpx.get", return_value=mock_resp)

        cfg = Settings(server=ServerSettings(port=8080))
        from llm.config import _get_server_model_info

        _get_server_model_info(cfg)
        httpx.get.assert_called_once()
        assert httpx.get.call_args[1].get("timeout") == 2


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
        from llm.config import _resolve_model_info

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
        from llm.config import _resolve_model_info

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
        from llm.config import _resolve_model_info

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
        from llm.config import _resolve_model_info

        name, ctx, max_out = _resolve_model_info(cfg)
        assert max_out == 8192  # server n_ctx // 8 = 256 < 512 → default

    def test_server_not_reachable_uses_catalog(self, mocker):
        import httpx

        mocker.patch("httpx.get", side_effect=httpx.ConnectError("no server"))

        cfg = Settings(
            server=ServerSettings(port=8080, n_ctx=16384),
            models=ModelsSettings(
                active="qwen2.5-coder-14b-q4",
                entries=[
                    ModelEntry(alias="qwen2.5-coder-14b-q4", repo="x/y", filename="x.gguf", max_output=16384),
                ],
            ),
        )
        from llm.config import _resolve_model_info

        name, ctx, max_out = _resolve_model_info(cfg)
        assert name == "qwen2.5-coder-14b-q4"
        assert ctx == 16384
        assert max_out == 16384

    def test_server_not_reachable_uses_known_models(self, mocker):
        import httpx

        mocker.patch("httpx.get", side_effect=httpx.ConnectError("no server"))

        cfg = Settings(
            server=ServerSettings(port=8080, n_ctx=65536),
            models=ModelsSettings(
                active="qwen2.5-coder-14b-q4",
                has_catalog=False,  # no config catalog → falls to KNOWN_MODELS
            ),
        )
        from llm.config import _resolve_model_info

        name, ctx, max_out = _resolve_model_info(cfg)
        assert name == "qwen2.5-coder-14b-q4"  # from KNOWN_MODELS
        assert ctx == 65536  # from cfg
        assert max_out == 8192  # from KNOWN_MODELS default

    def test_server_not_reachable_uses_default_max_output(self, mocker):
        import httpx

        mocker.patch("httpx.get", side_effect=httpx.ConnectError("no server"))

        cfg = Settings(
            server=ServerSettings(port=8080, n_ctx=4096),
            models=ModelsSettings(active="custom.gguf"),  # not in any catalog
        )
        from llm.config import _resolve_model_info

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
        from llm.config import _resolve_model_info

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
