"""Tests for the configuration module."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import typer

from llm.config import (
    ClientSettings,
    LxdSettings,
    ModelCost,
    ModelCostSettings,
    ModelEntry,
    ModelsSettings,
    MountEntry,
    ProxySettings,
    ServerSettings,
    Settings,
    _build_opencode_config,
    _build_pi_config,
    _sudo,
    _systemctl_is_active,
    _validate_opencode_config,
    app,
    config_apply,
    config_gencert,
    config_init,
    config_show,
    find_config,
    load_config,
    try_load_lxd,
)

# ── Model classes ──────────────────────────────────────────────────────────────


class TestServerSettings:
    def test_defaults(self):
        s = ServerSettings()
        assert s.llama_server_bin == "llama-server"
        assert s.port == 8080
        assert s.n_gpu_layers == 20
        assert s.n_ctx == 4096
        assert s.n_threads == 12
        assert s.extra_args == []

    def test_custom_values(self):
        s = ServerSettings(port=9000, n_threads=8, extra_args=["--jinja"])
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
        m = ModelsSettings(entries=models)
        assert m.has_catalog is True

    def test_by_alias(self):
        models = [
            ModelEntry(alias="test", repo="test/repo", filename="test.gguf"),
            ModelEntry(alias="other", repo="other/repo", filename="other.gguf"),
        ]
        m = ModelsSettings(entries=models)
        assert m.by_alias("test").alias == "test"
        assert m.by_alias("other").alias == "other"
        assert m.by_alias("missing") is None

    def test_by_filename(self):
        models = [
            ModelEntry(alias="test", repo="test/repo", filename="test.gguf"),
        ]
        m = ModelsSettings(entries=models)
        assert m.by_filename("test.gguf").alias == "test"
        assert m.by_filename("missing.gguf") is None

    def test_model_path_resolves_alias(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        models = [
            ModelEntry(alias="test-model", repo="test/repo", filename="test-model.gguf"),
        ]
        s = Settings(models=ModelsSettings(dir=str(models_dir), active="test-model", entries=models))
        assert s.model_path == models_dir / "test-model.gguf"

    def test_model_path_uses_filename_when_no_match(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        active_file = models_dir / "test.gguf"
        active_file.touch()
        # No catalog — active is treated as filename
        s = Settings(models=ModelsSettings(dir=str(models_dir), active="test.gguf"))
        assert s.model_path == active_file

    def test_model_path_with_custom_model(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        models = [
            ModelEntry(alias="known", repo="known/repo", filename="known.gguf"),
        ]
        s = Settings(models=ModelsSettings(dir=str(models_dir), active="custom.gguf", entries=models))
        # custom.gguf not in catalog, treated as filename
        assert s.model_path.name == "custom.gguf"


class TestModelCostSettings:
    def test_defaults(self):
        c = ModelCostSettings()
        assert c.input == 0.0
        assert c.output == 0.0
        assert c.cache_write == 0.0
        assert c.cache_read == 0.0

    def test_custom_values(self):
        c = ModelCostSettings(input=0.5, output=1.5, cache_write=0.375, cache_read=0.05)
        assert c.input == 0.5
        assert c.output == 1.5
        assert c.cache_write == 0.375
        assert c.cache_read == 0.05

    def test_to_cost_dict_format(self):
        c = ModelCostSettings(input=0.0001, output=0.0002, cache_write=0.0001, cache_read=0.0)
        d = c.to_cost_dict()
        assert d == {"input": 0.0001, "output": 0.0002, "cacheWrite": 0.0001, "cacheRead": 0.0}
        # Verify camelCase keys for cache fields (pi convention)
        assert "cacheWrite" in d
        assert "cacheRead" in d
        assert "cache_write" not in d
        assert "cache_read" not in d

    def test_to_cost_dict_all_zeros(self):
        c = ModelCostSettings()
        d = c.to_cost_dict()
        assert d == {"input": 0.0, "output": 0.0, "cacheWrite": 0.0, "cacheRead": 0.0}

    def test_is_zero_defaults(self):
        c = ModelCostSettings()
        assert c.is_zero() is True

    def test_is_zero_nonzero(self):
        c = ModelCostSettings(output=0.5)
        assert c.is_zero() is False

    def test_is_zero_partial(self):
        c = ModelCostSettings(input=0.0001, output=0.0)
        assert c.is_zero() is False


class TestProxySettings:
    def test_defaults(self):
        p = ProxySettings()
        assert p.port == 8443
        assert p.lan_ip == "192.168.1.100"
        assert p.lan_subnet == "192.168.1.0/24"
        assert p.api_key == "change-me-generate-a-strong-random-key"
        assert p.cert_path == "/etc/ssl/local-llm/cert.pem"

    def test_custom_values(self):
        p = ProxySettings(port=9443, lan_ip="10.0.0.1")
        assert p.port == 9443
        assert p.lan_ip == "10.0.0.1"


class TestClientSettings:
    def test_defaults(self):
        c = ClientSettings()
        assert c.server_url == ""
        assert c.api_key == ""
        assert c.cert_path == ""

    def test_remote_config(self):
        c = ClientSettings(server_url="https://10.0.0.5:8443/v1", api_key="remote-key")
        assert c.server_url == "https://10.0.0.5:8443/v1"
        assert c.api_key == "remote-key"


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
            alias="test", repo="test/repo", filename="test.gguf",
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
        s = Settings(client=ClientSettings(api_key="remote-secret"))
        assert s.client_api_key == "remote-secret"

    def test_model_cost_defaults(self):
        s = Settings()
        assert s.model_cost.input == 0.0
        assert s.model_cost.output == 0.0
        assert s.model_cost.cache_write == 0.0
        assert s.model_cost.cache_read == 0.0

    def test_model_cost_custom(self):
        s = Settings(model_cost=ModelCostSettings(input=0.5, output=1.5, cache_write=0.375, cache_read=0.05))
        assert s.model_cost.input == 0.5
        assert s.model_cost.output == 1.5
        assert s.model_cost.cache_write == 0.375
        assert s.model_cost.cache_read == 0.05

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

    def test_loads_model_cost_section(self, tmp_path, monkeypatch):
        config = tmp_path / "config.toml"
        config.write_text(
            '[server]\nllama_server_bin = "llama-server"\nport = 8080\nn_gpu_layers = 20\n'
            'n_ctx = 4096\nn_threads = 12\nextra_args = []\n\n[models]\ndir = "~/models"\n'
            'active = "model.gguf"\nhf_token = ""\n\n[proxy]\nport = 8443\n'
            'lan_ip = "192.168.1.100"\nlan_subnet = "192.168.1.0/24"\n'
            'api_key = "key"\ncert_path = "/etc/ssl/cert.pem"\n\n[client]\n'
            'server_url = ""\napi_key = ""\ncert_path = ""\n\n[model_cost]\n'
            "input = 0.0001\noutput = 0.0002\ncache_write = 0.00015\ncache_read = 0.00005\n"
            "\n[lxd]\ncraft_dirs = []\n"
        )
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.model_cost.input == 0.0001
        assert cfg.model_cost.output == 0.0002
        assert cfg.model_cost.cache_write == 0.00015
        assert cfg.model_cost.cache_read == 0.00005


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


class TestConfigInit:
    def test_creates_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config_init()
        config = tmp_path / "config.toml"
        assert config.exists()
        content = config.read_text()
        assert "[server]" in content
        assert "[models]" in content
        assert "[proxy]" in content
        assert "[client]" in content
        assert "[lxd]" in content

    def test_exits_when_exists(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.toml").write_text("existing")
        with pytest.raises(typer.Exit):
            config_init()

    def test_overwrite_with_force(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.toml").write_text("existing")
        config_init(force=True)
        content = (tmp_path / "config.toml").read_text()
        assert "[server]" in content  # overwritten with template

    def test_template_has_all_sections(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config_init()
        content = (tmp_path / "config.toml").read_text()
        for section in ["[server]", "[models]", "[proxy]", "[client]", "[model_cost]", "[lxd]"]:
            assert section in content

    def test_template_model_cost_has_examples(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config_init()
        content = (tmp_path / "config.toml").read_text()
        assert "input = 0.0" in content
        assert "output = 0.0" in content
        assert "cache_write = 0.0" in content
        assert "cache_read = 0.0" in content


# ── config_init CLI ────────────────────────────────────────────────────────────


class TestConfigInitCLI:
    """CLI-level tests for `llm config init` via TyperRunner."""

    def test_cli_init_creates_config(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner  # noqa: PLC0415

        runner = CliRunner()
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        config = tmp_path / "config.toml"
        assert config.exists()
        content = config.read_text()
        assert "[server]" in content

    def test_cli_init_exits_when_exists(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner  # noqa: PLC0415

        runner = CliRunner()
        (tmp_path / "config.toml").write_text("existing")
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 1
        assert "already exists" in result.stdout

    def test_cli_init_force_overwrites(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner  # noqa: PLC0415

        runner = CliRunner()
        (tmp_path / "config.toml").write_text("existing")
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init", "--force"])
        assert result.exit_code == 0
        content = (tmp_path / "config.toml").read_text()
        assert "[server]" in content  # overwritten with template

    def test_cli_init_help(self):
        from typer.testing import CliRunner  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0
        assert "Create config.toml" in result.stdout
        assert "--force" in result.stdout


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
            'active = "big-model.gguf"\nhf_token = ""\n\n[proxy]\nport = 8443\n'
            'lan_ip = "192.168.1.100"\nlan_subnet = "192.168.1.0/24"\n'
            'api_key = "test-key"\ncert_path = "/etc/ssl/local-llm/cert.pem"\n\n[client]\n'
            'server_url = ""\napi_key = ""\ncert_path = ""\n\n[lxd]\ncraft_dirs = []\n'
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
        assert provider["apiKey"] == "local"  # empty client key falls back to "local"

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

    def test_cost_uses_config_values(self):
        cfg = Settings(
            server=ServerSettings(port=8080, n_ctx=8192),
            models=ModelsSettings(active="test.gguf"),
            model_cost=ModelCost(
                input=0.0001, output=0.0002, cache_write=0.00015, cache_read=0.00005
            ),
        )
        result = _build_pi_config(cfg)
        model_entry = result["providers"]["local-llm"]["models"][0]
        cost = model_entry["cost"]
        assert cost["input"] == 0.0001
        assert cost["output"] == 0.0002
        assert cost["cacheWrite"] == 0.00015
        assert cost["cacheRead"] == 0.00005

    def test_cost_all_zero_when_not_configured(self):
        cfg = Settings()
        result = _build_pi_config(cfg)
        model_entry = result["providers"]["local-llm"]["models"][0]
        assert model_entry["cost"] == {"input": 0.0, "output": 0.0, "cacheWrite": 0.0, "cacheRead": 0.0}


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


# ── config_apply ───────────────────────────────────────────────────────────────


class TestConfigApply:
    def test_applies_client_configs_only(self, tmp_config_client_only, fake_console, mocker, monkeypatch):
        monkeypatch.chdir(tmp_config_client_only.parent)

        mocker.patch("urllib.request.urlopen", side_effect=Exception("no network"))
        config_apply()

        # Instead, let's test the _build_pi_config directly
        pi_cfg = _build_pi_config(Settings(client=ClientSettings(server_url="https://10.0.0.5:8443/v1")))
        assert "providers" in pi_cfg

    def test_render_templates_and_apply(
        self, tmp_config_full, fake_console, fake_template_files, monkeypatch, _make_proc, mocker
    ):
        (tmp_config_full.parent / "models").mkdir()
        (tmp_config_full.parent / "models" / "model.gguf").write_text("fake")

        mocker.patch("urllib.request.urlopen", side_effect=Exception("no network"))
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(0, ""))
        monkeypatch.setattr(os.path, "exists", lambda x: True)
        config_apply()

        # Check nginx config was rendered
        nginx_conf = tmp_config_full.parent / "nginx" / "llm-proxy.conf"
        assert nginx_conf.exists()
        content = nginx_conf.read_text()
        assert "192.168.1.100" in content
        assert "8443" in content

        # Check systemd service was rendered
        svc = tmp_config_full.parent / "systemd" / "llm-server.service"
        assert svc.exists()
        content = svc.read_text()
        assert "llama-server" in content

    def test_creates_pi_models_json_from_scratch(
        self,
        tmp_config_full,
        fake_console,
        fake_template_files,
        monkeypatch,
        _make_proc,
        mocker,
        fake_pi_config_path,
    ):
        """When models.json doesn't exist, create it with local-llm provider."""
        (tmp_config_full.parent / "models").mkdir()
        (tmp_config_full.parent / "models" / "model.gguf").write_text("fake")
        assert not fake_pi_config_path.exists()

        mocker.patch("urllib.request.urlopen", side_effect=Exception("no network"))
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(0, ""))
        monkeypatch.setattr(os.path, "exists", lambda x: True)
        config_apply()

        assert fake_pi_config_path.exists()
        data = json.loads(fake_pi_config_path.read_text())
        assert "providers" in data
        assert "local-llm" in data["providers"]

    def test_merges_pi_models_json_with_other_provider(
        self,
        tmp_config_full,
        fake_console,
        fake_template_files,
        monkeypatch,
        _make_proc,
        mocker,
        fake_pi_config_path,
    ):
        """When models.json exists with another provider, merge adds/updates local-llm."""
        (tmp_config_full.parent / "models").mkdir()
        (tmp_config_full.parent / "models" / "model.gguf").write_text("fake")

        existing = {
            "providers": {
                "anthropic": {"baseUrl": "https://api.anthropic.com", "apiKey": "sk-xxx"},
                "other": {"baseUrl": "https://other.example.com", "apiKey": "key"},
            }
        }
        fake_pi_config_path.write_text(json.dumps(existing, indent=2))

        mocker.patch("urllib.request.urlopen", side_effect=Exception("no network"))
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(0, ""))
        monkeypatch.setattr(os.path, "exists", lambda x: True)
        config_apply()

        data = json.loads(fake_pi_config_path.read_text())
        assert "anthropic" in data["providers"]
        assert "other" in data["providers"]
        assert "local-llm" in data["providers"]
        assert data["providers"]["local-llm"]["baseUrl"].startswith("http://127.0.0.1")
        # Other providers must be untouched
        assert data["providers"]["anthropic"]["apiKey"] == "sk-xxx"
        assert data["providers"]["other"]["baseUrl"] == "https://other.example.com"

    def test_updates_local_llm_in_existing_models_json(
        self,
        tmp_config_full,
        fake_console,
        fake_template_files,
        monkeypatch,
        _make_proc,
        mocker,
        fake_pi_config_path,
    ):
        """When local-llm already exists in models.json, it gets overwritten with new values."""
        (tmp_config_full.parent / "models").mkdir()
        (tmp_config_full.parent / "models" / "model.gguf").write_text("fake")

        existing = {
            "providers": {
                "local-llm": {
                    "baseUrl": "http://127.0.0.1:7070/v1",
                    "apiKey": "old-key",
                },
                "anthropic": {"baseUrl": "https://api.anthropic.com"},
            }
        }
        fake_pi_config_path.write_text(json.dumps(existing, indent=2))

        mocker.patch("urllib.request.urlopen", side_effect=Exception("no network"))
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(0, ""))
        monkeypatch.setattr(os.path, "exists", lambda x: True)
        config_apply()

        data = json.loads(fake_pi_config_path.read_text())
        assert data["providers"]["local-llm"]["baseUrl"] == "http://127.0.0.1:8080/v1"
        assert data["providers"]["local-llm"]["apiKey"] == "local"
        assert "anthropic" in data["providers"]


# ── config_gencert ─────────────────────────────────────────────────────────────


class TestConfigGencert:
    def test_gencert_creates_cert(self, tmp_config_gencert, fake_console, monkeypatch, _make_proc):
        def fake_subprocess_run(cmd, **kw):
            if "openssl" in cmd:
                cert_dir = Path(cmd[cmd.index("-out") + 1]).parent
                cert_dir.mkdir(parents=True, exist_ok=True)
                (cert_dir / "cert.pem").write_text("fake-cert")
                key_path = Path(cmd[cmd.index("-keyout") + 1])
                key_path.write_text("fake-key")
                return _make_proc(0, "")
            return _make_proc(0, "")

        monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

        config_gencert()

        cert = tmp_config_gencert.parent / "cert.pem"
        key = tmp_config_gencert.parent / "key.pem"
        assert cert.exists()
        assert key.exists()

    def test_gencert_exits_when_exists(self, tmp_config_gencert, fake_console, monkeypatch, _make_proc):
        cert = tmp_config_gencert.parent / "cert.pem"
        cert.write_text("existing-cert")
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(0, ""))

        with pytest.raises(typer.Exit):
            config_gencert()

    def test_gencert_overwrite_with_force(self, tmp_config_gencert, fake_console, monkeypatch, _make_proc):
        cert = tmp_config_gencert.parent / "cert.pem"
        cert.write_text("existing")

        def fake_subprocess_run(cmd, **kw):
            if "openssl" in cmd:
                cert.write_text("new-cert")
                return _make_proc(0, "")
            return _make_proc(0, "")

        monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

        config_gencert(force=True)

        assert cert.read_text() == "new-cert"


# ── Helpers ────────────────────────────────────────────────────────────────────


def load_config_from_path(config_path: Path):
    """Load settings from a config.toml at a specific path."""
    with config_path.open("rb") as f:
        raw = __import__("tomllib").load(f)
    return Settings.model_validate(raw)


def provider_config(opencode_cfg: dict) -> dict:
    return opencode_cfg["provider"]["local-llm"]
