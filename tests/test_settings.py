"""Tests for the configuration schema and the config.toml loader."""

from __future__ import annotations

import tomllib

import pytest
import typer
from pydantic import ValidationError

from llm.settings import (
    AuthSettings,
    BuildConfig,
    BuildProfile,
    ClientSettings,
    GitHubSandboxSettings,
    GitHubSettings,
    LxdSettings,
    ModelCost,
    ModelEntry,
    ModelsSettings,
    MountEntry,
    ProxySettings,
    ServerSettings,
    Settings,
    find_config,
    load_config,
    try_load_lxd,
)
from llm.settings.loader import CONFIG_TEMPLATE
from llm.settings.models import canonicalize_timezone


class TestServerSettings:
    def test_defaults(self):
        s = ServerSettings()
        assert s.enabled is True
        assert s.llama_server_bin == "llama-server"
        assert s.port == 8080
        assert s.n_gpu_layers == 20
        assert s.n_ctx == 131072
        assert s.n_threads == 12
        assert s.extra_args == []
        assert s.monitor is True
        assert s.monitor_interval == 30
        assert s.monitor_retention_days == 90

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
        assert m.active == "qwen3.6-35b-moe-q4"
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
        m = ModelsSettings(active="test", entries=models)  # ty: ignore[unknown-argument]
        assert m.has_catalog is True

    def test_by_alias(self):
        models = [
            ModelEntry(alias="test", repo="test/repo", filename="test.gguf"),
            ModelEntry(alias="other", repo="other/repo", filename="other.gguf"),
        ]
        m = ModelsSettings(active="test", entries=models)  # ty: ignore[unknown-argument]
        assert m.by_alias("test") is not None
        assert m.by_alias("test").alias == "test"  # ty: ignore[unresolved-attribute]
        assert m.by_alias("other") is not None
        assert m.by_alias("other").alias == "other"  # ty: ignore[unresolved-attribute]
        assert m.by_alias("missing") is None

    def test_by_filename(self):
        models = [
            ModelEntry(alias="test", repo="test/repo", filename="test.gguf"),
        ]
        m = ModelsSettings(active="test", entries=models)  # ty: ignore[unknown-argument]
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

    def test_git_username_default(self):
        g = GitHubSettings()
        assert g.git_username == "mr-cal-bot"

    def test_git_email_default(self):
        g = GitHubSettings()
        assert g.git_email == "callahanlovesshopping@gmail.com"

    def test_git_pat_default_empty(self):
        g = GitHubSettings()
        assert g.git_pat == ""

    def test_custom_git_identity(self):
        g = GitHubSettings(git_username="other-bot", git_email="other@example.com")
        assert g.git_username == "other-bot"
        assert g.git_email == "other@example.com"

    def test_custom_git_pat(self):
        g = GitHubSettings(git_pat="github_pat_abc123")
        assert g.git_pat == "github_pat_abc123"

    def test_sandbox_defaults(self):
        g = GitHubSettings()
        assert g.sandbox.token == ""
        assert g.sandbox.git_pat == ""
        assert g.sandbox.git_username == "mr-cal-bot"
        assert g.sandbox.git_email == "callahanlovesshopping@gmail.com"
        assert g.sandbox.is_authenticated() is False

    def test_custom_sandbox_credentials(self):
        g = GitHubSettings(
            sandbox=GitHubSandboxSettings(
                token="ghp_bot_token",
                git_pat="ghp_bot_pat",
                git_username="mr-cal-bot",
                git_email="bot@example.com",
            )
        )
        assert g.sandbox.token == "ghp_bot_token"
        assert g.sandbox.git_pat == "ghp_bot_pat"
        assert g.sandbox.is_authenticated() is True


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


class TestCanonicalizeTimezone:
    def test_empty_or_whitespace(self):
        assert canonicalize_timezone("") == ""
        assert canonicalize_timezone("   ") == ""

    def test_us_aliases(self):
        assert canonicalize_timezone("US/chicago") == "America/Chicago"
        assert canonicalize_timezone("us/chicago") == "America/Chicago"
        assert canonicalize_timezone("US/Central") == "America/Chicago"
        assert canonicalize_timezone("us/central") == "America/Chicago"
        assert canonicalize_timezone("US/Eastern") == "America/New_York"
        assert canonicalize_timezone("us/new_york") == "America/New_York"
        assert canonicalize_timezone("US/Pacific") == "America/Los_Angeles"
        assert canonicalize_timezone("us/los_angeles") == "America/Los_Angeles"
        assert canonicalize_timezone("US/Mountain") == "America/Denver"
        assert canonicalize_timezone("us/denver") == "America/Denver"
        assert canonicalize_timezone("US/Arizona") == "America/Phoenix"
        assert canonicalize_timezone("US/Alaska") == "America/Anchorage"
        assert canonicalize_timezone("US/Hawaii") == "Pacific/Honolulu"

    def test_canonical_and_case_insensitive(self):
        assert canonicalize_timezone("America/Chicago") == "America/Chicago"
        assert canonicalize_timezone("america/chicago") == "America/Chicago"
        assert canonicalize_timezone("UTC") == "UTC"
        assert canonicalize_timezone("utc") == "UTC"
        assert canonicalize_timezone("Europe/London") == "Europe/London"


class TestLxdSettings:
    def test_defaults(self):
        lxd = LxdSettings()
        assert lxd.timezone == "America/Chicago"
        assert lxd.craft_dirs == []
        assert lxd.mounts == []
        assert lxd.sandbox_mounts == []

    def test_custom_timezone(self):
        lxd = LxdSettings(timezone="America/New_York")
        assert lxd.timezone == "America/New_York"

    def test_timezone_normalization(self):
        lxd1 = LxdSettings(timezone="US/chicago")
        assert lxd1.timezone == "America/Chicago"
        lxd2 = LxdSettings(timezone="US/Central")
        assert lxd2.timezone == "America/Chicago"
        lxd3 = LxdSettings(timezone="us/pacific")
        assert lxd3.timezone == "America/Los_Angeles"

    def test_with_mounts(self):
        mounts = [
            MountEntry(host="/home/user/.agents"),
            MountEntry(host="/home/user/dev"),
        ]
        sandbox_mounts = [
            MountEntry(host="/home/user/dev/cal/chiptune"),
        ]
        lxd = LxdSettings(
            craft_dirs=["~/dev/craft/snapcraft"],
            mounts=mounts,
            sandbox_mounts=sandbox_mounts,
        )
        assert len(lxd.craft_dirs) == 1
        assert len(lxd.mounts) == 2
        assert len(lxd.sandbox_mounts) == 1
        assert lxd.sandbox_mounts[0].name == "chiptune"


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
        mocker.patch("llm.settings.loader.Path.cwd", return_value=tmp_config.parent)
        assert find_config() == tmp_config

    def test_walks_up_to_find_config(self, tmp_path, mocker):
        subdir = tmp_path / "deep" / "nested"
        subdir.mkdir(parents=True)
        (tmp_path / "config.toml").write_text("[server]\n")
        mocker.patch("llm.settings.loader.Path.cwd", return_value=subdir)
        assert find_config() == tmp_path / "config.toml"

    def test_stops_at_pyproject_toml(self, tmp_path, mocker):
        config = tmp_path / "config.toml"
        config.write_text("[server]\n")
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\n")
        # CWD is inside a subdir with pyproject but no config
        subdir = tmp_path / "sub"
        subdir.mkdir()
        mocker.patch("llm.settings.loader.Path.cwd", return_value=subdir)
        result = find_config()
        # Should walk up past subdir, find pyproject at tmp_path, stop
        # but config.toml is at tmp_path so it finds it first
        assert result == config

    def test_returns_cwd_when_no_config_found(self, tmp_path, mocker):
        mocker.patch("llm.settings.loader.Path.cwd", return_value=tmp_path)
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


# ── config_init ────────────────────────────────────────────────────────────────


# ── _build_opencode_config / _build_pi_config ──────────────────────────────────


class TestServerSettingsValidation:
    """Values that cannot work should be rejected at load, not at start."""

    @pytest.mark.parametrize("port", [0, -1, 65536, 99999])
    def test_rejects_out_of_range_ports(self, port):
        with pytest.raises(ValidationError):
            ServerSettings(port=port)

    @pytest.mark.parametrize("field", ["n_ctx", "n_threads", "monitor_interval", "monitor_retention_days"])
    def test_rejects_non_positive_counts(self, field):
        with pytest.raises(ValidationError):
            ServerSettings(**{field: 0})  # ty: ignore[invalid-argument-type]

    def test_rejects_negative_gpu_layers(self):
        with pytest.raises(ValidationError):
            ServerSettings(n_gpu_layers=-1)

    def test_accepts_zero_gpu_layers_for_cpu_only(self):
        assert ServerSettings(n_gpu_layers=0).n_gpu_layers == 0


class TestProxySettingsValidation:
    @pytest.mark.parametrize("lan_ip", ["not-an-ip", "", "192.168.1.999", "192.168.1.0/24"])
    def test_rejects_invalid_lan_ip(self, lan_ip):
        with pytest.raises(ValidationError):
            ProxySettings(lan_ip=lan_ip)

    @pytest.mark.parametrize("subnet", ["nonsense", "", "192.168.1.0/33"])
    def test_rejects_invalid_lan_subnet(self, subnet):
        with pytest.raises(ValidationError):
            ProxySettings(lan_subnet=subnet)

    def test_accepts_a_host_address_in_the_subnet_field(self):
        # strict=False, so 192.168.1.5/24 is accepted and means 192.168.1.0/24.
        assert ProxySettings(lan_subnet="192.168.1.5/24").lan_subnet == "192.168.1.5/24"

    def test_rejects_out_of_range_port(self):
        with pytest.raises(ValidationError):
            ProxySettings(port=0)


class TestActiveModelValidation:
    def _entries(self):
        return [ModelEntry(alias="known", repo="r/r", filename="known.gguf")]

    def test_rejects_an_active_alias_that_is_not_in_the_catalog(self):
        with pytest.raises(ValidationError, match="not a known alias"):
            ModelsSettings(active="typo", entries=self._entries())  # ty: ignore[unknown-argument]

    def test_accepts_a_catalog_alias(self):
        assert ModelsSettings(active="known", entries=self._entries()).active == "known"  # ty: ignore[unknown-argument]

    def test_accepts_a_catalog_filename(self):
        m = ModelsSettings(active="known.gguf", entries=self._entries())  # ty: ignore[unknown-argument]
        assert m.active == "known.gguf"

    def test_accepts_an_uncatalogued_gguf_filename(self):
        m = ModelsSettings(active="something-else.gguf", entries=self._entries())  # ty: ignore[unknown-argument]
        assert m.active == "something-else.gguf"

    def test_an_empty_catalog_validates_nothing(self):
        assert ModelsSettings(active="anything").active == "anything"

    def test_rejects_duplicate_aliases(self):
        entries = [
            ModelEntry(alias="dupe", repo="r/r", filename="a.gguf"),
            ModelEntry(alias="dupe", repo="r/r", filename="b.gguf"),
        ]
        with pytest.raises(ValidationError, match="Duplicate model aliases"):
            ModelsSettings(active="dupe", entries=entries)  # ty: ignore[unknown-argument]

    def test_rejects_an_empty_alias(self):
        with pytest.raises(ValidationError):
            ModelEntry(alias="", repo="r/r", filename="a.gguf")


class TestCrossFieldValidation:
    def test_rejects_the_proxy_listening_on_the_server_port(self):
        with pytest.raises(ValidationError, match="cannot listen on the port it forwards to"):
            Settings(server=ServerSettings(port=8443), proxy=ProxySettings(port=8443))

    def test_allows_the_collision_when_the_proxy_is_disabled(self):
        s = Settings(server=ServerSettings(port=8443), proxy=ProxySettings(port=8443, enabled=False))
        assert s.proxy.port == 8443

    def test_rejects_a_server_profile_that_does_not_exist(self):
        with pytest.raises(ValidationError, match="is not a build profile"):
            Settings(
                server=ServerSettings(profile="typo"),
                build=BuildConfig(profiles=[BuildProfile(name="vulkan", backend="vulkan")]),
            )

    def test_accepts_a_server_profile_that_exists(self):
        s = Settings(
            server=ServerSettings(profile="vulkan"),
            build=BuildConfig(profiles=[BuildProfile(name="vulkan", backend="vulkan")]),
        )
        assert s.server.profile == "vulkan"

    def test_an_empty_profile_list_validates_nothing(self):
        assert Settings(server=ServerSettings(profile="anything")).server.profile == "anything"


class TestBuildJobsValidation:
    @pytest.mark.parametrize("jobs", ["many", "", "0", "-4", "1.5"])
    def test_rejects_non_positive_integer_jobs(self, jobs):
        with pytest.raises(ValidationError):
            BuildConfig(jobs=jobs)

    @pytest.mark.parametrize("jobs", ["auto", "1", "16"])
    def test_accepts_auto_and_positive_integers(self, jobs):
        assert BuildConfig(jobs=jobs).jobs == jobs


class TestShippedTemplate:
    """The template `llm config init` writes must stay loadable.

    Defaults drifted apart before this test existed: the template's `active`
    named a model that no longer appeared in its own [[models.list]].
    """

    def test_the_template_validates(self):
        Settings.model_validate(tomllib.loads(CONFIG_TEMPLATE))

    def test_the_template_agrees_with_the_schema_defaults(self):
        parsed = Settings.model_validate(tomllib.loads(CONFIG_TEMPLATE))
        defaults = Settings()
        assert parsed.models.active == defaults.models.active
        assert parsed.server.n_ctx == defaults.server.n_ctx
        assert parsed.build.install_dir == defaults.build.install_dir
