"""Tests for the Hermes VM manager module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from llm.hermes import _format_credentials, _format_local_llm
from llm.provision.hermes_vm import HermesVmManager, _merge_env_file
from llm.settings import HermesSettings

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_cfg(
    provider: str = "local-llm",
    openrouter_key: str = "",
    telegram_token: str = "",
    telegram_allowed_users: str = "",
    github_token: str = "",
    mattermost_url: str = "",
    mattermost_token: str = "",
    mattermost_team: str = "canonical",
) -> HermesSettings:
    return HermesSettings(
        provider=provider,
        openrouter_key=openrouter_key,
        telegram_token=telegram_token,
        telegram_allowed_users=telegram_allowed_users,
        github_token=github_token,
        mattermost_url=mattermost_url,
        mattermost_token=mattermost_token,
        mattermost_team=mattermost_team,
    )


# ── _configure_credentials ────────────────────────────────────────────────────


class TestConfigureCredentials:
    """Tests for HermesVmManager._configure_credentials."""

    def _mgr(self, existing_env: str = ""):
        """Build a manager whose VM-side reads/writes are captured in memory."""
        mgr = HermesVmManager.__new__(HermesVmManager)
        mgr.container = "hermes"
        mgr.uid = 1000
        mgr.gid = 1000
        mgr._hermes_run = MagicMock()
        mgr._read_env_file = MagicMock(return_value=existing_env)
        return mgr

    @staticmethod
    def _written(mock_run) -> str:
        """Return the env-file contents sent to the VM over stdin."""
        for call in mock_run.call_args_list:
            if "input" in call.kwargs and call.kwargs["input"]:
                return call.kwargs["input"]
        raise AssertionError("No env-file write found in mock_run calls")

    @patch("llm.provision.hermes_vm.run")
    def test_no_credentials_skips(self, mock_run):
        """When no credentials are set, the method should log a warning and skip."""
        HermesVmManager._configure_credentials(self._mgr(), _make_cfg())
        mock_run.assert_not_called()

    @patch("llm.provision.hermes_vm.run")
    def test_single_credential_openrouter(self, mock_run):
        """Only the configured credential is written, and the provider is set."""
        mgr = self._mgr()
        cfg = _make_cfg(provider="openrouter", openrouter_key="sk-or-v1-test")
        HermesVmManager._configure_credentials(mgr, cfg)

        assert self._written(mock_run) == "OPENROUTER_API_KEY=sk-or-v1-test\n"
        mgr._hermes_run.assert_called_once_with(
            "config", "set", "model.provider", "openrouter", desc="set openrouter provider"
        )

    @patch("llm.provision.hermes_vm.run")
    def test_all_credentials_written(self, mock_run):
        """Every configured credential lands in the env file."""
        cfg = _make_cfg(
            provider="openrouter",
            openrouter_key="sk-or-v1-test",
            telegram_token="123:ABC",
            telegram_allowed_users="987654321",
            github_token="ghp_test",
        )
        HermesVmManager._configure_credentials(self._mgr(), cfg)

        assert self._written(mock_run).splitlines() == [
            "OPENROUTER_API_KEY=sk-or-v1-test",
            "TELEGRAM_BOT_TOKEN=123:ABC",
            "TELEGRAM_ALLOWED_USERS=987654321",
            "GITHUB_TOKEN=ghp_test",
        ]

    @patch("llm.provision.hermes_vm.run")
    def test_github_uses_has_github(self, mock_run):
        """Verify github_token uses has_github() guard (not raw truthiness)."""
        HermesVmManager._configure_credentials(self._mgr(), _make_cfg(github_token="   "))
        mock_run.assert_not_called()

    @patch("llm.provision.hermes_vm.run")
    def test_github_token_in_env_when_set(self, mock_run):
        """Verify GITHUB_TOKEN appears in env write when token is set."""
        HermesVmManager._configure_credentials(self._mgr(), _make_cfg(github_token="ghp_real"))
        assert self._written(mock_run) == "GITHUB_TOKEN=ghp_real\n"

    @patch("llm.provision.hermes_vm.run")
    def test_mattermost_credentials_written(self, mock_run):
        """Verify MATTERMOST_URL, TOKEN, TEAM in env and MCP server in config.yaml."""
        HermesVmManager._configure_credentials(
            self._mgr(),
            _make_cfg(
                mattermost_url="https://mm.example.com",
                mattermost_token="mm-secret-token",
                mattermost_team="custom-team",
            ),
        )
        assert mock_run.call_count == 3
        # First call: write env vars
        env_content = mock_run.call_args_list[0].kwargs["input"]
        assert env_content.splitlines() == [
            "MATTERMOST_URL=https://mm.example.com",
            "MATTERMOST_TOKEN=mm-secret-token",
            "MATTERMOST_TEAM=custom-team",
        ]
        # Second call: configure MCP in config.yaml
        mcp_cmd = mock_run.call_args_list[1].args[0]
        mcp_json = json.loads(mcp_cmd[-1])
        assert mcp_json["command"] == "uvx"
        assert mcp_json["args"] == [
            "--from",
            "mcp-server-mattermost==0.6.1",
            "mcp-server-mattermost",
        ]
        assert mcp_json["env"]["MATTERMOST_URL"] == "https://mm.example.com"
        assert mcp_json["env"]["MATTERMOST_TOKEN"] == "mm-secret-token"
        assert mcp_json["env"]["MATTERMOST_TEAM"] == "custom-team"
        assert "list_public_channels" in mcp_json["tools"]["include"]

    @patch("llm.provision.hermes_vm.run")
    def test_concurrency_limits_written(self, mock_run):
        """Verify max_concurrent_sessions and max_concurrent_children are written to config.yaml."""
        cfg = _make_cfg(
            provider="openrouter",
            openrouter_key="sk-or-v1-test",
        )
        cfg.max_concurrent_sessions = 1
        cfg.max_concurrent_children = 1
        HermesVmManager._configure_credentials(self._mgr(), cfg)

        concurrency_calls = [
            c
            for c in mock_run.call_args_list
            if c.kwargs.get("desc") == "configure concurrency limits in config.yaml"
        ]
        assert len(concurrency_calls) == 1
        cmd = concurrency_calls[0].args[0]
        payload = json.loads(cmd[-1])
        assert payload["max_concurrent_sessions"] == 1
        assert payload["max_concurrent_children"] == 1
        assert payload["approval_timeout"] == 3600

    @patch("llm.provision.hermes_vm.run")
    def test_secrets_never_appear_in_the_command(self, mock_run):
        """Credentials travel over stdin, never in argv where `ps` can see them."""
        cfg = _make_cfg(
            provider="openrouter",
            openrouter_key="sk-or-v1-secret",
            telegram_token="123:SECRETTOKEN",
            github_token="ghp_secretvalue",
        )
        HermesVmManager._configure_credentials(self._mgr(), cfg)

        for call in mock_run.call_args_list:
            argv = " ".join(str(a) for a in call.args[0])
            assert "sk-or-v1-secret" not in argv
            assert "123:SECRETTOKEN" not in argv
            assert "ghp_secretvalue" not in argv

    @patch("llm.provision.hermes_vm.run")
    def test_shell_metacharacters_survive_verbatim(self, mock_run):
        """A token containing shell syntax must be stored literally, not executed."""
        nasty = "tok'en|rm -rf /;$(whoami)"
        HermesVmManager._configure_credentials(self._mgr(), _make_cfg(github_token=nasty))
        assert self._written(mock_run) == f"GITHUB_TOKEN={nasty}\n"

    @patch("llm.provision.hermes_vm.run")
    def test_existing_keys_replaced_unrelated_preserved(self, mock_run):
        """Re-running updates known keys in place and leaves other entries alone."""
        existing = "SSL_CERT_FILE=/home/dev/.hermes/ca-bundle.pem\nGITHUB_TOKEN=ghp_old\n"
        HermesVmManager._configure_credentials(self._mgr(existing), _make_cfg(github_token="ghp_new"))
        assert self._written(mock_run).splitlines() == [
            "SSL_CERT_FILE=/home/dev/.hermes/ca-bundle.pem",
            "GITHUB_TOKEN=ghp_new",
        ]

    @patch("llm.provision.hermes_vm.run")
    @patch.object(HermesVmManager, "_configure_local_llm")
    def test_local_llm_calls_configure_local_llm(self, mock_local, mock_run):
        """When provider is local-llm, _configure_local_llm is called with all_cfg."""
        from llm.settings import AuthSettings, ProxySettings, ServerSettings, Settings

        all_cfg = Settings(
            auth=AuthSettings(api_key="test-api-key"),
            proxy=ProxySettings(enabled=False),
            server=ServerSettings(port=8080),
        )
        cfg = _make_cfg(provider="local-llm")
        HermesVmManager._configure_credentials(self._mgr(), cfg, all_cfg)
        mock_local.assert_called_once_with(all_cfg)

    @patch.object(HermesVmManager, "_configure_local_llm")
    def test_local_llm_skipped_without_all_cfg(self, mock_local):
        """When all_cfg is None, _configure_local_llm should not be called."""
        HermesVmManager._configure_credentials(self._mgr(), _make_cfg(provider="local-llm"))
        mock_local.assert_not_called()


# ── _configure_local_llm ───────────────────────────────────────────────────────


class TestConfigureLocalLlm:
    """Tests for HermesVmManager._configure_local_llm."""

    def _mgr(self):
        mgr = HermesVmManager.__new__(HermesVmManager)
        mgr.container = "hermes"
        mgr.uid = 1000
        mgr.gid = 1000
        mgr._hermes_run = MagicMock()
        mgr._install_ca_bundle = MagicMock()
        return mgr

    @staticmethod
    def _cfg(**proxy_kwargs):
        from llm.settings import (
            AuthSettings,
            ClientSettings,
            ProxySettings,
            ServerSettings,
            Settings,
        )

        return Settings(
            auth=AuthSettings(api_key="test-api-key"),
            client=ClientSettings(server_url=proxy_kwargs.pop("server_url", "")),
            proxy=ProxySettings(**proxy_kwargs),
            server=ServerSettings(port=8080),
        )

    @patch("llm.provision.hermes_vm.subprocess.run")
    def test_adds_etc_hosts_entry_proxy_disabled(self, mock_subprocess):
        """The 'local-llm' hostname must resolve inside the VM via /etc/hosts."""
        mgr = self._mgr()
        HermesVmManager._configure_local_llm(mgr, self._cfg(enabled=False, lan_ip="192.168.1.50"))

        hosts_calls = [c for c in mock_subprocess.call_args_list if "/etc/hosts" in str(c)]
        assert len(hosts_calls) == 1
        assert "192.168.1.50 local-llm" in " ".join(str(a) for a in hosts_calls[0].args[0])

    @patch("llm.provision.hermes_vm.subprocess.run")
    def test_adds_etc_hosts_entry_uses_server_url_host(self, mock_subprocess):
        """When client.server_url is set, its hostname is used for the hosts entry."""
        mgr = self._mgr()
        cfg = self._cfg(server_url="https://10.0.0.5:8443/v1", enabled=True, lan_ip="192.168.1.50")
        HermesVmManager._configure_local_llm(mgr, cfg)

        hosts_calls = [c for c in mock_subprocess.call_args_list if "/etc/hosts" in str(c)]
        assert len(hosts_calls) == 1
        assert "10.0.0.5 local-llm" in " ".join(str(a) for a in hosts_calls[0].args[0])

    @patch("llm.provision.hermes_vm.subprocess.run")
    def test_ca_bundle_installed_only_when_proxy_enabled(self, mock_subprocess):
        """A plain HTTP endpoint needs no CA cert."""
        mgr = self._mgr()
        HermesVmManager._configure_local_llm(mgr, self._cfg(enabled=False, lan_ip="192.168.1.50"))
        mgr._install_ca_bundle.assert_not_called()

        mgr = self._mgr()
        HermesVmManager._configure_local_llm(
            mgr, self._cfg(enabled=True, lan_ip="192.168.1.50", cert_path="/etc/ssl/x/cert.pem")
        )
        mgr._install_ca_bundle.assert_called_once_with("/etc/ssl/x/cert.pem")

    @patch("llm.provision.hermes_vm.subprocess.run")
    def test_configures_hermes_custom_provider(self, mock_subprocess):
        """Hermes must be configured with custom provider, base_url, api_key, and default model."""
        mgr = self._mgr()
        cfg = self._cfg(enabled=False, lan_ip="192.168.1.50")
        cfg.models.active = "qwen3.6-35b-moe-q4"
        HermesVmManager._configure_local_llm(mgr, cfg)

        mgr._hermes_run.assert_any_call(
            "config", "set", "model.provider", "custom", desc="set custom provider (local)"
        )
        mgr._hermes_run.assert_any_call(
            "config", "set", "model.base_url", "http://local-llm:8080/v1", desc="set local base url"
        )
        mgr._hermes_run.assert_any_call(
            "config", "set", "model.api_key", "test-api-key", desc="set local api key"
        )
        mgr._hermes_run.assert_any_call(
            "config", "set", "model.default", "qwen3.6-35b-moe-q4", desc="set local default model"
        )

    @patch("llm.provision.hermes_vm.subprocess.run")
    def test_configures_hermes_keyless_fallback(self, mock_subprocess):
        """When auth.api_key is empty, 'no-key-required' placeholder is passed."""
        from llm.settings import AuthSettings

        mgr = self._mgr()
        cfg = self._cfg(enabled=False, lan_ip="192.168.1.50")
        cfg.auth = AuthSettings(api_key="")
        HermesVmManager._configure_local_llm(mgr, cfg)

        mgr._hermes_run.assert_any_call(
            "config", "set", "model.api_key", "no-key-required", desc="set local api key"
        )


class TestInstallCaBundle:
    """Setup must fail loudly when the VM cannot be made to trust the proxy."""

    def _mgr(self):
        mgr = HermesVmManager.__new__(HermesVmManager)
        mgr.container = "hermes"
        mgr.uid = 1000
        mgr.gid = 1000
        mgr._write_env_vars = MagicMock()
        return mgr

    @patch("llm.provision.hermes_vm.subprocess.run")
    def test_raises_when_cert_push_fails(self, mock_subprocess):
        """Without the cert every request to the proxy fails, so don't report success."""
        mock_subprocess.return_value = MagicMock(returncode=1, stderr="no such file")

        with pytest.raises(RuntimeError, match="Failed to copy CA cert"):
            HermesVmManager._install_ca_bundle(self._mgr(), "/etc/ssl/local-llm/cert.pem")

    @patch("llm.provision.hermes_vm.run_capture")
    @patch("llm.provision.hermes_vm.subprocess.run")
    def test_raises_when_bundle_build_fails(self, mock_subprocess, mock_capture):
        """A pushed cert that isn't in the trust bundle is still unusable."""
        mock_subprocess.return_value = MagicMock(returncode=0, stderr="")
        mock_capture.return_value = MagicMock(returncode=1, stderr="certifi missing")

        mgr = self._mgr()
        with pytest.raises(RuntimeError, match="combined CA bundle"):
            HermesVmManager._install_ca_bundle(mgr, "/etc/ssl/local-llm/cert.pem")
        mgr._write_env_vars.assert_not_called()

    @patch("llm.provision.hermes_vm.run_capture")
    @patch("llm.provision.hermes_vm.subprocess.run")
    def test_points_ssl_cert_file_at_the_bundle(self, mock_subprocess, mock_capture):
        """httpx reads SSL_CERT_FILE, so it must point at the combined bundle."""
        mock_subprocess.return_value = MagicMock(returncode=0, stderr="")
        mock_capture.return_value = MagicMock(returncode=0, stderr="")

        mgr = self._mgr()
        HermesVmManager._install_ca_bundle(mgr, "/etc/ssl/local-llm/cert.pem")

        updates = mgr._write_env_vars.call_args.args[0]
        assert updates["SSL_CERT_FILE"].endswith("/.hermes/ca-bundle.pem")


# ── get_status ─────────────────────────────────────────────────────────────────


class TestGetStatus:
    """Tests for HermesVmManager.get_status()."""

    @pytest.fixture(autouse=True)
    def _default_config(self, monkeypatch):
        from llm.settings import Settings

        monkeypatch.setattr("llm.provision.hermes_vm.load_config", lambda: Settings())

    def _build_mgr(self):
        """Create a HermesVmManager with __init__ bypassed."""
        mgr = HermesVmManager.__new__(HermesVmManager)
        mgr.container = "hermes"
        mgr.uid = 1000
        mgr.gid = 1000
        return mgr

    @patch("llm.provision.hermes_vm.subprocess.run")
    def test_vm_not_running_returns_defaults(self, mock_run):
        """When VM is not running, all fields should be 'unknown' or 0."""
        mgr = self._build_mgr()
        mock_lxc = MagicMock()
        mock_lxc.stdout = json.dumps([{"status": "Stopped"}])
        mock_run.return_value = mock_lxc

        result = mgr.get_status()
        assert result["vm"] == "Stopped"
        assert result["gateway"] == "unknown"
        assert result["version"] == "unknown"
        assert result["uptime"] == "0"
        assert result["credentials_ok"] == "False"

    @patch("llm.provision.hermes_vm.subprocess.run")
    @patch("llm.provision.hermes_vm._cexec")
    def test_vm_running_gathering_all_fields(self, mock_cexec, mock_run):
        """When VM is Running, all fields should be populated."""
        mgr = self._build_mgr()

        def make_mock(stdout="", returncode=0):
            m = MagicMock()
            m.stdout = stdout
            m.returncode = returncode
            return m

        mock_run.side_effect = [
            make_mock(json.dumps([{"status": "Running"}])),  # lxc list
            make_mock("active\n", 0),  # systemctl is-active
            make_mock("ActiveEnterTimestampEpoch=1700000000\n"),  # systemctl show
            make_mock("hermes 3.0.0\n"),  # hermes --version
            make_mock("openrouter\n"),  # hermes config get
            make_mock("", 0),  # curl probe
        ]

        def cexec_side_effect(*args):
            return list(args)

        mock_cexec.side_effect = cexec_side_effect

        result = mgr.get_status()
        assert result["vm"] == "Running"
        assert result["gateway"] == "active"
        assert result["version"] == "hermes 3.0.0"
        assert int(result["uptime"]) > 0
        assert result["credentials_ok"] == "True"

    @patch("llm.provision.hermes_vm.subprocess.run")
    @patch("llm.provision.hermes_vm._cexec")
    def test_gateway_inactive(self, mock_cexec, mock_run):
        """Gateway in inactive state should still report correctly."""
        mgr = self._build_mgr()

        def make_mock(stdout="", returncode=0):
            m = MagicMock()
            m.stdout = stdout
            m.returncode = returncode
            return m

        mock_run.side_effect = [
            make_mock(json.dumps([{"status": "Running"}])),
            make_mock("inactive\n", 3),
            make_mock("ActiveEnterTimestampEpoch=0\n"),
            make_mock("hermes 2.5.0\n"),
            make_mock("openrouter\n"),
            make_mock("", 0),
        ]

        def cexec_side_effect(*args):
            return list(args)

        mock_cexec.side_effect = cexec_side_effect

        result = mgr.get_status()
        assert result["vm"] == "Running"
        assert result["gateway"] == "inactive"
        assert result["version"] == "hermes 2.5.0"
        assert result["credentials_ok"] == "True"

    @patch("llm.provision.hermes_vm.subprocess.run")
    @patch("llm.provision.hermes_vm._cexec")
    def test_credentials_invalid_when_curl_fails(self, mock_cexec, mock_run):
        """When curl to OpenRouter returns non-zero, credentials_ok is False."""
        mgr = self._build_mgr()

        def make_mock(stdout="", returncode=0):
            m = MagicMock()
            m.stdout = stdout
            m.returncode = returncode
            return m

        mock_run.side_effect = [
            make_mock(json.dumps([{"status": "Running"}])),
            make_mock("active\n", 0),
            make_mock("ActiveEnterTimestampEpoch=1700000000\n"),
            make_mock("hermes 3.0.0\n"),
            make_mock("openrouter\n"),
            make_mock("", 1),  # curl fails
        ]

        def cexec_side_effect(*args):
            return list(args)

        mock_cexec.side_effect = cexec_side_effect

        result = mgr.get_status()
        assert result["credentials_ok"] == "False"

    @patch("llm.provision.hermes_vm.subprocess.run")
    @patch("llm.provision.hermes_vm._cexec")
    def test_local_llm_probe_http(self, mock_cexec, mock_run):
        """When provider is custom and proxy is disabled, probe http endpoint."""
        from unittest.mock import patch as real_patch

        from llm.settings import AuthSettings, ProxySettings, ServerSettings, Settings

        mgr = self._build_mgr()

        def make_mock(stdout="", returncode=0):
            m = MagicMock()
            m.stdout = stdout
            m.returncode = returncode
            return m

        mock_run.side_effect = [
            make_mock(json.dumps([{"status": "Running"}])),
            make_mock("active\n", 0),
            make_mock("ActiveEnterTimestampEpoch=1700000000\n"),
            make_mock("hermes 3.0.0\n"),
            make_mock("custom\n"),
            make_mock("", 0),  # curl probe for local-llm
        ]

        def cexec_side_effect(*args):
            return list(args)

        mock_cexec.side_effect = cexec_side_effect

        with real_patch("llm.provision.hermes_vm.load_config") as mock_load:
            mock_load.return_value = Settings(
                auth=AuthSettings(api_key="test-key"),
                proxy=ProxySettings(enabled=False),
                server=ServerSettings(port=8080),
            )
            result = mgr.get_status()

        # Should have probed the local HTTP endpoint
        http_calls = [c for c in mock_run.call_args_list if "local-llm" in str(c)]
        assert len(http_calls) >= 1
        cmd_str = " ".join(str(a) for a in http_calls[0].args[0])
        assert "http://local-llm:8080" in cmd_str
        assert result["credentials_ok"] == "True"

    @patch("llm.provision.hermes_vm.subprocess.run")
    def test_bad_json_in_lxc_list(self, mock_run):
        """When lxc list returns invalid JSON, vm should default to 'unknown'."""
        mgr = self._build_mgr()
        mock_lxc = MagicMock()
        mock_lxc.stdout = "not json"
        mock_run.return_value = mock_lxc

        result = mgr.get_status()
        assert result["vm"] == "unknown"
        assert result["gateway"] == "unknown"
        assert result["version"] == "unknown"
        assert result["uptime"] == "0"
        assert result["credentials_ok"] == "False"

    @patch("llm.provision.hermes_vm.subprocess.run")
    @patch("llm.provision.hermes_vm._cexec")
    def test_version_empty_defaults_to_unknown(self, mock_cexec, mock_run):
        """When hermes --version returns empty, version is 'unknown'."""
        mgr = self._build_mgr()

        def make_mock(stdout="", returncode=0):
            m = MagicMock()
            m.stdout = stdout
            m.returncode = returncode
            return m

        mock_run.side_effect = [
            make_mock(json.dumps([{"status": "Running"}])),
            make_mock("active\n", 0),
            make_mock("ActiveEnterTimestampEpoch=1700000000\n"),
            make_mock(""),  # Empty version
            make_mock("openrouter\n"),
            make_mock("", 0),
        ]

        def cexec_side_effect(*args):
            return list(args)

        mock_cexec.side_effect = cexec_side_effect

        result = mgr.get_status()
        assert result["version"] == "unknown"

    @patch("llm.provision.hermes_vm.subprocess.run")
    @patch("llm.provision.hermes_vm._cexec")
    def test_return_keys_match_expected(self, mock_cexec, mock_run):
        """Verify get_status returns exactly the expected keys."""
        mgr = self._build_mgr()

        def make_mock(stdout="", returncode=0):
            m = MagicMock()
            m.stdout = stdout
            m.returncode = returncode
            return m

        mock_run.side_effect = [
            make_mock(json.dumps([{"status": "Running"}])),
            make_mock("active\n", 0),
            make_mock("ActiveEnterTimestampEpoch=1700000000\n"),
            make_mock("hermes 3.0.0\n"),
            make_mock("openrouter\n"),
            make_mock("", 0),
        ]

        def cexec_side_effect(*args):
            return list(args)

        mock_cexec.side_effect = cexec_side_effect

        result = mgr.get_status()
        expected_keys = {
            "vm",
            "gateway",
            "version",
            "uptime",
            "provider",
            "credentials_ok",
            "mattermost",
        }
        assert set(result.keys()) == expected_keys
        assert all(isinstance(v, str) for v in result.values())

    @patch("llm.provision.hermes_vm.subprocess.run")
    @patch("llm.provision.hermes_vm._cexec")
    def test_mattermost_status_connected(self, mock_cexec, mock_run, monkeypatch):
        """When Mattermost is configured and curl succeeds, status is connected."""
        from llm.settings import Settings

        mm_cfg = Settings(
            hermes=HermesSettings(
                mattermost_url="https://mm.example.com",
                mattermost_token="mm-tok",
            )
        )
        monkeypatch.setattr("llm.provision.hermes_vm.load_config", lambda: mm_cfg)

        mgr = self._build_mgr()

        def make_mock(stdout="", returncode=0):
            m = MagicMock()
            m.stdout = stdout
            m.returncode = returncode
            return m

        mock_run.side_effect = [
            make_mock(json.dumps([{"status": "Running"}])),
            make_mock("active\n", 0),
            make_mock("ActiveEnterTimestampEpoch=1700000000\n"),
            make_mock("hermes 3.0.0\n"),
            make_mock("openrouter\n"),
            make_mock("", 0),  # openrouter probe
            make_mock("", 0),  # mattermost probe succeeds
        ]

        def cexec_side_effect(*args):
            return list(args)

        mock_cexec.side_effect = cexec_side_effect

        result = mgr.get_status()
        assert result["mattermost"] == "connected"

    @patch("llm.provision.hermes_vm.subprocess.run")
    @patch("llm.provision.hermes_vm._cexec")
    def test_mattermost_status_unreachable(self, mock_cexec, mock_run, monkeypatch):
        """When Mattermost curl returns non-zero, status is unreachable."""
        from llm.settings import Settings

        mm_cfg = Settings(
            hermes=HermesSettings(
                mattermost_url="https://mm.example.com",
                mattermost_token="mm-tok",
            )
        )
        monkeypatch.setattr("llm.provision.hermes_vm.load_config", lambda: mm_cfg)

        mgr = self._build_mgr()

        def make_mock(stdout="", returncode=0):
            m = MagicMock()
            m.stdout = stdout
            m.returncode = returncode
            return m

        mock_run.side_effect = [
            make_mock(json.dumps([{"status": "Running"}])),
            make_mock("active\n", 0),
            make_mock("ActiveEnterTimestampEpoch=1700000000\n"),
            make_mock("hermes 3.0.0\n"),
            make_mock("openrouter\n"),
            make_mock("", 0),  # openrouter probe
            make_mock("", 1),  # mattermost probe fails
        ]

        def cexec_side_effect(*args):
            return list(args)

        mock_cexec.side_effect = cexec_side_effect

        result = mgr.get_status()
        assert result["mattermost"] == "unreachable"


# ── Formatting helpers ─────────────────────────────────────────────────────────


class TestFormatCredentials:
    """Tests for hermes._format_credentials."""

    def _import_helpers(self):
        return _format_credentials

    def test_valid(self):
        f = self._import_helpers()
        label, color = f("True")
        assert label == "valid"
        assert color == "green"

    def test_invalid(self):
        f = self._import_helpers()
        label, color = f("False")
        assert label == "invalid"
        assert color == "yellow"

    def test_unknown(self):
        f = self._import_helpers()
        label, color = f("unknown")
        assert label == "unknown"
        assert color == "red"


class TestFormatLocalLlm:
    """Tests for hermes._format_local_llm."""

    def test_connected(self):
        label, color = _format_local_llm("True")
        assert label == "connected"
        assert color == "green"

    def test_unreachable(self):
        label, color = _format_local_llm("False")
        assert label == "unreachable"
        assert color == "yellow"

    def test_unknown(self):
        label, color = _format_local_llm("unknown")
        assert label == "unknown"
        assert color == "red"


class TestStatusCommand:
    """Tests for hermes.status() displaying local LLM connectivity."""

    @patch("llm.hermes.console")
    @patch("llm.provision.exec.container_exists", return_value=True)
    @patch.object(HermesVmManager, "get_status")
    def test_status_shows_local_llm_line_for_local_provider(self, mock_get_status, mock_exists, mock_console):
        """When hermes is configured for the local llama-server, show 'Local LLM'."""
        from llm.hermes import status

        mock_get_status.return_value = {
            "vm": "Running",
            "gateway": "active",
            "version": "hermes 3.0.0",
            "uptime": "60",
            "provider": "openai",
            "credentials_ok": "True",
        }
        status()

        printed = " ".join(str(c.args[0]) for c in mock_console.print.call_args_list)
        assert "Local LLM:   connected" in printed
        assert "Credentials:" not in printed

    @patch("llm.hermes.console")
    @patch("llm.provision.exec.container_exists", return_value=True)
    @patch.object(HermesVmManager, "get_status")
    def test_status_shows_credentials_line_for_openrouter(self, mock_get_status, mock_exists, mock_console):
        """When hermes is configured for OpenRouter, show 'Credentials'."""
        from llm.hermes import status

        mock_get_status.return_value = {
            "vm": "Running",
            "gateway": "active",
            "version": "hermes 3.0.0",
            "uptime": "60",
            "provider": "openrouter",
            "credentials_ok": "False",
        }
        status()

        printed = " ".join(str(c.args[0]) for c in mock_console.print.call_args_list)
        assert "Credentials: invalid" in printed
        assert "Local LLM:" not in printed


# ── _merge_env_file ───────────────────────────────────────────────────────────


class TestMergeEnvFile:
    """The env file is rewritten wholesale, so the merge must not lose entries."""

    def test_creates_file_from_nothing(self):
        assert _merge_env_file("", {"A": "1"}) == "A=1\n"

    def test_replaces_in_place_preserving_order(self):
        existing = "A=old\nB=keep\n"
        assert _merge_env_file(existing, {"A": "new"}) == "A=new\nB=keep\n"

    def test_appends_unknown_keys(self):
        assert _merge_env_file("A=1\n", {"B": "2"}) == "A=1\nB=2\n"

    def test_preserves_unrelated_lines(self):
        existing = "# comment\nSSL_CERT_FILE=/x/y.pem\n"
        merged = _merge_env_file(existing, {"GITHUB_TOKEN": "t"})
        assert merged == "# comment\nSSL_CERT_FILE=/x/y.pem\nGITHUB_TOKEN=t\n"

    def test_values_containing_equals_are_kept_whole(self):
        assert _merge_env_file("", {"A": "b=c=d"}) == "A=b=c=d\n"

    def test_is_idempotent(self):
        first = _merge_env_file("", {"A": "1", "B": "2"})
        assert _merge_env_file(first, {"A": "1", "B": "2"}) == first

    def test_ends_with_a_single_trailing_newline(self):
        """Repeated writes must not accumulate blank lines at the end of the file."""
        assert _merge_env_file("A=1\n\n\n", {"A": "1"}) == "A=1\n"


# ── Timezone ──────────────────────────────────────────────────────────────────


class TestTimezone:
    """Tests for timezone configuration in Hermes VM."""

    def _mgr(self, timezone: str = "America/Chicago"):
        mgr = HermesVmManager.__new__(HermesVmManager)
        mgr.container = "hermes"
        mgr.uid = 1000
        mgr.gid = 1000
        mgr.timezone = timezone
        return mgr

    @patch("llm.provision.vm.run")
    def test_set_timezone_default(self, mock_run):
        mgr = self._mgr("America/Chicago")
        mgr._set_timezone()
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "timedatectl set-timezone America/Chicago" in cmd[-1]
        assert "ln -sf /usr/share/zoneinfo/America/Chicago /etc/localtime" in cmd[-1]
        assert "echo 'America/Chicago' > /etc/timezone" in cmd[-1]

    def test_set_timezone_raises_for_unrecognized(self):
        mgr = self._mgr("US/chicago")
        with pytest.raises(ValueError, match="Unknown IANA timezone"):
            mgr._set_timezone()

    @patch.object(HermesVmManager, "_set_timezone")
    @patch("llm.provision.hermes_vm.run_with_retry")
    @patch("llm.provision.hermes_vm.run")
    @patch.object(HermesVmManager, "_hermes_run")
    @patch.object(HermesVmManager, "_configure_credentials")
    @patch("llm.provision.hermes_vm.subprocess.run")
    def test_refresh_configures_timezone(
        self, mock_sub, mock_creds, mock_hermes_run, mock_run, mock_retry, mock_set_tz
    ):
        mock_sub.return_value = MagicMock(returncode=1)
        mgr = self._mgr()
        cfg = HermesSettings(timezone="America/New_York")
        mgr.refresh(cfg)
        mock_set_tz.assert_called_once_with("America/New_York")
