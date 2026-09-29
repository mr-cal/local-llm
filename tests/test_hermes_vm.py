"""Tests for the Hermes VM manager module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from llm.config import HermesSettings
from llm.hermes import _format_credentials, _format_local_llm, _format_uptime
from llm.hermes_vm import HermesVmManager, _merge_env_file

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_cfg(
    provider: str = "local-llm",
    openrouter_key: str = "",
    telegram_token: str = "",
    telegram_allowed_users: str = "",
    github_token: str = "",
) -> HermesSettings:
    return HermesSettings(
        provider=provider,
        openrouter_key=openrouter_key,
        telegram_token=telegram_token,
        telegram_allowed_users=telegram_allowed_users,
        github_token=github_token,
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
        assert mock_run.call_count == 1
        return mock_run.call_args.kwargs["input"]

    @patch("llm.hermes_vm.run")
    def test_no_credentials_skips(self, mock_run):
        """When no credentials are set, the method should log a warning and skip."""
        HermesVmManager._configure_credentials(self._mgr(), _make_cfg())
        mock_run.assert_not_called()

    @patch("llm.hermes_vm.run")
    def test_single_credential_openrouter(self, mock_run):
        """Only the configured credential is written, and the provider is set."""
        mgr = self._mgr()
        cfg = _make_cfg(provider="openrouter", openrouter_key="sk-or-v1-test")
        HermesVmManager._configure_credentials(mgr, cfg)

        assert self._written(mock_run) == "OPENROUTER_API_KEY=sk-or-v1-test\n"
        mgr._hermes_run.assert_called_once_with(
            "config", "set", "model.provider", "openrouter", desc="set openrouter provider"
        )

    @patch("llm.hermes_vm.run")
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

    @patch("llm.hermes_vm.run")
    def test_github_uses_has_github(self, mock_run):
        """Verify github_token uses has_github() guard (not raw truthiness)."""
        HermesVmManager._configure_credentials(self._mgr(), _make_cfg(github_token="   "))
        mock_run.assert_not_called()

    @patch("llm.hermes_vm.run")
    def test_github_token_in_env_when_set(self, mock_run):
        """Verify GITHUB_TOKEN appears in env write when token is set."""
        HermesVmManager._configure_credentials(self._mgr(), _make_cfg(github_token="ghp_real"))
        assert self._written(mock_run) == "GITHUB_TOKEN=ghp_real\n"

    @patch("llm.hermes_vm.run")
    def test_secrets_never_appear_in_the_command(self, mock_run):
        """Credentials travel over stdin, never in argv where `ps` can see them."""
        cfg = _make_cfg(
            provider="openrouter",
            openrouter_key="sk-or-v1-secret",
            telegram_token="123:SECRETTOKEN",
            github_token="ghp_secretvalue",
        )
        HermesVmManager._configure_credentials(self._mgr(), cfg)

        argv = " ".join(str(a) for a in mock_run.call_args.args[0])
        assert "sk-or-v1-secret" not in argv
        assert "123:SECRETTOKEN" not in argv
        assert "ghp_secretvalue" not in argv

    @patch("llm.hermes_vm.run")
    def test_shell_metacharacters_survive_verbatim(self, mock_run):
        """A token containing shell syntax must be stored literally, not executed."""
        nasty = "tok'en|rm -rf /;$(whoami)"
        HermesVmManager._configure_credentials(self._mgr(), _make_cfg(github_token=nasty))
        assert self._written(mock_run) == f"GITHUB_TOKEN={nasty}\n"

    @patch("llm.hermes_vm.run")
    def test_existing_keys_replaced_unrelated_preserved(self, mock_run):
        """Re-running updates known keys in place and leaves other entries alone."""
        existing = "SSL_CERT_FILE=/home/dev/.hermes/ca-bundle.pem\nGITHUB_TOKEN=ghp_old\n"
        HermesVmManager._configure_credentials(self._mgr(existing), _make_cfg(github_token="ghp_new"))
        assert self._written(mock_run).splitlines() == [
            "SSL_CERT_FILE=/home/dev/.hermes/ca-bundle.pem",
            "GITHUB_TOKEN=ghp_new",
        ]

    @patch.object(HermesVmManager, "_configure_local_llm")
    def test_local_llm_calls_configure_local_llm(self, mock_local):
        """When provider is local-llm, _configure_local_llm is called with all_cfg."""
        from llm.config import AuthSettings, ProxySettings, ServerSettings, Settings

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

    @patch("llm.hermes_vm.subprocess.run")
    def test_adds_etc_hosts_entry_proxy_disabled(self, mock_subprocess):
        """The 'local-llm' hostname must resolve inside the VM via /etc/hosts."""
        from llm.config import AuthSettings, ProxySettings, ServerSettings, Settings

        mgr = HermesVmManager.__new__(HermesVmManager)
        mgr.container = "hermes"
        mgr.uid = 1000
        mgr.gid = 1000
        mgr._hermes_run = MagicMock()
        all_cfg = Settings(
            auth=AuthSettings(api_key="test-api-key"),
            proxy=ProxySettings(enabled=False, lan_ip="192.168.1.50"),
            server=ServerSettings(port=8080),
        )
        HermesVmManager._configure_local_llm(mgr, all_cfg)

        hosts_calls = [c for c in mock_subprocess.call_args_list if "/etc/hosts" in str(c)]
        assert len(hosts_calls) == 1
        cmd_str = " ".join(str(a) for a in hosts_calls[0].args[0])
        assert "192.168.1.50 local-llm" in cmd_str

    @patch("llm.hermes_vm.subprocess.run")
    def test_adds_etc_hosts_entry_uses_server_url_host(self, mock_subprocess):
        """When client.server_url is set, its hostname is used for the hosts entry."""
        from llm.config import AuthSettings, ClientSettings, ProxySettings, ServerSettings, Settings

        mgr = HermesVmManager.__new__(HermesVmManager)
        mgr.container = "hermes"
        mgr.uid = 1000
        mgr.gid = 1000
        mgr._hermes_run = MagicMock()
        all_cfg = Settings(
            auth=AuthSettings(api_key="test-api-key"),
            client=ClientSettings(server_url="https://10.0.0.5:8443/v1"),
            proxy=ProxySettings(enabled=True, lan_ip="192.168.1.50"),
            server=ServerSettings(port=8080),
        )
        HermesVmManager._configure_local_llm(mgr, all_cfg)

        hosts_calls = [c for c in mock_subprocess.call_args_list if "/etc/hosts" in str(c)]
        assert len(hosts_calls) == 1
        cmd_str = " ".join(str(a) for a in hosts_calls[0].args[0])
        assert "10.0.0.5 local-llm" in cmd_str


# ── get_status ─────────────────────────────────────────────────────────────────


class TestGetStatus:
    """Tests for HermesVmManager.get_status()."""

    def _build_mgr(self):
        """Create a HermesVmManager with __init__ bypassed."""
        mgr = HermesVmManager.__new__(HermesVmManager)
        mgr.container = "hermes"
        mgr.uid = 1000
        mgr.gid = 1000
        return mgr

    @patch("llm.hermes_vm.subprocess.run")
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

    @patch("llm.hermes_vm.subprocess.run")
    @patch("llm.hermes_vm._cexec")
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

    @patch("llm.hermes_vm.subprocess.run")
    @patch("llm.hermes_vm._cexec")
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

    @patch("llm.hermes_vm.subprocess.run")
    @patch("llm.hermes_vm._cexec")
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

    @patch("llm.hermes_vm.subprocess.run")
    @patch("llm.hermes_vm._cexec")
    def test_local_llm_probe_http(self, mock_cexec, mock_run):
        """When provider is openai and proxy is disabled, probe http endpoint."""
        from unittest.mock import patch as real_patch

        from llm.config import AuthSettings, ProxySettings, ServerSettings, Settings

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
            make_mock("openai\n"),
            make_mock("", 0),  # curl probe for local-llm
        ]

        def cexec_side_effect(*args):
            return list(args)

        mock_cexec.side_effect = cexec_side_effect

        with real_patch("llm.config.load_config") as mock_load:
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

    @patch("llm.hermes_vm.subprocess.run")
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

    @patch("llm.hermes_vm.subprocess.run")
    @patch("llm.hermes_vm._cexec")
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

    @patch("llm.hermes_vm.subprocess.run")
    @patch("llm.hermes_vm._cexec")
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
        expected_keys = {"vm", "gateway", "version", "uptime", "provider", "credentials_ok"}
        assert set(result.keys()) == expected_keys
        assert all(isinstance(v, str) for v in result.values())


# ── Formatting helpers ─────────────────────────────────────────────────────────


class TestFormatUptime:
    """Tests for hermes._format_uptime."""

    def _import_helpers(self):
        return _format_uptime

    def test_zero_seconds(self):
        f = self._import_helpers()
        assert f(0) == "0s"

    def test_seconds(self):
        f = self._import_helpers()
        assert f(45) == "45s"

    def test_minutes(self):
        f = self._import_helpers()
        assert f(120) == "2m 0s"

    def test_minutes_with_remainder(self):
        f = self._import_helpers()
        assert f(125) == "2m 5s"

    def test_hours(self):
        f = self._import_helpers()
        assert f(3600) == "1h 0m 0s"

    def test_hours_with_minutes(self):
        f = self._import_helpers()
        assert f(7200) == "2h 0m 0s"

    def test_hours_minutes_seconds(self):
        f = self._import_helpers()
        assert f(3725) == "1h 2m 5s"

    def test_days(self):
        f = self._import_helpers()
        assert f(86400) == "1d 0h 0m"

    def test_days_hours(self):
        f = self._import_helpers()
        assert f(180000) == "2d 2h 0m"

    def test_large_days(self):
        f = self._import_helpers()
        assert f(2592000) == "30d 0h 0m"


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
    @patch("llm.lxd.container_exists", return_value=True)
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
    @patch("llm.lxd.container_exists", return_value=True)
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
