"""Tests for the Hermes VM manager module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from llm.config import HermesSettings
from llm.hermes import _format_credentials, _format_uptime
from llm.hermes_vm import HermesVmManager

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

    @patch("llm.hermes_vm.subprocess.run")
    def test_no_credentials_skips(self, mock_run):
        """When no credentials are set, the method should log a warning and skip."""
        mgr = MagicMock()
        mgr.container = "hermes"
        HermesVmManager._configure_credentials(mgr, _make_cfg())
        mock_run.assert_not_called()

    @patch("llm.hermes_vm.subprocess.run")
    def test_single_credential_openrouter(self, mock_run):
        """When only openrouter_key is set, provider config and env write occur."""
        mgr = MagicMock()
        mgr.container = "hermes"
        mgr.uid = 1000
        mgr.gid = 1000
        cfg = _make_cfg(provider="openrouter", openrouter_key="sk-or-v1-test")
        HermesVmManager._configure_credentials(mgr, cfg)
        # One call for setting provider, one for writing env
        assert mock_run.call_count == 2
        provider_call = mock_run.call_args_list[0]
        cmd = provider_call.args[0]
        cmd_str = " ".join(str(c) for c in cmd)
        assert "hermes" in cmd_str
        assert "config" in cmd_str
        assert "set" in cmd_str
        assert "model.provider" in cmd_str
        assert "openrouter" in cmd_str

    @patch("llm.hermes_vm.subprocess.run")
    def test_all_credentials_written(self, mock_run):
        """When all credentials are set, all are written to .env."""
        mgr = MagicMock()
        mgr.container = "hermes"
        mgr.uid = 1000
        mgr.gid = 1000
        cfg = _make_cfg(
            provider="openrouter",
            openrouter_key="sk-or-v1-test",
            telegram_token="123:ABC",
            telegram_allowed_users="987654321",
            github_token="ghp_test",
        )
        HermesVmManager._configure_credentials(mgr, cfg)
        # One call for provider config, one for env write
        assert mock_run.call_count == 2

    @patch("llm.hermes_vm.subprocess.run")
    def test_github_uses_has_github(self, mock_run):
        """Verify github_token uses has_github() guard (not raw truthiness)."""
        mgr = MagicMock()
        mgr.container = "hermes"
        mgr.uid = 1000
        mgr.gid = 1000
        # Whitespace-only token should be treated as not set
        cfg = _make_cfg(github_token="   ")
        HermesVmManager._configure_credentials(mgr, cfg)
        mock_run.assert_not_called()

    @patch("llm.hermes_vm.subprocess.run")
    def test_github_token_in_env_when_set(self, mock_run):
        """Verify GITHUB_TOKEN appears in env write when token is set."""
        mgr = MagicMock()
        mgr.container = "hermes"
        mgr.uid = 1000
        mgr.gid = 1000
        cfg = _make_cfg(github_token="ghp_real")
        HermesVmManager._configure_credentials(mgr, cfg)
        env_call = mock_run.call_args_list[-1]
        cmd_str = " ".join(str(a) for a in env_call.args[0])
        assert "GITHUB_TOKEN=" in cmd_str

    @patch.object(HermesVmManager, "_configure_local_llm")
    def test_local_llm_calls_configure_local_llm(self, mock_local):
        """When provider is local-llm, _configure_local_llm is called with all_cfg."""
        from llm.config import AuthSettings, ProxySettings, ServerSettings, Settings

        mgr = HermesVmManager.__new__(HermesVmManager)
        mgr.container = "hermes"
        mgr.uid = 1000
        mgr.gid = 1000
        all_cfg = Settings(
            auth=AuthSettings(api_key="test-api-key"),
            proxy=ProxySettings(enabled=False),
            server=ServerSettings(port=8080),
        )
        cfg = _make_cfg(provider="local-llm")
        HermesVmManager._configure_credentials(mgr, cfg, all_cfg)
        mock_local.assert_called_once_with(all_cfg)

    @patch.object(HermesVmManager, "_configure_local_llm")
    def test_local_llm_skipped_without_all_cfg(self, mock_local):
        """When all_cfg is None, _configure_local_llm should not be called."""
        mgr = HermesVmManager.__new__(HermesVmManager)
        mgr.container = "hermes"
        mgr.uid = 1000
        mgr.gid = 1000
        cfg = _make_cfg(provider="local-llm")
        HermesVmManager._configure_credentials(mgr, cfg)
        mock_local.assert_not_called()


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
        expected_keys = {"vm", "gateway", "version", "uptime", "credentials_ok"}
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
