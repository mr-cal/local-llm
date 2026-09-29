"""Tests for the server management module."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest
import typer

import llm.server as server

# ── nginx helpers ──────────────────────────────────────────────────────────────


class TestNginxHelpers:
    def test_nginx_is_active_true(self, monkeypatch, _make_proc):
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(0, "active"))
        assert server._nginx_is_active() is True

    def test_nginx_is_active_false(self, monkeypatch, _make_proc):
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(1, "inactive"))
        assert server._nginx_is_active() is False

    def test_nginx_start_success(self, monkeypatch, _make_proc):
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(0, ""))
        assert server._nginx_start() is True

    def test_nginx_start_failure(self, monkeypatch, _make_proc):
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(1, ""))
        assert server._nginx_start() is False

    def test_nginx_reload_success(self, monkeypatch, _make_proc):
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(0, ""))
        assert server._nginx_reload() is True

    def test_nginx_reload_failure(self, monkeypatch, _make_proc):
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(1, ""))
        assert server._nginx_reload() is False

    def test_nginx_stop_success(self, monkeypatch, _make_proc):
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(0, ""))
        assert server._nginx_stop() is True

    def test_nginx_stop_failure(self, monkeypatch, _make_proc):
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(1, ""))
        assert server._nginx_stop() is False

    def test_nginx_ensure_running_starts_when_inactive(self, monkeypatch, fake_console, _make_proc):
        start_called = []
        reload_called = []

        def fake_run(cmd, **kw):
            cmd_str = " ".join(str(c) for c in cmd)
            if "is-active" in cmd_str:
                return _make_proc(1, "inactive")
            if "start" in cmd_str and "nginx" in cmd_str:
                start_called.append(cmd)
                return _make_proc(0, "")
            if "reload" in cmd_str and "nginx" in cmd_str:
                reload_called.append(cmd)
                return _make_proc(0, "")
            return _make_proc(0, "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        server._nginx_ensure_running()

        assert start_called
        assert not reload_called

    def test_nginx_ensure_running_reloads_when_active(self, monkeypatch, fake_console, _make_proc):
        reload_called = []

        def fake_run(cmd, **kw):
            cmd_str = " ".join(str(c) for c in cmd)
            if "is-active" in cmd_str:
                return _make_proc(0, "active")
            if "reload" in cmd_str and "nginx" in cmd_str:
                reload_called.append(cmd)
                return _make_proc(0, "")
            return _make_proc(0, "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        server._nginx_ensure_running()

        assert reload_called


# ── process discovery ────────────────────────────────────────────────────────


def _ss_output(port: int, pid: int) -> str:
    """Build `ss -tlnp` output showing llama-server listening on *port*."""
    return (
        "State  Recv-Q Send-Q Local Address:Port Peer Address:Port Process\n"
        f"LISTEN 0      4096         0.0.0.0:{port}         0.0.0.0:*     "
        f'users:(("llama-server",pid={pid},fd=3))\n'
    )


class TestServerPid:
    """llama-server is supervised by systemd, so systemd is the source of truth."""

    def test_returns_unit_main_pid(self, mocker):
        mocker.patch.object(server, "_unit_main_pid", return_value=4242)
        assert server._server_pid(8080) == 4242

    def test_falls_back_to_port_probe(self, mocker):
        """A server started outside the unit must still be detected."""
        mocker.patch.object(server, "_unit_main_pid", return_value=None)
        mocker.patch("subprocess.run", return_value=MagicMock(stdout=_ss_output(8080, 777)))
        mocker.patch("os.kill", lambda pid, sig: None)
        assert server._server_pid(8080) == 777

    def test_ignores_dead_process_from_port_probe(self, mocker):
        mocker.patch.object(server, "_unit_main_pid", return_value=None)
        mocker.patch("subprocess.run", return_value=MagicMock(stdout=_ss_output(8080, 777)))
        mocker.patch("os.kill", MagicMock(side_effect=ProcessLookupError))
        assert server._server_pid(8080) is None

    def test_returns_none_when_nothing_is_running(self, mocker):
        mocker.patch.object(server, "_unit_main_pid", return_value=None)
        mocker.patch("subprocess.run", return_value=MagicMock(stdout=""))
        assert server._server_pid(8080) is None

    def test_no_port_and_no_unit_pid(self, mocker):
        mocker.patch.object(server, "_unit_main_pid", return_value=None)
        assert server._server_pid() is None


class TestUnitMainPid:
    """systemd reports MainPID=0 for an inactive unit."""

    def test_parses_main_pid(self, mocker, _make_proc):
        mocker.patch("subprocess.run", return_value=_make_proc(0, "4242"))
        assert server._unit_main_pid() == 4242

    def test_zero_means_not_running(self, mocker, _make_proc):
        mocker.patch("subprocess.run", return_value=_make_proc(0, "0"))
        assert server._unit_main_pid() is None

    def test_unknown_unit_returns_none(self, mocker, _make_proc):
        mocker.patch("subprocess.run", return_value=_make_proc(1, ""))
        assert server._unit_main_pid() is None

    def test_unparseable_output_returns_none(self, mocker, _make_proc):
        mocker.patch("subprocess.run", return_value=_make_proc(0, "nonsense"))
        assert server._unit_main_pid() is None


# ── start command ─────────────────────────────────────────────────────────────


class TestStartCommand:
    def test_start_no_server_configured(self, tmp_config_server, fake_console, monkeypatch):
        config, tmp_path = tmp_config_server
        content = config.read_text().replace('"llama-server"', '""')
        config.write_text(content)
        with pytest.raises(typer.Exit):
            server.start(wait=0)

    def test_start_requires_installed_unit(self, tmp_config_server, fake_console, mocker):
        """Without the unit there is nothing to start, so say so rather than fail opaquely."""
        mocker.patch.object(server.proc, "ensure_sudo")
        mocker.patch.object(server, "_llm_server_unit_installed", return_value=False)
        systemctl = mocker.patch.object(server, "_llm_server_systemctl")

        with pytest.raises(typer.Exit):
            server.start(wait=0)
        systemctl.assert_not_called()

    def test_start_server_already_running(self, tmp_config_server, fake_console, mocker):
        mocker.patch.object(server.proc, "ensure_sudo")
        mocker.patch.object(server, "_llm_server_unit_installed", return_value=True)
        mocker.patch.object(server, "_server_pid", return_value=12345)
        systemctl = mocker.patch.object(server, "_llm_server_systemctl")

        with pytest.raises(typer.Exit):
            server.start(wait=0)
        systemctl.assert_not_called()

    def test_start_model_not_found(self, tmp_config_server, fake_console, mocker):
        config, tmp_path = tmp_config_server
        (tmp_path / "models" / "model.gguf").unlink()
        mocker.patch.object(server.proc, "ensure_sudo")
        mocker.patch.object(server, "_llm_server_unit_installed", return_value=True)
        mocker.patch.object(server, "_server_pid", return_value=None)

        with pytest.raises(typer.Exit):
            server.start(wait=0)

    def test_start_uses_systemctl(self, tmp_config_server, fake_console, mocker):
        """The unit owns the command line; start must not spawn llama-server itself."""
        mocker.patch.object(server.proc, "ensure_sudo")
        mocker.patch.object(server, "_llm_server_unit_installed", return_value=True)
        mocker.patch.object(server, "_server_pid", return_value=None)
        mocker.patch.object(server, "_unit_main_pid", return_value=54321)
        mocker.patch.object(server, "_nginx_ensure_running")
        mocker.patch.object(server, "_start_monitor")
        popen = mocker.patch("subprocess.Popen")
        systemctl = mocker.patch.object(server, "_llm_server_systemctl", return_value=True)

        server.start(wait=0)

        systemctl.assert_called_once_with("start")
        popen.assert_not_called()

    def test_start_exits_when_systemctl_fails(self, tmp_config_server, fake_console, mocker):
        mocker.patch.object(server.proc, "ensure_sudo")
        mocker.patch.object(server, "_llm_server_unit_installed", return_value=True)
        mocker.patch.object(server, "_server_pid", return_value=None)
        mocker.patch.object(server, "_llm_server_systemctl", return_value=False)
        nginx = mocker.patch.object(server, "_nginx_ensure_running")

        with pytest.raises(typer.Exit):
            server.start(wait=0)
        nginx.assert_not_called()

    def test_start_waits_for_ready(self, tmp_config_server, fake_console, mock_httpx_get, mocker):
        mocker.patch.object(server.proc, "ensure_sudo")
        mocker.patch.object(server, "_llm_server_unit_installed", return_value=True)
        mocker.patch.object(server, "_server_pid", return_value=None)
        mocker.patch.object(server, "_unit_main_pid", return_value=54321)
        mocker.patch.object(server, "_llm_server_systemctl", return_value=True)
        mocker.patch.object(server, "_nginx_ensure_running")
        wait_for_ready = mocker.patch.object(server, "_wait_until_ready")

        server.start(wait=3)

        assert wait_for_ready.call_args.args[1] == 3


# ── stop command ──────────────────────────────────────────────────────────────


class TestStopCommand:
    def test_stop_not_running(self, tmp_config_server, fake_console, mocker):
        mocker.patch.object(server.proc, "ensure_sudo")
        mocker.patch.object(server, "_server_pid", return_value=None)
        mocker.patch.object(server, "_nginx_is_active", return_value=False)
        with pytest.raises(typer.Exit):
            server.stop()

    def test_stop_nginx_when_server_already_stopped(self, tmp_config_server, fake_console, mocker):
        """nginx is still running after a previous failed stop; should stop it."""
        mocker.patch.object(server.proc, "ensure_sudo")
        mocker.patch.object(server, "_server_pid", return_value=None)
        mocker.patch.object(server, "_nginx_is_active", return_value=True)
        mocker.patch.object(server, "_stop_monitor")
        nginx_stop = mocker.patch.object(server, "_nginx_stop", return_value=True)

        server.stop()

        nginx_stop.assert_called_once()

    def test_stop_actually_stops_the_unit(self, tmp_config_server, fake_console, mocker):
        """The old implementation only disabled the unit, leaving the server running."""
        mocker.patch.object(server.proc, "ensure_sudo")
        mocker.patch.object(server, "_server_pid", return_value=12345)
        mocker.patch.object(server, "_nginx_is_active", return_value=False)
        mocker.patch.object(server, "_stop_monitor")
        systemctl = mocker.patch.object(server, "_llm_server_systemctl", return_value=True)

        server.stop()

        systemctl.assert_called_once_with("stop")

    def test_stop_does_not_disable_the_unit(self, tmp_config_server, fake_console, mocker):
        """Enablement controls start-on-boot and is owned by `server apply`."""
        mocker.patch.object(server.proc, "ensure_sudo")
        mocker.patch.object(server, "_server_pid", return_value=12345)
        mocker.patch.object(server, "_nginx_is_active", return_value=False)
        mocker.patch.object(server, "_stop_monitor")
        systemctl = mocker.patch.object(server, "_llm_server_systemctl", return_value=True)

        server.stop()

        assert "disable" not in [c.args[0] for c in systemctl.call_args_list]

    def test_stop_warns_about_a_server_it_cannot_stop(self, tmp_config_server, fake_console, mocker):
        """systemctl reports success for an inactive unit, so a manually
        started llama-server would otherwise be reported as stopped."""
        mocker.patch.object(server.proc, "ensure_sudo")
        mocker.patch.object(server, "_server_pid", return_value=12345)
        mocker.patch.object(server, "_nginx_is_active", return_value=False)
        mocker.patch.object(server, "_stop_monitor")
        mocker.patch.object(server, "_llm_server_systemctl", return_value=True)

        server.stop()

        assert any("still running" in line for line in fake_console)

    def test_stop_stops_the_monitor(self, tmp_config_server, fake_console, mocker):
        mocker.patch.object(server.proc, "ensure_sudo")
        mocker.patch.object(server, "_server_pid", return_value=12345)
        mocker.patch.object(server, "_nginx_is_active", return_value=False)
        mocker.patch.object(server, "_llm_server_systemctl", return_value=True)
        stop_monitor = mocker.patch.object(server, "_stop_monitor")

        server.stop()

        stop_monitor.assert_called_once()


class TestSystemctlHelpers:
    def test_runs_the_requested_action(self, mocker, _make_proc):
        run = mocker.patch.object(server.proc, "sudo", return_value=_make_proc(0, ""))
        assert server._llm_server_systemctl("restart") is True
        run.assert_called_once_with(["systemctl", "restart", "llm-server"])

    def test_returns_false_on_failure(self, mocker, _make_proc):
        mocker.patch.object(server.proc, "sudo", return_value=_make_proc(1, ""))
        assert server._llm_server_systemctl("start") is False

    def test_unit_installed_reflects_the_unit_path(self, mocker):
        mocker.patch.object(server._UNIT_PATH.__class__, "exists", lambda self: True)
        assert server._llm_server_unit_installed() is True


# ── restart command ───────────────────────────────────────────────────────────


class TestRestartCommand:
    def test_restart_uses_a_single_systemctl_restart(self, tmp_config_server, fake_console, mocker):
        """systemctl restart is atomic; stop-then-start raced with the restart policy."""
        mocker.patch.object(server.proc, "ensure_sudo")
        mocker.patch.object(server, "_llm_server_unit_installed", return_value=True)
        mocker.patch.object(server, "_unit_main_pid", return_value=54321)
        mocker.patch.object(server, "_stop_monitor")
        mocker.patch.object(server, "_wait_until_ready")
        mocker.patch.object(server, "_nginx_ensure_running")
        systemctl = mocker.patch.object(server, "_llm_server_systemctl", return_value=True)

        server.restart()

        systemctl.assert_called_once_with("restart")

    def test_restart_ensures_nginx_is_running(self, tmp_config_server, fake_console, mocker):
        mocker.patch.object(server.proc, "ensure_sudo")
        mocker.patch.object(server, "_llm_server_unit_installed", return_value=True)
        mocker.patch.object(server, "_unit_main_pid", return_value=54321)
        mocker.patch.object(server, "_stop_monitor")
        mocker.patch.object(server, "_wait_until_ready")
        mocker.patch.object(server, "_llm_server_systemctl", return_value=True)
        nginx = mocker.patch.object(server, "_nginx_ensure_running")

        server.restart()

        nginx.assert_called_once()

    def test_restart_exits_when_systemctl_fails(self, tmp_config_server, fake_console, mocker):
        mocker.patch.object(server.proc, "ensure_sudo")
        mocker.patch.object(server, "_llm_server_unit_installed", return_value=True)
        mocker.patch.object(server, "_stop_monitor")
        mocker.patch.object(server, "_llm_server_systemctl", return_value=False)
        nginx = mocker.patch.object(server, "_nginx_ensure_running")

        with pytest.raises(typer.Exit):
            server.restart()
        nginx.assert_not_called()


# ── status command ────────────────────────────────────────────────────────────


class TestStatusCommand:
    def test_status_no_server_configured(self, tmp_config_server, fake_console, monkeypatch):
        config, tmp_path = tmp_config_server
        content = config.read_text().replace('"llama-server"', '""')
        config.write_text(content)
        # Should not crash
        server.status()

    def test_status_running(self, tmp_config_server, fake_console, mocker):
        config, tmp_path = tmp_config_server
        # Use the default model name for the "running" status test
        content = config.read_text().replace('"model.gguf"', '"Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"')
        config.write_text(content)

        mocker.patch.object(server, "_server_pid", return_value=12345)
        mocker.patch.object(server, "_nginx_is_active", return_value=True)
        server.status()

    def test_status_stopped(self, tmp_config_server, fake_console, mocker):
        config, tmp_path = tmp_config_server

        mocker.patch.object(server, "_server_pid", return_value=None)
        mocker.patch.object(server, "_nginx_is_active", return_value=False)
        server.status()


class TestUptimeHelpers:
    def test_process_uptime_seconds_unknown_pid(self):
        assert server._process_uptime_seconds(2_147_483_647) is None

    def test_process_uptime_seconds_live_pid(self):
        import os

        assert isinstance(server._process_uptime_seconds(os.getpid()), int)


# ── logs command ─────────────────────────────────────────────────────────────


class TestLogsCommand:
    """Logs come from the journal now that systemd owns the process."""

    def test_reads_the_journal_for_the_unit(self, fake_console, mocker):
        run = mocker.patch("subprocess.run")

        server.logs(lines=25)

        cmd = run.call_args.args[0]
        assert cmd[:3] == ["journalctl", "-u", "llm-server"]
        assert "-n" in cmd
        assert "25" in cmd
        assert "-f" not in cmd

    def test_follow_passes_f(self, fake_console, mocker):
        run = mocker.patch("subprocess.run")

        server.logs(follow=True)

        assert "-f" in run.call_args.args[0]
