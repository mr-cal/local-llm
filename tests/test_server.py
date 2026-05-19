"""Tests for the server management module."""

from __future__ import annotations

import signal
import subprocess
import time
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


# ── PID file management ──────────────────────────────────────────────────────


class TestPidFile:
    def test_pid_file_path(self):
        assert server._pid_file().name == ".server.pid"

    def test_log_file_path(self):
        assert server._log_file().name == ".server.log"

    def test_read_pid_returns_none_when_no_file(self, tmp_path, mocker):

        mocker.patch.object(server, "_PID_FILE", tmp_path / "nonexistent.pid")
        mocker.patch("os.kill")
        result = server._read_pid()
        assert result is None

    def test_read_pid_returns_pid_when_running(self, tmp_path, fake_pid_file, mocker):

        fake_pid_file.write_text("99999")

        def fake_kill(pid, sig):
            if sig == 0:
                pass  # simulate process exists
            return None

        mocker.patch.object(server, "_PID_FILE", fake_pid_file)
        mocker.patch("os.kill", fake_kill)
        result = server._read_pid()
        assert result == 99999

    def test_read_pid_returns_none_when_process_gone(self, tmp_path, fake_pid_file, mocker):

        fake_pid_file.write_text("99999")

        def fake_kill(pid, sig):
            if sig == 0:
                raise ProcessLookupError()
            return None

        mocker.patch.object(server, "_PID_FILE", fake_pid_file)
        mocker.patch("os.kill", fake_kill)
        result = server._read_pid()
        assert result is None

    def test_read_pid_handles_corrupt_file(self, tmp_path, fake_pid_file, mocker):

        fake_pid_file.write_text("not-a-pid")

        mocker.patch.object(server, "_PID_FILE", fake_pid_file)
        mocker.patch("os.kill")
        result = server._read_pid()
        assert result is None

    def test_read_pid_handles_permission_error(self, tmp_path, fake_pid_file, mocker):

        fake_pid_file.write_text("99999")

        def fake_kill(pid, sig):
            if sig == 0:
                raise PermissionError()
            return None

        mocker.patch.object(server, "_PID_FILE", fake_pid_file)
        mocker.patch("os.kill", fake_kill)
        result = server._read_pid()
        assert result is None


# ── start command ─────────────────────────────────────────────────────────────


class TestStartCommand:
    def test_start_no_server_configured(self, tmp_config_server, fake_console, monkeypatch):
        config, tmp_path = tmp_config_server
        content = config.read_text().replace('"llama-server"', '""')
        config.write_text(content)
        with pytest.raises(typer.Exit):
            server.start(wait=0)

    def test_start_server_already_running(self, tmp_config_server, fake_console, monkeypatch, mocker):
        config, tmp_path = tmp_config_server
        pid_file = tmp_path / ".server.pid"
        pid_file.write_text("12345")

        def fake_kill(pid, sig):
            if sig == 0:
                pass  # process exists
            return None

        mocker.patch("os.kill", fake_kill)
        with pytest.raises(typer.Exit):
            server.start(wait=0)

    def test_start_model_not_found(self, tmp_config_server, fake_console, monkeypatch):
        config, tmp_path = tmp_config_server
        # Remove the model file so start fails
        (tmp_path / "models" / "model.gguf").unlink()
        with pytest.raises(typer.Exit):
            server.start(wait=0)

    def test_start_success(self, tmp_config_server, fake_console, monkeypatch, _make_proc, mocker):
        config, tmp_path = tmp_config_server

        proc = MagicMock()
        proc.pid = 54321

        def fake_popen(cmd, **kw):
            return proc

        def fake_kill(pid, sig):
            if sig == 0:
                raise ProcessLookupError()
            return None

        def fake_run(cmd, **kw):
            return _make_proc(0, "")

        mocker.patch("subprocess.Popen", fake_popen)
        mocker.patch("os.kill", fake_kill)
        mocker.patch("time.monotonic", return_value=999999)
        mocker.patch("subprocess.run", fake_run)
        server.start(wait=0)

        # Check PID file was written
        pid_file = tmp_path / ".server.pid"
        assert pid_file.exists()
        assert pid_file.read_text().strip() == "54321"

    def test_start_with_extra_args(self, tmp_config_server, fake_console, monkeypatch, _make_proc, mocker):
        config, tmp_path = tmp_config_server
        # Add extra args
        content = config.read_text()
        content = content.replace('extra_args = []', 'extra_args = ["--jinja", "--flash-attn"]')
        config.write_text(content)

        proc = MagicMock()
        proc.pid = 54321
        started_cmd = None

        def fake_popen(cmd, **kw):
            nonlocal started_cmd
            started_cmd = cmd
            return proc

        def fake_kill(pid, sig):
            if sig == 0:
                raise ProcessLookupError()
            return None

        def fake_run(cmd, **kw):
            return _make_proc(0, "")

        mocker.patch("subprocess.Popen", fake_popen)
        mocker.patch("os.kill", fake_kill)
        mocker.patch("time.monotonic", return_value=999999)
        mocker.patch("subprocess.run", fake_run)
        server.start(wait=0)

        assert started_cmd is not None
        assert "--jinja" in started_cmd
        assert "--flash-attn" in started_cmd

    def test_start_binary_not_found(self, tmp_config_server, fake_console, monkeypatch, mocker):
        config, tmp_path = tmp_config_server
        content = config.read_text().replace('"llama-server"', '"/nonexistent/llama-server"')
        config.write_text(content)

        def fake_popen(cmd, **kw):
            raise FileNotFoundError("no such file")

        mocker.patch("subprocess.Popen", fake_popen)
        with pytest.raises(typer.Exit):
            server.start(wait=0)

    def test_start_waits_for_ready(
        self, tmp_config_server, fake_console, mock_httpx_get, monkeypatch, _make_proc, mocker
    ):
        config, tmp_path = tmp_config_server

        proc = MagicMock()
        proc.pid = 54321

        def fake_popen(cmd, **kw):
            return proc

        def fake_kill(pid, sig):
            if sig == 0:
                raise ProcessLookupError()
            return None

        def fake_run(cmd, **kw):
            return _make_proc(0, "")

        ready_calls = []

        def fake_monotonic():
            ready_calls.append(time.time())
            return 999999  # always "past deadline" for timeout test

        mocker.patch("subprocess.Popen", fake_popen)
        mocker.patch("os.kill", fake_kill)
        mocker.patch("time.monotonic", fake_monotonic)
        mocker.patch("subprocess.run", fake_run)
        server.start(wait=1)

        # Should have checked readiness
        assert len(ready_calls) > 0


# ── stop command ──────────────────────────────────────────────────────────────


class TestStopCommand:
    def test_stop_not_running(self, tmp_config_server, fake_console, mocker):
        mocker.patch.object(server, "_read_pid", return_value=None)
        with pytest.raises(typer.Exit):
            server.stop()

    def test_stop_success(self, tmp_config_server, fake_console, mocker):
        _, tmp_path = tmp_config_server
        pid_file = tmp_path / ".server.pid"
        pid_file.write_text("12345")
        stopped = False

        def fake_kill(pid, sig):
            if sig == signal.SIGTERM:
                pass
            if sig == 0:
                nonlocal stopped
                if not stopped:
                    stopped = True
                    raise ProcessLookupError()
                return None

        mocker.patch.object(server, "_read_pid", return_value=12345)
        mocker.patch("os.kill", fake_kill)
        server.stop()

        assert not pid_file.exists()

    def test_stop_removes_pid_file(self, tmp_config_server, fake_console, mocker):
        _, tmp_path = tmp_config_server
        pid_file = tmp_path / ".server.pid"
        pid_file.write_text("12345")

        def fake_kill(pid, sig):
            if sig == 0:
                raise ProcessLookupError()
            return None

        mocker.patch.object(server, "_read_pid", return_value=12345)
        mocker.patch("os.kill", fake_kill)
        server.stop()

        assert not pid_file.exists()


# ── restart command ───────────────────────────────────────────────────────────


class TestRestartCommand:
    def test_restart_calls_stop_then_start(
        self, tmp_config_server, fake_console, mock_httpx_get, monkeypatch, mocker
    ):
        config, tmp_path = tmp_config_server

        pid_file = tmp_path / ".server.pid"
        pid_file.write_text("12345")
        proc = MagicMock()
        proc.pid = 54321

        stop_calls = []
        start_calls = []

        def fake_stop():
            stop_calls.append(True)
            pid_file.unlink(missing_ok=True)

        def fake_start(wait=5):
            start_calls.append(True)

        def fake_popen(cmd, **kw):
            return proc

        def fake_kill(pid, sig):
            if sig == 0:
                raise ProcessLookupError()
            return None

        mocker.patch.object(server, "_read_pid", return_value=12345)
        mocker.patch.object(server, "_nginx_is_active", return_value=False)
        mocker.patch.object(server, "_nginx_start", return_value=True)
        mocker.patch.object(server, "stop", fake_stop)
        mocker.patch.object(server, "start", fake_start)
        mocker.patch("subprocess.Popen", fake_popen)
        mocker.patch("os.kill", fake_kill)
        mocker.patch("time.monotonic", return_value=999999)
        server.restart()

        assert stop_calls
        assert start_calls


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

        mocker.patch.object(server, "_read_pid", return_value=12345)
        mocker.patch.object(server, "_nginx_is_active", return_value=True)
        server.status()

    def test_status_stopped(self, tmp_config_server, fake_console, mocker):
        config, tmp_path = tmp_config_server

        mocker.patch.object(server, "_read_pid", return_value=None)
        mocker.patch.object(server, "_nginx_is_active", return_value=False)
        server.status()


# ── logs command ─────────────────────────────────────────────────────────────


class TestLogsCommand:
    def test_logs_no_file(self, tmp_config_server, mocker):
        _, tmp_path = tmp_config_server
        mocker.patch.object(server, "_log_file", return_value=tmp_path / "nonexistent.log")
        with pytest.raises(typer.Exit):
            server.logs(lines=50)

    def test_logs_show_lines(self, tmp_config_server, fake_console, mocker):
        _, tmp_path = tmp_config_server
        log_file = tmp_path / ".server.log"
        log_file.write_text("line 1\nline 2\nline 3\n")

        captured = []

        def fake_run(cmd, **kw):
            captured.append(list(cmd))

        mocker.patch.object(server, "_log_file", return_value=log_file)
        mocker.patch("subprocess.run", fake_run)
        server.logs(lines=2)

        assert captured
        assert "-2" in captured[0]
        assert str(log_file) in captured[0]

    def test_logs_follow(self, tmp_config_server, mocker):
        _, tmp_path = tmp_config_server
        log_file = tmp_path / ".server.log"
        log_file.write_text("line 1\n")

        captured = []

        def fake_run(cmd, **kw):
            captured.append(list(cmd))

        mocker.patch.object(server, "_log_file", return_value=log_file)
        mocker.patch("subprocess.run", fake_run)
        server.logs(follow=True)

        assert captured
        assert "-f" in captured[0]
