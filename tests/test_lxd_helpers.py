from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock

import pytest
import typer

import llm.lxd as lxd


def _make_proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    p = MagicMock()
    p.returncode = returncode
    p.stdout = stdout
    p.stderr = stderr
    return p


def test_run_capture_returns_result(monkeypatch):
    expected = _make_proc(0, "hello")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: expected)
    result = lxd.run_capture(["lxc", "list"])
    assert result is expected


def test_container_exists_true(monkeypatch):
    data = json.dumps([{"name": "craft-llm-1", "status": "Running"}])
    monkeypatch.setattr(lxd, "run_capture", lambda _: _make_proc(0, data))
    assert lxd.container_exists("craft-llm-1") is True


def test_container_exists_false_empty_list(monkeypatch):
    data = json.dumps([])
    monkeypatch.setattr(lxd, "run_capture", lambda _: _make_proc(0, data))
    assert lxd.container_exists("craft-llm-1") is False


def test_container_exists_false_different_name(monkeypatch):
    data = json.dumps([{"name": "craft-llm-2", "status": "Running"}])
    monkeypatch.setattr(lxd, "run_capture", lambda _: _make_proc(0, data))
    assert lxd.container_exists("craft-llm-1") is False


def test_container_exists_false_on_nonzero_returncode(monkeypatch):
    monkeypatch.setattr(lxd, "run_capture", lambda _: _make_proc(1, ""))
    assert lxd.container_exists("craft-llm-1") is False


def test_check_passes():
    result = lxd.check("always passes", lambda: None)
    assert result is True


def test_check_fails_on_exception():
    def boom():
        raise AssertionError("bad")

    result = lxd.check("always fails", boom)
    assert result is False


def test_check_fails_on_arbitrary_exception():
    result = lxd.check("runtime error", lambda: 1 / 0)
    assert result is False


def test_wait_for_container_succeeds_immediately(monkeypatch):
    exec_proc = _make_proc(0)
    cloud_proc = _make_proc(0, stdout='{"status": "done"}')
    call_count = {"n": 0}

    def fake_run(cmd, **kwargs):
        call_count["n"] += 1
        if "true" in cmd:
            return exec_proc
        return cloud_proc

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(lxd.time, "sleep", lambda _: None)
    lxd.wait_for_container("craft-llm-1", timeout=10)


def test_wait_for_container_retries_then_succeeds(monkeypatch):
    attempts = {"exec": 0, "cloud": 0}

    def fake_run(cmd, **kwargs):
        if "true" in cmd:
            attempts["exec"] += 1
            code = 1 if attempts["exec"] < 2 else 0
            return _make_proc(code)
        attempts["cloud"] += 1
        return _make_proc(0, stdout='{"status": "done"}')

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(lxd.time, "sleep", lambda _: None)
    lxd.wait_for_container("craft-llm-1", timeout=10)
    assert attempts["exec"] >= 2


def test_wait_for_container_timeout(monkeypatch):
    _calls = [0]

    def fake_time():
        _calls[0] += 1
        return 0.0 if _calls[0] == 1 else 9999.0

    monkeypatch.setattr(lxd.time, "time", fake_time)
    monkeypatch.setattr(lxd.time, "sleep", lambda _: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(1))

    with pytest.raises(typer.Exit):
        lxd.wait_for_container("craft-llm-1", timeout=1)
