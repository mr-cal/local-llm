from __future__ import annotations

import json
import os
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


def test_github_mount_check_skipped_when_not_in_mounts(monkeypatch):
    subprocess_calls = []

    def fake_subprocess_run(cmd, **kwargs):
        subprocess_calls.append(list(cmd))
        return _make_proc(0, "")

    monkeypatch.setattr(lxd, "MOUNTS", [("dev", "/home/user/dev", "/home/user/dev")])
    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    container_data = json.dumps([{"name": "craft-llm-1", "status": "Running"}])

    def fake_run_capture(cmd):
        return _make_proc(0, container_data)

    monkeypatch.setattr(lxd, "run_capture", fake_run_capture)
    monkeypatch.setattr(lxd, "MAKE_SETUP_DIRS", [])
    monkeypatch.setattr(os.path, "exists", lambda _: False)

    fake_stat = MagicMock()
    fake_stat.st_uid = lxd.HOST_UID
    fake_stat.st_gid = lxd.HOST_GID
    monkeypatch.setattr(os, "stat", lambda _: fake_stat)
    monkeypatch.setattr(os, "unlink", lambda _, **kw: None)

    def fake_full_run(cmd, **kwargs):
        cmd_str = " ".join(str(x) for x in cmd)
        if "list" in cmd_str:
            return _make_proc(0, container_data)
        if "stat" in cmd_str:
            return _make_proc(0, lxd.CONTAINER_USER)
        if "ls" in cmd_str and "dev/craft" in cmd_str:
            return _make_proc(0, "snapcraft\n")
        if "id" in cmd_str:
            return _make_proc(0, lxd.CONTAINER_USER)
        if "cat" in cmd_str and "opencode.json" in cmd_str:
            return _make_proc(0, json.dumps({"provider": {"local-llm": {}}}))
        if "cat" in cmd_str and "lsp-config" in cmd_str:
            return _make_proc(0, json.dumps({"lspServers": {"python": {"command": "pylsp"}}}))
        if "touch" in cmd_str:
            return _make_proc(0, "")
        return _make_proc(0, "")

    monkeypatch.setattr(subprocess, "run", fake_full_run)

    lxd.run_tests("craft-llm-1")
    github_calls = [c for c in subprocess_calls if ".github" in " ".join(str(x) for x in c)]
    assert not github_calls


def test_github_mount_check_runs_when_in_mounts(monkeypatch):
    monkeypatch.setattr(lxd, "MOUNTS", [("github", "/home/user/.github", "/home/user/.github")])

    github_checked = []

    def fake_subprocess_run(cmd, **kwargs):
        cmd_str = " ".join(str(x) for x in cmd)
        if ".github" in cmd_str:
            github_checked.append(True)
            return _make_proc(0, "")
        return _make_proc(0, "")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    def t_github_mount():
        if not any(name == "github" for name, _, _ in lxd.MOUNTS):
            return
        subprocess.run(
            ["lxc", "exec", "craft-llm-1", "--", "ls", f"{lxd.CONTAINER_HOME}/.github"],
            capture_output=True,
            check=True,
        )

    result = lxd.check(".github mount works", t_github_mount)
    assert result is True
    assert github_checked


def test_run_tests_all_pass(monkeypatch):
    container_data = json.dumps([{"name": "craft-llm-1", "status": "Running"}])

    def fake_subprocess_run(cmd, **kwargs):
        cmd_str = " ".join(str(x) for x in cmd)
        if "list" in cmd_str:
            return _make_proc(0, container_data)
        if "stat" in cmd_str:
            return _make_proc(0, lxd.CONTAINER_USER)
        if "ls" in cmd_str and "dev/craft" in cmd_str:
            return _make_proc(0, "snapcraft\n")
        if "id" in cmd_str:
            return _make_proc(0, lxd.CONTAINER_USER)
        if "cat" in cmd_str and "opencode.json" in cmd_str:
            return _make_proc(0, json.dumps({"provider": {"local-llm": {}}}))
        if "cat" in cmd_str and "lsp-config" in cmd_str:
            cfg = json.dumps({"lspServers": {"python": {"command": "pylsp"}}})
            return _make_proc(0, cfg)
        if "touch" in cmd_str:
            return _make_proc(0, "")
        return _make_proc(0, "")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    fake_stat = MagicMock()
    fake_stat.st_uid = lxd.HOST_UID
    fake_stat.st_gid = lxd.HOST_GID
    monkeypatch.setattr(os, "stat", lambda _: fake_stat)
    monkeypatch.setattr(os, "unlink", lambda _, **kw: None)
    monkeypatch.setattr(os.path, "exists", lambda _: False)
    monkeypatch.setattr(lxd, "MAKE_SETUP_DIRS", [])

    lxd.run_tests("craft-llm-1")


def test_run_tests_fails_exits(monkeypatch):
    container_data = json.dumps([{"name": "craft-llm-1", "status": "Stopped"}])

    def fake_subprocess_run(cmd, **kwargs):
        if "list" in " ".join(str(x) for x in cmd):
            return _make_proc(0, container_data)
        return _make_proc(0, "")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(lxd, "MAKE_SETUP_DIRS", [])
    monkeypatch.setattr(os.path, "exists", lambda _: False)

    with pytest.raises(typer.Exit) as exc_info:
        lxd.run_tests("craft-llm-1")
    assert exc_info.value.exit_code == 1
