from __future__ import annotations

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


def test_run_craft_setup_tests_skips_missing_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(lxd, "MAKE_SETUP_DIRS", [str(tmp_path / "nonexistent")])
    lxd.run_craft_setup_tests("craft-llm-1")


def test_run_craft_setup_tests_checks_venv(monkeypatch, tmp_path):
    existing_dir = tmp_path / "my-project"
    existing_dir.mkdir()
    monkeypatch.setattr(lxd, "MAKE_SETUP_DIRS", [str(existing_dir)])

    def fake_subprocess_run(cmd, **kwargs):
        if ".venv" in " ".join(str(x) for x in cmd):
            return _make_proc(1, "")
        return _make_proc(0, "")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    with pytest.raises(typer.Exit) as exc_info:
        lxd.run_craft_setup_tests("craft-llm-1")
    assert exc_info.value.exit_code == 1


def test_pylsp_lsp_config_structure():
    cfg = lxd.PYLSP_LSP_CONFIG
    assert "lspServers" in cfg
    assert "python" in cfg["lspServers"]
    server = cfg["lspServers"]["python"]
    assert server["command"] == "pylsp"
    assert "fileExtensions" in server
    assert ".py" in server["fileExtensions"]


def test_container_prefix():
    assert lxd.CONTAINER_PREFIX == "craft-llm"


def test_container_uid_gid():
    assert lxd.CONTAINER_UID == 1000
    assert lxd.CONTAINER_GID == 1000


def _setup_crafts_with_config(monkeypatch, tmp_path):
    cfg = tmp_path / "craft-dirs.toml"
    cfg.write_text("dirs = []\n")
    monkeypatch.setattr(lxd, "CRAFT_DIRS_CONFIG", cfg)
    monkeypatch.setattr(lxd, "MAKE_SETUP_DIRS", [])


def test_setup_crafts_errors_when_container_missing(monkeypatch, tmp_path):
    _setup_crafts_with_config(monkeypatch, tmp_path)
    monkeypatch.setattr(lxd, "container_exists", lambda _: False)

    with pytest.raises(typer.Exit) as exc_info:
        lxd.setup_crafts(1)
    assert exc_info.value.exit_code == 1


def test_setup_crafts_inits_config_when_missing(monkeypatch, tmp_path):
    cfg = tmp_path / "craft-dirs.toml"
    monkeypatch.setattr(lxd, "CRAFT_DIRS_CONFIG", cfg)

    with pytest.raises(typer.Exit) as exc_info:
        lxd.setup_crafts(1)
    assert exc_info.value.exit_code == 0
    assert cfg.exists()
    assert "dirs" in cfg.read_text()


def test_setup_crafts_calls_make_setup_and_tests(monkeypatch, tmp_path):
    _setup_crafts_with_config(monkeypatch, tmp_path)
    make_calls = []
    test_calls = []

    monkeypatch.setattr(lxd, "container_exists", lambda _: True)
    monkeypatch.setattr(lxd, "container_is_vm", lambda _: False)
    monkeypatch.setattr(lxd, "run_make_setup", lambda c, uid, gid: make_calls.append((c, uid, gid)))
    monkeypatch.setattr(lxd, "run_craft_setup_tests", lambda c: test_calls.append(c))

    lxd.setup_crafts(1)

    assert make_calls == [("craft-llm-1", lxd.CONTAINER_UID, lxd.CONTAINER_GID)]
    assert test_calls == ["craft-llm-1"]


def test_setup_crafts_uses_host_uid_for_vm(monkeypatch, tmp_path):
    _setup_crafts_with_config(monkeypatch, tmp_path)
    make_calls = []

    monkeypatch.setattr(lxd, "container_exists", lambda _: True)
    monkeypatch.setattr(lxd, "container_is_vm", lambda _: True)
    monkeypatch.setattr(lxd, "run_make_setup", lambda c, uid, gid: make_calls.append((uid, gid)))
    monkeypatch.setattr(lxd, "run_craft_setup_tests", lambda _: None)

    lxd.setup_crafts(1)

    assert make_calls == [(lxd.HOST_UID, lxd.HOST_GID)]


def test_create_prints_setup_crafts_hint(monkeypatch):
    printed = []

    monkeypatch.setattr(lxd, "container_exists", lambda _: False)
    monkeypatch.setattr(lxd, "create_container", lambda *a, **kw: None)
    monkeypatch.setattr(lxd, "configure_idmap", lambda *a, **kw: None)
    monkeypatch.setattr(lxd, "add_mounts", lambda *a, **kw: None)
    monkeypatch.setattr(lxd, "install_packages", lambda *a, **kw: None)
    monkeypatch.setattr(lxd, "install_pylsp", lambda *a, **kw: None)
    monkeypatch.setattr(lxd, "run_tests", lambda *a, **kw: None)
    monkeypatch.setattr(lxd.console, "print", lambda s, **kw: printed.append(str(s)))

    lxd.create(1)

    assert any("setup-crafts" in s for s in printed)
