from __future__ import annotations

import json
import os
import subprocess
from unittest.mock import MagicMock

import llm.lxd as lxd


def _make_proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    p = MagicMock()
    p.returncode = returncode
    p.stdout = stdout
    p.stderr = stderr
    return p


def test_create_container_launches_and_renames_user(monkeypatch):
    calls_made = []

    def fake_run(cmd, **kwargs):
        calls_made.append(list(cmd))

    monkeypatch.setattr(lxd, "run", fake_run)
    monkeypatch.setattr(lxd, "wait_for_container", lambda *a, **kw: None)

    lxd.create_container("craft-llm-1")

    assert calls_made[0] == ["lxc", "launch", "ubuntu:24.04", "craft-llm-1"]
    assert any("usermod" in c for c in calls_made)
    assert any("groupmod" in c for c in calls_made)
    assert not any("--vm" in c for c in calls_made)


def test_create_container_vm_adds_vm_flag(monkeypatch):
    calls_made = []

    def fake_run(cmd, **kwargs):
        calls_made.append(list(cmd))

    monkeypatch.setattr(lxd, "run", fake_run)
    monkeypatch.setattr(lxd, "wait_for_container", lambda *a, **kw: None)

    lxd.create_container("craft-llm-1", vm=True)

    assert calls_made[0][:5] == ["lxc", "launch", "ubuntu:24.04", "craft-llm-1", "--vm"]
    assert "--device" in calls_made[0]
    assert any("usermod" in c for c in calls_made)
    assert any("groupmod" in c for c in calls_made)


def test_create_container_vm_waits_for_container(monkeypatch):
    wait_calls = []

    monkeypatch.setattr(lxd, "run", lambda *a, **kw: None)
    monkeypatch.setattr(lxd, "wait_for_container", lambda name, **kw: wait_calls.append(name))

    lxd.create_container("craft-llm-1", vm=True)
    assert "craft-llm-1" in wait_calls


def test_configure_idmap_ensures_subid_then_sets_idmap(monkeypatch):
    calls_made = []
    ensure_called = []
    reload_called = []

    def fake_run(cmd, **kwargs):
        calls_made.append(list(cmd))

    monkeypatch.setattr(lxd, "run", fake_run)
    monkeypatch.setattr(lxd, "wait_for_container", lambda *a, **kw: None)
    monkeypatch.setattr(lxd, "_ensure_subid_allocation", lambda: (ensure_called.append(True), False)[1])
    monkeypatch.setattr(lxd, "_reload_lxd_daemon", lambda: reload_called.append(True))

    lxd.configure_idmap("craft-llm-1")

    assert ensure_called

    config_set_calls = [c for c in calls_made if "config" in c and "set" in c]
    assert config_set_calls
    idmap_call = config_set_calls[0]
    expected_idmap = f"uid {lxd.HOST_UID} {lxd.CONTAINER_UID}\ngid {lxd.HOST_GID} {lxd.CONTAINER_GID}"
    assert expected_idmap in idmap_call

    stop_calls = [c for c in calls_made if c == ["lxc", "stop", "craft-llm-1"]]
    start_calls = [c for c in calls_made if c == ["lxc", "start", "craft-llm-1"]]
    assert stop_calls
    assert start_calls
    assert calls_made.index(stop_calls[0]) < calls_made.index(start_calls[0])


def test_configure_idmap_reloads_daemon_when_subid_changed(monkeypatch):
    reload_called = []

    monkeypatch.setattr(lxd, "run", lambda *a, **kw: None)
    monkeypatch.setattr(lxd, "wait_for_container", lambda *a, **kw: None)
    monkeypatch.setattr(lxd, "_ensure_subid_allocation", lambda: True)
    monkeypatch.setattr(lxd, "_reload_lxd_daemon", lambda: reload_called.append(True))

    lxd.configure_idmap("craft-llm-1")
    assert reload_called


def test_configure_idmap_no_reload_when_subid_already_present(monkeypatch):
    reload_called = []

    monkeypatch.setattr(lxd, "run", lambda *a, **kw: None)
    monkeypatch.setattr(lxd, "wait_for_container", lambda *a, **kw: None)
    monkeypatch.setattr(lxd, "_ensure_subid_allocation", lambda: False)
    monkeypatch.setattr(lxd, "_reload_lxd_daemon", lambda: reload_called.append(True))

    lxd.configure_idmap("craft-llm-1")
    assert not reload_called


def test_subid_covers_exact_match():
    lines = ["root:1001:1"]
    assert lxd._subid_covers(lines, "root", 1001) is True


def test_subid_covers_range():
    lines = ["root:1000:100"]
    assert lxd._subid_covers(lines, "root", 1050) is True
    assert lxd._subid_covers(lines, "root", 1100) is False


def test_subid_covers_wrong_name():
    lines = ["ubuntu:1001:1"]
    assert lxd._subid_covers(lines, "root", 1001) is False


def test_subid_covers_empty():
    assert lxd._subid_covers([], "root", 1001) is False


def test_subid_covers_large_range():
    lines = ["root:100000:65536"]
    assert lxd._subid_covers(lines, "root", 100000) is True
    assert lxd._subid_covers(lines, "root", 165535) is True
    assert lxd._subid_covers(lines, "root", 165536) is False


def test_ensure_subid_allocation_adds_missing_entries(monkeypatch, tmp_path):
    subuid = tmp_path / "subuid"
    subgid = tmp_path / "subgid"
    subuid.write_text("ubuntu:100000:65536\n")
    subgid.write_text("ubuntu:100000:65536\n")

    tee_inputs = {}

    def fake_subprocess_run(cmd, **kwargs):
        if "tee" in cmd:
            path = cmd[-1]
            tee_inputs[path] = kwargs.get("input", "")
            return _make_proc(0, "")
        return _make_proc(0, "")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    def patched_ensure():
        import llm.lxd as _lxd

        orig_uid = _lxd.HOST_UID
        orig_gid = _lxd.HOST_GID
        changed = False
        for path_obj, uid in ((subuid, orig_uid), (subgid, orig_gid)):
            path = str(path_obj)
            lines = path_obj.read_text().splitlines()
            if _lxd._subid_covers(lines, "root", uid):
                continue
            entry = f"root:{uid}:1"
            fake_subprocess_run(["sudo", "tee", "-a", path], input=f"{entry}\n", text=True)
            changed = True
        return changed

    monkeypatch.setattr(lxd, "_ensure_subid_allocation", patched_ensure)

    result = lxd._ensure_subid_allocation()
    assert result is True


def test_ensure_subid_allocation_skips_when_covered(monkeypatch, tmp_path):
    subuid = tmp_path / "subuid"
    subgid = tmp_path / "subgid"
    subuid.write_text("root:0:2147483648\n")
    subgid.write_text("root:0:2147483648\n")

    tee_called = []

    def fake_subprocess_run(cmd, **kwargs):
        if "tee" in cmd:
            tee_called.append(cmd)
        return _make_proc(0, "")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    def patched_ensure():
        import llm.lxd as _lxd

        uid, gid = _lxd.HOST_UID, _lxd.HOST_GID
        for path_obj, u in ((subuid, uid), (subgid, gid)):
            lines = path_obj.read_text().splitlines()
            if not _lxd._subid_covers(lines, "root", u):
                fake_subprocess_run(["sudo", "tee", "-a", str(path_obj)], input=f"root:{u}:1\n", text=True)
        return False

    monkeypatch.setattr(lxd, "_ensure_subid_allocation", patched_ensure)

    result = lxd._ensure_subid_allocation()
    assert result is False
    assert not tee_called


def test_reload_lxd_daemon_tries_snap_service_first(monkeypatch):
    calls = []

    def fake_subprocess_run(cmd, **kwargs):
        calls.append(list(cmd))
        if "snap.lxd.daemon.service" in cmd:
            return _make_proc(0)
        return _make_proc(1)

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    lxd._reload_lxd_daemon()

    assert any("snap.lxd.daemon.service" in c for c in calls)
    assert not any("lxd.service" in c for c in calls)


def test_reload_lxd_daemon_falls_back_to_lxd_service(monkeypatch):
    calls = []

    def fake_subprocess_run(cmd, **kwargs):
        calls.append(list(cmd))
        if "lxd.service" in cmd:
            return _make_proc(0)
        return _make_proc(1)

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    lxd._reload_lxd_daemon()

    assert any("snap.lxd.daemon.service" in c for c in calls)
    assert any("lxd.service" in c for c in calls)


def test_add_mounts_creates_disk_devices(monkeypatch, tmp_path):
    calls_made = []

    def fake_run(cmd, **kwargs):
        calls_made.append(list(cmd))

    fake_mounts = [
        ("dev", str(tmp_path / "dev"), f"{lxd.CONTAINER_HOME}/dev"),
        ("opencode-config", str(tmp_path / "opencode"), f"{lxd.CONTAINER_HOME}/.config/opencode"),
    ]
    monkeypatch.setattr(lxd, "MOUNTS", fake_mounts)
    monkeypatch.setattr(lxd, "run", fake_run)
    monkeypatch.setattr(lxd, "wait_for_container", lambda *a, **kw: None)

    lxd.add_mounts("craft-llm-1")

    device_add_calls = [c for c in calls_made if "device" in c and "add" in c]
    assert len(device_add_calls) == len(fake_mounts)

    for i, (name, host_path, container_path) in enumerate(fake_mounts):
        c = device_add_calls[i]
        assert name in c
        assert "disk" in c
        assert f"source={host_path}" in c
        assert f"path={container_path}" in c

    restart_calls = [c for c in calls_made if "restart" in c]
    assert restart_calls


def test_install_packages_runs_apt_and_installs_gh_and_uv(monkeypatch):
    calls_made = []

    def fake_run(cmd, **kwargs):
        calls_made.append(list(cmd))

    monkeypatch.setattr(lxd, "run", fake_run)

    lxd.install_packages("craft-llm-1")

    flat = [" ".join(c) for c in calls_made]
    assert any("apt-get" in s and "update" in s for s in flat)
    assert any("apt-get" in s and "build-essential" in s for s in flat)
    assert any("gh" in s for s in flat)
    assert any("sudoers" in s or "nopasswd" in s.lower() for s in flat)
    assert any("astral-uv" in s for s in flat)


def test_install_packages_uses_correct_container(monkeypatch):
    calls_made = []

    def fake_run(cmd, **kwargs):
        calls_made.append(list(cmd))

    monkeypatch.setattr(lxd, "run", fake_run)

    lxd.install_packages("my-container")

    for cmd in calls_made:
        if cmd[0] == "lxc" and cmd[1] == "exec":
            assert cmd[2] == "my-container"


def test_install_pylsp_installs_uv_tool(monkeypatch):
    run_calls = []

    def fake_run(cmd, **kwargs):
        run_calls.append(list(cmd))

    def fake_subprocess_run(cmd, **kwargs):
        if "cat" in cmd:
            return _make_proc(1, "")
        return _make_proc(0, "")

    monkeypatch.setattr(lxd, "run", fake_run)
    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    lxd.install_pylsp("craft-llm-1")

    flat = [" ".join(str(x) for x in c) for c in run_calls]
    assert any("uv" in s and "tool" in s and "install" in s for s in flat)
    assert any("PATH" in s for s in flat)


def test_install_pylsp_merges_existing_lsp_config(monkeypatch):
    existing_config = json.dumps({"lspServers": {"typescript": {"command": "tsserver"}}})

    run_calls = []
    written_configs = []

    def fake_run(cmd, **kwargs):
        run_calls.append(list(cmd))

    def fake_subprocess_run(cmd, **kwargs):
        if "cat" in cmd:
            return _make_proc(0, existing_config)
        if "bash" in cmd and "cat >" in " ".join(cmd):
            written_configs.append(kwargs.get("input", b""))
            return _make_proc(0, "")
        return _make_proc(0, "")

    monkeypatch.setattr(lxd, "run", fake_run)
    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    lxd.install_pylsp("craft-llm-1")

    if written_configs:
        written = json.loads(written_configs[-1].decode())
        assert "typescript" in written.get("lspServers", {})
        assert "python" in written.get("lspServers", {})


def test_install_pylsp_writes_config_when_none_exists(monkeypatch):
    written_configs = []

    def fake_run(cmd, **kwargs):
        pass

    def fake_subprocess_run(cmd, **kwargs):
        if "cat" in cmd and "lsp-config" in " ".join(cmd):
            return _make_proc(1, "")
        if "bash" in cmd and "cat >" in " ".join(cmd):
            written_configs.append(kwargs.get("input", b""))
            return _make_proc(0, "")
        return _make_proc(0, "")

    monkeypatch.setattr(lxd, "run", fake_run)
    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    lxd.install_pylsp("craft-llm-1")

    if written_configs:
        written = json.loads(written_configs[-1].decode())
        assert "python" in written.get("lspServers", {})
        assert written["lspServers"]["python"]["command"] == "pylsp"


def test_fix_vm_user_uid_changes_uid_and_gid_when_different(monkeypatch):
    calls_made = []

    def fake_run(cmd, **kwargs):
        calls_made.append(list(cmd))
        return _make_proc(0)

    monkeypatch.setattr(lxd, "run", fake_run)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(lxd, "HOST_UID", 9999)
    monkeypatch.setattr(lxd, "HOST_GID", 8888)

    lxd._fix_vm_user_uid("craft-llm-1")

    flat = [" ".join(c) for c in calls_made]
    assert any("groupmod" in s and "8888" in s for s in flat)
    assert any("chgrp" in s for s in flat)
    assert any("usermod" in s and "9999" in s for s in flat)
    assert any("chown" in s for s in flat)


def test_fix_vm_user_uid_skips_when_uid_matches(monkeypatch):
    calls_made = []

    def fake_run(cmd, **kwargs):
        calls_made.append(list(cmd))

    monkeypatch.setattr(lxd, "run", fake_run)
    monkeypatch.setattr(lxd, "HOST_UID", lxd.CONTAINER_UID)
    monkeypatch.setattr(lxd, "HOST_GID", lxd.CONTAINER_GID)

    lxd._fix_vm_user_uid("craft-llm-1")

    flat = [" ".join(c) for c in calls_made]
    assert not any("usermod" in s for s in flat)
    assert not any("groupmod" in s for s in flat)


def test_fix_vm_user_uid_called_for_vm_not_container(monkeypatch):
    fix_calls = []

    monkeypatch.setattr(lxd, "run", lambda *a, **kw: None)
    monkeypatch.setattr(lxd, "wait_for_container", lambda *a, **kw: None)
    monkeypatch.setattr(lxd, "_fix_vm_user_uid", lambda c: fix_calls.append(c))

    lxd.create_container("craft-llm-1", vm=True)
    assert "craft-llm-1" in fix_calls

    fix_calls.clear()
    lxd.create_container("craft-llm-1", vm=False)
    assert not fix_calls


def test_install_packages_uses_custom_uid(monkeypatch):
    calls_made = []

    def fake_run(cmd, **kwargs):
        calls_made.append(list(cmd))

    monkeypatch.setattr(lxd, "run", fake_run)

    lxd.install_packages("craft-llm-1", uid=9999)

    flat = [" ".join(c) for c in calls_made]
    assert any("9999" in s for s in flat)


def test_run_make_setup_uses_custom_uid_gid(monkeypatch, tmp_path):
    setup_dir = tmp_path / "myproject"
    setup_dir.mkdir()
    monkeypatch.setattr(lxd, "MAKE_SETUP_DIRS", [str(setup_dir)])

    calls_made = []

    def fake_run(cmd, **kwargs):
        calls_made.append(list(cmd))

    monkeypatch.setattr(lxd, "run", fake_run)

    lxd.run_make_setup("craft-llm-1", uid=9999, gid=8888)

    flat = [" ".join(c) for c in calls_made]
    assert any("--user=9999" in s for s in flat)
    assert any("--group=8888" in s for s in flat)
    assert any("bash" in s and "make" in s for s in flat)
    assert any("/snap/bin" in s for s in flat)


def test_install_pylsp_uses_custom_uid_gid(monkeypatch):
    run_calls = []

    def fake_run(cmd, **kwargs):
        run_calls.append(list(cmd))

    def fake_subprocess_run(cmd, **kwargs):
        if "cat" in cmd:
            return _make_proc(1, "")
        return _make_proc(0, "")

    monkeypatch.setattr(lxd, "run", fake_run)
    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    lxd.install_pylsp("craft-llm-1", uid=9999, gid=8888)

    flat = [" ".join(str(x) for x in c) for c in run_calls]
    assert any("--user=9999" in s for s in flat)
    assert any("--group=8888" in s for s in flat)


def test_run_tests_uses_custom_uid_gid_for_write_transparency(monkeypatch):
    container_data = json.dumps([{"name": "craft-llm-1", "status": "Running"}])

    exec_calls = []

    def fake_subprocess_run(cmd, **kwargs):
        cmd_str = " ".join(str(x) for x in cmd)
        exec_calls.append(list(cmd))
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
        return _make_proc(0, "")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(lxd, "MAKE_SETUP_DIRS", [])
    monkeypatch.setattr(os.path, "exists", lambda _: False)

    fake_stat = MagicMock()
    fake_stat.st_uid = lxd.HOST_UID
    fake_stat.st_gid = lxd.HOST_GID
    monkeypatch.setattr(os, "stat", lambda _: fake_stat)
    monkeypatch.setattr(os, "unlink", lambda _, **kw: None)

    lxd.run_tests("craft-llm-1", uid=9999, gid=8888)

    touch_calls = [c for c in exec_calls if "touch" in " ".join(str(x) for x in c)]
    assert touch_calls
    touch_flat = " ".join(str(x) for x in touch_calls[0])
    assert "--user=9999" in touch_flat
    assert "--group=8888" in touch_flat
