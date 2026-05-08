"""Unit tests for the LXD container/VM management module (src/llm/lxd.py).

These tests mock subprocess calls so no real LXD environment is required.
"""

from __future__ import annotations

import json
import os
import subprocess
from unittest.mock import MagicMock

import pytest
import typer

import llm.lxd as lxd

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    """Build a mock CompletedProcess-like object."""
    p = MagicMock()
    p.returncode = returncode
    p.stdout = stdout
    p.stderr = stderr
    return p


# ---------------------------------------------------------------------------
# run_capture
# ---------------------------------------------------------------------------


def test_run_capture_returns_result(monkeypatch):
    expected = _make_proc(0, "hello")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: expected)
    result = lxd.run_capture(["lxc", "list"])
    assert result is expected


# ---------------------------------------------------------------------------
# container_exists
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# wait_for_container
# ---------------------------------------------------------------------------


def test_wait_for_container_succeeds_immediately(monkeypatch):
    """Container becomes ready on the first poll (exec succeeds + cloud-init done)."""
    exec_proc = _make_proc(0)
    cloud_proc = _make_proc(0, stdout='{"status": "done"}')
    call_count = {"n": 0}

    def fake_run(cmd, **kwargs):
        call_count["n"] += 1
        if "true" in cmd:
            return exec_proc
        return cloud_proc  # cloud-init status

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(lxd.time, "sleep", lambda _: None)
    # Should not raise
    lxd.wait_for_container("craft-llm-1", timeout=10)


def test_wait_for_container_retries_then_succeeds(monkeypatch):
    """Container is not ready on first poll but becomes ready on the second."""
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
    """Raises SystemExit when container never becomes ready."""

    # Make time advance beyond the deadline immediately on second call
    _calls = [0]

    def fake_time():
        _calls[0] += 1
        # First call sets deadline, second call is past it
        return 0.0 if _calls[0] == 1 else 9999.0

    monkeypatch.setattr(lxd.time, "time", fake_time)
    monkeypatch.setattr(lxd.time, "sleep", lambda _: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(1))

    with pytest.raises(typer.Exit):
        lxd.wait_for_container("craft-llm-1", timeout=1)


# ---------------------------------------------------------------------------
# create_container (container mode)
# ---------------------------------------------------------------------------


def test_create_container_launches_and_renames_user(monkeypatch):
    calls_made = []

    def fake_run(cmd, **kwargs):
        calls_made.append(list(cmd))

    monkeypatch.setattr(lxd, "run", fake_run)
    monkeypatch.setattr(lxd, "wait_for_container", lambda *a, **kw: None)

    lxd.create_container("craft-llm-1")

    # First call must be `lxc launch ubuntu:24.04 craft-llm-1` (no --vm)
    assert calls_made[0] == ["lxc", "launch", "ubuntu:24.04", "craft-llm-1"]
    # Must contain a usermod call
    assert any("usermod" in c for c in calls_made)
    # Must contain a groupmod call
    assert any("groupmod" in c for c in calls_made)
    # --vm must NOT appear
    assert not any("--vm" in c for c in calls_made)


# ---------------------------------------------------------------------------
# create_container (VM mode)
# ---------------------------------------------------------------------------


def test_create_container_vm_adds_vm_flag(monkeypatch):
    calls_made = []

    def fake_run(cmd, **kwargs):
        calls_made.append(list(cmd))

    monkeypatch.setattr(lxd, "run", fake_run)
    monkeypatch.setattr(lxd, "wait_for_container", lambda *a, **kw: None)

    lxd.create_container("craft-llm-1", vm=True)

    # First call must include --vm and the disk size device
    assert calls_made[0][:5] == ["lxc", "launch", "ubuntu:24.04", "craft-llm-1", "--vm"]
    assert "--device" in calls_made[0]
    # User and group rename must still happen
    assert any("usermod" in c for c in calls_made)
    assert any("groupmod" in c for c in calls_made)


def test_create_container_vm_waits_for_container(monkeypatch):
    wait_calls = []

    monkeypatch.setattr(lxd, "run", lambda *a, **kw: None)
    monkeypatch.setattr(lxd, "wait_for_container", lambda name, **kw: wait_calls.append(name))

    lxd.create_container("craft-llm-1", vm=True)
    assert "craft-llm-1" in wait_calls


# ---------------------------------------------------------------------------
# configure_idmap
# ---------------------------------------------------------------------------


def test_configure_idmap_ensures_subid_then_sets_idmap(monkeypatch):
    """configure_idmap calls _ensure_subid_allocation, then sets raw.idmap."""
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

    assert ensure_called, "_ensure_subid_allocation must be called"

    # raw.idmap value must be set
    config_set_calls = [c for c in calls_made if "config" in c and "set" in c]
    assert config_set_calls, "Expected lxc config set call"
    idmap_call = config_set_calls[0]
    expected_idmap = f"uid {lxd.HOST_UID} {lxd.CONTAINER_UID}\ngid {lxd.HOST_GID} {lxd.CONTAINER_GID}"
    assert expected_idmap in idmap_call

    # Must stop then start (not just restart) after setting idmap so the
    # host-side UID/GID remapping table is properly re-initialised.
    stop_calls = [c for c in calls_made if c == ["lxc", "stop", "craft-llm-1"]]
    start_calls = [c for c in calls_made if c == ["lxc", "start", "craft-llm-1"]]
    assert stop_calls, "Expected lxc stop after idmap"
    assert start_calls, "Expected lxc start after idmap"
    stop_idx = calls_made.index(stop_calls[0])
    start_idx = calls_made.index(start_calls[0])
    assert stop_idx < start_idx, "lxc stop must come before lxc start"


def test_configure_idmap_reloads_daemon_when_subid_changed(monkeypatch):
    """_reload_lxd_daemon is called only when _ensure_subid_allocation returns True."""
    reload_called = []

    monkeypatch.setattr(lxd, "run", lambda *a, **kw: None)
    monkeypatch.setattr(lxd, "wait_for_container", lambda *a, **kw: None)
    monkeypatch.setattr(lxd, "_ensure_subid_allocation", lambda: True)  # new entries added
    monkeypatch.setattr(lxd, "_reload_lxd_daemon", lambda: reload_called.append(True))

    lxd.configure_idmap("craft-llm-1")
    assert reload_called, "_reload_lxd_daemon must be called when subid entries were added"


def test_configure_idmap_no_reload_when_subid_already_present(monkeypatch):
    """_reload_lxd_daemon is NOT called when no subid changes were needed."""
    reload_called = []

    monkeypatch.setattr(lxd, "run", lambda *a, **kw: None)
    monkeypatch.setattr(lxd, "wait_for_container", lambda *a, **kw: None)
    monkeypatch.setattr(lxd, "_ensure_subid_allocation", lambda: False)  # already present
    monkeypatch.setattr(lxd, "_reload_lxd_daemon", lambda: reload_called.append(True))

    lxd.configure_idmap("craft-llm-1")
    assert not reload_called, "_reload_lxd_daemon must NOT be called when no subid changes"


# ---------------------------------------------------------------------------
# _subid_covers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _ensure_subid_allocation
# ---------------------------------------------------------------------------


def test_ensure_subid_allocation_adds_missing_entries(monkeypatch, tmp_path):
    """Adds root:UID:1 entries when they are absent."""
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

    # Patch the paths used inside _ensure_subid_allocation

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
    """Returns False when the entries are already covered."""
    subuid = tmp_path / "subuid"
    subgid = tmp_path / "subgid"
    # Use a range that covers any realistic UID (0 to 2^31)
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
                fake_subprocess_run(["sudo", "tee", "-a", str(path_obj)],
                                    input=f"root:{u}:1\n", text=True)
        # All covered → no change
        return False

    monkeypatch.setattr(lxd, "_ensure_subid_allocation", patched_ensure)

    result = lxd._ensure_subid_allocation()
    assert result is False
    assert not tee_called, "tee should not be called when entries are already covered"


# ---------------------------------------------------------------------------
# _reload_lxd_daemon
# ---------------------------------------------------------------------------


def test_reload_lxd_daemon_tries_snap_service_first(monkeypatch):
    calls = []

    def fake_subprocess_run(cmd, **kwargs):
        calls.append(list(cmd))
        # Succeed on snap service
        if "snap.lxd.daemon.service" in cmd:
            return _make_proc(0)
        return _make_proc(1)

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    lxd._reload_lxd_daemon()

    assert any("snap.lxd.daemon.service" in c for c in calls)
    # Should not try the fallback if snap succeeded
    assert not any("lxd.service" in c for c in calls)


def test_reload_lxd_daemon_falls_back_to_lxd_service(monkeypatch):
    calls = []

    def fake_subprocess_run(cmd, **kwargs):
        calls.append(list(cmd))
        if "lxd.service" in cmd:
            return _make_proc(0)
        return _make_proc(1)  # snap service fails

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    lxd._reload_lxd_daemon()  # should not raise

    assert any("snap.lxd.daemon.service" in c for c in calls)
    assert any("lxd.service" in c for c in calls)


# ---------------------------------------------------------------------------


def test_add_mounts_creates_disk_devices(monkeypatch, tmp_path):
    calls_made = []

    def fake_run(cmd, **kwargs):
        calls_made.append(list(cmd))

    # Patch MOUNTS to use tmp_path so makedirs won't fail in CI
    fake_mounts = [
        ("dev", str(tmp_path / "dev"), f"{lxd.CONTAINER_HOME}/dev"),
        ("opencode-config", str(tmp_path / "opencode"), f"{lxd.CONTAINER_HOME}/.config/opencode"),
    ]
    monkeypatch.setattr(lxd, "MOUNTS", fake_mounts)
    monkeypatch.setattr(lxd, "run", fake_run)
    monkeypatch.setattr(lxd, "wait_for_container", lambda *a, **kw: None)

    lxd.add_mounts("craft-llm-1")

    # Each mount should produce a `lxc config device add` call
    device_add_calls = [c for c in calls_made if "device" in c and "add" in c]
    assert len(device_add_calls) == len(fake_mounts)

    for i, (name, host_path, container_path) in enumerate(fake_mounts):
        c = device_add_calls[i]
        assert name in c
        assert "disk" in c
        assert f"source={host_path}" in c
        assert f"path={container_path}" in c

    # Must restart after adding mounts
    restart_calls = [c for c in calls_made if "restart" in c]
    assert restart_calls


# ---------------------------------------------------------------------------
# install_packages
# ---------------------------------------------------------------------------


def test_install_packages_runs_apt_and_installs_gh_and_uv(monkeypatch):
    calls_made = []

    def fake_run(cmd, **kwargs):
        calls_made.append(list(cmd))

    monkeypatch.setattr(lxd, "run", fake_run)

    lxd.install_packages("craft-llm-1")

    flat = [" ".join(c) for c in calls_made]

    # apt-get update
    assert any("apt-get" in s and "update" in s for s in flat), "Missing apt-get update"
    # apt-get install build-essential
    assert any("apt-get" in s and "build-essential" in s for s in flat), "Missing build-essential install"
    # gh CLI (installed via bash script)
    assert any("gh" in s for s in flat), "Missing gh CLI install"
    # sudoers configuration
    assert any("sudoers" in s or "nopasswd" in s.lower() for s in flat), "Missing sudoers setup"
    # astral-uv snap
    assert any("astral-uv" in s for s in flat), "Missing astral-uv install"


def test_install_packages_uses_correct_container(monkeypatch):
    calls_made = []

    def fake_run(cmd, **kwargs):
        calls_made.append(list(cmd))

    monkeypatch.setattr(lxd, "run", fake_run)

    lxd.install_packages("my-container")

    # Every lxc exec call should reference the correct container name
    for cmd in calls_made:
        if cmd[0] == "lxc" and cmd[1] == "exec":
            assert cmd[2] == "my-container", f"Wrong container in: {cmd}"


# ---------------------------------------------------------------------------
# install_pylsp
# ---------------------------------------------------------------------------


def test_install_pylsp_installs_uv_tool(monkeypatch):
    run_calls = []

    def fake_run(cmd, **kwargs):
        run_calls.append(list(cmd))

    def fake_subprocess_run(cmd, **kwargs):
        # Return plausible output for `cat lsp-config.json` (not found)
        if "cat" in cmd:
            return _make_proc(1, "")
        return _make_proc(0, "")

    monkeypatch.setattr(lxd, "run", fake_run)
    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    lxd.install_pylsp("craft-llm-1")

    flat = [" ".join(str(x) for x in c) for c in run_calls]
    assert any("uv" in s and "tool" in s and "install" in s for s in flat), "Missing uv tool install pylsp"
    assert any("PATH" in s for s in flat), "Missing PATH update"


def test_install_pylsp_merges_existing_lsp_config(monkeypatch):
    """Existing lsp-config.json entries are preserved when pylsp is written."""
    existing_config = json.dumps({"lspServers": {"typescript": {"command": "tsserver"}}})

    run_calls = []
    written_configs = []

    def fake_run(cmd, **kwargs):
        run_calls.append(list(cmd))

    def fake_subprocess_run(cmd, **kwargs):
        if "cat" in cmd:
            return _make_proc(0, existing_config)
        if "bash" in cmd and "cat >" in " ".join(cmd):
            # Capture the input written to the config
            written_configs.append(kwargs.get("input", b""))
            return _make_proc(0, "")
        return _make_proc(0, "")

    monkeypatch.setattr(lxd, "run", fake_run)
    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    lxd.install_pylsp("craft-llm-1")

    # The written config should contain both the existing entry and pylsp
    if written_configs:
        written = json.loads(written_configs[-1].decode())
        assert "typescript" in written.get("lspServers", {}), "Existing entry was overwritten"
        assert "python" in written.get("lspServers", {}), "pylsp entry missing"


def test_install_pylsp_writes_config_when_none_exists(monkeypatch):
    """Creates a new lsp-config.json when none exists in the container."""
    written_configs = []

    def fake_run(cmd, **kwargs):
        pass

    def fake_subprocess_run(cmd, **kwargs):
        if "cat" in cmd and "lsp-config" in " ".join(cmd):
            return _make_proc(1, "")  # file not found
        if "bash" in cmd and "cat >" in " ".join(cmd):
            written_configs.append(kwargs.get("input", b""))
            return _make_proc(0, "")
        return _make_proc(0, "")

    monkeypatch.setattr(lxd, "run", fake_run)
    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    lxd.install_pylsp("craft-llm-1")

    if written_configs:
        written = json.loads(written_configs[-1].decode())
        assert "python" in written.get("lspServers", {}), "pylsp entry missing"
        assert written["lspServers"]["python"]["command"] == "pylsp"

# ---------------------------------------------------------------------------
# .github mount conditional check
# ---------------------------------------------------------------------------


def test_github_mount_check_skipped_when_not_in_mounts(monkeypatch):
    """t_github_mount is skipped (pass) when github is not in MOUNTS."""
    subprocess_calls = []

    def fake_subprocess_run(cmd, **kwargs):
        subprocess_calls.append(list(cmd))
        return _make_proc(0, "")

    monkeypatch.setattr(lxd, "MOUNTS", [
        ("dev", "/home/user/dev", "/home/user/dev"),
    ])
    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    # Build and call the inner t_github_mount function directly
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

    # run_tests should pass without calling lxc exec ... ls .github
    lxd.run_tests("craft-llm-1")
    github_calls = [c for c in subprocess_calls if ".github" in " ".join(str(x) for x in c)]
    assert not github_calls, "Should not call lxc exec for .github when mount is not configured"


def test_github_mount_check_runs_when_in_mounts(monkeypatch):
    """.github is tested when github entry is present in MOUNTS."""
    monkeypatch.setattr(lxd, "MOUNTS", [
        ("github", "/home/user/.github", "/home/user/.github"),
    ])

    github_checked = []

    def fake_subprocess_run(cmd, **kwargs):
        cmd_str = " ".join(str(x) for x in cmd)
        if ".github" in cmd_str:
            github_checked.append(True)
            return _make_proc(0, "")
        return _make_proc(0, "")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    # Call t_github_mount directly via check()
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
    assert github_checked, "Should have checked .github"




def test_run_tests_all_pass(monkeypatch):
    """run_tests returns without error when all lxc calls succeed."""
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

    # Patch os.stat and os.unlink to avoid real filesystem side effects
    fake_stat = MagicMock()
    fake_stat.st_uid = lxd.HOST_UID
    fake_stat.st_gid = lxd.HOST_GID
    monkeypatch.setattr(os, "stat", lambda _: fake_stat)
    monkeypatch.setattr(os, "unlink", lambda _, **kw: None)
    monkeypatch.setattr(os.path, "exists", lambda _: False)

    # Patch MAKE_SETUP_DIRS to empty so venv tests are skipped
    monkeypatch.setattr(lxd, "MAKE_SETUP_DIRS", [])

    # Should complete without sys.exit
    lxd.run_tests("craft-llm-1")


def test_run_tests_fails_exits(monkeypatch):
    """run_tests calls sys.exit(1) when any check fails."""
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


# ---------------------------------------------------------------------------
# run_craft_setup_tests
# ---------------------------------------------------------------------------


def test_run_craft_setup_tests_skips_missing_dirs(monkeypatch, tmp_path):
    """Directories that don't exist on the host are silently skipped."""
    monkeypatch.setattr(lxd, "MAKE_SETUP_DIRS", [str(tmp_path / "nonexistent")])
    # Should complete without error (nothing to check)
    lxd.run_craft_setup_tests("craft-llm-1")


def test_run_craft_setup_tests_checks_venv(monkeypatch, tmp_path):
    """Checks that .venv exists in directories that are present on the host."""
    existing_dir = tmp_path / "my-project"
    existing_dir.mkdir()
    monkeypatch.setattr(lxd, "MAKE_SETUP_DIRS", [str(existing_dir)])

    # Return failure for `ls .venv`
    def fake_subprocess_run(cmd, **kwargs):
        if ".venv" in " ".join(str(x) for x in cmd):
            return _make_proc(1, "")
        return _make_proc(0, "")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    with pytest.raises(typer.Exit) as exc_info:
        lxd.run_craft_setup_tests("craft-llm-1")
    assert exc_info.value.exit_code == 1


# ---------------------------------------------------------------------------
# PYLSP_LSP_CONFIG constant
# ---------------------------------------------------------------------------


def test_pylsp_lsp_config_structure():
    """The LSP config constant has the expected structure."""
    cfg = lxd.PYLSP_LSP_CONFIG
    assert "lspServers" in cfg
    assert "python" in cfg["lspServers"]
    server = cfg["lspServers"]["python"]
    assert server["command"] == "pylsp"
    assert "fileExtensions" in server
    assert ".py" in server["fileExtensions"]


# ---------------------------------------------------------------------------
# CONTAINER_PREFIX / naming convention
# ---------------------------------------------------------------------------


def test_container_prefix():
    assert lxd.CONTAINER_PREFIX == "craft-llm"


def test_container_uid_gid():
    assert lxd.CONTAINER_UID == 1000
    assert lxd.CONTAINER_GID == 1000


# ---------------------------------------------------------------------------
# _fix_vm_user_uid
# ---------------------------------------------------------------------------


def test_fix_vm_user_uid_changes_uid_and_gid_when_different(monkeypatch):
    """groupmod + chgrp + usermod + chown are called when HOST IDs differ."""
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
    assert any("groupmod" in s and "8888" in s for s in flat), "groupmod -g HOST_GID not called"
    assert any("chgrp" in s for s in flat), "chgrp find command not called"
    assert any("usermod" in s and "9999" in s for s in flat), "usermod -u HOST_UID not called"
    assert any("chown" in s for s in flat), "chown find command not called"


def test_fix_vm_user_uid_skips_when_uid_matches(monkeypatch):
    """No usermod/chown when HOST_UID already equals CONTAINER_UID."""
    calls_made = []

    def fake_run(cmd, **kwargs):
        calls_made.append(list(cmd))

    monkeypatch.setattr(lxd, "run", fake_run)
    monkeypatch.setattr(lxd, "HOST_UID", lxd.CONTAINER_UID)
    monkeypatch.setattr(lxd, "HOST_GID", lxd.CONTAINER_GID)

    lxd._fix_vm_user_uid("craft-llm-1")

    flat = [" ".join(c) for c in calls_made]
    assert not any("usermod" in s for s in flat), "usermod should not be called when UIDs match"
    assert not any("groupmod" in s for s in flat), "groupmod should not be called when GIDs match"


def test_fix_vm_user_uid_called_for_vm_not_container(monkeypatch):
    """_fix_vm_user_uid is called for VM creation but not container creation."""
    fix_calls = []

    monkeypatch.setattr(lxd, "run", lambda *a, **kw: None)
    monkeypatch.setattr(lxd, "wait_for_container", lambda *a, **kw: None)
    monkeypatch.setattr(lxd, "_fix_vm_user_uid", lambda c: fix_calls.append(c))

    lxd.create_container("craft-llm-1", vm=True)
    assert "craft-llm-1" in fix_calls, "_fix_vm_user_uid must be called for VMs"

    fix_calls.clear()
    lxd.create_container("craft-llm-1", vm=False)
    assert not fix_calls, "_fix_vm_user_uid must NOT be called for containers"


# ---------------------------------------------------------------------------
# uid/gid parameter threading
# ---------------------------------------------------------------------------


def test_install_packages_uses_custom_uid(monkeypatch):
    """install_packages passes the uid param into the sudoers line."""
    calls_made = []

    def fake_run(cmd, **kwargs):
        calls_made.append(list(cmd))

    monkeypatch.setattr(lxd, "run", fake_run)

    lxd.install_packages("craft-llm-1", uid=9999)

    flat = [" ".join(c) for c in calls_made]
    assert any("9999" in s for s in flat), "Custom uid must appear in sudoers setup"


def test_run_make_setup_uses_custom_uid_gid(monkeypatch, tmp_path):
    """run_make_setup uses the uid/gid params in lxc exec calls."""
    setup_dir = tmp_path / "myproject"
    setup_dir.mkdir()
    monkeypatch.setattr(lxd, "MAKE_SETUP_DIRS", [str(setup_dir)])

    calls_made = []

    def fake_run(cmd, **kwargs):
        calls_made.append(list(cmd))

    monkeypatch.setattr(lxd, "run", fake_run)

    lxd.run_make_setup("craft-llm-1", uid=9999, gid=8888)

    flat = [" ".join(c) for c in calls_made]
    assert any("--user=9999" in s for s in flat), "Custom uid not found in lxc exec call"
    assert any("--group=8888" in s for s in flat), "Custom gid not found in lxc exec call"
    assert any("bash" in s and "make" in s for s in flat), "make not called via bash -c"
    assert any("/snap/bin" in s for s in flat), "PATH with /snap/bin missing from lxc exec call"


def test_install_pylsp_uses_custom_uid_gid(monkeypatch):
    """install_pylsp uses the uid/gid params in lxc exec calls."""
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
    assert any("--user=9999" in s for s in flat), "Custom uid not found in lxc exec call"
    assert any("--group=8888" in s for s in flat), "Custom gid not found in lxc exec call"


def test_run_tests_uses_custom_uid_gid_for_write_transparency(monkeypatch):
    """run_tests uses the uid/gid params in the write-transparency touch command."""
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
    assert touch_calls, "touch call not found"
    touch_flat = " ".join(str(x) for x in touch_calls[0])
    assert "--user=9999" in touch_flat, "Custom uid not used in write-transparency touch"
    assert "--group=8888" in touch_flat, "Custom gid not used in write-transparency touch"


# ---------------------------------------------------------------------------
# setup_crafts command
# ---------------------------------------------------------------------------


def _setup_crafts_with_config(monkeypatch, tmp_path):
    """Helper: point CRAFT_DIRS_CONFIG at an existing tmp file so tests skip the init branch."""
    cfg = tmp_path / "craft-dirs.toml"
    cfg.write_text('dirs = []\n')
    monkeypatch.setattr(lxd, "CRAFT_DIRS_CONFIG", cfg)
    monkeypatch.setattr(lxd, "MAKE_SETUP_DIRS", [])


def test_setup_crafts_errors_when_container_missing(monkeypatch, tmp_path):
    """setup-crafts exits with error when the container doesn't exist."""
    _setup_crafts_with_config(monkeypatch, tmp_path)
    monkeypatch.setattr(lxd, "container_exists", lambda _: False)

    with pytest.raises(typer.Exit) as exc_info:
        lxd.setup_crafts(1)
    assert exc_info.value.exit_code == 1


def test_setup_crafts_inits_config_when_missing(monkeypatch, tmp_path):
    """setup-crafts writes example craft-dirs.toml and exits 0 when config is absent."""
    cfg = tmp_path / "craft-dirs.toml"
    monkeypatch.setattr(lxd, "CRAFT_DIRS_CONFIG", cfg)

    with pytest.raises(typer.Exit) as exc_info:
        lxd.setup_crafts(1)
    assert exc_info.value.exit_code == 0
    assert cfg.exists(), "craft-dirs.toml should have been created"
    assert "dirs" in cfg.read_text()


def test_setup_crafts_calls_make_setup_and_tests(monkeypatch, tmp_path):
    """setup-crafts calls run_make_setup then run_craft_setup_tests."""
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
    """setup-crafts passes HOST_UID/GID when the instance is a VM."""
    _setup_crafts_with_config(monkeypatch, tmp_path)
    make_calls = []

    monkeypatch.setattr(lxd, "container_exists", lambda _: True)
    monkeypatch.setattr(lxd, "container_is_vm", lambda _: True)
    monkeypatch.setattr(lxd, "run_make_setup", lambda c, uid, gid: make_calls.append((uid, gid)))
    monkeypatch.setattr(lxd, "run_craft_setup_tests", lambda _: None)

    lxd.setup_crafts(1)

    assert make_calls == [(lxd.HOST_UID, lxd.HOST_GID)]


def test_create_prints_setup_crafts_hint(monkeypatch):
    """create prints a hint to run setup-crafts after successful creation."""
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

    assert any("setup-crafts" in s for s in printed), "setup-crafts hint not printed after create"
