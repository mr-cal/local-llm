"""Post-provisioning smoke tests run after `llm client setup` and `crafts`.

These run on every setup, so a regression in provisioning surfaces immediately
rather than the next time somebody opens a shell in the VM.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
from collections.abc import Callable

from llm.core.console import console
from llm.provision.exec import (
    CONTAINER_HOME,
    CONTAINER_USER,
    HOST_GID,
    HOST_HOME,
    HOST_UID,
    LSP_CONFIG_PATH,
    _cexec,
    run_capture,
)

_OMP_CONTAINER_CONFIG = f"{CONTAINER_HOME}/.omp/agent/models.yml"


def check(name, fn):
    try:
        fn()
        console.print(f"  [green]PASS[/green]  {name}")
        return True
    except Exception as e:
        console.print(f"  [red]FAIL[/red]  {name}: {e}")
        return False


def _t_venv_exists(container: str, craft_dirs: list[str]) -> None:
    """Assert that .venv exists in every configured craft directory inside the container."""
    missing = []
    for directory in craft_dirs:
        if not os.path.isdir(directory):
            continue
        venv = os.path.join(directory, ".venv")
        r = subprocess.run(
            ["lxc", "exec", container, "--", "ls", venv],
            capture_output=True,
        )
        if r.returncode != 0:
            missing.append(directory)
    assert not missing, f"missing .venv in: {missing}"


def _t_venv_interpreter_valid(craft_dirs: list[str]) -> None:
    """Assert that the venv Python interpreter is executable on the host."""
    failures = []
    for directory in craft_dirs:
        if not os.path.isdir(directory):
            continue
        python = os.path.join(directory, ".venv", "bin", "python3")
        if not os.path.exists(python):
            failures.append(f"not found: {python}")
            continue
        r = subprocess.run([python, "--version"], capture_output=True, text=True)
        if r.returncode != 0:
            failures.append(f"{python}: exit {r.returncode}: {r.stderr.strip()}")
    assert not failures, "\n".join(failures)


def run_tests(
    container: str,
    mounts: list[tuple[str, str, str]],
    craft_dirs: list[str],
    uid: int,
    gid: int,
    sandbox: bool = False,
) -> None:
    """Run verification tests against the configured container."""
    console.print("\n-- Verification tests ----------------------------------------------------------")

    target_host_dir = f"{HOST_HOME}/dev"
    target_container_dir = f"{CONTAINER_HOME}/dev"
    if sandbox and mounts:
        for _, h_path, c_path in mounts:
            if not c_path.endswith(".config/opencode") and not c_path.endswith(".config/helix"):
                target_host_dir = h_path
                target_container_dir = c_path
                break

    def t_running() -> None:
        data = json.loads(run_capture(["lxc", "list", container, "--format=json"]).stdout)
        matches = [c for c in data if c["name"] == container]
        assert matches and matches[0]["status"] == "Running", (
            f"status={matches[0]['status'] if matches else 'not found'}"
        )

    def t_nested_kvm_available() -> None:
        # LXD VMs automatically pass through CPU virtualisation flags when the
        # host has nested KVM enabled, so nested VMs (e.g. snapcraft --vm builds)
        # work without any extra VM config.
        r = subprocess.run(
            ["lxc", "exec", container, "--", "grep", "-c", r"vmx\|svm", "/proc/cpuinfo"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0 and int(r.stdout.strip() or 0) > 0, (
            "No VMX/SVM CPU flags in VM — host nested KVM may not be enabled"
        )

    def t_build_essential() -> None:
        subprocess.run(
            ["lxc", "exec", container, "--", "dpkg", "-l", "build-essential"],
            capture_output=True,
            check=True,
        )

    def t_gh_installed() -> None:
        subprocess.run(
            ["lxc", "exec", container, "--", "gh", "--version"],
            capture_output=True,
            check=True,
        )

    def t_dev_mount_read() -> None:
        r = subprocess.run(
            ["lxc", "exec", container, "--", "stat", "-c", "%a", target_container_dir],
            capture_output=True,
            text=True,
            check=True,
        )
        mode = r.stdout.strip()
        assert mode != "", f"{target_container_dir} is not accessible in the container"

    def t_dev_ownership() -> None:
        r = subprocess.run(
            [
                "lxc",
                "exec",
                container,
                "--",
                "stat",
                "-c",
                "%U",
                target_container_dir,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        owner = r.stdout.strip()
        assert owner == CONTAINER_USER, f"owner is {owner!r}, expected {CONTAINER_USER!r}"

    def t_github_mount() -> None:
        if not any(name == "github" for name, _, _ in mounts):
            return
        subprocess.run(
            ["lxc", "exec", container, "--", "ls", f"{CONTAINER_HOME}/.github"],
            capture_output=True,
            check=True,
        )

    def t_opencode_config_mount() -> None:
        r = subprocess.run(
            [
                "lxc",
                "exec",
                container,
                "--",
                "cat",
                f"{CONTAINER_HOME}/.config/opencode/config.json",
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"opencode config not found in container: {r.stderr.strip()}"
        config = json.loads(r.stdout)
        assert "provider" in config, f"'provider' key missing from opencode config: {config}"

    def t_write_transparency() -> None:
        test_file = os.path.join(target_host_dir, f".{container}_test_file")
        test_path = f"{target_container_dir}/.{container}_test_file"
        subprocess.run(
            _cexec(container, uid, gid, "touch", test_path),
            check=True,
        )
        try:
            st = os.stat(test_file)
            assert st.st_uid == HOST_UID, f"uid={st.st_uid}, expected {HOST_UID}"
            assert st.st_gid == HOST_GID, f"gid={st.st_gid}, expected {HOST_GID}"
        finally:
            if os.path.exists(test_file):
                os.unlink(test_file)

    def t_passwordless_sudo() -> None:
        subprocess.run(
            _cexec(container, uid, gid, "sudo", "-n", "true"),
            capture_output=True,
            check=True,
        )

    def t_uv_installed() -> None:
        subprocess.run(
            ["lxc", "exec", container, "--", "uv", "--version"],
            capture_output=True,
            check=True,
        )

    def t_snap_runs() -> None:
        # Verifies that systemd-run (used by _snap_install) works in the VM.
        # systemd-run creates a transient cgroup scope that snapd accepts,
        # avoiding the lxd-agent.service cgroup rejection error.
        r = subprocess.run(
            ["lxc", "exec", container, "--", "systemd-run", "--wait", "true"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"systemd-run failed in VM (cgroup issue?): {r.stderr.strip()}"
        # Confirm astral-uv snap is installed and accessible.
        r2 = subprocess.run(
            ["lxc", "exec", container, "--", "snap", "list", "astral-uv"],
            capture_output=True,
            text=True,
        )
        assert r2.returncode == 0, f"astral-uv snap not installed or not accessible: {r2.stderr.strip()}"

    def t_fish_installed() -> None:
        subprocess.run(
            ["lxc", "exec", container, "--", "fish", "--version"],
            capture_output=True,
            check=True,
        )

    def t_fish_default_shell() -> None:
        r = subprocess.run(
            ["lxc", "exec", container, "--", "getent", "passwd", CONTAINER_USER],
            capture_output=True,
            text=True,
            check=True,
        )
        shell = r.stdout.strip().split(":")[-1]
        assert shell == "/usr/bin/fish", f"shell is {shell!r}, expected '/usr/bin/fish'"

    def t_path_in_fish_conf() -> None:
        fish_conf_dir = f"{CONTAINER_HOME}/.config/fish/conf.d"
        r = subprocess.run(
            ["lxc", "exec", container, "--", "cat", f"{fish_conf_dir}/path.fish"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"fish path.fish not found: {r.stderr.strip()}"
        assert ".local/bin" in r.stdout, f".local/bin not in fish path.fish: {r.stdout}"
        assert ".bun/bin" in r.stdout, f".bun/bin not in fish path.fish: {r.stdout}"
        assert ".cargo/bin" in r.stdout, f".cargo/bin not in fish path.fish: {r.stdout}"

    def t_path_in_bashrc() -> None:
        r = subprocess.run(
            ["lxc", "exec", container, "--", "grep", ".local/bin", f"{CONTAINER_HOME}/.bashrc"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f".local/bin not in ~/.bashrc: {r.stderr.strip()}"
        r2 = subprocess.run(
            ["lxc", "exec", container, "--", "grep", ".bun/bin", f"{CONTAINER_HOME}/.bashrc"],
            capture_output=True,
            text=True,
        )
        assert r2.returncode == 0, f".bun/bin not in ~/.bashrc: {r2.stderr.strip()}"
        r3 = subprocess.run(
            ["lxc", "exec", container, "--", "grep", ".cargo/bin", f"{CONTAINER_HOME}/.bashrc"],
            capture_output=True,
            text=True,
        )
        assert r3.returncode == 0, f".cargo/bin not in ~/.bashrc: {r3.stderr.strip()}"

    def t_pi_installed() -> None:
        r = subprocess.run(
            ["lxc", "exec", container, "--", f"{CONTAINER_HOME}/.local/bin/pi", "--version"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"pi not found in container: {r.stderr.strip()}"

    def t_omp_config() -> None:
        if sandbox:
            return
        r = subprocess.run(
            ["lxc", "exec", container, "--", "cat", _OMP_CONTAINER_CONFIG],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"models.yml not found in container: {r.stderr.strip()}"
        assert "local-llm" in r.stdout, f"local-llm provider missing in models.yml: {r.stdout}"
        assert "baseUrl" in r.stdout, f"baseUrl missing in models.yml: {r.stdout}"

    def t_pi_mount() -> None:
        if sandbox:
            return
        r = subprocess.run(
            ["lxc", "exec", container, "--", "stat", "-c", "%a", f"{CONTAINER_HOME}/.pi"],
            capture_output=True,
            text=True,
            check=True,
        )
        mode = r.stdout.strip()
        assert mode != "", "~/.pi is not accessible in the container"

    def t_venv_exists() -> None:
        _t_venv_exists(container, craft_dirs)

    def t_container_user() -> None:
        r = subprocess.run(
            ["lxc", "exec", container, "--", "id", "-un", f"{uid}"],
            capture_output=True,
            text=True,
            check=True,
        )
        name = r.stdout.strip()
        assert name == CONTAINER_USER, f"uid {uid} maps to {name!r}, expected {CONTAINER_USER!r}"

    def t_venv_interpreter_valid() -> None:
        _t_venv_interpreter_valid(craft_dirs)

    def t_pylsp_installed() -> None:
        pylsp_bin = f"{CONTAINER_HOME}/.local/bin/pylsp"
        r = subprocess.run(
            _cexec(container, uid, gid, pylsp_bin, "--version"),
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"pylsp not found in container at {pylsp_bin}: {r.stderr.strip()}"

    def t_pylsp_lsp_config() -> None:
        container_config = f"{CONTAINER_HOME}/.copilot/lsp-config.json"
        r = subprocess.run(
            ["lxc", "exec", container, "--", "cat", container_config],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"lsp-config.json not found in container at {container_config}"
        config = json.loads(r.stdout)
        servers = config.get("lspServers", {})
        assert "python" in servers, f"'python' server missing from lspServers: {servers}"
        assert servers["python"]["command"] == "pylsp", (
            f"unexpected command: {servers['python']['command']!r}"
        )

    def t_nested_lxd() -> None:
        r = subprocess.run(
            [
                "lxc",
                "exec",
                container,
                f"--user={uid}",
                f"--env=HOME={CONTAINER_HOME}",
                "--",
                "sg",
                "lxd",
                "-c",
                "lxc list --format=json",
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"lxc list failed inside VM: {r.stderr.strip()}"
        instances = json.loads(r.stdout)
        assert isinstance(instances, list), f"expected JSON list, got: {r.stdout!r}"

    # Auto-discover test functions (t_*) defined in this method's local scope.
    # New tests are discovered automatically; no list to maintain.
    frame = inspect.currentframe()
    local_tests: dict[str, Callable] = {
        k: v for k, v in (frame.f_locals if frame else {}).items() if k.startswith("t_") and callable(v)
    }
    assert frame is not None
    del frame  # avoid reference cycle (PEP 557)
    # Display names are derived from function names: t_foo_bar → "foo bar"
    tests: list[tuple[str, Callable]] = [
        (name.replace("t_", "").replace("_", " "), fn) for name, fn in local_tests.items()
    ]
    results = [check(name, fn) for name, fn in tests]
    passed = sum(results)
    total = len(results)

    console.print()
    if all(results):
        console.print("=" * 60)
        console.print("craft-llm container is ready!")
        console.print(f"  Mounts: ~/.github, ~/dev, ~/.config/opencode  ->  {CONTAINER_HOME}/{{...}}")
        console.print(
            f"  UID/GID mapping: transparent (host {HOST_UID}:{HOST_GID} <-> container {CONTAINER_USER})"
        )
        console.print(f"  Container user: {CONTAINER_USER}")
        console.print("  Packages: build-essential, gh, gh-copilot, astral-uv, pi, bun")
        console.print("  sudo: passwordless for container user")
        console.print("  Next: run 'gh auth login', 'gh copilot setup', and '/allow-all'")
        console.print(" PAT token perms: all repos, actions, issues, merge queues, metadata, pull requests")
        console.print("            user: copilot, gists")
        console.print(f"  pylsp: installed in container (~/.local/bin), config at {LSP_CONFIG_PATH}")
        console.print(f"  All {total} tests passed.")
        console.print("=" * 60)
    else:
        console.print(f"[red]{passed}/{total} tests passed. See failures above.[/red]")
        raise RuntimeError(f"{passed}/{total} tests passed.")


def run_craft_setup_tests(container: str, craft_dirs: list[str]) -> None:
    """Run craft setup verification tests."""
    console.print("\n-- Craft setup verification ----------------------------------------------------")

    tests = [
        ("make setup completed (.venv)", lambda: _t_venv_exists(container, craft_dirs)),
        (
            "venv Python interpreters valid on host",
            lambda: _t_venv_interpreter_valid(craft_dirs),
        ),
    ]

    results = [check(name, fn) for name, fn in tests]
    passed = sum(results)
    total = len(results)

    console.print()
    if all(results):
        console.print(f"[green]All {total} craft setup tests passed.[/green]")
    else:
        console.print(f"[red]{passed}/{total} tests passed. See failures above.[/red]")
