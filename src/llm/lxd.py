"""LXD container and VM management — library functions.

All container creation, configuration, and verification logic lives here
as plain functions.  CLI commands are in ``client.py`` and ``server.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from rich.console import Console

from llm.config import (
    _build_pi_config_for_container,
    _get_lxd_bridge_info,
    load_config,
    try_load_lxd,
)

# Version stored in LXD container metadata for future compatibility handling.
LOCAL_LLM_VERSION = 1

CONTAINER_PREFIX = "craft-llm"
HOST_UID = os.getuid()
HOST_GID = os.getgid()
HOST_HOME = os.path.expanduser("~")
HOST_USER = os.path.basename(os.path.normpath(HOST_HOME))

# The container user is renamed to match the host, so paths are identical.
CONTAINER_USER = HOST_USER
CONTAINER_UID = 1000
CONTAINER_GID = 1000
CONTAINER_HOME = HOST_HOME

_DEFAULT_MOUNTS: list[tuple[str, str, str]] = [
    ("agents", f"{HOST_HOME}/.agents", f"{CONTAINER_HOME}/.agents"),
    ("github", f"{HOST_HOME}/.github", f"{CONTAINER_HOME}/.github"),
    ("dev", f"{HOST_HOME}/dev", f"{CONTAINER_HOME}/dev"),
    ("opencode-config", f"{HOST_HOME}/.config/opencode", f"{CONTAINER_HOME}/.config/opencode"),
]

LSP_CONFIG_PATH = f"{CONTAINER_HOME}/.copilot/lsp-config.json"

PYLSP_LSP_CONFIG = {
    "lspServers": {
        "python": {
            "command": "pylsp",
            "args": [],
            "fileExtensions": {
                ".py": "python",
            },
        }
    }
}

console = Console()


def load_lxd_settings() -> tuple[list[tuple[str, str, str]], list[str]]:
    """Load mounts and craft_dirs from config.toml [lxd], falling back to defaults.

    Called at command time (not import time) to pick up config changes.
    """
    lxd = try_load_lxd()
    if lxd is None:
        return _DEFAULT_MOUNTS, []
    mounts = (
        [(m.name, str(Path(m.host).expanduser()), str(Path(m.container).expanduser())) for m in lxd.mounts]
        if lxd.mounts
        else _DEFAULT_MOUNTS
    )
    return mounts, [str(Path(d).expanduser()) for d in lxd.craft_dirs]


# -- Helpers ------------------------------------------------------------------


def _cexec(container: str, uid: int, gid: int, *cmd: str) -> list[str]:
    """Build an `lxc exec` command running *cmd* as uid/gid inside *container*."""
    return [
        "lxc",
        "exec",
        container,
        f"--user={uid}",
        f"--group={gid}",
        f"--env=HOME={CONTAINER_HOME}",
        "--",
        *cmd,
    ]


def run(cmd, desc: str | None = None, **kwargs):
    console.print(f"  $ {' '.join(str(a) for a in cmd)}")
    try:
        subprocess.run(cmd, check=True, **kwargs)
    except subprocess.CalledProcessError as e:
        label = desc or " ".join(str(a) for a in cmd[:3])
        raise subprocess.CalledProcessError(e.returncode, e.cmd, e.output, e.stderr) from Exception(
            f"Command failed ({label}): exit {e.returncode}"
        )


def run_capture(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def wait_for_container(container, timeout=90):
    console.print(f"  Waiting for {container} to be ready...", end="", highlight=False)
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = subprocess.run(
            ["lxc", "exec", container, "--", "true"],
            capture_output=True,
        )
        if r.returncode == 0:
            console.print(" ready.")
            break
        console.print(".", end="", highlight=False)
        time.sleep(2)
    else:
        console.print()
        console.print(f"[red]ERROR:[/red] {container} did not become ready within {timeout}s.")
        raise RuntimeError(f"{container} did not become ready within {timeout}s.")

    console.print("  Waiting for cloud-init...", end="", highlight=False)
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = subprocess.run(
            ["lxc", "exec", container, "--", "cloud-init", "status", "--format=json"],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            try:
                if json.loads(r.stdout).get("status") == "done":
                    console.print(" done.")
                    return
            except (json.JSONDecodeError, AttributeError):
                pass
        console.print(".", end="", highlight=False)
        time.sleep(2)
    console.print()
    console.print(f"[red]ERROR:[/red] cloud-init did not finish within {timeout}s.")
    raise RuntimeError(f"cloud-init did not finish within {timeout}s.")


def container_exists(container):
    r = run_capture(["lxc", "list", container, "--format=json"])
    if r.returncode != 0:
        return False
    return any(c["name"] == container for c in json.loads(r.stdout))


def container_is_vm(container):
    """Return True if the named LXD instance is a virtual machine."""
    r = run_capture(["lxc", "list", container, "--format=json"])
    if r.returncode != 0:
        return False
    instances = json.loads(r.stdout)
    return any(c["name"] == container and c.get("type") == "virtual-machine" for c in instances)


# The custom LXD config key used to mark containers managed by this tool.
# Containers tagged with this key are discovered by `llm lxd refresh` when
# no explicit container number is given.
_MANAGED_TAG = "user.local-llm-managed"


def _tag_as_managed(container: str) -> None:
    """Set the managed tag on *container* so it is discovered by `llm lxd refresh`."""
    run(["lxc", "config", "set", container, f"{_MANAGED_TAG}=true"])


def _list_managed_containers() -> list[str]:
    """Return names of all running LXD instances tagged as managed by this tool."""
    r = run_capture(["lxc", "list", "--format=json"])
    if r.returncode != 0:
        return []
    instances = json.loads(r.stdout)
    return [
        inst["name"]
        for inst in instances
        if inst.get("config", {}).get(_MANAGED_TAG) == "true"
        and inst.get("status") == "Running"
    ]


VM_ROOT_DISK_SIZE = "50GB"
VM_MEMORY = "4GiB"
VM_SWAP_SIZE = "4G"


def _setup_vm_swap(container: str) -> None:
    """Create a persistent swapfile inside the VM and enable it on boot."""
    run(
        [
            "lxc",
            "exec",
            container,
            "--",
            "bash",
            "-c",
            f"fallocate -l {VM_SWAP_SIZE} /swapfile"
            f" && chmod 600 /swapfile"
            f" && mkswap /swapfile"
            f" && swapon /swapfile"
            f" && echo '/swapfile none swap sw 0 0' >> /etc/fstab",
        ],
        desc="create swapfile",
    )


def create_container(container, vm: bool = False):
    kind = "VM" if vm else "container"
    total_steps = 4 if vm else 5
    console.print(f"\n[bold][1/{total_steps}][/bold] Launching {container} (ubuntu:24.04) as {kind}...")
    launch_cmd = ["lxc", "launch", "ubuntu:24.04", container]
    if vm:
        launch_cmd += [
            "--vm",
            "--config",
            f"limits.memory={VM_MEMORY}",
            "--device",
            f"root,size={VM_ROOT_DISK_SIZE}",
        ]
    run(launch_cmd)
    wait_for_container(container)
    if vm:
        _setup_vm_swap(container)
    # Rename the default ubuntu user/group to match the host user, and move the
    # home directory to the same path as on the host.  This ensures venv scripts
    # (whose shebangs reference HOST_HOME) resolve correctly in both environments
    # without any symlinks or re-syncing.
    run(
        [
            "lxc",
            "exec",
            container,
            "--",
            "usermod",
            "--badname",
            "--login",
            CONTAINER_USER,
            "--home",
            CONTAINER_HOME,
            "--move-home",
            "ubuntu",
        ]
    )
    run(
        [
            "lxc",
            "exec",
            container,
            "--",
            "groupmod",
            "--new-name",
            CONTAINER_USER,
            "ubuntu",
        ]
    )

    if vm:
        _fix_vm_user_uid(container)


def _fix_vm_user_uid(container):
    """Change the in-VM user's UID/GID to HOST_UID/HOST_GID.

    VMs share the host UID namespace — virtiofs/9p passes file UIDs as-is, so
    bind-mounted files owned by HOST_UID appear with that same UID inside the VM.
    Changing the VM user to HOST_UID/HOST_GID makes ownership transparent.
    raw.idmap is a container-only feature and cannot be used here.
    """
    if HOST_GID != CONTAINER_GID:
        console.print(f"  Changing in-VM group GID {CONTAINER_GID} → {HOST_GID} to match host...")
        run(["lxc", "exec", container, "--", "groupmod", "-g", str(HOST_GID), CONTAINER_USER])
        r = subprocess.run(
            [
                "lxc",
                "exec",
                container,
                "--",
                "bash",
                "-c",
                f"find / -xdev -group {CONTAINER_GID} -exec chgrp {HOST_GID} {{}} + 2>&1",
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            console.print(f"[yellow]  Warning:[/yellow] chgrp may have missed some files: {r.stderr.strip()}")
    if HOST_UID != CONTAINER_UID:
        console.print(f"  Changing in-VM user UID {CONTAINER_UID} → {HOST_UID} to match host...")
        run(["lxc", "exec", container, "--", "usermod", "-u", str(HOST_UID), CONTAINER_USER])
        r = subprocess.run(
            [
                "lxc",
                "exec",
                container,
                "--",
                "bash",
                "-c",
                f"find / -xdev -user {CONTAINER_UID} -exec chown {HOST_UID} {{}} + 2>&1",
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            console.print(f"[yellow]  Warning:[/yellow] chown may have missed some files: {r.stderr.strip()}")


def _subid_covers(lines: list[str], name: str, uid: int) -> bool:
    """Return True if any existing subuid/subgid entry for *name* covers *uid*."""
    for line in lines:
        parts = line.strip().split(":")
        if len(parts) != 3 or parts[0] != name:
            continue
        try:
            start, count = int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if start <= uid < start + count:
            return True
    return False


def _ensure_subid_allocation() -> bool:
    """Ensure HOST_UID and HOST_GID are in root's subuid/subgid allocations.

    LXD raw.idmap silently ignores mappings whose host UID/GID are not in the
    LXD daemon's allowed range (root's entries in /etc/subuid and /etc/subgid).
    This function adds the missing single-entry allocations and returns True if
    any changes were made (the caller must then reload the LXD daemon).
    """
    changed = False
    for path, uid in (("/etc/subuid", HOST_UID), ("/etc/subgid", HOST_GID)):
        try:
            lines = Path(path).read_text().splitlines()
        except FileNotFoundError:
            lines = []
        if _subid_covers(lines, "root", uid):
            console.print(f"  [dim]{path}[/dim] already covers {uid}")
            continue
        entry = f"root:{uid}:1"
        subprocess.run(
            ["sudo", "tee", "-a", path],
            input=f"{entry}\n",
            text=True,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        console.print(f"  Added [cyan]{entry}[/cyan] to {path}")
        changed = True
    return changed


def _reload_lxd_daemon() -> None:
    """Tell the LXD daemon to reload its subuid/subgid allocations.

    Tries the snap service name first (most Ubuntu installations), then the
    plain systemd unit name as a fallback.
    """
    for svc in ("snap.lxd.daemon.service", "lxd.service"):
        r = subprocess.run(["sudo", "systemctl", "reload", svc], capture_output=True)
        if r.returncode == 0:
            console.print(f"  Reloaded [cyan]{svc}[/cyan]")
            return
    console.print(
        "[yellow]  Warning:[/yellow] could not reload LXD daemon — "
        "raw.idmap may not take effect until the daemon is restarted."
    )


def configure_idmap(container, step: str = "2/5"):
    console.print(
        f"\n[bold][{step}][/bold] Configuring UID/GID mapping "
        f"(host {HOST_UID}:{HOST_GID} -> container {CONTAINER_UID}:{CONTAINER_GID})..."
    )
    # raw.idmap silently does nothing if the host UID/GID is not in the range
    # allocated to root in /etc/subuid and /etc/subgid.  Add single-entry
    # allocations and reload the LXD daemon before touching the container config.
    console.print(
        f"  Ensuring host UID {HOST_UID} / GID {HOST_GID} "
        "are in /etc/subuid + /etc/subgid (raw.idmap prerequisite)..."
    )
    if _ensure_subid_allocation():
        _reload_lxd_daemon()
    idmap = f"uid {HOST_UID} {CONTAINER_UID}\ngid {HOST_GID} {CONTAINER_GID}"
    run(["lxc", "config", "set", container, "raw.idmap", idmap])
    # raw.idmap requires a full stop+start cycle to take effect — lxc restart
    # is not sufficient because it performs an in-place reboot that doesn't
    # re-initialise the host-side UID/GID remapping table.
    run(["lxc", "stop", container])
    run(["lxc", "start", container])
    wait_for_container(container)


def add_mounts(
    container,
    mounts: list[tuple[str, str, str]],
    step: str = "3/5",
    uid: int = CONTAINER_UID,
    gid: int = CONTAINER_GID,
):
    console.print(f"\n[bold][{step}][/bold] Adding bind mounts...")

    # Pre-create mount-point parent directories as the correct user so LXD doesn't
    # create them as root when it sets up the disk devices on the next boot.
    parent_dirs = {str(Path(container_path).parent) for _, _, container_path in mounts}
    for parent in sorted(parent_dirs):
        run(_cexec(container, uid, gid, "mkdir", "-p", parent))

    for name, host_path, container_path in mounts:
        os.makedirs(host_path, exist_ok=True)
        run(
            [
                "lxc",
                "config",
                "device",
                "add",
                container,
                name,
                "disk",
                f"source={host_path}",
                f"path={container_path}",
            ]
        )
    run(["lxc", "restart", container])
    wait_for_container(container)


def install_packages(container, step: str = "4/5", uid: int = CONTAINER_UID):
    console.print(f"\n[bold][{step}][/bold] Installing packages...")
    run(["lxc", "exec", container, "--", "apt-get", "update", "-q"])
    run(
        [
            "lxc",
            "exec",
            container,
            "--",
            "apt-get",
            "install",
            "-y",
            "build-essential",
            "jq",
            "kitty-terminfo",
            "fish",
        ]
    )

    console.print("  Installing gh CLI...")
    gh_setup = (
        "set -euo pipefail\n"
        "curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg"
        " -o /usr/share/keyrings/githubcli-archive-keyring.gpg\n"
        "chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg\n"
        "echo 'deb"
        " [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg]"
        " https://cli.github.com/packages stable main'"
        " | tee /etc/apt/sources.list.d/github-cli.list\n"
        "apt-get update -q\n"
        "apt-get install -y gh"
    )
    run(["lxc", "exec", container, "--", "bash", "-c", gh_setup])

    console.print("  Configuring passwordless sudo...")
    # sudoers.d ignores files containing '.' - use a safe filename.
    # Use User_Alias with #uid to avoid issues with '@' in the username.
    run(
        [
            "lxc",
            "exec",
            container,
            "--",
            "bash",
            "-c",
            f"printf 'User_Alias CONTAINERUSER = #{uid}\\nCONTAINERUSER ALL=(ALL) NOPASSWD:ALL\\n'"
            f" > /etc/sudoers.d/nopasswd-user"
            f" && chmod 440 /etc/sudoers.d/nopasswd-user",
        ]
    )

    console.print("  Installing astral-uv...")
    run(["lxc", "exec", container, "--", "snap", "install", "astral-uv", "--classic"])

    console.print("  Installing helix...")
    run(["lxc", "exec", container, "--", "snap", "install", "helix", "--classic"])

    console.print("  Installing nodejs (for pi)...")
    run(
        [
            "lxc", "exec", container, "--",
            "bash", "-c", "set -euo pipefail && "
            "apt-get update -q && "
            "apt-get install -y nodejs npm curl && "
            "curl -fsSL https://deb.nodesource.com/setup_22.x "
            "| bash - && "
            "apt-get install -y nodejs"
        ]
    )

    console.print("  Installing gh-copilot extension...")
    run(
        ["lxc", "exec", container, "--", "gh", "extension", "install", "github/gh-copilot"],
    )

    console.print("  Installing pi (@earendil-works/pi-coding-agent)...")
    run(
        ["lxc", "exec", container, "--", "npm", "install", "-g", "@earendil-works/pi-coding-agent"],
    )

    console.print("  Setting fish as the default shell...")
    run(
        ["lxc", "exec", container, "--", "chsh", "-s", "/usr/bin/fish", CONTAINER_USER],
    )

    console.print("  Cleaning up unused packages...")
    run(["lxc", "exec", container, "--", "apt-get", "autoremove", "-y"])
    run(["lxc", "exec", container, "--", "apt-get", "clean"])


def run_make_setup(container, craft_dirs: list[str], uid: int = CONTAINER_UID, gid: int = CONTAINER_GID):
    """Run ``make setup`` in each craft project directory inside the container.

    The directories live under ~/dev which is bind-mounted, so the resulting
    venvs are visible on the host at the same paths.  ``make setup`` may be
    interactive (it installs apt packages via sudo), so stdin is inherited from
    the calling terminal.
    """
    console.print("\nRunning make setup in craft directories (in container)...")
    for directory in craft_dirs:
        if not os.path.isdir(directory):
            console.print(f"  [yellow]WARNING:[/yellow] directory not found on host, skipping: {directory}")
            continue
        console.print(f"  Running make setup in {directory}...")
        run(
            [
                "lxc",
                "exec",
                container,
                f"--user={uid}",
                f"--group={gid}",
                f"--env=HOME={CONTAINER_HOME}",
                f"--env=USER={CONTAINER_USER}",
                f"--env=LOGNAME={CONTAINER_USER}",
                # lxc exec with --env replaces the entire environment, so
                # PATH must be set explicitly to include snap and user binaries.
                f"--env=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin:{CONTAINER_HOME}/.local/bin",
                "--env=CI=1",
                "--",
                "bash",
                "-c",
                f"make -C {directory} setup",
            ],
        )


def install_pylsp(container, step: str = "5/5", uid: int = CONTAINER_UID, gid: int = CONTAINER_GID):
    """Install python-lsp-server via uv tool inside the container, ensure it is
    on PATH, and write the gh copilot LSP config into the container."""
    console.print(f"\n[bold][{step}][/bold] Installing pylsp (python-lsp-server) in container...")

    run(_cexec(container, uid, gid, "uv", "tool", "install", "python-lsp-server"))
    # uv tool update-shell can't detect the shell via lxc exec, so append directly.
    run(
        _cexec(
            container,
            uid,
            gid,
            "bash",
            "-c",
            r'grep -qxF "export PATH=$HOME/.local/bin:$PATH" ~/.bashrc'
            r' || echo "export PATH=$HOME/.local/bin:$PATH" >> ~/.bashrc',
        )
    )
    # Shorten the prompt to just the current directory — the default includes
    # username and hostname which are too long in a container context.
    run(
        _cexec(
            container,
            uid,
            gid,
            "bash",
            "-c",
            r'grep -qxF "export PS1=\"\w\$ \"" ~/.bashrc'
            r' || echo "export PS1=\"\w\$ \"" >> ~/.bashrc',
        )
    )

    # Write a fish conf.d snippet so `llm <n>` can pass CRAFT_CWD via the
    # environment and have the login shell cd there automatically.  This avoids
    # the "no job control" errors that arise from `su - user -c "exec $SHELL"`.
    fish_conf_dir = f"{CONTAINER_HOME}/.config/fish/conf.d"
    craft_cwd_fish = (
        "# cd to CRAFT_CWD when set (passed by the host `llm` fish function)\n"
        "if set -q CRAFT_CWD; and test -d $CRAFT_CWD\n"
        "    cd $CRAFT_CWD\n"
        "end\n"
    )
    run(_cexec(container, uid, gid, "mkdir", "-p", fish_conf_dir))
    subprocess.run(
        _cexec(container, uid, gid, "bash", "-c", f"cat > {fish_conf_dir}/craft-cwd.fish"),
        input=craft_cwd_fish.encode(),
        check=True,
    )

    # Minimal fish prompt — just the current directory, no user@host, no git info.
    prompt_fish = (
        "# Minimal prompt: current directory + arrow (no user@host, no git hash)\n"
        "function fish_prompt\n"
        '    echo -n (set_color blue)(prompt_pwd)(set_color normal) " ❯ "\n'
        "end\n"
    )
    subprocess.run(
        _cexec(container, uid, gid, "bash", "-c", f"cat > {fish_conf_dir}/prompt.fish"),
        input=prompt_fish.encode(),
        check=True,
    )

    # pi "full context" helper: bump contextWindow and maxTokens in models.json
    # so pi uses the server's full context window for long agentic tasks.
    pif_fish = (
        "function pif\n"
        "    set tmp (mktemp)\n"
        "    jq '.providers[\"local-llm\"].models[0].contextWindow = 131072"
        " | .providers[\"local-llm\"].models[0].maxTokens = 16384'"
        " ~/.pi/agent/models.json > $tmp\n"
        "    and mv $tmp ~/.pi/agent/models.json\n"
        "end\n"
    )
    subprocess.run(
        _cexec(container, uid, gid, "bash", "-c", f"cat > {fish_conf_dir}/pif.fish"),
        input=pif_fish.encode(),
        check=True,
    )

    console.print(f"  Writing LSP config to {CONTAINER_HOME}/.copilot/lsp-config.json in container...")
    run(_cexec(container, uid, gid, "mkdir", "-p", f"{CONTAINER_HOME}/.copilot"))

    # Read any existing config from the container, then merge and write back.
    r = subprocess.run(
        _cexec(container, uid, gid, "cat", f"{CONTAINER_HOME}/.copilot/lsp-config.json"),
        capture_output=True,
        text=True,
    )
    existing: dict = {}
    if r.returncode == 0 and r.stdout.strip():
        try:
            existing = json.loads(r.stdout)
            if not isinstance(existing, dict):
                console.print(
                    "  [yellow]WARNING:[/yellow] lsp-config.json is not a JSON object; overwriting."
                )
                existing = {}
        except json.JSONDecodeError as e:
            console.print(f"  [yellow]WARNING:[/yellow] lsp-config.json is invalid JSON ({e}); overwriting.")
    existing.setdefault("lspServers", {}).update(PYLSP_LSP_CONFIG["lspServers"])
    config_json = json.dumps(existing, indent=2) + "\n"
    # Validate before writing — guards against bugs in PYLSP_LSP_CONFIG.
    json.loads(config_json)

    subprocess.run(
        _cexec(container, uid, gid, "bash", "-c", f"cat > {CONTAINER_HOME}/.copilot/lsp-config.json"),
        input=config_json.encode(),
        check=True,
    )


def setup_nested_lxd(container, step: str = "5/5", uid: int = HOST_UID):
    """Install and initialise LXD inside the VM so nested containers can run.

    Steps taken:
    1. Install the ``lxd`` snap (if not already present).
    2. Initialise LXD with ``lxd init --auto``.
    3. Add the VM user to the ``lxd`` group.
    4. Launch a minimal Ubuntu container to verify nesting works, then delete it.
    """
    console.print(f"\n[bold][{step}][/bold] Setting up nested LXD inside VM...")

    console.print("  Installing lxd snap...")
    run(["lxc", "exec", container, "--", "snap", "install", "lxd"])

    console.print("  Initialising LXD (lxd init --auto)...")
    run(["lxc", "exec", container, "--", "lxd", "init", "--auto"])

    console.print(f"  Adding uid {uid} to lxd group...")
    r = subprocess.run(
        ["lxc", "exec", container, "--", "id", "-un", str(uid)],
        capture_output=True,
        text=True,
        check=True,
    )
    vm_username = r.stdout.strip()
    run(["lxc", "exec", container, "--", "usermod", "-aG", "lxd", vm_username])

    console.print("  Launching nested test container (ubuntu:24.04) to verify nesting...")
    # usermod -aG doesn't activate the new group in the current session, so use
    # `sg lxd` to re-exec under the lxd group without requiring a new login.
    run(
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
            "lxc launch ubuntu:24.04 nested-test",
        ]
    )
    console.print("  Nested container launched. Deleting it...")
    run(
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
            "lxc delete --force nested-test",
        ]
    )
    console.print("  Nested LXD ready.")


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
    """Assert that the venv Python interpreter is executable on the host in all setup dirs."""
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


# -- Pi harness setup --------------------------------------------------------


# Path inside the container where pi reads its model configuration.
_PI_CONTAINER_CONFIG = f"{CONTAINER_HOME}/.pi/agent/models.json"
_NODE_CA_CERTS_DIR = f"{CONTAINER_HOME}/.config/local-llm"
_NODE_CA_CERTS_FILE = f"{_NODE_CA_CERTS_DIR}/cert.pem"


def _run_capture(container: str, *cmd: str) -> subprocess.CompletedProcess[str]:
    """Run a command inside the container and return the result."""
    return subprocess.run(
        ["lxc", "exec", container, "--", *cmd],
        capture_output=True,
        text=True,
    )


def _get_lxd_bridge_ip() -> str:
    """Return the host IP on the lxdbr0 bridge (e.g. '10.113.167.1').

    Returns an empty string if lxdbr0 is not found or ip(8) fails.
    Delegates to :func:`llm.config._get_lxd_bridge_info` to avoid duplication.
    """
    ip, _ = _get_lxd_bridge_info()
    return ip


def setup_gh_auth_in_container(
    container: str,
    gh_token: str,
    *,
    effective_uid: int = CONTAINER_UID,
    effective_gid: int = CONTAINER_GID,
) -> None:
    """Authenticate the GitHub CLI (gh) inside the container using a PAT.

    Runs ``gh auth login --with-token`` inside the container so that tools
    like ``gh copilot`` can access GitHub APIs without interactive login.

    Args:
        container: Name of the LXD container.
        gh_token: GitHub personal access token (PAT).
        effective_uid: UID to run commands as inside the container.
        effective_gid: GID to run commands as inside the container.
    """
    if not gh_token:
        console.print("  [yellow]⚠[/yellow] No GitHub token configured — skipping gh auth.")
        return

    console.print(f"  [bold]Authenticating gh CLI in {container}...[/bold]")

    # Use gh auth login --with-token to authenticate non-interactively.
    # This writes the token to ~/.config/gh/hosts.yml which gh uses for auth.
    subprocess.run(
        _cexec(container, effective_uid, effective_gid, "gh", "auth", "login", "--with-token"),
        input=gh_token.encode(),
        check=True,
    )
    console.print(f"  [green]✓[/green] gh authenticated — can access GitHub APIs")


def setup_pi_in_container(
    container: str,
    bridge_ip: str,
    cert_pem: str | None = None,
    uid: int = CONTAINER_UID,
    gid: int = CONTAINER_GID,
) -> None:
    """Set up the Pi harness inside the container so it can reach the LLM server.

    Four things are configured inside the container:

    1. **models.json** — generated with URL ``https://local-llm:<port>/v1``
       so pi connects to the nginx proxy using the ``local-llm`` hostname
       (which is already in the cert's SubjectAltName).

    2. **/etc/hosts entry** — maps ``local-llm`` → ``proxy.lan_ip`` (the
       server's LAN IP) so traffic routes correctly whether the container is
       on the same machine as the server or on a remote client machine.
       LXD's MASQUERADE rules ensure nginx's subnet allowlist passes in both
       cases.

    3. **TLS certificate** — copied from the host and stored at
       ``~/.config/local-llm/cert.pem`` for Node.js to trust.

    4. **NODE_EXTRA_CA_CERTS** — set in ``~/.bashrc`` and
       ``~/.config/fish/conf.d/`` so pi always trusts the self-signed cert.

    Args:
        container: Name of the LXD container/VM.
        bridge_ip: Unused; kept for backward compatibility.
        cert_pem: PEM cert string for the nginx TLS proxy. When provided it
                  is written to ``_NODE_CA_CERTS_FILE``.  If *None* the cert
                  is read from ``cfg.proxy.cert_path`` on the host.
        uid: UID to run container commands as.
        gid: GID to run container commands as.
    """
    console.print(f"\n[bold]Setting up Pi harness in {container}...[/bold]")

    cfg = load_config()

    # ── Step 1: Add /etc/hosts entry so 'local-llm' resolves inside container ─
    # Use the server's LAN IP (proxy.lan_ip) rather than the lxdbr0 bridge IP.
    # nginx listens on lan_ip:port, and LXD's MASQUERADE NAT ensures nginx's
    # subnet allowlist passes regardless of whether the server is local or remote.
    server_ip = cfg.proxy.lan_ip
    console.print(f"  Adding /etc/hosts entry: {server_ip} local-llm...")
    hosts_cmd = (
        f"grep -qxF '{server_ip} local-llm' /etc/hosts || "
        f"echo '{server_ip} local-llm' >> /etc/hosts"
    )
    subprocess.run(
        ["lxc", "exec", container, "--", "bash", "-c", hosts_cmd],
        check=True,
    )

    # ── Step 2: Generate models.json with the 'local-llm' proxy URL ──────────
    console.print("  Generating models.json with proxy URL...")
    pi_cfg = _build_pi_config_for_container(cfg, "local-llm")

    # Ensure the pi config directory exists (no longer created by a bind-mount).
    pi_config_dir = str(Path(_PI_CONTAINER_CONFIG).parent)
    subprocess.run(
        _cexec(container, uid, gid, "bash", "-c", f"mkdir -p {pi_config_dir}"),
        check=True,
    )

    # Read existing config from the container, merge in our provider, write back.
    r = _run_capture(container, "cat", _PI_CONTAINER_CONFIG)
    existing: dict = {}
    if r.returncode == 0 and r.stdout.strip():
        try:
            existing = json.loads(r.stdout)
            if not isinstance(existing, dict):
                console.print("    [yellow]WARNING:[/yellow] existing config is not a JSON object")
                existing = {}
        except json.JSONDecodeError:
            existing = {}
    existing.setdefault("providers", {}).update(pi_cfg.get("providers", {}))
    merged_json = json.dumps(existing, indent=2) + "\n"

    subprocess.run(
        _cexec(container, uid, gid, "bash", "-c", f"cat > {_PI_CONTAINER_CONFIG}"),
        input=merged_json.encode(),
        check=True,
    )
    console.print(f"    Written to {Path(_PI_CONTAINER_CONFIG).relative_to(CONTAINER_HOME)}")

    # ── Step 3: Install TLS certificate ──────────────────────────────────────
    console.print("  Installing TLS certificate...")

    # Resolve cert: caller-supplied > config.toml cert_path
    if cert_pem is None:
        cfg_cert_path = Path(cfg.proxy.cert_path)
        if cfg_cert_path.exists():
            cert_pem = cfg_cert_path.read_text()

    if cert_pem:
        subprocess.run(
            _cexec(container, uid, gid, "bash", "-c", f"mkdir -p {_NODE_CA_CERTS_DIR}"),
            check=True,
        )
        subprocess.run(
            _cexec(container, uid, gid, "bash", "-c", f"cat > {_NODE_CA_CERTS_FILE}"),
            input=cert_pem.encode(),
            check=True,
        )
        console.print(f"    Written to {Path(_NODE_CA_CERTS_FILE).relative_to(CONTAINER_HOME)}")
    else:
        console.print(
            f"    [yellow]Warning:[/yellow] cert not found at {cfg.proxy.cert_path}. "
            "Run 'uv run llm config gencert' on the server first."
        )

    # ── Step 4: Set NODE_EXTRA_CA_CERTS in shell profiles ────────────────────
    console.print("  Configuring NODE_EXTRA_CA_CERTS in shell profiles...")
    bash_export = f'export NODE_EXTRA_CA_CERTS="{_NODE_CA_CERTS_FILE}"'
    fish_export = f'set -x NODE_EXTRA_CA_CERTS "{_NODE_CA_CERTS_FILE}"'

    # Add to ~/.bashrc if not already present
    bashrc_cmd = (
        f"grep -qxF '{bash_export}' ~/.bashrc || echo '{bash_export}' >> ~/.bashrc"
    )
    subprocess.run(_cexec(container, uid, gid, "bash", "-c", bashrc_cmd), check=True)

    # Add to fish conf.d (fish uses 'set -x', not 'export VAR=val')
    fish_conf_dir = f"{CONTAINER_HOME}/.config/fish/conf.d"
    fish_conf = f"{fish_conf_dir}/node-ca-certs.fish"
    fish_cmd = (
        f"mkdir -p {fish_conf_dir} && "
        f"grep -qxF '{fish_export}' {fish_conf} 2>/dev/null || "
        f"echo '{fish_export}' > {fish_conf}"
    )
    subprocess.run(_cexec(container, uid, gid, "bash", "-c", fish_cmd), check=True)

    console.print("    Added to ~/.bashrc and ~/.config/fish/conf.d/node-ca-certs.fish")
    console.print("  [green]✓[/green] Pi harness configured — it can now reach the LLM server")


# -- Verification tests -------------------------------------------------------


def check(name, fn):
    try:
        fn()
        console.print(f"  [green]PASS[/green]  {name}")
        return True
    except Exception as e:
        console.print(f"  [red]FAIL[/red]  {name}: {e}")
        return False


def run_tests(
    container,
    mounts: list[tuple[str, str, str]] | None = None,
    uid: int = CONTAINER_UID,
    gid: int = CONTAINER_GID,
):
    console.print("\n-- Verification tests ----------------------------------------------------------")

    def t_running():
        data = json.loads(run_capture(["lxc", "list", container, "--format=json"]).stdout)
        matches = [c for c in data if c["name"] == container]
        assert matches and matches[0]["status"] == "Running", (
            f"status={matches[0]['status'] if matches else 'not found'}"
        )

    def t_build_essential():
        subprocess.run(
            ["lxc", "exec", container, "--", "dpkg", "-l", "build-essential"],
            capture_output=True,
            check=True,
        )

    def t_gh_installed():
        subprocess.run(
            ["lxc", "exec", container, "--", "gh", "--version"],
            capture_output=True,
            check=True,
        )

    def t_gh_copilot_extension():
        subprocess.run(
            ["lxc", "exec", container, "--", "gh", "extension", "list"],
            capture_output=True,
            check=True,
        )

    def t_dev_mount_read():
        # Verify ~/dev mount point is readable in the container.
        # We check the mount point itself rather than a specific subdirectory
        # because users may have ~/dev/cal, ~/dev/craft, both, or neither.
        r = subprocess.run(
            ["lxc", "exec", container, "--", "stat", "-c", "%a", f"{CONTAINER_HOME}/dev"],
            capture_output=True,
            text=True,
            check=True,
        )
        mode = r.stdout.strip()
        assert mode != "", "~/dev is not accessible in the container"

    def t_dev_ownership():
        # Verify ~/dev is owned by the container user.
        r = subprocess.run(
            [
                "lxc",
                "exec",
                container,
                "--",
                "stat",
                "-c",
                "%U",
                f"{CONTAINER_HOME}/dev",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        owner = r.stdout.strip()
        assert owner == CONTAINER_USER, f"owner is {owner!r}, expected {CONTAINER_USER!r}"

    def t_github_mount():
        effective_mounts = mounts or _DEFAULT_MOUNTS
        if not any(name == "github" for name, _, _ in effective_mounts):
            return
        subprocess.run(
            ["lxc", "exec", container, "--", "ls", f"{CONTAINER_HOME}/.github"],
            capture_output=True,
            check=True,
        )

    def t_opencode_config_mount():
        r = subprocess.run(
            ["lxc", "exec", container, "--", "cat", f"{CONTAINER_HOME}/.config/opencode/config.json"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"opencode config not found in container: {r.stderr.strip()}"
        config = json.loads(r.stdout)
        assert "provider" in config, f"'provider' key missing from opencode config: {config}"

    def t_write_transparency():
        test_file = f"{HOST_HOME}/dev/.{container}_test_file"
        subprocess.run(
            _cexec(container, uid, gid, "touch", f"{CONTAINER_HOME}/dev/.{container}_test_file"),
            check=True,
        )
        try:
            st = os.stat(test_file)
            assert st.st_uid == HOST_UID, f"uid={st.st_uid}, expected {HOST_UID}"
            assert st.st_gid == HOST_GID, f"gid={st.st_gid}, expected {HOST_GID}"
        finally:
            if os.path.exists(test_file):
                os.unlink(test_file)

    def t_passwordless_sudo():
        subprocess.run(
            _cexec(container, uid, gid, "sudo", "-n", "true"),
            capture_output=True,
            check=True,
        )

    def t_uv_installed():
        subprocess.run(
            ["lxc", "exec", container, "--", "uv", "--version"],
            capture_output=True,
            check=True,
        )

    def t_fish_installed():
        subprocess.run(
            ["lxc", "exec", container, "--", "fish", "--version"],
            capture_output=True,
            check=True,
        )

    def t_fish_default_shell():
        r = subprocess.run(
            ["lxc", "exec", container, "--", "getent", "passwd", CONTAINER_USER],
            capture_output=True,
            text=True,
            check=True,
        )
        shell = r.stdout.strip().split(":")[-1]
        assert shell == "/usr/bin/fish", f"shell is {shell!r}, expected '/usr/bin/fish'"

    def t_pi_installed():
        r = subprocess.run(
            ["lxc", "exec", container, "--", "pi", "--version"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"pi not found in container: {r.stderr.strip()}"

    def t_pi_mount():
        # Verify ~/.pi mount point exists in the container.
        r = subprocess.run(
            ["lxc", "exec", container, "--", "stat", "-c", "%a", f"{CONTAINER_HOME}/.pi"],
            capture_output=True,
            text=True,
            check=True,
        )
        mode = r.stdout.strip()
        assert mode != "", "~/.pi is not accessible in the container"

    def t_venv_exists():
        _t_venv_exists(container)

    def t_container_user():
        r = subprocess.run(
            ["lxc", "exec", container, "--", "id", "-un", f"{uid}"],
            capture_output=True,
            text=True,
            check=True,
        )
        name = r.stdout.strip()
        assert name == CONTAINER_USER, f"uid {uid} maps to {name!r}, expected {CONTAINER_USER!r}"

    def t_venv_interpreter_valid():
        _t_venv_interpreter_valid()

    def t_pylsp_installed():
        pylsp_bin = f"{CONTAINER_HOME}/.local/bin/pylsp"
        r = subprocess.run(
            _cexec(container, uid, gid, pylsp_bin, "--version"),
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"pylsp not found in container at {pylsp_bin}: {r.stderr.strip()}"

    def t_pylsp_lsp_config():
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

    def t_nested_lxd():
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
        # Confirm the output is valid JSON (list of instances)
        instances = json.loads(r.stdout)
        assert isinstance(instances, list), f"expected JSON list, got: {r.stdout!r}"

    is_vm = container_is_vm(container)
    tests = [
        ("Container running", t_running),
        ("build-essential installed", t_build_essential),
        ("gh installed", t_gh_installed),
        ("gh-copilot extension installed", t_gh_copilot_extension),
        ("passwordless sudo works", t_passwordless_sudo),
        ("uv installed", t_uv_installed),
        ("fish installed", t_fish_installed),
        ("fish is the default shell", t_fish_default_shell),
        ("pi installed", t_pi_installed),
        ("dev mount readable", t_dev_mount_read),
        ("dev mount ownership transparent", t_dev_ownership),
        (".github mount works", t_github_mount),
        ("Write transparency", t_write_transparency),
        (f"container user is {CONTAINER_USER!r}", t_container_user),
        ("opencode config mounted", t_opencode_config_mount),
        ("pi mount exists", t_pi_mount),
        ("pylsp installed", t_pylsp_installed),
        ("pylsp registered in lsp-config.json", t_pylsp_lsp_config),
        *([("nested lxd: lxc list works inside VM", t_nested_lxd)] if is_vm else []),
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
        console.print("  Packages: build-essential, gh, gh-copilot, astral-uv, pi")
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


def run_craft_setup_tests(container, craft_dirs: list[str]):
    console.print("\n-- Craft setup verification ----------------------------------------------------")

    tests = [
        ("make setup completed (.venv)", lambda: _t_venv_exists(container, craft_dirs)),
        ("venv Python interpreters valid on host", lambda: _t_venv_interpreter_valid(craft_dirs)),
    ]

    results = [check(name, fn) for name, fn in tests]
    passed = sum(results)
    total = len(results)

    console.print()
    if all(results):
        console.print(f"[green]All {total} craft setup tests passed.[/green]")
    else:
        console.print(f"[red]{passed}/{total} tests passed. See failures above.[/red]")


def create_and_setup(
    container_name: str,
    *,
    mounts: list[tuple[str, str, str]],
    recreate: bool = False,
    lxd_vm: bool = False,
    cert_pem: str | None = None,
) -> None:
    """Create and configure an LXD container (or VM) for local LLM development.

    This is the library equivalent of the old ``llm lxd create`` command.
    Raises ``RuntimeError`` on fatal errors instead of ``typer.Exit``.
    """
    kind = "VM" if lxd_vm else "container"

    if container_exists(container_name):
        if not recreate:
            raise RuntimeError(
                f"{kind} '{container_name}' already exists. "
                "Pass recreate=True to delete and recreate it."
            )
        console.print(f"Deleting existing {kind}: {container_name}")
        run(["lxc", "delete", "--force", container_name])

    console.print(f"Creating {kind}: {container_name}")

    # Prepend a helix config bind-mount if the directory exists on the host.
    helix_host = os.path.join(HOST_HOME, ".config", "helix")
    helix_container = f"{CONTAINER_HOME}/.config/helix"
    all_mounts = list(mounts)
    if os.path.isdir(helix_host):
        all_mounts = [("helix-config", helix_host, helix_container), *all_mounts]
    else:
        console.print(f"  [dim]~/.config/helix not found on host — skipping helix config mount[/dim]")

    if lxd_vm:
        create_container(container_name, vm=True)
        add_mounts(container_name, all_mounts, step="2/5", uid=HOST_UID, gid=HOST_GID)
        install_packages(container_name, step="3/5", uid=HOST_UID)
        install_pylsp(container_name, step="4/5", uid=HOST_UID, gid=HOST_GID)
        setup_nested_lxd(container_name, step="5/5", uid=HOST_UID)
    else:
        create_container(container_name, vm=False)
        configure_idmap(container_name, step="2/5")
        add_mounts(container_name, all_mounts, step="3/5")
        install_packages(container_name, step="4/5")
        install_pylsp(container_name, step="5/5")

    effective_uid = HOST_UID if lxd_vm else CONTAINER_UID
    effective_gid = HOST_GID if lxd_vm else CONTAINER_GID

    bridge_ip = _get_lxd_bridge_ip()
    if not bridge_ip:
        console.print(
            "[yellow]Warning:[/yellow] Could not detect lxdbr0 bridge IP. "
            "Pi in the container may not be able to reach the LLM server."
        )

    setup_pi_in_container(
        container_name,
        bridge_ip=bridge_ip,
        cert_pem=cert_pem,
        uid=effective_uid,
        gid=effective_gid,
    )

    _tag_as_managed(container_name)

    run_tests(container_name, mounts=all_mounts, uid=effective_uid, gid=effective_gid)


def do_setup_crafts(
    container_name: str,
    craft_dirs: list[str],
    *,
    lxd_vm: bool = False,
) -> None:
    """Run 'make setup' in all craft project directories inside the container/VM.

    Raises ``RuntimeError`` if no craft_dirs are configured or container doesn't exist.
    """
    if not craft_dirs:
        raise RuntimeError(
            "No craft_dirs configured. "
            "Add them to the [lxd] section of config.toml, then re-run."
        )

    if not container_exists(container_name):
        raise RuntimeError(f"'{container_name}' does not exist.")

    is_vm = lxd_vm or container_is_vm(container_name)
    kind = "VM" if is_vm else "container"
    console.print(f"  Detected {container_name} as {kind}.")

    effective_uid = HOST_UID if is_vm else CONTAINER_UID
    effective_gid = HOST_GID if is_vm else CONTAINER_GID
    run_make_setup(container_name, craft_dirs, uid=effective_uid, gid=effective_gid)
    run_craft_setup_tests(container_name, craft_dirs)


def _refresh_one(container: str, cert_pem: str | None, uid: int, gid: int) -> None:
    """Run all refresh steps for a single managed container.

    Steps (in order):
    1. apt update + upgrade + autoremove
    2. npm update for pi (``@earendil-works/pi-coding-agent``)
    3. gh copilot update (self-update the Copilot CLI binary)
    4. Re-apply pi config (bridge /etc/hosts, models.json, cert, shell env vars)
    """
    console.print(f"\n[bold cyan]── Refreshing {container} ──[/bold cyan]")

    # 1. apt
    console.print("\n  [bold]apt:[/bold] update + upgrade + autoremove...")
    run(["lxc", "exec", container, "--", "apt-get", "update", "-q"])
    run(["lxc", "exec", container, "--", "apt-get", "upgrade", "-y"])
    run(["lxc", "exec", container, "--", "apt-get", "autoremove", "-y"])
    run(["lxc", "exec", container, "--", "apt-get", "clean"])

    # 2. pi (npm global)
    console.print("\n  [bold]pi:[/bold] updating @earendil-works/pi-coding-agent...")
    run(
        ["lxc", "exec", container, "--", "npm", "update", "-g", "@earendil-works/pi-coding-agent"],
    )

    # 3. copilot (self-update via built-in binary at ~/.local/share/gh/copilot/)
    console.print("\n  [bold]copilot:[/bold] updating gh copilot...")
    run(
        _cexec(container, uid, gid, "gh", "copilot", "update"),
    )

    # 4. pi config (bridge IP, cert, models.json, shell env vars)
    bridge_ip = _get_lxd_bridge_ip()
    if not bridge_ip:
        console.print(
            "    [yellow]Warning:[/yellow] Could not detect lxdbr0 bridge IP. "
            "/etc/hosts entry for 'local-llm' will not be updated."
        )
    setup_pi_in_container(container, bridge_ip=bridge_ip, cert_pem=cert_pem, uid=uid, gid=gid)

    console.print(f"\n  [green]✓[/green] {container} refresh complete")


def refresh_containers(
    container_name: str | None = None,
    *,
    cert_pem: str | None = None,
    lxd_vm: bool = False,
) -> None:
    """Update packages and re-apply config in managed LXD container(s).

    When *container_name* is ``None``, discovers every running container tagged
    with ``user.local-llm-managed=true`` and refreshes all of them.

    Raises ``RuntimeError`` if a named container doesn't exist or no managed
    containers are found.
    """
    if container_name is not None:
        if not container_exists(container_name):
            raise RuntimeError(f"'{container_name}' does not exist.")
        is_vm = lxd_vm or container_is_vm(container_name)
        uid = HOST_UID if is_vm else CONTAINER_UID
        gid = HOST_GID if is_vm else CONTAINER_GID
        _refresh_one(container_name, cert_pem=cert_pem, uid=uid, gid=gid)
    else:
        managed = _list_managed_containers()
        if not managed:
            raise RuntimeError(
                f"No running containers tagged with {_MANAGED_TAG}=true found."
            )

        console.print(
            f"Found [bold]{len(managed)}[/bold] managed container(s): "
            + ", ".join(managed)
        )
        for container in managed:
            is_vm = container_is_vm(container)
            uid = HOST_UID if is_vm else CONTAINER_UID
            gid = HOST_GID if is_vm else CONTAINER_GID
            _refresh_one(container, cert_pem=cert_pem, uid=uid, gid=gid)

        console.print(f"\n[green]✓[/green] All {len(managed)} container(s) refreshed.")
