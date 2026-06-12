"""LXD container and VM management.

Provides the LxdVmManager class for encapsulating all container operations,
plus module-level helper functions and wrapper functions for backward
compatibility.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import time
from collections.abc import Callable
from enum import Enum
from pathlib import Path

from rich.console import Console
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm import omp
from llm.config import (
    _build_omp_config_for_container,
    _build_pi_config_for_container,
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
            "fileExtensions": {".py": "python"},
        }
    }
}

console = Console()


# ── LxdVmManager class ──────────────────────────────────────────────────────


class SetupStep(Enum):
    """Ordered steps for LXD VM setup, used in progress labels like '2/4'."""

    MOUNTS = 2
    PACKAGES = 3
    PYLSP = 4
    NESTED_LXD = 4  # shares the same step as pylsp

    def label(self, total: int) -> str:
        """Return a display label like '2/4'."""
        return f"{self.value}/{total}"


class LxdVmManager:
    """Encapsulates all LXD VM operations for local LLM development.

    All container creation, configuration, verification, and refresh logic
    is managed through this class. Module-level functions at the bottom of
    this file delegate to this class for backward compatibility.

    Example::

        mgr = LxdVmManager("craft-llm-1", mounts=mounts)
        mgr.create_and_setup()

    Attributes:
        container: Name of the LXD container/VM.
        mounts: List of (name, host_path, container_path) tuples.
        craft_dirs: List of craft project directories.
        uid: UID for running commands inside the container.
        gid: GID for running commands inside the container.
    """

    def __init__(
        self,
        container: str,
        mounts: list[tuple[str, str, str]] | None = None,
        craft_dirs: list[str] | None = None,
        uid: int = HOST_UID,
        gid: int = HOST_GID,
    ) -> None:
        self.container = container
        self.mounts = mounts or list(_DEFAULT_MOUNTS)
        self.craft_dirs = craft_dirs or []
        self.uid = uid
        self.gid = gid

    # ── Container lifecycle ───────────────────────────────────────────────

    def create_container(self) -> None:
        """Launch an LXD VM (ubuntu:24.04) with default resources."""
        console.print(
            f"\n[bold][1/4][/bold] Launching {self.container} (ubuntu:24.04) as VM..."
        )
        launch_cmd = [
            "lxc",
            "launch",
            "ubuntu:24.04",
            self.container,
            "--vm",
            "--config",
            f"limits.memory={VM_MEMORY}",
            "--device",
            f"root,size={VM_ROOT_DISK_SIZE}",
        ]
        run_with_retry(launch_cmd, desc="lxc launch")
        wait_for_container(self.container)
        self._setup_vm_swap()
        # Rename the default ubuntu user/group to match the host user, and move
        # the home directory to the same path as on the host.
        run(
            [
                "lxc",
                "exec",
                self.container,
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
                self.container,
                "--",
                "groupmod",
                "--new-name",
                CONTAINER_USER,
                "ubuntu",
            ]
        )
        run(
            [
                "lxc",
                "exec",
                self.container,
                "--",
                "git",
                "config",
                "--global",
                "user.email",
                "mr-cal-bot@users.no-reply.github.com",
            ]
        )
        run(
            [
                "lxc",
                "exec",
                self.container,
                "--",
                "git",
                "config",
                "--global",
                "user.name",
                "mr-cal-bot",
            ]
        )
        self._fix_vm_user_uid()

    def _add_mounts(
        self,
        mounts: list[tuple[str, str, str]] | None = None,
        step: SetupStep = SetupStep.MOUNTS,
        total_steps: int = 4,
    ) -> None:
        """Add bind mounts to the container."""
        all_mounts = mounts or self.mounts
        label = step.label(total_steps)
        console.print(f"\n[bold][{label}][/bold] Adding bind mounts...")

        # Pre-create mount-point parent directories as the correct user so LXD
        # doesn't create them as root when it sets up the disk devices on the
        # next boot.
        parent_dirs = {str(Path(cp).parent) for _, _, cp in all_mounts}
        for parent in sorted(parent_dirs):
            run(_cexec(self.container, self.uid, self.gid, "mkdir", "-p", parent))

        for name, host_path, container_path in all_mounts:
            os.makedirs(host_path, exist_ok=True)
            run(
                [
                    "lxc",
                    "config",
                    "device",
                    "add",
                    self.container,
                    name,
                    "disk",
                    f"source={host_path}",
                    f"path={container_path}",
                ]
            )
        run(["lxc", "restart", self.container])
        wait_for_container(self.container)

    def _install_packages(
        self,
        step: SetupStep = SetupStep.PACKAGES,
        total_steps: int = 4,
        uid: int = CONTAINER_UID,
    ) -> None:
        """Install packages in the container."""
        label = step.label(total_steps)
        console.print(f"\n[bold][{label}][/bold] Installing packages...")
        run_with_retry(
            ["lxc", "exec", self.container, "--", "apt-get", "update", "-q"],
            desc="apt-get update",
        )
        run(
            [
                "lxc",
                "exec",
                self.container,
                "--",
                "apt-get",
                "install",
                "-y",
                "build-essential",
                "jq",
                "sponge",
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
        run(["lxc", "exec", self.container, "--", "bash", "-c", gh_setup])

        console.print("  Configuring passwordless sudo...")
        run(
            [
                "lxc",
                "exec",
                self.container,
                "--",
                "bash",
                "-c",
                f"printf 'User_Alias CONTAINERUSER = #{uid}\\n"
                f"CONTAINERUSER ALL=(ALL) NOPASSWD:ALL\\n'"
                f" > /etc/sudoers.d/nopasswd-user"
                f" && chmod 440 /etc/sudoers.d/nopasswd-user",
            ]
        )

        console.print("  Installing astral-uv...")
        run(["lxc", "exec", self.container, "--", "snap", "install", "astral-uv", "--classic"])

        console.print("  Installing helix...")
        run(["lxc", "exec", self.container, "--", "snap", "install", "helix", "--classic"])

        console.print("  Installing nodejs (for pi)...")
        run(
            [
                "lxc",
                "exec",
                self.container,
                "--",
                "bash",
                "-c",
                "set -euo pipefail && "
                "apt-get update -q && "
                "apt-get install -y nodejs npm curl && "
                "curl -fsSL https://deb.nodesource.com/setup_22.x "
                "| bash - && "
                "apt-get install -y nodejs",
            ]
        )

        console.print("  Installing oh-my-pi...")
        run(
            ["lxc", "exec", self.container, "--", "bash", "-c", "curl -fsSL https://omp.sh/install | sh"],
        )

        console.print("  Setting fish as the default shell...")
        run(
            ["lxc", "exec", self.container, "--", "chsh", "-s", "/usr/bin/fish", CONTAINER_USER],
        )

        console.print("  Cleaning up unused packages...")
        run(["lxc", "exec", self.container, "--", "apt-get", "autoremove", "-y"])
        run(["lxc", "exec", self.container, "--", "apt-get", "clean"])

    def _install_pylsp(
        self,
        step: SetupStep = SetupStep.PYLSP,
        total_steps: int = 4,
        uid: int = CONTAINER_UID,
        gid: int = CONTAINER_GID,
    ) -> None:
        """Install python-lsp-server via uv tool inside the container."""
        label = step.label(total_steps)
        console.print(
            f"\n[bold][{label}][/bold] "
            "Installing pylsp (python-lsp-server) in container..."
        )

        run(_cexec(self.container, uid, gid, "uv", "tool", "install", "python-lsp-server"))
        # Shorten the prompt
        run(
            _cexec(
                self.container,
                uid,
                gid,
                "bash",
                "-c",
                r'grep -qxF "export PS1=\"\w\$ \"" ~/.bashrc'
                r' || echo "export PS1=\"\w\$ \"" >> ~/.bashrc',
            )
        )

        # Write a fish conf.d snippet so `llm <n>` can pass CRAFT_CWD via the
        # environment and have the login shell cd there automatically.
        fish_conf_dir = f"{CONTAINER_HOME}/.config/fish/conf.d"
        craft_cwd_fish = (
            "# cd to CRAFT_CWD when set (passed by the host `llm` fish function)\n"
            "if set -q CRAFT_CWD; and test -d $CRAFT_CWD\n"
            "    cd $CRAFT_CWD\n"
            "end\n"
        )
        run(_cexec(self.container, uid, gid, "mkdir", "-p", fish_conf_dir))

        # Ensure ~/.local/bin is on PATH for both bash and fish.
        path_bash_line = "export PATH=$HOME/.local/bin:$PATH"
        path_fish_line = "set -gx PATH $HOME/.local/bin $PATH"
        run(
            _cexec(
                self.container,
                uid,
                gid,
                "bash",
                "-c",
                f"grep -qxF '{path_bash_line}' ~/.bashrc 2>/dev/null || "
                f"echo '{path_bash_line}' >> ~/.bashrc",
            )
        )
        run(
            _cexec(
                self.container,
                uid,
                gid,
                "bash",
                "-c",
                f"grep -qxF '{path_fish_line}' {fish_conf_dir}/path.fish 2>/dev/null || "
                f"echo '{path_fish_line}' > {fish_conf_dir}/path.fish",
            )
        )

        subprocess.run(
            _cexec(self.container, uid, gid, "bash", "-c", f"cat > {fish_conf_dir}/craft-cwd.fish"),
            input=craft_cwd_fish.encode(),
            check=True,
        )

        # Minimal fish prompt
        prompt_fish = (
            "# Minimal prompt: current directory + arrow (no user@host, no git hash)\n"
            "function fish_prompt\n"
            '    echo -n (set_color blue)(prompt_pwd)(set_color normal) " ❯ "\n'
            "end\n"
        )
        subprocess.run(
            _cexec(self.container, uid, gid, "bash", "-c", f"cat > {fish_conf_dir}/prompt.fish"),
            input=prompt_fish.encode(),
            check=True,
        )

        # pi "full context" helper: bump contextWindow and maxTokens in models.json
        pif_fish = (
            "function pif\n"
            '    jq \'.providers["local-llm"].models[0].contextWindow = 131092'
            ' | .providers["local-llm"].models[0].maxTokens = 32768\''
            " ~/.pi/agent/models.json | sponge ~/.pi/agent/models.json\n"
            "    pi\n"
            "end\n"
        )
        subprocess.run(
            _cexec(self.container, uid, gid, "bash", "-c", f"cat > {fish_conf_dir}/pif.fish"),
            input=pif_fish.encode(),
            check=True,
        )

        # Bash version of pif
        pif_bash = (
            '# pi "full context" helper: bump contextWindow and maxTokens, then run pi\n'
            'pif() { jq \'.providers["local-llm"].models[0].contextWindow = 131092 '
            '| .providers["local-llm"].models[0].maxTokens = 32768\''
            " ~/.pi/agent/models.json | sponge ~/.pi/agent/models.json; pi; }\n"
        )
        bashrc_cmd = (
            "grep -qxF 'pif()' ~/.bashrc 2>/dev/null || "
            '(tmp=$(mktemp) && cat > "$tmp" && cat "$tmp" >> ~/.bashrc && rm "$tmp")'
        )
        subprocess.run(
            _cexec(self.container, uid, gid, "bash", "-c", bashrc_cmd),
            input=pif_bash.encode(),
            check=True,
        )

        console.print(f"  Writing LSP config to {CONTAINER_HOME}/.copilot/lsp-config.json in container...")
        run(_cexec(self.container, uid, gid, "mkdir", "-p", f"{CONTAINER_HOME}/.copilot"))

        # Read any existing config from the container, then merge and write back.
        r = subprocess.run(
            _cexec(self.container, uid, gid, "cat", f"{CONTAINER_HOME}/.copilot/lsp-config.json"),
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
                console.print(
                    f"  [yellow]WARNING:[/yellow] lsp-config.json is invalid JSON ({e}); overwriting."
                )
        existing.setdefault("lspServers", {}).update(PYLSP_LSP_CONFIG["lspServers"])
        config_json = json.dumps(existing, indent=2) + "\n"
        # Validate before writing - guards against bugs in PYLSP_LSP_CONFIG.
        json.loads(config_json)

        lsp_path = f"{CONTAINER_HOME}/.copilot/lsp-config.json"
        subprocess.run(
            _cexec(self.container, uid, gid, "bash", "-c", f"cat > {lsp_path}"),
            input=config_json.encode(),
            check=True,
        )

    def _setup_nested_lxd(
        self,
        step: SetupStep = SetupStep.NESTED_LXD,
        total_steps: int = 4,
        uid: int = HOST_UID,
    ) -> None:
        """Install and initialise LXD inside the VM so nested containers can run."""
        label = step.label(total_steps)
        console.print(f"\n[bold][{label}][/bold] Setting up nested LXD inside VM...")

        console.print("  Installing lxd snap...")
        run(["lxc", "exec", self.container, "--", "snap", "install", "lxd"])

        console.print("  Initialising LXD (lxd init --auto)...")
        run(["lxc", "exec", self.container, "--", "lxd", "init", "--auto"])

        console.print(f"  Adding uid {uid} to lxd group...")
        r = subprocess.run(
            ["lxc", "exec", self.container, "--", "id", "-un", str(uid)],
            capture_output=True,
            text=True,
            check=True,
        )
        vm_username = r.stdout.strip()
        run(["lxc", "exec", self.container, "--", "usermod", "-aG", "lxd", vm_username])

        console.print("  Launching nested test container (ubuntu:24.04) to verify nesting...")
        run(
            [
                "lxc",
                "exec",
                self.container,
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
                self.container,
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

    # ── Setup workflows ───────────────────────────────────────────────────

    def create_and_setup(
        self, recreate: bool = False, cert_pem: str | None = None
    ) -> None:
        """Create and configure an LXD VM for local LLM development.

        This is the full setup workflow: create the VM, add mounts, install
        packages, configure tools, and run verification tests.
        """
        if container_exists(self.container):
            if not recreate:
                raise RuntimeError(
                    f"VM '{self.container}' already exists. "
                    "Pass recreate=True to delete and recreate it."
                )
            console.print(f"Deleting existing VM: {self.container}")
            run(["lxc", "delete", "--force", self.container])

        console.print(f"Creating VM: {self.container}")

        # Prepend a helix config bind-mount if the directory exists on the host.
        helix_host = os.path.join(HOST_HOME, ".config", "helix")
        helix_container = f"{CONTAINER_HOME}/.config/helix"
        all_mounts = list(self.mounts)
        if os.path.isdir(helix_host):
            all_mounts = [("helix-config", helix_host, helix_container), *all_mounts]
        else:
            console.print(
                "  [dim]~/.config/helix not found on host - "
                "skipping helix config mount[/dim]"
            )

        self.create_container()
        self._add_mounts(all_mounts, step=SetupStep.MOUNTS, total_steps=4)
        self._install_packages(step=SetupStep.PACKAGES, total_steps=4, uid=HOST_UID)
        self._install_pylsp(
            step=SetupStep.PYLSP, total_steps=4, uid=HOST_UID, gid=HOST_GID
        )
        self._setup_nested_lxd(
            step=SetupStep.NESTED_LXD, total_steps=4, uid=HOST_UID
        )

        self.setup_pi(cert_pem=cert_pem)
        self._tag_as_managed()

        self.run_tests()

    def do_setup_crafts(self) -> None:
        """Run 'make setup' in all craft project directories inside the VM.

        Raises ``RuntimeError`` if no craft_dirs are configured or VM doesn't exist.
        """
        if not self.craft_dirs:
            raise RuntimeError(
                "No craft_dirs configured. Add them to the [lxd] section of "
                "config.toml, then re-run."
            )

        if not container_exists(self.container):
            raise RuntimeError(f"'{self.container}' does not exist.")

        self.run_make_setup()
        self.run_craft_setup_tests()

    # ── Verification ──────────────────────────────────────────────────────

    def run_make_setup(self) -> None:
        """Run ``make setup`` in each craft project directory inside the container."""
        console.print("\nRunning make setup in craft directories (in container)...")
        for directory in self.craft_dirs:
            if not os.path.isdir(directory):
                console.print(
                    f"  [yellow]WARNING:[/yellow] directory not found on host, "
                    f"skipping: {directory}"
                )
                continue
            console.print(f"  Running make setup in {directory}...")
            run(
                [
                    "lxc",
                    "exec",
                    self.container,
                    f"--user={self.uid}",
                    f"--group={self.gid}",
                    f"--env=HOME={CONTAINER_HOME}",
                    f"--env=USER={CONTAINER_USER}",
                    f"--env=LOGNAME={CONTAINER_USER}",
                    f"--env=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:"
                    f"/usr/bin:/sbin:/bin:/snap/bin:{CONTAINER_HOME}/.local/bin",
                    "--env=CI=1",
                    "--",
                    "bash",
                    "-c",
                    f"make -C {directory} setup",
                ],
            )

    def run_tests(self) -> None:
        """Run verification tests against the configured container."""
        console.print("\n-- Verification tests ----------------------------------------------------------")

        def t_running() -> None:
            data = json.loads(
                run_capture(["lxc", "list", self.container, "--format=json"]).stdout
            )
            matches = [c for c in data if c["name"] == self.container]
            assert matches and matches[0]["status"] == "Running", (
                f"status={matches[0]['status'] if matches else 'not found'}"
            )

        def t_build_essential() -> None:
            subprocess.run(
                ["lxc", "exec", self.container, "--", "dpkg", "-l", "build-essential"],
                capture_output=True,
                check=True,
            )

        def t_gh_installed() -> None:
            subprocess.run(
                ["lxc", "exec", self.container, "--", "gh", "--version"],
                capture_output=True,
                check=True,
            )

        def t_dev_mount_read() -> None:
            r = subprocess.run(
                ["lxc", "exec", self.container, "--", "stat", "-c", "%a", f"{CONTAINER_HOME}/dev"],
                capture_output=True,
                text=True,
                check=True,
            )
            mode = r.stdout.strip()
            assert mode != "", "~/dev is not accessible in the container"

        def t_dev_ownership() -> None:
            r = subprocess.run(
                [
                    "lxc",
                    "exec",
                    self.container,
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

        def t_github_mount() -> None:
            if not any(name == "github" for name, _, _ in self.mounts):
                return
            subprocess.run(
                ["lxc", "exec", self.container, "--", "ls", f"{CONTAINER_HOME}/.github"],
                capture_output=True,
                check=True,
            )

        def t_opencode_config_mount() -> None:
            r = subprocess.run(
                [
                    "lxc",
                    "exec",
                    self.container,
                    "--",
                    "cat",
                    f"{CONTAINER_HOME}/.config/opencode/config.json",
                ],
                capture_output=True,
                text=True,
            )
            assert r.returncode == 0, (
                f"opencode config not found in container: {r.stderr.strip()}"
            )
            config = json.loads(r.stdout)
            assert "provider" in config, (
                f"'provider' key missing from opencode config: {config}"
            )

        def t_write_transparency() -> None:
            test_file = f"{HOST_HOME}/dev/.{self.container}_test_file"
            test_path = f"{CONTAINER_HOME}/dev/.{self.container}_test_file"
            subprocess.run(
                _cexec(self.container, self.uid, self.gid, "touch", test_path),
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
                _cexec(self.container, self.uid, self.gid, "sudo", "-n", "true"),
                capture_output=True,
                check=True,
            )

        def t_uv_installed() -> None:
            subprocess.run(
                ["lxc", "exec", self.container, "--", "uv", "--version"],
                capture_output=True,
                check=True,
            )

        def t_fish_installed() -> None:
            subprocess.run(
                ["lxc", "exec", self.container, "--", "fish", "--version"],
                capture_output=True,
                check=True,
            )

        def t_fish_default_shell() -> None:
            r = subprocess.run(
                ["lxc", "exec", self.container, "--", "getent", "passwd", CONTAINER_USER],
                capture_output=True,
                text=True,
                check=True,
            )
            shell = r.stdout.strip().split(":")[-1]
            assert shell == "/usr/bin/fish", f"shell is {shell!r}, expected '/usr/bin/fish'"

        def t_path_in_fish_conf() -> None:
            fish_conf_dir = f"{CONTAINER_HOME}/.config/fish/conf.d"
            r = subprocess.run(
                ["lxc", "exec", self.container, "--", "cat", f"{fish_conf_dir}/path.fish"],
                capture_output=True,
                text=True,
            )
            assert r.returncode == 0, f"fish path.fish not found: {r.stderr.strip()}"
            assert ".local/bin" in r.stdout, f".local/bin not in fish path.fish: {r.stdout}"

        def t_path_in_bashrc() -> None:
            r = subprocess.run(
                ["lxc", "exec", self.container, "--", "grep", ".local/bin", "~/.bashrc"],
                capture_output=True,
                text=True,
            )
            assert r.returncode == 0, f".local/bin not in ~/.bashrc: {r.stderr.strip()}"

        def t_pi_installed() -> None:
            r = subprocess.run(
                ["lxc", "exec", self.container, "--", "pi", "--version"],
                capture_output=True,
                text=True,
            )
            assert r.returncode == 0, f"pi not found in container: {r.stderr.strip()}"

        def t_omp_config() -> None:
            r = subprocess.run(
                ["lxc", "exec", self.container, "--", "cat", _OMP_CONTAINER_CONFIG],
                capture_output=True,
                text=True,
            )
            assert r.returncode == 0, f"models.yml not found in container: {r.stderr.strip()}"
            assert "local-llm" in r.stdout, (
                f"local-llm provider missing in models.yml: {r.stdout}"
            )
            assert "baseUrl" in r.stdout, f"baseUrl missing in models.yml: {r.stdout}"

        def t_pi_mount() -> None:
            r = subprocess.run(
                ["lxc", "exec", self.container, "--", "stat", "-c", "%a", f"{CONTAINER_HOME}/.pi"],
                capture_output=True,
                text=True,
                check=True,
            )
            mode = r.stdout.strip()
            assert mode != "", "~/.pi is not accessible in the container"

        def t_venv_exists() -> None:
            _t_venv_exists(self.container, self.craft_dirs)

        def t_container_user() -> None:
            r = subprocess.run(
                ["lxc", "exec", self.container, "--", "id", "-un", f"{self.uid}"],
                capture_output=True,
                text=True,
                check=True,
            )
            name = r.stdout.strip()
            assert name == CONTAINER_USER, (
                f"uid {self.uid} maps to {name!r}, expected {CONTAINER_USER!r}"
            )

        def t_venv_interpreter_valid() -> None:
            _t_venv_interpreter_valid(self.craft_dirs)

        def t_pylsp_installed() -> None:
            pylsp_bin = f"{CONTAINER_HOME}/.local/bin/pylsp"
            r = subprocess.run(
                _cexec(self.container, self.uid, self.gid, pylsp_bin, "--version"),
                capture_output=True,
                text=True,
            )
            assert r.returncode == 0, (
                f"pylsp not found in container at {pylsp_bin}: {r.stderr.strip()}"
            )

        def t_pylsp_lsp_config() -> None:
            container_config = f"{CONTAINER_HOME}/.copilot/lsp-config.json"
            r = subprocess.run(
                ["lxc", "exec", self.container, "--", "cat", container_config],
                capture_output=True,
                text=True,
            )
            assert r.returncode == 0, (
                f"lsp-config.json not found in container at {container_config}"
            )
            config = json.loads(r.stdout)
            servers = config.get("lspServers", {})
            assert "python" in servers, (
                f"'python' server missing from lspServers: {servers}"
            )
            assert servers["python"]["command"] == "pylsp", (
                f"unexpected command: {servers['python']['command']!r}"
            )

        def t_nested_lxd() -> None:
            r = subprocess.run(
                [
                    "lxc",
                    "exec",
                    self.container,
                    f"--user={self.uid}",
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
            k: v
            for k, v in (frame.f_locals if frame else {}).items()
            if k.startswith("t_") and callable(v)
        }
        assert frame is not None
        del frame  # avoid reference cycle (PEP 557)
        # Display names are derived from function names: t_foo_bar → "foo bar"
        tests: list[tuple[str, Callable]] = [
            (name.replace("t_", "").replace("_", " "), fn)
            for name, fn in local_tests.items()
        ]
        results = [check(name, fn) for name, fn in tests]
        passed = sum(results)
        total = len(results)

        console.print()
        if all(results):
            console.print("=" * 60)
            console.print("craft-llm container is ready!")
            console.print(
                f"  Mounts: ~/.github, ~/dev, ~/.config/opencode  ->  {CONTAINER_HOME}/{{...}}"
            )
            console.print(
                f"  UID/GID mapping: transparent (host {HOST_UID}:{HOST_GID} "
                f"<-> container {CONTAINER_USER})"
            )
            console.print(f"  Container user: {CONTAINER_USER}")
            console.print(
                "  Packages: build-essential, gh, gh-copilot, astral-uv, pi"
            )
            console.print("  sudo: passwordless for container user")
            console.print(
                "  Next: run 'gh auth login', 'gh copilot setup', and '/allow-all'"
            )
            console.print(
                " PAT token perms: all repos, actions, issues, merge queues, metadata, pull requests"
            )
            console.print("            user: copilot, gists")
            console.print(
                f"  pylsp: installed in container (~/.local/bin), config at {LSP_CONFIG_PATH}"
            )
            console.print(f"  All {total} tests passed.")
            console.print("=" * 60)
        else:
            console.print(
                f"[red]{passed}/{total} tests passed. See failures above.[/red]"
            )
            raise RuntimeError(f"{passed}/{total} tests passed.")

    def run_craft_setup_tests(self) -> None:
        """Run craft setup verification tests."""
        console.print("\n-- Craft setup verification ----------------------------------------------------")

        tests = [
            ("make setup completed (.venv)", lambda: _t_venv_exists(self.container, self.craft_dirs)),
            (
                "venv Python interpreters valid on host",
                lambda: _t_venv_interpreter_valid(self.craft_dirs),
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

    # ── Pi harness ────────────────────────────────────────────────────────

    def setup_pi(self, cert_pem: str | None = None) -> None:
        """Set up the Pi harness inside the container."""
        console.print(f"\n[bold]Setting up Pi harness in {self.container}...[/bold]")

        cfg = load_config()

        # Step 1: Add /etc/hosts entry
        from urllib.parse import urlparse  # noqa: PLC0415

        if cfg.client.server_url:
            server_ip = urlparse(cfg.client.server_url).hostname or cfg.proxy.lan_ip
        else:
            server_ip = cfg.proxy.lan_ip
        console.print(f"  Adding /etc/hosts entry: {server_ip} local-llm...")
        hosts_cmd = (
            f"grep -qxF '{server_ip} local-llm' /etc/hosts || "
            f"echo '{server_ip} local-llm' >> /etc/hosts"
        )
        subprocess.run(
            ["lxc", "exec", self.container, "--", "bash", "-c", hosts_cmd],
            check=True,
        )

        # Step 2: Generate models.json
        console.print("  Generating models.json with proxy URL...")
        pi_cfg = _build_pi_config_for_container(cfg, "local-llm")
        pi_config_dir = str(Path(_PI_CONTAINER_CONFIG).parent)
        subprocess.run(
            _cexec(self.container, self.uid, self.gid, "bash", "-c", f"mkdir -p {pi_config_dir}"),
            check=True,
        )

        r = _run_capture(self.container, "cat", _PI_CONTAINER_CONFIG)
        existing: dict = {}
        if r.returncode == 0 and r.stdout.strip():
            try:
                existing = json.loads(r.stdout)
                if not isinstance(existing, dict):
                    console.print(
                        "    [yellow]WARNING:[/yellow] existing config is not a JSON object"
                    )
                    existing = {}
            except json.JSONDecodeError:
                existing = {}
        existing.setdefault("providers", {}).update(pi_cfg.get("providers", {}))
        merged_json = json.dumps(existing, indent=2) + "\n"

        subprocess.run(
            _cexec(self.container, self.uid, self.gid, "bash", "-c", f"cat > {_PI_CONTAINER_CONFIG}"),
            input=merged_json.encode(),
            check=True,
        )
        console.print(
            f"    Written to {Path(_PI_CONTAINER_CONFIG).relative_to(CONTAINER_HOME)}"
        )

        # Step 2.5: Generate models.yml for oh-my-pi
        console.print("  Generating models.yml for oh-my-pi...")
        omp_cfg = _build_omp_config_for_container(cfg, "local-llm")
        omp_yaml = omp.build_omp_yaml(omp_cfg)
        omp_config_dir = str(Path(_OMP_CONTAINER_CONFIG).parent)
        subprocess.run(
            _cexec(self.container, self.uid, self.gid, "bash", "-c", f"mkdir -p {omp_config_dir}"),
            check=True,
        )
        subprocess.run(
            _cexec(self.container, self.uid, self.gid, "bash", "-c", f"cat > {_OMP_CONTAINER_CONFIG}"),
            input=omp_yaml.encode(),
            check=True,
        )
        console.print(
            f"    Written to {Path(_OMP_CONTAINER_CONFIG).relative_to(CONTAINER_HOME)}"
        )

        # Step 3: Install TLS certificate
        console.print("  Installing TLS certificate...")
        if cert_pem is None:
            cfg_cert_path = Path(cfg.client.cert_path or cfg.proxy.cert_path).expanduser()
            if cfg_cert_path.exists():
                cert_pem = cfg_cert_path.read_text()

        if cert_pem:
            subprocess.run(
                _cexec(self.container, self.uid, self.gid, "bash", "-c", f"mkdir -p {_NODE_CA_CERTS_DIR}"),
                check=True,
            )
            subprocess.run(
                _cexec(self.container, self.uid, self.gid, "bash", "-c", f"cat > {_NODE_CA_CERTS_FILE}"),
                input=cert_pem.encode(),
                check=True,
            )
            console.print(
                f"    Written to {Path(_NODE_CA_CERTS_FILE).relative_to(CONTAINER_HOME)}"
            )
        else:
            console.print(
                f"    [yellow]Warning:[/yellow] cert not found at {cfg.proxy.cert_path}. "
                "Run 'uv run llm config gencert' on the server first."
            )

        # Step 4: Set NODE_EXTRA_CA_CERTS in shell profiles
        console.print("  Configuring NODE_EXTRA_CA_CERTS in shell profiles...")
        bash_export = f'export NODE_EXTRA_CA_CERTS="{_NODE_CA_CERTS_FILE}"'
        fish_export = f'set -x NODE_EXTRA_CA_CERTS "{_NODE_CA_CERTS_FILE}"'

        bashrc_cmd = (
            f"grep -qxF '{bash_export}' ~/.bashrc || "
            f"echo '{bash_export}' >> ~/.bashrc"
        )
        subprocess.run(
            _cexec(self.container, self.uid, self.gid, "bash", "-c", bashrc_cmd),
            check=True,
        )

        fish_conf_dir = f"{CONTAINER_HOME}/.config/fish/conf.d"
        fish_conf = f"{fish_conf_dir}/node-ca-certs.fish"
        fish_cmd = (
            f"mkdir -p {fish_conf_dir} && "
            f"grep -qxF '{fish_export}' {fish_conf} 2>/dev/null || "
            f"echo '{fish_export}' > {fish_conf}"
        )
        subprocess.run(
            _cexec(self.container, self.uid, self.gid, "bash", "-c", fish_cmd),
            check=True,
        )

        console.print(
            "    Added to ~/.bashrc and ~/.config/fish/conf.d/node-ca-certs.fish"
        )
        console.print(
            "  [green]✓[/green] Pi harness configured - it can now reach the LLM server"
        )

    # ── Refresh ───────────────────────────────────────────────────────────

    def _refresh(self, cert_pem: str | None = None) -> None:
        """Run all refresh steps for this container."""
        console.print(f"\n[bold cyan]── Refreshing {self.container} ──[/bold cyan]")

        # 1. apt
        console.print("\n  [bold]apt:[/bold] update + upgrade + autoremove...")
        run_with_retry(
            ["lxc", "exec", self.container, "--", "apt-get", "update", "-q"],
            desc="apt-get update",
        )
        run_with_retry(
            ["lxc", "exec", self.container, "--", "apt-get", "upgrade", "-y"],
            desc="apt-get upgrade",
        )
        run(["lxc", "exec", self.container, "--", "apt-get", "autoremove", "-y"])
        run(["lxc", "exec", self.container, "--", "apt-get", "clean"])

        # 2. pi (oh-my-pi)
        console.print("\n  [bold]pi:[/bold] updating oh-my-pi...")
        run_with_retry(
            ["lxc", "exec", self.container, "--", "bash", "-c",
             "curl -fsSL https://omp.sh/install | sh"],
            desc="oh-my-pi install",
        )

        # 3. copilot
        console.print("\n  [bold]copilot:[/bold] updating gh copilot...")
        run(
            _cexec(self.container, self.uid, self.gid, "gh", "copilot", "update"),
        )

        # 4. pi + oh-my-pi config
        self.setup_pi(cert_pem=cert_pem)
        _refresh_omp_config(self.container, self.uid, self.gid)

        console.print(f"\n  [green]✓[/green] {self.container} refresh complete")

    def _tag_as_managed(self) -> None:
        """Set the managed tag on this container."""
        run(["lxc", "config", "set", self.container, f"{_MANAGED_TAG}=true"])

    # ── GH auth ───────────────────────────────────────────────────────────

    def setup_gh_auth(
        self,
        gh_token: str,
        *,
        effective_uid: int = CONTAINER_UID,
        effective_gid: int = CONTAINER_GID,
    ) -> None:
        """Authenticate the GitHub CLI inside the container using a PAT."""
        if not gh_token:
            console.print("  [yellow]⚠[/yellow] No GitHub token configured - skipping gh auth.")
            return

        console.print(f"  [bold]Authenticating gh CLI in {self.container}...[/bold]")

        subprocess.run(
            _cexec(self.container, effective_uid, effective_gid, "gh", "auth", "login", "--with-token"),
            input=gh_token.encode(),
            check=True,
        )
        console.print("  [green]✓[/green] gh authenticated - can access GitHub APIs")

    # ── Internal helpers ──────────────────────────────────────────────────

    def _setup_vm_swap(self) -> None:
        """Create a persistent swapfile inside the VM and enable it on boot."""
        run(
            [
                "lxc",
                "exec",
                self.container,
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

    def _fix_vm_user_uid(self) -> None:
        """Change the in-VM user's UID/GID to HOST_UID/HOST_GID."""
        if HOST_GID != CONTAINER_GID:
            console.print(
                f"  Changing in-VM group GID {CONTAINER_GID} → {HOST_GID} to match host..."
            )
            run(
                ["lxc", "exec", self.container, "--", "groupmod", "-g", str(HOST_GID), CONTAINER_USER]
            )
            r = subprocess.run(
                [
                    "lxc",
                    "exec",
                    self.container,
                    "--",
                    "bash",
                    "-c",
                    f"find / -xdev -group {CONTAINER_GID} -exec chgrp {HOST_GID} {{}} + 2>&1",
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                console.print(
                    f"[yellow]  Warning:[/yellow] chgrp may have missed some files: {r.stderr.strip()}"
                )
        if HOST_UID != CONTAINER_UID:
            console.print(
                f"  Changing in-VM user UID {CONTAINER_UID} → {HOST_UID} to match host..."
            )
            run(
                ["lxc", "exec", self.container, "--", "usermod", "-u", str(HOST_UID), CONTAINER_USER]
            )
            r = subprocess.run(
                [
                    "lxc",
                    "exec",
                    self.container,
                    "--",
                    "bash",
                    "-c",
                    f"find / -xdev -user {CONTAINER_UID} -exec chown {HOST_UID} {{}} + 2>&1",
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                console.print(
                    f"[yellow]  Warning:[/yellow] chown may have missed some files: {r.stderr.strip()}"
                )


# ── Constants (paths inside the container) ────────────────────────────────────

_PI_CONTAINER_CONFIG = f"{CONTAINER_HOME}/.pi/agent/models.json"
_OMP_CONTAINER_CONFIG = f"{CONTAINER_HOME}/.omp/agent/models.yml"
_NODE_CA_CERTS_DIR = f"{CONTAINER_HOME}/.config/local-llm"
_NODE_CA_CERTS_FILE = f"{_NODE_CA_CERTS_DIR}/cert.pem"
_MANAGED_TAG = "user.local-llm-managed"

VM_ROOT_DISK_SIZE = "50GB"
VM_MEMORY = "4GiB"
VM_SWAP_SIZE = "4G"


# ── YAML helpers ─────────────────────────────────────────────────────────────


def _refresh_omp_config(container: str, uid: int, gid: int) -> None:
    """Re-apply the oh-my-pi models.yml inside the container."""
    from llm.config import load_config  # noqa: PLC0415

    cfg = load_config()
    omp_cfg = _build_omp_config_for_container(cfg, "local-llm")

    # Read existing config from the container
    r = _run_capture(container, "cat", _OMP_CONTAINER_CONFIG)

    # Parse existing YAML into a dict to preserve other providers
    existing: dict = omp.parse_omp_yaml(r.stdout)
    merged: dict = {**existing}
    merged.setdefault("providers", {})
    merged["providers"].update(omp_cfg.get("providers", {}))

    merged_yaml = omp.build_omp_yaml(merged)
    subprocess.run(
        _cexec(container, uid, gid, "bash", "-c", f"cat > {_OMP_CONTAINER_CONFIG}"),
        input=merged_yaml.encode(),
        check=True,
    )


# ── Load settings ────────────────────────────────────────────────────────────


def load_lxd_settings() -> tuple[list[tuple[str, str, str]], list[str]]:
    """Load mounts and craft_dirs from config.toml [lxd], falling back to defaults."""
    lxd = try_load_lxd()
    if lxd is None:
        return _DEFAULT_MOUNTS, []
    mounts = (
        [
            (m.name, str(Path(m.host).expanduser()), str(Path(m.container).expanduser()))
            for m in lxd.mounts
        ]
        if lxd.mounts
        else _DEFAULT_MOUNTS
    )
    return mounts, [str(Path(d).expanduser()) for d in lxd.craft_dirs]


# ── Helper functions ─────────────────────────────────────────────────────────


def _cexec(container: str, uid: int, gid: int, *cmd: str) -> list[str]:
    """Build an ``lxc exec`` command running *cmd* as uid/gid inside *container*."""
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
        raise subprocess.CalledProcessError(
            e.returncode, e.cmd, e.output, e.stderr
        ) from Exception(f"Command failed ({label}): exit {e.returncode}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((subprocess.CalledProcessError, OSError)),
    reraise=True,
)
def run_with_retry(cmd, desc: str | None = None, **kwargs):
    """Run a command with automatic retry on transient failures.

    Retries up to 3 times with exponential backoff (2s → 4s → 8s) on
    ``subprocess.CalledProcessError`` or ``OSError`` (network timeout,
    LXD daemon busy, etc.). Intended for network-dependent operations
    like ``apt-get`` updates, ``curl``-based installs, and ``lxc launch``.
    """
    run(cmd, desc=desc, **kwargs)


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


# ── Tag / list helpers ──────────────────────────────────────────────────────


def _tag_as_managed(container: str) -> None:
    """Set the managed tag on *container* so it is discovered by ``llm client refresh``."""
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


# ── GH auth helper ──────────────────────────────────────────────────────────


def setup_gh_auth_in_container(
    container: str,
    gh_token: str,
    *,
    effective_uid: int = CONTAINER_UID,
    effective_gid: int = CONTAINER_GID,
) -> None:
    """Authenticate the GitHub CLI (gh) inside the container using a PAT."""
    mgr = LxdVmManager(container, uid=effective_uid, gid=effective_gid)
    mgr.setup_gh_auth(gh_token, effective_uid=effective_uid, effective_gid=effective_gid)


# ── Pi harness helper (standalone wrapper) ──────────────────────────────────


def setup_pi_in_container(
    container: str,
    cert_pem: str | None = None,
    uid: int = CONTAINER_UID,
    gid: int = CONTAINER_GID,
) -> None:
    """Set up the Pi harness inside the container so it can reach the LLM server."""
    mgr = LxdVmManager(container, uid=uid, gid=gid)
    mgr.setup_pi(cert_pem=cert_pem)


# ── Verification helpers ────────────────────────────────────────────────────


def _run_capture(container: str, *cmd: str) -> subprocess.CompletedProcess[str]:
    """Run a command inside the container and return the result."""
    return subprocess.run(
        ["lxc", "exec", container, "--", *cmd],
        capture_output=True,
        text=True,
    )


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


# ── Public wrapper functions (backward compatibility) ────────────────────────


def create_and_setup(
    container_name: str,
    *,
    mounts: list[tuple[str, str, str]],
    recreate: bool = False,
    cert_pem: str | None = None,
) -> None:
    """Create and configure an LXD VM for local LLM development.

    This is the library equivalent of the ``llm client setup --container``
    CLI command. Raises ``RuntimeError`` on fatal errors instead of
    ``typer.Exit``.

    Args:
        container_name: Name for the new LXD VM.
        mounts: List of (name, host_path, container_path) tuples.
        recreate: If True, delete an existing VM before creating a new one.
        cert_pem: Optional PEM certificate string for the nginx TLS proxy.
    """
    mgr = LxdVmManager(container_name, mounts=mounts)
    mgr.create_and_setup(recreate=recreate, cert_pem=cert_pem)


def do_setup_crafts(container_name: str, craft_dirs: list[str]) -> None:
    """Run 'make setup' in all craft project directories inside the VM.

    Args:
        container_name: Name of the existing LXD VM.
        craft_dirs: List of craft project directories on the host.

    Raises:
        RuntimeError: If no craft_dirs are configured or VM doesn't exist.
    """
    mgr = LxdVmManager(container_name, craft_dirs=craft_dirs)
    mgr.do_setup_crafts()


def refresh_containers(
    container_name: str | None = None,
    *,
    cert_pem: str | None = None,
) -> None:
    """Update packages and re-apply config in managed LXD VM(s).

    When *container_name* is ``None``, discovers every running VM tagged
    with ``user.local-llm-managed=true`` and refreshes all of them.

    Raises:
        RuntimeError: If a named VM doesn't exist or no managed VMs are found.
    """
    if container_name is not None:
        if not container_exists(container_name):
            raise RuntimeError(f"'{container_name}' does not exist.")
        mgr = LxdVmManager(container_name, uid=HOST_UID, gid=HOST_GID)
        mgr._refresh(cert_pem=cert_pem)
    else:
        managed = _list_managed_containers()
        if not managed:
            raise RuntimeError(
                f"No running VMs tagged with {_MANAGED_TAG}=true found."
            )

        console.print(
            f"Found [bold]{len(managed)}[/bold] managed VM(s): "
            + ", ".join(managed)
        )
        for container in managed:
            mgr = LxdVmManager(container, uid=HOST_UID, gid=HOST_GID)
            mgr._refresh(cert_pem=cert_pem)

        console.print(f"\n[green]✓[/green] All {len(managed)} VM(s) refreshed.")
