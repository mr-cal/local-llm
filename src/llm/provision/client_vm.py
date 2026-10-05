"""The developer client VM: packages, language servers, crafts and agents."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
from pathlib import Path

from llm import omp
from llm.core.console import console
from llm.provision import checks
from llm.provision.exec import (
    _DEFAULT_MOUNTS,
    _DEFAULT_SANDBOX_MOUNTS,
    _MANAGED_TAG,
    _SANDBOX_TAG,
    CONTAINER_GID,
    CONTAINER_HOME,
    CONTAINER_UID,
    CONTAINER_USER,
    HOST_GID,
    HOST_HOME,
    HOST_UID,
    KIND_CLIENT,
    KIND_HERMES,
    PYLSP_LSP_CONFIG,
    _cexec,
    _list_managed_containers,
    _run_capture,
    add_hosts_entry,
    container_exists,
    get_container_kind,
    is_container_sandbox,
    mkdir_p,
    run,
    run_with_retry,
    wait_for_container,
    write_file,
)
from llm.provision.vm import SetupStep, _BaseVmManager
from llm.render.client_configs import (
    _build_omp_config_for_container,
    _build_pi_config_for_container,
)
from llm.settings import load_config, try_load_lxd

_PI_CONTAINER_CONFIG = f"{CONTAINER_HOME}/.pi/agent/models.json"
_OMP_CONTAINER_CONFIG = f"{CONTAINER_HOME}/.omp/agent/models.yml"
_NODE_CA_CERTS_DIR = f"{CONTAINER_HOME}/.config/local-llm"
_NODE_CA_CERTS_FILE = f"{_NODE_CA_CERTS_DIR}/cert.pem"


class LxdVmManager(_BaseVmManager):
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
        sandbox: bool = False,
    ) -> None:
        super().__init__(container, uid=uid, gid=gid)
        self.sandbox = sandbox
        if mounts is not None:
            self.mounts = mounts
        else:
            self.mounts = list(_DEFAULT_SANDBOX_MOUNTS if sandbox else _DEFAULT_MOUNTS)
        self.craft_dirs = craft_dirs or []

    # ── Container lifecycle ───────────────────────────────────────────────

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
                "moreutils",
                "kitty-terminfo",
                "fish",
                "unzip",
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

        self._configure_sudo()

        console.print("  Installing astral-uv...")
        self._snap_install("astral-uv", "--classic")

        console.print("  Installing helix...")
        self._snap_install("helix", "--classic")

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

        console.print("  Installing pi...")
        run(
            _cexec(
                self.container,
                uid,
                CONTAINER_GID,
                "npm",
                "config",
                "set",
                "prefix",
                f"{CONTAINER_HOME}/.local",
            ),
        )
        run(
            _cexec(
                self.container,
                uid,
                CONTAINER_GID,
                "npm",
                "install",
                "-g",
                "--ignore-scripts",
                "@earendil-works/pi-coding-agent",
            ),
        )

        console.print("  Installing oh-my-pi...")
        run(
            _cexec(
                self.container,
                uid,
                CONTAINER_GID,
                "npm",
                "install",
                "-g",
                "--ignore-scripts",
                "@oh-my-pi/pi-coding-agent",
            ),
        )

        console.print("  Installing bun...")
        run(
            [
                "lxc",
                "exec",
                self.container,
                f"--user={uid}",
                f"--group={CONTAINER_GID}",
                f"--env=HOME={CONTAINER_HOME}",
                "--",
                "bash",
                "-c",
                "curl -fsSL https://bun.sh/install | bash",
            ],
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
        console.print(f"\n[bold][{label}][/bold] Installing pylsp (python-lsp-server) in container...")

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

        # Write a fish conf.d snippet so `llm <n>` can pass the target CWD via
        # /tmp/llm-cwd, which the login shell reads and removes on startup.
        # File-based handoff works reliably regardless of how su propagates env.
        fish_conf_dir = f"{CONTAINER_HOME}/.config/fish/conf.d"
        craft_cwd_fish = (
            "# cd to the directory pushed by the host `llm` function via /tmp/llm-cwd\n"
            "if test -f /tmp/llm-cwd\n"
            "    set -l cwd (string trim (cat /tmp/llm-cwd))\n"
            "    rm -f /tmp/llm-cwd\n"
            "    if test -d $cwd\n"
            "        cd $cwd\n"
            "    end\n"
            "end\n"
        )
        run(_cexec(self.container, uid, gid, "mkdir", "-p", fish_conf_dir))

        # Ensure ~/.local/bin, ~/.bun/bin, and ~/.cargo/bin are on PATH for both bash and fish.
        path_bash_line = "export PATH=$HOME/.local/bin:$HOME/.bun/bin:$HOME/.cargo/bin:$PATH"
        path_fish_line = "set -gx PATH $HOME/.local/bin $HOME/.bun/bin $HOME/.cargo/bin $PATH"
        run(
            _cexec(
                self.container,
                uid,
                gid,
                "bash",
                "-c",
                f"grep -qxF '{path_bash_line}' ~/.bashrc 2>/dev/null || echo '{path_bash_line}' >> ~/.bashrc",
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

        write_file(self.container, f"{fish_conf_dir}/craft-cwd.fish", craft_cwd_fish, uid, gid)

        # Minimal fish prompt
        prompt_fish = (
            "# Minimal prompt: current directory + arrow (no user@host, no git hash)\n"
            "function fish_prompt\n"
            '    echo -n (set_color blue)(prompt_pwd)(set_color normal) " ❯ "\n'
            "end\n"
        )
        write_file(self.container, f"{fish_conf_dir}/prompt.fish", prompt_fish, uid, gid)

        # pi "full context" helper: bump contextWindow and maxTokens in models.json
        pif_fish = (
            "function pif\n"
            '    jq \'.providers["local-llm"].models[0].contextWindow = 131092'
            ' | .providers["local-llm"].models[0].maxTokens = 32768\''
            " ~/.pi/agent/models.json | sponge ~/.pi/agent/models.json\n"
            "    pi\n"
            "end\n"
        )
        write_file(self.container, f"{fish_conf_dir}/pif.fish", pif_fish, uid, gid)

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
        write_file(self.container, lsp_path, config_json, uid, gid)

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
        self._snap_install("lxd")

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

    def create_and_setup(self, recreate: bool = False, cert_pem: str | None = None) -> None:
        """Create and configure an LXD VM for local LLM development.

        This is the full setup workflow: create the VM, add mounts, install
        packages, configure tools, and run verification tests.
        """
        if container_exists(self.container):
            if not recreate:
                raise RuntimeError(
                    f"VM '{self.container}' already exists. Pass recreate=True to delete and recreate it."
                )
            console.print(f"Deleting existing VM: {self.container}")
            # ZFS can transiently fail to remove the VM's mountpoint directory
            # right after stop (unmount race), so retry on failure.
            run_with_retry(["lxc", "delete", "--force", self.container], desc="lxc delete")

        console.print(f"Creating VM: {self.container}")

        # Prepend a helix config bind-mount if the directory exists on the host (non-sandbox only).
        helix_host = os.path.join(HOST_HOME, ".config", "helix")
        helix_container = f"{CONTAINER_HOME}/.config/helix"
        all_mounts = list(self.mounts)
        if not self.sandbox and os.path.isdir(helix_host):
            all_mounts = [("helix-config", helix_host, helix_container), *all_mounts]
        elif not self.sandbox:
            console.print("  [dim]~/.config/helix not found on host - skipping helix config mount[/dim]")

        self.create_container()
        self._add_mounts(all_mounts, step=SetupStep.MOUNTS, total_steps=4)
        self._install_packages(step=SetupStep.PACKAGES, total_steps=4, uid=HOST_UID)
        self._install_pylsp(step=SetupStep.PYLSP, total_steps=4, uid=HOST_UID, gid=HOST_GID)
        self._setup_nested_lxd(step=SetupStep.NESTED_LXD, total_steps=4, uid=HOST_UID)

        if not self.sandbox:
            self.setup_pi(cert_pem=cert_pem)
        else:
            console.print("  [dim]Sandbox container: omitting local LLM proxy setup & certs[/dim]")
        self._tag_as_managed()
        if self.sandbox:
            run(["lxc", "config", "set", self.container, f"{_SANDBOX_TAG}=true"])

        self.run_tests()

    def do_setup_crafts(self) -> None:
        """Run 'make setup' in all craft project directories inside the VM.

        Raises ``RuntimeError`` if no craft_dirs are configured or VM doesn't exist.
        """
        if not self.craft_dirs:
            raise RuntimeError(
                "No craft_dirs configured. Add them to the [lxd] section of config.toml, then re-run."
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
                    f"  [yellow]WARNING:[/yellow] directory not found on host, skipping: {directory}"
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
                    f"/usr/bin:/sbin:/bin:/snap/bin:{CONTAINER_HOME}/.local/bin:{CONTAINER_HOME}/.bun/bin:{CONTAINER_HOME}/.cargo/bin",
                    "--env=CI=1",
                    "--",
                    "bash",
                    "-c",
                    f"make -C {directory} setup",
                ],
            )

    def run_tests(self) -> None:
        """Run verification tests against the configured container."""
        checks.run_tests(
            self.container, self.mounts, self.craft_dirs, self.uid, self.gid, sandbox=self.sandbox
        )

    def run_craft_setup_tests(self) -> None:
        """Run craft setup verification tests."""
        checks.run_craft_setup_tests(self.container, self.craft_dirs)

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
        add_hosts_entry(self.container, server_ip, "local-llm")

        # Step 2: Generate models.json
        console.print("  Generating models.json with proxy URL...")
        pi_cfg = _build_pi_config_for_container(cfg, "local-llm")
        pi_config_dir = str(Path(_PI_CONTAINER_CONFIG).parent)
        mkdir_p(self.container, pi_config_dir, self.uid, self.gid)

        r = _run_capture(self.container, "cat", _PI_CONTAINER_CONFIG)
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

        write_file(self.container, _PI_CONTAINER_CONFIG, merged_json, self.uid, self.gid)
        console.print(f"    Written to {Path(_PI_CONTAINER_CONFIG).relative_to(CONTAINER_HOME)}")

        # Step 2.5: Generate models.yml for oh-my-pi
        console.print("  Generating models.yml for oh-my-pi...")
        omp_cfg = _build_omp_config_for_container(cfg, "local-llm")
        omp_yaml = omp.build_omp_yaml(omp_cfg)
        write_file(self.container, _OMP_CONTAINER_CONFIG, omp_yaml, self.uid, self.gid)
        console.print(f"    Written to {Path(_OMP_CONTAINER_CONFIG).relative_to(CONTAINER_HOME)}")

        # Step 3: Install TLS certificate
        console.print("  Installing TLS certificate...")
        if cert_pem is None:
            cfg_cert_path = Path(cfg.client.cert_path or cfg.proxy.cert_path).expanduser()
            if cfg_cert_path.exists():
                cert_pem = cfg_cert_path.read_text()

        if cert_pem:
            write_file(self.container, _NODE_CA_CERTS_FILE, cert_pem, self.uid, self.gid)
            console.print(f"    Written to {Path(_NODE_CA_CERTS_FILE).relative_to(CONTAINER_HOME)}")
        else:
            console.print(
                f"    [yellow]Warning:[/yellow] cert not found at {cfg.proxy.cert_path}. "
                "Run 'uv run llm config gencert' on the server first."
            )

        # Step 4: Set NODE_EXTRA_CA_CERTS in shell profiles
        console.print("  Configuring NODE_EXTRA_CA_CERTS in shell profiles...")
        bash_export = f'export NODE_EXTRA_CA_CERTS="{_NODE_CA_CERTS_FILE}"'
        fish_export = f'set -x NODE_EXTRA_CA_CERTS "{_NODE_CA_CERTS_FILE}"'

        bashrc_cmd = f"grep -qxF '{bash_export}' ~/.bashrc || echo '{bash_export}' >> ~/.bashrc"
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

        console.print("    Added to ~/.bashrc and ~/.config/fish/conf.d/node-ca-certs.fish")
        console.print("  [green]✓[/green] Pi harness configured - it can now reach the LLM server")

    # ── Refresh ───────────────────────────────────────────────────────────

    def _refresh(
        self,
        cert_pem: str | None = None,
        gh_token: str = "",
        git_username: str = "",
        git_email: str = "",
        git_pat: str = "",
    ) -> None:
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
            _cexec(
                self.container,
                self.uid,
                CONTAINER_GID,
                "npm",
                "config",
                "set",
                "prefix",
                f"{CONTAINER_HOME}/.local",
            ),
            desc="npm prefix",
        )
        run_with_retry(
            _cexec(
                self.container,
                self.uid,
                CONTAINER_GID,
                "npm",
                "install",
                "-g",
                "--ignore-scripts",
                "@earendil-works/pi-coding-agent",
            ),
            desc="pi install",
        )
        run_with_retry(
            _cexec(
                self.container,
                self.uid,
                CONTAINER_GID,
                "npm",
                "install",
                "-g",
                "--ignore-scripts",
                "@oh-my-pi/pi-coding-agent",
            ),
            desc="oh-my-pi install",
        )

        # 3. copilot
        console.print("\n  [bold]copilot:[/bold] updating gh copilot...")
        run(
            _cexec(self.container, self.uid, self.gid, "gh", "copilot", "update"),
        )

        # 4. pi + oh-my-pi config
        if not self.sandbox:
            self.setup_pi(cert_pem=cert_pem)
            _refresh_omp_config(self.container, self.uid, self.gid)

        # 5. gh auth + git identity
        self.setup_gh_auth(gh_token, effective_uid=self.uid, effective_gid=self.gid)
        self.setup_git_config(git_username, git_email, git_pat)

        console.print(f"\n  [green]✓[/green] {self.container} refresh complete")

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

    def setup_git_config(
        self,
        git_username: str,
        git_email: str,
        git_pat: str = "",
    ) -> None:
        """Configure git identity and push credentials inside the container.

        Runs as the container user so config lands in the user's ~/.gitconfig.
        When git_pat is set, configures git credential store and writes
        ~/.git-credentials so HTTPS pushes use the dedicated push token rather
        than the gh CLI credential helper.
        """
        if not git_username and not git_email and not git_pat:
            return

        console.print(f"  [bold]Configuring git in {self.container}...[/bold]")
        if git_username:
            run(
                _cexec(
                    self.container, self.uid, self.gid, "git", "config", "--global", "user.name", git_username
                )
            )
        if git_email:
            run(
                _cexec(
                    self.container, self.uid, self.gid, "git", "config", "--global", "user.email", git_email
                )
            )
        if git_pat:
            run(
                _cexec(
                    self.container,
                    self.uid,
                    self.gid,
                    "git",
                    "config",
                    "--global",
                    "credential.https://github.com.helper",
                    "store",
                )
            )
            creds = f"https://{git_username}:{git_pat}@github.com\n"
            write_file(
                self.container,
                f"{CONTAINER_HOME}/.git-credentials",
                creds,
                self.uid,
                self.gid,
                mode="600",
            )
        console.print(f"  [green]✓[/green] git identity: {git_username} <{git_email}>")
        if git_pat:
            console.print("  [green]✓[/green] git push credentials configured for github.com")


# ── YAML helpers ─────────────────────────────────────────────────────────────


def _refresh_omp_config(container: str, uid: int, gid: int) -> None:
    """Re-apply the oh-my-pi models.yml inside the container."""

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
    write_file(container, _OMP_CONTAINER_CONFIG, merged_yaml, uid, gid)


# ── Load settings ────────────────────────────────────────────────────────────


def load_lxd_settings(sandbox: bool = False) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Load mounts and craft_dirs from config.toml [lxd], falling back to defaults."""
    lxd = try_load_lxd()
    if lxd is None:
        return (_DEFAULT_SANDBOX_MOUNTS if sandbox else _DEFAULT_MOUNTS), []
    if sandbox:
        mounts = (
            [
                (m.name, str(Path(m.host).expanduser()), str(Path(m.container).expanduser()))
                for m in lxd.sandbox_mounts
            ]
            if lxd.sandbox_mounts
            else _DEFAULT_SANDBOX_MOUNTS
        )
    else:
        mounts = (
            [
                (m.name, str(Path(m.host).expanduser()), str(Path(m.container).expanduser()))
                for m in lxd.mounts
            ]
            if lxd.mounts
            else _DEFAULT_MOUNTS
        )
    return mounts, [str(Path(d).expanduser()) for d in lxd.craft_dirs]


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


def setup_git_config_in_container(
    container: str,
    git_username: str,
    git_email: str,
    git_pat: str = "",
    *,
    uid: int = CONTAINER_UID,
    gid: int = CONTAINER_GID,
) -> None:
    """Configure git identity and push credentials inside the container."""
    mgr = LxdVmManager(container, uid=uid, gid=gid)
    mgr.setup_git_config(git_username, git_email, git_pat)


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


# ── Public wrapper functions (backward compatibility) ────────────────────────


def create_and_setup(
    container_name: str,
    *,
    mounts: list[tuple[str, str, str]],
    recreate: bool = False,
    cert_pem: str | None = None,
    sandbox: bool = False,
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
        sandbox: If True, configure as an isolated sandbox container.
    """
    mgr = LxdVmManager(container_name, mounts=mounts, sandbox=sandbox)
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
    gh_token: str = "",
    git_username: str = "",
    git_email: str = "",
    git_pat: str = "",
) -> None:
    """Update packages and re-apply config in managed dev client VM(s).

    When *container_name* is ``None``, discovers every running VM tagged
    ``user.local-llm-managed=true`` whose kind is ``client`` and refreshes all
    of them. The Hermes agent VM is excluded: refreshing it would reconfigure
    it as a dev client and overwrite its agent setup. Use ``llm hermes setup``
    to update it instead.

    Raises:
        RuntimeError: If a named VM doesn't exist, is not a dev client, or if
            no managed client VMs are found.
    """
    cfg = None
    with contextlib.suppress(Exception):
        cfg = load_config()

    if container_name is not None:
        if not container_exists(container_name):
            raise RuntimeError(f"'{container_name}' does not exist.")
        if get_container_kind(container_name) == KIND_HERMES:
            raise RuntimeError(
                f"'{container_name}' is the Hermes agent VM, not a dev client. "
                "Run 'llm hermes setup' to update it."
            )
        is_sandbox = is_container_sandbox(container_name)
        if is_sandbox and cfg is not None:
            gh_token = cfg.github.sandbox.token
            git_username = cfg.github.sandbox.git_username
            git_email = cfg.github.sandbox.git_email
            git_pat = cfg.github.sandbox.git_pat
            cert_pem = None
        mgr = LxdVmManager(container_name, uid=HOST_UID, gid=HOST_GID, sandbox=is_sandbox)
        mgr._refresh(
            cert_pem=cert_pem,
            gh_token=gh_token,
            git_username=git_username,
            git_email=git_email,
            git_pat=git_pat,
        )
    else:
        managed = _list_managed_containers(kind=KIND_CLIENT)
        if not managed:
            raise RuntimeError(f"No running client VMs tagged with {_MANAGED_TAG}=true found.")

        console.print(f"Found [bold]{len(managed)}[/bold] managed VM(s): " + ", ".join(managed))
        for container in managed:
            is_sandbox = is_container_sandbox(container)
            c_gh_token = gh_token
            c_git_username = git_username
            c_git_email = git_email
            c_git_pat = git_pat
            c_cert_pem = cert_pem
            if is_sandbox and cfg is not None:
                c_gh_token = cfg.github.sandbox.token
                c_git_username = cfg.github.sandbox.git_username
                c_git_email = cfg.github.sandbox.git_email
                c_git_pat = cfg.github.sandbox.git_pat
                c_cert_pem = None
            mgr = LxdVmManager(container, uid=HOST_UID, gid=HOST_GID, sandbox=is_sandbox)
            mgr._refresh(
                cert_pem=c_cert_pem,
                gh_token=c_gh_token,
                git_username=c_git_username,
                git_email=c_git_email,
                git_pat=c_git_pat,
            )

        console.print(f"\n[green]✓[/green] All {len(managed)} VM(s) refreshed.")
