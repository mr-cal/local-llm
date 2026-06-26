"""Hermes agent VM management.

Provides ``HermesVmManager`` for creating and maintaining the ``hermes`` LXD VM,
which runs the Nous Research Hermes agent
(https://hermes-agent.nousresearch.com/).

The hermes VM is deliberately minimal — no dev tools, no bind-mounts.
Credentials from ``config.toml [hermes]`` are injected into the VM after
the Hermes agent installs itself.
"""

from __future__ import annotations

import json
import subprocess

from rich.console import Console

from llm.config import HermesSettings

# Import shared LXD infrastructure from lxd.py
from llm.lxd import (
    CONTAINER_HOME,
    HOST_GID,
    HOST_UID,
    _BaseVmManager,
    _cexec,
    container_exists,
    run,
    run_with_retry,
)

console = Console()

HERMES_CONTAINER_NAME = "hermes"

# Hermes install script URL (official one-liner)
_HERMES_INSTALL_URL = "https://hermes-agent.nousresearch.com/install.sh"

# Minimal system packages — the Hermes install script handles uv, Python,
# Node.js, ripgrep, and ffmpeg itself.
_PREREQ_PACKAGES = [
    "curl",
    "git",
    "ca-certificates",
    "jq",
]


class HermesVmManager(_BaseVmManager):
    """Manages the hermes LXD VM running the Nous Research Hermes agent.

    Unlike ``LxdVmManager`` (which sets up full development environments),
    this manager installs only what Hermes needs and injects credentials from
    ``config.toml``.  No bind-mounts are created.

    Example::

        cfg = load_config().hermes
        mgr = HermesVmManager()
        mgr.create_and_setup(cfg)
    """

    def __init__(self) -> None:
        super().__init__(HERMES_CONTAINER_NAME, uid=HOST_UID, gid=HOST_GID)

    # ── Full setup workflow ───────────────────────────────────────────────

    def create_and_setup(self, cfg: HermesSettings, *, recreate: bool = False) -> None:
        """Create the hermes VM and configure the Hermes agent.

        Steps:
        1. Launch ubuntu:24.04 VM
        2. Configure passwordless sudo
        3. Install system prerequisites
        4. Run the Hermes install script
        5. Inject credentials from config.toml
        6. Install and enable the gateway systemd service
        7. Tag as managed
        """
        if container_exists(self.container):
            if not recreate:
                raise RuntimeError(
                    f"VM '{self.container}' already exists. Pass recreate=True to delete and recreate it."
                )
            console.print(f"Deleting existing VM: {self.container}")
            run(["lxc", "delete", "--force", self.container])

        console.print("\n[bold cyan]═══ Setting up hermes VM ═══[/bold cyan]\n")

        console.print("[bold][1/6][/bold] Launching VM...")
        self.create_container()
        self._configure_sudo()

        console.print("\n[bold][2/6][/bold] Installing prerequisites...")
        self._install_prerequisites()

        console.print("\n[bold][3/6][/bold] Installing Hermes agent...")
        self._install_hermes()

        console.print("\n[bold][4/6][/bold] Configuring credentials...")
        self._configure_credentials(cfg)

        console.print("\n[bold][5/6][/bold] Setting up gateway service...")
        self._setup_gateway_service()

        console.print("\n[bold][6/6][/bold] Tagging as managed...")
        self._tag_as_managed()

        console.print("\n[bold green]✓ hermes VM is ready![/bold green]")
        console.print(
            f"\n  Enter the VM:    [bold]lxc exec {self.container} -- su -l $USER[/bold]"
            f"\n  Start chatting:  [bold]hermes[/bold]"
            f"\n  Gateway status:  [bold]uv run llm hermes status[/bold]"
        )

    def refresh(self, cfg: HermesSettings) -> None:
        """Update packages, Hermes agent, and re-inject credentials."""
        console.print(f"\n[bold cyan]── Refreshing {self.container} ──[/bold cyan]")

        console.print("\n  [bold]apt:[/bold] update + upgrade...")
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

        console.print("\n  [bold]hermes:[/bold] updating...")
        run(_cexec(self.container, self.uid, self.gid, "hermes", "update"), desc="hermes update")

        console.print("\n  [bold]credentials:[/bold] re-injecting...")
        self._configure_credentials(cfg)

        # Restart the gateway service if it's already installed
        r = subprocess.run(
            _cexec(
                self.container,
                self.uid,
                self.gid,
                "systemctl",
                "--user",
                "is-active",
                "--quiet",
                "hermes-gateway",
            ),
            capture_output=True,
        )
        if r.returncode == 0:
            console.print("\n  [bold]gateway:[/bold] restarting service...")
            run(
                _cexec(
                    self.container,
                    self.uid,
                    self.gid,
                    "systemctl",
                    "--user",
                    "restart",
                    "hermes-gateway",
                ),
                desc="restart gateway",
            )
            console.print("  [green]✓[/green] gateway restarted")

        console.print(f"\n  [green]✓[/green] {self.container} refresh complete")

    def get_status(self) -> dict[str, str]:
        """Return VM and gateway service status.

        Returns a dict with keys ``vm`` and ``gateway``.
        """
        # VM status
        r = subprocess.run(
            ["lxc", "list", self.container, "--format=json"],
            capture_output=True,
            text=True,
        )
        vm_status = "unknown"
        try:
            data = json.loads(r.stdout)
            if data:
                vm_status = data[0].get("status", "unknown")
        except (json.JSONDecodeError, IndexError):
            pass

        # Gateway service status
        gateway_status = "unknown"
        if vm_status == "Running":
            r2 = subprocess.run(
                _cexec(
                    self.container,
                    self.uid,
                    self.gid,
                    "systemctl",
                    "--user",
                    "is-active",
                    "hermes-gateway",
                ),
                capture_output=True,
                text=True,
            )
            gateway_status = r2.stdout.strip() or ("active" if r2.returncode == 0 else "inactive")

        return {"vm": vm_status, "gateway": gateway_status}

    # ── Internal steps ────────────────────────────────────────────────────

    def _install_prerequisites(self) -> None:
        """Install minimal system packages needed before the Hermes install script."""
        run_with_retry(
            ["lxc", "exec", self.container, "--", "apt-get", "update", "-q"],
            desc="apt-get update",
        )
        run(["lxc", "exec", self.container, "--", "apt-get", "install", "-y", *_PREREQ_PACKAGES])
        run(["lxc", "exec", self.container, "--", "apt-get", "clean"])
        console.print("  [green]✓[/green] prerequisites installed")

    def _install_hermes(self) -> None:
        """Run the Hermes one-liner install script as the container user."""
        install_cmd = f"curl -fsSL {_HERMES_INSTALL_URL} | bash"
        run(
            _cexec(
                self.container,
                self.uid,
                self.gid,
                "bash",
                "-c",
                f"source /etc/profile && {install_cmd}",
            ),
            desc="hermes install",
        )
        console.print("  [green]✓[/green] Hermes agent installed")

    def _configure_credentials(self, cfg: HermesSettings) -> None:
        """Write API keys and tokens into ~/.hermes/.env inside the VM.

        Writes each configured secret directly to the Hermes env file.
        Running ``hermes config set`` non-interactively is fragile; writing
        .env directly is the reliable approach.
        """
        env_lines: list[str] = []

        if cfg.has_openrouter():
            env_lines.append(f"OPENROUTER_API_KEY={cfg.openrouter_key}")
            # Set OpenRouter as default provider via config.yaml
            run(
                _cexec(
                    self.container,
                    self.uid,
                    self.gid,
                    "hermes",
                    "config",
                    "set",
                    "model.provider",
                    "openrouter",
                ),
                desc="set openrouter provider",
            )

        if cfg.telegram_token:
            env_lines.append(f"TELEGRAM_BOT_TOKEN={cfg.telegram_token}")
        if cfg.telegram_allowed_users:
            env_lines.append(f"TELEGRAM_ALLOWED_USERS={cfg.telegram_allowed_users}")

        if cfg.github_token:
            env_lines.append(f"GITHUB_TOKEN={cfg.github_token}")

        if not env_lines:
            console.print("  [yellow]⚠[/yellow] No credentials configured — skipping.")
            console.print("  Set openrouter_key, telegram_token, etc. in [hermes] config.toml")
            return

        # Merge into ~/.hermes/.env (append or create)
        env_path = f"{CONTAINER_HOME}/.hermes/.env"
        subprocess.run(
            _cexec(
                self.container,
                self.uid,
                self.gid,
                "bash",
                "-c",
                # Ensure the .hermes dir exists, then write/replace each key
                f"mkdir -p {CONTAINER_HOME}/.hermes && "
                + " && ".join(
                    f"grep -qF '{line.split('=')[0]}=' {env_path} 2>/dev/null "
                    f"&& sed -i 's|^{line.split('=')[0]}=.*|{line}|' {env_path} "
                    f"|| echo '{line}' >> {env_path}"
                    for line in env_lines
                ),
            ),
            check=True,
        )
        console.print(f"  [green]✓[/green] credentials written to {env_path}")
        if cfg.has_openrouter():
            console.print("  [green]✓[/green] OpenRouter set as default provider")
        if cfg.has_telegram():
            console.print("  [green]✓[/green] Telegram gateway credentials configured")

    def _setup_gateway_service(self) -> None:
        """Install the Hermes gateway as a systemd user service.

        Runs ``hermes gateway install`` which generates
        ~/.config/systemd/user/hermes-gateway.service and enables it.
        Also enables linger so the service survives after logout.
        """
        run(
            _cexec(
                self.container,
                self.uid,
                self.gid,
                "hermes",
                "gateway",
                "install",
            ),
            desc="hermes gateway install",
        )
        # Enable linger so the user service persists after logout
        run(
            ["lxc", "exec", self.container, "--", "loginctl", "enable-linger", str(self.uid)],
            desc="loginctl enable-linger",
        )
        console.print("  [green]✓[/green] gateway service installed and linger enabled")
        console.print(
            "  Start with: [bold]lxc exec hermes -- "
            "su -l $USER -c 'systemctl --user start hermes-gateway'[/bold]"
        )
