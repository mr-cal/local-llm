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
from typing import Any

from rich.console import Console

from llm.config import HermesSettings, Settings, load_config

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
# Node.js, ripgrep, and ffmpeg itself. libatomic1 isn't pulled in by the base
# image but is required by the prebuilt Node.js binary the script downloads;
# without it `node` fails with "error while loading shared libraries:
# libatomic.so.1" (which the install script reports as exit 127 / "Command
# not found").
_PREREQ_PACKAGES = [
    "curl",
    "git",
    "ca-certificates",
    "jq",
    "dbus",
    "dbus-user-session",
    "systemd",
    "libatomic1",
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

    def _hermes_run(self, *args: str, desc: str | None = None, **kwargs: Any) -> None:
        """Run a ``hermes`` CLI command inside the container with profile sourced.

        The ``hermes`` binary is installed into ``~/.local/bin/`` which only appears
        in PATH after ``/etc/profile`` is sourced (non-login shells skip it).
        """
        run(
            self._hermes_exec(*args),
            desc=desc or f"hermes {' '.join(args)}",
            **kwargs,
        )

    def _hermes_exec(self, *args: str) -> list[str]:
        """Build an ``lxc exec`` command that runs ``hermes`` with ~/.profile sourced.

        The ``hermes`` binary is installed into ``~/.local/bin/`` which is only in
        PATH after ``~/.profile`` is sourced (a login shell or explicit source).
        """
        return _cexec(
            self.container,
            self.uid,
            self.gid,
            "bash",
            "-c",
            'source ~/.profile && exec hermes "$@"',
            "_",
            *args,
        )

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

        self._configure_credentials(cfg, load_config())

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
        self._hermes_run("update", desc="hermes update")

        console.print("\n  [bold]credentials:[/bold] re-injecting...")
        self._configure_credentials(cfg, load_config())

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

        Returns a dict with keys ``vm``, ``gateway``, ``version``,
        ``uptime`` (seconds since gateway started), and ``credentials_ok``.
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
        uptime_seconds = 0
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

            # Uptime: systemctl show returns ActiveEnterTimestampEpoch in epoch seconds
            r3 = subprocess.run(
                _cexec(
                    self.container,
                    self.uid,
                    self.gid,
                    "systemctl",
                    "--user",
                    "show",
                    "--property=ActiveEnterTimestampEpoch",
                    "hermes-gateway",
                ),
                capture_output=True,
                text=True,
            )
            try:
                epoch = int(r3.stdout.split("=")[1].strip() or "0")
                if epoch > 0:
                    import time

                    uptime_seconds = max(0, int(time.time()) - epoch)
            except (IndexError, ValueError):
                pass

        # Version: run hermes --version inside the container
        version = "unknown"
        if vm_status == "Running":
            r4 = subprocess.run(
                self._hermes_exec("--version"),
                capture_output=True,
                text=True,
            )
            version = r4.stdout.strip() or "unknown"
        # Credentials check: probe the configured provider
        credentials_ok = False
        if vm_status == "Running":
            r5 = subprocess.run(
                self._hermes_exec("config", "get", "model.provider"),
                capture_output=True,
                text=True,
            )
            provider = r5.stdout.strip()
            if provider == "openrouter":
                # Quick probe: check that OpenRouter responds (non-401)
                r6 = subprocess.run(
                    _cexec(
                        self.container,
                        self.uid,
                        self.gid,
                        "curl",
                        "-fsSL",
                        "-H",
                        "Authorization: Bearer $OPENROUTER_API_KEY",
                        "https://openrouter.ai/api/v1/models",
                    ),
                    capture_output=True,
                    text=True,
                )
                credentials_ok = r6.returncode == 0

            elif provider in ("openai", "local"):
                # Probe the local llama-server endpoint
                from llm.config import load_config  # noqa: PLC0415

                all_cfg = load_config()
                if all_cfg.proxy.enabled:
                    url = f"https://local-llm:{all_cfg.proxy.port}/v1/models"
                    cert = f"{CONTAINER_HOME}/.hermes/cert.pem"
                    r6 = subprocess.run(
                        _cexec(
                            self.container,
                            self.uid,
                            self.gid,
                            "curl",
                            "-fsSL",
                            "--cacert",
                            cert,
                            "-H",
                            f"Authorization: Bearer {all_cfg.auth.api_key}",
                            url,
                        ),
                        capture_output=True,
                        text=True,
                    )
                    credentials_ok = r6.returncode == 0
                else:
                    port = all_cfg.server.port
                    r6 = subprocess.run(
                        _cexec(
                            self.container,
                            self.uid,
                            self.gid,
                            "curl",
                            "-fsSL",
                            f"http://local-llm:{port}/v1/models",
                        ),
                        capture_output=True,
                        text=True,
                    )
                    credentials_ok = r6.returncode == 0

        return {
            "vm": vm_status,
            "gateway": gateway_status,
            "version": version,
            "uptime": str(uptime_seconds),
            "credentials_ok": str(credentials_ok),
        }

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

    def _configure_local_llm(self, cfg: Settings) -> None:
        """Configure the Hermes agent to use the local llama-server.

        Sets up model.provider, endpoint, api_key, and optionally copies
        the CA cert into the VM when the TLS proxy is enabled.
        """
        # Determine endpoint based on whether the proxy is enabled
        if cfg.proxy.enabled:
            local_url = f"https://local-llm:{cfg.proxy.port}/v1"
            # Copy the CA cert into the VM so TLS is trusted
            cert_src = cfg.proxy.cert_path  # e.g. /etc/ssl/local-llm/cert.pem
            cert_dst = f"{CONTAINER_HOME}/.hermes/cert.pem"
            subprocess.run(
                [
                    "lxc",
                    "file",
                    "copy",
                    self.container,
                    "/",
                    "--",
                    f"--path=0{cert_src}",
                    f"{self.container}/{cert_dst.lstrip('/')}",
                ],
                capture_output=True,
            )
            console.print("  [green]✓[/green] CA cert copied into VM")
        else:
            local_url = f"http://local-llm:{cfg.server.port}/v1"

        local_api_key = cfg.auth.api_key

        self._hermes_run("config", "set", "model.provider", "openai", desc="set openai provider (local)")
        self._hermes_run("config", "set", "model.endpoint", local_url, desc="set local endpoint")
        self._hermes_run("config", "set", "model.api_key", local_api_key, desc="set local api key")

    def _configure_credentials(self, cfg: HermesSettings, all_cfg: Settings | None = None) -> None:
        """Write API keys and tokens into ~/.hermes/.env inside the VM.

        Writes each configured secret directly to the Hermes env file.
        Running ``hermes config set`` non-interactively is fragile; writing
        .env directly is the reliable approach.
        """
        env_lines: list[str] = []

        if cfg.has_openrouter():
            env_lines.append(f"OPENROUTER_API_KEY={cfg.openrouter_key}")
            # Set OpenRouter as default provider via config.yaml
            self._hermes_run("config", "set", "model.provider", "openrouter", desc="set openrouter provider")

        if cfg.has_local_llm() and all_cfg is not None:
            self._configure_local_llm(all_cfg)

        if cfg.telegram_token:
            env_lines.append(f"TELEGRAM_BOT_TOKEN={cfg.telegram_token}")
        if cfg.telegram_allowed_users:
            env_lines.append(f"TELEGRAM_ALLOWED_USERS={cfg.telegram_allowed_users}")

        if cfg.has_github():
            env_lines.append(f"GITHUB_TOKEN={cfg.github_token}")

        if not env_lines:
            console.print("  [yellow]⚠[/yellow] No credentials configured — skipping.")
            console.print("  Set openrouter_key, telegram_token, etc. in [hermes] config.toml")
            return

        # Merge into ~/.hermes/.env (append or create)
        env_path = f"{CONTAINER_HOME}/.hermes/.env"
        run(
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
            desc="write hermes credentials",
        )
        console.print(f"  [green]✓[/green] credentials written to {env_path}")
        if cfg.has_openrouter():
            console.print("  [green]✓[/green] OpenRouter set as default provider")
        if cfg.has_local_llm():
            console.print("  [green]✓[/green] Local LLM set as default provider")
        if cfg.has_telegram():
            console.print("  [green]✓[/green] Telegram gateway credentials configured")

    def _setup_gateway_service(self) -> None:
        """Install the Hermes gateway as a systemd user service.

        Runs ``hermes gateway install`` which generates
        ~/.config/systemd/user/hermes-gateway.service and enables it.
        Also enables linger so the service survives after logout.
        """
        # Start dbus-daemon and user systemd before gateway install.
        # In a fresh container, user systemd hasn't been initialized yet.
        # dbus-daemon must be started first (systemctl --user needs a running bus),
        # then systemd-user can connect and daemon-reload will work.
        run(
            [
                "lxc", "exec", self.container, "--",
                "bash", "-c",
                "dbus-daemon --session --fork --print-pid && "
                "systemctl --user daemon-reload",
            ],
            desc="init user dbus/systemd",
        )
        self._hermes_run("gateway", "install", desc="hermes gateway install")
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
