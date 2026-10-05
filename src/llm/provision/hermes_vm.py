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

from llm.core.console import console
from llm.core.proc import register_secrets

# Import shared LXD infrastructure from lxd.py
from llm.provision.exec import (
    CONTAINER_HOME,
    CONTAINER_USER,
    HERMES_CONTAINER_NAME,
    HOST_GID,
    HOST_UID,
    KIND_HERMES,
    _cexec,
    add_hosts_entry,
    container_exists,
    run,
    run_capture,
    run_with_retry,
)
from llm.provision.vm import _BaseVmManager
from llm.settings import HermesSettings, Settings, load_config

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
    "systemd",
    "libatomic1",
]

# Pinned version for mcp-server-mattermost. Updates are performed intentionally
# by maintainers after auditing releases and security advisories.
MATTERMOST_MCP_VERSION = "0.6.1"

# Read-only tool whitelist for Mattermost MCP integration.
# Restricting to read/search operations prevents unauthorized edits or mutations.
READ_ONLY_MATTERMOST_TOOLS = [
    "list_public_channels",
    "list_my_channels",
    "get_channel",
    "get_channel_by_name",
    "get_channel_messages",
    "get_thread",
    "search_messages",
    "get_team",
    "list_teams",
    "get_me",
    "get_user",
]


def _merge_env_file(existing: str, updates: dict[str, str]) -> str:
    """Merge *updates* into the contents of a KEY=VALUE env file.

    Keys already present are replaced in place so unrelated entries and their
    ordering survive a re-run; new keys are appended. Returns the full file
    contents, ready to be written over stdin.
    """
    remaining = dict(updates)
    lines_out: list[str] = []

    for line in existing.splitlines():
        key = line.split("=", 1)[0].strip()
        if key in remaining:
            lines_out.append(f"{key}={remaining.pop(key)}")
        else:
            lines_out.append(line)

    lines_out.extend(f"{key}={value}" for key, value in remaining.items())
    return "\n".join(lines_out).strip("\n") + "\n"


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
        self._tag_as_managed(kind=KIND_HERMES)

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

        # Restart the gateway service if it's already installed. The gateway
        # runs as a system-level systemd service (see _setup_gateway_service),
        # so this is a plain `systemctl`, not `systemctl --user`.
        r = subprocess.run(
            ["lxc", "exec", self.container, "--", "systemctl", "is-active", "--quiet", "hermes-gateway"],
            capture_output=True,
        )
        if r.returncode == 0:
            console.print("\n  [bold]gateway:[/bold] restarting service...")
            run(
                ["lxc", "exec", self.container, "--", "systemctl", "restart", "hermes-gateway"],
                desc="restart gateway",
            )
            console.print("  [green]✓[/green] gateway restarted")

        console.print(f"\n  [green]✓[/green] {self.container} refresh complete")

    def get_status(self, all_cfg: Settings | None = None) -> dict[str, str]:
        """Return VM and gateway service status.

        Returns a dict with keys ``vm``, ``gateway``, ``version``,
        ``uptime`` (seconds since gateway started), ``credentials_ok``,
        and ``mattermost``.
        """
        if all_cfg is None:
            try:
                all_cfg = load_config()
            except Exception:
                all_cfg = None

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
            # The gateway runs as a system-level systemd service (see
            # _setup_gateway_service), so plain `systemctl`, not `systemctl --user`.
            r2 = subprocess.run(
                ["lxc", "exec", self.container, "--", "systemctl", "is-active", "hermes-gateway"],
                capture_output=True,
                text=True,
            )
            gateway_status = r2.stdout.strip() or ("active" if r2.returncode == 0 else "inactive")

            # Uptime: systemctl show returns ActiveEnterTimestampEpoch in epoch seconds
            r3 = subprocess.run(
                [
                    "lxc",
                    "exec",
                    self.container,
                    "--",
                    "systemctl",
                    "show",
                    "--property=ActiveEnterTimestampEpoch",
                    "hermes-gateway",
                ],
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
        provider = "unknown"
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

            elif provider in ("custom", "openai", "local"):
                # Probe the local llama-server endpoint

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

        mattermost_status = "none"
        if vm_status == "Running" and all_cfg is not None and all_cfg.hermes.has_mattermost():
            r_mm = subprocess.run(
                _cexec(
                    self.container,
                    self.uid,
                    self.gid,
                    "curl",
                    "-fsSL",
                    "-H",
                    f"Authorization: Bearer {all_cfg.hermes.mattermost_token}",
                    f"{all_cfg.hermes.mattermost_url.rstrip('/')}/api/v4/users/me",
                ),
                capture_output=True,
                text=True,
            )
            mattermost_status = "connected" if r_mm.returncode == 0 else "unreachable"

        return {
            "vm": vm_status,
            "gateway": gateway_status,
            "version": version,
            "uptime": str(uptime_seconds),
            "provider": provider,
            "credentials_ok": str(credentials_ok),
            "mattermost": mattermost_status,
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

        Sets up model.provider, base_url, api_key, model.default, and
        optionally copies the CA cert into the VM when the TLS proxy is enabled.
        """
        # Add /etc/hosts entry so the "local-llm" hostname (used in the
        # endpoint URL and the cert's SubjectAltName) resolves inside the VM.
        from urllib.parse import urlparse  # noqa: PLC0415

        if cfg.client.server_url:
            server_ip = urlparse(cfg.client.server_url).hostname or cfg.proxy.lan_ip
        else:
            server_ip = cfg.proxy.lan_ip
        add_hosts_entry(self.container, server_ip, "local-llm")

        # Determine endpoint based on whether the proxy is enabled
        if cfg.proxy.enabled:
            local_url = f"https://local-llm:{cfg.proxy.port}/v1"
            self._install_ca_bundle(cfg.proxy.cert_path)
        else:
            local_url = f"http://local-llm:{cfg.server.port}/v1"

        local_api_key = cfg.auth.api_key or "no-key-required"
        register_secrets(local_api_key)

        self._hermes_run("config", "set", "model.provider", "custom", desc="set custom provider (local)")
        self._hermes_run("config", "set", "model.base_url", local_url, desc="set local base url")
        self._hermes_run("config", "set", "model.api_key", local_api_key, desc="set local api key")
        model_name = cfg.models.active or "local"
        self._hermes_run("config", "set", "model.default", model_name, desc="set local default model")

    def _install_ca_bundle(self, cert_src: str) -> None:
        """Copy the proxy's CA cert into the VM and trust it for HTTPS calls.

        Raises:
            RuntimeError: If the cert cannot be installed. Without it every
                request to the TLS proxy fails, so setup must not report
                success.
        """
        cert_dst = f"{CONTAINER_HOME}/.hermes/cert.pem"
        copy_result = subprocess.run(
            [
                "lxc",
                "file",
                "push",
                "--create-dirs",
                f"--uid={self.uid}",
                f"--gid={self.gid}",
                cert_src,
                f"{self.container}/{cert_dst.lstrip('/')}",
            ],
            capture_output=True,
            text=True,
        )
        if copy_result.returncode != 0:
            raise RuntimeError(
                f"Failed to copy CA cert {cert_src} into the VM: {copy_result.stderr.strip()}\n"
                "Hermes cannot reach the TLS proxy without it. Run 'llm server certs' "
                "to regenerate the cert, or disable proxy.enabled in config.toml."
            )
        console.print("  [green]✓[/green] CA cert copied into VM")

        # Hermes' Python OpenAI client (httpx) verifies TLS against certifi's
        # bundled CAs, not the OS trust store, so curl trusting our self-signed
        # cert isn't enough. Pointing SSL_CERT_FILE straight at our self-signed
        # cert would work for local-llm but replaces the *entire* trust store
        # for the process, breaking every other HTTPS call (Telegram,
        # OpenRouter, GitHub, ...). Instead, build a combined bundle — the
        # venv's certifi CAs plus our cert — and point SSL_CERT_FILE there.
        # Rebuilt on every setup/refresh so it tracks certifi upgrades and
        # cert rotation.
        bundle_dst = f"{CONTAINER_HOME}/.hermes/ca-bundle.pem"
        venv_python = f"{CONTAINER_HOME}/.hermes/hermes-agent/venv/bin/python"
        bundle_result = run_capture(
            _cexec(
                self.container,
                self.uid,
                self.gid,
                "bash",
                "-c",
                f"certifi_bundle=$({venv_python} -c 'import certifi; print(certifi.where())') && "
                f'cat "$certifi_bundle" {cert_dst} > {bundle_dst}',
            ),
        )
        if bundle_result.returncode != 0:
            raise RuntimeError(
                f"Failed to build the combined CA bundle in the VM: {bundle_result.stderr.strip()}\n"
                "Hermes would not trust the TLS proxy."
            )
        console.print("  [green]✓[/green] combined CA bundle built")
        self._write_env_vars(
            {"SSL_CERT_FILE": bundle_dst},
            desc="set SSL_CERT_FILE to combined CA bundle",
        )

    def _read_env_file(self, env_path: str) -> str:
        """Return the current contents of *env_path* in the VM, or "" if absent."""
        result = run_capture(_cexec(self.container, self.uid, self.gid, "cat", env_path))
        return result.stdout if result.returncode == 0 else ""

    def _write_env_vars(self, updates: dict[str, str], desc: str) -> str:
        """Merge *updates* into the VM's Hermes env file and return its path.

        The merged contents are sent over stdin rather than interpolated into a
        shell command, so credential values never appear in the process
        arguments (visible to ``ps``) or in echoed output, and tokens
        containing quotes, pipes or newlines cannot corrupt the command.
        """
        env_path = f"{CONTAINER_HOME}/.hermes/.env"
        merged = _merge_env_file(self._read_env_file(env_path), updates)
        run(
            _cexec(
                self.container,
                self.uid,
                self.gid,
                "bash",
                "-c",
                # env_path is a constant, not user input. 0600 keeps the
                # credentials out of reach of other users in the VM.
                f"mkdir -p {CONTAINER_HOME}/.hermes && cat > {env_path} && chmod 600 {env_path}",
            ),
            desc=desc,
            input=merged,
            text=True,
        )
        return env_path

    def _configure_credentials(self, cfg: HermesSettings, all_cfg: Settings | None = None) -> None:
        """Write API keys and tokens into ~/.hermes/.env inside the VM."""
        register_secrets(
            cfg.openrouter_key,
            cfg.telegram_token,
            cfg.github_token,
            cfg.mattermost_token,
        )

        env_vars: dict[str, str] = {}

        if cfg.has_openrouter():
            env_vars["OPENROUTER_API_KEY"] = cfg.openrouter_key
            # Set OpenRouter as default provider via config.yaml
            self._hermes_run("config", "set", "model.provider", "openrouter", desc="set openrouter provider")

        if cfg.has_local_llm() and all_cfg is not None:
            self._configure_local_llm(all_cfg)

        if cfg.telegram_token:
            env_vars["TELEGRAM_BOT_TOKEN"] = cfg.telegram_token
        if cfg.telegram_allowed_users:
            env_vars["TELEGRAM_ALLOWED_USERS"] = cfg.telegram_allowed_users

        if cfg.has_github():
            env_vars["GITHUB_TOKEN"] = cfg.github_token

        if cfg.mattermost_url:
            env_vars["MATTERMOST_URL"] = cfg.mattermost_url
        if cfg.mattermost_token:
            env_vars["MATTERMOST_TOKEN"] = cfg.mattermost_token
        if cfg.has_mattermost() and cfg.mattermost_team:
            env_vars["MATTERMOST_TEAM"] = cfg.mattermost_team

        if not env_vars and (not cfg.has_local_llm() or all_cfg is None):
            console.print("  [yellow]⚠[/yellow] No credentials configured — skipping.")
            console.print("  Set openrouter_key, telegram_token, etc. in [hermes] config.toml")
            return

        if env_vars:
            env_path = self._write_env_vars(env_vars, desc="write hermes credentials")
            console.print(f"  [green]✓[/green] credentials written to {env_path}")
        if cfg.has_openrouter():
            console.print("  [green]✓[/green] OpenRouter set as default provider")
        if cfg.has_local_llm() and all_cfg is not None:
            console.print("  [green]✓[/green] Local LLM set as default provider")
        if cfg.has_telegram():
            console.print("  [green]✓[/green] Telegram gateway credentials configured")
        if cfg.has_mattermost():
            console.print("  [green]✓[/green] Mattermost credentials configured")
            self._configure_mattermost_mcp(cfg)

        self._configure_concurrency(cfg)

    def _configure_concurrency(self, cfg: HermesSettings) -> None:
        """Configure concurrency limits in ~/.hermes/config.yaml."""
        venv_python = f"{CONTAINER_HOME}/.hermes/hermes-agent/venv/bin/python"
        config_path = f"{CONTAINER_HOME}/.hermes/config.yaml"
        concurrency_payload = {
            "max_concurrent_sessions": cfg.max_concurrent_sessions,
            "max_concurrent_children": cfg.max_concurrent_children,
        }
        update_script = (
            "import sys, json\n"
            "from pathlib import Path\n"
            "try:\n"
            "    from ruamel.yaml import YAML\n"
            "    yaml = YAML()\n"
            "    yaml.preserve_quotes = True\n"
            "except ImportError:\n"
            "    import yaml\n"
            "config_file = Path(sys.argv[1])\n"
            "opts = json.loads(sys.argv[2])\n"
            "data = {}\n"
            "if config_file.exists() and config_file.stat().st_size > 0:\n"
            "    with open(config_file) as f:\n"
            "        data = yaml.load(f) or {}\n"
            "data['max_concurrent_sessions'] = opts['max_concurrent_sessions']\n"
            "gateway = data.setdefault('gateway', {})\n"
            "gateway['max_concurrent_sessions'] = opts['max_concurrent_sessions']\n"
            "delegation = data.setdefault('delegation', {})\n"
            "delegation['max_concurrent_children'] = opts['max_concurrent_children']\n"
            "with open(config_file, 'w') as f:\n"
            "    yaml.dump(data, f)\n"
        )

        run(
            _cexec(
                self.container,
                self.uid,
                self.gid,
                venv_python,
                "-c",
                update_script,
                config_path,
                json.dumps(concurrency_payload),
            ),
            desc="configure concurrency limits in config.yaml",
        )
        console.print(
            f"  [green]✓[/green] Concurrency limits set: "
            f"max_concurrent_sessions={cfg.max_concurrent_sessions}, "
            f"max_concurrent_children={cfg.max_concurrent_children}"
        )

    def _configure_mattermost_mcp(self, cfg: HermesSettings) -> None:
        """Register the Mattermost MCP server in ~/.hermes/config.yaml."""
        if not cfg.has_mattermost():
            return

        mcp_config = {
            "command": "uvx",
            "args": [
                "--from",
                f"mcp-server-mattermost=={MATTERMOST_MCP_VERSION}",
                "mcp-server-mattermost",
            ],
            "env": {
                "MATTERMOST_URL": cfg.mattermost_url,
                "MATTERMOST_TOKEN": cfg.mattermost_token,
                "MATTERMOST_TEAM": cfg.mattermost_team,
            },
            "tools": {
                "include": READ_ONLY_MATTERMOST_TOOLS,
            },
        }

        venv_python = f"{CONTAINER_HOME}/.hermes/hermes-agent/venv/bin/python"
        config_path = f"{CONTAINER_HOME}/.hermes/config.yaml"
        update_script = (
            "import sys, json\n"
            "from pathlib import Path\n"
            "try:\n"
            "    from ruamel.yaml import YAML\n"
            "    yaml = YAML()\n"
            "    yaml.preserve_quotes = True\n"
            "except ImportError:\n"
            "    import yaml\n"
            "config_file = Path(sys.argv[1])\n"
            "data = {}\n"
            "if config_file.exists() and config_file.stat().st_size > 0:\n"
            "    with open(config_file) as f:\n"
            "        data = yaml.load(f) or {}\n"
            "servers = data.setdefault('mcp_servers', {})\n"
            "servers['mattermost'] = json.loads(sys.argv[2])\n"
            "with open(config_file, 'w') as f:\n"
            "    yaml.dump(data, f)\n"
        )

        run(
            _cexec(
                self.container,
                self.uid,
                self.gid,
                venv_python,
                "-c",
                update_script,
                config_path,
                json.dumps(mcp_config),
            ),
            desc="configure mattermost mcp in config.yaml",
        )
        console.print("  [green]✓[/green] Mattermost MCP server registered in config.yaml")

    def _setup_gateway_service(self) -> None:
        """Install the Hermes gateway as a system-level systemd service.

        ``hermes gateway install`` defaults to a per-user service under
        ``~/.config/systemd/user/``, which needs a running session bus.
        A fresh, non-interactive LXD VM has no logged-in session, so
        ``systemctl --user`` (and any workaround to bootstrap one) is
        fragile. Passing ``--system --run-as-user`` instead installs
        ``/etc/systemd/system/hermes-gateway.service``, managed directly by
        PID 1 — no session bus or linger required.
        """
        hermes_bin = f"{CONTAINER_HOME}/.local/bin/hermes"
        run(
            [
                "lxc",
                "exec",
                self.container,
                "--",
                "sudo",
                hermes_bin,
                "gateway",
                "install",
                "--system",
                "--run-as-user",
                CONTAINER_USER,
                "--start-now",
                "--start-on-login",
                "--force",
            ],
            desc="hermes gateway install --system",
        )
        console.print("  [green]✓[/green] gateway system service installed and started")

        # `sudo hermes gateway install` runs the Hermes CLI as root just long
        # enough to write the systemd unit, but importing Python modules
        # along the way leaves root-owned __pycache__ dirs inside the venv.
        # Left in place, those block later `uv sync`/pip operations run as
        # the unprivileged container user with "Permission denied" — reset
        # ownership back to that user so the venv stays writable.
        run(
            [
                "lxc",
                "exec",
                self.container,
                "--",
                "chown",
                "-R",
                f"{self.uid}:{self.gid}",
                f"{CONTAINER_HOME}/.hermes/hermes-agent/venv",
            ],
            desc="restore venv ownership after gateway install",
        )
        console.print("  Manage with: [bold]lxc exec hermes -- systemctl status hermes-gateway[/bold]")
