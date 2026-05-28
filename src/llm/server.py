"""Server management: setup, start, stop, restart, status, logs."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from llm.config import CONFIG_FILENAME, load_config

app = typer.Typer(help="Manage the llama-server process.", no_args_is_help=True)
console = Console()


# ── server setup ──────────────────────────────────────────────────────────────


def _prompt(prompt: str, default: str = "") -> str:
    """Prompt the user for input, showing a default."""
    suffix = f" [{default}]" if default else ""
    result = input(f"{prompt}{suffix}: ").strip()
    return result or default


def _prompt_choice(prompt: str, choices: list[str], default: str = "") -> str:
    """Prompt the user to pick from a list of choices."""
    for i, c in enumerate(choices, 1):
        marker = " (default)" if c == default else ""
        print(f"  {i}. {c}{marker}")
    raw = input(f"{prompt} [1-{len(choices)}]: ").strip()
    if not raw and default:
        return default
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(choices):
            return choices[idx]
    except ValueError:
        if raw in choices:
            return raw
    return default or choices[0]


@app.command("setup")
def setup(
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing TLS cert even if present."),
    ] = False,
) -> None:
    """Guided setup for the server (and local client).

    Creates or updates config.toml, generates TLS cert + API key,
    renders nginx/systemd configs, and configures the local client
    (opencode, pi, shell env vars).  Safe to re-run.
    """
    from llm.config import (  # noqa: PLC0415
        apply_client_configs,
        apply_server_configs,
        configure_shell_env_host,
        detect_lan_ip,
        generate_api_key,
        generate_tls_cert,
        write_config_toml,
    )

    project_root = Path.cwd()
    config_path = project_root / CONFIG_FILENAME
    existing_cfg: dict | None = None  # type: ignore[type-arg]

    console.print("\n[bold cyan]═══ local-llm server setup ═══[/bold cyan]\n")

    # ── Load existing config if present ───────────────────────────────────
    if config_path.exists():
        import tomllib  # noqa: PLC0415

        with config_path.open("rb") as f:
            existing_cfg = tomllib.load(f)
        console.print(f"[dim]Found existing config: {config_path}[/dim]\n")

    def _get(section: str, key: str, fallback: str = "") -> str:
        if existing_cfg and section in existing_cfg:
            return str(existing_cfg[section].get(key, fallback))
        return fallback

    # ── Step 1: LAN IP ────────────────────────────────────────────────────
    console.print("[bold]Step 1/6[/bold] — Network")
    detected_ip = detect_lan_ip()
    current_ip = _get("proxy", "lan_ip", detected_ip or "192.168.1.100")
    lan_ip = _prompt("  LAN IP", current_ip)

    # Derive subnet from IP (e.g. 192.168.1.100 → 192.168.1.0/24)
    parts = lan_ip.rsplit(".", 1)
    default_subnet = f"{parts[0]}.0/24" if len(parts) == 2 else "192.168.1.0/24"
    current_subnet = _get("proxy", "lan_subnet", default_subnet)
    lan_subnet = _prompt("  LAN subnet", current_subnet)

    proxy_port = int(_prompt("  Proxy port (HTTPS)", _get("proxy", "port", "8443")))
    server_port = int(_prompt("  Server port (internal)", _get("server", "port", "8080")))

    # ── Step 2: Auth ──────────────────────────────────────────────────────
    console.print("\n[bold]Step 2/6[/bold] — Authentication")
    current_key = _get("auth", "api_key", "")
    if current_key:
        console.print(f"  [dim]API key already set ({current_key[:8]}…)[/dim]")
        api_key = current_key
    else:
        api_key = generate_api_key()
        console.print(f"  [green]Generated API key:[/green] {api_key[:16]}…")

    # ── Step 3: Server tuning ─────────────────────────────────────────────
    console.print("\n[bold]Step 3/6[/bold] — Server tuning")
    n_gpu_layers = int(_prompt("  GPU layers", _get("server", "n_gpu_layers", "20")))
    n_ctx = int(_prompt("  Context size", _get("server", "n_ctx", "65536")))
    n_threads = int(_prompt("  Threads", _get("server", "n_threads", "12")))

    # ── Step 4: Model ─────────────────────────────────────────────────────
    console.print("\n[bold]Step 4/6[/bold] — Model")
    models_dir = _prompt("  Models directory", _get("models", "dir", "~/models"))
    active_model = _get("models", "active", "qwen2.5-coder-14b-q4")
    console.print(f"  Active model: [bold]{active_model}[/bold]")
    console.print("  [dim](Change models later with: llm model switch)[/dim]")

    # ── Build config dict ─────────────────────────────────────────────────
    cfg_dict: dict = {}  # type: ignore[type-arg]
    if existing_cfg:
        cfg_dict = dict(existing_cfg)

    cfg_dict["proxy"] = {
        **cfg_dict.get("proxy", {}),
        "lan_ip": lan_ip,
        "lan_subnet": lan_subnet,
        "port": proxy_port,
    }
    cfg_dict["server"] = {
        **cfg_dict.get("server", {}),
        "port": server_port,
        "n_gpu_layers": n_gpu_layers,
        "n_ctx": n_ctx,
        "n_threads": n_threads,
    }
    cfg_dict["auth"] = {"api_key": api_key}
    cfg_dict["models"] = {
        **cfg_dict.get("models", {}),
        "dir": models_dir,
        "active": active_model,
    }

    # ── Write config.toml ─────────────────────────────────────────────────
    console.print("\n[bold]Step 5/6[/bold] — Writing config")
    write_config_toml(cfg_dict, config_path)
    console.print(f"  [green]✓[/green] {config_path}")

    # Reload config from the file we just wrote
    cfg = load_config()

    # ── Generate TLS cert ─────────────────────────────────────────────────
    if not generate_tls_cert(cfg, force=force):
        console.print("  [red]✗[/red] TLS cert generation failed")
        raise typer.Exit(1)

    # ── Step 6: Apply configs ─────────────────────────────────────────────
    console.print("\n[bold]Step 6/6[/bold] — Applying configs")

    # Server configs (nginx, systemd)
    apply_server_configs(cfg, project_root)

    # Client configs for the local machine (opencode, pi, shell env)
    apply_client_configs(cfg)

    # Shell env vars
    cert_path = cfg.proxy.cert_path
    base_url = f"https://{lan_ip}:{proxy_port}/v1"
    actions = configure_shell_env_host(base_url, api_key, cert_path)
    for action in actions:
        console.print(f"  [green]✓[/green] {action}")

    # ── Done ──────────────────────────────────────────────────────────────
    console.print("\n[bold green]✓ Server setup complete![/bold green]")
    console.print(
        "\n  Start the server:     [bold]uv run llm server start[/bold]"
        "\n  Check server status:  [bold]uv run llm server status[/bold]"
        "\n  Set up a container:   [bold]uv run llm client setup --container <name>[/bold]"
    )


# ── nginx helpers ─────────────────────────────────────────────────────────────


def _nginx_is_active() -> bool:
    result = subprocess.run(
        ["systemctl", "is-active", "nginx"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() == "active"


def _nginx_start() -> bool:
    """Start nginx via systemctl. Returns True on success."""
    result = subprocess.run(["sudo", "systemctl", "start", "nginx"], capture_output=True, text=True)
    return result.returncode == 0


def _nginx_reload() -> bool:
    """Reload nginx config. Returns True on success."""
    result = subprocess.run(["sudo", "systemctl", "reload", "nginx"], capture_output=True, text=True)
    return result.returncode == 0


def _nginx_stop() -> bool:
    """Stop nginx via systemctl. Returns True on success."""
    result = subprocess.run(["sudo", "systemctl", "stop", "nginx"], capture_output=True, text=True)
    return result.returncode == 0


def _nginx_ensure_running() -> None:
    """Start nginx if it isn't already running; reload if it is."""
    if _nginx_is_active():
        console.print("[dim]  (sudo systemctl reload nginx)[/dim]")
        if _nginx_reload():
            console.print("[green]nginx[/green]       reloaded")
        else:
            console.print("[yellow]nginx[/yellow]       reload failed — check: sudo nginx -t")
    else:
        console.print("[dim]  (sudo systemctl start nginx)[/dim]")
        if _nginx_start():
            console.print("[green]nginx[/green]       started")
        else:
            console.print("[yellow]nginx[/yellow]       failed to start — check: sudo systemctl status nginx")


# Runtime files live alongside the config in the project directory.
# Both are gitignored.
_PID_FILE = Path(".server.pid")
_LOG_FILE = Path(".server.log")


def _pid_file() -> Path:
    """Resolve PID file path relative to CWD (project root)."""
    return _PID_FILE


def _log_file() -> Path:
    return _LOG_FILE


def _read_pid(port: int | None = None) -> int | None:
    """Return running server PID, or None if not running.

    Tries two strategies:
    1. Read the PID file (fast path)
    2. Fall back to finding the process listening on *port* via ss
    """
    # Strategy 1: PID file
    pf = _pid_file()
    if pf.exists():
        try:
            pid = int(pf.read_text().strip())
            os.kill(pid, 0)  # signal 0 = existence check
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            pf.unlink(missing_ok=True)

    # Strategy 2: probe the port (fallback when PID file is missing)
    if port is not None:
        result = subprocess.run(
            ["ss", "-tlnp", f"sport = :{port}"],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines()[1:]:  # skip header
            if "llama-server" in line:
                m = re.search(r"pid=(\d+)", line)
                if m:
                    try:
                        pid = int(m.group(1))
                        os.kill(pid, 0)  # verify it's still alive
                        return pid
                    except (ValueError, ProcessLookupError, PermissionError):
                        pass
    return None


@app.command("start")
def start(
    wait: Annotated[int, typer.Option("--wait", help="Seconds to wait for server to be ready.")] = 5,
    profile: Annotated[
        str | None,
        typer.Option(
            "--profile", "-p",
            help="Build profile to use (overrides config.toml [server] profile).",
        ),
    ] = None,
) -> None:
    """Start llama-server using settings from config.toml."""
    cfg = load_config()

    if not cfg.has_local_server:
        console.print(
            "[yellow]No local server configured.[/yellow]  "
            "Set [server] llama_server_bin or configure [[build.profiles]] in config.toml."
        )
        raise typer.Exit(1)

    existing = _read_pid(cfg.server.port)
    if existing:
        console.print(f"[yellow]Server already running[/yellow] (PID {existing})")
        raise typer.Exit(1)

    if not cfg.model_path.exists():
        console.print(f"[red]Model file not found:[/red] {cfg.model_path}")
        console.print("Run [bold]uv run llm model list[/bold] to see available models.")
        raise typer.Exit(1)

    # Resolve binary: --profile flag > config override > auto-resolve
    if profile:
        p = cfg.build.get_profile(profile)
        if p is None:
            console.print(
                f"[red]Unknown profile:[/red] '{profile}'\n"
                f"Available: {', '.join(cfg.build.profile_names())}"
            )
            raise typer.Exit(1)
        bin_path = str(p.installed_server_bin(cfg.build.install_path))
    else:
        bin_path = cfg.resolve_llama_server_bin()

    cmd: list[str] = [
        bin_path,
        "--model",
        str(cfg.model_path),
        "--port",
        str(cfg.server.port),
        "--n-gpu-layers",
        str(cfg.server.n_gpu_layers),
        "--ctx-size",
        str(cfg.server.n_ctx),
        "--threads",
        str(cfg.server.n_threads),
        *cfg.server.extra_args,
    ]

    log_path = _log_file()
    log_fh = log_path.open("a")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,  # detach from current session
        )
    except FileNotFoundError:
        log_fh.close()
        console.print(f"[red]Binary not found:[/red] {bin_path}")
        console.print(
            "\nBuild llama.cpp and install the binary:\n"
            "  [bold]uv run llm build init[/bold]   # initialize submodule\n"
            "  [bold]uv run llm build run[/bold]    # build active profile\n"
            "\nOr set [bold]llama_server_bin[/bold] explicitly in config.toml:\n"
            '  llama_server_bin = "~/.local/bin/llama-server"'
        )
        raise typer.Exit(1) from None

    _pid_file().write_text(str(proc.pid))
    console.print(f"[green]Started[/green] llama-server (PID {proc.pid})")
    console.print(f"  Model  : {cfg.models.active}")
    console.print(f"  Port   : {cfg.server.port}")
    console.print(f"  Layers : {cfg.server.n_gpu_layers} (Vulkan iGPU)")
    if cfg.server.extra_args:
        console.print(f"  Extra  : {' '.join(cfg.server.extra_args)}")
    console.print(f"  Logs   : {log_path.resolve()}")

    if wait > 0:
        console.print(f"Waiting up to {wait}s for server to be ready...", end="")
        import httpx

        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            time.sleep(0.5)
            try:
                httpx.get(f"{cfg.internal_url}/health", timeout=1).raise_for_status()
                console.print(" [green]ready[/green]")
                break
            except Exception:
                pass
        else:
            console.print(" [yellow]timeout (server may still be loading)[/yellow]")

    _nginx_ensure_running()


@app.command("stop")
def stop() -> None:
    """Stop the running llama-server and nginx."""
    cfg = load_config()
    pid = _read_pid(cfg.server.port)
    if pid is None:
        console.print("[yellow]Server is not running[/yellow]")
        raise typer.Exit(1)

    os.kill(pid, signal.SIGTERM)
    # Wait briefly for clean shutdown
    for _ in range(20):
        time.sleep(0.25)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break

    _pid_file().unlink(missing_ok=True)
    console.print(f"[green]Stopped[/green] llama-server (PID {pid})")

    if _nginx_is_active():
        if _nginx_stop():
            console.print("[green]Stopped[/green] nginx")
        else:
            console.print("[yellow]nginx[/yellow]       failed to stop — check: sudo systemctl status nginx")


@app.command("restart")
def restart() -> None:
    """Restart llama-server (stop then start)."""
    cfg = load_config()
    pid = _read_pid(cfg.server.port)
    if pid:
        stop()
    start()


@app.command("status")
def status() -> None:
    """Show whether llama-server and nginx are running."""
    from llm.config import load_config  # noqa: PLC0415 (already imported at top)

    cfg = load_config()

    if not cfg.has_local_server:
        console.print("[dim]No local server configured — client-only mode.[/dim]")
        console.print(f"  Connecting to: [cyan]{cfg.client_url}[/cyan]")
        if _nginx_is_active():
            console.print("[green]● nginx[/green]         active")
        return

    pid = _read_pid(cfg.server.port)
    if pid:
        from llm.models import KNOWN_MODELS  # noqa: PLC0415

        # Resolve: check config catalog first, then KNOWN_MODELS
        entry = cfg.models.by_alias(cfg.models.active)
        if entry is None:
            entry = cfg.models.by_filename(cfg.models.active)
        if entry is None and cfg.models.has_catalog is False:
            entry = next((m for m in KNOWN_MODELS if m.filename == cfg.models.active), None)
        display = entry.alias if entry else cfg.models.active
        console.print(f"[green]● llama-server[/green]  PID {pid}  port {cfg.server.port}")
        console.print(f"  Model  : {display}  [dim]({cfg.models.active})[/dim]")
        console.print(f"  Layers : {cfg.server.n_gpu_layers}")
        if cfg.server.extra_args:
            console.print(f"  Extra  : {' '.join(cfg.server.extra_args)}")
        # Show active build profile if configured
        active_profile = cfg.build.active_profile
        if active_profile:
            profile_name = cfg.server.profile or active_profile.name
            console.print(f"  Profile: {profile_name}")
        log = _log_file().resolve()
        console.print(f"  Logs   : {log}  [dim](uv run llm server logs -f)[/dim]")
    else:
        console.print("[red]● llama-server[/red]  stopped")
        console.print("  Run [bold]uv run llm server start[/bold] to start.")

    if _nginx_is_active():
        console.print("[green]● nginx[/green]         active")
    else:
        console.print("[red]● nginx[/red]         stopped")


@app.command("logs")
def logs(
    lines: Annotated[int, typer.Option("-n", help="Number of lines to show.")] = 50,
    follow: Annotated[bool, typer.Option("-f", "--follow", help="Follow log output.")] = False,
) -> None:
    """Show server logs."""
    log_path = _log_file()
    if not log_path.exists():
        console.print(f"[yellow]No log file found:[/yellow] {log_path}")
        raise typer.Exit(1)

    cmd = ["tail", f"-{lines}", str(log_path)]
    if follow:
        cmd.insert(1, "-f")
    subprocess.run(cmd, check=False)
