"""Server management: setup, start, stop, restart, status, logs."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from llm.config import CONFIG_FILENAME, find_config, load_config
from llm.core import fmt, http, proc
from llm.core.console import console

app = typer.Typer(help="Manage the llama-server process.", no_args_is_help=True)

# systemd is the only supervisor for llama-server: it owns the process, the
# restart policy and the logs. The unit is rendered from config.toml and
# installed by `llm server apply`.
SERVICE_UNIT = "llm-server"
_UNIT_PATH = Path("/etc/systemd/system/llm-server.service")


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
    (opencode, pi, shell env vars). Safe to re-run.
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

    proc.ensure_sudo()

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
    console.print("[bold]Step 1/6[/bold] - Network")
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
    console.print("\n[bold]Step 2/6[/bold] - Authentication")
    current_key = _get("auth", "api_key", "")
    if current_key:
        console.print(f"  [dim]API key already set ({current_key[:8]}…)[/dim]")
        api_key = current_key
    else:
        api_key = generate_api_key()
        console.print(f"  [green]Generated API key:[/green] {api_key[:16]}…")

    # ── Step 3: Server tuning ─────────────────────────────────────────────
    console.print("\n[bold]Step 3/6[/bold] - Server tuning")
    n_gpu_layers = int(_prompt("  GPU layers", _get("server", "n_gpu_layers", "20")))
    n_ctx = int(_prompt("  Context size", _get("server", "n_ctx", "65536")))
    n_threads = int(_prompt("  Threads", _get("server", "n_threads", "12")))

    # ── Step 4: Model ─────────────────────────────────────────────────────
    console.print("\n[bold]Step 4/6[/bold] - Model")
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
    console.print("\n[bold]Step 5/6[/bold] - Writing config")
    write_config_toml(cfg_dict, config_path)
    console.print(f"  [green]✓[/green] {config_path}")

    # Reload config from the file we just wrote
    cfg = load_config()

    # ── Generate TLS cert ─────────────────────────────────────────────────
    if not generate_tls_cert(cfg, force=force):
        console.print("  [red]✗[/red] TLS cert generation failed")
        raise typer.Exit(1)

    # ── Step 6: Apply configs ─────────────────────────────────────────────
    console.print("\n[bold]Step 6/6[/bold] - Applying configs")

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
    return proc.unit_is_active("nginx")


def _nginx_start() -> bool:
    """Start nginx via systemctl. Returns True on success."""
    result = proc.sudo(["systemctl", "start", "nginx"])
    return result.returncode == 0


def _nginx_reload() -> bool:
    """Reload nginx config. Returns True on success."""
    result = proc.sudo(["systemctl", "reload", "nginx"])
    return result.returncode == 0


def _nginx_stop() -> bool:
    """Stop nginx via systemctl. Returns True on success."""
    result = proc.sudo(["systemctl", "stop", "nginx"])
    return result.returncode == 0


def _llm_server_unit_installed() -> bool:
    """Return True when the llm-server unit has been installed."""
    return _UNIT_PATH.exists()


def _require_unit_installed() -> None:
    """Exit with guidance when the llm-server unit has not been installed yet."""
    if _llm_server_unit_installed():
        return
    console.print(f"[red]systemd unit not installed:[/red] {_UNIT_PATH}")
    console.print("Render and install it with: [bold]uv run llm server apply[/bold]")
    raise typer.Exit(1)


def _llm_server_systemctl(action: str) -> bool:
    """Run ``systemctl <action> llm-server``. Returns True on success."""
    result = proc.sudo(["systemctl", action, SERVICE_UNIT])
    if result.returncode != 0 and result.stderr.strip():
        console.print(f"  [dim]{result.stderr.strip()}[/dim]")
    return result.returncode == 0


def _unit_main_pid() -> int | None:
    """Return the llm-server unit's main PID, or None if it isn't running.

    systemd reports MainPID=0 for an inactive unit.
    """
    result = subprocess.run(
        ["systemctl", "show", "-p", "MainPID", "--value", SERVICE_UNIT],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        pid = int(result.stdout.strip())
    except ValueError:
        return None
    return pid or None


def _nginx_ensure_running() -> None:
    """Start nginx if it isn't already running; reload if it is."""
    if _nginx_is_active():
        console.print("[dim]  (sudo systemctl reload nginx)[/dim]")
        if _nginx_reload():
            console.print("[green]nginx[/green]       reloaded")
        else:
            console.print("[yellow]nginx[/yellow]       reload failed - check: sudo nginx -t")
    else:
        console.print("[dim]  (sudo systemctl start nginx)[/dim]")
        if _nginx_start():
            console.print("[green]nginx[/green]       started")
        else:
            console.print("[yellow]nginx[/yellow]       failed to start - check: sudo systemctl status nginx")


def _project_root() -> Path:
    """Return the project root directory (parent of config.toml)."""
    return find_config().parent


def _configs_are_stale() -> bool:
    """Return True if config.toml is newer than the last-rendered nginx/systemd files.

    Rendered files are only produced by ``server apply`` (or ``server setup``).
    Missing rendered files are treated as stale.
    """
    root = _project_root()
    config_file = root / CONFIG_FILENAME
    if not config_file.exists():
        return False
    config_mtime = config_file.stat().st_mtime
    rendered = [
        root / "nginx" / "llm-proxy.conf",
        root / "systemd" / "llm-server.service",
    ]
    return any(not f.exists() or config_mtime > f.stat().st_mtime for f in rendered)


def _warn_if_stale() -> None:
    """Print a warning if rendered configs are older than config.toml."""
    if _configs_are_stale():
        console.print(
            "[yellow]⚠  config.toml has changed since configs were last applied.[/yellow]\n"
            "   Run [bold]uv run llm server apply[/bold] to update nginx/systemd configs.\n"
        )


# Runtime files for the memory monitor live alongside the config in the
# project directory. Both are gitignored. llama-server itself is supervised by
# systemd and logs to the journal, so it has no PID or log file of its own.
_MONITOR_PID_FILE = Path(".server-monitor.pid")
_MONITOR_LOG_FILE = Path(".server-monitor.log")


def _server_pid(port: int | None = None) -> int | None:
    """Return the running llama-server PID, or None if it isn't running.

    Asks systemd first, then falls back to probing *port*. The fallback still
    matters: it finds a llama-server started outside the unit, which callers
    such as the benchmark sweep need to know about.
    """
    pid = _unit_main_pid()
    if pid is not None:
        return pid

    if port is not None:
        result = subprocess.run(
            ["ss", "-tlnp", f"sport = :{port}"],
            capture_output=True,
            text=True,
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


def _server_is_ready(port: int) -> bool:
    """Check if llama-server is ready to accept requests via /health.

    Returns True if the server responds with 200 OK on the /health
    endpoint, False otherwise (still loading, crashed, etc.).
    """
    return http.is_healthy(f"http://127.0.0.1:{port}/health")


def _read_monitor_pid() -> int | None:
    """Return the memory-monitor PID if it is still alive."""
    pf = _MONITOR_PID_FILE
    if not pf.exists():
        return None
    try:
        pid = int(pf.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        pf.unlink(missing_ok=True)
        return None


def _start_monitor(cfg: object, server_pid: int) -> None:
    """Spawn the detached memory-monitor daemon for the running server."""
    from llm.config import Settings  # noqa: PLC0415

    assert isinstance(cfg, Settings)
    if cfg.server.monitor_interval <= 0:
        return

    cmd = [
        sys.executable,
        "-m",
        "llm.monitor",
        str(server_pid),
        "--interval",
        str(cfg.server.monitor_interval),
        "--retention-days",
        str(cfg.server.monitor_retention_days),
    ]

    log_fh = _MONITOR_LOG_FILE.open("a")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=log_fh,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        log_fh.close()
        console.print("[yellow]Could not start memory monitor[/yellow]")
        return

    _MONITOR_PID_FILE.write_text(str(proc.pid))
    console.print(
        f"[dim]Memory monitor started[/dim] (PID {proc.pid}, every "
        f"{cfg.server.monitor_interval}s → logs/memory-monitor.csv)"
    )


def _stop_monitor() -> None:
    """Stop the memory-monitor daemon if it is running."""
    pid = _read_monitor_pid()
    if pid is None:
        return
    os.kill(pid, signal.SIGTERM)
    _MONITOR_PID_FILE.unlink(missing_ok=True)


def _check_oom() -> None:
    """Best-effort scan of the kernel log for OOM kills."""
    result = subprocess.run(
        ["journalctl", "-k", "--no-pager", "--grep", "Out of memory"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print("[yellow]Could not read the kernel log.[/yellow]")
        console.print("  Try: [bold]sudo journalctl -k | grep -i 'out of memory'[/bold]")
        return
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        console.print("[green]No OOM kills found in the kernel log.[/green]")
        return
    console.print(f"[red]{len(lines)} OOM kill(s)[/red] in kernel log:")
    for ln in lines[-10:]:
        console.print(f"  {ln}")


def _process_uptime_seconds(pid: int) -> int | None:
    """Return how long *pid* has been running in seconds, or None if unknown."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        boot_uptime = float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError):
        return None
    # /proc/<pid>/stat field 2 (comm) is parenthesized and may itself contain
    # spaces or parens, so parse everything after the last ")". Field 22
    # (starttime) is then the 20th token (index 19) of what remains.
    rest = stat.rpartition(")")[2].split()
    try:
        clk_tck = os.sysconf("SC_CLK_TCK")
    except (ValueError, OSError):
        clk_tck = 100  # USER_HZ fallback on x86 Linux
    try:
        start_since_boot = int(rest[19]) / clk_tck
    except (IndexError, ValueError):
        return None
    return max(0, int(boot_uptime - start_since_boot))


@app.command("start")
def start(
    wait: Annotated[int, typer.Option("--wait", help="Seconds to wait for server to be ready.")] = 5,
) -> None:
    """Start llama-server via its systemd unit.

    The command line comes from the unit rendered by ``llm server apply``, so
    change settings (including the build profile) in config.toml and re-apply
    rather than passing overrides here.
    """
    proc.ensure_sudo()
    _warn_if_stale()
    cfg = load_config()

    if not cfg.has_local_server:
        console.print(
            "[yellow]No local server configured.[/yellow]  "
            "Set [server] llama_server_bin or configure [[build.profiles]] in config.toml."
        )
        raise typer.Exit(1)

    _require_unit_installed()

    existing = _server_pid(cfg.server.port)
    if existing:
        console.print(f"[yellow]Server already running[/yellow] (PID {existing})")
        raise typer.Exit(1)

    if not cfg.model_path.exists():
        console.print(f"[red]Model file not found:[/red] {cfg.model_path}")
        console.print("Run [bold]uv run llm model list[/bold] to see available models.")
        raise typer.Exit(1)

    if not _llm_server_systemctl("start"):
        console.print("[red]Failed to start llm-server.[/red]")
        console.print("  Check: [bold]systemctl status llm-server[/bold]")
        raise typer.Exit(1)

    pid = _unit_main_pid()
    suffix = f" (PID {pid})" if pid else ""
    console.print(f"[green]Started[/green] llama-server{suffix}")
    console.print(f"  Model  : {cfg.models.active}")
    console.print(f"  Port   : {cfg.server.port}")
    console.print(f"  Layers : {cfg.server.n_gpu_layers} (Vulkan iGPU)")
    if cfg.server.extra_args:
        console.print(f"  Extra  : {' '.join(cfg.server.extra_args)}")
    console.print("  Logs   : journalctl -u llm-server  [dim](uv run llm server logs -f)[/dim]")

    if wait > 0:
        _wait_until_ready(cfg, wait)

    if cfg.server.monitor and pid:
        _start_monitor(cfg, pid)

    _nginx_ensure_running()


def _wait_until_ready(cfg: object, wait: int) -> None:
    """Poll /health until the server answers or *wait* seconds elapse."""
    from llm.config import Settings  # noqa: PLC0415

    assert isinstance(cfg, Settings)
    console.print(f"Waiting up to {wait}s for server to be ready...", end="")
    if http.wait_until_healthy(f"{cfg.internal_url}/health", timeout=wait, probe_timeout=1):
        console.print(" [green]ready[/green]")
    else:
        console.print(" [yellow]timeout (server may still be loading)[/yellow]")


@app.command("stop")
def stop() -> None:
    """Stop llama-server and nginx."""
    proc.ensure_sudo()
    cfg = load_config()
    pid = _server_pid(cfg.server.port)
    nginx_active = _nginx_is_active()

    if pid is None and not nginx_active:
        console.print("[yellow]Server is not running[/yellow]")
        raise typer.Exit(1)

    if pid is not None:
        if _llm_server_systemctl("stop"):
            console.print(f"[green]Stopped[/green] llama-server (PID {pid})")
        else:
            console.print("[yellow]llm-server[/yellow]   failed to stop - check: systemctl status llm-server")
        # systemctl reports success even when the unit was already inactive, so
        # confirm nothing is still listening. A survivor was started outside the
        # unit, which systemd has no way to stop.
        survivor = _server_pid(cfg.server.port)
        if survivor is not None:
            console.print(
                f"[yellow]⚠  llama-server (PID {survivor}) is still running.[/yellow]\n"
                f"   It was not started by systemd - stop it with: [bold]kill {survivor}[/bold]"
            )

    if nginx_active:
        if _nginx_stop():
            console.print("[green]Stopped[/green] nginx")
        else:
            console.print("[yellow]nginx[/yellow]       failed to stop - check: sudo systemctl status nginx")

    _stop_monitor()


@app.command("restart")
def restart() -> None:
    """Restart llama-server and make sure nginx is running."""
    proc.ensure_sudo()
    _warn_if_stale()
    cfg = load_config()
    _require_unit_installed()

    _stop_monitor()
    if not _llm_server_systemctl("restart"):
        console.print("[red]Failed to restart llm-server.[/red]")
        console.print("  Check: [bold]systemctl status llm-server[/bold]")
        raise typer.Exit(1)

    pid = _unit_main_pid()
    suffix = f" (PID {pid})" if pid else ""
    console.print(f"[green]Restarted[/green] llama-server{suffix}")

    _wait_until_ready(cfg, 5)

    if cfg.server.monitor and pid:
        _start_monitor(cfg, pid)

    _nginx_ensure_running()


@app.command("status")
def status() -> None:
    """Show whether llama-server and nginx are running."""
    _warn_if_stale()
    cfg = load_config()

    if not cfg.has_local_server:
        console.print("[dim]No local server configured - client-only mode.[/dim]")
        console.print(f"  Connecting to: [cyan]{cfg.client_url}[/cyan]")
        if _nginx_is_active():
            console.print("[green]● nginx[/green]         active")
        return

    pid = _server_pid(cfg.server.port)
    if pid:
        from llm.models import KNOWN_MODELS  # noqa: PLC0415

        # Resolve: check config catalog first, then KNOWN_MODELS
        entry = cfg.models.by_alias(cfg.models.active)
        if entry is None:
            entry = cfg.models.by_filename(cfg.models.active)
        if entry is None and cfg.models.has_catalog is False:
            entry = next((m for m in KNOWN_MODELS if m.filename == cfg.models.active), None)
        display = entry.alias if entry else cfg.models.active
        ready = _server_is_ready(cfg.server.port)
        status_icon = "[green]●[/green]" if ready else "[yellow]●[/yellow] loading"
        uptime = _process_uptime_seconds(pid)
        console.print(f"{status_icon} llama-server  PID {pid} port {cfg.server.port}")
        console.print(f"  Model  : {display}  [dim]({cfg.models.active})[/dim]")
        if uptime is not None:
            console.print(f"  Uptime : {fmt.duration(uptime)}")
        console.print(f"  Layers : {cfg.server.n_gpu_layers}")
        if cfg.server.extra_args:
            console.print(f"  Extra  : {' '.join(cfg.server.extra_args)}")
        # Show active build profile if configured
        active_profile = cfg.build.active_profile
        if active_profile:
            profile_name = cfg.server.profile or active_profile.name
            console.print(f"  Profile: {profile_name}")
        console.print("  Logs   : journalctl -u llm-server  [dim](uv run llm server logs -f)[/dim]")
    elif not _llm_server_unit_installed():
        console.print("[red]● llama-server[/red] no systemd unit installed")
        console.print("  Run [bold]uv run llm server apply[/bold] to install it.")
    else:
        console.print("[red]● llama-server[/red] stopped")
        console.print("  Run [bold]uv run llm server start[/bold] to start.")

    # Memory monitor status
    monitor_pid = _read_monitor_pid()
    if monitor_pid:
        console.print(f"[green]● monitor[/green]       PID {monitor_pid} → logs/memory-monitor.csv")
    else:
        console.print("[dim]● monitor[/dim]       stopped")

    if _nginx_is_active():
        console.print("[green]● nginx[/green]         active")
    else:
        console.print("[red]● nginx[/red]         stopped")


@app.command("logs")
def logs(
    lines: Annotated[int, typer.Option("-n", help="Number of lines to show.")] = 50,
    follow: Annotated[bool, typer.Option("-f", "--follow", help="Follow log output.")] = False,
) -> None:
    """Show server logs from the systemd journal."""
    cmd = ["journalctl", "-u", SERVICE_UNIT, "-n", str(lines), "--no-pager"]
    if follow:
        cmd.append("-f")
    subprocess.run(cmd, check=False)


@app.command("memory")
def memory(
    last: Annotated[int, typer.Option("-n", help="Number of samples to show.")] = 10,
    oom: Annotated[bool, typer.Option("--oom", help="Scan the kernel log for OOM kills.")] = False,
) -> None:
    """Show recent memory samples recorded by the monitor."""
    from llm.monitor import MONITOR_CSV, read_recent_rows

    if oom:
        _check_oom()
        return

    if not MONITOR_CSV.exists():
        console.print("[yellow]No memory samples yet.[/yellow]")
        console.print("  Start the server to enable the monitor: [bold]uv run llm server start[/bold]")
        raise typer.Exit(1)

    rows = read_recent_rows(MONITOR_CSV, last)
    if not rows:
        console.print("[yellow]Memory sample file is empty.[/yellow]")
        raise typer.Exit(1)

    table = Table(title="Memory monitor — logs/memory-monitor.csv")
    table.add_column("Time (UTC)", no_wrap=True)
    table.add_column("Event")
    table.add_column("MemAvail", justify="right")
    table.add_column("RSS", justify="right")
    table.add_column("GPU GTT", justify="right")
    for row in rows:
        table.add_row(
            row.get("timestamp", ""),
            row.get("event", ""),
            fmt.mib(row.get("mem_avail_kb", ""), "kb"),
            fmt.mib(row.get("rss_kb", ""), "kb"),
            fmt.mib(row.get("gpu_gtt_used_mb", ""), "mb"),
        )
    console.print(table)


@app.command("apply")
def apply() -> None:
    """Render nginx/systemd templates from config.toml and install them.

    Re-runs template rendering and installs the resulting files to
    /etc/nginx/sites-available/llm and /etc/systemd/system/llm-server.service.
    Reloads nginx if it is already running.
    """
    from llm.config import apply_server_configs  # noqa: PLC0415

    cfg = load_config()
    proc.ensure_sudo()
    apply_server_configs(cfg, _project_root())
