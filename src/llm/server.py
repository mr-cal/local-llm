"""Server management: start, stop, restart, status, logs."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from llm.config import load_config

app = typer.Typer(help="Manage the llama-server process.")
console = Console()

# Runtime files live alongside the config in the project directory.
# Both are gitignored.
_PID_FILE = Path(".server.pid")
_LOG_FILE = Path(".server.log")


def _pid_file() -> Path:
    """Resolve PID file path relative to CWD (project root)."""
    return _PID_FILE


def _log_file() -> Path:
    return _LOG_FILE


def _read_pid() -> int | None:
    """Return running server PID, or None if not running."""
    pf = _pid_file()
    if not pf.exists():
        return None
    try:
        pid = int(pf.read_text().strip())
        os.kill(pid, 0)  # signal 0 = existence check, raises if not running
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        pf.unlink(missing_ok=True)
        return None


@app.command("start")
def start(
    wait: Annotated[int, typer.Option("--wait", help="Seconds to wait for server to be ready.")] = 5,
) -> None:
    """Start llama-server using settings from config.toml."""
    existing = _read_pid()
    if existing:
        console.print(f"[yellow]Server already running[/yellow] (PID {existing})")
        raise typer.Exit(1)

    cfg = load_config()

    if not cfg.model_path.exists():
        console.print(f"[red]Model file not found:[/red] {cfg.model_path}")
        console.print("Run [bold]uv run llm model list[/bold] to see available models.")
        raise typer.Exit(1)

    cmd: list[str] = [
        cfg.server.llama_server_bin,
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
        bin_path = cfg.server.llama_server_bin
        console.print(f"[red]Binary not found:[/red] {bin_path}")
        console.print(
            "\nBuild llama.cpp and install the binary:\n"
            "  [bold]cmake --build llama.cpp/build -j$(nproc)[/bold]\n"
            "  [bold]cp llama.cpp/build/bin/llama-server ~/.local/bin/[/bold]\n"
            "\nThen update [bold]llama_server_bin[/bold] in config.toml if needed:\n"
            '  llama_server_bin = "~/.local/bin/llama-server"'
        )
        raise typer.Exit(1) from None

    _pid_file().write_text(str(proc.pid))
    console.print(f"[green]Started[/green] llama-server (PID {proc.pid})")
    console.print(f"  Model  : {cfg.models.active}")
    console.print(f"  Port   : {cfg.server.port}")
    console.print(f"  Layers : {cfg.server.n_gpu_layers} (Vulkan iGPU)")
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
                return
            except Exception:
                pass
        console.print(" [yellow]timeout (server may still be loading)[/yellow]")


@app.command("stop")
def stop() -> None:
    """Stop the running llama-server."""
    pid = _read_pid()
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


@app.command("restart")
def restart() -> None:
    """Restart llama-server (stop then start)."""
    pid = _read_pid()
    if pid:
        stop()
    start()


@app.command("status")
def status() -> None:
    """Show whether llama-server is running."""
    pid = _read_pid()
    if pid:
        cfg = load_config()
        console.print(f"[green]● Running[/green]  PID {pid}  port {cfg.server.port}")
        console.print(f"  Model : {cfg.models.active}")
        console.print(f"  Logs  : {_log_file().resolve()}")
    else:
        console.print("[red]● Stopped[/red]")
        console.print("Run [bold]uv run llm server start[/bold] to start.")


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
