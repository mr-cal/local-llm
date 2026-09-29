"""CLI commands for the hermes VM: setup, refresh, status."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.markup import escape

from llm.core import fmt
from llm.core.console import console
from llm.settings import find_config, load_config

app = typer.Typer(help="Hermes agent VM management.", no_args_is_help=True)


@app.command("setup")
def setup(
    recreate: Annotated[
        bool,
        typer.Option("--recreate", help="Delete and recreate the VM if it already exists."),
    ] = False,
) -> None:
    """Create and configure the hermes LXD VM.

    Installs the Nous Research Hermes agent and configures it using the
    [hermes] section of config.toml (OpenRouter key, Telegram token, etc.).
    """
    from llm.provision.hermes_vm import HermesVmManager  # noqa: PLC0415

    cfg_path = find_config()
    if not cfg_path.exists():
        console.print(
            "[red]ERROR:[/red] config.toml not found.\n  Run [bold]uv run llm config init[/bold] first."
        )
        raise typer.Exit(1)

    cfg = load_config()

    try:
        mgr = HermesVmManager()
        mgr.create_and_setup(cfg.hermes, recreate=recreate)
    except RuntimeError as e:
        console.print(f"[red]ERROR:[/red] {escape(str(e))}")
        raise typer.Exit(1) from None


@app.command("refresh")
def refresh() -> None:
    """Update packages and Hermes agent, and re-apply credentials."""
    from llm.provision.exec import container_exists  # noqa: PLC0415
    from llm.provision.hermes_vm import HermesVmManager  # noqa: PLC0415

    cfg_path = find_config()
    if not cfg_path.exists():
        console.print("[red]ERROR:[/red] config.toml not found.")
        raise typer.Exit(1)

    cfg = load_config()
    mgr = HermesVmManager()

    if not container_exists(mgr.container):
        console.print(
            f"[red]ERROR:[/red] VM '{mgr.container}' does not exist.\n"
            "  Run [bold]uv run llm hermes setup[/bold] first."
        )
        raise typer.Exit(1)

    try:
        mgr.refresh(cfg.hermes)
    except RuntimeError as e:
        console.print(f"[red]ERROR:[/red] {escape(str(e))}")
        raise typer.Exit(1) from None


def _format_credentials(cred: str) -> tuple[str, str]:
    """Return (label, color) for the credentials status string."""
    if cred == "True":
        return "valid", "green"
    if cred == "False":
        return "invalid", "yellow"
    return "unknown", "red"


def _format_local_llm(ok: str) -> tuple[str, str]:
    """Return (label, color) for the local LLM connectivity status string."""
    if ok == "True":
        return "connected", "green"
    if ok == "False":
        return "unreachable", "yellow"
    return "unknown", "red"


@app.command("status")
def status() -> None:
    """Show VM and gateway service status for the hermes VM."""
    from llm.provision.exec import container_exists  # noqa: PLC0415
    from llm.provision.hermes_vm import HermesVmManager  # noqa: PLC0415

    mgr = HermesVmManager()

    if not container_exists(mgr.container):
        console.print("  [yellow]●[/yellow] hermes  (VM does not exist)")
        console.print("\n  Create it with: [bold]uv run llm hermes setup[/bold]")
        return

    s = mgr.get_status()

    vm_color = "green" if s["vm"] == "Running" else "yellow"
    console.print(f"  [{vm_color}]●[/{vm_color}] VM:       {s['vm']}  (hermes)")

    gw_color = "green" if s["gateway"] == "active" else "yellow"
    console.print(f"  [{gw_color}]●[/{gw_color}] Gateway:  {s['gateway']}  (hermes-gateway.service)")

    ver = s.get("version", "unknown")
    console.print(f"  Version:     {ver}")

    uptime_secs = int(s.get("uptime", "0"))
    console.print(f"  Uptime:      {fmt.duration(uptime_secs)}")

    # The "Credentials" check probes different things depending on the
    # configured provider: OpenRouter API key validity, or reachability of
    # the local llama-server through the hermes VM's network setup.
    provider = s.get("provider", "unknown")
    ok = s.get("credentials_ok", "unknown")
    if provider in ("openai", "local"):
        label, color = _format_local_llm(ok)
        console.print(f"  [{color}]●[/{color}] Local LLM:   {label}")
    else:
        label, color = _format_credentials(ok)
        console.print(f"  [{color}]●[/{color}] Credentials: {label}")
