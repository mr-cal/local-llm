"""CLI commands for the hermes VM: setup, refresh, status."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from llm.config import find_config, load_config

app = typer.Typer(help="Hermes agent VM management.", no_args_is_help=True)
console = Console()


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
    from llm.hermes_vm import HermesVmManager  # noqa: PLC0415

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
    from llm.hermes_vm import HermesVmManager  # noqa: PLC0415
    from llm.lxd import container_exists  # noqa: PLC0415

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


@app.command("status")
def status() -> None:
    """Show VM and gateway service status for the hermes VM."""
    from llm.hermes_vm import HermesVmManager  # noqa: PLC0415
    from llm.lxd import container_exists  # noqa: PLC0415

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
