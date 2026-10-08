"""CLI commands for Jotter container management."""

from __future__ import annotations

from typing import Annotated

import typer

from llm.core.console import console
from llm.provision.jotter_vm import JotterVmManager
from llm.settings import load_config

app = typer.Typer(help="Jotter container provisioning & management.", no_args_is_help=True)


@app.command("setup")
def setup(
    recreate: Annotated[
        bool,
        typer.Option("--recreate", help="Delete and recreate container if it exists."),
    ] = False,
) -> None:
    """Create and configure the Jotter LXD container."""
    cfg = load_config()
    mgr = JotterVmManager(timezone=cfg.jotter.effective_timezone(cfg))
    mgr.create_and_setup(cfg, recreate=recreate)


@app.command("status")
def status() -> None:
    """Show Jotter container status."""
    cfg = load_config()
    mgr = JotterVmManager(timezone=cfg.jotter.effective_timezone(cfg))
    info = mgr.status()
    st = info.get("status")
    if st == "running":
        console.print("[green]● jotter[/green] container is running")
    elif st == "stopped":
        console.print("[yellow]● jotter[/yellow] container is stopped")
    else:
        console.print("[dim]● jotter[/dim] container not created. Run [bold]llm jotter setup[/bold]")
