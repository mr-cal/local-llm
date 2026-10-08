"""CLI commands for digest generation and automation."""

from __future__ import annotations

from typing import Annotated

import typer

from llm.core.console import console
from llm.digest.engine import DigestEngine
from llm.digest.schedule import Slot
from llm.settings import load_config

app = typer.Typer(help="Digest generation commands for background automation.", no_args_is_help=True)


@app.command("run")
def run(
    slot: Annotated[
        str,
        typer.Option("--slot", help="Digest slot: 'morning' or 'afternoon'."),
    ] = "morning",
) -> None:
    """Run headless digest generation and save markdown output."""
    try:
        target_slot = Slot(slot.lower())
    except ValueError:
        console.print(f"[red]Error:[/red] Invalid slot: {slot!r}. Choose 'morning' or 'afternoon'.")
        raise typer.Exit(1) from None

    cfg = load_config()
    engine = DigestEngine(cfg)

    console.print(f"[bold cyan]Running {target_slot.value} digest...[/bold cyan]")
    try:
        summary = engine.run_digest(slot=target_slot)
        console.print(f"[green]✓[/green] Digest completed successfully ({len(summary)} characters).")
    except Exception as exc:
        console.print(f"[red]Error generating digest:[/red] {exc}")
        raise typer.Exit(1) from None
