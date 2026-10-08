"""Main CLI entry point. All subcommand groups are registered here."""

from __future__ import annotations

import typer

from llm import benchmark, build, client, config, models, server
from llm.core.console import console
from llm.core.errors import LlmError
from llm.digest.cli import app as digest_app
from llm.discourse.cli import app as discourse_app
from llm.jotter_cli import app as jotter_app

app = typer.Typer(
    name="llm",
    help="Local LLM server management (llama.cpp + nginx + opencode).",
    no_args_is_help=True,
)

app.add_typer(server.app, name="server")
app.add_typer(models.app, name="model")
app.add_typer(benchmark.app, name="benchmark")
app.add_typer(build.app, name="build")
app.add_typer(client.app, name="client")
app.add_typer(config.app, name="config")
app.add_typer(discourse_app, name="discourse")
app.add_typer(digest_app, name="digest")
app.add_typer(jotter_app, name="jotter")


@app.command("tui")
def tui() -> None:
    """Launch the interactive Jotter Textual TUI."""
    from llm.tui.app import run_tui  # noqa: PLC0415

    run_tui()


def main() -> None:
    """Run the CLI, reporting expected failures without a traceback."""
    try:
        app()
    except LlmError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from None
