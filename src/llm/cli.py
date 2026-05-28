"""Main CLI entry point — all subcommand groups are registered here."""

from __future__ import annotations

import typer

from llm import benchmark, build, client, config, models, server

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


def main() -> None:
    app()
