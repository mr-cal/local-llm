"""Discourse CLI commands."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.markup import escape

from llm.core.console import console
from llm.discourse.client import DiscourseClient, get_default_summary_window
from llm.settings import find_config, load_config

app = typer.Typer(help="Discourse forum integration & MCP commands.", no_args_is_help=True)


def _get_discourse_creds() -> tuple[str, str]:
    """Retrieve Discourse URL and session cookie from config or env."""
    cfg_path = find_config()
    if cfg_path.exists():
        cfg = load_config()
        if cfg.hermes.discourse_url and cfg.hermes.discourse_session_cookie:
            return cfg.hermes.discourse_url, cfg.hermes.discourse_session_cookie

    import os

    url = os.getenv("DISCOURSE_URL", "https://canonical.discourse.com")
    cookie = os.getenv("DISCOURSE_SESSION_COOKIE", "")
    return url, cookie


@app.command("check-auth")
def check_auth() -> None:
    """Check Discourse session cookie authentication."""
    url, cookie = _get_discourse_creds()
    if not cookie:
        console.print(
            "[yellow]⚠[/yellow] No Discourse session cookie found in config.toml [hermes] "
            "or DISCOURSE_SESSION_COOKIE."
        )
        raise typer.Exit(1)

    console.print(f"Checking auth for [bold]{url}[/bold]...")
    client = DiscourseClient(base_url=url, session_cookie=cookie)
    status = client.check_auth()

    if status.get("authenticated"):
        console.print(
            f"[green]✓[/green] Authenticated as [bold]{status.get('username')}[/bold] ({status.get('name')})"
        )
        if status.get("admin"):
            console.print("  [cyan]Role:[/cyan] Admin")
    else:
        console.print(f"[red]✗[/red] Authentication failed: {status.get('error', 'Unknown error')}")
        raise typer.Exit(1)


@app.command("summary-context")
def summary_context(
    hours: Annotated[
        int | None,
        typer.Option("--hours", help="Lookback window in hours (default: 72 on Monday, 24 otherwise)."),
    ] = None,
) -> None:
    """Fetch recent topics and posts formatted for daily summarization."""
    url, cookie = _get_discourse_creds()
    if not cookie:
        console.print("[yellow]⚠[/yellow] No Discourse session cookie configured.")
        raise typer.Exit(1)

    import datetime as dt

    if hours is not None:
        since = dt.datetime.now(dt.UTC) - dt.timedelta(hours=hours)
        window_hours = hours
    else:
        since, window_hours = get_default_summary_window()

    console.print(f"Fetching posts since {since.isoformat()} ({window_hours} hours)...")
    client = DiscourseClient(base_url=url, session_cookie=cookie)
    topics = client.get_posts_since(since=since)

    console.print(f"[green]✓[/green] Found [bold]{len(topics)}[/bold] active topic(s):")
    for t in topics:
        console.print(f"  • [bold]{escape(t['title'])}[/bold] ({len(t.get('recent_posts', []))} new post(s))")
        console.print(f"    [dim]{t['url']}[/dim]")


@app.command("serve-mcp")
def serve_mcp() -> None:
    """Run the Discourse MCP server (stdio transport)."""
    from llm.discourse.server import create_discourse_mcp_server

    url, cookie = _get_discourse_creds()
    server = create_discourse_mcp_server(url=url, session_cookie=cookie)
    server.run(transport="stdio")
