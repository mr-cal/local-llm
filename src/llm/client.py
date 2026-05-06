"""Client setup helper: print connection instructions for the LXD container."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.syntax import Syntax

from llm.config import load_config

app = typer.Typer(help="Print client setup instructions.")
console = Console()


@app.command("setup")
def setup() -> None:
    """Print export commands for configuring opencode in the LXD container."""
    cfg = load_config()

    if cfg.proxy.api_key == "change-me-generate-a-strong-random-key":
        console.print(
            "[yellow]Warning:[/yellow] api_key is still the default placeholder.\n"
            "Generate a real key:\n"
            '  python -c "import secrets; print(secrets.token_hex(32))"\n'
            "Then update [proxy] api_key in config.toml.\n"
        )

    base_url = f"https://{cfg.proxy.lan_ip}:{cfg.proxy.port}/v1"

    console.print("[bold]opencode / any OpenAI-compatible client[/bold]\n")
    console.print("Add these to ~/.bashrc (or the LXD container profile):\n")

    export_block = f'export OPENAI_BASE_URL="{base_url}"\nexport OPENAI_API_KEY="{cfg.proxy.api_key}"\n'
    console.print(Syntax(export_block, "bash", theme="monokai"))

    console.print("[bold]opencode config (alternative)[/bold]\n")
    opencode_block = (
        "[providers.local-llm]\n"
        f'  api_key = "{cfg.proxy.api_key}"\n'
        f'  base_url = "{base_url}"\n'
        f'  model = "{cfg.models.active}"\n'
    )
    console.print(Syntax(opencode_block, "toml", theme="monokai"))

    console.print("[bold]Test connectivity from the container:[/bold]\n")
    curl_cmd = (
        f'curl -sk {base_url.replace("/v1", "")}/health \\\n  -H "Authorization: Bearer {cfg.proxy.api_key}"'
    )
    console.print(Syntax(curl_cmd, "bash", theme="monokai"))

    console.print(
        "\n[dim]Note: the self-signed TLS cert will cause curl to warn unless you "
        "pass -k or install the cert. See README.md → TLS section.[/dim]"
    )
