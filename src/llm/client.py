"""Client setup helper: print connection instructions for the LXD container."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.syntax import Syntax

from llm.config import find_config, load_config

app = typer.Typer(help="Print client setup instructions.")
console = Console()

_PLACEHOLDER_URL = "https://<SERVER_LAN_IP>:8443/v1"
_PLACEHOLDER_KEY = "<your-api-key>"
_PLACEHOLDER_MODEL = "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"


@app.command("setup")
def setup(
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="Override server base URL (e.g. https://192.168.1.x:8443/v1)."),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="Override API key."),
    ] = None,
) -> None:
    """Print export commands for configuring opencode in the LXD container.

    Run this on the SERVER to generate instructions, then paste the output
    into the LXD container. The client itself does not need this CLI or a
    config.toml — only opencode and the two env vars below.
    """
    # Try loading config; fall back to placeholders if not on the server
    config_path = find_config()
    if config_path.exists():
        cfg = load_config()
        resolved_base_url = base_url or f"https://{cfg.proxy.lan_ip}:{cfg.proxy.port}/v1"
        resolved_key = api_key or cfg.proxy.api_key
        resolved_model = cfg.models.active
        if resolved_key == "change-me-generate-a-strong-random-key":
            console.print(
                "[yellow]Warning:[/yellow] api_key is still the default placeholder.\n"
                "Generate a real key:\n"
                '  python -c "import secrets; print(secrets.token_hex(32))"\n'
                "Then update [proxy] api_key in config.toml.\n"
            )
    else:
        console.print(
            "[yellow]Note:[/yellow] No config.toml found — showing placeholder values.\n"
            "Run this command on the server (where config.toml lives) to see real values,\n"
            "or pass them directly with --base-url and --api-key flags.\n"
        )
        resolved_base_url = base_url or _PLACEHOLDER_URL
        resolved_key = api_key or _PLACEHOLDER_KEY
        resolved_model = _PLACEHOLDER_MODEL

    health_url = resolved_base_url.removesuffix("/v1")

    console.print("[bold]1. Environment variables[/bold]  (add to ~/.bashrc in the container)\n")
    export_block = f'export OPENAI_BASE_URL="{resolved_base_url}"\nexport OPENAI_API_KEY="{resolved_key}"\n'
    console.print(Syntax(export_block, "bash", theme="monokai"))

    console.print("[bold]2. opencode config[/bold]  (~/.config/opencode/opencode.json)\n")
    # Model ID within the provider (strip .gguf for a cleaner model ID)
    model_id = resolved_model.removesuffix(".gguf")
    opencode_block = (
        "{\n"
        '  "$schema": "https://opencode.ai/config.json",\n'
        '  "provider": {\n'
        '    "local-llm": {\n'
        '      "api": "openai",\n'
        '      "name": "Local LLM (llama-server)",\n'
        '      "options": {\n'
        f'        "apiKey": "{resolved_key}",\n'
        f'        "baseURL": "{resolved_base_url}"\n'
        "      },\n"
        '      "models": {\n'
        f'        "{model_id}": {{\n'
        f'          "name": "{model_id}"\n'
        "        }\n"
        "      }\n"
        "    }\n"
        "  },\n"
        f'  "model": "local-llm/{model_id}"\n'
        "}"
    )
    console.print(Syntax(opencode_block, "jsonc", theme="monokai"))

    console.print("[bold]3. Test connectivity from the container:[/bold]\n")
    curl_cmd = f'curl -sk {health_url}/health \\\n  -H "Authorization: Bearer {resolved_key}"'
    console.print(Syntax(curl_cmd, "bash", theme="monokai"))

    console.print(
        "\n[dim]Tip: pass -k to curl (or --insecure) to skip self-signed cert verification on the LAN.[/dim]"
    )
