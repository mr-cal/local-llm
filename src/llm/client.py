"""Client setup helper: print connection instructions for the LXD container."""

from __future__ import annotations

from pathlib import Path
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
_PLACEHOLDER_IP = "<SERVER_LAN_IP>"


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
    cert_pem: str | None = None
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
        cert_file = Path(cfg.proxy.cert_path)
        if cert_file.exists():
            cert_pem = cert_file.read_text().strip()
        else:
            console.print(
                f"[yellow]Cert not found at {cert_file}[/yellow] — generate it first:\n"
                "  [bold]uv run llm config gencert[/bold]\n"
                "(Sets the required SubjectAltName — plain CN= certs are rejected by Node.js)\n"
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
    model_id = resolved_model.removesuffix(".gguf")

    # ── Step 1: TLS cert ─────────────────────────────────────────────────────
    console.print("[bold]Step 1 — Install TLS certificate[/bold]  (in the LXD container)\n")
    if cert_pem:
        console.print(
            "Save the following certificate to a file on the container, "
            "then trust it for Node.js:\n"
        )
        console.print(Syntax(cert_pem, "text", theme="monokai"))
        console.print()
        cert_install = (
            "# Save cert (paste the PEM block above into this file)\n"
            f"mkdir -p ~/.config/opencode\n"
            f"cat > ~/.config/opencode/local-llm.pem << 'EOF'\n"
            f"{cert_pem}\n"
            "EOF\n\n"
            "# Tell Node.js (opencode) to trust it — add to ~/.bashrc\n"
            "echo 'export NODE_EXTRA_CA_CERTS=\"$HOME/.config/opencode/local-llm.pem\"' >> ~/.bashrc\n"
            "export NODE_EXTRA_CA_CERTS=\"$HOME/.config/opencode/local-llm.pem\""
        )
        console.print(Syntax(cert_install, "bash", theme="monokai"))
    else:
        console.print(
            "Once the cert exists at the configured path, re-run this command to get\n"
            "the cert content and installation instructions.\n\n"
            "Alternatively, copy the cert manually from the server and set:\n"
            '  [bold]export NODE_EXTRA_CA_CERTS="/path/to/local-llm.pem"[/bold]'
        )
    console.print()

    # ── Step 2: env vars ─────────────────────────────────────────────────────
    console.print("[bold]Step 2 — Environment variables[/bold]  (add to ~/.bashrc)\n")
    export_block = (
        f'export OPENAI_BASE_URL="{resolved_base_url}"\n'
        f'export OPENAI_API_KEY="{resolved_key}"\n'
        f'export NODE_EXTRA_CA_CERTS="$HOME/.config/opencode/local-llm.pem"'
    )
    console.print(Syntax(export_block, "bash", theme="monokai"))
    console.print()

    # ── Step 3: opencode config ───────────────────────────────────────────────
    console.print("[bold]Step 3 — opencode config[/bold]  (~/.config/opencode/opencode.json)\n")
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
    console.print()

    # ── Step 4: verify ────────────────────────────────────────────────────────
    console.print("[bold]Step 4 — Verify connectivity[/bold]\n")
    curl_cmd = (
        f"curl -s --cacert ~/.config/opencode/local-llm.pem \\\n"
        f"  {health_url}/health \\\n"
        f'  -H "Authorization: Bearer {resolved_key}"'
    )
    console.print(Syntax(curl_cmd, "bash", theme="monokai"))
