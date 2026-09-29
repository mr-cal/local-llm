"""The `llm config` commands: interactive init, and printing the active config."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import typer
from pydantic import BaseModel
from rich.syntax import Syntax

from llm.core.console import console
from llm.render.client_configs import (
    _build_opencode_config,
    _build_pi_config,
    _validate_opencode_config,
)
from llm.settings import (
    CONFIG_FILENAME,
    find_config,
    load_config,
    write_config_toml,
)

app = typer.Typer(help="Manage configuration and render templates.", no_args_is_help=True)


def _prompt_init(prompt: str, default: str = "") -> str:
    """Prompt the user for input during config init, showing a default.

    If no default is provided the user must enter a non-empty value.
    """
    suffix = f" [{default}]" if default else ""
    while True:
        result = input(f"{prompt}{suffix}: ").strip()
        if result:
            return result
        if default:
            return default
        console.print("  [red]Please enter a non-empty value.[/red]")


@app.command("init")
def config_init() -> None:
    """Create a minimal client-only config.toml interactively.

    Use this on a machine that connects to a remote server but does not
    run llama-server itself.  For a server machine, use instead:

        uv run llm server setup
    """

    config_path = find_config()

    console.print("\n[bold cyan]═══ local-llm client init ═══[/bold cyan]\n")

    if config_path.exists():
        console.print(f"[yellow]Config already exists:[/yellow] {config_path}")
        answer = _prompt_init("  Overwrite?", "n")
        if answer.lower() not in ("y", "yes"):
            console.print("Aborted.")
            raise typer.Exit(0)
    else:
        config_path = Path.cwd() / CONFIG_FILENAME

    # ── Step 1: Server connection ─────────────────────────────────────────
    console.print("[bold]Step 1/3[/bold] - Server connection")
    server_url = _prompt_init("  Server URL", "https://192.168.1.x:8443/v1")

    # ── Step 2: Auth ──────────────────────────────────────────────────────
    console.print("\n[bold]Step 2/3[/bold] - Authentication")
    console.print("  [dim]Find the API key in config.toml on the server (auth.api_key).[/dim]")
    api_key = _prompt_init("  API key")

    # ── Step 3: TLS certificate ───────────────────────────────────────────
    console.print("\n[bold]Step 3/3[/bold] - TLS certificate")
    default_cert = "~/.config/local-llm/cert.pem"
    cert_path = _prompt_init("  Local cert path", default_cert)
    cert_expanded = Path(cert_path).expanduser()

    if not cert_expanded.exists():
        console.print(f"\n  [yellow]Cert not found at {cert_expanded}[/yellow]")
        fetch = _prompt_init("  Fetch from server via scp? (y/n)", "y")
        if fetch.lower() in ("y", "yes"):
            server_host = _prompt_init("  Server SSH host (e.g. user@192.168.1.64)")
            remote_cert = _prompt_init("  Remote cert path", "/etc/ssl/local-llm/cert.pem")
            cert_expanded.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                ["scp", f"{server_host}:{remote_cert}", str(cert_expanded)],
                check=False,
            )
            if result.returncode == 0:
                console.print(f"  [green]✓[/green] Cert copied to {cert_expanded}")
            else:
                console.print(
                    f"  [red]✗[/red] scp failed - copy it manually:\n"
                    f"    scp {server_host}:{remote_cert} {cert_expanded}"
                )

    # ── Build and write config ────────────────────────────────────────────
    match = re.match(r"https?://([^:/]+):(\d+)", server_url)
    lan_ip = match.group(1) if match else "192.168.1.100"
    proxy_port = int(match.group(2)) if match else 8443

    cfg_dict = {
        "server": {"enabled": False},
        "proxy": {
            "enabled": False,
            "lan_ip": lan_ip,
            "port": proxy_port,
            "cert_path": str(cert_expanded),
        },
        "client": {
            "enabled": True,
            "server_url": server_url,
            "cert_path": str(cert_expanded),
        },
        "auth": {"api_key": api_key},
    }

    write_config_toml(cfg_dict, config_path)
    console.print(f"\n  [green]✓[/green] Config written: {config_path}")
    console.print(
        "\n[bold green]✓ Config created![/bold green]\n"
        "\n  Next, set up client tools (opencode, pi, shell env):"
        "\n    [bold]uv run llm client setup[/bold]"
    )


def _is_secret_field(field: object) -> bool:
    """True when a pydantic field was tagged with the SECRET marker."""
    extra = getattr(field, "json_schema_extra", None)
    return isinstance(extra, dict) and bool(extra.get("secret"))


def mask_secrets(model: BaseModel) -> dict:  # type: ignore[type-arg]
    """Dump *model* with every SECRET-tagged field replaced by a placeholder.

    Walks nested models so a credential added anywhere in the settings tree is
    masked automatically. Empty values are left empty rather than masked, so
    "unset" stays visually distinct from "set but hidden".
    """
    out: dict[str, object] = {}
    for name, field in type(model).model_fields.items():
        value = getattr(model, name)
        key = field.alias or name
        if isinstance(value, BaseModel):
            out[key] = mask_secrets(value)
        elif isinstance(value, list):
            out[key] = [mask_secrets(v) if isinstance(v, BaseModel) else v for v in value]
        elif _is_secret_field(field):
            out[key] = "***" if value else ""
        else:
            out[key] = value
    return out


def _mask_api_keys(obj: object) -> object:
    """Recursively mask apiKey/api_key values in a generated client config.

    The opencode and pi configs embed the live API key. `config show` prints
    them for inspection, so mask the credential without disturbing the rest of
    the structure (the unmasked config is what actually gets written to disk).
    """
    if isinstance(obj, dict):
        return {k: ("***" if k in ("apiKey", "api_key") and v else _mask_api_keys(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_mask_api_keys(v) for v in obj]
    return obj


@app.command("show")
def config_show() -> None:
    """Print current configuration (credentials are masked) and opencode config."""
    import json  # noqa: PLC0415

    cfg = load_config()
    masked = mask_secrets(cfg)

    def _to_toml_ish(d: dict, indent: int = 0) -> str:  # type: ignore[type-arg]
        lines_out: list[str] = []
        prefix = "  " * indent
        for k, v in d.items():
            if isinstance(v, dict):
                lines_out.append(f"\n{prefix}[{k}]")
                lines_out.append(_to_toml_ish(v, indent + 1))
            elif isinstance(v, list):
                lines_out.append(f"{prefix}{k} = {v!r}")
            elif isinstance(v, str):
                lines_out.append(f'{prefix}{k} = "{v}"')
            else:
                lines_out.append(f"{prefix}{k} = {v}")
        return "\n".join(lines_out)

    output = _to_toml_ish(masked)
    console.print(Syntax(output, "toml", theme="monokai"))

    # Show model catalog if present
    if cfg.models.has_catalog:
        console.print(f"\n[bold]Model catalog ({len(cfg.models.entries)} models)[/bold]")
        for m in cfg.models.entries:
            active_marker = " ▶" if m.alias == cfg.models.active else "  "
            cost_str = ""
            if not m.cost.is_zero():
                cost_str = f"  cost: {m.cost.input:.4g}/{m.cost.output:.4g}"
            console.print(f"  {active_marker} {m.alias}  {m.size:>8}  {m.description}{cost_str}")
        console.print("\n[dim]▶ = active[/dim]")

    opencode_cfg = _build_opencode_config(cfg)
    console.print("\n[bold]opencode config[/bold] (~/.config/opencode/config.json):")
    console.print(Syntax(json.dumps(_mask_api_keys(opencode_cfg), indent=2), "json", theme="monokai"))

    console.print("\n[dim]Validating against opencode.ai/config.json schema…[/dim]")
    errors = _validate_opencode_config(opencode_cfg)
    fetch_warning = next((e for e in errors if e.startswith("⚠")), None)
    real_errors = [e for e in errors if not e.startswith("⚠")]
    if fetch_warning:
        console.print(f"  [yellow]{fetch_warning}[/yellow]")
    elif real_errors:
        for err in real_errors:
            console.print(f"  [red]✗[/red]  {err}")
    else:
        console.print("  [green]✓[/green] Schema valid")

    pi_cfg = _build_pi_config(cfg)
    console.print("\n[bold]pi config[/bold] (~/.pi/agent/models.json):")
    console.print(Syntax(json.dumps(_mask_api_keys(pi_cfg), indent=2), "json", theme="monokai"))
