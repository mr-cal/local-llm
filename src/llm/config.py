"""Config loading and management."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, Field
from rich.console import Console
from rich.syntax import Syntax

console = Console()

CONFIG_FILENAME = "config.toml"

# Template written by `llm config init` — every option is a commented example
# so the file is self-documenting. Values below are usable defaults where safe;
# sensitive fields are left as obvious placeholders.
_CONFIG_TEMPLATE = """\
# Local LLM Server Configuration
# Edit this file, then run:
#   uv run llm config apply   — render nginx + systemd files
#   uv run llm server start   — start llama-server
#
# Sensitive values (api_key, lan_ip, hf_token) live only here — never committed.

[server]
# Path to the llama-server binary.
# After building llama.cpp: ./llama.cpp/build/bin/llama-server
# Or after `make install`:    /usr/local/bin/llama-server
llama_server_bin = "llama-server"

# Internal HTTP port — llama-server listens here; nginx proxies to it.
port = 8080

# Model layers to offload to Vulkan iGPU (Radeon 890M).
# 0 = CPU only. Try 20–40 and tune up/down for speed.
# Each offloaded layer uses ~100–200 MB of shared VRAM depending on the model.
n_gpu_layers = 20

# Context window in tokens. Larger uses more RAM. 4096 is a safe default.
# For a 14B Q4 model with 62 GB RAM you can safely go up to 32768.
n_ctx = 4096

# CPU inference threads. Recommended: physical core count (not hyperthreads).
# Ryzen AI 9 HX 370 has 12 physical cores → set 12.
n_threads = 12

# Extra llama-server flags (list of strings).
# Useful options: "--flash-attn"  "--mlock"  "--no-mmap"
# Example: extra_args = ["--flash-attn"]
extra_args = []

[models]
# Directory containing GGUF model files.
dir = "~/models"

# Filename of the active model (must be present in dir above).
# Run `uv run llm model list` to see what's downloaded.
#
# Recommended Qwen models for this machine (62 GB RAM):
#   Qwen2.5-Coder-7B-Instruct-Q8_0.gguf          ~8 GB   fastest
#   Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf        ~8.5 GB best balance  <- default
#   Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf        ~18 GB  strong coding
#   Qwen2.5-Coder-32B-Instruct-Q8_0.gguf           ~34 GB  high precision
#   Qwen2.5-72B-Instruct-Q4_K_M.gguf               ~42 GB  near-frontier  <- 62 GB advantage
#   Qwen3-30B-A3B-Q4_K_M.gguf                      ~17 GB  MoE, efficient
active = "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"

# HuggingFace token — only needed for gated/private models. Leave empty otherwise.
# Generate at: https://huggingface.co/settings/tokens
hf_token = ""

[proxy]
# External HTTPS port nginx listens on (clients connect here).
port = 8443

# This machine's LAN IP address.
# Find it with: ip -4 addr show | grep 'inet ' | grep -v '127\\.0\\.0\\.1'
lan_ip = "192.168.1.100"

# LAN subnet allowed through nginx (CIDR). Requests from outside are rejected.
lan_subnet = "192.168.1.0/24"

# Bearer token clients must include in the Authorization header.
# Generate a strong key:
#   python -c "import secrets; print(secrets.token_hex(32))"
api_key = "change-me-generate-a-strong-random-key"

# Path to the self-signed TLS certificate (public cert only, not the key).
# Used by `llm client setup` to display installation instructions for the LXD container.
# Generate with: uv run llm config gencert
# (this sets the correct SubjectAltName for the LAN IP — required by modern TLS / Node.js)
cert_path = "/etc/ssl/local-llm/cert.pem"
"""


class ServerSettings(BaseModel):
    llama_server_bin: str = "llama-server"
    port: int = 8080
    n_gpu_layers: int = 20
    n_ctx: int = 4096
    n_threads: int = 12
    extra_args: list[str] = Field(default_factory=list)


class ModelsSettings(BaseModel):
    dir: str = "~/models"
    active: str = "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"
    hf_token: str = ""


class ProxySettings(BaseModel):
    port: int = 8443
    lan_ip: str = "192.168.1.100"
    lan_subnet: str = "192.168.1.0/24"
    api_key: str = "change-me-generate-a-strong-random-key"
    cert_path: str = "/etc/ssl/local-llm/cert.pem"


class Settings(BaseModel):
    server: ServerSettings = Field(default_factory=ServerSettings)
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    proxy: ProxySettings = Field(default_factory=ProxySettings)

    @property
    def models_path(self) -> Path:
        return Path(self.models.dir).expanduser().resolve()

    @property
    def model_path(self) -> Path:
        return self.models_path / self.models.active

    @property
    def internal_url(self) -> str:
        return f"http://127.0.0.1:{self.server.port}"

    @property
    def proxy_url(self) -> str:
        return f"https://{self.proxy.lan_ip}:{self.proxy.port}"


def find_config() -> Path:
    """Walk up from CWD to find config.toml."""
    here = Path.cwd()
    for directory in [here, *here.parents]:
        candidate = directory / CONFIG_FILENAME
        if candidate.exists():
            return candidate
        if (directory / "pyproject.toml").exists():
            break  # stop at project root even if config.toml is missing
    return Path.cwd() / CONFIG_FILENAME


def load_config() -> Settings:
    """Load settings from config.toml. Raises SystemExit with a helpful message if not found."""
    config_path = find_config()
    if not config_path.exists():
        console.print(
            f"[red]Config file not found:[/red] {config_path}\n"
            "Run [bold]uv run llm config init[/bold] to create it."
        )
        raise typer.Exit(1)
    with config_path.open("rb") as f:
        raw = tomllib.load(f)
    return Settings.model_validate(raw)


# ── Typer app ─────────────────────────────────────────────────────────────────

app = typer.Typer(help="Manage configuration and render templates.")


@app.command("init")
def config_init(
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing config.")] = False,
) -> None:
    """Create config.toml with commented examples (gitignored)."""
    config_path = Path.cwd() / CONFIG_FILENAME
    if config_path.exists() and not force:
        console.print(
            f"[yellow]Config already exists:[/yellow] {config_path}\nUse [bold]--force[/bold] to overwrite."
        )
        raise typer.Exit(1)
    config_path.write_text(_CONFIG_TEMPLATE)
    console.print(f"[green]Created[/green] {config_path}")
    console.print("Edit it, then run: [bold]uv run llm config apply[/bold]")


@app.command("show")
def config_show() -> None:
    """Print current configuration (masks api_key and hf_token)."""
    cfg = load_config()
    _CONFIG_TEMPLATE.splitlines()
    # Build a display-safe version by masking secrets

    masked = cfg.model_dump()
    masked["proxy"]["api_key"] = "***"
    masked["models"]["hf_token"] = "***" if masked["models"]["hf_token"] else ""

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


@app.command("apply")
def config_apply() -> None:
    """Render nginx and systemd templates using current config."""
    cfg = load_config()
    project_root = find_config().parent

    replacements = {
        "%%LAN_IP%%": cfg.proxy.lan_ip,
        "%%LAN_SUBNET%%": cfg.proxy.lan_subnet,
        "%%PROXY_PORT%%": str(cfg.proxy.port),
        "%%SERVER_PORT%%": str(cfg.server.port),
        "%%API_KEY%%": cfg.proxy.api_key,
        "%%LLAMA_SERVER_BIN%%": cfg.server.llama_server_bin,
        "%%MODELS_DIR%%": str(cfg.models_path),
        "%%ACTIVE_MODEL%%": cfg.models.active,
        "%%N_GPU_LAYERS%%": str(cfg.server.n_gpu_layers),
        "%%N_CTX%%": str(cfg.server.n_ctx),
        "%%N_THREADS%%": str(cfg.server.n_threads),
        "%%USER%%": os.environ.get("USER", os.environ.get("LOGNAME", "nobody")),
    }

    templates = [
        (project_root / "nginx" / "llm-proxy.conf.template", project_root / "nginx" / "llm-proxy.conf"),
        (
            project_root / "systemd" / "llm-server.service.template",
            project_root / "systemd" / "llm-server.service",
        ),
    ]

    for src, dst in templates:
        if not src.exists():
            console.print(f"[yellow]Template not found, skipping:[/yellow] {src}")
            continue
        text = src.read_text()
        for placeholder, value in replacements.items():
            text = text.replace(placeholder, value)
        dst.write_text(text)
        console.print(f"[green]Rendered[/green] {dst}")

    console.print("\nNext steps:")
    console.print("  nginx:")
    console.print("    sudo cp nginx/llm-proxy.conf /etc/nginx/sites-available/llm")
    console.print("    sudo nginx -t && sudo systemctl reload nginx")
    console.print("  systemd:")
    console.print("    sudo cp systemd/llm-server.service /etc/systemd/system/")
    console.print("    sudo systemctl daemon-reload && sudo systemctl enable --now llm-server")


@app.command("gencert")
def config_gencert(
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing cert.")] = False,
) -> None:
    """Generate a self-signed TLS certificate with the correct SubjectAltName.

    IMPORTANT: The cert MUST include a SAN matching the server's LAN IP.
    Node.js (and modern browsers) reject certs that only have a CN — they require
    a SubjectAltName. This command uses the lan_ip from config.toml automatically.

    Writes cert + key to the paths configured in [proxy] cert_path (key stored
    alongside as key.pem). Requires openssl and sudo.
    """
    import subprocess  # noqa: PLC0415

    cfg = load_config()
    cert_path = Path(cfg.proxy.cert_path)
    key_path = cert_path.parent / "key.pem"
    lan_ip = cfg.proxy.lan_ip

    if cert_path.exists() and not force:
        console.print(
            f"[yellow]Cert already exists:[/yellow] {cert_path}\n"
            "Use [bold]--force[/bold] to regenerate."
        )
        raise typer.Exit(1)

    console.print(f"Generating cert for IP [bold]{lan_ip}[/bold] → {cert_path}")

    cmd = [
        "sudo",
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:4096",
        "-keyout",
        str(key_path),
        "-out",
        str(cert_path),
        "-days",
        "3650",
        "-nodes",
        "-subj",
        "/CN=local-llm",
        "-addext",
        f"subjectAltName=IP:{lan_ip},DNS:local-llm",
    ]

    # Ensure the directory exists first
    mkdir_result = subprocess.run(["sudo", "mkdir", "-p", str(cert_path.parent)], check=False)
    if mkdir_result.returncode != 0:
        console.print(f"[red]Failed to create directory:[/red] {cert_path.parent}")
        raise typer.Exit(1)

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]openssl failed:[/red]\n{result.stderr}")
        raise typer.Exit(1)

    # Restrict permissions on the private key
    subprocess.run(["sudo", "chmod", "600", str(key_path)], check=False)

    console.print(f"[green]Certificate written:[/green] {cert_path}")
    console.print(f"[green]Private key written:[/green]  {key_path}")
    console.print(
        "\n[bold]Next:[/bold] Update nginx to reference the new cert, then reload:\n"
        "  uv run llm config apply\n"
        "  sudo cp nginx/llm-proxy.conf /etc/nginx/sites-available/llm\n"
        "  sudo nginx -t && sudo systemctl reload nginx\n"
        "\nThen re-run [bold]uv run llm client setup[/bold] to get updated container instructions."
    )
