"""Config loading and management."""

from __future__ import annotations

import os
import subprocess
import tomllib
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, Field, model_validator
from rich.console import Console
from rich.syntax import Syntax

console = Console()

CONFIG_FILENAME = "config.toml"

# Template written by `llm config init` — every option is a commented example
# so the file is self-documenting. Values below are usable defaults where safe;
# sensitive fields are left as obvious placeholders.
_CONFIG_TEMPLATE = """\
# Local LLM Configuration
# Edit this file, then run:  uv run llm config apply
#
# Machine roles — set the sections that apply to this machine:
#   server+client  — runs llama-server AND opencode/pi locally (most common)
#   server-only    — runs llama-server + nginx for remote clients; no local tools
#   client-only    — runs opencode/pi and connects to a remote server
#
# Sensitive values (api_key, hf_token) live only here — never committed.

# ── SERVER ────────────────────────────────────────────────────────────────────
# Skip [server] and [models] entirely on a client-only machine.

[server]
# Path to the llama-server binary.
# After building llama.cpp: ./llama.cpp/build/bin/llama-server
# Or after copying to PATH:  ~/.local/bin/llama-server
llama_server_bin = "llama-server"

# Internal HTTP port — llama-server listens here; nginx proxies to it.
port = 8080

# Model layers to offload to Vulkan iGPU (Radeon 890M).
# 0 = CPU only. Try 20–40 and tune up/down for speed.
n_gpu_layers = 20

# Context window in tokens. 65536+ recommended for agentic coding tasks.
n_ctx = 65536

# CPU inference threads. Recommended: physical core count (not hyperthreads).
n_threads = 12

# Extra llama-server flags (list of strings).
# Example: extra_args = ["--flash-attn", "--cache-type-k", "q8_0", "--jinja"]
extra_args = []

[models]
# Directory containing GGUF model files.
dir = "~/models"

# Active model filename. Run `uv run llm model list` to see options.
active = "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"

# HuggingFace token — only needed for gated/private models.
# Generate at: https://huggingface.co/settings/tokens
hf_token = ""

# ── PROXY ─────────────────────────────────────────────────────────────────────
# nginx TLS proxy — allows remote clients to reach llama-server securely.
# Skip [proxy] on a client-only machine.

[proxy]
# External HTTPS port nginx listens on (remote clients connect here).
port = 8443

# This machine's LAN IP address.
# Find it with: ip -4 addr show | grep 'inet ' | grep -v '127\\.0\\.0\\.1'
lan_ip = "192.168.1.100"

# LAN subnet allowed through nginx (CIDR). Requests from outside are rejected.
lan_subnet = "192.168.1.0/24"

# Bearer token remote clients must include in the Authorization header.
# Generate: python -c "import secrets; print(secrets.token_hex(32))"
api_key = "change-me-generate-a-strong-random-key"

# Path to the self-signed TLS certificate (public cert only, not the key).
# Generate with: uv run llm config gencert
cert_path = "/etc/ssl/local-llm/cert.pem"

# ── CLIENT ────────────────────────────────────────────────────────────────────
# How opencode, Pi, and other tools on THIS machine connect to the LLM.
#
# server+client machine: leave server_url empty — tools connect directly to
#   the local llama-server at http://127.0.0.1:<port> (no TLS, no auth).
#
# client-only machine: set server_url to the remote server's proxy URL,
#   and set api_key / cert_path to authenticate and trust the TLS cert.

[client]
# Leave empty on a server+client machine (uses local server directly).
# Set to remote proxy URL on a client-only machine:
#   server_url = "https://192.168.1.x:8443/v1"
server_url = ""

# API key for remote server (from that server's [proxy] api_key).
api_key = ""

# Path to the remote server's TLS cert PEM, for Node.js tools (Pi, opencode).
# Copy from the server: scp server:/etc/ssl/local-llm/cert.pem ~/.config/local-llm/cert.pem
# Then set NODE_EXTRA_CA_CERTS to this path in your shell profile.
cert_path = ""

# ── MODEL COST ────────────────────────────────────────────────────────────────
# Per-token pricing used when generating pi's models.json.
# Set to 0 for free local models.  For cloud APIs, use the provider's rates.
# Values are in USD per token.

[model_cost]
input = 0.0          # cost per input token
output = 0.0         # cost per output token
cache_write = 0.0    # cost per cached token written (5M tokens in API docs)
cache_read = 0.0     # cost per cached token read (1M tokens in API docs)

[lxd]
# Craft project directories to run `make setup` in via `llm lxd setup-crafts`.
craft_dirs = [
    # "~/dev/craft/snapcraft/snapcraft-a",
]

# Directories to bind-mount from the host into the LXD container/VM.
[[lxd.mounts]]
host = "~/.agents"

[[lxd.mounts]]
host = "~/.github"

[[lxd.mounts]]
host = "~/dev"

[[lxd.mounts]]
name = "opencode-config"
host = "~/.config/opencode"
"""


class ServerSettings(BaseModel):
    llama_server_bin: str = "llama-server"
    port: int = 8080
    n_gpu_layers: int = 20
    n_ctx: int = 4096
    n_threads: int = 12
    extra_args: list[str] = Field(default_factory=list)


class ModelCostSettings(BaseModel):
    """Per-token cost for pi's models.json.

    Prices are in USD per token.  Defaults are zero because local models
    are free — override for cloud-hosted APIs or when you want cost tracking.
    """

    input: float = 0.0  # $ per input token
    output: float = 0.0  # $ per output token
    cache_write: float = 0.0  # $ per cached (KV cache) token write
    cache_read: float = 0.0  # $ per cached (KV cache) token read

    def to_cost_dict(self) -> dict:  # type: ignore[type-arg]
        return {
            "input": self.input,
            "output": self.output,
            "cacheWrite": self.cache_write,
            "cacheRead": self.cache_read,
        }


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


class ClientSettings(BaseModel):
    """How client tools (opencode, Pi) on this machine connect to the LLM.

    server+client machine: leave server_url empty — defaults to the local
    llama-server at http://127.0.0.1:<port> (no TLS, no auth needed).

    client-only machine: set server_url to the remote proxy URL, and
    api_key / cert_path as needed.
    """

    server_url: str = ""
    api_key: str = ""
    cert_path: str = ""  # local path to remote server's TLS cert (PEM)


class MountEntry(BaseModel):
    host: str
    name: str = ""
    container: str = ""

    @model_validator(mode="after")
    def derive_defaults(self) -> MountEntry:
        host_expanded = str(Path(self.host).expanduser())
        if not self.name:
            self.name = Path(host_expanded).name.lstrip(".")
        if not self.container:
            self.container = host_expanded
        return self


class LxdSettings(BaseModel):
    craft_dirs: list[str] = Field(default_factory=list)
    mounts: list[MountEntry] = Field(default_factory=list)


class Settings(BaseModel):
    server: ServerSettings = Field(default_factory=ServerSettings)
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    model_cost: ModelCostSettings = Field(default_factory=ModelCostSettings, alias="model_cost")
    proxy: ProxySettings = Field(default_factory=ProxySettings)
    client: ClientSettings = Field(default_factory=ClientSettings)
    lxd: LxdSettings = Field(default_factory=LxdSettings)

    @property
    def has_local_server(self) -> bool:
        """True if this machine is configured to run llama-server."""
        return bool(self.server.llama_server_bin)

    @property
    def client_url(self) -> str:
        """Base URL (including /v1) for local tools to connect to the LLM.

        Defaults to the local llama-server (no TLS, no auth).
        Override with [client] server_url for a remote server.
        """
        if self.client.server_url:
            return self.client.server_url
        return f"{self.internal_url}/v1"

    @property
    def client_api_key(self) -> str:
        """API key for client tools. Empty when connecting to the local server."""
        return self.client.api_key

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


def try_load_lxd() -> LxdSettings | None:
    """Load only the [lxd] section from config.toml; return None if the file doesn't exist."""
    config_path = find_config()
    if not config_path.exists():
        return None
    with config_path.open("rb") as f:
        raw = tomllib.load(f)
    return LxdSettings.model_validate(raw.get("lxd", {}))


# ── Typer app ─────────────────────────────────────────────────────────────────

app = typer.Typer(help="Manage configuration and render templates.", no_args_is_help=True)


def _sudo(*args: str, desc: str) -> bool:
    """Run a sudo command, printing what runs. Returns True on success."""
    cmd = ["sudo", *args]
    console.print(f"  [dim]$ {' '.join(cmd)}[/dim]")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        msg = (result.stderr or result.stdout).strip()
        console.print(f"  [red]✗[/red]  {desc}" + (f": {msg}" if msg else ""))
        return False
    console.print(f"  [green]✓[/green]  {desc}")
    return True


def _systemctl_is_active(unit: str) -> bool:
    result = subprocess.run(["systemctl", "is-active", unit], capture_output=True, text=True)
    return result.stdout.strip() == "active"


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


def _build_opencode_config(cfg: Settings) -> dict:  # type: ignore[type-arg]
    """Build the opencode provider config dict from current settings."""
    from llm.models import KNOWN_MODELS  # noqa: PLC0415

    active = cfg.models.active
    entry = next((m for m in KNOWN_MODELS if m.filename == active), None)
    display_name = entry.alias if entry else active
    max_output = entry.max_output if entry else 8192

    # Use a stable generic key so the opencode config never needs updating when
    # switching models. llama-server ignores the model name in chat completion
    # requests and serves whatever is currently loaded.
    model_key = "local"

    # Without limit.input set, opencode ignores compaction.reserved and fires at
    # context - max_output (= 32K for Qwen3). Setting limit.input = n_ctx unlocks
    # the reserved path: usable = n_ctx - reserved, so compaction fires at ~57K.
    _compaction_reserved = 8192

    return {
        "$schema": "https://opencode.ai/config.json",
        "snapshot": True,
        "watcher": {
            "ignore": [".venv", "**/*.pyc", "**/__pycache__", "**/node_modules"],
        },
        "permission": "allow",
        "model": f"local-llm/{model_key}",
        "agent": {
            "build": {
                "temperature": 0.3,
                "steps": 50,
            },
            "plan": {
                "temperature": 0.1,
            },
        },
        # Compaction: fire late, keep lots of recent context verbatim so the
        # model doesn't forget what it was doing mid-task.
        "compaction": {
            "reserved": _compaction_reserved,
            "tail_turns": 10,
            "preserve_recent_tokens": 20000,
        },
        "provider": {
            "local-llm": {
                "name": "Local LLM",
                "npm": "@ai-sdk/openai-compatible",
                "api": cfg.client_url,
                "options": {"apiKey": cfg.client_api_key or "local"},
                "models": {
                    model_key: {
                        "name": display_name,
                        "limit": {
                            "context": cfg.server.n_ctx,
                            # Setting input = context activates the reserved path in
                            # opencode's overflow calc: usable = input - reserved.
                            # Without this, reserved is ignored and compaction fires
                            # at context - max_output (half the window for Qwen3).
                            "input": cfg.server.n_ctx,
                            "output": max_output,
                        },
                        "tool_call": True,
                        "options": {"repeat_penalty": 1.2},
                    }
                },
            }
        },
    }


_OPENCODE_SCHEMA_URL = "https://opencode.ai/config.json"
_OPENCODE_CONFIG_PATH = Path("~/.config/opencode/config.json")
_PI_CONFIG_PATH = Path("~/.pi/agent/models.json")


def _build_pi_config(cfg: Settings) -> dict:  # type: ignore[type-arg]
    """Build the pi-harness models.json config dict from current settings."""
    from llm.models import KNOWN_MODELS  # noqa: PLC0415

    active = cfg.models.active
    entry = next((m for m in KNOWN_MODELS if m.filename == active), None)
    display_name = entry.alias if entry else active
    max_output = entry.max_output if entry else 8192

    # apiKey is required by pi even for unauthenticated local servers.
    # Fall back to "local" so the field is always present.
    api_key = cfg.client_api_key or "local"

    return {
        "providers": {
            "local-llm": {
                "baseUrl": cfg.client_url,
                "api": "openai-completions",
                "apiKey": api_key,
                # llama-server compat: no developer role, no reasoning_effort,
                # uses max_tokens (not max_completion_tokens).
                "compat": {
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                    "maxTokensField": "max_tokens",
                },
                "models": [
                    {
                        "id": "local",
                        "name": display_name,
                        "contextWindow": cfg.server.n_ctx,
                        "maxTokens": max_output,
                        "cost": cfg.model_cost.to_cost_dict(),
                    }
                ],
            }
        }
    }


def _validate_opencode_config(cfg_dict: dict) -> list[str]:  # type: ignore[type-arg]
    """Validate opencode config dict against the live schema. Returns list of error strings."""
    import copy  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415
    import warnings  # noqa: PLC0415

    import jsonschema  # noqa: PLC0415

    try:
        req = urllib.request.Request(
            _OPENCODE_SCHEMA_URL,
            headers={"User-Agent": "local-llm-config-validator/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            schema = __import__("json").loads(resp.read())
    except Exception as exc:
        return [f"⚠ Could not fetch schema ({exc}) — skipping validation"]

    # Strip the $ref from 'model' field: it points to models.dev enum of known
    # cloud providers. Custom local providers will never be in that list, so we
    # validate it as a plain string only. Use setdefault to handle any schema shape.
    schema = copy.deepcopy(schema)
    schema.setdefault("properties", {})["model"] = {"type": "string"}

    errors = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for err in jsonschema.Draft202012Validator(schema).iter_errors(cfg_dict):
            path = " → ".join(str(p) for p in err.absolute_path) or "(root)"
            # Skip residual model-enum errors — local providers are never in the
            # cloud-provider enum that models.dev maintains.
            if path == "model":
                continue
            errors.append(f"{path}: {err.message}")
    return errors


@app.command("show")
def config_show() -> None:
    """Print current configuration (masks api_key and hf_token) and opencode config."""
    import json  # noqa: PLC0415

    cfg = load_config()
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

    opencode_cfg = _build_opencode_config(cfg)
    console.print("\n[bold]opencode config[/bold] (~/.config/opencode/config.json):")
    console.print(Syntax(json.dumps(opencode_cfg, indent=2), "json", theme="monokai"))

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
    console.print(Syntax(json.dumps(pi_cfg, indent=2), "json", theme="monokai"))


@app.command("apply")
def config_apply() -> None:
    """Render templates and install configs based on this machine's role."""
    import json  # noqa: PLC0415

    cfg = load_config()
    project_root = find_config().parent

    # ── Role summary ──────────────────────────────────────────────────────────
    server_tag = "[green]server[/green]" if cfg.has_local_server else "[dim]client-only[/dim]"
    client_tag = "[green]client[/green]"
    client_url_note = cfg.client_url
    console.print(
        f"Role: {server_tag} + {client_tag}  →  client connects to [cyan]{client_url_note}[/cyan]\n"
    )

    # ── Client configs (always) ───────────────────────────────────────────────
    pi_path = _PI_CONFIG_PATH.expanduser()
    pi_path.parent.mkdir(parents=True, exist_ok=True)
    pi_cfg = _build_pi_config(cfg)

    # Merge into existing models.json rather than overwriting, preserving other providers.
    existing: dict = {}
    if pi_path.exists():
        with pi_path.open("r") as f:
            existing = json.load(f)
    merged = {
        **existing,
        "providers": {**existing.get("providers", {}), **pi_cfg.get("providers", {})},
    }
    pi_path.write_text(json.dumps(merged, indent=2) + "\n")
    console.print(f"[green]Rendered[/green] {pi_path}")

    opencode_path = _OPENCODE_CONFIG_PATH.expanduser()
    opencode_path.parent.mkdir(parents=True, exist_ok=True)
    opencode_cfg = _build_opencode_config(cfg)

    console.print("[dim]Validating opencode config against schema…[/dim]")
    errors = _validate_opencode_config(opencode_cfg)
    fetch_warning = next((e for e in errors if e.startswith("⚠")), None)
    real_errors = [e for e in errors if not e.startswith("⚠")]

    if fetch_warning:
        console.print(f"  [yellow]{fetch_warning}[/yellow]")
    elif real_errors:
        for err in real_errors:
            console.print(f"  [red]✗[/red]  {err}")
        console.print("\n[red]opencode config has schema errors — not written.[/red]")
        raise typer.Exit(1)
    else:
        console.print("  [green]✓[/green] Schema valid")

    opencode_path.write_text(json.dumps(opencode_cfg, indent=2) + "\n")
    console.print(f"[green]Rendered[/green] {opencode_path}")

    if not cfg.has_local_server:
        console.print("\n[dim]Skipping nginx/systemd — no local server configured.[/dim]")
        return

    # ── Server-side: render templates ─────────────────────────────────────────
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

    console.print()
    for src, dst in templates:
        if not src.exists():
            console.print(f"[yellow]Template not found, skipping:[/yellow] {src}")
            continue
        text = src.read_text()
        for placeholder, value in replacements.items():
            text = text.replace(placeholder, value)
        dst.write_text(text)
        console.print(f"[green]Rendered[/green] {dst}")

    # ── nginx ─────────────────────────────────────────────────────────────────
    console.print("\n[bold]nginx[/bold]")
    nginx_src = project_root / "nginx" / "llm-proxy.conf"
    nginx_avail = Path("/etc/nginx/sites-available/llm")
    nginx_enabled = Path("/etc/nginx/sites-enabled/llm")
    if not nginx_src.exists():
        console.print("  [yellow]nginx/llm-proxy.conf not found — skipping[/yellow]")
    else:
        if _sudo("cp", str(nginx_src), str(nginx_avail), desc="install conf"):
            if not nginx_enabled.exists():
                _sudo("ln", "-sf", str(nginx_avail), str(nginx_enabled), desc="enable site")
            test = subprocess.run(["sudo", "nginx", "-t"], capture_output=True, text=True)
            if test.returncode != 0:
                console.print(f"  [red]✗[/red]  nginx -t failed:\n{test.stderr.strip()}")
            else:
                console.print("  [green]✓[/green]  nginx -t passed")
                if _systemctl_is_active("nginx"):
                    _sudo("systemctl", "reload", "nginx", desc="reload nginx")
                else:
                    _sudo("systemctl", "start", "nginx", desc="start nginx")

    # ── systemd ───────────────────────────────────────────────────────────────
    console.print("\n[bold]systemd[/bold]")
    svc_src = project_root / "systemd" / "llm-server.service"
    svc_dst = Path("/etc/systemd/system/llm-server.service")
    if not svc_src.exists():
        console.print("  [yellow]systemd/llm-server.service not found — skipping[/yellow]")
    else:
        if _sudo("cp", str(svc_src), str(svc_dst), desc="install service"):
            _sudo("systemctl", "daemon-reload", desc="daemon-reload")
            _sudo("systemctl", "enable", "llm-server", desc="enable llm-server")
            if _systemctl_is_active("llm-server"):
                console.print(
                    "  [dim]llm-server is running — restart to pick up changes:[/dim]\n"
                    "    [bold]uv run llm server restart[/bold]"
                )


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
            f"[yellow]Cert already exists:[/yellow] {cert_path}\nUse [bold]--force[/bold] to regenerate."
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
