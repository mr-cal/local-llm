"""Config loading and management."""

from __future__ import annotations

import os
import subprocess
import tomllib
from importlib import resources
from pathlib import Path

import typer
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rich.console import Console
from rich.syntax import Syntax

console = Console()

CONFIG_FILENAME = "config.toml"

# Template written by `llm config init` — loaded from config_template.toml.
# Every option is a commented example so the file is self-documenting.
_CONFIG_TEMPLATE = (
    resources.files("llm").joinpath("config_template.toml").read_text(encoding="utf-8")
)


BACKEND_FLAGS: dict[str, str] = {
    "vulkan": "-DGGML_VULKAN=ON",
    "metal": "-DGGML_METAL=ON",
    "cuda": "-DGGML_CUDA=ON",
    "blis": "-DGGML_BLIS=ON",
    "hipblas": "-DGGML_HIPBLAS=ON",
    "coreml": "-DGGML_COREML=ON",
    "kluster": "-DGGML_KLUSTER=ON",
}


class BuildProfile(BaseModel):
    """A single build profile — a named set of cmake flags."""

    name: str
    backend: str | None = None  # convenience shorthand, e.g. "vulkan" → -DGGML_VULKAN=ON
    extra_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_backend(self) -> BuildProfile:
        if self.backend and self.backend not in BACKEND_FLAGS:
            raise ValueError(
                f"Unknown backend '{self.backend}'. "
                f"Valid options: {', '.join(sorted(BACKEND_FLAGS))}"
            )
        return self

    def get_full_flags(self) -> list[str]:
        """Return the complete list of cmake flags for this profile."""
        flags = []
        if self.backend:
            flags.append(BACKEND_FLAGS[self.backend])
        flags.extend(self.extra_flags)
        return flags

    @property
    def build_dir_name(self) -> str:
        """Name of the out-of-tree cmake build directory (inside llama.cpp/)."""
        return f"build-{self.name}"

    def installed_server_bin(self, install_dir: Path) -> Path:
        """Resolved path to llama-server installed for this profile."""
        return install_dir / self.name / "llama-server"

    def installed_bench_bin(self, install_dir: Path) -> Path:
        """Resolved path to llama-bench installed for this profile."""
        return install_dir / self.name / "llama-bench"


class BuildConfig(BaseModel):
    """Configuration for the llama.cpp build system."""

    enabled: bool = True
    repo: str = "https://github.com/ggerganov/llama.cpp"
    commit: str = "HEAD"
    install_dir: str = "~/.local/bin"
    jobs: str = "auto"  # "auto" = nproc; number for specific count
    release: bool = True
    profiles: list[BuildProfile] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_profile_names(self) -> BuildConfig:
        names = [p.name for p in self.profiles]
        if len(names) != len(set(names)):
            dupes = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate profile names: {dupes}")
        return self

    @property
    def install_path(self) -> Path:
        return Path(self.install_dir).expanduser().resolve()

    @property
    def active_profile(self) -> BuildProfile | None:
        """Return the first profile, or None if no profiles are configured."""
        return self.profiles[0] if self.profiles else None

    def get_profile(self, name: str | None = None) -> BuildProfile | None:
        """Return profile by name, or active_profile if name is None."""
        if name is None:
            return self.active_profile
        return next((p for p in self.profiles if p.name == name), None)

    def profile_names(self) -> list[str]:
        return [p.name for p in self.profiles]

    def jobs_count(self) -> int:
        """Return the number of parallel build jobs."""
        if self.jobs == "auto":
            import os  # noqa: PLC0415

            return os.cpu_count() or 1
        return int(self.jobs)


class ServerSettings(BaseModel):
    enabled: bool = True
    llama_server_bin: str = "llama-server"
    # Name of the build profile whose binary to use when llama_server_bin is empty.
    # If both are empty, falls back to 'llama-server' on PATH.
    profile: str = ""
    port: int = 8080
    n_gpu_layers: int = 20
    n_ctx: int = 4096
    n_threads: int = 12
    extra_args: list[str] = Field(default_factory=list)


class ModelCost(BaseModel):
    """Per-token cost for a single model.

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

    def is_zero(self) -> bool:
        """True when all cost fields are zero (not configured)."""
        return self.input == 0.0 and self.output == 0.0 and self.cache_write == 0.0 and self.cache_read == 0.0


class AuthSettings(BaseModel):
    """Bearer token used by remote clients to authenticate with this server."""

    api_key: str = ""  # Generate: python -c "import secrets; print(secrets.token_hex(32))"


class ModelEntry(BaseModel):
    """One model in the [[models.list]] catalog."""

    alias: str
    repo: str
    filename: str
    size: str = ""
    description: str = ""
    max_output: int = 8192
    cost: ModelCost = Field(default_factory=ModelCost)

    @property
    def id(self) -> str:
        """Unique identifier — alias."""
        return self.alias


class ModelsSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    dir: str = "~/models"
    active: str = "qwen2.5-coder-14b-q4"  # alias, not filename
    hf_token: str = ""
    entries: list[ModelEntry] = Field(default_factory=list, alias="list")

    @model_validator(mode="after")
    def validate_active(self) -> ModelsSettings:
        # If active is an alias, check it exists in the list (for non-custom models)
        if self.entries and self.active:
            by_alias = {m.alias for m in self.entries}
            if self.active not in by_alias:
                # Still allow custom/uncatalogued models (legacy compat)
                pass
        return self

    @property
    def models_path(self) -> Path:
        return Path(self.dir).expanduser().resolve()

    @property
    def model_path(self) -> Path:
        """Path to the active model file, resolving alias→filename when possible."""
        # If active is already a filename (e.g. legacy config or custom model)
        if self.active.endswith(".gguf"):
            return self.models_path / self.active
        # Try to resolve via catalog
        entry = self.by_alias(self.active)
        if entry:
            return self.models_path / entry.filename
        entry = self.by_filename(self.active)
        if entry:
            return self.models_path / entry.filename
        # Fallback: treat active as a filename
        return self.models_path / self.active

    def by_alias(self, alias: str) -> ModelEntry | None:
        return next((m for m in self.entries if m.alias == alias), None)

    def by_filename(self, filename: str) -> ModelEntry | None:
        return next((m for m in self.entries if m.filename == filename), None)

    @property
    def has_catalog(self) -> bool:
        return len(self.entries) > 0


class ProxySettings(BaseModel):
    enabled: bool = True
    port: int = 8443
    lan_ip: str = "192.168.1.100"
    lan_subnet: str = "192.168.1.0/24"
    cert_path: str = "/etc/ssl/local-llm/cert.pem"


class GitHubSettings(BaseModel):
    """GitHub CLI (gh) authentication token."""

    token: str = ""  # GitHub personal access token for gh CLI auth

    def is_authenticated(self) -> bool:
        """True when a non-empty token is configured."""
        return bool(self.token.strip())


class ClientSettings(BaseModel):
    """How client tools (opencode, Pi) on this machine connect to the LLM.

    server+client machine: leave server_url empty — defaults to the local
    llama-server at http://127.0.0.1:<port> (no TLS, no auth needed).

    client-only machine: set server_url to the remote proxy URL, and
    cert_path as needed (auth is read from [auth]).
    """

    enabled: bool = True
    server_url: str = ""
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
    auth: AuthSettings = Field(default_factory=AuthSettings)
    proxy: ProxySettings = Field(default_factory=ProxySettings)
    client: ClientSettings = Field(default_factory=ClientSettings)
    lxd: LxdSettings = Field(default_factory=LxdSettings)
    build: BuildConfig = Field(default_factory=BuildConfig)
    github: GitHubSettings = Field(default_factory=GitHubSettings)

    @property
    def has_local_server(self) -> bool:
        """True if this machine is configured to run llama-server."""
        return bool(self.server.llama_server_bin or self.server.profile or self.build.profiles)

    def resolve_llama_server_bin(self) -> str:
        """Resolve the llama-server binary path using the configured priority:

        1. Explicit ``[server] llama_server_bin`` (non-empty string)
        2. Profile-resolved: ``<build.install_dir>/<profile>/llama-server``
        3. ``llama-server`` on PATH (shutil.which fallback)
        """
        import shutil  # noqa: PLC0415

        if self.server.llama_server_bin:
            return self.server.llama_server_bin

        # Try profile-based resolution
        profile_name = self.server.profile or (
            self.build.active_profile.name if self.build.active_profile else None
        )
        if profile_name and self.build.profiles:
            profile = self.build.get_profile(profile_name)
            if profile:
                return str(profile.installed_server_bin(self.build.install_path))

        return shutil.which("llama-server") or "llama-server"

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
        return self.auth.api_key

    @property
    def models_path(self) -> Path:
        return Path(self.models.dir).expanduser().resolve()

    @property
    def model_path(self) -> Path:
        return self.models.model_path

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


def _resolve_active(active: str, model_list: list[ModelEntry]) -> str:
    """Resolve a filename or alias in the TOML 'active' field to an alias.

    If the active value matches a catalog filename → return the alias.
    Otherwise return the value as-is (custom/uncatalogued model).
    """
    if not model_list:
        return active
    entry = next((m for m in model_list if m.filename == active), None)
    return entry.alias if entry else active


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
    import re  # noqa: PLC0415

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
    console.print("[bold]Step 1/3[/bold] — Server connection")
    server_url = _prompt_init("  Server URL", "https://192.168.1.x:8443/v1")

    # ── Step 2: Auth ──────────────────────────────────────────────────────
    console.print("\n[bold]Step 2/3[/bold] — Authentication")
    console.print("  [dim]Find the API key in config.toml on the server (auth.api_key).[/dim]")
    api_key = _prompt_init("  API key")

    # ── Step 3: TLS certificate ───────────────────────────────────────────
    console.print("\n[bold]Step 3/3[/bold] — TLS certificate")
    default_cert = "~/.config/local-llm/cert.pem"
    cert_path = _prompt_init("  Local cert path", default_cert)
    cert_expanded = Path(cert_path).expanduser()

    if not cert_expanded.exists():
        console.print(f"\n  [yellow]Cert not found at {cert_expanded}[/yellow]")
        fetch = _prompt_init("  Fetch from server via scp? (y/n)", "y")
        if fetch.lower() in ("y", "yes"):
            server_host = _prompt_init("  Server SSH host (e.g. user@192.168.1.209)")
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
                    f"  [red]✗[/red] scp failed — copy it manually:\n"
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


def _build_opencode_config(cfg: Settings) -> dict:  # type: ignore[type-arg]
    """Build the opencode provider config dict from current settings."""
    from llm.models import KNOWN_MODELS  # noqa: PLC0415

    active = cfg.models.active
    # First check config catalog, then fall back to KNOWN_MODELS
    entry = cfg.models.by_alias(active) or cfg.models.by_filename(active)
    if entry is None and cfg.models.has_catalog is False:
        # Dual catalog fallback: search KNOWN_MODELS by filename
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


def _get_lxd_bridge_info() -> tuple[str, str]:
    """Return ``(host_ip, subnet)`` for the lxdbr0 bridge.

    Example return: ``("10.113.167.1", "10.113.167.0/24")``.
    Returns ``("", "")`` when lxdbr0 is not found or ``ip(8)`` fails.
    """
    import ipaddress  # noqa: PLC0415

    result = subprocess.run(
        ["ip", "-4", "addr", "show", "lxdbr0"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "", ""
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("inet "):
            ip_cidr = line.split()[1]
            host_ip = ip_cidr.split("/")[0]
            network = str(ipaddress.ip_interface(ip_cidr).network)
            return host_ip, network
    return "", ""


def _build_pi_config(cfg: Settings) -> dict:  # type: ignore[type-arg]
    """Build the pi-harness models.json config dict from current settings."""
    from llm.models import KNOWN_MODELS  # noqa: PLC0415

    active = cfg.models.active
    # First check config catalog, then fall back to KNOWN_MODELS
    entry = cfg.models.by_alias(active) or cfg.models.by_filename(active)
    if entry is None and cfg.models.has_catalog is False:
        # Dual catalog fallback: search KNOWN_MODELS by filename
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
                        "cost": (
                            entry.cost.to_cost_dict()
                            if entry
                            else {
                                "input": 0.0,
                                "output": 0.0,
                                "cacheWrite": 0.0,
                                "cacheRead": 0.0,
                            }
                        ),
                    }
                ],
            }
        }
    }


def _build_pi_config_for_container(cfg: Settings, server_host: str) -> dict:  # type: ignore[type-arg]
    """Build the pi-harness models.json config dict for use INSIDE an LXD container.

    The container cannot reach the host's llama-server at 127.0.0.1, so it
    connects via the nginx TLS proxy using *server_host* (typically the
    ``local-llm`` hostname, which is already in the cert's SubjectAltName).
    """
    from llm.models import KNOWN_MODELS  # noqa: PLC0415

    active = cfg.models.active
    entry = cfg.models.by_alias(active) or cfg.models.by_filename(active)
    if entry is None and cfg.models.has_catalog is False:
        entry = next((m for m in KNOWN_MODELS if m.filename == active), None)
    display_name = entry.alias if entry else active
    max_output = entry.max_output if entry else 8192
    api_key = cfg.client_api_key or "local"

    # Container connects via the nginx TLS proxy using the provided host.
    # Using the 'local-llm' hostname (present in the cert's SAN) avoids
    # TLS verification failures that would occur with a raw IP address.
    base_url = f"https://{server_host}:{cfg.proxy.port}/v1"

    return {
        "providers": {
            "local-llm": {
                "baseUrl": base_url,
                "api": "openai-completions",
                "apiKey": api_key,
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
                        "cost": (
                            entry.cost.to_cost_dict()
                            if entry
                            else {
                                "input": 0.0,
                                "output": 0.0,
                                "cacheWrite": 0.0,
                                "cacheRead": 0.0,
                            }
                        ),
                    }
                ],
            }
        }
    }


def _build_opencode_config_for_container(cfg: Settings, server_host: str) -> dict:  # type: ignore[type-arg]
    """Build the opencode config dict for use INSIDE an LXD container.

    Like :func:`_build_opencode_config`, but connects via the nginx TLS proxy
    at ``https://<server_host>:<port>/v1`` instead of localhost.
    """
    from llm.models import KNOWN_MODELS  # noqa: PLC0415

    active = cfg.models.active
    entry = cfg.models.by_alias(active) or cfg.models.by_filename(active)
    if entry is None and cfg.models.has_catalog is False:
        entry = next((m for m in KNOWN_MODELS if m.filename == active), None)
    display_name = entry.alias if entry else active
    max_output = entry.max_output if entry else 8192

    model_key = "local"
    _compaction_reserved = 8192
    api_key = cfg.client_api_key or "local"
    base_url = f"https://{server_host}:{cfg.proxy.port}/v1"

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
        "compaction": {
            "reserved": _compaction_reserved,
            "tail_turns": 10,
            "preserve_recent_tokens": 20000,
        },
        "provider": {
            "local-llm": {
                "name": "Local LLM",
                "npm": "@ai-sdk/openai-compatible",
                "api": base_url,
                "options": {"apiKey": api_key},
                "models": {
                    model_key: {
                        "name": display_name,
                        "limit": {
                            "context": cfg.server.n_ctx,
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


# ── Reusable apply helpers ────────────────────────────────────────────────────
# Extracted from the old config_apply command so they can be called from
# server.py and client.py setup commands.


def apply_client_configs(cfg: Settings) -> None:
    """Write opencode and pi configs for the host machine.

    Writes ~/.config/opencode/config.json and ~/.pi/agent/models.json.
    Validates the opencode config against the live schema before writing.
    """
    import json  # noqa: PLC0415

    # ── Pi config ─────────────────────────────────────────────────────────
    pi_path = _PI_CONFIG_PATH.expanduser()
    pi_path.parent.mkdir(parents=True, exist_ok=True)
    pi_cfg = _build_pi_config(cfg)

    # Merge into existing models.json, preserving other providers.
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

    # ── Opencode config ───────────────────────────────────────────────────
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


def apply_server_configs(cfg: Settings, project_root: Path) -> None:
    """Render nginx/systemd templates and install them.

    Handles template rendering, nginx site install + reload, and systemd
    service install + daemon-reload + enable.
    """
    from llm.models import KNOWN_MODELS  # noqa: PLC0415

    # Resolve alias → filename for template replacements
    active = cfg.models.active
    active_filename = active
    entry = cfg.models.by_alias(active)
    if entry:
        active_filename = entry.filename
    else:
        entry = next((m for m in KNOWN_MODELS if m.alias == active), None)
        if entry:
            active_filename = entry.filename

    replacements = {
        "%%LAN_IP%%": cfg.proxy.lan_ip,
        "%%LAN_SUBNET%%": cfg.proxy.lan_subnet,
        "%%PROXY_PORT%%": str(cfg.proxy.port),
        "%%SERVER_PORT%%": str(cfg.server.port),
        "%%API_KEY%%": cfg.auth.api_key,
        "%%LLAMA_SERVER_BIN%%": cfg.server.llama_server_bin,
        "%%MODELS_DIR%%": str(cfg.models_path),
        "%%ACTIVE_MODEL%%": active_filename,
        "%%N_GPU_LAYERS%%": str(cfg.server.n_gpu_layers),
        "%%N_CTX%%": str(cfg.server.n_ctx),
        "%%N_THREADS%%": str(cfg.server.n_threads),
        "%%USER%%": os.environ.get("USER", os.environ.get("LOGNAME", "nobody")),
    }

    # Auto-detect lxdbr0 bridge to allow LXD containers to reach the proxy.
    lxd_bridge_ip, lxd_bridge_subnet = _get_lxd_bridge_info()
    if lxd_bridge_ip and lxd_bridge_subnet:
        console.print(
            f"  [dim]LXD bridge detected: {lxd_bridge_ip} ({lxd_bridge_subnet})[/dim]"
        )
        replacements["%%LXD_ALLOW_LINE%%"] = f"    allow {lxd_bridge_subnet};\n"
    else:
        replacements["%%LXD_ALLOW_LINE%%"] = ""

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

    # ── nginx ─────────────────────────────────────────────────────────────
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

    # ── systemd ───────────────────────────────────────────────────────────
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


def generate_tls_cert(cfg: Settings, force: bool = False) -> bool:
    """Generate a self-signed TLS certificate with the correct SubjectAltName.

    Returns True on success, False on failure. Skips if cert already exists
    unless *force* is True.
    """
    cert_path = Path(cfg.proxy.cert_path)
    key_path = cert_path.parent / "key.pem"
    lan_ip = cfg.proxy.lan_ip

    if cert_path.exists() and not force:
        console.print(f"[dim]Cert already exists:[/dim] {cert_path}")
        return True

    console.print(f"Generating cert for IP [bold]{lan_ip}[/bold] → {cert_path}")

    cmd = [
        "sudo", "openssl", "req", "-x509", "-newkey", "rsa:4096",
        "-keyout", str(key_path), "-out", str(cert_path),
        "-days", "3650", "-nodes",
        "-subj", "/CN=local-llm",
        "-addext", f"subjectAltName=IP:{lan_ip},DNS:local-llm",
    ]

    mkdir_result = subprocess.run(["sudo", "mkdir", "-p", str(cert_path.parent)], check=False)
    if mkdir_result.returncode != 0:
        console.print(f"[red]Failed to create directory:[/red] {cert_path.parent}")
        return False

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]openssl failed:[/red]\n{result.stderr}")
        return False

    subprocess.run(["sudo", "chmod", "600", str(key_path)], check=False)
    console.print(f"[green]Certificate written:[/green] {cert_path}")
    console.print(f"[green]Private key written:[/green]  {key_path}")
    return True


def generate_api_key() -> str:
    """Generate a random hex API key."""
    import secrets  # noqa: PLC0415

    return secrets.token_hex(32)


def detect_lan_ip() -> str:
    """Auto-detect the machine's LAN IP address.

    Returns the first non-loopback IPv4 address, or an empty string if detection fails.
    """
    result = subprocess.run(
        ["ip", "-4", "addr", "show"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return ""
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("inet ") and "127.0.0.1" not in line:
            return line.split()[1].split("/")[0]
    return ""


def write_config_toml(cfg_dict: dict, path: Path | None = None) -> Path:  # type: ignore[type-arg]
    """Write a config dict as TOML to *path* (default: CWD/config.toml).

    Returns the path written to.
    """
    import tomli_w  # noqa: PLC0415

    if path is None:
        path = Path.cwd() / CONFIG_FILENAME
    path.write_text(tomli_w.dumps(cfg_dict))
    return path


# ── Shell profile helpers ─────────────────────────────────────────────────────


def _ensure_line_in_file(path: Path, line: str) -> bool:
    """Ensure *line* is present in *path*. Creates the file if needed.

    Returns True if the line was added, False if already present.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        content = path.read_text()
        if line in content:
            return False
    with path.open("a") as f:
        f.write(f"\n{line}\n")
    return True


def configure_shell_env_host(
    base_url: str,
    api_key: str,
    cert_path: str,
) -> list[str]:
    """Configure shell env vars on the host for bash and fish.

    Writes to dedicated config files:
    - ~/.config/local-llm/env (sourced from ~/.bashrc)
    - ~/.config/fish/conf.d/local-llm.fish

    Returns a list of actions taken.
    """
    home = Path.home()
    actions: list[str] = []

    # ── Bash: dedicated env file sourced from bashrc ──────────────────────
    env_dir = home / ".config" / "local-llm"
    env_file = env_dir / "env"
    env_dir.mkdir(parents=True, exist_ok=True)

    env_content = (
        "# Generated by local-llm — do not edit manually\n"
        f'export OPENAI_BASE_URL="{base_url}"\n'
        f'export OPENAI_API_KEY="{api_key}"\n'
        f'export NODE_EXTRA_CA_CERTS="{cert_path}"\n'
    )
    env_file.write_text(env_content)
    actions.append(f"Wrote {env_file}")

    # Source from ~/.bashrc if not already present
    source_line = f'[ -f "{env_file}" ] && source "{env_file}"'
    bashrc = home / ".bashrc"
    if _ensure_line_in_file(bashrc, source_line):
        actions.append(f"Added source line to {bashrc}")

    # ── Fish: conf.d snippet ──────────────────────────────────────────────
    fish_conf = home / ".config" / "fish" / "conf.d" / "local-llm.fish"
    fish_content = (
        "# Generated by local-llm — do not edit manually\n"
        f'set -gx OPENAI_BASE_URL "{base_url}"\n'
        f'set -gx OPENAI_API_KEY "{api_key}"\n'
        f'set -gx NODE_EXTRA_CA_CERTS "{cert_path}"\n'
    )
    fish_conf.parent.mkdir(parents=True, exist_ok=True)
    fish_conf.write_text(fish_content)
    actions.append(f"Wrote {fish_conf}")

    return actions


@app.command("show")
def config_show() -> None:
    """Print current configuration (masks api_key and hf_token) and opencode config."""
    import json  # noqa: PLC0415

    cfg = load_config()
    # Build a display-safe version by masking secrets

    masked = cfg.model_dump()
    masked["auth"]["api_key"] = "***"
    masked["models"]["hf_token"] = "***" if masked["models"]["hf_token"] else ""
    masked["github"]["token"] = "***" if masked["github"]["token"] else ""

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
            console.print(
                f"  {active_marker} {m.alias}  {m.size:>8}  {m.description}{cost_str}"
            )
        console.print("\n[dim]▶ = active[/dim]")

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
