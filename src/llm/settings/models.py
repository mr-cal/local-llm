"""The pydantic models that define the config.toml schema."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_FILENAME = "config.toml"

# Marks a field as holding a credential. `llm config show` masks every field
# tagged this way, so adding a new secret to a model is enough to keep it out
# of the printed configuration - there is no separate list to keep in sync.
SECRET: dict[str, object] = {"secret": True}


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
    """A single build profile - a named set of cmake flags."""

    name: str
    backend: str | None = None  # shortcut, e.g. "vulkan" → -DGGML_VULKAN=ON
    extra_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_backend(self) -> BuildProfile:
        if self.backend and self.backend not in BACKEND_FLAGS:
            raise ValueError(
                f"Unknown backend '{self.backend}'. Valid options: {', '.join(sorted(BACKEND_FLAGS))}"
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
    monitor: bool = True
    monitor_interval: int = 30
    monitor_retention_days: int = 90


class ModelCost(BaseModel):
    """Per-token cost for a single model.

    Prices are in USD per token.  Defaults are zero because local models
    are free - override for cloud-hosted APIs or when you want cost tracking.
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

    # Generate: python -c "import secrets; print(secrets.token_hex(32))"
    api_key: str = Field(default="", json_schema_extra=SECRET)


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
        """Unique identifier - alias."""
        return self.alias


class ModelsSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    dir: str = "~/models"
    active: str = "qwen2.5-coder-14b-q4"  # alias, not filename
    hf_token: str = Field(default="", json_schema_extra=SECRET)
    entries: list[ModelEntry] = Field(default_factory=list, alias="list")

    @model_validator(mode="after")
    def validate_active(self) -> ModelsSettings:
        # If active is an alias, check it exists in the list (for non-custom models)
        if self.entries and self.active:
            by_alias = {m.alias for m in self.entries}
            if self.active not in by_alias:
                # Allow custom/uncatalogued models referenced by filename
                pass
        return self

    @property
    def models_path(self) -> Path:
        return Path(self.dir).expanduser().resolve()

    @property
    def model_path(self) -> Path:
        """Path to the active model file, resolving alias→filename when possible."""
        # If active is already a filename (custom model without a catalog entry)
        if self.active.endswith(".gguf"):
            return self.models_path / Path(self.active).name
        # Try to resolve via catalog
        entry = self.by_alias(self.active)
        if entry:
            return self.models_path / Path(entry.filename).name
        entry = self.by_filename(self.active)
        if entry:
            return self.models_path / Path(entry.filename).name
        # Fallback: treat active as a filename
        return self.models_path / Path(self.active).name

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
    """GitHub CLI (gh) authentication and git identity settings."""

    # GitHub personal access token for gh CLI auth
    token: str = Field(default="", json_schema_extra=SECRET)
    # Separate PAT used for git push (HTTPS credential)
    git_pat: str = Field(default="", json_schema_extra=SECRET)
    git_username: str = "mr-cal-bot"
    git_email: str = "callahanlovesshopping@gmail.com"

    def is_authenticated(self) -> bool:
        """True when a non-empty token is configured."""
        return bool(self.token.strip())


class ClientSettings(BaseModel):
    """How client tools (opencode, Pi) on this machine connect to the LLM.

    server+client machine: leave server_url empty - defaults to the local
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


class HermesSettings(BaseModel):
    """Configuration for the Hermes agent LXD VM."""

    # LLM backend for the Hermes agent: "local-llm" or "openrouter".
    provider: str = "local-llm"

    # OpenRouter API key (used when provider = "openrouter").
    # https://openrouter.ai/keys
    openrouter_key: str = Field(default="", json_schema_extra=SECRET)

    # Telegram bot token from @BotFather.
    telegram_token: str = Field(default="", json_schema_extra=SECRET)

    # Comma-separated numeric Telegram user IDs allowed to talk to the bot.
    # Get your ID from @userinfobot on Telegram.
    telegram_allowed_users: str = ""

    # GitHub PAT for Hermes GitHub MCP tool (needs repo + read:org scope).
    github_token: str = Field(default="", json_schema_extra=SECRET)

    def has_openrouter(self) -> bool:
        """True when OpenRouter backend is selected with a key."""
        return self.provider == "openrouter" and bool(self.openrouter_key.strip())

    def has_local_llm(self) -> bool:
        """True when local-llm backend is selected."""
        return self.provider == "local-llm"

    def has_telegram(self) -> bool:
        """True when both a bot token and at least one allowed user are configured."""
        return bool(self.telegram_token.strip()) and bool(self.telegram_allowed_users.strip())

    def has_github(self) -> bool:
        """True when a GitHub PAT is configured."""
        return bool(self.github_token.strip())


class Settings(BaseModel):
    server: ServerSettings = Field(default_factory=ServerSettings)
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    proxy: ProxySettings = Field(default_factory=ProxySettings)
    client: ClientSettings = Field(default_factory=ClientSettings)
    lxd: LxdSettings = Field(default_factory=LxdSettings)
    build: BuildConfig = Field(default_factory=BuildConfig)
    github: GitHubSettings = Field(default_factory=GitHubSettings)
    hermes: HermesSettings = Field(default_factory=HermesSettings)

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

    def resolve_llama_bench_bin(self) -> Path | None:
        """Resolve the llama-bench binary path using the same priority as the server bin:

        1. Profile-resolved: ``<build.install_dir>/<profile>/llama-bench``
        2. ``llama-bench`` on PATH (shutil.which fallback)
        """
        import shutil  # noqa: PLC0415

        profile_name = self.server.profile or (
            self.build.active_profile.name if self.build.active_profile else None
        )
        if profile_name and self.build.profiles:
            profile = self.build.get_profile(profile_name)
            if profile:
                candidate = profile.installed_bench_bin(self.build.install_path)
                if candidate.exists():
                    return candidate

        found = shutil.which("llama-bench")
        return Path(found) if found else None

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
