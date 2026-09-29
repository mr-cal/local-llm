"""Generation of the opencode, pi and omp client configurations.

These are derived entirely from config.toml plus, where available, what the
running server reports about the loaded model.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer
from pydantic import BaseModel, ConfigDict, Field

from llm.core import http
from llm.core.console import console
from llm.settings import Settings

# ── Pydantic models for config builders ──────────────────────────────────────
# These models provide runtime validation at construction time and
# self-document the config schema. The builder functions construct instances
# and serialize with model_dump(mode="json", by_alias=True).


class _OpencodeWatcher(BaseModel):
    ignore: list[str] = [
        ".venv",
        "**/*.pyc",
        "**/__pycache__",
        "**/node_modules",
    ]


class _OpencodeAgentBuild(BaseModel):
    temperature: float = 0.3
    steps: int = 50


class _OpencodeAgentPlan(BaseModel):
    temperature: float = 0.1


class _OpencodeAgent(BaseModel):
    build: _OpencodeAgentBuild = Field(default_factory=_OpencodeAgentBuild)
    plan: _OpencodeAgentPlan = Field(default_factory=_OpencodeAgentPlan)


class _OpencodeModelLimit(BaseModel):
    context: int
    input: int
    output: int


class _OpencodeModel(BaseModel):
    name: str
    limit: _OpencodeModelLimit
    tool_call: bool = True
    options: dict = Field(default_factory=lambda: {"repeat_penalty": 1.2})


class _OpencodeProviderLocalLlm(BaseModel):
    name: str = "Local LLM"
    npm: str = "@ai-sdk/openai-compatible"
    api: str
    options: dict = Field(default_factory=lambda: {"apiKey": "local"})
    models: dict[str, _OpencodeModel] = Field(default_factory=dict)


class _OpencodeConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_field: str = Field(default="https://opencode.ai/config.json", alias="$schema")
    snapshot: bool = True
    watcher: _OpencodeWatcher = Field(default_factory=_OpencodeWatcher)
    permission: str = "allow"
    model: str
    agent: _OpencodeAgent = Field(default_factory=_OpencodeAgent)
    compaction: dict = Field(
        default_factory=lambda: {
            "reserved": 8192,
            "tail_turns": 10,
            "preserve_recent_tokens": 20000,
        }
    )
    provider: dict[str, _OpencodeProviderLocalLlm] = Field(default_factory=dict)


class _PiModelCompat(BaseModel):
    supportsDeveloperRole: bool = False
    supportsReasoningEffort: bool = False
    maxTokensField: str = "max_tokens"


class _PiModel(BaseModel):
    id: str = "local"
    name: str
    contextWindow: int
    maxTokens: int
    cost: dict[str, float] = Field(
        default_factory=lambda: {
            "input": 0.0,
            "output": 0.0,
            "cacheWrite": 0.0,
            "cacheRead": 0.0,
        }
    )


class _PiProviderLocalLlm(BaseModel):
    baseUrl: str
    api: str = "openai-completions"
    apiKey: str
    compat: _PiModelCompat = Field(default_factory=_PiModelCompat)
    models: list[_PiModel] = Field(default_factory=list)


class _PiConfig(BaseModel):
    providers: dict[str, _PiProviderLocalLlm] = Field(default_factory=dict)


class _OmpModelCost(BaseModel):
    input: float = 0.0
    output: float = 0.0
    cacheRead: float = 0.0
    cacheWrite: float = 0.0


class _OmpModel(BaseModel):
    id: str = "local"
    name: str
    reasoning: bool = True
    input: list[str] = Field(default_factory=lambda: ["text"])
    contextWindow: int = 4096
    maxTokens: int = 8192
    cost: _OmpModelCost = Field(default_factory=_OmpModelCost)


class _OmpProviderLocalLlm(BaseModel):
    baseUrl: str
    apiKey: str
    api: str = "openai-completions"
    auth: str = "apiKey"
    models: list[_OmpModel] = Field(default_factory=list)


class _OmpConfig(BaseModel):
    providers: dict[str, _OmpProviderLocalLlm] = Field(default_factory=dict)


def _build_opencode_config(cfg: Settings) -> dict:  # type: ignore[type-arg]
    """Build the opencode provider config dict from current settings."""
    # Resolve model info: server-reported → catalog → config defaults
    display_name, context_window, max_output = _resolve_model_info(cfg)

    # Use a stable generic key so the opencode config never needs updating when
    # switching models. llama-server ignores the model name in chat completion
    # requests and serves whatever is currently loaded.
    model_key = "local"
    api_key = cfg.client_api_key or "local"

    # Without limit.input set, opencode ignores compaction.reserved and fires at
    # context - max_output (= 32K for Qwen3). Setting limit.input = n_ctx unlocks
    # the reserved path: usable = n_ctx - reserved, so compaction fires at ~57K.
    return _OpencodeConfig(
        model=f"local-llm/{model_key}",
        provider={
            "local-llm": _OpencodeProviderLocalLlm(
                api=cfg.client_url,
                options={"apiKey": api_key},
                models={
                    model_key: _OpencodeModel(
                        name=display_name,
                        limit=_OpencodeModelLimit(
                            context=context_window,
                            input=context_window,
                            output=max_output,
                        ),
                    )
                },
            )
        },
    ).model_dump(mode="json", by_alias=True)


_OPENCODE_SCHEMA_URL = "https://opencode.ai/config.json"
_OPENCODE_CONFIG_PATH = Path("~/.config/opencode/config.json")
_PI_CONFIG_PATH = Path("~/.pi/agent/models.json")


def _get_server_model_info(cfg: Settings) -> dict | None:
    """Query llama.cpp's /model_info endpoint for server-reported model metadata.

    Returns a dict with keys like ``model_name``, ``ctx_size``, ``n_embd``, etc.
    Returns ``None`` if the server is unreachable or the endpoint is unavailable
    (e.g. old llama.cpp version without this endpoint).
    """
    return http.get_json(f"{cfg.internal_url}/model_info")


def _resolve_model_info(
    cfg: Settings,
) -> tuple[str, int, int]:
    """Resolve model name, context window, and max output using server-reported values.

    Resolution priority:
    1. Query the live server via ``/model_info``
    2. Look up in the local catalog (``KNOWN_MODELS`` or ``[[models.list]]``)
    3. Fall back to config values and defaults

    Returns ``(display_name, context_window, max_output)``.
    """
    from llm.models import KNOWN_MODELS  # noqa: PLC0415

    # ── 1. Server-reported (most accurate) ─────────────────────────────────
    server_info = _get_server_model_info(cfg)

    if server_info:
        model_name = server_info.get("model_name", "")
        ctx_size = server_info.get("ctx_size", 0) or server_info.get("n_ctx", 0)
        # Try to get n_ctx from the server info as well
        if not ctx_size:
            ctx_size = cfg.server.n_ctx
        # Use ctx_size for max output heuristic (server reports ctx_size, not n_ctx)
        server_ctx = ctx_size or server_info.get("n_ctx", 0)
        max_output = (server_ctx // 8) if server_ctx >= 4096 else 8192
        if max_output < 512:
            max_output = 8192
        return model_name, ctx_size, max_output

    # ── 2. Local catalog ──────────────────────────────────────────────────
    active = cfg.models.active
    entry = cfg.models.by_alias(active) or cfg.models.by_filename(active)
    if entry is None and cfg.models.has_catalog is False:
        entry = next((m for m in KNOWN_MODELS if m.filename == active), None)

    display_name = entry.alias if entry else active
    max_output = entry.max_output if entry else 8192
    return display_name, cfg.server.n_ctx, max_output


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
    # Resolve model info: server-reported → catalog → config defaults
    display_name, context_window, max_output = _resolve_model_info(cfg)

    # Resolve catalog entry for cost lookup (server_info may not have costs)
    active = cfg.models.active
    entry = cfg.models.by_alias(active) or cfg.models.by_filename(active)
    if entry is None and cfg.models.has_catalog is False:
        from llm.models import KNOWN_MODELS  # noqa: PLC0415

        entry = next((m for m in KNOWN_MODELS if m.filename == active), None)

    # apiKey is required by pi even for unauthenticated local servers.
    # Fall back to "local" so the field is always present.
    api_key = cfg.client_api_key or "local"

    return _PiConfig(
        providers={
            "local-llm": _PiProviderLocalLlm(
                baseUrl=cfg.client_url,
                apiKey=api_key,
                models=[
                    _PiModel(
                        name=display_name,
                        contextWindow=context_window,
                        maxTokens=max_output,
                        cost=(
                            entry.cost.to_cost_dict()
                            if entry
                            else {
                                "input": 0.0,
                                "output": 0.0,
                                "cacheWrite": 0.0,
                                "cacheRead": 0.0,
                            }
                        ),
                    )
                ],
            )
        },
    ).model_dump(mode="json", by_alias=True)


def _build_omp_config_for_container(cfg: Settings, server_host: str) -> dict:  # type: ignore[type-arg]
    """Build the oh-my-pi models.yml config dict for use INSIDE an LXD container.

    The container cannot reach the host's llama-server at 127.0.0.1, so it
    connects via the nginx TLS proxy using *server_host* (typically the
    ``local-llm`` hostname, which is already in the cert's SubjectAltName).
    """
    # Resolve model info: server-reported → catalog → config defaults
    display_name, context_window, max_output = _resolve_model_info(cfg)

    # Resolve catalog entry for cost lookup (server_info may not have costs)
    active = cfg.models.active
    entry = cfg.models.by_alias(active) or cfg.models.by_filename(active)
    if entry is None and cfg.models.has_catalog is False:
        from llm.models import KNOWN_MODELS  # noqa: PLC0415

        entry = next((m for m in KNOWN_MODELS if m.filename == active), None)

    api_key = cfg.client_api_key or "local"

    base_url = f"https://{server_host}:{cfg.proxy.port}/v1"

    return _OmpConfig(
        providers={
            "local-llm": _OmpProviderLocalLlm(
                baseUrl=base_url,
                apiKey=api_key,
                models=[
                    _OmpModel(
                        name=display_name,
                        contextWindow=context_window,
                        maxTokens=max_output,
                        cost=_OmpModelCost(
                            input=entry.cost.input if entry else 0.0,
                            output=entry.cost.output if entry else 0.0,
                            cacheRead=entry.cost.cache_read if entry else 0.0,
                            cacheWrite=entry.cost.cache_write if entry else 0.0,
                        ),
                    )
                ],
            )
        },
    ).model_dump(mode="json", by_alias=True)


def _build_pi_config_for_container(cfg: Settings, server_host: str) -> dict:  # type: ignore[type-arg]
    """Build the pi-harness models.json config dict for use INSIDE an LXD container.

    The container cannot reach the host's llama-server at 127.0.0.1, so it
    connects via the nginx TLS proxy using *server_host* (typically the
    ``local-llm`` hostname, which is already in the cert's SubjectAltName).
    """
    # Resolve model info: server-reported → catalog → config defaults
    display_name, context_window, max_output = _resolve_model_info(cfg)

    # Resolve catalog entry for cost lookup (server_info may not have costs)
    active = cfg.models.active
    entry = cfg.models.by_alias(active) or cfg.models.by_filename(active)
    if entry is None and cfg.models.has_catalog is False:
        from llm.models import KNOWN_MODELS  # noqa: PLC0415

        entry = next((m for m in KNOWN_MODELS if m.filename == active), None)

    api_key = cfg.client_api_key or "local"

    # Container connects via the nginx TLS proxy using the provided host.
    # Using the 'local-llm' hostname (present in the cert's SAN) avoids
    # TLS verification failures that would occur with a raw IP address.
    base_url = f"https://{server_host}:{cfg.proxy.port}/v1"

    return _PiConfig(
        providers={
            "local-llm": _PiProviderLocalLlm(
                baseUrl=base_url,
                apiKey=api_key,
                models=[
                    _PiModel(
                        name=display_name,
                        contextWindow=context_window,
                        maxTokens=max_output,
                        cost=(
                            entry.cost.to_cost_dict()
                            if entry
                            else {
                                "input": 0.0,
                                "output": 0.0,
                                "cacheWrite": 0.0,
                                "cacheRead": 0.0,
                            }
                        ),
                    )
                ],
            )
        },
    ).model_dump(mode="json", by_alias=True)


def _build_opencode_config_for_container(cfg: Settings, server_host: str) -> dict:  # type: ignore[type-arg]
    """Build the opencode config dict for use INSIDE an LXD container.

    Like :func:`_build_opencode_config`, but connects via the nginx TLS proxy
    at ``https://<server_host>:<port>/v1`` instead of localhost.
    """
    # Resolve model info: server-reported → catalog → config defaults
    display_name, context_window, max_output = _resolve_model_info(cfg)

    model_key = "local"
    api_key = cfg.client_api_key or "local"
    base_url = f"https://{server_host}:{cfg.proxy.port}/v1"

    return _OpencodeConfig(
        model=f"local-llm/{model_key}",
        provider={
            "local-llm": _OpencodeProviderLocalLlm(
                api=base_url,
                options={"apiKey": api_key},
                models={
                    model_key: _OpencodeModel(
                        name=display_name,
                        limit=_OpencodeModelLimit(
                            context=context_window,
                            input=context_window,
                            output=max_output,
                        ),
                    )
                },
            )
        },
    ).model_dump(mode="json", by_alias=True)


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
        return [f"⚠ Could not fetch schema ({exc}) - skipping validation"]

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
            # Skip residual model-enum errors - local providers are never in the
            # cloud-provider enum that models.dev maintains.
            if path == "model":
                continue
            errors.append(f"{path}: {err.message}")
    return errors


def apply_client_configs(cfg: Settings) -> None:
    """Write opencode and pi configs for the host machine.

    Writes ~/.config/opencode/config.json and ~/.pi/agent/models.json.
    Validates the opencode config against the live schema before writing.
    """

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
        console.print("\n[red]opencode config has schema errors - not written.[/red]")
        raise typer.Exit(1)
    else:
        console.print("  [green]✓[/green] Schema valid")

    opencode_path.write_text(json.dumps(opencode_cfg, indent=2) + "\n")
    console.print(f"[green]Rendered[/green] {opencode_path}")
