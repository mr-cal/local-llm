"""Client commands: setup (host + container), check, show, refresh, crafts, list."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from llm.config import CONFIG_FILENAME, find_config, load_config

app = typer.Typer(help="Client setup and management.", no_args_is_help=True)
console = Console()


# ── client setup ──────────────────────────────────────────────────────────────


def _setup_host_client() -> None:
    """Set up the current machine as a client (opencode, pi, shell env)."""
    from llm.config import (  # noqa: PLC0415
        apply_client_configs,
        configure_shell_env_host,
    )

    config_path = find_config()
    if not config_path.exists():
        console.print(
            f"[red]ERROR:[/red] {CONFIG_FILENAME} not found.\n"
            "  On a client-only machine, run [bold]uv run llm config init[/bold] first.\n"
            "  On the server machine, run [bold]uv run llm server setup[/bold] first."
        )
        raise typer.Exit(1)

    cfg = load_config()
    console.print("\n[bold cyan]═══ local-llm client setup ═══[/bold cyan]\n")

    # Client configs (opencode, pi)
    apply_client_configs(cfg)

    # Shell env vars - prefer explicit client settings, fall back to proxy settings.
    base_url = cfg.client.server_url or f"https://{cfg.proxy.lan_ip}:{cfg.proxy.port}/v1"
    cert_path = cfg.client.cert_path or cfg.proxy.cert_path
    api_key = cfg.auth.api_key
    actions = configure_shell_env_host(base_url, api_key, cert_path)
    for action in actions:
        console.print(f"  [green]✓[/green] {action}")

    console.print("\n[bold green]✓ Host client setup complete![/bold green]")
    console.print(
        "\n  Verify connectivity: [bold]uv run llm client check[/bold]"
        "\n  Restart your shell to pick up env vars, or run:"
        "\n    source ~/.config/local-llm/env"
    )


def _setup_container_client(
    container_name: str,
    *,
    recreate: bool = False,
    lxd_vm: bool = False,
) -> None:
    """Create an LXD container and fully configure it as a client."""
    from llm.config import _build_opencode_config_for_container  # noqa: PLC0415
    from llm.lxd import (  # noqa: PLC0415
        CONTAINER_GID,
        CONTAINER_HOME,
        CONTAINER_UID,
        HOST_GID,
        HOST_UID,
        LOCAL_LLM_VERSION,
        _cexec,
        create_and_setup,
        load_lxd_settings,
    )

    config_path = find_config()
    if not config_path.exists():
        console.print(
            f"[red]ERROR:[/red] {CONFIG_FILENAME} not found.\n"
            "  Run [bold]uv run llm server setup[/bold] first."
        )
        raise typer.Exit(1)

    cfg = load_config()
    mounts, _ = load_lxd_settings()

    console.print(f"\n[bold cyan]═══ Setting up container: {container_name} ═══[/bold cyan]\n")

    # Read TLS cert from host
    cert_pem: str | None = None
    cert_file = Path(cfg.proxy.cert_path)
    if cert_file.exists():
        cert_pem = cert_file.read_text()
    else:
        console.print(
            f"[yellow]Warning:[/yellow] Cert not found at {cert_file}.\n"
            "  Container setup will skip cert installation.\n"
            "  Generate it first: [bold]uv run llm server setup[/bold]"
        )

    # Create and configure the container (LXD, packages, mounts, pi)
    create_and_setup(
        container_name,
        mounts=mounts,
        recreate=recreate,
        lxd_vm=lxd_vm,
        cert_pem=cert_pem,
    )

    # Set up opencode config inside the container (separate from host bind mount)
    effective_uid = HOST_UID if lxd_vm else CONTAINER_UID
    effective_gid = HOST_GID if lxd_vm else CONTAINER_GID

    console.print("\n[bold]Setting up opencode config in container...[/bold]")
    opencode_cfg = _build_opencode_config_for_container(cfg, "local-llm")
    opencode_json = json.dumps(opencode_cfg, indent=2) + "\n"
    opencode_path = f"{CONTAINER_HOME}/.config/opencode/config.json"
    subprocess.run(
        _cexec(container_name, effective_uid, effective_gid, "bash", "-c", f"cat > {opencode_path}"),
        input=opencode_json.encode(),
        check=True,
    )
    console.print(f"  [green]✓[/green] Wrote opencode config to {opencode_path}")

    # Authenticate gh CLI inside the container
    gh_token = cfg.github.token if cfg.github.is_authenticated() else ""
    from llm.lxd import setup_gh_auth_in_container  # noqa: PLC0415

    setup_gh_auth_in_container(
        container_name,
        gh_token,
        effective_uid=effective_uid,
        effective_gid=effective_gid,
    )

    # Tag with version
    subprocess.run(
        ["lxc", "config", "set", container_name, f"user.local-llm-version={LOCAL_LLM_VERSION}"],
        check=True,
    )

    console.print(f"\n[bold green]✓ Container '{container_name}' is ready![/bold green]")
    console.print(
        f"\n  Enter the container:  [bold]lxc exec {container_name} -- su -l $USER[/bold]"
        f"\n  Set up craft dirs:    [bold]uv run llm client crafts {container_name}[/bold]"
    )


@app.command("setup")
def setup(
    container: Annotated[
        str | None,
        typer.Option(
            "--container",
            "-c",
            help="Create an LXD container with this name and set it up as a client.",
        ),
    ] = None,
    recreate: Annotated[
        bool,
        typer.Option("--recreate", help="Delete and recreate the container if it already exists."),
    ] = False,
    lxd_vm: Annotated[
        bool,
        typer.Option("--lxd-vm", help="Create a full LXD VM instead of a container."),
    ] = False,
) -> None:
    """Set up a client (either the current host or an LXD container).

    Without --container: configures opencode, pi, and shell env vars on this machine.
    With --container: creates an LXD container and configures it as a client.
    """
    if container:
        _setup_container_client(container, recreate=recreate, lxd_vm=lxd_vm)
    else:
        _setup_host_client()


# ── client check ──────────────────────────────────────────────────────────────


@app.command("check")
def check() -> None:
    """Test connectivity to the configured LLM server.

    Hits the health endpoint and reports connection status, TLS, auth,
    and model info.
    """
    import time  # noqa: PLC0415

    import httpx  # noqa: PLC0415

    config_path = find_config()
    if not config_path.exists():
        console.print(
            f"[red]ERROR:[/red] {CONFIG_FILENAME} not found.\n"
            "  Run [bold]uv run llm server setup[/bold] first."
        )
        raise typer.Exit(1)

    cfg = load_config()
    base_url = f"https://{cfg.proxy.lan_ip}:{cfg.proxy.port}"
    health_url = f"{base_url}/health"
    cert_path = cfg.proxy.cert_path

    console.print(f"\n[bold]Checking server at {base_url}[/bold]\n")

    # Health check
    try:
        start = time.monotonic()
        resp = httpx.get(
            health_url,
            headers={"Authorization": f"Bearer {cfg.auth.api_key}"},
            verify=cert_path if Path(cert_path).exists() else False,
            timeout=10,
        )
        latency = (time.monotonic() - start) * 1000
        console.print(f"  [green]✓[/green] Health:  {resp.status_code}  ({latency:.0f}ms)")
    except httpx.ConnectError as e:
        console.print(f"  [red]✗[/red] Connection failed: {e}")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"  [red]✗[/red] Error: {e}")
        raise typer.Exit(1) from None

    # TLS check
    if Path(cert_path).exists():
        console.print(f"  [green]✓[/green] TLS:     cert at {cert_path}")
    else:
        console.print(f"  [yellow]⚠[/yellow] TLS:     cert not found at {cert_path}")

    # Model info
    try:
        models_resp = httpx.get(
            f"{base_url}/v1/models",
            headers={"Authorization": f"Bearer {cfg.auth.api_key}"},
            verify=cert_path if Path(cert_path).exists() else False,
            timeout=5,
        )
        if models_resp.status_code == 200:
            data = models_resp.json()
            models = data.get("data", [])
            if models:
                model_id = models[0].get("id", "unknown")
                console.print(f"  [green]✓[/green] Model:   {model_id}")
            else:
                console.print("  [yellow]⚠[/yellow] Model:   no models loaded")
        else:
            console.print(f"  [yellow]⚠[/yellow] Model:   HTTP {models_resp.status_code}")
    except Exception:
        console.print("  [yellow]⚠[/yellow] Model:   could not query")

    console.print("\n[bold green]✓ Server is reachable![/bold green]")


# ── client show ───────────────────────────────────────────────────────────────


@app.command("show")
def show() -> None:
    """Print current client connection info (URL, model, cert)."""
    config_path = find_config()
    if not config_path.exists():
        console.print(f"[yellow]{CONFIG_FILENAME} not found.[/yellow]")
        raise typer.Exit(1)

    cfg = load_config()
    base_url = f"https://{cfg.proxy.lan_ip}:{cfg.proxy.port}/v1"
    cert_path = cfg.proxy.cert_path
    cert_exists = Path(cert_path).exists()

    console.print(f"  URL:    [cyan]{base_url}[/cyan]")
    console.print(f"  Model:  {cfg.models.active}")
    key_display = f"{cfg.auth.api_key[:16]}…" if cfg.auth.api_key else "[yellow]not set[/yellow]"
    console.print(f"  Key:    {key_display}")
    cert_status = "[green]exists[/green]" if cert_exists else "[red]missing[/red]"
    console.print(f"  Cert:   {cert_path}  {cert_status}")


# ── client list ───────────────────────────────────────────────────────────────


@app.command("list")
def list_containers() -> None:
    """List all managed LXD containers with their status and version."""
    from llm.lxd import _list_managed_containers  # noqa: PLC0415

    managed = _list_managed_containers()
    if not managed:
        console.print(
            "[yellow]No managed containers found.[/yellow]\n"
            "  Create one with: [bold]uv run llm client setup --container <name>[/bold]"
        )
        return

    console.print(f"\n[bold]Managed containers ({len(managed)}):[/bold]\n")
    for name in managed:
        # Get version from metadata
        r = subprocess.run(
            ["lxc", "config", "get", name, "user.local-llm-version"],
            capture_output=True,
            text=True,
        )
        version = r.stdout.strip() or "unknown"

        # Get status
        r2 = subprocess.run(
            ["lxc", "list", name, "--format=json"],
            capture_output=True,
            text=True,
        )
        status = "unknown"
        try:
            data = json.loads(r2.stdout)
            if data:
                status = data[0].get("status", "unknown")
        except (json.JSONDecodeError, IndexError):
            pass

        status_color = "green" if status == "Running" else "yellow"
        console.print(f"  [{status_color}]●[/{status_color}] {name}  v{version}  ({status})")


# ── client refresh ────────────────────────────────────────────────────────────


@app.command("refresh")
def refresh(
    container: Annotated[
        str | None,
        typer.Argument(help="Container name. Omit to refresh all managed containers."),
    ] = None,
    lxd_vm: Annotated[
        bool,
        typer.Option("--lxd-vm", help="Force VM mode for the named container."),
    ] = False,
) -> None:
    """Update packages and re-apply client config in managed LXD container(s).

    Without arguments, refreshes all containers tagged as managed.
    """
    from llm.lxd import refresh_containers  # noqa: PLC0415

    # Read cert from host
    cert_pem: str | None = None
    config_path = find_config()
    if config_path.exists():
        cfg = load_config()
        cert_file = Path(cfg.proxy.cert_path)
        if cert_file.exists():
            cert_pem = cert_file.read_text()

    try:
        refresh_containers(container, cert_pem=cert_pem, lxd_vm=lxd_vm)
    except RuntimeError as e:
        console.print(f"[red]ERROR:[/red] {escape(str(e))}")
        raise typer.Exit(1) from None


# ── client crafts ─────────────────────────────────────────────────────────────


@app.command("crafts")
def crafts(
    container: Annotated[
        str,
        typer.Argument(help="Container name to run 'make setup' in."),
    ],
    lxd_vm: Annotated[
        bool,
        typer.Option("--lxd-vm", help="Force VM mode."),
    ] = False,
) -> None:
    """Run 'make setup' in all configured craft directories inside a container."""
    from llm.lxd import do_setup_crafts, load_lxd_settings  # noqa: PLC0415

    _, craft_dirs = load_lxd_settings()

    try:
        do_setup_crafts(container, craft_dirs, lxd_vm=lxd_vm)
    except RuntimeError as e:
        console.print(f"[red]ERROR:[/red] {escape(str(e))}")
        raise typer.Exit(1) from None
