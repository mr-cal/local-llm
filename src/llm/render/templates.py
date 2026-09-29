"""Rendering and installation of the nginx and systemd templates."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from llm.core import proc
from llm.core.console import console
from llm.render.client_configs import _get_lxd_bridge_info
from llm.settings import Settings


def _template_replacements(cfg: Settings) -> dict[str, str]:
    """Build the placeholder → value map for the nginx and systemd templates.

    The systemd unit is the only thing that starts llama-server, so every
    setting that affects its command line must appear here; a missing
    placeholder means the server runs with different options than configured.
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
        "%%LLAMA_SERVER_BIN%%": cfg.resolve_llama_server_bin(),
        "%%MODELS_DIR%%": str(cfg.models_path),
        "%%ACTIVE_MODEL%%": active_filename,
        "%%N_GPU_LAYERS%%": str(cfg.server.n_gpu_layers),
        "%%N_CTX%%": str(cfg.server.n_ctx),
        "%%N_THREADS%%": str(cfg.server.n_threads),
        # Continues the unit's line-continuation style so extra args land on
        # their own lines; empty when no extra args are configured.
        "%%EXTRA_ARGS%%": "".join(f" \\\n    {arg}" for arg in cfg.server.extra_args),
        "%%USER%%": os.environ.get("USER", os.environ.get("LOGNAME", "nobody")),
    }

    # Auto-detect lxdbr0 bridge to allow LXD containers to reach the proxy.
    lxd_bridge_ip, lxd_bridge_subnet = _get_lxd_bridge_info()
    if lxd_bridge_ip and lxd_bridge_subnet:
        console.print(f"  [dim]LXD bridge detected: {lxd_bridge_ip} ({lxd_bridge_subnet})[/dim]")
        replacements["%%LXD_ALLOW_LINE%%"] = f"    allow {lxd_bridge_subnet};\n"
    else:
        replacements["%%LXD_ALLOW_LINE%%"] = ""

    return replacements


def apply_server_configs(cfg: Settings, project_root: Path) -> None:
    """Render nginx/systemd templates and install them.

    Handles template rendering, nginx site install + reload, and systemd
    service install + daemon-reload + enable.
    """
    replacements = _template_replacements(cfg)

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
        console.print("  [yellow]nginx/llm-proxy.conf not found - skipping[/yellow]")
    else:
        if proc.sudo_step(["cp", str(nginx_src), str(nginx_avail)], desc="install conf"):
            if not nginx_enabled.exists():
                proc.sudo_step(["ln", "-sf", str(nginx_avail), str(nginx_enabled)], desc="enable site")
            test = subprocess.run(["sudo", "nginx", "-t"], capture_output=True, text=True)
            if test.returncode != 0:
                console.print(f"  [red]✗[/red]  nginx -t failed:\n{test.stderr.strip()}")
            else:
                console.print("  [green]✓[/green]  nginx -t passed")
                if proc.unit_is_active("nginx"):
                    proc.sudo_step(["systemctl", "reload", "nginx"], desc="reload nginx")
                else:
                    proc.sudo_step(["systemctl", "start", "nginx"], desc="start nginx")

    # ── systemd ───────────────────────────────────────────────────────────
    console.print("\n[bold]systemd[/bold]")
    svc_src = project_root / "systemd" / "llm-server.service"
    svc_dst = Path("/etc/systemd/system/llm-server.service")
    if not svc_src.exists():
        console.print("  [yellow]systemd/llm-server.service not found - skipping[/yellow]")
    else:
        if proc.sudo_step(["cp", str(svc_src), str(svc_dst)], desc="install service"):
            proc.sudo_step(["systemctl", "daemon-reload"], desc="daemon-reload")
            proc.sudo_step(["systemctl", "enable", "llm-server"], desc="enable llm-server")
            if proc.unit_is_active("llm-server"):
                console.print(
                    "  [dim]llm-server is running - restart to pick up changes:[/dim]\n"
                    "    [bold]uv run llm server restart[/bold]"
                )
