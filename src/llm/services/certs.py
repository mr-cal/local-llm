"""TLS certificate generation and the network details it depends on."""

from __future__ import annotations

import secrets
import subprocess
from pathlib import Path

from llm.core.console import console
from llm.settings import Settings


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

    return secrets.token_hex(32)


def detect_lan_ip() -> str:
    """Auto-detect the machine's LAN IP address.

    Returns the first non-loopback IPv4 address, or an empty string if detection fails.
    """
    result = subprocess.run(
        ["ip", "-4", "addr", "show"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("inet ") and "127.0.0.1" not in line:
            return line.split()[1].split("/")[0]
    return ""
