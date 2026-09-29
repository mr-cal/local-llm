"""The LXD execution layer: running commands in and against containers.

Everything that shells out to ``lxc`` goes through here, so command echoing,
secret redaction and the container's uid/gid/HOME conventions are defined in
exactly one place.
"""

from __future__ import annotations

import ipaddress
import json
import os
import string
import subprocess
import time

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm.core.console import console
from llm.core.errors import LlmError
from llm.core.proc import redact

LOCAL_LLM_VERSION = 1

CONTAINER_PREFIX = "craft-llm"
HOST_UID = os.getuid()
HOST_GID = os.getgid()
HOST_HOME = os.path.expanduser("~")
HOST_USER = os.path.basename(os.path.normpath(HOST_HOME))

# The container user is renamed to match the host, so paths are identical.
CONTAINER_USER = HOST_USER
CONTAINER_UID = 1000
CONTAINER_GID = 1000
CONTAINER_HOME = HOST_HOME

_DEFAULT_MOUNTS: list[tuple[str, str, str]] = [
    ("agents", f"{HOST_HOME}/.agents", f"{CONTAINER_HOME}/.agents"),
    ("github", f"{HOST_HOME}/.github", f"{CONTAINER_HOME}/.github"),
    ("dev", f"{HOST_HOME}/dev", f"{CONTAINER_HOME}/dev"),
    ("opencode-config", f"{HOST_HOME}/.config/opencode", f"{CONTAINER_HOME}/.config/opencode"),
]

LSP_CONFIG_PATH = f"{CONTAINER_HOME}/.copilot/lsp-config.json"

PYLSP_LSP_CONFIG = {
    "lspServers": {
        "python": {
            "command": "pylsp",
            "args": [],
            "fileExtensions": {".py": "python"},
        }
    }
}


_MANAGED_TAG = "user.local-llm-managed"
_KIND_TAG = "user.local-llm-kind"

# Managed VMs come in two flavours with incompatible provisioning. Dev client
# VMs are refreshed by ``llm client refresh``; the Hermes agent VM must not be,
# because reconfiguring it as a dev client would overwrite its agent setup.
KIND_CLIENT = "client"
KIND_HERMES = "hermes"

# The Hermes VM always uses this reserved name.
HERMES_CONTAINER_NAME = "hermes"


# ── Helper functions ─────────────────────────────────────────────────────────


def _cexec(container: str, uid: int, gid: int, *cmd: str) -> list[str]:
    """Build an ``lxc exec`` command running *cmd* as uid/gid inside *container*."""
    return [
        "lxc",
        "exec",
        container,
        f"--user={uid}",
        f"--group={gid}",
        f"--env=HOME={CONTAINER_HOME}",
        "--",
        *cmd,
    ]


# Credential values registered here are scrubbed from every echoed command and
# captured output. Registration is belt-and-braces: secrets should reach the
# container over stdin rather than in argv, but anything that does slip into a
# command line must not also be printed to the terminal or CI logs.
# A permissive but shell-safe hostname: letters, digits, dots and hyphens only.
# Anything outside this set (quotes, spaces, ``;``, ``$(...)``) could break out
# of the single-quoted /etc/hosts command below.
_HOSTNAME_CHARS = set(string.ascii_letters + string.digits + ".-")


def validate_host(value: str) -> str:
    """Return *value* if it is a usable IP address or hostname, else raise."""
    try:
        ipaddress.ip_address(value)
    except ValueError:
        if not value or not set(value) <= _HOSTNAME_CHARS:
            raise LlmError(
                f"{value!r} is not a valid IP address or hostname; "
                "check client.server_url and proxy.lan_ip in config.toml"
            ) from None
    return value


def add_hosts_entry(container: str, address: str, hostname: str) -> None:
    """Idempotently map *hostname* to *address* in the container's /etc/hosts."""
    address = validate_host(address)
    console.print(f"  Adding /etc/hosts entry: {address} {hostname}...")
    entry = f"{address} {hostname}"
    subprocess.run(
        [
            "lxc",
            "exec",
            container,
            "--",
            "bash",
            "-c",
            f"grep -qxF '{entry}' /etc/hosts || echo '{entry}' >> /etc/hosts",
        ],
        check=True,
    )


def run(cmd, desc: str | None = None, **kwargs):
    console.print(f"  $ {redact(' '.join(str(a) for a in cmd))}")
    try:
        result = subprocess.run(cmd, check=True, **kwargs)
    except subprocess.CalledProcessError as e:
        # When callers opt into capture_output, echo it back since it wasn't
        # streamed live to the terminal.
        if e.stdout:
            console.print(redact(e.stdout))
        if e.stderr:
            console.print(redact(e.stderr))
        label = desc or " ".join(str(a) for a in cmd[:3])
        raise subprocess.CalledProcessError(e.returncode, e.cmd, e.output, e.stderr) from Exception(
            f"Command failed ({label}): exit {e.returncode}"
        )
    if kwargs.get("capture_output") and result.stdout:
        console.print(redact(result.stdout))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((subprocess.CalledProcessError, OSError)),
    reraise=True,
)
def run_with_retry(cmd, desc: str | None = None, **kwargs):
    """Run a command with automatic retry on transient failures.

    Retries up to 3 times with exponential backoff (2s → 4s → 8s) on
    ``subprocess.CalledProcessError`` or ``OSError`` (network timeout,
    LXD daemon busy, etc.). Intended for network-dependent operations
    like ``apt-get`` updates, ``curl``-based installs, and ``lxc launch``.
    """
    run(cmd, desc=desc, **kwargs)


def run_capture(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def wait_for_container(container, timeout=90):
    console.print(f"  Waiting for {container} to be ready...", end="", highlight=False)
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = subprocess.run(
            ["lxc", "exec", container, "--", "true"],
            capture_output=True,
        )
        if r.returncode == 0:
            console.print(" ready.")
            break
        console.print(".", end="", highlight=False)
        time.sleep(2)
    else:
        console.print()
        console.print(f"[red]ERROR:[/red] {container} did not become ready within {timeout}s.")
        raise RuntimeError(f"{container} did not become ready within {timeout}s.")

    console.print("  Waiting for cloud-init...", end="", highlight=False)
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = subprocess.run(
            ["lxc", "exec", container, "--", "cloud-init", "status", "--format=json"],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            try:
                if json.loads(r.stdout).get("status") == "done":
                    console.print(" done.")
                    return
            except (json.JSONDecodeError, AttributeError):
                pass
        console.print(".", end="", highlight=False)
        time.sleep(2)
    console.print()
    console.print(f"[red]ERROR:[/red] cloud-init did not finish within {timeout}s.")
    raise RuntimeError(f"cloud-init did not finish within {timeout}s.")


def container_exists(container):
    r = run_capture(["lxc", "list", container, "--format=json"])
    if r.returncode != 0:
        return False
    return any(c["name"] == container for c in json.loads(r.stdout))


# ── Tag / list helpers ──────────────────────────────────────────────────────


def _tag_as_managed(container: str, kind: str = KIND_CLIENT) -> None:
    """Tag *container* as managed so it is discovered by ``llm client refresh``."""
    run(["lxc", "config", "set", container, f"{_MANAGED_TAG}=true"])
    run(["lxc", "config", "set", container, f"{_KIND_TAG}={kind}"])


def _container_kind(instance: dict) -> str:
    """Return the managed kind of an LXD instance from ``lxc list --format=json``.

    VMs provisioned before the kind tag existed carry no kind. The Hermes VM
    has a reserved name, so it can still be told apart from a dev client.
    """
    kind = instance.get("config", {}).get(_KIND_TAG)
    if kind:
        return kind
    return KIND_HERMES if instance.get("name") == HERMES_CONTAINER_NAME else KIND_CLIENT


def get_container_kind(container: str) -> str:
    """Return the managed kind of *container* (``client`` or ``hermes``)."""
    r = run_capture(["lxc", "list", container, "--format=json"])
    if r.returncode != 0:
        return KIND_CLIENT
    for inst in json.loads(r.stdout):
        if inst.get("name") == container:
            return _container_kind(inst)
    return KIND_CLIENT


def _list_managed_containers(kind: str | None = KIND_CLIENT) -> list[str]:
    """Return names of running LXD instances managed by this tool.

    Args:
        kind: Restrict to one kind of VM, or ``None`` for every managed VM.
    """
    r = run_capture(["lxc", "list", "--format=json"])
    if r.returncode != 0:
        return []
    instances = json.loads(r.stdout)
    return [
        inst["name"]
        for inst in instances
        if inst.get("config", {}).get(_MANAGED_TAG) == "true"
        and inst.get("status") == "Running"
        and (kind is None or _container_kind(inst) == kind)
    ]


def _run_capture(container: str, *cmd: str) -> subprocess.CompletedProcess[str]:
    """Run a command inside the container and return the result."""
    return subprocess.run(
        ["lxc", "exec", container, "--", *cmd],
        capture_output=True,
        text=True,
    )


def mkdir_p(container: str, path: str, uid: int = CONTAINER_UID, gid: int = CONTAINER_GID) -> None:
    """Create *path* and its parents inside the container."""
    subprocess.run(_cexec(container, uid, gid, "mkdir", "-p", path), check=True)


def write_file(
    container: str,
    path: str,
    content: str,
    uid: int = CONTAINER_UID,
    gid: int = CONTAINER_GID,
    parents: bool = True,
    mode: str | None = None,
) -> None:
    """Write *content* to *path* inside the container.

    The content travels over stdin rather than the command line, so it is never
    echoed, never shell-quoted and may contain arbitrary bytes - which matters
    because some of these files hold credentials.
    """
    if parents:
        parent = path.rsplit("/", 1)[0]
        if parent:
            mkdir_p(container, parent, uid, gid)
    script = f"cat > {path}"
    if mode:
        script += f" && chmod {mode} {path}"
    subprocess.run(
        _cexec(container, uid, gid, "bash", "-c", script),
        input=content.encode(),
        check=True,
    )
