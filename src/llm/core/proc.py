"""The one place this project shells out to other programs.

Centralising subprocess use gives a single definition of three policies that
were previously spelled differently in every module: how the terminal is
restored after an interrupted sudo prompt, how credentials are kept out of
echoed commands, and what a non-zero return code means.
"""

from __future__ import annotations

import subprocess

from llm.core.console import console

# ── secret redaction ──────────────────────────────────────────────────────────

_SECRET_VALUES: set[str] = set()

# Shorter values are too likely to collide with ordinary command text (and are
# not plausible credentials), so redacting them would corrupt the output.
_MIN_REDACTABLE_LEN = 8


def register_secrets(*values: str | None) -> None:
    """Register credential values to scrub from echoed commands and output."""
    for value in values:
        if value and len(value) >= _MIN_REDACTABLE_LEN:
            _SECRET_VALUES.add(value)


def redact(text: str) -> str:
    """Replace every registered credential in *text* with a placeholder."""
    for secret in _SECRET_VALUES:
        text = text.replace(secret, "***")
    return text


# ── running commands ──────────────────────────────────────────────────────────


def _restore_terminal() -> None:
    """Undo the terminal echo suppression a killed sudo prompt leaves behind.

    sudo disables echo while reading the password; if this process dies before
    sudo restores it, the user's shell stays unusable until `stty sane` runs.
    """
    subprocess.run(["stty", "sane"], check=False)


def run(
    cmd: list[str],
    *,
    check: bool = False,
    capture: bool = True,
    **kwargs,
) -> subprocess.CompletedProcess[str]:
    """Run *cmd*, always as an argv list, never through a shell.

    Interrupting the command restores the terminal before re-raising, so a
    Ctrl+C at a sudo prompt cannot leave the shell without echo.
    """
    try:
        return subprocess.run(cmd, check=check, capture_output=capture, text=True, **kwargs)
    except KeyboardInterrupt:
        _restore_terminal()
        raise


def succeeded(cmd: list[str], **kwargs) -> bool:
    """True when *cmd* exits zero. Output is captured and discarded."""
    return run(cmd, **kwargs).returncode == 0


# ── sudo ──────────────────────────────────────────────────────────────────────


def sudo(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    """Run *cmd* under sudo. Pass the command without a leading "sudo"."""
    return run(["sudo", *cmd], **kwargs)


def sudo_step(cmd: list[str], desc: str) -> bool:
    """Run a sudo command as a reported setup step. Returns True on success."""
    console.print(f"  [dim]$ sudo {' '.join(cmd)}[/dim]")
    result = sudo(cmd)
    if result.returncode != 0:
        msg = (result.stderr or result.stdout).strip()
        console.print(f"  [red]✗[/red]  {desc}" + (f": {msg}" if msg else ""))
        return False
    console.print(f"  [green]✓[/green]  {desc}")
    return True


def ensure_sudo() -> None:
    """Prompt for the sudo password up front, before any other work.

    Without this the prompt appears only once a privileged step is reached -
    often after a long wait - which is surprising. Authenticating first caches
    the credential for the rest of the command.
    """
    run(["sudo", "-v"], capture=False)


# ── systemd ───────────────────────────────────────────────────────────────────


def unit_is_active(unit: str) -> bool:
    """True when systemd reports *unit* as active."""
    return run(["systemctl", "is-active", unit]).stdout.strip() == "active"
