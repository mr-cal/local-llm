"""Writing files without leaving a half-written one behind."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_atomic(path: Path, content: str, mode: int | None = None, encoding: str = "utf-8") -> None:
    """Replace *path* with *content* in a single step.

    The content is written to a temporary file in the same directory - so the
    rename stays within one filesystem and is therefore atomic - then moved
    over the destination. A crash or a full disk leaves the previous file
    intact rather than truncated, which matters most for config.toml: a
    truncated config is unrecoverable without a backup.
    """
    path = Path(path)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        else:
            # mkstemp creates 0600; match the permissions the file would have
            # had from a plain write so existing readers are not locked out.
            os.chmod(tmp, 0o644 & ~_umask())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _umask() -> int:
    """Read the process umask without permanently changing it."""
    current = os.umask(0o022)
    os.umask(current)
    return current
