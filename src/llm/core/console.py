"""The single Console shared by every command module.

Each module used to build its own Console(), which meant rich's terminal
detection, width and colour settings were resolved nine separate times and
tests had to patch each one individually.
"""

from __future__ import annotations

from rich.console import Console

console = Console()
