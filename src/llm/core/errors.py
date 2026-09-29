"""The error type command code raises to abort with a message.

Commands should raise LlmError rather than calling typer.Exit deep inside
helpers: it keeps exit-code policy at the CLI boundary and lets helpers stay
usable from other helpers.
"""

from __future__ import annotations


class LlmError(Exception):
    """A failure that should be reported to the user, not as a traceback."""
