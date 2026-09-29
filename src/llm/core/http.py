"""Shared HTTP probes against the llama-server API.

Readiness polling was written three times with three different timeouts and
three different exception lists; a probe that treats a TLS error as "still
starting" hides real failures, so the policy lives here instead.
"""

from __future__ import annotations

import time
from typing import Any

import httpx


def is_healthy(url: str, timeout: float = 2.0) -> bool:
    """True when *url* answers 200. Any failure means "not ready"."""
    try:
        return httpx.get(url, timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def wait_until_healthy(
    url: str,
    timeout: float,
    interval: float = 0.5,
    probe_timeout: float = 2.0,
    sleep: Any = time.sleep,
) -> bool:
    """Poll *url* until it answers 200 or *timeout* seconds elapse.

    ``sleep`` is injectable so tests do not have to wait in real time.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_healthy(url, timeout=probe_timeout):
            return True
        sleep(interval)
    return False


def get_json(url: str, timeout: float = 2.0) -> Any | None:
    """Return the decoded JSON body of *url*, or None if it is unavailable."""
    try:
        resp = httpx.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPError, ValueError):
        return None
