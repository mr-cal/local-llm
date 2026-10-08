"""LXD container provisioning for the Jotter digest assistant."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from llm.core.console import console
from llm.provision.exec import (
    CONTAINER_HOME,
    HOST_GID,
    HOST_UID,
    JOTTER_CONTAINER_NAME,
    KIND_JOTTER,
    _tag_as_managed,
    container_exists,
    run,
    run_capture,
    run_with_retry,
    wait_for_container,
)

if TYPE_CHECKING:
    from llm.settings.models import Settings


class JotterVmManager:
    """Manages the lifecycle of the dedicated Jotter LXD container."""

    def __init__(
        self, container_name: str = JOTTER_CONTAINER_NAME, timezone: str = "America/Chicago"
    ) -> None:
        self.container = container_name
        self.timezone = timezone
        self.uid = HOST_UID
        self.gid = HOST_GID

    def create_and_setup(self, cfg: Settings, recreate: bool = False) -> None:
        """Create and configure the Jotter container."""
        exists = container_exists(self.container)
        if exists and recreate:
            console.print(f"Deleting existing container {self.container}...")
            run_with_retry(["lxc", "delete", "--force", self.container], desc="lxc delete")
            exists = False

        if not exists:
            console.print(f"Creating Jotter container: {self.container}...")
            run(["lxc", "launch", "ubuntu:24.04", self.container])
            wait_for_container(self.container)
            _tag_as_managed(self.container, KIND_JOTTER)

        # Set timezone
        run(["lxc", "exec", self.container, "--", "timedatectl", "set-timezone", self.timezone])

        # Bind-mount ~/dev/craft read-only
        craft_dir = Path(cfg.digest.craft_dir).expanduser()
        if craft_dir.exists():
            console.print(f"Mounting read-only craft directory: {craft_dir}...")
            os.makedirs(craft_dir, exist_ok=True)
            # Remove existing mount if present
            run_capture(["lxc", "config", "device", "remove", self.container, "craft-dev"])
            run(
                [
                    "lxc",
                    "config",
                    "device",
                    "add",
                    self.container,
                    "craft-dev",
                    "disk",
                    f"source={craft_dir}",
                    f"path={CONTAINER_HOME}/dev/craft",
                    "readonly=true",
                ]
            )

        # Install packages
        console.print("Installing base packages...")
        run_with_retry(
            ["lxc", "exec", self.container, "--", "apt-get", "update", "-q"],
            desc="apt-get update",
        )
        run(
            [
                "lxc",
                "exec",
                self.container,
                "--",
                "apt-get",
                "install",
                "-y",
                "python3-pip",
                "python3-venv",
                "git",
                "ripgrep",
                "curl",
                "jq",
            ]
        )

        console.print(f"[green]✓[/green] Jotter container {self.container} provisioned successfully.")

    def status(self) -> dict[str, str]:
        """Check status of Jotter container."""
        exists = container_exists(self.container)
        if not exists:
            return {"status": "not_created"}
        r = run_capture(["lxc", "info", self.container])
        status = "running" if "Status: RUNNING" in r.stdout else "stopped"
        return {"status": status}
