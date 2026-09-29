"""The lifecycle shared by every managed VM: create, boot, swap, users."""

from __future__ import annotations

import subprocess
from enum import Enum

from llm.core.console import console
from llm.provision.exec import (
    _KIND_TAG,
    _MANAGED_TAG,
    CONTAINER_GID,
    CONTAINER_HOME,
    CONTAINER_UID,
    CONTAINER_USER,
    HOST_GID,
    HOST_UID,
    KIND_CLIENT,
    run,
    run_with_retry,
    wait_for_container,
)

VM_ROOT_DISK_SIZE = "90GB"
VM_MEMORY = "4GiB"
VM_SWAP_SIZE = "4G"


class SetupStep(Enum):
    """Ordered steps for LXD VM setup, used in progress labels like '2/4'."""

    MOUNTS = 2
    PACKAGES = 3
    PYLSP = 4
    NESTED_LXD = 4  # shares the same step as pylsp

    def label(self, total: int) -> str:
        """Return a display label like '2/4'."""
        return f"{self.value}/{total}"


class _BaseVmManager:
    """Shared LXD VM lifecycle: create, configure user, swap, sudo, and tag.

    Subclasses add workload-specific package installs and configuration.
    Do not instantiate directly — use ``LxdVmManager`` or ``HermesVmManager``.

    Attributes:
        container: Name of the LXD container/VM.
        uid: UID for running commands inside the container.
        gid: GID for running commands inside the container.
    """

    def __init__(
        self,
        container: str,
        uid: int = HOST_UID,
        gid: int = HOST_GID,
    ) -> None:
        self.container = container
        self.uid = uid
        self.gid = gid

    # ── VM lifecycle ─────────────────────────────────────────────────────

    def create_container(self) -> None:
        """Launch an LXD VM (ubuntu:24.04) with default resources."""
        console.print(f"\n[bold][1/4][/bold] Launching {self.container} (ubuntu:24.04) as VM...")
        launch_cmd = [
            "lxc",
            "launch",
            "ubuntu:24.04",
            self.container,
            "--vm",
            "--config",
            f"limits.memory={VM_MEMORY}",
            "--device",
            f"root,size={VM_ROOT_DISK_SIZE}",
        ]
        try:
            run_with_retry(launch_cmd, desc="lxc launch", capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            if e.stderr and "no /dev/kvm" in e.stderr:
                console.print(
                    "[bold yellow]Hint:[/bold yellow] LXD reported missing KVM support, but this is "
                    "often a stale check left over from before /dev/kvm became available (e.g. after "
                    "loading kernel modules or restarting VirtualBox). Try:\n"
                    "  sudo systemctl restart snap.lxd.daemon\n"
                    "then re-run this command."
                )
            raise
        wait_for_container(self.container)
        self._setup_vm_swap()
        # Rename the default ubuntu user/group to match the host user, and move
        # the home directory to the same path as on the host.
        run(
            [
                "lxc",
                "exec",
                self.container,
                "--",
                "usermod",
                "--badname",
                "--login",
                CONTAINER_USER,
                "--home",
                CONTAINER_HOME,
                "--move-home",
                "ubuntu",
            ]
        )
        run(
            [
                "lxc",
                "exec",
                self.container,
                "--",
                "groupmod",
                "--new-name",
                CONTAINER_USER,
                "ubuntu",
            ]
        )
        self._fix_vm_user_uid()

    def _configure_sudo(self) -> None:
        """Configure passwordless sudo for the container user."""
        console.print("  Configuring passwordless sudo...")
        run(
            [
                "lxc",
                "exec",
                self.container,
                "--",
                "bash",
                "-c",
                f"printf 'User_Alias CONTAINERUSER = #{self.uid}\\n"
                f"CONTAINERUSER ALL=(ALL) NOPASSWD:ALL\\n'"
                f" > /etc/sudoers.d/nopasswd-user"
                f" && chmod 440 /etc/sudoers.d/nopasswd-user",
            ]
        )

    def _tag_as_managed(self, kind: str = KIND_CLIENT) -> None:
        """Tag this container as managed, recording which kind of VM it is."""
        run(["lxc", "config", "set", self.container, f"{_MANAGED_TAG}=true"])
        run(["lxc", "config", "set", self.container, f"{_KIND_TAG}={kind}"])

    def _snap_install(self, *args: str) -> None:
        """Install a snap inside the VM via systemd-run to avoid lxd-agent cgroup errors.

        Running snap directly under lxc exec puts the process in the
        lxd-agent.service cgroup, which snapd rejects. systemd-run creates a
        transient scope with its own cgroup that snapd accepts.
        """
        run(["lxc", "exec", self.container, "--", "systemd-run", "--wait", "snap", "install", *args])

    def _setup_vm_swap(self) -> None:
        """Create a persistent swapfile inside the VM and enable it on boot."""
        run(
            [
                "lxc",
                "exec",
                self.container,
                "--",
                "bash",
                "-c",
                f"fallocate -l {VM_SWAP_SIZE} /swapfile"
                f" && chmod 600 /swapfile"
                f" && mkswap /swapfile"
                f" && swapon /swapfile"
                f" && echo '/swapfile none swap sw 0 0' >> /etc/fstab",
            ],
            desc="create swapfile",
        )

    def _fix_vm_user_uid(self) -> None:
        """Change the in-VM user's UID/GID to HOST_UID/HOST_GID."""
        if HOST_GID != CONTAINER_GID:
            console.print(f"  Changing in-VM group GID {CONTAINER_GID} → {HOST_GID} to match host...")
            run(["lxc", "exec", self.container, "--", "groupmod", "-g", str(HOST_GID), CONTAINER_USER])
            r = subprocess.run(
                [
                    "lxc",
                    "exec",
                    self.container,
                    "--",
                    "bash",
                    "-c",
                    f"find / -xdev -group {CONTAINER_GID} -exec chgrp {HOST_GID} {{}} + 2>&1",
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                console.print(
                    f"[yellow]  Warning:[/yellow] chgrp may have missed some files: {r.stderr.strip()}"
                )
        if HOST_UID != CONTAINER_UID:
            console.print(f"  Changing in-VM user UID {CONTAINER_UID} → {HOST_UID} to match host...")
            run(["lxc", "exec", self.container, "--", "usermod", "-u", str(HOST_UID), CONTAINER_USER])
            r = subprocess.run(
                [
                    "lxc",
                    "exec",
                    self.container,
                    "--",
                    "bash",
                    "-c",
                    f"find / -xdev -user {CONTAINER_UID} -exec chown {HOST_UID} {{}} + 2>&1",
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                console.print(
                    f"[yellow]  Warning:[/yellow] chown may have missed some files: {r.stderr.strip()}"
                )
