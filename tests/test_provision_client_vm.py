"""Tests for provisioning the developer client VM."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock

import pytest

from llm.provision import client_vm


class TestSetupPiInContainer:
    """Tests for setup_pi_in_container - subprocess calls are mocked."""

    def _make_completed(self, returncode=0, stdout=""):
        p = MagicMock()
        p.returncode = returncode
        p.stdout = stdout
        return p

    def test_adds_hosts_entry_using_proxy_lan_ip(self, monkeypatch, tmp_path):
        """The /etc/hosts entry should use proxy.lan_ip (the server's LAN IP)."""
        import tomli_w

        # Write a minimal config so load_config() works.
        config = tmp_path / "config.toml"
        config.write_text(
            tomli_w.dumps(
                {
                    "server": {"port": 8080},
                    "proxy": {
                        "port": 8443,
                        "lan_ip": "192.168.1.1",
                        "lan_subnet": "192.168.1.0/24",
                        "cert_path": str(tmp_path / "cert.pem"),
                    },
                    "auth": {"api_key": "key"},
                    "models": {"active": "model.gguf", "dir": str(tmp_path)},
                    "lxd": {"craft_dirs": [], "mounts": []},
                }
            )
        )
        monkeypatch.chdir(tmp_path)

        calls: list[list] = []

        def _run(cmd, **kwargs):
            calls.append(list(cmd))
            return self._make_completed(0, "")

        monkeypatch.setattr(subprocess, "run", _run)

        from llm.provision.client_vm import setup_pi_in_container

        setup_pi_in_container("craft-llm-1")

        # The /etc/hosts command must use proxy.lan_ip.
        hosts_calls = [c for c in calls if any("/etc/hosts" in str(a) for a in c)]
        assert hosts_calls, "Expected an /etc/hosts manipulation command"
        hosts_cmd_str = " ".join(str(a) for a in hosts_calls[0])
        assert "192.168.1.1" in hosts_cmd_str, "Expected proxy.lan_ip in /etc/hosts entry"
        assert "10.113.167.1" not in hosts_cmd_str, "Expected bridge_ip NOT in /etc/hosts entry"

    def test_adds_hosts_entry_even_when_no_bridge_ip(self, monkeypatch, tmp_path):
        """The /etc/hosts entry is always written using proxy.lan_ip regardless of bridge_ip."""
        import tomli_w

        config = tmp_path / "config.toml"
        config.write_text(
            tomli_w.dumps(
                {
                    "server": {"port": 8080},
                    "proxy": {
                        "port": 8443,
                        "lan_ip": "192.168.1.1",
                        "lan_subnet": "192.168.1.0/24",
                        "cert_path": str(tmp_path / "cert.pem"),
                    },
                    "auth": {"api_key": "key"},
                    "models": {"active": "model.gguf", "dir": str(tmp_path)},
                    "lxd": {"craft_dirs": [], "mounts": []},
                }
            )
        )
        monkeypatch.chdir(tmp_path)

        calls: list[list] = []

        def _run(cmd, **kwargs):
            calls.append(list(cmd))
            return self._make_completed(0, "")

        monkeypatch.setattr(subprocess, "run", _run)

        from llm.provision.client_vm import setup_pi_in_container

        setup_pi_in_container("craft-llm-1")

        # Should still write the /etc/hosts entry using proxy.lan_ip
        hosts_calls = [c for c in calls if any("/etc/hosts" in str(a) for a in c)]
        assert hosts_calls, "Expected /etc/hosts command even when bridge_ip is empty"
        hosts_cmd_str = " ".join(str(a) for a in hosts_calls[0])
        assert "192.168.1.1" in hosts_cmd_str, "Expected proxy.lan_ip in /etc/hosts entry"

    def test_writes_cert_when_cert_pem_provided(self, monkeypatch, tmp_path):
        """Cert content should be piped into the container when cert_pem is set."""
        import tomli_w

        config = tmp_path / "config.toml"
        fake_cert = "-----BEGIN CERTIFICATE-----\nMIIfake\n-----END CERTIFICATE-----\n"
        config.write_text(
            tomli_w.dumps(
                {
                    "server": {"port": 8080},
                    "proxy": {
                        "port": 8443,
                        "lan_ip": "192.168.1.1",
                        "lan_subnet": "192.168.1.0/24",
                        "cert_path": str(tmp_path / "cert.pem"),
                    },
                    "auth": {"api_key": "key"},
                    "models": {"active": "model.gguf", "dir": str(tmp_path)},
                    "lxd": {"craft_dirs": [], "mounts": []},
                }
            )
        )
        monkeypatch.chdir(tmp_path)

        stdin_inputs: list[bytes] = []

        def _run(cmd, **kwargs):
            if "input" in kwargs and kwargs["input"]:
                stdin_inputs.append(kwargs["input"])
            return self._make_completed(0, "")

        monkeypatch.setattr(subprocess, "run", _run)

        from llm.provision.client_vm import setup_pi_in_container

        setup_pi_in_container("craft-llm-1", cert_pem=fake_cert)

        assert fake_cert.encode() in stdin_inputs, (
            "Expected cert PEM to be piped as stdin to a container command"
        )

    def test_reads_cert_from_config_when_not_provided(self, monkeypatch, tmp_path):
        """Cert should be read from cert_path in config when cert_pem is not given."""
        import tomli_w

        cert_file = tmp_path / "cert.pem"
        cert_content = "-----BEGIN CERTIFICATE-----\nMIIcert\n-----END CERTIFICATE-----\n"
        cert_file.write_text(cert_content)

        config = tmp_path / "config.toml"
        config.write_text(
            tomli_w.dumps(
                {
                    "server": {"port": 8080},
                    "proxy": {
                        "port": 8443,
                        "lan_ip": "192.168.1.1",
                        "lan_subnet": "192.168.1.0/24",
                        "cert_path": str(cert_file),
                    },
                    "auth": {"api_key": "key"},
                    "models": {"active": "model.gguf", "dir": str(tmp_path)},
                    "lxd": {"craft_dirs": [], "mounts": []},
                }
            )
        )
        monkeypatch.chdir(tmp_path)

        stdin_inputs: list[bytes] = []

        def _run(cmd, **kwargs):
            if "input" in kwargs and kwargs["input"]:
                stdin_inputs.append(kwargs["input"])
            return self._make_completed(0, "")

        monkeypatch.setattr(subprocess, "run", _run)

        from llm.provision.client_vm import setup_pi_in_container

        # No cert_pem passed - should be read from config cert_path
        setup_pi_in_container("craft-llm-1", cert_pem=None)

        assert cert_content.encode() in stdin_inputs, (
            "Expected cert content from config cert_path to be piped into container"
        )

    def test_models_json_uses_local_llm_hostname(self, monkeypatch, tmp_path):
        """models.json written into the container should use the 'local-llm' hostname URL."""

        import tomli_w

        config = tmp_path / "config.toml"
        config.write_text(
            tomli_w.dumps(
                {
                    "server": {"port": 8080},
                    "proxy": {
                        "port": 8443,
                        "lan_ip": "192.168.1.1",
                        "lan_subnet": "192.168.1.0/24",
                        "cert_path": str(tmp_path / "cert.pem"),
                    },
                    "auth": {"api_key": "key"},
                    "models": {"active": "model.gguf", "dir": str(tmp_path)},
                    "lxd": {"craft_dirs": [], "mounts": []},
                }
            )
        )
        monkeypatch.chdir(tmp_path)

        stdin_inputs: list[bytes] = []

        def _run(cmd, **kwargs):
            if "input" in kwargs and kwargs["input"]:
                stdin_inputs.append(kwargs["input"])
            return self._make_completed(0, "")

        monkeypatch.setattr(subprocess, "run", _run)

        from llm.provision.client_vm import setup_pi_in_container

        setup_pi_in_container("craft-llm-1")

        # The first stdin input should be the merged models.json
        json_inputs = [b for b in stdin_inputs if b.strip().startswith(b"{")]
        assert json_inputs, "Expected JSON to be piped to container"
        parsed = json.loads(json_inputs[0])
        base_url = parsed["providers"]["local-llm"]["baseUrl"]
        assert "local-llm" in base_url, f"Expected 'local-llm' hostname in baseUrl, got: {base_url}"
        assert base_url.startswith("https://"), "Expected HTTPS scheme"


# ── _tag_as_managed / _list_managed_containers ────────────────────────────────


class TestPathVerification:
    """Tests that bun, pi, and omp end up on the container's PATH."""

    def _make_completed(self, returncode=0, stdout=""):
        p = MagicMock()
        p.returncode = returncode
        p.stdout = stdout
        return p

    def test_bun_install_uses_CONTAINER_HOME_not_literal_dollar_home(self, monkeypatch, tmp_path):
        """bun install must run with HOME=CONTAINER_HOME, not a literal '$HOME' string.

        A literal '$HOME' passed via --env would cause bun to create a directory
        literally named $HOME instead of installing under the container user's home.
        """
        import tomli_w

        config = tmp_path / "config.toml"
        config.write_text(
            tomli_w.dumps(
                {
                    "server": {"port": 8080},
                    "proxy": {
                        "port": 8443,
                        "lan_ip": "192.168.1.1",
                        "lan_subnet": "192.168.1.0/24",
                        "cert_path": str(tmp_path / "cert.pem"),
                    },
                    "auth": {"api_key": "key"},
                    "models": {"active": "model.gguf", "dir": str(tmp_path)},
                    "lxd": {"craft_dirs": [], "mounts": []},
                }
            )
        )
        monkeypatch.chdir(tmp_path)

        calls: list[list] = []

        def _run(cmd, **kwargs):
            calls.append(list(cmd))
            return self._make_completed(0, "")

        monkeypatch.setattr(subprocess, "run", _run)

        from llm.provision.client_vm import LxdVmManager
        from llm.provision.exec import CONTAINER_HOME

        mgr = LxdVmManager("test-vm", mounts=[])
        mgr._install_packages(uid=1000)

        bun_calls = [c for c in calls if "bun.sh" in " ".join(str(a) for a in c)]
        assert bun_calls, "Expected a bun install command"
        cmd_str = " ".join(str(a) for a in bun_calls[0])
        assert f"HOME={CONTAINER_HOME}" in cmd_str, (
            f"Expected HOME={CONTAINER_HOME} in bun install call: {cmd_str}"
        )
        assert "BUN_INSTALL" not in cmd_str, (
            f"BUN_INSTALL env var should not be set (uses HOME instead): {cmd_str}"
        )

    def test_path_fish_contains_bun_bin(self, monkeypatch, tmp_path):
        """path.fish must contain ~/.bun/bin on PATH so bun is discoverable."""
        import tomli_w

        config = tmp_path / "config.toml"
        config.write_text(
            tomli_w.dumps(
                {
                    "server": {"port": 8080},
                    "proxy": {
                        "port": 8443,
                        "lan_ip": "192.168.1.1",
                        "lan_subnet": "192.168.1.0/24",
                        "cert_path": str(tmp_path / "cert.pem"),
                    },
                    "auth": {"api_key": "key"},
                    "models": {"active": "model.gguf", "dir": str(tmp_path)},
                    "lxd": {"craft_dirs": [], "mounts": []},
                }
            )
        )
        monkeypatch.chdir(tmp_path)

        calls: list[list] = []

        def _run(cmd, **kwargs):
            calls.append(list(cmd))
            return self._make_completed(0, "")

        monkeypatch.setattr(subprocess, "run", _run)

        from llm.provision.client_vm import LxdVmManager

        mgr = LxdVmManager("test-vm", mounts=[])
        mgr._install_pylsp(uid=1000, gid=1000)

        # Check that path.fish content includes .bun/bin
        path_fish_calls = [c for c in calls if "path.fish" in " ".join(str(a) for a in c)]
        assert path_fish_calls, "Expected path.fish PATH setup commands"
        # The echo command that writes path.fish should contain .bun/bin
        all_path_cmd = " ".join(str(a) for c in path_fish_calls for a in c)
        assert ".bun/bin" in all_path_cmd, f"Expected '.bun/bin' in path.fish setup: {all_path_cmd[:600]}"
        assert ".cargo/bin" in all_path_cmd, f"Expected '.cargo/bin' in path.fish setup: {all_path_cmd[:600]}"

    def test_bashrc_contains_bun_bin(self, monkeypatch, tmp_path):
        """~/.bashrc must contain ~/.bun/bin on PATH so bun is discoverable."""
        import tomli_w

        config = tmp_path / "config.toml"
        config.write_text(
            tomli_w.dumps(
                {
                    "server": {"port": 8080},
                    "proxy": {
                        "port": 8443,
                        "lan_ip": "192.168.1.1",
                        "lan_subnet": "192.168.1.0/24",
                        "cert_path": str(tmp_path / "cert.pem"),
                    },
                    "auth": {"api_key": "key"},
                    "models": {"active": "model.gguf", "dir": str(tmp_path)},
                    "lxd": {"craft_dirs": [], "mounts": []},
                }
            )
        )
        monkeypatch.chdir(tmp_path)

        calls: list[list] = []

        def _run(cmd, **kwargs):
            calls.append(list(cmd))
            return self._make_completed(0, "")

        monkeypatch.setattr(subprocess, "run", _run)

        from llm.provision.client_vm import LxdVmManager

        mgr = LxdVmManager("test-vm", mounts=[])
        mgr._install_pylsp(uid=1000, gid=1000)

        # Check that .bashrc setup includes .bun/bin
        bashrc_calls = [c for c in calls if ".bashrc" in " ".join(str(a) for a in c)]
        assert bashrc_calls, "Expected .bashrc PATH setup commands"
        all_bash_cmd = " ".join(str(a) for c in bashrc_calls for a in c)
        assert ".bun/bin" in all_bash_cmd, f"Expected '.bun/bin' in .bashrc setup: {all_bash_cmd[:600]}"
        assert ".cargo/bin" in all_bash_cmd, f"Expected '.cargo/bin' in .bashrc setup: {all_bash_cmd[:600]}"

    def test_pi_and_omp_npm_runs_as_container_user(self, monkeypatch, tmp_path):
        """pi and oh-my-pi (omp) npm installs run as container user so they land on PATH."""
        import tomli_w

        config = tmp_path / "config.toml"
        config.write_text(
            tomli_w.dumps(
                {
                    "server": {"port": 8080},
                    "proxy": {
                        "port": 8443,
                        "lan_ip": "192.168.1.1",
                        "lan_subnet": "192.168.1.0/24",
                        "cert_path": str(tmp_path / "cert.pem"),
                    },
                    "auth": {"api_key": "key"},
                    "models": {"active": "model.gguf", "dir": str(tmp_path)},
                    "lxd": {"craft_dirs": [], "mounts": []},
                }
            )
        )
        monkeypatch.chdir(tmp_path)

        calls: list[list] = []

        def _run(cmd, **kwargs):
            calls.append(list(cmd))
            return self._make_completed(0, "")

        monkeypatch.setattr(subprocess, "run", _run)

        from llm.provision.client_vm import LxdVmManager

        mgr = LxdVmManager("test-vm", mounts=[])
        mgr._install_packages(uid=1000)

        npm_calls = [c for c in calls if "npm install" in " ".join(str(a) for a in c)]
        assert len(npm_calls) >= 2, "Expected npm install commands for pi and oh-my-pi"
        for npm_call in npm_calls:
            cmd_str = " ".join(str(a) for a in npm_call)
            # Should run as container user (uid=1000)
            assert "--user=1000" in cmd_str, f"npm install should run as container user: {cmd_str}"

        # Verify npm config set prefix (for .local) also runs as container user
        config_calls = [c for c in calls if "npm config set prefix" in " ".join(str(a) for a in c)]
        assert config_calls, "Expected npm config set prefix command"
        config_str = " ".join(str(a) for a in config_calls[0])
        assert "--user=1000" in config_str, (
            f"npm config set prefix should run as container user: {config_str}"
        )


# ── Snap cgroup and nested VM tests ──────────────────────────────────────────


class TestSnapInstall:
    """Tests that snap installs use systemd-run to avoid lxd-agent cgroup errors."""

    def _make_completed(self, returncode=0, stdout=""):
        p = MagicMock()
        p.returncode = returncode
        p.stdout = stdout
        return p

    def test_snap_install_uses_systemd_run(self, monkeypatch, tmp_path):
        """_snap_install must wrap snap with systemd-run --wait to avoid cgroup rejection."""
        import tomli_w

        config = tmp_path / "config.toml"
        config.write_text(
            tomli_w.dumps(
                {
                    "server": {"port": 8080},
                    "proxy": {
                        "port": 8443,
                        "lan_ip": "192.168.1.1",
                        "lan_subnet": "192.168.1.0/24",
                        "cert_path": str(tmp_path / "cert.pem"),
                    },
                    "auth": {"api_key": "key"},
                    "models": {"active": "model.gguf", "dir": str(tmp_path)},
                    "lxd": {"craft_dirs": [], "mounts": []},
                }
            )
        )
        monkeypatch.chdir(tmp_path)

        calls: list[list] = []

        def _run(cmd, **kwargs):
            calls.append(list(cmd))
            return self._make_completed(0, "")

        monkeypatch.setattr(subprocess, "run", _run)

        from llm.provision.client_vm import LxdVmManager

        mgr = LxdVmManager("test-vm", mounts=[])
        mgr._snap_install("astral-uv", "--classic")

        assert len(calls) == 1
        cmd = calls[0]
        cmd_str = " ".join(cmd)
        assert "systemd-run" in cmd_str, f"snap install should use systemd-run: {cmd_str}"
        assert "--wait" in cmd_str, f"systemd-run should pass --wait: {cmd_str}"
        assert "snap" in cmd_str, f"snap command should be present: {cmd_str}"
        assert "astral-uv" in cmd_str, f"snap name should be present: {cmd_str}"
        assert "--classic" in cmd_str, f"--classic flag should be present: {cmd_str}"

    def test_install_packages_snaps_use_systemd_run(self, monkeypatch, tmp_path):
        """astral-uv and helix snap installs in _install_packages must use systemd-run."""
        import tomli_w

        config = tmp_path / "config.toml"
        config.write_text(
            tomli_w.dumps(
                {
                    "server": {"port": 8080},
                    "proxy": {
                        "port": 8443,
                        "lan_ip": "192.168.1.1",
                        "lan_subnet": "192.168.1.0/24",
                        "cert_path": str(tmp_path / "cert.pem"),
                    },
                    "auth": {"api_key": "key"},
                    "models": {"active": "model.gguf", "dir": str(tmp_path)},
                    "lxd": {"craft_dirs": [], "mounts": []},
                }
            )
        )
        monkeypatch.chdir(tmp_path)

        calls: list[list] = []

        def _run(cmd, **kwargs):
            calls.append(list(cmd))
            return self._make_completed(0, "")

        monkeypatch.setattr(subprocess, "run", _run)

        from llm.provision.client_vm import LxdVmManager

        mgr = LxdVmManager("test-vm", mounts=[])
        mgr._install_packages(uid=1000)

        snap_install_calls = [c for c in calls if "snap" in c and "install" in c]
        assert snap_install_calls, "Expected snap install commands in _install_packages"
        for call in snap_install_calls:
            cmd_str = " ".join(str(a) for a in call)
            assert "systemd-run" in cmd_str, (
                f"snap install should use systemd-run to avoid cgroup errors: {cmd_str}"
            )
            assert "--wait" in cmd_str, f"systemd-run should pass --wait: {cmd_str}"

    def test_setup_nested_lxd_snap_uses_systemd_run(self, monkeypatch, tmp_path):
        """lxd snap install in _setup_nested_lxd must use systemd-run."""
        import tomli_w

        config = tmp_path / "config.toml"
        config.write_text(
            tomli_w.dumps(
                {
                    "server": {"port": 8080},
                    "proxy": {
                        "port": 8443,
                        "lan_ip": "192.168.1.1",
                        "lan_subnet": "192.168.1.0/24",
                        "cert_path": str(tmp_path / "cert.pem"),
                    },
                    "auth": {"api_key": "key"},
                    "models": {"active": "model.gguf", "dir": str(tmp_path)},
                    "lxd": {"craft_dirs": [], "mounts": []},
                }
            )
        )
        monkeypatch.chdir(tmp_path)

        calls: list[list] = []

        def _run(cmd, **kwargs):
            calls.append(list(cmd))
            return self._make_completed(0, '{"status":"done"}')

        monkeypatch.setattr(subprocess, "run", _run)

        from llm.provision.client_vm import LxdVmManager

        mgr = LxdVmManager("test-vm", mounts=[])
        mgr._setup_nested_lxd()

        lxd_install_calls = [c for c in calls if "snap" in c and "install" in c and "lxd" in c]
        assert lxd_install_calls, "Expected 'snap install lxd' in _setup_nested_lxd"
        cmd_str = " ".join(str(a) for a in lxd_install_calls[0])
        assert "systemd-run" in cmd_str, (
            f"lxd snap install should use systemd-run to avoid cgroup errors: {cmd_str}"
        )
        assert "--wait" in cmd_str, f"systemd-run should pass --wait: {cmd_str}"


class TestNestedVmSupport:
    """Tests for nested VM support (KVM passthrough via the host's nested KVM)."""

    def _make_completed(self, returncode=0, stdout=""):
        p = MagicMock()
        p.returncode = returncode
        p.stdout = stdout
        return p

    def test_create_container_is_a_vm(self, monkeypatch, tmp_path):
        """lxc launch must use --vm so the instance is a full VM with KVM passthrough."""
        import tomli_w

        config = tmp_path / "config.toml"
        config.write_text(
            tomli_w.dumps(
                {
                    "server": {"port": 8080},
                    "proxy": {
                        "port": 8443,
                        "lan_ip": "192.168.1.1",
                        "lan_subnet": "192.168.1.0/24",
                        "cert_path": str(tmp_path / "cert.pem"),
                    },
                    "auth": {"api_key": "key"},
                    "models": {"active": "model.gguf", "dir": str(tmp_path)},
                    "lxd": {"craft_dirs": [], "mounts": []},
                }
            )
        )
        monkeypatch.chdir(tmp_path)

        calls: list[list] = []

        def _run(cmd, **kwargs):
            calls.append(list(cmd))
            return self._make_completed(0, '{"status":"done"}')

        monkeypatch.setattr(subprocess, "run", _run)

        from llm.provision.client_vm import LxdVmManager

        mgr = LxdVmManager("test-vm", mounts=[])
        mgr.create_container()

        launch_calls = [c for c in calls if "lxc" in c and "launch" in c]
        assert launch_calls, "Expected an lxc launch command"
        assert "--vm" in launch_calls[0], (
            "lxc launch should use --vm; LXD VMs automatically pass through CPU "
            "virtualisation flags so nested VMs work when the host has nested KVM enabled"
        )

    def test_create_container_hints_at_stale_kvm_check(self, monkeypatch, tmp_path, capsys):
        """A 'no /dev/kvm' failure should surface a hint to restart the lxd daemon.

        LXD caches its KVM support check at daemon startup, so this error can be
        stale even when /dev/kvm is actually available; restarting the daemon
        forces a fresh check.
        """
        import tomli_w

        config = tmp_path / "config.toml"
        config.write_text(
            tomli_w.dumps(
                {
                    "server": {"port": 8080},
                    "proxy": {
                        "port": 8443,
                        "lan_ip": "192.168.1.1",
                        "lan_subnet": "192.168.1.0/24",
                        "cert_path": str(tmp_path / "cert.pem"),
                    },
                    "auth": {"api_key": "key"},
                    "models": {"active": "model.gguf", "dir": str(tmp_path)},
                    "lxd": {"craft_dirs": [], "mounts": []},
                }
            )
        )
        monkeypatch.chdir(tmp_path)

        def _run(cmd, **kwargs):
            raise subprocess.CalledProcessError(
                1,
                cmd,
                output="",
                stderr="Error: Failed instance creation: Failed creating instance record: "
                'Instance type "virtual-machine" is not supported on this server: '
                "KVM support is missing (no /dev/kvm)",
            )

        monkeypatch.setattr(subprocess, "run", _run)
        monkeypatch.setattr("llm.provision.exec.time.sleep", lambda *a, **k: None)

        from llm.provision.client_vm import LxdVmManager

        mgr = LxdVmManager("test-vm", mounts=[])
        with pytest.raises(subprocess.CalledProcessError):
            mgr.create_container()

        captured = capsys.readouterr()
        assert "sudo systemctl restart snap.lxd.daemon" in captured.out


# ── secret redaction ─────────────────────────────────────────────────────────


class TestRefreshExcludesHermes:
    """Refreshing the Hermes VM would overwrite its agent configuration."""

    def test_named_hermes_vm_is_refused(self, monkeypatch):
        monkeypatch.setattr(client_vm, "container_exists", MagicMock(return_value=True))
        monkeypatch.setattr(client_vm, "get_container_kind", MagicMock(return_value=client_vm.KIND_HERMES))
        refresh = MagicMock()
        monkeypatch.setattr(client_vm.LxdVmManager, "_refresh", refresh)

        with pytest.raises(RuntimeError, match="llm hermes setup"):
            client_vm.refresh_containers("hermes")

        refresh.assert_not_called()

    def test_named_client_vm_is_refreshed(self, monkeypatch):
        monkeypatch.setattr(client_vm, "container_exists", MagicMock(return_value=True))
        monkeypatch.setattr(client_vm, "get_container_kind", MagicMock(return_value=client_vm.KIND_CLIENT))
        refresh = MagicMock()
        monkeypatch.setattr(client_vm.LxdVmManager, "_refresh", refresh)

        client_vm.refresh_containers("dev")

        refresh.assert_called_once()

    def test_bulk_refresh_only_covers_clients(self, monkeypatch):
        listed = MagicMock(return_value=["dev"])
        monkeypatch.setattr(client_vm, "_list_managed_containers", listed)
        monkeypatch.setattr(client_vm.LxdVmManager, "_refresh", MagicMock())

        client_vm.refresh_containers()

        assert listed.call_args.kwargs["kind"] == client_vm.KIND_CLIENT


# ── Sandbox tests ────────────────────────────────────────────────────────────


class TestSandboxProvisioning:
    """Tests for sandbox container creation and settings."""

    def test_load_lxd_settings_default_sandbox(self, monkeypatch):
        monkeypatch.setattr("llm.provision.client_vm.try_load_lxd", lambda: None)
        from llm.provision.client_vm import load_lxd_settings
        from llm.provision.exec import _DEFAULT_MOUNTS, _DEFAULT_SANDBOX_MOUNTS

        mounts, _ = load_lxd_settings(sandbox=True)
        assert mounts == _DEFAULT_SANDBOX_MOUNTS
        assert mounts != _DEFAULT_MOUNTS

    def test_load_lxd_settings_configured_sandbox(self, monkeypatch):
        from llm.provision.client_vm import load_lxd_settings
        from llm.settings.models import LxdSettings, MountEntry

        fake_lxd = LxdSettings(
            mounts=[MountEntry(host="/home/user/dev")],
            sandbox_mounts=[MountEntry(host="/home/user/dev/cal/chiptune")],
        )
        monkeypatch.setattr("llm.provision.client_vm.try_load_lxd", lambda: fake_lxd)

        mounts, _ = load_lxd_settings(sandbox=True)
        assert len(mounts) == 1
        assert mounts[0][0] == "chiptune"

    def test_lxd_vm_manager_sandbox_init(self):
        from llm.provision.client_vm import LxdVmManager
        from llm.provision.exec import _DEFAULT_SANDBOX_MOUNTS

        mgr = LxdVmManager("sandbox-vm", sandbox=True)
        assert mgr.sandbox is True
        assert mgr.mounts == list(_DEFAULT_SANDBOX_MOUNTS)

    def test_create_and_setup_sandbox_tags_and_skips_pi(self, monkeypatch):
        from llm.provision.client_vm import LxdVmManager
        from llm.provision.exec import _SANDBOX_TAG

        calls: list[list] = []

        def _fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            p = MagicMock()
            p.returncode = 0
            p.stdout = ""
            return p

        monkeypatch.setattr("llm.provision.client_vm.run", _fake_run)
        monkeypatch.setattr("llm.provision.client_vm.run_with_retry", _fake_run)
        monkeypatch.setattr("llm.provision.client_vm.container_exists", lambda c: False)
        monkeypatch.setattr(LxdVmManager, "create_container", lambda self: None)
        monkeypatch.setattr(LxdVmManager, "_add_mounts", lambda self, m, **k: None)
        monkeypatch.setattr(LxdVmManager, "_install_packages", lambda self, **k: None)
        monkeypatch.setattr(LxdVmManager, "_install_pylsp", lambda self, **k: None)
        monkeypatch.setattr(LxdVmManager, "_setup_nested_lxd", lambda self, **k: None)
        monkeypatch.setattr(LxdVmManager, "_tag_as_managed", lambda self: None)
        monkeypatch.setattr(LxdVmManager, "run_tests", lambda self: None)

        pi_called = False

        def _fake_setup_pi(self, cert_pem=None):
            nonlocal pi_called
            pi_called = True

        monkeypatch.setattr(LxdVmManager, "setup_pi", _fake_setup_pi)

        mgr = LxdVmManager("craft-sandbox", sandbox=True)
        mgr.create_and_setup()

        assert pi_called is False, "setup_pi should not be called for sandbox container"
        sandbox_tag_calls = [
            c
            for c in calls
            if len(c) >= 5
            and c[0] == "lxc"
            and c[1] == "config"
            and c[2] == "set"
            and f"{_SANDBOX_TAG}=true" in c[4]
        ]
        assert sandbox_tag_calls, f"Expected container to be tagged with {_SANDBOX_TAG}=true"

    def test_refresh_sandbox_container_uses_sandbox_credentials(self, monkeypatch):
        from llm.provision import client_vm
        from llm.settings.models import GitHubSandboxSettings, GitHubSettings, Settings

        cfg = Settings(
            github=GitHubSettings(
                token="personal-token",
                git_pat="personal-pat",
                sandbox=GitHubSandboxSettings(
                    token="bot-token",
                    git_pat="bot-pat",
                    git_username="mr-cal-bot",
                    git_email="bot@example.com",
                ),
            )
        )
        monkeypatch.setattr(client_vm, "container_exists", MagicMock(return_value=True))
        monkeypatch.setattr(client_vm, "get_container_kind", MagicMock(return_value=client_vm.KIND_CLIENT))
        monkeypatch.setattr(client_vm, "is_container_sandbox", MagicMock(return_value=True))
        monkeypatch.setattr(client_vm, "load_config", MagicMock(return_value=cfg))
        mock_refresh = MagicMock()
        monkeypatch.setattr(client_vm.LxdVmManager, "_refresh", mock_refresh)

        client_vm.refresh_containers("craft-sandbox", cert_pem="cert-data")

        mock_refresh.assert_called_once_with(
            cert_pem=None,
            gh_token="bot-token",
            git_username="mr-cal-bot",
            git_email="bot@example.com",
            git_pat="bot-pat",
        )
