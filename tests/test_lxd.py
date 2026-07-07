"""Tests for the LXD container management module."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

from llm.config import _get_lxd_bridge_info

# ── _get_lxd_bridge_info (shared impl used by lxd.py and config.py) ──────────


class TestGetLxdBridgeInfo:
    """Tests for _get_lxd_bridge_info which underpins both config and lxd modules."""

    _IP_OUTPUT = (
        "3: lxdbr0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP\n"
        "    link/ether 00:16:3e:xx:xx:xx brd ff:ff:ff:ff:ff:ff\n"
        "    inet 10.113.167.1/24 scope global lxdbr0\n"
        "       valid_lft forever preferred_lft forever\n"
    )

    def test_parses_bridge_ip(self, monkeypatch):
        def _run(cmd, **kw):
            p = MagicMock()
            p.returncode = 0
            p.stdout = self._IP_OUTPUT
            return p

        monkeypatch.setattr(subprocess, "run", _run)
        ip, subnet = _get_lxd_bridge_info()
        assert ip == "10.113.167.1"

    def test_parses_bridge_subnet(self, monkeypatch):
        def _run(cmd, **kw):
            p = MagicMock()
            p.returncode = 0
            p.stdout = self._IP_OUTPUT
            return p

        monkeypatch.setattr(subprocess, "run", _run)
        ip, subnet = _get_lxd_bridge_info()
        assert subnet == "10.113.167.0/24"

    def test_returns_empty_when_no_bridge(self, monkeypatch):
        def _run(cmd, **kw):
            p = MagicMock()
            p.returncode = 1
            p.stdout = ""
            return p

        monkeypatch.setattr(subprocess, "run", _run)
        ip, subnet = _get_lxd_bridge_info()
        assert ip == ""
        assert subnet == ""

    def test_returns_empty_when_no_inet_line(self, monkeypatch):
        def _run(cmd, **kw):
            p = MagicMock()
            p.returncode = 0
            p.stdout = "3: lxdbr0: <BROADCAST,MULTICAST,UP>\n"
            return p

        monkeypatch.setattr(subprocess, "run", _run)
        ip, subnet = _get_lxd_bridge_info()
        assert ip == ""
        assert subnet == ""

    def test_handles_different_subnet_size(self, monkeypatch):
        """/16 subnets should also parse correctly."""

        def _run(cmd, **kw):
            p = MagicMock()
            p.returncode = 0
            p.stdout = "    inet 172.16.0.1/16 scope global lxdbr0\n"
            return p

        monkeypatch.setattr(subprocess, "run", _run)
        ip, subnet = _get_lxd_bridge_info()
        assert ip == "172.16.0.1"
        assert subnet == "172.16.0.0/16"


# ── setup_pi_in_container ─────────────────────────────────────────────────────


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

        from llm.lxd import setup_pi_in_container

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

        from llm.lxd import setup_pi_in_container

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

        from llm.lxd import setup_pi_in_container

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

        from llm.lxd import setup_pi_in_container

        # No cert_pem passed - should be read from config cert_path
        setup_pi_in_container("craft-llm-1", cert_pem=None)

        assert cert_content.encode() in stdin_inputs, (
            "Expected cert content from config cert_path to be piped into container"
        )

    def test_models_json_uses_local_llm_hostname(self, monkeypatch, tmp_path):
        """models.json written into the container should use the 'local-llm' hostname URL."""
        import json

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

        from llm.lxd import setup_pi_in_container

        setup_pi_in_container("craft-llm-1")

        # The first stdin input should be the merged models.json
        json_inputs = [b for b in stdin_inputs if b.strip().startswith(b"{")]
        assert json_inputs, "Expected JSON to be piped to container"
        parsed = json.loads(json_inputs[0])
        base_url = parsed["providers"]["local-llm"]["baseUrl"]
        assert "local-llm" in base_url, f"Expected 'local-llm' hostname in baseUrl, got: {base_url}"
        assert base_url.startswith("https://"), "Expected HTTPS scheme"


# ── _tag_as_managed / _list_managed_containers ────────────────────────────────


class TestManagedTag:
    """Tests for container tagging and managed-container discovery."""

    def _make_completed(self, returncode=0, stdout=""):
        p = MagicMock()
        p.returncode = returncode
        p.stdout = stdout
        return p

    def test_tag_as_managed_issues_lxc_config_set(self, monkeypatch):
        """_tag_as_managed should run 'lxc config set <container> user.local-llm-managed=true'."""
        calls: list[list] = []

        def _run(cmd, **kwargs):
            calls.append(list(cmd))
            return self._make_completed(0, "")

        monkeypatch.setattr(subprocess, "run", _run)

        from llm.lxd import _tag_as_managed

        _tag_as_managed("craft-llm-1")

        config_set_calls = [c for c in calls if "config" in c and "set" in c]
        assert config_set_calls, "Expected an 'lxc config set' call"
        full = " ".join(config_set_calls[0])
        assert "user.local-llm-managed=true" in full

    def test_list_managed_containers_returns_tagged_running(self, monkeypatch):
        """Only Running containers with the managed tag should be returned."""
        import json as _json

        instances = [
            {
                "name": "craft-llm-1",
                "status": "Running",
                "config": {"user.local-llm-managed": "true"},
            },
            {
                "name": "craft-llm-2",
                "status": "Stopped",
                "config": {"user.local-llm-managed": "true"},
            },
            {
                "name": "craft-llm-3",
                "status": "Running",
                "config": {},  # not managed
            },
            {
                "name": "other-container",
                "status": "Running",
                "config": {"user.local-llm-managed": "true"},
            },
        ]

        def _run(cmd, **kwargs):
            p = MagicMock()
            p.returncode = 0
            p.stdout = _json.dumps(instances)
            return p

        monkeypatch.setattr(subprocess, "run", _run)

        from llm.lxd import _list_managed_containers

        result = _list_managed_containers()
        # craft-llm-1 is Running + tagged; craft-llm-2 is Stopped; craft-llm-3 not tagged
        # other-container is Running + tagged but also returned (no prefix filter)
        assert "craft-llm-1" in result
        assert "craft-llm-2" not in result, "Stopped containers should be excluded"
        assert "craft-llm-3" not in result, "Untagged containers should be excluded"

    def test_list_managed_containers_empty_when_lxc_fails(self, monkeypatch):
        """If lxc list fails, return an empty list (don't crash)."""

        def _run(cmd, **kwargs):
            p = MagicMock()
            p.returncode = 1
            p.stdout = ""
            return p

        monkeypatch.setattr(subprocess, "run", _run)

        from llm.lxd import _list_managed_containers

        result = _list_managed_containers()
        assert result == []


# ── PATH / bin verification tests ────────────────────────────────────────────


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

        from llm.lxd import CONTAINER_HOME, LxdVmManager

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

        from llm.lxd import LxdVmManager

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

        from llm.lxd import LxdVmManager

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

        from llm.lxd import LxdVmManager

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

        from llm.lxd import LxdVmManager

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

        from llm.lxd import LxdVmManager

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

        from llm.lxd import LxdVmManager

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

        from llm.lxd import LxdVmManager

        mgr = LxdVmManager("test-vm", mounts=[])
        mgr.create_container()

        launch_calls = [c for c in calls if "lxc" in c and "launch" in c]
        assert launch_calls, "Expected an lxc launch command"
        assert "--vm" in launch_calls[0], (
            "lxc launch should use --vm; LXD VMs automatically pass through CPU "
            "virtualisation flags so nested VMs work when the host has nested KVM enabled"
        )
