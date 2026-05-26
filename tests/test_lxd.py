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
    """Tests for setup_pi_in_container — subprocess calls are mocked."""

    def _make_completed(self, returncode=0, stdout=""):
        p = MagicMock()
        p.returncode = returncode
        p.stdout = stdout
        return p

    def test_adds_hosts_entry_when_bridge_ip_present(self, monkeypatch, tmp_path):
        """When bridge_ip is provided a hosts entry command should be issued."""
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

        setup_pi_in_container("craft-llm-1", bridge_ip="10.113.167.1")

        # At least one call should touch /etc/hosts
        hosts_calls = [c for c in calls if any("/etc/hosts" in str(a) for a in c)]
        assert hosts_calls, "Expected an /etc/hosts manipulation command"

    def test_skips_hosts_entry_when_no_bridge_ip(self, monkeypatch, tmp_path, capsys):
        """When bridge_ip is empty, the /etc/hosts step should be skipped."""
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

        setup_pi_in_container("craft-llm-1", bridge_ip="")

        # No /etc/hosts command should have been issued
        hosts_calls = [c for c in calls if any("/etc/hosts" in str(a) for a in c)]
        assert not hosts_calls, "Expected no /etc/hosts command when bridge_ip is empty"

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

        setup_pi_in_container("craft-llm-1", bridge_ip="10.113.167.1", cert_pem=fake_cert)

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

        # No cert_pem passed — should be read from config cert_path
        setup_pi_in_container("craft-llm-1", bridge_ip="10.113.167.1", cert_pem=None)

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

        setup_pi_in_container("craft-llm-1", bridge_ip="10.113.167.1")

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
