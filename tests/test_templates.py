"""Tests for the rendered nginx and systemd unit templates."""

from __future__ import annotations

import stat
from pathlib import Path

from llm.render import templates
from llm.settings import ServerSettings, Settings


class TestSystemdUnitRendering:
    """The unit is the only thing that starts llama-server, so it must carry
    every setting that affects the command line."""

    TEMPLATE = Path(__file__).resolve().parents[1] / "systemd" / "llm-server.service.template"

    def _render(self, cfg) -> str:
        text = self.TEMPLATE.read_text()
        for placeholder, value in templates._template_replacements(cfg).items():
            text = text.replace(placeholder, value)
        return text

    def _exec_start(self, unit: str) -> str:
        """Return the ExecStart line with its line continuations collapsed."""
        body = unit.split("ExecStart=", 1)[1]
        collapsed = body.replace("\\\n", " ")
        return " ".join(collapsed.splitlines()[0].split())

    def test_every_placeholder_is_substituted(self):
        """A leftover %%...%% would be passed to llama-server verbatim."""
        rendered = self._render(Settings())
        assert "%%" not in rendered

    def test_extra_args_reach_the_unit(self):
        """Previously the unit had no EXTRA_ARGS placeholder, so systemd and
        the old direct-start path ran different commands."""
        cfg = Settings(server=ServerSettings(extra_args=["--jinja", "--flash-attn"]))
        exec_start = self._exec_start(self._render(cfg))
        assert "--jinja" in exec_start
        assert "--flash-attn" in exec_start

    def test_no_extra_args_leaves_a_clean_command(self):
        cfg = Settings(server=ServerSettings(extra_args=[]))
        unit = self._render(cfg)
        assert "--threads" in unit
        assert not self._exec_start(unit).rstrip().endswith("\\")

    def test_core_server_settings_reach_the_unit(self):
        cfg = Settings(server=ServerSettings(port=9099, n_gpu_layers=42, n_ctx=8192, n_threads=7))
        exec_start = self._exec_start(self._render(cfg))
        assert "--port 9099" in exec_start
        assert "--n-gpu-layers 42" in exec_start
        assert "--ctx-size 8192" in exec_start
        assert "--threads 7" in exec_start


class TestNginxRendering:
    """The proxy conf is what enforces the API key and the LAN allow-list, so
    a missing substitution is a security failure rather than a cosmetic one."""

    TEMPLATE = Path(__file__).resolve().parents[1] / "nginx" / "llm-proxy.conf.template"

    def _render(self, cfg, monkeypatch, bridge=(None, None)) -> str:
        monkeypatch.setattr(templates, "_get_lxd_bridge_info", lambda: bridge)
        text = self.TEMPLATE.read_text()
        for placeholder, value in templates._template_replacements(cfg).items():
            text = text.replace(placeholder, value)
        return text

    def test_every_placeholder_is_substituted(self, monkeypatch):
        assert "%%" not in self._render(Settings(), monkeypatch)

    def test_the_api_key_is_embedded(self, monkeypatch):
        cfg = Settings()
        cfg.auth.api_key = "sk-test-key"
        assert "sk-test-key" in self._render(cfg, monkeypatch)

    def test_ports_and_subnet_reach_the_conf(self, monkeypatch):
        cfg = Settings()
        cfg.server.port = 9099
        cfg.proxy.port = 9443
        cfg.proxy.lan_subnet = "10.1.2.0/24"
        rendered = self._render(cfg, monkeypatch)
        assert "server 127.0.0.1:9099;" in rendered
        assert "listen 9443 ssl;" in rendered
        assert "allow 10.1.2.0/24;" in rendered

    def test_an_lxd_bridge_adds_an_allow_line(self, monkeypatch):
        rendered = self._render(Settings(), monkeypatch, bridge=("10.5.0.1", "10.5.0.0/24"))
        assert "allow 10.5.0.0/24;" in rendered

    def test_no_lxd_bridge_leaves_the_allow_list_alone(self, monkeypatch):
        rendered = self._render(Settings(), monkeypatch)
        assert "\n    deny  all;" in rendered


class TestApplyServerConfigs:
    def test_the_rendered_nginx_conf_is_not_world_readable(self, tmp_path, monkeypatch, fake_console):
        """It embeds the API key in cleartext."""
        monkeypatch.setattr(templates, "_get_lxd_bridge_info", lambda: (None, None))
        monkeypatch.setattr(templates.proc, "sudo_step", lambda *a, **k: False)
        nginx_dir = tmp_path / "nginx"
        nginx_dir.mkdir()
        (nginx_dir / "llm-proxy.conf.template").write_text('key "%%API_KEY%%";\n')

        templates.apply_server_configs(Settings(), tmp_path)

        rendered = nginx_dir / "llm-proxy.conf"
        assert stat.S_IMODE(rendered.stat().st_mode) == 0o600
