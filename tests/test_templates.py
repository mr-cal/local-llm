"""Tests for the rendered nginx and systemd unit templates."""

from __future__ import annotations

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
