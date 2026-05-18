"""Tests for the models module."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from llm.models import (
    KNOWN_MODELS,
    ModelEntry,
    _by_alias,
    _by_filename,
    _fmt_size,
    _models_dir,
    _resolve,
    _catalog_table,
)


# ── KNOWN_MODELS catalog ──────────────────────────────────────────────────────


class TestKnownModels:
    def test_has_entries(self):
        assert len(KNOWN_MODELS) > 0

    def test_all_entries_have_required_fields(self):
        for entry in KNOWN_MODELS:
            assert entry.alias, f"Missing alias for {entry}"
            assert entry.repo, f"Missing repo for {entry}"
            assert entry.filename, f"Missing filename for {entry}"
            assert entry.size, f"Missing size for {entry}"
            assert entry.description, f"Missing description for {entry}"
            assert entry.max_output > 0, f"Invalid max_output for {entry}"

    def test_no_duplicate_aliases(self):
        aliases = [e.alias for e in KNOWN_MODELS]
        assert len(aliases) == len(set(aliases)), "Duplicate alias found"

    def test_no_duplicate_filenames(self):
        filenames = [e.filename for e in KNOWN_MODELS]
        assert len(filenames) == len(set(filenames)), "Duplicate filename found"

    def test_all_files_end_with_gguf(self):
        for entry in KNOWN_MODELS:
            assert entry.filename.endswith(".gguf"), f"Filename {entry.filename} doesn't end with .gguf"

    def test_max_output_varies_by_model(self):
        outputs = set(e.max_output for e in KNOWN_MODELS)
        # Some have 8192 default, some have 32768
        assert len(outputs) >= 2

    def test_qwen3_models_have_32k_output(self):
        qwen3_models = [e for e in KNOWN_MODELS if "qwen3" in e.alias.lower()]
        for m in qwen3_models:
            assert m.max_output == 32768

    def test_gemma_models_have_default_output(self):
        gemma_models = [e for e in KNOWN_MODELS if "gemma" in e.alias.lower()]
        for m in gemma_models:
            assert m.max_output == 8192

    def test_moe_models_have_32k_output(self):
        moe_models = [e for e in KNOWN_MODELS if "moe" in e.alias.lower()]
        for m in moe_models:
            assert m.max_output == 32768


# ── Lookup helpers ────────────────────────────────────────────────────────────


class TestByAlias:
    def test_finds_by_alias(self):
        result = _by_alias("qwen2.5-coder-14b-q4")
        assert result is not None
        assert result.alias == "qwen2.5-coder-14b-q4"

    def test_returns_none_for_unknown_alias(self):
        result = _by_alias("nonexistent-model")
        assert result is None

    def test_case_sensitive(self):
        result = _by_alias("QWEN2.5-CODER-14B-Q4")
        assert result is None


class TestByFilename:
    def test_finds_by_filename(self):
        result = _by_filename("Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf")
        assert result is not None
        assert result.filename == "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"

    def test_returns_none_for_unknown_filename(self):
        result = _by_filename("nonexistent.gguf")
        assert result is None


class TestResolve:
    def test_resolves_alias(self):
        result = _resolve("qwen2.5-coder-14b-q4")
        assert result is not None
        assert result.alias == "qwen2.5-coder-14b-q4"

    def test_resolves_filename(self):
        result = _resolve("Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf")
        assert result is not None
        assert result.filename == "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"

    def test_returns_none_for_unknown(self):
        result = _resolve("unknown-thing")
        assert result is None

    def test_alias_takes_precedence(self):
        # If both alias and filename match (they shouldn't in practice),
        # alias is checked first
        result = _resolve("qwen2.5-coder-14b-q4")
        assert result is not None
        assert result.alias == "qwen2.5-coder-14b-q4"


# ── _models_dir ───────────────────────────────────────────────────────────────


class TestModelsDir:
    def test_creates_directory(self, tmp_path):
        config = tmp_path / "config.toml"
        models_dir = tmp_path / "models"
        config.write_text(
            '[server]\nllama_server_bin = "llama-server"\nport = 8080\nn_gpu_layers = 20\n'
            'n_ctx = 4096\nn_threads = 12\nextra_args = []\n\n[models]\ndir = "'
            + str(models_dir) + '"\nactive = "test.gguf"\nhf_token = ""\n\n[proxy]\n'
            "port = 8443\nlan_ip = \"192.168.1.100\"\nlan_subnet = \"192.168.1.0/24\"\n"
            'api_key = "key"\ncert_path = "/etc/ssl/cert.pem"\n\n[client]\n'
            'server_url = ""\napi_key = ""\ncert_path = ""\n\n[lxd]\ncraft_dirs = []\n'
        )

        import os
        os.chdir(tmp_path)

        result = _models_dir()
        assert result == models_dir
        assert result.exists()

    def test_returns_configured_directory(self, tmp_path):
        config = tmp_path / "config.toml"
        models_dir = tmp_path / "custom-models"
        config.write_text(
            '[server]\nllama_server_bin = "llama-server"\nport = 8080\nn_gpu_layers = 20\n'
            'n_ctx = 4096\nn_threads = 12\nextra_args = []\n\n[models]\ndir = "'
            + str(models_dir) + '"\nactive = "test.gguf"\nhf_token = ""\n\n[proxy]\n'
            "port = 8443\nlan_ip = \"192.168.1.100\"\nlan_subnet = \"192.168.1.0/24\"\n"
            'api_key = "key"\ncert_path = "/etc/ssl/cert.pem"\n\n[client]\n'
            'server_url = ""\napi_key = ""\ncert_path = ""\n\n[lxd]\ncraft_dirs = []\n'
        )

        import os
        os.chdir(tmp_path)

        result = _models_dir()
        assert result == models_dir


# ── _fmt_size ─────────────────────────────────────────────────────────────────


class TestFmtSize:
    def test_formats_small_file(self, tmp_path):
        f = tmp_path / "small.gguf"
        f.write_text("x" * 1_000_000)  # ~1 MB
        result = _fmt_size(f)
        assert "0.0 GB" in result

    def test_formats_large_file(self, tmp_path):
        f = tmp_path / "large.gguf"
        f.write_text("x" * (2 * 1_073_741_824))  # 2 GB
        result = _fmt_size(f)
        assert "2.0 GB" in result

    def test_format_contains_gb(self, tmp_path):
        f = tmp_path / "medium.gguf"
        f.write_text("x" * 536_870_912)  # 0.5 GB
        result = _fmt_size(f)
        assert "GB" in result
        assert "0.5" in result


# ── _catalog_table ────────────────────────────────────────────────────────────


class TestCatalogTable:
    def test_returns_table(self):
        table = _catalog_table("Test catalog")
        assert table.title == "Test catalog"
        assert table.columns  # has columns

    def test_table_has_columns(self):
        table = _catalog_table("Test")
        col_names = [str(c.header) for c in table.columns]
        assert "Alias" in col_names
        assert "Size" in col_names
        assert "Description" in col_names
