"""Tests for the models module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from llm.config import ModelEntry
from llm.models import (
    KNOWN_MODELS,
    _by_alias,
    _by_filename,
    _catalog_table,
    _fmt_size,
    _models_dir,
    _resolve,
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
    def test_resolves_alias_from_catalog(self):
        catalog = [
            ModelEntry(
                alias="qwen2.5-coder-14b-q4",
                repo="a/b",
                filename="Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
            )
        ]
        result = _resolve("qwen2.5-coder-14b-q4", _fallback_list=catalog)
        assert result is not None
        assert result.alias == "qwen2.5-coder-14b-q4"

    def test_resolves_filename_from_catalog(self):
        catalog = [
            ModelEntry(
                alias="qwen2.5-coder-14b-q4",
                repo="a/b",
                filename="Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
            )
        ]
        result = _resolve("Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf", _fallback_list=catalog)
        assert result is not None
        assert result.filename == "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"

    def test_returns_none_for_unknown(self):
        result = _resolve("unknown-thing", _fallback_list=[])
        assert result is None

    def test_falls_back_to_known_models(self):
        catalog = [ModelEntry(alias="other", repo="a/b", filename="other.gguf")]
        result = _resolve("qwen2.5-coder-14b-q4", _fallback_list=catalog)
        assert result is not None
        assert result.alias == "qwen2.5-coder-14b-q4"

    def test_config_catalog_takes_precedence(self):
        catalog = [
            ModelEntry(
                alias="qwen2.5-coder-14b-q4",
                repo="other/repo",
                filename="other.gguf",
            )
        ]
        result = _resolve("qwen2.5-coder-14b-q4", _fallback_list=catalog)
        assert result is not None
        assert result.repo == "other/repo"  # from catalog, not KNOWN_MODELS


# ── _models_dir ───────────────────────────────────────────────────────────────


class TestModelsDir:
    def test_creates_directory(self, tmp_config_with_models_dir):
        result = _models_dir()
        expected = tmp_config_with_models_dir.parent / "models"
        assert result == expected
        assert result.exists()

    def test_returns_configured_directory(self, tmp_config_with_models_dir, monkeypatch):
        config = tmp_config_with_models_dir
        models_dir = tmp_config_with_models_dir.parent / "custom-models"
        content = config.read_text()
        content = content.replace(
            'dir = "' + str(tmp_config_with_models_dir.parent / "models") + '"',
            'dir = "' + str(models_dir) + '"',
        )
        config.write_text(content)
        result = _models_dir()
        assert result == models_dir


# ── _fmt_size ─────────────────────────────────────────────────────────────────


class TestFmtSize:
    def test_formats_small_file(self, monkeypatch):
        fake_stat = MagicMock(st_size=1_000_000)
        monkeypatch.setattr(Path, "stat", lambda self: fake_stat)
        result = _fmt_size(Path("small.gguf"))
        assert "0.0 GB" in result

    def test_formats_large_file(self, monkeypatch):
        fake_stat = MagicMock(st_size=2 * 1_073_741_824)
        monkeypatch.setattr(Path, "stat", lambda self: fake_stat)
        result = _fmt_size(Path("large.gguf"))
        assert "2.0 GB" in result

    def test_format_contains_gb(self, monkeypatch):
        fake_stat = MagicMock(st_size=536_870_912)
        monkeypatch.setattr(Path, "stat", lambda self: fake_stat)
        result = _fmt_size(Path("medium.gguf"))
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


# ── _by_alias / _by_filename with model_list parameter ────────────────────────


class TestByAliasWithList:
    def test_finds_in_custom_list(self):
        catalog = [ModelEntry(alias="custom", repo="a/b", filename="custom.gguf")]
        result = _by_alias("custom", catalog)
        assert result is not None
        assert result.alias == "custom"

    def test_uses_known_models_when_list_empty(self):
        result = _by_alias("qwen2.5-coder-14b-q4", [])
        assert result is not None
        assert result.alias == "qwen2.5-coder-14b-q4"


class TestByFilenameWithList:
    def test_finds_in_custom_list(self):
        catalog = [ModelEntry(alias="custom", repo="a/b", filename="custom.gguf")]
        result = _by_filename("custom.gguf", catalog)
        assert result is not None
        assert result.filename == "custom.gguf"


# ── New commands: init-catalog, show, cost ────────────────────────────────────


class TestInitCatalog:
    def test_init_catalog_adds_entries(self, tmp_path, monkeypatch, fake_console):
        """Test that init-catalog adds model entries to config.toml."""
        config = tmp_path / "config.toml"
        config.write_text('[server]\nllama_server_bin = "llama-server"\nport = 8080\n')
        monkeypatch.chdir(tmp_path)

        from typer.testing import CliRunner  # noqa: PLC0415

        from llm.cli import app  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(app, ["model", "init-catalog"])
        assert result.exit_code == 0, result.output

        # Verify entries were added
        content = config.read_text()
        assert "[[models.list]]" in content
        assert 'alias = "qwen2.5-coder-14b-q4"' in content
        assert 'alias = "qwen3-8b-q8"' in content


class TestModelShow:
    def test_show_existing_model(self, tmp_path, monkeypatch, fake_console):
        """Test showing details for an existing model."""
        config = tmp_path / "config.toml"
        config.write_text('[server]\nllama_server_bin = "llama-server"\nport = 8080\n')
        monkeypatch.chdir(tmp_path)

        from typer.testing import CliRunner  # noqa: PLC0415

        from llm.cli import app  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(app, ["model", "show", "qwen2.5-coder-14b-q4"])
        assert result.exit_code == 0, result.output
        assert "qwen2.5-coder-14b-q4" in fake_console[0]
        assert "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf" in fake_console[2]

    def test_show_unknown_model(self, tmp_path, monkeypatch, fake_console):
        """Test showing details for an unknown model."""
        config = tmp_path / "config.toml"
        config.write_text('[server]\nllama_server_bin = "llama-server"\nport = 8080\n')
        monkeypatch.chdir(tmp_path)

        from typer.testing import CliRunner  # noqa: PLC0415

        from llm.cli import app  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(app, ["model", "show", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in fake_console[0]


class TestModelCost:
    def test_show_cost_for_model(self, tmp_path, monkeypatch, fake_console):
        """Test showing cost for a specific model."""
        config = tmp_path / "config.toml"
        config.write_text('[server]\nllama_server_bin = "llama-server"\nport = 8080\n')
        monkeypatch.chdir(tmp_path)

        from typer.testing import CliRunner  # noqa: PLC0415

        from llm.cli import app  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(app, ["model", "cost", "qwen2.5-coder-14b-q4"])
        assert result.exit_code == 0, result.output
        assert "qwen2.5-coder-14b-q4" in fake_console[0]
        assert "(all costs are zero" in fake_console[5]

    def test_show_cost_all(self, tmp_path, monkeypatch, fake_console):
        """Test showing cost for all models (empty when all zero)."""
        config = tmp_path / "config.toml"
        config.write_text('[server]\nllama_server_bin = "llama-server"\nport = 8080\n')
        monkeypatch.chdir(tmp_path)

        from typer.testing import CliRunner  # noqa: PLC0415

        from llm.cli import app  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(app, ["model", "cost"])
        assert result.exit_code == 0, result.output

    def test_show_cost_unknown(self, tmp_path, monkeypatch, fake_console):
        """Test showing cost for unknown model."""
        config = tmp_path / "config.toml"
        config.write_text('[server]\nllama_server_bin = "llama-server"\nport = 8080\n')
        monkeypatch.chdir(tmp_path)

        from typer.testing import CliRunner  # noqa: PLC0415

        from llm.cli import app  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(app, ["model", "cost", "nonexistent"])
        assert result.exit_code != 0
