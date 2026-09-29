"""Tests for the models module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import click
import pytest

import llm.models as models
from llm.models import (
    KNOWN_MODELS,
    _by_alias,
    _by_filename,
    _catalog_table,
    _flatten_conflicts,
    _fmt_size,
    _models_dir,
    _resolve,
    _set_active_in_config,
    _split_subfolder,
    _verify_size,
)
from llm.settings import ModelEntry

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


# ── config `active` rewriting ─────────────────────────────────────────────────


class TestSetActiveInConfig:
    def test_rewrites_value_in_models_table(self):
        text = '[models]\nactive = "old.gguf"\n'
        assert _set_active_in_config(text, "new-alias") == '[models]\nactive = "new-alias"\n'

    def test_preserves_inline_comment(self):
        text = '[models]\nactive = "old"  # the current model\n'
        assert _set_active_in_config(text, "new") == '[models]\nactive = "new"  # the current model\n'

    def test_preserves_indentation_and_spacing(self):
        text = "[models]\n  active   =   'old'\n"
        assert _set_active_in_config(text, "new") == '[models]\n  active   =   "new"\n'

    def test_handles_single_quoted_values(self):
        text = "[models]\nactive = 'old.gguf'\n"
        assert _set_active_in_config(text, "new") == '[models]\nactive = "new"\n'

    def test_ignores_active_in_other_tables(self):
        text = '[server]\nactive = "keep-me"\n\n[models]\nactive = "old"\n'
        expected = '[server]\nactive = "keep-me"\n\n[models]\nactive = "new"\n'
        assert _set_active_in_config(text, "new") == expected

    def test_ignores_active_in_nested_model_entries(self):
        text = '[[models.list]]\nactive = "nested"\n\n[models]\nactive = "old"\n'
        result = _set_active_in_config(text, "new") or ""
        assert 'active = "nested"' in result
        assert 'active = "new"' in result

    def test_leaves_other_keys_untouched(self):
        text = '[models]\ndir = "~/models"\nactive = "old"\nhf_token = ""\n'
        result = _set_active_in_config(text, "new") or ""
        assert 'dir = "~/models"' in result
        assert 'hf_token = ""' in result

    def test_returns_none_when_key_missing(self):
        assert _set_active_in_config('[models]\ndir = "~/models"\n', "new") is None

    def test_returns_none_when_models_table_missing(self):
        assert _set_active_in_config('[server]\nactive = "x"\n', "new") is None

    def test_preserves_missing_trailing_newline(self):
        assert _set_active_in_config('[models]\nactive = "old"', "new") == '[models]\nactive = "new"'

    def test_rejects_a_value_containing_a_quote(self):
        with pytest.raises(ValueError, match="quote"):
            _set_active_in_config('[models]\nactive = "old"\n', 'ev"il')


# ── download helpers ──────────────────────────────────────────────────────────


class TestSplitSubfolder:
    def test_flat_filename_has_no_subfolder(self):
        assert _split_subfolder("model.gguf") == (None, "model.gguf")

    def test_nested_filename_is_split(self):
        assert _split_subfolder("gguf/model.gguf") == ("gguf", "model.gguf")

    def test_deeply_nested_filename_keeps_full_subfolder(self):
        assert _split_subfolder("a/b/model.gguf") == ("a/b", "model.gguf")


class TestFlattenConflicts:
    def _entry(self, alias, filename):
        return ModelEntry(alias=alias, repo="r/x", filename=filename)

    def test_no_conflict_for_distinct_basenames(self):
        entries = [self._entry("a", "a.gguf"), self._entry("b", "b.gguf")]
        assert _flatten_conflicts("a.gguf", "a.gguf", entries) == []

    def test_detects_entries_that_flatten_onto_the_same_name(self):
        entries = [self._entry("a", "x/model.gguf"), self._entry("b", "y/model.gguf")]
        conflicts = _flatten_conflicts("model.gguf", "x/model.gguf", entries)
        assert [e.alias for e in conflicts] == ["b"]

    def test_the_entry_being_downloaded_is_not_its_own_conflict(self):
        entries = [self._entry("a", "x/model.gguf")]
        assert _flatten_conflicts("model.gguf", "x/model.gguf", entries) == []


class TestVerifySize:
    def test_accepts_matching_size(self, tmp_path):
        f = tmp_path / "m.gguf"
        f.write_bytes(b"12345")
        _verify_size(f, 5)

    def test_accepts_unknown_expected_size(self, tmp_path):
        f = tmp_path / "m.gguf"
        f.write_bytes(b"12345")
        _verify_size(f, None)

    def test_rejects_empty_file(self, tmp_path):
        f = tmp_path / "m.gguf"
        f.touch()
        with pytest.raises(click.exceptions.Exit):
            _verify_size(f, None)

    def test_rejects_short_file(self, tmp_path):
        f = tmp_path / "m.gguf"
        f.write_bytes(b"12")
        with pytest.raises(click.exceptions.Exit):
            _verify_size(f, 100)


# ── download command ──────────────────────────────────────────────────────────


class TestDownload:
    @pytest.fixture
    def models_dir(self, tmp_config_with_models_dir, monkeypatch):
        d = tmp_config_with_models_dir.parent / "models"
        d.mkdir(exist_ok=True)
        monkeypatch.setattr(models, "_remote_size", lambda *a, **kw: None)
        return d

    def _fake_hub(self, monkeypatch, content=b"weights", filename="test.gguf"):
        """Simulate hf_hub_download writing into the staging dir we hand it."""
        import huggingface_hub

        def _download(repo_id, filename=filename, subfolder=None, local_dir="", token=None):
            out = Path(local_dir) / (f"{subfolder}/{filename}" if subfolder else filename)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(content)
            return str(out)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)

    def test_downloads_into_the_models_dir(self, models_dir, monkeypatch):
        self._fake_hub(monkeypatch)
        models.download("repo/id", filename="test.gguf")
        assert (models_dir / "test.gguf").read_bytes() == b"weights"

    def test_saves_nested_files_flat(self, models_dir, monkeypatch):
        self._fake_hub(monkeypatch, filename="model.gguf")
        models.download("repo/id", filename="gguf/model.gguf")
        assert (models_dir / "model.gguf").exists()

    def test_leaves_no_staging_directory_behind(self, models_dir, monkeypatch):
        self._fake_hub(monkeypatch)
        models.download("repo/id", filename="test.gguf")
        assert [p.name for p in models_dir.iterdir()] == ["test.gguf"]

    def test_skips_an_already_downloaded_model(self, models_dir, monkeypatch):
        existing = models_dir / "test.gguf"
        existing.write_bytes(b"original")
        self._fake_hub(monkeypatch)
        with pytest.raises(click.exceptions.Exit) as exc:
            models.download("repo/id", filename="test.gguf")
        assert exc.value.exit_code == 0
        assert existing.read_bytes() == b"original"

    def test_force_overwrites_an_existing_model(self, models_dir, monkeypatch):
        existing = models_dir / "test.gguf"
        existing.write_bytes(b"original")
        self._fake_hub(monkeypatch)
        models.download("repo/id", filename="test.gguf", force=True)
        assert existing.read_bytes() == b"weights"

    def test_an_incomplete_download_never_reaches_the_models_dir(self, models_dir, monkeypatch):
        self._fake_hub(monkeypatch, content=b"tru")
        monkeypatch.setattr(models, "_remote_size", lambda *a, **kw: 9999)
        with pytest.raises(click.exceptions.Exit):
            models.download("repo/id", filename="test.gguf")
        assert list(models_dir.iterdir()) == []

    def test_a_failed_download_leaves_no_staging_directory(self, models_dir, monkeypatch):
        import huggingface_hub

        def _boom(**kwargs):
            raise OSError("network died")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom)
        with pytest.raises(OSError, match="network died"):
            models.download("repo/id", filename="test.gguf")
        assert list(models_dir.iterdir()) == []

    def test_requires_a_filename_for_raw_repo_ids(self, models_dir, monkeypatch):
        with pytest.raises(click.exceptions.Exit):
            models.download("repo/id")


# ── switch command ────────────────────────────────────────────────────────────


class TestSwitch:
    @pytest.fixture
    def catalogued(self, tmp_config_with_models_dir):
        """A config carrying one catalog entry, with its GGUF already on disk."""
        models_dir = tmp_config_with_models_dir.parent / "models"
        tmp_config_with_models_dir.write_text(
            "[server]\n"
            'llama_server_bin = "llama-server"\n'
            "[models]\n"
            f'dir = "{models_dir}"\n'
            'active = "test.gguf"  # current model\n'
            "[[models.list]]\n"
            "alias = 'coder'\n"
            "repo = 'bartowski/Coder'\n"
            "filename = 'Coder-Q4.gguf'\n"
        )
        models_dir.mkdir(exist_ok=True)
        (models_dir / "Coder-Q4.gguf").touch()
        return tmp_config_with_models_dir

    def test_records_the_alias_not_the_filename(self, catalogued):
        models.switch("Coder-Q4.gguf", restart=False)
        assert 'active = "coder"' in catalogued.read_text()

    def test_accepts_an_alias(self, catalogued):
        models.switch("coder", restart=False)
        assert 'active = "coder"' in catalogued.read_text()

    def test_records_the_filename_for_uncatalogued_models(self, catalogued):
        (catalogued.parent / "models" / "custom.gguf").touch()
        models.switch("custom.gguf", restart=False)
        assert 'active = "custom.gguf"' in catalogued.read_text()

    def test_the_new_active_model_loads_back(self, catalogued):
        from llm.settings import load_config

        models.switch("coder", restart=False)
        assert load_config().models.active == "coder"

    def test_rejects_a_missing_model(self, catalogued):
        with pytest.raises(click.exceptions.Exit):
            models.switch("not-here.gguf", restart=False)

    def test_leaves_catalog_entries_untouched(self, catalogued):
        models.switch("coder", restart=False)
        text = catalogued.read_text()
        assert "filename = 'Coder-Q4.gguf'" in text
        assert "alias = 'coder'" in text
