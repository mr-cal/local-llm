"""Tests for the benchmark module."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
import click

import llm.benchmark as benchmark


def _make_proc(returncode=0, stdout="", stderr=""):
    p = MagicMock()
    p.returncode = returncode
    p.stdout = stdout
    p.stderr = stderr
    return p


# ── _parse_bench_csv ──────────────────────────────────────────────────────────


class TestParseBenchCsv:
    def test_parses_valid_csv(self):
        csv_text = "n_gpu_layers,n_prompt,n_gen,avg_ts\n20,512,0,15.3\n20,0,128,25.7\n"
        result = benchmark._parse_bench_csv(csv_text)
        assert len(result) == 2
        assert result[0]["n_gpu_layers"] == "20"
        assert result[0]["avg_ts"] == "15.3"
        assert result[1]["avg_ts"] == "25.7"

    def test_strips_ggml_lines(self):
        csv_text = (
            "ggml_init_cublas: CUDA version 1204\n"
            "llama model loaded\n"
            "main: prompt eval time\n"
            "n_gpu_layers,n_prompt,n_gen,avg_ts\n"
            "20,512,0,15.3\n"
        )
        result = benchmark._parse_bench_csv(csv_text)
        assert len(result) == 1

    def test_returns_empty_for_empty_input(self):
        assert benchmark._parse_bench_csv("") == []

    def test_returns_empty_for_only_header(self):
        result = benchmark._parse_bench_csv("n_gpu_layers,n_prompt,n_gen,avg_ts\n")
        assert result == []

    def test_strips_load_underscore_prefix(self):
        csv_text = "load_: loading model structure\nn_gpu_layers,n_prompt,n_gen,avg_ts\n20,0,128,25.0\n"
        result = benchmark._parse_bench_csv(csv_text)
        assert len(result) == 1

    def test_handles_missing_columns(self):
        csv_text = "n_gpu_layers,n_prompt,avg_ts\n20,512,15.0\n"
        result = benchmark._parse_bench_csv(csv_text)
        assert len(result) == 1

    def test_skips_empty_lines(self):
        csv_text = "\n\nn_gpu_layers,n_prompt,n_gen,avg_ts\n20,512,0,15.0\n\n\n"
        result = benchmark._parse_bench_csv(csv_text)
        assert len(result) == 1

    def test_strips_llama_prefix(self):
        csv_text = "llama_kv_cache_init: kv_size=4096\nn_gpu_layers,n_prompt,n_gen,avg_ts\n20,100,0,10.0\n"
        result = benchmark._parse_bench_csv(csv_text)
        assert len(result) == 1

    def test_strips_main_prefix(self):
        csv_text = "main: starting\nn_gpu_layers,n_prompt,n_gen,avg_ts\n20,100,0,10.0\n"
        result = benchmark._parse_bench_csv(csv_text)
        assert len(result) == 1

    def test_multiple_log_lines_before_header(self):
        csv_text = (
            "ggml_cuda_init: GPU 0: NVIDIA\n"
            "load_: loading model\n"
            "main: starting\n"
            "llama_kv_cache_init: kv_size=4096\n"
            "n_gpu_layers,n_prompt,n_gen,avg_ts\n"
            "20,100,0,10.0\n"
        )
        result = benchmark._parse_bench_csv(csv_text)
        assert len(result) == 1


# ── _bench_tps ────────────────────────────────────────────────────────────────


class TestBenchTps:
    def test_returns_pp_and_tg(self):
        rows = [
            {"n_prompt": "512", "n_gen": "0", "avg_ts": "15.3"},
            {"n_prompt": "0", "n_gen": "128", "avg_ts": "25.7"},
        ]
        pp, tg = benchmark._bench_tps(rows)
        assert pp == pytest.approx(15.3)
        assert tg == pytest.approx(25.7)

    def test_averages_multiple_rows(self):
        rows = [
            {"n_prompt": "512", "n_gen": "0", "avg_ts": "10.0"},
            {"n_prompt": "512", "n_gen": "0", "avg_ts": "20.0"},
            {"n_prompt": "0", "n_gen": "128", "avg_ts": "30.0"},
            {"n_prompt": "0", "n_gen": "128", "avg_ts": "50.0"},
        ]
        pp, tg = benchmark._bench_tps(rows)
        assert pp == pytest.approx(15.0)
        assert tg == pytest.approx(40.0)

    def test_filters_by_gpu_layers(self):
        rows = [
            {"n_gpu_layers": "0", "n_prompt": "512", "n_gen": "0", "avg_ts": "5.0"},
            {"n_gpu_layers": "20", "n_prompt": "512", "n_gen": "0", "avg_ts": "15.0"},
            {"n_gpu_layers": "20", "n_prompt": "0", "n_gen": "128", "avg_ts": "25.0"},
        ]
        pp, tg = benchmark._bench_tps(rows, n_gpu_layers=20)
        assert pp == pytest.approx(15.0)
        assert tg == pytest.approx(25.0)

    def test_filters_by_flash_attn(self):
        rows = [
            {"n_prompt": "512", "n_gen": "0", "avg_ts": "10.0", "flash_attn": "0"},
            {"n_prompt": "512", "n_gen": "0", "avg_ts": "20.0", "flash_attn": "1"},
        ]
        pp_off, _ = benchmark._bench_tps(rows, flash_attn=0)
        pp_on, _ = benchmark._bench_tps(rows, flash_attn=1)
        assert pp_off == pytest.approx(10.0)
        assert pp_on == pytest.approx(20.0)

    def test_filters_by_type_k(self):
        rows = [
            {"n_prompt": "512", "n_gen": "0", "avg_ts": "10.0", "type_k": "f16"},
            {"n_prompt": "512", "n_gen": "0", "avg_ts": "12.0", "type_k": "q8_0"},
        ]
        pp_f16, _ = benchmark._bench_tps(rows, type_k="f16")
        pp_q8, _ = benchmark._bench_tps(rows, type_k="q8_0")
        assert pp_f16 == pytest.approx(10.0)
        assert pp_q8 == pytest.approx(12.0)

    def test_averages_mixed_gpu_layers_when_no_filter(self):
        rows = [
            {"n_gpu_layers": "0", "n_prompt": "512", "n_gen": "0", "avg_ts": "5.0"},
            {"n_gpu_layers": "20", "n_prompt": "512", "n_gen": "0", "avg_ts": "15.0"},
        ]
        pp, _ = benchmark._bench_tps(rows)
        assert pp == pytest.approx(10.0)

    def test_empty_result_with_no_match(self):
        rows = [
            {"n_gpu_layers": "0", "n_prompt": "512", "n_gen": "0", "avg_ts": "5.0"},
        ]
        pp, tg = benchmark._bench_tps(rows, n_gpu_layers=20)
        assert pp == 0.0
        assert tg == 0.0

    def test_handles_invalid_avg_ts(self):
        rows = [
            {"n_prompt": "512", "n_gen": "0", "avg_ts": "not-a-number"},
            {"n_prompt": "512", "n_gen": "0", "avg_ts": "10.0"},
        ]
        pp, _ = benchmark._bench_tps(rows)
        assert pp == pytest.approx(10.0)

    def test_handles_invalid_prompt_gen_values(self):
        rows = [
            {"n_prompt": "bad", "n_gen": "0", "avg_ts": "10.0"},
        ]
        pp, tg = benchmark._bench_tps(rows)
        assert pp == 0.0
        assert tg == 0.0

    def test_handles_both_prompt_and_gen_nonzero(self):
        rows = [
            {"n_prompt": "100", "n_gen": "50", "avg_ts": "15.0"},
        ]
        pp, tg = benchmark._bench_tps(rows)
        assert pp == 0.0
        assert tg == 0.0

    def test_handles_both_prompt_and_gen_zero(self):
        rows = [
            {"n_prompt": "0", "n_gen": "0", "avg_ts": "15.0"},
        ]
        pp, tg = benchmark._bench_tps(rows)
        assert pp == 0.0
        assert tg == 0.0

    def test_classifies_prompt_vs_gen(self):
        rows = [
            {"n_prompt": "512", "n_gen": "0", "avg_ts": "10.0"},
            {"n_prompt": "0", "n_gen": "128", "avg_ts": "20.0"},
            {"n_prompt": "512", "n_gen": "0", "avg_ts": "15.0"},
            {"n_prompt": "0", "n_gen": "128", "avg_ts": "25.0"},
        ]
        pp, tg = benchmark._bench_tps(rows)
        assert pp == pytest.approx(12.5)
        assert tg == pytest.approx(22.5)

    def test_no_rows(self):
        pp, tg = benchmark._bench_tps([])
        assert pp == 0.0
        assert tg == 0.0

    def test_multiple_gpu_layers_with_filter(self):
        rows = [
            {"n_gpu_layers": "0", "n_prompt": "512", "n_gen": "0", "avg_ts": "3.0"},
            {"n_gpu_layers": "20", "n_prompt": "512", "n_gen": "0", "avg_ts": "10.0"},
            {"n_gpu_layers": "20", "n_prompt": "0", "n_gen": "128", "avg_ts": "20.0"},
            {"n_gpu_layers": "48", "n_prompt": "512", "n_gen": "0", "avg_ts": "15.0"},
            {"n_gpu_layers": "48", "n_prompt": "0", "n_gen": "128", "avg_ts": "25.0"},
        ]
        pp, tg = benchmark._bench_tps(rows, n_gpu_layers=20)
        assert pp == pytest.approx(10.0)
        assert tg == pytest.approx(20.0)

    def test_combined_filters(self):
        rows = [
            {"n_gpu_layers": "20", "flash_attn": "0", "n_prompt": "512", "n_gen": "0", "avg_ts": "10.0"},
            {"n_gpu_layers": "20", "flash_attn": "1", "n_prompt": "512", "n_gen": "0", "avg_ts": "15.0"},
            {"n_gpu_layers": "20", "flash_attn": "1", "n_prompt": "0", "n_gen": "128", "avg_ts": "25.0"},
        ]
        pp, tg = benchmark._bench_tps(rows, n_gpu_layers=20, flash_attn=1)
        assert pp == pytest.approx(15.0)
        assert tg == pytest.approx(25.0)


# ── _run_llama_bench ──────────────────────────────────────────────────────────


class TestRunLlamaBench:
    def test_runs_bench_with_correct_args(self, tmp_path, fake_console):
        bench = tmp_path / "llama-bench"
        bench.write_text("#!/bin/bash\necho 'n_gpu_layers,n_prompt,n_gen,avg_ts'\n"
                         "echo '20,512,0,10.0'\n")
        bench.chmod(0o755)

        model = tmp_path / "model.gguf"
        model.touch()

        rows = benchmark._run_llama_bench(
            bench_bin=bench,
            model_path=model,
            n_threads=12,
            ngl_values=[20, 48],
            flash_attn_values=[0, 1],
            ctk_values=["f16"],
            n_prompt=512,
            n_gen=128,
            repetitions=2,
        )
        assert len(rows) == 1

    def test_returns_empty_on_failure(self, tmp_path, fake_console):
        bench = tmp_path / "llama-bench"
        bench.write_text("#!/bin/bash\necho 'error'\nexit 1\n")
        bench.chmod(0o755)

        model = tmp_path / "model.gguf"
        model.touch()

        rows = benchmark._run_llama_bench(
            bench_bin=bench,
            model_path=model,
            n_threads=12,
            ngl_values=[20],
            flash_attn_values=[0],
            ctk_values=["f16"],
        )
        assert rows == []


# ── _apply_config ─────────────────────────────────────────────────────────────


class TestApplyConfig:
    def test_updates_n_gpu_layers(self, tmp_config_bench, fake_console, monkeypatch):
        monkeypatch.chdir(tmp_config_bench.parent)

        benchmark._apply_config(n_gpu_layers=48, flash_attn=False, ctk="f16")
        content = tmp_config_bench.read_text()
        assert "n_gpu_layers = 48" in content

    def test_updates_extra_args(self, tmp_config_bench, fake_console, monkeypatch):
        monkeypatch.chdir(tmp_config_bench.parent)

        benchmark._apply_config(n_gpu_layers=20, flash_attn=True, ctk="q8_0")
        content = tmp_config_bench.read_text()
        assert "--flash-attn" in content
        assert "--cache-type-k" in content
        assert "q8_0" in content


# ── _ensure_history_file / _append_result ─────────────────────────────────────


class TestHistoryFileOps:
    def test_creates_history_file(self, tmp_path, fake_console):
        history_path = tmp_path / "logs" / "benchmark-history.csv"
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(benchmark, "HISTORY_FILE", history_path)

        benchmark._ensure_history_file()

        assert history_path.exists()
        content = history_path.read_text()
        assert "timestamp" in content

    def test_appends_result(self, tmp_path, fake_console):
        history_path = tmp_path / "logs" / "benchmark-history.csv"
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(benchmark, "HISTORY_FILE", history_path)

        benchmark._ensure_history_file()

        benchmark._append_result({
            "timestamp": "2026-01-01T00:00:00",
            "model": "test-model",
            "backend": "llama-server",
            "pp_tps": "10.0",
            "tg_tps": "20.0",
            "ctx": "4096",
            "n_tokens": 100,
            "n_gpu_layers": 20,
        })

        content = history_path.read_text()
        assert "test-model" in content
        assert "10.0" in content


# ── _build_extra_args ─────────────────────────────────────────────────────────


class TestBuildExtraArgs:
    def test_empty_when_flash_off_and_f16(self):
        result = benchmark._build_extra_args(False, "f16", [])
        assert result == []

    def test_includes_flash_attn(self):
        result = benchmark._build_extra_args(True, "f16", [])
        assert "--flash-attn" in result

    def test_includes_cache_type(self):
        result = benchmark._build_extra_args(False, "q8_0", [])
        assert "--cache-type-k" in result
        assert "q8_0" in result

    def test_preserves_non_tuning_args(self):
        result = benchmark._build_extra_args(True, "f16", ["--jinja", "--log-disable"])
        assert "--jinja" in result
        assert "--log-disable" in result
        assert "--flash-attn" in result

    def test_removes_old_flash_attn_when_turned_off(self):
        result = benchmark._build_extra_args(False, "f16", ["--flash-attn", "--jinja"])
        assert "--flash-attn" not in result
        assert "--jinja" in result

    def test_removes_old_cache_type_when_f16(self):
        result = benchmark._build_extra_args(False, "f16", ["--cache-type-k", "q8_0", "--jinja"])
        assert "--cache-type-k" not in result
        assert "q8_0" not in result
        assert "--jinja" in result

    def test_keeps_flash_attn_when_already_present(self):
        result = benchmark._build_extra_args(True, "f16", ["--flash-attn"])
        assert result.count("--flash-attn") == 1

    def test_replaces_q4_0_cache_type(self):
        result = benchmark._build_extra_args(False, "f16", ["--cache-type-k", "q4_0"])
        assert "--cache-type-k" not in result
        assert "q4_0" not in result

    def test_replaces_q4_1_cache_type(self):
        result = benchmark._build_extra_args(False, "f16", ["--cache-type-k", "q4_1"])
        assert "--cache-type-k" not in result
        assert "q4_1" not in result


# ── _run_llama_bench_raw ──────────────────────────────────────────────────────


class TestRunLlamaBenchRaw:
    def test_skips_when_no_bench(self, tmp_config_bench, fake_console, monkeypatch):
        monkeypatch.chdir(tmp_config_bench.parent)

        from llm.config import load_config
        cfg = load_config()

        def fake_find():
            return None

        with patch.object(benchmark, "_find_bench_bin", fake_find):
            benchmark._run_llama_bench_raw(cfg)


# ── history command ───────────────────────────────────────────────────────────


class TestHistoryCommand:
    def test_no_history_shows_warning(self, tmp_path, fake_console, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(benchmark, "HISTORY_FILE", Path("/nonexistent/file.csv"))

        with pytest.raises(click.exceptions.Exit):
            benchmark.history()

    def test_empty_history_shows_warning(self, tmp_bench_history, fake_console):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(benchmark, "HISTORY_FILE", tmp_bench_history)

        # No data rows -> returns without exit
        benchmark.history()

    def test_shows_last_n_records(self, tmp_path, fake_console, monkeypatch):
        monkeypatch.chdir(tmp_path)

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        history_file = logs_dir / "benchmark-history.csv"
        history_file.write_text(
            "timestamp,model,backend,pp_tps,tg_tps,ctx,n_tokens,n_gpu_layers\n"
            "2026-01-01,model1,llama-server,10.0,20.0,4096,100,20\n"
            "2026-01-02,model2,llama-server,12.0,22.0,8192,200,30\n"
            "2026-01-03,model3,llama-server,14.0,24.0,16384,300,48\n"
        )

        monkeypatch.setattr(benchmark, "HISTORY_FILE", history_file)

        benchmark.history(last=2)

    def test_history_show_all_when_n_large(self, tmp_path, fake_console, monkeypatch):
        monkeypatch.chdir(tmp_path)

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        history_file = logs_dir / "benchmark-history.csv"
        history_file.write_text(
            "timestamp,model,backend,pp_tps,tg_tps,ctx,n_tokens,n_gpu_layers\n"
            "2026-01-01,model1,llama-server,10.0,20.0,4096,100,20\n"
            "2026-01-02,model2,llama-server,12.0,22.0,8192,200,30\n"
        )

        monkeypatch.setattr(benchmark, "HISTORY_FILE", history_file)

        benchmark.history(last=100)


# ── Module-level constants ───────────────────────────────────────────────────


class TestModuleConstants:
    def test_history_file_path(self):
        assert benchmark.HISTORY_FILE.name == "benchmark-history.csv"

    def test_history_headers(self):
        expected = ["timestamp", "model", "backend", "pp_tps", "tg_tps", "ctx", "n_tokens", "n_gpu_layers"]
        assert benchmark.HISTORY_HEADERS == expected

    def test_default_prompt(self):
        assert len(benchmark._DEFAULT_PROMPT) > 0
        assert "Python" in benchmark._DEFAULT_PROMPT
        assert "prime" in benchmark._DEFAULT_PROMPT.lower()


# ── App definition ───────────────────────────────────────────────────────────


class TestAppDefinition:
    def test_app_is_typer(self):
        assert hasattr(benchmark, "app")

    def test_app_help(self):
        # app is a Typer object with registered commands
        assert hasattr(benchmark, "app")
        assert hasattr(benchmark.app, "command")
