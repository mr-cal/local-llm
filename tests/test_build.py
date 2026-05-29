"""Tests for build.py: BuildProfile, BuildConfig, BACKEND_FLAGS, and build commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import tomli_w
from click.exceptions import Exit as ClickExit

from llm.config import BACKEND_FLAGS, BuildConfig, BuildProfile

# ── BACKEND_FLAGS ───────────────────────────────────────────────────────────


class TestBackendFlags:
    def test_vulkan_present(self):
        assert "vulkan" in BACKEND_FLAGS
        assert BACKEND_FLAGS["vulkan"] == "-DGGML_VULKAN=ON"

    def test_cuda_present(self):
        assert "cuda" in BACKEND_FLAGS
        assert BACKEND_FLAGS["cuda"] == "-DGGML_CUDA=ON"

    def test_metal_present(self):
        assert "metal" in BACKEND_FLAGS

    def test_hipblas_present(self):
        assert "hipblas" in BACKEND_FLAGS

    def test_all_values_are_cmake_flags(self):
        for key, val in BACKEND_FLAGS.items():
            assert val.startswith("-D"), f"{key!r} flag doesn't start with -D"


# ── BuildProfile ─────────────────────────────────────────────────────────────


class TestBuildProfile:
    def test_get_full_flags_backend_only(self):
        p = BuildProfile(name="vk", backend="vulkan")
        assert p.get_full_flags() == ["-DGGML_VULKAN=ON"]

    def test_get_full_flags_extra_flags_only(self):
        p = BuildProfile(name="cpu", extra_flags=["-DFOO=BAR", "-DBAZ=1"])
        assert p.get_full_flags() == ["-DFOO=BAR", "-DBAZ=1"]

    def test_get_full_flags_backend_and_extra(self):
        p = BuildProfile(name="vk-flash", backend="vulkan", extra_flags=["-DGGML_FLASH_ATTN=ON"])
        flags = p.get_full_flags()
        assert "-DGGML_VULKAN=ON" in flags
        assert "-DGGML_FLASH_ATTN=ON" in flags
        assert flags.index("-DGGML_VULKAN=ON") < flags.index("-DGGML_FLASH_ATTN=ON")

    def test_get_full_flags_empty(self):
        p = BuildProfile(name="plain")
        assert p.get_full_flags() == []

    def test_unknown_backend_raises(self):
        with pytest.raises(Exception, match="Unknown backend"):
            BuildProfile(name="bad", backend="notabackend")

    def test_build_dir_name(self):
        p = BuildProfile(name="vulkan-default")
        assert p.build_dir_name == "build-vulkan-default"

    def test_installed_server_bin(self):
        p = BuildProfile(name="myprofile")
        path = p.installed_server_bin(Path("/install"))
        assert path == Path("/install/myprofile/llama-server")

    def test_installed_bench_bin(self):
        p = BuildProfile(name="myprofile")
        path = p.installed_bench_bin(Path("/install"))
        assert path == Path("/install/myprofile/llama-bench")


# ── BuildConfig ──────────────────────────────────────────────────────────────


class TestBuildConfig:
    def _make(self, **kwargs) -> BuildConfig:
        return BuildConfig(**kwargs)

    def test_active_profile_returns_first(self):
        bc = self._make(profiles=[
            BuildProfile(name="a"),
            BuildProfile(name="b"),
        ])
        assert bc.active_profile is not None
        assert bc.active_profile.name == "a"

    def test_active_profile_empty(self):
        bc = self._make()
        assert bc.active_profile is None

    def test_get_profile_by_name(self):
        bc = self._make(profiles=[
            BuildProfile(name="x"),
            BuildProfile(name="y"),
        ])
        p = bc.get_profile("y")
        assert p is not None
        assert p.name == "y"

    def test_get_profile_unknown_returns_none(self):
        bc = self._make(profiles=[BuildProfile(name="a")])
        assert bc.get_profile("zzz") is None

    def test_get_profile_none_returns_active(self):
        bc = self._make(profiles=[BuildProfile(name="first"), BuildProfile(name="second")])
        assert bc.get_profile(None) == bc.active_profile

    def test_profile_names(self):
        bc = self._make(profiles=[BuildProfile(name="p1"), BuildProfile(name="p2")])
        assert bc.profile_names() == ["p1", "p2"]

    def test_profile_names_empty(self):
        bc = self._make()
        assert bc.profile_names() == []

    def test_duplicate_names_raises(self):
        with pytest.raises(Exception, match="Duplicate profile names"):
            BuildConfig(profiles=[BuildProfile(name="dup"), BuildProfile(name="dup")])

    def test_jobs_count_auto(self):
        bc = self._make(jobs="auto")
        count = bc.jobs_count()
        assert count >= 1

    def test_jobs_count_explicit(self):
        bc = self._make(jobs="4")
        assert bc.jobs_count() == 4

    def test_install_path_expands_home(self, tmp_path):
        bc = self._make(install_dir=str(tmp_path / "install"))
        assert bc.install_path == tmp_path / "install"


# ── build commands (mocked subprocess) ──────────────────────────────────────


def _write_config(tmp_path: Path, extra: dict | None = None) -> Path:
    """Write a minimal config.toml for tests."""
    data: dict = {
        "server": {"enabled": True, "llama_server_bin": "llama-server"},
        "models": {"dir": str(tmp_path / "models"), "active": "test.gguf"},
        "build": {
            "install_dir": str(tmp_path / "install"),
            "jobs": "2",
            "profiles": [
                {"name": "vulkan-default", "backend": "vulkan"},
                {"name": "vulkan-flash", "backend": "vulkan", "extra_flags": ["-DGGML_FLASH_ATTN=ON"]},
            ],
        },
    }
    if extra:
        data.update(extra)
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_bytes(tomli_w.dumps(data).encode())
    return tmp_path


def _make_fake_submodule(tmp_path: Path) -> Path:
    """Create a fake llama.cpp submodule directory."""
    sm = tmp_path / "llama.cpp"
    sm.mkdir()
    (sm / "CMakeLists.txt").write_text("# fake")
    return sm


class TestBuildInit:
    def test_init_runs_git_submodule_update(self, tmp_path, monkeypatch):
        from llm import build

        monkeypatch.chdir(_write_config(tmp_path))
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return MagicMock(returncode=0, stdout="abc1234\n")

        with patch("llm.build.subprocess.run", side_effect=fake_run):
            build.init()

        assert any("submodule" in str(c) for c in calls), f"submodule not in calls: {calls}"

    def test_init_skips_if_already_present(self, tmp_path, monkeypatch):
        from llm import build

        monkeypatch.chdir(_write_config(tmp_path))
        _make_fake_submodule(tmp_path)
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return MagicMock(returncode=0, stdout="abc1234\n")

        with patch("llm.build.subprocess.run", side_effect=fake_run):
            build.init()

        submodule_update_calls = [c for c in calls if "submodule" in str(c)]
        assert not submodule_update_calls, "Should not re-run git submodule update when already present"


class TestBuildRun:
    def test_build_run_calls_cmake(self, tmp_path, monkeypatch):
        from llm import build

        monkeypatch.chdir(_write_config(tmp_path))
        _make_fake_submodule(tmp_path)
        cmake_calls = []

        def fake_run(cmd, **kwargs):
            cmake_calls.append(cmd)
            # Simulate cmake creating a fake binary
            if cmd[0] == "cmake" and "--build" not in str(cmd):
                # configure step - create fake bin dir
                build_dir = tmp_path / "llama.cpp" / "build-vulkan-default"
                bin_dir = build_dir / "bin"
                bin_dir.mkdir(parents=True, exist_ok=True)
                (bin_dir / "llama-server").write_text("fake")
                (bin_dir / "llama-bench").write_text("fake")
            return MagicMock(returncode=0)

        with patch("llm.build.subprocess.run", side_effect=fake_run):
            build.build_run(profile="vulkan-default", all_profiles=False)

        configure_calls = [c for c in cmake_calls if "cmake" in c[0] and "--build" not in c]
        assert configure_calls, "No cmake configure call found"
        assert any("-DGGML_VULKAN=ON" in str(c) for c in configure_calls)

    def test_build_run_exits_if_no_submodule(self, tmp_path, monkeypatch):
        from llm import build

        monkeypatch.chdir(_write_config(tmp_path))
        # No submodule directory

        with pytest.raises((SystemExit, ClickExit)):
            build.build_run(profile=None, all_profiles=False)

    def test_build_run_exits_if_no_profiles(self, tmp_path, monkeypatch):
        from llm import build

        no_profiles = {"build": {"install_dir": str(tmp_path / "install"), "profiles": []}}
        monkeypatch.chdir(_write_config(tmp_path, extra=no_profiles))
        _make_fake_submodule(tmp_path)

        with pytest.raises((SystemExit, ClickExit)):
            build.build_run(profile=None, all_profiles=False)

    def test_build_run_unknown_profile_exits(self, tmp_path, monkeypatch):
        from llm import build

        monkeypatch.chdir(_write_config(tmp_path))
        _make_fake_submodule(tmp_path)

        with pytest.raises((SystemExit, ClickExit)):
            build.build_run(profile="does-not-exist", all_profiles=False)


class TestBuildClean:
    def test_clean_removes_build_dir(self, tmp_path, monkeypatch):
        from llm import build

        monkeypatch.chdir(_write_config(tmp_path))
        _make_fake_submodule(tmp_path)
        build_dir = tmp_path / "llama.cpp" / "build-vulkan-default"
        build_dir.mkdir(parents=True)

        build.clean(profile="vulkan-default", all_profiles=False)

        assert not build_dir.exists()

    def test_clean_all_removes_all_build_dirs(self, tmp_path, monkeypatch):
        from llm import build

        monkeypatch.chdir(_write_config(tmp_path))
        _make_fake_submodule(tmp_path)
        for name in ("build-vulkan-default", "build-vulkan-flash"):
            (tmp_path / "llama.cpp" / name).mkdir(parents=True)

        build.clean(profile=None, all_profiles=True)

        for name in ("build-vulkan-default", "build-vulkan-flash"):
            assert not (tmp_path / "llama.cpp" / name).exists()

    def test_clean_exits_if_no_args(self, tmp_path, monkeypatch):
        from llm import build

        monkeypatch.chdir(_write_config(tmp_path))

        with pytest.raises((SystemExit, ClickExit)):
            build.clean(profile=None, all_profiles=False)


class TestInstallPathResolution:
    def test_installed_server_bin_path_structure(self, tmp_path):
        install_dir = tmp_path / "install"
        p = BuildProfile(name="vulkan-default", backend="vulkan")
        expected = install_dir / "vulkan-default" / "llama-server"
        assert p.installed_server_bin(install_dir) == expected

    def test_installed_bench_bin_path_structure(self, tmp_path):
        install_dir = tmp_path / "install"
        p = BuildProfile(name="vulkan-flash", backend="vulkan", extra_flags=["-DGGML_FLASH_ATTN=ON"])
        expected = install_dir / "vulkan-flash" / "llama-bench"
        assert p.installed_bench_bin(install_dir) == expected

    def test_install_path_exists_after_build(self, tmp_path, monkeypatch):
        from llm import build

        monkeypatch.chdir(_write_config(tmp_path))
        _make_fake_submodule(tmp_path)

        def fake_run(cmd, **kwargs):
            if "--build" not in str(cmd):
                build_dir = tmp_path / "llama.cpp" / "build-vulkan-default"
                bin_dir = build_dir / "bin"
                bin_dir.mkdir(parents=True, exist_ok=True)
                (bin_dir / "llama-server").write_text("fake-binary")
                (bin_dir / "llama-bench").write_text("fake-bench")
            return MagicMock(returncode=0)

        with patch("llm.build.subprocess.run", side_effect=fake_run):
            build.build_run(profile="vulkan-default", all_profiles=False)

        install_path = tmp_path / "install" / "vulkan-default" / "llama-server"
        assert install_path.exists()
