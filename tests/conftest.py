from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def tmp_config_cwd(tmp_config: Path, monkeypatch) -> Path:
    """Write a standard config.toml and chdir to its parent.

    Combines the common pattern of tmp_config + monkeypatch.chdir.
    """
    monkeypatch.chdir(tmp_config.parent)
    return tmp_config


@pytest.fixture
def tmp_config_with_lxd(tmp_path: Path, monkeypatch) -> Path:
    """Write a config.toml with an lxd section (mounts) and chdir to parent."""
    config = tmp_path / "config.toml"
    config.write_text(
        '[server]\nllama_server_bin = "llama-server"\nport = 8080\nn_gpu_layers = 20\n'
        'n_ctx = 4096\nn_threads = 12\nextra_args = []\n\n[models]\ndir = "~/models"\n'
        'active = "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"\nhf_token = ""\n\n[proxy]\n'
        "port = 8443\nlan_ip = \"192.168.1.100\"\nlan_subnet = \"192.168.1.0/24\"\n"
        'api_key = "test-key"\ncert_path = "/etc/ssl/cert.pem"\n\n[client]\n'
        'server_url = ""\napi_key = ""\ncert_path = ""\n\n[lxd]\ncraft_dirs = ["~/dev/craft"]\n'
        "\n[[lxd.mounts]]\n"
        'host = "~/.agents"\n'
        "\n[[lxd.mounts]]\n"
        'host = "~/dev"\n'
    )
    monkeypatch.chdir(tmp_path)
    return config


@pytest.fixture
def tmp_config_full(tmp_path: Path, monkeypatch) -> Path:
    """Write a full config.toml with all sections and chdir to parent.

    Suitable for tests that exercise config-init, config-gencert, config-apply.
    """
    config = tmp_path / "config.toml"
    config.write_text(
        '[server]\nllama_server_bin = "llama-server"\nport = 8080\nn_gpu_layers = 20\n'
        'n_ctx = 4096\nn_threads = 12\nextra_args = []\n\n[models]\ndir = "~/models"\n'
        'active = "model.gguf"\nhf_token = ""\n\n[proxy]\nport = 8443\n'
        'lan_ip = "192.168.1.100"\nlan_subnet = "192.168.1.0/24"\n'
        'api_key = "test-key"\ncert_path = "cert.pem"\n\n[client]\n'
        'server_url = ""\napi_key = ""\ncert_path = ""\n\n[lxd]\ncraft_dirs = []\n'
    )
    monkeypatch.chdir(tmp_path)
    return config


@pytest.fixture
def tmp_config_with_models_dir(tmp_path: Path, monkeypatch) -> Path:
    """Write a full config.toml with a configurable models directory and chdir to parent.

    Returns the path to the config file. The caller can read/modify the file
    before using it with load_config() or _models_dir().
    """
    config = tmp_path / "config.toml"
    models_dir = tmp_path / "models"
    config.write_text(
        '[server]\nllama_server_bin = "llama-server"\nport = 8080\n'
        'n_gpu_layers = 20\nn_ctx = 4096\nn_threads = 12\n'
        'extra_args = []\n\n[models]\ndir = "'
        + str(models_dir)
        + '"\nactive = "test.gguf"\nhf_token = ""\n'
        '\n[proxy]\nport = 8443\n'
        'lan_ip = "192.168.1.100"\nlan_subnet = "192.168.1.0/24"\n'
        'api_key = "key"\ncert_path = "/etc/ssl/cert.pem"\n'
        '\n[client]\nserver_url = ""\napi_key = ""\ncert_path = ""\n'
        '\n[lxd]\ncraft_dirs = []\n'
    )
    monkeypatch.chdir(tmp_path)
    return config


@pytest.fixture
def tmp_config_server(tmp_path: Path, monkeypatch):
    """Set up a full server test environment.

    Creates config.toml with a working server config, a models directory
    with model.gguf, and chdirs to tmp_path. Returns the tuple
    (config_path, tmp_path) so tests can modify the config or inspect
    created files.
    """
    config = tmp_path / "config.toml"
    models_dir = tmp_path / "models"
    config.write_text(
        '[server]\nllama_server_bin = "llama-server"\nport = 8080\n'
        'n_gpu_layers = 20\nn_ctx = 4096\nn_threads = 12\n'
        'extra_args = []\n\n[models]\ndir = "'
        + str(models_dir)
        + '"\nactive = "model.gguf"\nhf_token = ""\n'
        '\n[proxy]\nport = 8443\n'
        'lan_ip = "192.168.1.100"\nlan_subnet = "192.168.1.0/24"\n'
        'api_key = "key"\ncert_path = "/etc/ssl/cert.pem"\n'
        '\n[client]\nserver_url = ""\napi_key = ""\ncert_path = ""\n'
        '\n[lxd]\ncraft_dirs = []\n'
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "model.gguf").touch()
    monkeypatch.chdir(tmp_path)
    return config, tmp_path


@pytest.fixture
def tmp_config_gencert(tmp_path: Path, monkeypatch) -> Path:
    """Write a full config.toml with relative cert_path and chdir to parent.

    Similar to tmp_config_full but with cert_path pointing to tmp_path,
    which is required for config-gencert tests (certs are written to config dir).
    """
    config = tmp_path / "config.toml"
    config.write_text(
        '[server]\nllama_server_bin = "llama-server"\nport = 8080\nn_gpu_layers = 20\n'
        'n_ctx = 4096\nn_threads = 12\nextra_args = []\n\n[models]\ndir = "~/models"\n'
        'active = "model.gguf"\nhf_token = ""\n\n[proxy]\nport = 8443\n'
        'lan_ip = "192.168.1.100"\nlan_subnet = "192.168.1.0/24"\n'
        'api_key = "test-key"\ncert_path = "cert.pem"\n\n[client]\n'
        'server_url = ""\napi_key = ""\ncert_path = ""\n\n[lxd]\ncraft_dirs = []\n'
    )
    monkeypatch.chdir(tmp_path)
    return config


@pytest.fixture
def mock_urlopen(monkeypatch):
    """Mock urllib.request.urlopen to raise an exception (no-network simulation).

    Returns the patched open function for tests that need to re-use it.
    """
    def _urlopen(*a, **kw):
        raise Exception("no network")
    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    return _urlopen


@pytest.fixture
def tmp_config_bench(tmp_path: Path) -> Path:
    """Write a minimal benchmark config.toml and return its path.

    Suitable for tests that exercise _apply_config or _run_llama_bench_raw.
    """
    config = tmp_path / "config.toml"
    config.write_text(
        '[server]\nllama_server_bin = "llama-server"\nport = 8080\nn_gpu_layers = 20\n'
        'n_ctx = 4096\nn_threads = 12\nextra_args = []\n\n[models]\ndir = "~/models"\n'
        'active = "model.gguf"\nhf_token = ""\n\n[proxy]\nport = 8443\n'
        'lan_ip = "192.168.1.100"\nlan_subnet = "192.168.1.0/24"\n'
        'api_key = "key"\ncert_path = "/etc/ssl/cert.pem"\n\n[client]\n'
        'server_url = ""\napi_key = ""\ncert_path = ""\n\n[lxd]\ncraft_dirs = []\n'
    )
    return config


@pytest.fixture
def tmp_bench_history(tmp_path: Path, monkeypatch):
    """Create a benchmark history CSV file under tmp_path/logs and chdir to tmp_path.

    Returns the path to the history file.
    """
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    history_file = logs_dir / "benchmark-history.csv"
    history_file.write_text(
        "timestamp,model,backend,pp_tps,tg_tps,ctx,n_tokens,n_gpu_layers\n"
    )
    monkeypatch.chdir(tmp_path)
    return history_file


@pytest.fixture
def fake_console(monkeypatch):
    """Capture console.print() calls for modules using the shared console."""
    calls = []

    def _print(*args, **kwargs):
        calls.append(str(args[0]) if args else "")

    monkeypatch.setattr("llm.config.console.print", lambda *a, **kw: _print(*a, **kw))
    monkeypatch.setattr("llm.server.console.print", lambda *a, **kw: _print(*a, **kw))
    monkeypatch.setattr("llm.benchmark.console.print", lambda *a, **kw: _print(*a, **kw))
    monkeypatch.setattr("llm.client.console.print", lambda *a, **kw: _print(*a, **kw))
    monkeypatch.setattr("llm.models.console.print", lambda *a, **kw: _print(*a, **kw))
    return calls


@pytest.fixture
def fake_file(tmp_path: Path):
    """Create a file under tmp_path with configurable content and return its path."""
    def _inner(name: str = "file", content: bytes | str = b"") -> Path:
        f = tmp_path / name
        f.write_bytes(content if isinstance(content, bytes) else content.encode())
        return f
    return _inner


@pytest.fixture
def _make_proc():
    """Create a MagicMock that mimics subprocess.CompletedProcess."""
    def _inner(
        returncode: int = 0, stdout: str = "", stderr: str = "", stdout_bytes: bytes = b""
    ) -> MagicMock:
        p = MagicMock()
        p.returncode = returncode
        p.stdout = stdout
        p.stderr = stderr
        p.stdout_bytes = stdout_bytes
        return p
    return _inner


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    """Write a minimal valid config.toml and return its path.

    The caller can read/modify the file before using it.
    """
    config = tmp_path / "config.toml"
    config.write_text(
        '[server]\nllama_server_bin = "llama-server"\nport = 8080\nn_gpu_layers = 20\n'
        'n_ctx = 4096\nn_threads = 12\nextra_args = []\n\n[models]\ndir = "~/models"\n'
        'active = "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"\nhf_token = ""\n\n[proxy]\n'
        "port = 8443\nlan_ip = \"192.168.1.100\"\nlan_subnet = \"192.168.1.0/24\"\n"
        'api_key = "test-key-1234"\ncert_path = "/etc/ssl/local-llm/cert.pem"\n\n[client]\n'
        'server_url = ""\napi_key = ""\ncert_path = ""\n\n[lxd]\n'
        "craft_dirs = []\n"
    )
    return config


@pytest.fixture
def tmp_client_config(tmp_path: Path) -> Path:
    """Write a client-test config.toml and return its path.

    Suitable for tests exercising the client module's setup command.
    The caller can read/modify the file before using it.
    """
    config = tmp_path / "config.toml"
    config.write_text(
        '[server]\nllama_server_bin = "llama-server"\nport = 8080\nn_gpu_layers = 20\n'
        'n_ctx = 4096\nn_threads = 12\nextra_args = []\n\n[models]\ndir = "~/models"\n'
        'active = "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"\nhf_token = ""\n\n[proxy]\n'
        "port = 8443\nlan_ip = \"192.168.1.100\"\nlan_subnet = \"192.168.1.0/24\"\n"
        'api_key = "test-key"\ncert_path = "/etc/ssl/cert.pem"\n\n[client]\n'
        'server_url = ""\napi_key = ""\ncert_path = ""\n\n[lxd]\n'
        "craft_dirs = []\n"
    )
    return config


@pytest.fixture
def fake_find_config(monkeypatch, tmp_path: Path):
    """Patch client.find_config to return a path, and chdir to tmp_path.

    Returns the path so the caller can write/modify the config file.
    """
    config_path = tmp_path / "config.toml"

    def _fake_find():
        return config_path

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("llm.client.find_config", _fake_find)
    return config_path


@pytest.fixture
def fake_no_config(monkeypatch, tmp_path: Path):
    """Patch client.find_config to return a nonexistent path, and chdir to tmp_path."""
    nonexistent = tmp_path / "nonexistent.toml"

    def _fake_find():
        return nonexistent

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("llm.client.find_config", _fake_find)
    return nonexistent


@pytest.fixture
def tmp_config_client_only(tmp_path: Path) -> Path:
    """Write a client-only config.toml (no [server] section)."""
    config = tmp_path / "config.toml"
    config.write_text(
        '[proxy]\nport = 8443\nlan_ip = "10.0.0.5"\n'
        'lan_subnet = "10.0.0.0/24"\napi_key = "remote-key"\n'
        'cert_path = "/etc/ssl/local-llm/cert.pem"\n\n[client]\n'
        'server_url = "https://10.0.0.5:8443/v1"\napi_key = "remote-key"\n'
        'cert_path = "/home/user/.config/local-llm/cert.pem"\n\n'
        "[lxd]\ncraft_dirs = []\n"
    )
    return config


@pytest.fixture
def mock_httpx_get(monkeypatch):
    """Return a mock httpx.get that simulates server health."""

    def _get(url, **kw):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        return resp

    monkeypatch.setattr("httpx.get", _get)
    return _get


@pytest.fixture
def mock_httpx_post(monkeypatch):
    """Return a mock httpx.post that simulates a chat completion."""

    def _post(url, json=None, **kw):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Here is the function you requested.",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 42,
                "completion_tokens": 128,
            },
            "timings": {
                "prompt_per_second": 15.3,
                "predicted_per_second": 45.2,
            },
        }
        return resp

    monkeypatch.setattr("httpx.post", _post)
    return _post


@pytest.fixture
def mock_subprocess_run(monkeypatch, _make_proc):
    """Return a function that creates a fake CompletedProcess."""
    fake_calls = []

    def _run(cmd, **kwargs):
        fake_calls.append(list(cmd))
        return _make_proc(0, "")

    monkeypatch.setattr(subprocess, "run", _run)
    return _run, fake_calls


@pytest.fixture
def mock_subprocess_run_fail(monkeypatch, _make_proc):
    """Return a function that creates a fake CompletedProcess with failure."""

    def _run(cmd, **kwargs):
        return _make_proc(1, "", "command not found")

    monkeypatch.setattr(subprocess, "run", _run)
    return _run


@pytest.fixture
def mock_subprocess_run_active(monkeypatch, _make_proc):
    """Return a function that reports a unit/service as active."""

    def _run(cmd, **kwargs):
        cmd_str = " ".join(str(c) for c in cmd)
        if "is-active" in cmd_str:
            return _make_proc(0, "active")
        if "daemon-reload" in cmd_str:
            return _make_proc(0, "")
        if "enable" in cmd_str:
            return _make_proc(0, "")
        return _make_proc(0, "")

    monkeypatch.setattr(subprocess, "run", _run)
    return _run


@pytest.fixture
def mock_subprocess_run_not_active(monkeypatch, _make_proc):
    """Return a function that reports a unit/service as not active."""

    def _run(cmd, **kwargs):
        cmd_str = " ".join(str(c) for c in cmd)
        if "is-active" in cmd_str:
            return _make_proc(1, "inactive")
        return _make_proc(0, "")

    monkeypatch.setattr(subprocess, "run", _run)
    return _run


@pytest.fixture
def mock_systemctl_start(monkeypatch, _make_proc):
    """Return a function that simulates successful systemctl start."""

    def _run(cmd, **kwargs):
        cmd_str = " ".join(str(c) for c in cmd)
        if "start" in cmd_str or "reload" in cmd_str or "stop" in cmd_str:
            return _make_proc(0, "")
        return _make_proc(0, "")

    monkeypatch.setattr(subprocess, "run", _run)
    return _run


@pytest.fixture
def mock_model_entry():
    """A sample ModelEntry for testing."""
    from llm.models import ModelEntry

    return ModelEntry(
        alias="test-model",
        repo="test/repo",
        filename="test-model.gguf",
        size="~1 GB",
        description="A test model",
        max_output=4096,
    )


@pytest.fixture
def mock_model_entry_long_output():
    """A ModelEntry with large max_output."""
    from llm.models import ModelEntry

    return ModelEntry(
        alias="big-model",
        repo="test/big",
        filename="big-model.gguf",
        size="~4 GB",
        description="A big model",
        max_output=32768,
    )


@pytest.fixture
def fake_pid_file(tmp_path: Path):
    """Write a fake PID file and return its path."""
    pid_file = tmp_path / ".server.pid"
    pid_file.write_text("12345")
    return pid_file


@pytest.fixture
def fake_log_file(tmp_path: Path):
    """Write a fake log file and return its path."""
    log_file = tmp_path / ".server.log"
    log_file.write_text("line 1\nline 2\n")
    return log_file


@pytest.fixture
def fake_models_dir(tmp_path: Path):
    """Create a models directory with a fake GGUF file."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf").write_text("fake model data")
    return models_dir


@pytest.fixture
def fake_opencode_config(tmp_path: Path) -> Path:
    """Create a fake opencode config directory and file."""
    opencode_dir = tmp_path / ".config" / "opencode"
    opencode_dir.mkdir(parents=True)
    config = opencode_dir / "config.json"
    config.write_text(json.dumps({"provider": {"local-llm": {}}}))
    return config


@pytest.fixture
def fake_pi_config(tmp_path: Path) -> Path:
    """Create a fake pi config directory and file."""
    pi_dir = tmp_path / ".pi" / "agent"
    pi_dir.mkdir(parents=True)
    config = pi_dir / "models.json"
    config.write_text(json.dumps({"providers": {"local-llm": {}}}))
    return config


@pytest.fixture
def fake_cert_file(tmp_path: Path) -> Path:
    """Create a fake TLS certificate file."""
    cert_dir = tmp_path / "etc" / "ssl" / "local-llm"
    cert_dir.mkdir(parents=True)
    cert = cert_dir / "cert.pem"
    cert.write_text("-----BEGIN CERTIFICATE-----\nMIIBkTCB+wIJAKHHCgVZU\n-----END CERTIFICATE-----\n")
    return cert


@pytest.fixture
def fake_history_file(tmp_path: Path) -> Path:
    """Create a benchmark history CSV with a header."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    history_file = logs_dir / "benchmark-history.csv"
    history_file.write_text(
        "timestamp,model,backend,pp_tps,tg_tps,ctx,n_tokens,n_gpu_layers\n"
        "2026-01-01T00:00:00,my-model,llama-server,10.0,20.0,4096,100,20\n"
    )
    return history_file


@pytest.fixture
def fake_bench_bin(tmp_path: Path) -> Path:
    """Create a fake llama-bench binary that outputs CSV."""
    bench = tmp_path / "llama-bench"
    bench.write_text("#!/bin/bash\necho 'prompt_tokens,generation_tokens,avg_ts'\n"
                     'echo "100,0,10.5"\n'
                     'echo "0,200,25.3"\n')
    bench.chmod(0o755)
    return bench


@pytest.fixture
def fake_bench_csv(monkeypatch, fake_bench_bin):
    """Add fake_bench_bin to PATH so _find_bench_bin finds it."""
    import os

    path = os.environ.get("PATH", "")
    monkeypatch.setenv("PATH", f"{fake_bench_bin.parent}{path and ':'}{path}")


@pytest.fixture
def fake_nginx_conf(tmp_path: Path) -> Path:
    """Create a fake nginx config template."""
    nginx_dir = tmp_path / "nginx"
    nginx_dir.mkdir()
    conf = nginx_dir / "llm-proxy.conf.template"
    conf.write_text(
        "server {\n"
        "    listen %%PROXY_PORT%% ssl;\n"
        "    location /v1/ {\n"
        "        proxy_pass http://127.0.0.1:%%SERVER_PORT%%;\n"
        "    }\n"
        "}\n"
    )
    return conf


@pytest.fixture
def fake_service_template(tmp_path: Path) -> Path:
    """Create a fake systemd service template."""
    systemd_dir = tmp_path / "systemd"
    systemd_dir.mkdir()
    svc = systemd_dir / "llm-server.service.template"
    svc.write_text(
        "[Service]\n"
        "ExecStart=%%LLAMA_SERVER_BIN%% --model %%MODELS_DIR%%/%%ACTIVE_MODEL%%\n"
    )
    return svc


@pytest.fixture
def fake_template_files(tmp_path: Path):
    """Create both nginx and systemd template files."""
    nginx_conf = tmp_path / "nginx" / "llm-proxy.conf.template"
    nginx_conf.parent.mkdir(parents=True)
    nginx_conf.write_text(
        "%%LAN_IP%% %%LAN_SUBNET%% %%PROXY_PORT%% %%SERVER_PORT%% %%API_KEY%%\n"
    )
    svc = tmp_path / "systemd" / "llm-server.service.template"
    svc.parent.mkdir(parents=True)
    svc.write_text(
        "%%LLAMA_SERVER_BIN%% %%MODELS_DIR%% %%ACTIVE_MODEL%%\n"
        "%%N_GPU_LAYERS%% %%N_CTX%% %%N_THREADS%% %%USER%%\n"
    )
    return {"nginx": nginx_conf, "systemd": svc}
