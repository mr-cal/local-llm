# agents.md

## Before completing any task

Always run:

```bash
make format   # ruff format + ruff check --fix
make lint     # ruff check
make test     # pytest
```

Ensure all commands pass before marking a task complete.

---

## Project layout

```
src/llm/
  cli.py       → Typer CLI entry point (routes to sub-modules)
  client.py    → client setup/check/show/refresh/commands
  lxd.py       → Container creation, config, verification, gh auth
  config.py    → Settings (pydantic), template rendering, shell env setup
  build.py     → llama.cpp build profiles
  server.py    → llama-server start/stop/status
  models.py    → Model catalog + model helpers
  benchmark.py → Benchmark parsing and reporting
tests/         → All tests; key files: test_lxd.py, test_client.py, test_config.py
nginx/         → Nginx proxy config templates
systemd/       → systemd service templates
config.toml    → User configuration
```

## Configuration

The app uses a single `config.toml` with these top-level sections:

| Section | Required | Purpose |
|---------|----------|---------|
| `server` | Conditional | llama-server settings (`enabled`, `port`, `n_gpu_layers`, etc.). Set `enabled = false` on client-only machines. |
| `proxy` | Conditional | nginx TLS proxy settings (`lan_ip`, `port`, `cert_path`). Set `enabled = false` on client-only machines. |
| `client` | Conditional | How tools connect (`server_url`, `cert_path`). Set `enabled = true` on client-only machines. |
| `auth` | Conditional | Bearer API key (`api_key`). Generated on the server. |
| `models` | — | Model catalog (`dir`, `active`, `list`, `hf_token`). |
| `build` | — | llama.cpp build config (`repo`, `commit`, `profiles`, `install_dir`). |
| `github` | **Optional** | GitHub CLI auth (`token`). Used by `llm client setup --container` to run `gh auth login` inside containers. Leave blank to skip. |
| `lxd` | — | LXD-specific settings (`mounts`, `craft_dirs`). |

**Secrets** (`auth.api_key`, `models.hf_token`, `github.token`) are masked in `llm config show` output.

### Bind-mount defaults (in `lxd.py` `_DEFAULT_MOUNTS`)

| Mount name | Host path | Container path |
|------------|-----------|----------------|
| `agents` | `~/.agents` | `~/.agents` |
| `github` | `~/.github` | `~/.github` |
| `dev` | `~/dev` | `~/dev` |
| `opencode-config` | `~/.config/opencode` | `~/.config/opencode` |

These are used by `create_and_setup()` and `refresh_containers()`. Custom mounts override via the `[lxd] mounts` section.

## Container workflow (`llm client setup --container`)

The full sequence in `_setup_container_client()` / `create_and_setup()`:

1. **Launch** — `lxc launch ubuntu:24.04` (container or VM)
2. **ID mapping** — `raw.idmap` to map host UID/GID → container UID/GID (containers only; VMs share namespace)
3. **Bind mounts** — disk devices for `.agents`, `.github`, `~/dev`, `.config/opencode`
4. **Package install** — `build-essential`, `gh`, `gh-copilot`, `astral-uv`, `helix`, `nodejs`, `pylsp`
5. **Pi harness** — `/etc/hosts` entry, `models.json`, TLS cert, `NODE_EXTRA_CA_CERTS` shell vars
6. **Opencode config** — written to container's `~/.config/opencode/config.json`
7. **gh auth** — `gh auth login --with-token` (if `[github] token` is set)
8. **Version tag** — `lxc config set <name> user.local-llm-version=<ver>`

### `effective_uid` / `effective_gid` convention

VMs and containers use different UID/GID for `lxc exec`:

```python
effective_uid = HOST_UID if lxd_vm else CONTAINER_UID
effective_gid = HOST_GID if lxd_vm else CONTAINER_GID
```

Always compute this once and pass as keyword args: `effective_uid=effective_uid, effective_gid=effective_gid`.

## Injecting content into containers

The pattern is consistent across cert injection, config files, and other content:

```python
subprocess.run(
    _cexec(container, effective_uid, effective_gid, "bash", "-c", "cat > <path>"),
    input=payload.encode(),
    check=True,
)
```

Where `_cexec()` builds the `lxc exec --user=... --group=... --env=HOME=... --` prefix.

For `gh auth login --with-token`, pipe the token directly as stdin to the `gh` command (no intermediate file needed):

```python
subprocess.run(
    _cexec(container, effective_uid, effective_gid, "gh", "auth", "login", "--with-token"),
    input=gh_token.encode(),
    check=True,
)
```

## Testing

- Tests use `monkeypatch.setattr(subprocess, "run", _run)` to intercept all subprocess calls.
- Use the `_make_completed(returncode=0, stdout="")` helper to build mock `CompletedProcess` objects.
- Collect calls in a `list[list]` to verify what commands were issued.
- Key test files: `tests/test_lxd.py`, `tests/test_client.py`, `tests/test_config.py`.
- All 267 tests must pass.

## Verification

- Run `make format`, `make lint`, and `make test` before marking any task complete.
- Evidence (test output) before assertions — always show passing test results when claiming success.
