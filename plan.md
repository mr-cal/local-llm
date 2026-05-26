# Plan: llama.cpp Submodule + Config-Driven Build System

## Overview

Move from the manual "clone + cmake + cp" workflow to a fully automated, config-driven build system backed by a git submodule. The `uv run llm` CLI becomes the single entry point for building, updating, and installing llama.cpp.

---

## Step 1 — Add llama.cpp as a git submodule

```bash
git submodule add https://github.com/ggerganov/llama.cpp llama.cpp
git submodule update --init --recursive
```

**What this does:**
- Clones llama.cpp into `./llama.cpp/` in this repo
- Records the submodule URL and commit in `.gitmodules`
- `.gitmodules` is committed; actual source lives in the submodule

**Why a submodule (not a submodule with a pinned branch):**
- Pinning a specific commit gives reproducible builds
- `llm build` will update the submodule to a specific commit (configurable)

---

## Step 2 — Extend `config.toml` with `[build]` section

Add a new configuration section to manage build params and installation. Build **profiles** are a first-class concept — each profile is a dynamically-defined set of build flags that produces a distinct binary. Profiles become a dimension for benchmarking and tuning, alongside `n_gpu_layers` and `n_ctx`.

```toml
# ── BUILD ─────────────────────────────────────────────────────────────────────
# llama.cpp build configuration.

[build]
enabled = true
repo = "https://github.com/ggerganov/llama.cpp"
commit = "HEAD"          # git commit/branch/tag to build; "HEAD" = latest
install_dir = "~/.local/bin"

# Thread parallelism for the build itself.
# "auto" = nproc; set to a number for a specific count.
jobs = "auto"

# Whether to build in Release mode.
release = true

# Build profiles — a flat list of dynamically-defined profiles.
# Each profile is a self-contained set of cmake flags (including the backend flag).
# These are the dimensions for `llm benchmark tune --profiles`.
# The first profile is the active profile by default.

[[build.profiles]]
name = "vulkan-default"
# Backend is a shortcut — the CLI expands it to the correct cmake flag below.
backend = "vulkan"
# Extra cmake flags appended after the backend flag.
extra_flags = []

[[build.profiles]]
name = "vulkan-flash"
backend = "vulkan"
extra_flags = ["-DGGML_FLASH_ATTN=ON"]

[[build.profiles]]
name = "vulkan-optimized"
backend = "vulkan"
# You can define anything here — custom defines, optimization flags, etc.
extra_flags = ["-DGGML_FLASH_ATTN=ON", "-DCMAKE_C_FLAGS=-O3", "-DCMAKE_CXX_FLAGS=-O3"]

[[build.profiles]]
name = "cuda-debug"
backend = "cuda"
# Debug builds: no optimization, include debug symbols.
extra_flags = ["-DCMAKE_BUILD_TYPE=Debug", "-DGGML_CUDA_DMMV_X=64"]

[[build.profiles]]
name = "bare-metal"
# Profiles don't need a backend shortcut — you can write cmake flags directly.
extra_flags = ["-DGGML_VULKAN=ON", "-DCMAKE_C_FLAGS=-O3"]
```

**Rationale for the config shape:**
- `build.profiles` is a flat TOML array-of-tables (`[[...]]`) — users can freely add/remove profiles without nested table keys
- Each profile has a `name` (display identifier) and either a `backend` shortcut or raw `extra_flags` (or both)
- `backend` is a convenience shorthand — the CLI expands `backend = "vulkan"` to `-DGGML_VULKAN=ON` automatically
- `extra_flags` lets users write **any** cmake flag, custom define, or compiler flag — no predefined schema
- If both `backend` and `extra_flags` are present, the backend flag is prepended (so `backend = "vulkan"` + `extra_flags = ["-DFOO=BAR"]` → `-DGGML_VULKAN=ON -DFOO=BAR`)
- The first profile in the list is the active profile by default (configurable via `[server].profile`)
- `commit = "HEAD"` defaults to the latest; pin to a SHA for reproducibility
- `jobs` defaults to "auto" for parallel builds
- `install_dir` mirrors the existing `llama_server_bin` pattern but at the directory level

---

## Step 3 — Add `llm build` subcommands

Create `src/llm/build.py` with the following commands:

| Command | Description |
|---|---|
| `llm build init` | Clone/pull the submodule (if not present) to the configured commit |
| `llm build` | Build llama.cpp with the `active_profile` and install to `install_dir` |
| `llm build --profile <name>` | Build a specific profile (e.g. `vulkan-flash`) |
| `llm build all` | Build **all** configured profiles sequentially |
| `llm build update` | Pull the latest llama.cpp and rebuild the active profile |
| `llm build update --commit <sha>` | Update to a specific commit and rebuild |
| `llm build clean --name <name>` | Clean a specific profile's build directory |
| `llm build clean --all` | Clean **all** profile build directories |
| `llm build info` | Show current commit, active profile, all configured profiles (with full flag list), and binary locations |

**Implementation details:**
- `init` checks if `./llama.cpp/` exists; if not, runs `git submodule update --init`
- Each profile builds into its own CMake out-of-tree directory: `llama.cpp/build-{profile_name}/`
- `build` resolves the active profile, sets up cmake flags, builds, and copies `llama-server` and `llama-bench` to `<build.install_dir>/{profile_name}/llama-server` (namespaced by profile)
- `llm build all` iterates all profiles, builds each (parallelizable with `-j`), and installs all binaries
- Backend flag mapping:

  | Profile backend key | CMake flag |
  |---|---|
  | `vulkan` | `-DGGML_VULKAN=ON` |
  | `metal` | `-DGGML_METAL=ON` |
  | `cuda` | `-DGGML_CUDA=ON` |
  | `blis` | `-DGGML_BLIS=ON` |
  | `hipblas` | `-DGGML_HIPBLAS=ON` |
  | `coreml` | `-DGGML_COREML=ON` |
  | `kluster` | `-DGGML_KLUSTER=ON` |

- Extra cmake flags from the profile are appended verbatim
- Validates that each profile's backend is known; errors otherwise

---

## Step 3.5 — Extend benchmark with build-profile dimension

Build profiles are a first-class dimension for benchmarking and tuning, alongside `n_gpu_layers` and `n_ctx`. Each profile produces a distinct binary with different performance characteristics.

### New CLI commands

| Command | Description |
|---|---|
| `llm benchmark run --profile <name>` | Benchmark a specific profile (builds it first if needed) |
| `llm benchmark run --profiles` | Sweep all profiles — build each, then benchmark each against the same model/params |
| `llm benchmark run --profile vulkan-flash --profile cuda-debug` | Benchmark specific named profiles only |
| `llm benchmark tune --profiles` | Full sweep — for each profile, tune `n_gpu_layers` (and optionally `n_ctx`) |
| `llm benchmark compare` | Compare all profile results from history |

### CSV format update

Append `profile` and `flags_hash` columns to `logs/benchmark-history.csv`:

```csv
timestamp,alias,profile,backend,model,n_gpu_layers,n_ctx,flags_hash,backend_tokens_sec,api_tokens_sec,latency_ms,raw_tokens_sec
2026-05-22T14:00:00,qwen2.5-coder-14b-q4,vulkan-default,vulkan,Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf,20,131072,a1b2c3,45.2,42.1,120,0.0
2026-05-22T14:05:00,qwen2.5-coder-14b-q4,vulkan-flash,vulkan,Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf,20,131072,d4e5f6,52.8,49.3,105,0.0
```

The `flags_hash` is a short SHA of the full cmake flag list — useful for tracking results when profiles are edited or recreated.

### `llm benchmark run --profiles` behavior

1. **Build phase** — runs `llm build all` (builds all profiles if not already built)
2. **Benchmark phase** — for each profile:
   - Starts llama-server with that profile's binary
   - Runs the benchmark (same model, same server params, same prompt)
   - Records results with the `profile` column
   - Restarts with the next profile
3. **Output** — prints a comparison table:

```
Profile          Backend  Tokens/s  API/s   Latency  Δ vs baseline
───────────────  ───────  ────────  ──────  ────────  ─────────────
vulkan-default   vulkan   42.1      45.2    120ms     baseline
vulkan-flash     vulkan   49.3      52.8    105ms     +17.1% / +16.7%
vulkan-quant     vulkan   38.4      41.0    135ms     -8.8% / -9.3%
```

### `llm benchmark tune --profiles` behavior

1. Builds all profiles (if not built)
2. For each profile, sweeps `n_gpu_layers` (and optionally `n_ctx`):
   - e.g. `n_gpu_layers` = {0, 10, 20, 30, 40, all}
   - e.g. `n_ctx` = {8192, 16384, 32768, 65536}
3. Produces a combined report:

```
=== Profile: vulkan-default ===
n_gpu  Tokens/s  API/s    Latency
───    ────────  ──────   ───────
0      12.1      11.8     450ms
10     38.4      36.2     180ms
20     45.2      42.1     120ms  ← optimal
30     45.1      42.0     121ms
40     44.8      41.9     122ms
all    44.5      41.6     124ms

=== Profile: vulkan-flash ===
n_gpu  Tokens/s  API/s    Latency
───    ────────  ──────   ───────
0      10.8      10.5     520ms
10     42.1      39.8     160ms
20     52.8      49.3     105ms  ← optimal
30     52.5      49.0     106ms
40     52.0      48.5     108ms
all    51.8      48.2     109ms

=== Comparison ===
Profile          Best n_gpu  Best Tokens/s  Δ vs vulkan-default
───────────────  ──────────  ─────────────  ───────────────────
vulkan-default   20          45.2           baseline
vulkan-flash     20          52.8           +16.8%
```

### Profile-aware server management

- `llm server start --profile <name>` — starts llama-server from the specified profile's binary
- `llm server stop` — stops whatever is running (tracks PID + profile)
- `llm server status` — shows which profile is active
- Default `llm server start` uses the configured `active_profile`

### Server section config update

```toml
[server]
enabled = true
llama_server_bin = ""              # empty = auto-detect from active profile
profile = "vulkan-default"         # which profile's binary to use (by name)
port = 8080
n_gpu_layers = 20
n_ctx = 65536
n_threads = 12
extra_args = ['--jinja', '--cache-type-k', 'q8_0']
```

The `llama_server_bin` field now resolves to `<build.install_dir>/<profile>/llama-server` when empty.

Add to `src/llm/cli.py`:

```python
@cli.group()
def build():
    """Build and install llama.cpp."""
    pass

@build.command("init")
def build_init(): ...

@build.command()
@typer.option("--profile", "-p", help="Build a specific profile (default: active)")
@typer.option("--all", "-a", is_flag=True, help="Build all profiles")
def build_run(profile: str | None, all: bool): ...

@build.command("update")
@typer.argument("commit", required=False)
def build_update(commit: str | None): ...

@build.command("clean")
@typer.option("--all", is_flag=True, help="Clean all profile builds")
def build_clean(all: bool): ...

@build.command("info")
def build_info(): ...

# ── Benchmark extensions ─────────────────────────────────────────────

@benchmark.command("run")
@typer.option("--profile", "-p", multiple=True,
              help="Benchmark specific profile(s)")
@typer.option("--profiles", is_flag=True,
              help="Sweep all build profiles")
def benchmark_run(profile: tuple[str, ...], profiles: bool): ...

@benchmark.command("tune")
@typer.option("--profiles", is_flag=True,
              help="Tune across all build profiles")
def benchmark_tune(profiles: bool): ...

@benchmark.command("compare")
def benchmark_compare(): ...
```

Register the `build` group alongside existing groups (`server`, `model`, `config`, `benchmark`, `client`, `lxd`).
Add `--profile` / `--profiles` flags to existing benchmark commands.

---

## Step 5 — Update `config.py` for the new `[build]` section

- Add a `BuildProfile` Pydantic model:

```python
class BuildProfile(BaseModel):
    name: str                              # human-readable profile name
    backend: str | None = None             # convenience shorthand (expanded to cmake flag)
    extra_flags: list[str] = []            # arbitrary cmake/compiler flags
    _backend_flag: str = Field(
        default="",
        init_var=False,
        repr=False,
    )

    def get_full_flags(self) -> list[str]:
        """Return the complete list of cmake flags for this profile."""
        flags = []
        if self.backend:
            flags.append(BACKEND_FLAGS[self.backend])  # e.g. vulkan → -DGGML_VULKAN=ON
        flags.extend(self.extra_flags)
        return flags
```

- Add a `BuildConfig` Pydantic model:

```python
class BuildConfig(BaseModel):
    enabled: bool = True
    repo: str = "https://github.com/ggerganov/llama.cpp"
    commit: str = "HEAD"
    install_dir: str = "~/.local/bin"
    profiles: list[BuildProfile] = []
    jobs: str = "auto"
    release: bool = True

    @property
    def active_profile(self) -> BuildProfile:
        """Return the first profile (default active)."""
        if not self.profiles:
            return BuildProfile(name="default")
        return self.profiles[0]

    def get_profile(self, name: str | None = None) -> BuildProfile:
        """Get profile by name (or active_profile)."""
        if name:
            for p in self.profiles:
                if p.name == name:
                    return p
            raise ValueError(f"Unknown profile: {name}")
        return self.active_profile

    def profile_names(self) -> list[str]:
        """Return all profile names."""
        return [p.name for p in self.profiles]
```

- Add `build` field to the main config model
- Add `profile` field to the `[server]` section (which profile's binary to use, referenced by `name`)
- Add a `BACKEND_FLAGS` constant mapping:

```python
BACKEND_FLAGS = {
    "vulkan": "-DGGML_VULKAN=ON",
    "metal": "-DGGML_METAL=ON",
    "cuda": "-DGGML_CUDA=ON",
    "blis": "-DGGML_BLIS=ON",
    "hipblas": "-DGGML_HIPBLAS=ON",
    "coreml": "-DGGML_COREML=ON",
    "kluster": "-DGGML_KLUSTER=ON",
}
```

- Add validation:
  - Each profile's `backend` (if set) must be a known backend string
  - Profile `name` values must be unique
  - Profile names in `[server].profile` must exist in `build.profiles`
- Add to the config template generated by `config init`

---

## Step 6 — Update the server runner for profile-aware binary resolution

In `src/llm/server.py`:
- The `[server]` section's `llama_server_bin` resolution order:
  1. Explicit full path (current behavior)
  2. Profile-resolved: `<build.install_dir>/<profile>/llama-server` (when `llama_server_bin` is empty)
  3. `shutil.which("llama-server")` — on PATH
- The `profile` field from `[server]` selects which build profile's binary to use
- The server process tracks which profile it's running (written to `.server.pid` alongside PID)
- `server status` shows the active profile
- `server start --profile <name>` overrides the configured profile for this invocation
- Update `config apply` to render the systemd service with the resolved path
- Add a pre-flight check in `server start`: if `build.enabled` and the binary doesn't exist, suggest running `llm build`

In `src/llm/build.py`:
- `build_run(profile=None, all=False)` — builds the specified profile or `active_profile`
- `build_all()` — iterates all profiles, builds each into `llama.cpp/build-{profile}/`
- `build_info()` — shows commit, all profiles, active profile, and binary paths

---

## Step 7 — Update the Makefile

```makefile
.PHONY: install lint format test build

install:
	uv sync

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

test:
	uv run pytest tests/ -q

build:
	uv run llm build
```

---

## Step 8 — Update `.gitignore`

Add:
```gitignore
# llama.cpp build artifacts
llama.cpp/build/
llama.cpp/build-*/
llama.cpp/ggml/src/ggml*.so
llama.cpp/ggml/src/ggml*.dylib
llama.cpp/models/*.*.bin
llama.cpp/models/*.*.gguf

# Profile-specific installed binaries (optional — uncomment if you want to version-control builds)
# ~/.local/bin/*/llama-server
```

---

## Step 9 — Update README.md

### Replace the manual build section (Step 2 in Quick Start):

**Before:**
```bash
sudo apt install cmake libvulkan-dev glslc spirv-headers spirv-tools nginx
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j$(nproc)
mkdir -p ~/.local/bin
cp build/bin/llama-server build/bin/llama-bench ~/.local/bin/
```

**After:**
```bash
sudo apt install cmake build-essential libvulkan-dev glslc spirv-headers spirv-tools nginx
uv sync
uv run llm build init   # clones the submodule
uv run llm build         # builds + installs to ~/.local/bin
```

### Update the Machine Setup section (Prerequisites):
- Add `libvulkan-dev` and `spirv-tools` to Vulkan requirements
- Remove the manual clone + cmake instructions

### Add a "Updating llama.cpp" section:
```bash
uv run llm build update    # pull latest, rebuild
uv run llm build info      # show current commit
```

### Add a "Profile-based benchmarking" section:
```bash
# Benchmark a specific build profile
uv run llm benchmark run --profile vulkan-flash

# Sweep all profiles — build each, benchmark each, compare
uv run llm benchmark run --profiles

# Full tuning sweep: for each profile, tune n_gpu_layers
uv run llm benchmark tune --profiles

# Compare profile results from history
uv run llm benchmark compare
```

Show the comparison table output and explain how profiles form a tuning dimension alongside `n_gpu_layers`.

---

## Step 10 — Update the config template

The config template (`src/llm/config_template.toml`) needs to include the new `[build]` section with sensible defaults matching the current hardware (Vulkan for Radeon).

---

## Step 11 — Add tests

New test file `tests/test_build.py`:
- Test `BuildProfile.get_full_flags()` — backend shortcut + extra_flags composition
- Test `BuildConfig.get_profile(name)` — lookup, default, unknown name error
- Test `BuildConfig.profile_names()` list
- Test `BACKEND_FLAGS` constant completeness
- Test install path resolution (profile namespaced)
- Mock subprocess calls for build commands

---

## File change summary

| File | Action |
|---|---|
| `llama.cpp/` (dir) | Added via git submodule |
| `.gitmodules` | Added (auto by `git submodule add`) |
| `.gitignore` | Updated — add build artifact ignores |
| `config.toml` | Updated — add `[build]` section with flat-list profiles |
| `src/llm/config_template.toml` | Updated — add `[build]` flat-list profiles + `server.profile` |
| `src/llm/config.py` | Updated — add `BuildProfile`, `BuildConfig` models, `BACKEND_FLAGS`, `server.profile` field |
| `src/llm/cli.py` | Updated — add `build` group + `llm build` subcommands |
| `src/llm/build.py` | **New** — build logic, profile management, subprocess calls |
| `src/llm/server.py` | Updated — profile-aware binary resolution, PID+profile tracking |
| `src/llm/benchmark.py` | Updated — `--profile`/`--profiles` flags, profile sweep, comparison report |
| `Makefile` | Updated — add `build` target |
| `README.md` | Updated — replace manual build with `llm build`, add profile docs |
| `tests/test_build.py` | **New** — unit tests for build module |
| `tests/test_benchmark.py` | Updated — add profile sweep + comparison tests |

---

## Execution order

1. **Step 1** — `git submodule add` (one-time, manual)
2. **Step 2** — Write `[build]` config section with flat-list profiles to `config.toml`
3. **Step 5** — `BuildProfile` / `BuildConfig` Pydantic models, `BACKEND_FLAGS`, `server.profile` field
4. **Step 3** — `src/llm/build.py` (core build logic, flat-list profile iteration)
5. **Step 4** — Wire `build` into `src/llm/cli.py` + add benchmark profile flags
6. **Step 3.5** — Extend `benchmark.py` with `--profile`/`--profiles`, sweep, comparison
7. **Step 6** — Update `server.py` for profile-aware binary resolution
8. **Step 7–8** — Makefile + `.gitignore` updates
9. **Step 9** — README.md update (build + profile benchmark docs)
10. **Step 11** — Tests (`test_build.py` + updated `test_benchmark.py`)
11. **Step 10** — Config template (can be done in parallel with Step 5)

---

## Post-merge: one-time setup for existing installs

After merging, any machine using the repo runs:
```bash
git submodule update --init --recursive
uv sync
uv run llm build   # builds from the submodule
```

No manual clone, cmake, or cp needed — the CLI handles everything.
