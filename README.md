# local-llm

Run a local model on your machine and expose it over your LAN as an OpenAI-compatible API. Works with [opencode](https://opencode.ai) or any compatible client.

**Engine:** [llama.cpp](https://github.com/ggerganov/llama.cpp) (`llama-server`) with Vulkan backend for iGPU acceleration
**Proxy:** nginx (HTTPS + Bearer-token auth + LAN IP allowlist)
**CLI:** single `uv run llm` entry point

---

## How it works

```
LAN client (opencode, curl, …)
        │
        │  HTTPS :8443  +  Bearer token
        ▼
     nginx proxy
     ├─ TLS termination (self-signed cert with SAN)
     ├─ Bearer token check  (rejects wrong / missing key)
     └─ subnet allowlist    (rejects requests outside LAN)
        │
        │  HTTP :8080  (localhost only)
        ▼
  llama-server (llama.cpp)
  └─ loads your GGUF model
```

---

## Quick Start

### 1 - Install dependencies

```bash
git clone <this-repo>
cd local-llm
uv sync
```

### 2 - Build llama-server (or install a pre-built binary)

```bash
uv run llm build init    # initialize llama.cpp submodule
uv run llm build run     # build with active profile (Vulkan by default)
```

Or install manually (see the [llama.cpp build guide](https://github.com/ggerganov/llama.cpp#build)).

### 3 - Server setup (guided wizard)

```bash
uv run llm server setup
```

This command:
1. Creates/updates `config.toml` (auto-detects your LAN IP, prompts for tuning params)
2. Generates an API key
3. Generates the TLS certificate (with correct SubjectAltName)
4. Renders and installs nginx + systemd configs
5. Configures the local client (opencode, pi, shell env vars)

Safe to re-run (detects existing config and offers to update).

### 4 - Download a model

```bash
uv run llm model list
uv run llm model download qwen2.5-coder-14b-q4
```

### 5 - Start the server

```bash
uv run llm server start
uv run llm server status
```

### 6 - Verify connectivity

```bash
uv run llm client check
```

---

## LXD Container Setup

Create a fully configured LXD VM as a development client:

```bash
uv run llm client setup --container craft-llm-1
```

This creates the VM, installs packages, configures mounts, and sets up the full client (opencode, pi, TLS cert, shell env vars).

```bash
# Enter the VM
lxc exec craft-llm-1 -- su -l $USER

# Run make setup in configured craft directories
uv run llm client crafts craft-llm-1

# Refresh packages and configs in all managed VMs
uv run llm client refresh

# List managed VMs
uv run llm client list
```

Configure mounts and craft project paths in `config.toml` under `[lxd]`:
```toml
[lxd]
craft_dirs = ["~/dev/craft/snapcraft"]

[[lxd.mounts]]
host = "~/dev"

[[lxd.mounts]]
name = "opencode-config"
host = "~/.config/opencode"
```

---

## All Commands

```
Server
  uv run llm server setup          Guided setup wizard (config, cert, nginx, systemd, client)
  uv run llm server start          Start llama-server (also starts nginx)
  uv run llm server stop           Stop llama-server and nginx
  uv run llm server restart        Restart llama-server
  uv run llm server status         Show running status
  uv run llm server logs [-f]      Tail server logs

Client
  uv run llm client setup          Set up this machine as a client (opencode, pi, shell env)
  uv run llm client setup -c NAME  Create an LXD VM and set it up as a client
  uv run llm client check          Test connectivity to the server
  uv run llm client show           Print current client connection info
  uv run llm client list           List managed LXD VMs
  uv run llm client refresh [NAME] Update packages + re-apply config in VMs
  uv run llm client crafts NAME    Run make setup in craft directories inside a VM

Models
  uv run llm model list            List downloaded models
  uv run llm model download <id>   Download a model from HuggingFace
  uv run llm model switch <name>   Set active model + restart

Config
  uv run llm config show           Print current settings + opencode/pi config

Build
  uv run llm build init            Initialize llama.cpp submodule
  uv run llm build run             Build llama.cpp with active profile
  uv run llm build update          Update llama.cpp to latest commit
  uv run llm build clean           Clean build artifacts
  uv run llm build info            Show build info

Benchmark
  uv run llm benchmark run         Run API benchmark
  uv run llm benchmark history     Show past benchmark results
```

---

## Security Notes

- `config.toml` is **gitignored** (contains your API key, LAN IP, and HF token).
- nginx enforces both a Bearer token and a source-IP subnet allowlist.
- TLS (self-signed) encrypts traffic on the LAN.
- The server only listens internally (`127.0.0.1`); nginx handles the LAN exposure.

---

## Development

```bash
uv run ruff check src/
uv run ruff format src/
uv run pytest tests/
```
