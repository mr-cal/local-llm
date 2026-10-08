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
uv run llm model download qwen3.6-35b-moe-q4
```

### 5 - Start the server

```bash
uv run llm server start
uv run llm server status
```

`llama-server` runs under systemd as the `llm-server` unit, which owns its
restart policy and logging. `start`, `stop` and `restart` are thin wrappers
around `systemctl`, and `logs` reads the journal.

The unit's command line is generated from `config.toml`, so changing
`[server]` settings (port, `n_ctx`, `n_gpu_layers`, `extra_args`, build
profile) means re-running `uv run llm server apply` followed by
`uv run llm server restart`. `status` warns when `config.toml` is newer than
the installed unit.

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

# Refresh packages and configs in all managed dev client VMs
# (the Jotter container is excluded; use `llm jotter setup` for that)
uv run llm client refresh

# List managed VMs/containers, with their kind (client or jotter)
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

Every command is `uv run llm <group> <command>`; the `uv run` prefix is omitted
below. Add `--help` to any group or command for its full options.

```
server setup            Guided setup for the server (and local client)
server start            Start llama-server via its systemd unit
server stop             Stop llama-server and nginx
server restart          Restart llama-server and make sure nginx is running
server status           Show whether llama-server and nginx are running
server logs [-f]        Show server logs from the systemd journal
server memory           Show recent memory samples recorded by the monitor
server apply            Render nginx/systemd templates and install them

client setup [-c NAME]  Set up a client: the current host, or an LXD VM
client check            Test connectivity to the configured LLM server
client show             Print current client connection info (URL, model, cert)
client list             List managed LXD VMs with kind, status and version
client refresh [NAME]   Update packages and re-apply config in managed VMs
client crafts NAME      Run 'make setup' in configured craft dirs inside a VM

model list              List known models, with download and active status
model download <id>     Download a GGUF model from HuggingFace
                        (skips models already present; --force re-downloads)
model switch <name>     Set the active model (accepts alias or filename)
model show <name>       Show detailed info for a model

tui                     Launch the Jotter interactive Textual TUI
digest run [--slot]     Generate morning or afternoon digest headlessly
jotter setup            Create and configure the Jotter LXD container
jotter status           Show Jotter container status

build init              Initialize the llama.cpp git submodule
build run               Build llama.cpp with a profile and install binaries
build update            Pull the latest llama.cpp and rebuild
build clean             Remove profile build directories
build info              Show submodule commit, profiles and binary paths

benchmark run           Run an end-to-end API benchmark and record results
benchmark tune          3-phase sweep: GPU layers -> flash-attn -> KV quant
benchmark history       Display benchmark history

config init             Create a minimal client-only config.toml interactively
config show             Print current config (credentials masked) + client config
```

---

## Jotter Assistant & Daily Digest TUI

`jotter` provides a transparent, deterministic digest engine and interactive
Textual TUI (`llm tui`) running either directly or in an isolated LXD container.

- **Daily Digests**: Weekday scheduling for morning (5:30 AM, captures weekend on Monday)
  and afternoon (2:00 PM) slots.
- **Multi-Source Ingestion**: Pulls from Discourse, GitHub, Mattermost, Matrix, Google
  Workspace (Calendar, Gmail, Drive), Jira, and read-only local code in `~/dev/craft/`.
- **Thread Updates & Deltas**: Automatically calculates thread reply counts and deltas
  between morning and afternoon slots.
- **Fast Local Cache & Embeddings**: SQLite database with full-text search (FTS5) and
  vector semantic similarity using the restored local embedding server on port 8081.
- **Calibrated Inspection & Streaming**: Measures prompt evaluation speed and self-calibrates
  actual tokens/sec to give a live estimated progress bar before streaming `<think>`
  reasoning and answers character-by-character.

```bash
uv run llm tui                # Launch the interactive Textual interface
uv run llm digest run         # Generate morning digest headlessly
uv run llm digest run --slot afternoon  # Generate afternoon digest headlessly
uv run llm jotter setup       # Provision the isolated LXD container
uv run llm jotter status      # Check container status
```

---

## Security Notes

- `config.toml` is **gitignored**; it holds your API key, HF token, and external
  service credentials.
- `llm config show` masks every secret-bearing field.
- Generated files that embed a credential (the rendered nginx conf, the client
  configs, the shell env files) are written `0600`.
- nginx enforces both a Bearer-token check and a source-IP subnet allowlist.
- TLS (self-signed, with a correct SubjectAltName) encrypts traffic on the LAN.
- `llama-server` only listens on `127.0.0.1`; nginx handles the LAN exposure.

---

## Development

```bash
make lint      # ruff check + ty
make format    # ruff format
make test      # pytest
```
