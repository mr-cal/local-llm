# local-llm

Run a local model on your machine and expose it securely over your LAN as an OpenAI-compatible API — ready for use with [opencode](https://opencode.ai) or any compatible client.

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

**nginx** sits between the network and `llama-server`, which only binds to `127.0.0.1`. This means:
- `llama-server` is never directly reachable from the LAN.
- All remote requests go through nginx, which enforces TLS, the Bearer token, and a subnet allowlist.
- Client tools on the *same* machine connect directly to `http://127.0.0.1:8080` — no TLS or auth needed.

**TLS cert:** Generated with `uv run llm config gencert`. The cert must include a SubjectAltName (SAN) matching the server's LAN IP — Node.js (and modern clients) reject certs that only have a `CN=`. The `gencert` command handles this automatically.

**API key:** A random hex token stored in `config.toml` under `[proxy] api_key`. nginx checks that every request includes `Authorization: Bearer <key>`. Generate one with:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## Machine roles

| Role | Description |
|---|---|
| **server+client** | Runs llama-server AND opencode/pi on the same machine. Client tools connect directly to `http://127.0.0.1:8080` — no TLS or auth overhead. Most common setup. |
| **server-only** | Runs llama-server + nginx proxy. Remote clients connect over HTTPS with a Bearer token. |
| **client-only** | Runs opencode/pi only; connects to a remote server via HTTPS with a Bearer token. No `[server]`, `[models]`, or `[proxy]` sections needed. |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12+ | `python3 --version` |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| nginx | `sudo apt install nginx` |
| cmake + build tools | `sudo apt install cmake build-essential` |
| Vulkan dev libraries | `sudo apt install libvulkan-dev vulkan-tools` |

---

## Quick Start

### 1 — Install Python dependencies

```bash
git clone <this-repo>
cd local-llm
uv sync
```

### 2 — Build llama-server with Vulkan

```bash
sudo apt install cmake libvulkan-dev glslc spirv-headers spirv-tools nginx
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j$(nproc)
mkdir -p ~/.local/bin
cp build/bin/llama-server build/bin/llama-bench ~/.local/bin/
```

Ensure `~/.local/bin` is on your PATH.

**bash** — add to `~/.bashrc`:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

**fish** — run once:
```fish
fish_add_path ~/.local/bin
```

Verify:
```bash
llama-server --version
```

### 3 — Create your config

```bash
uv run llm config init
$EDITOR config.toml
```

This creates `config.toml` (gitignored) with every option pre-commented. Key sections:

**Server + proxy** (skip on a client-only machine):
```toml
[server]
llama_server_bin = "llama-server"  # full path if not on PATH
n_gpu_layers = 20      # tune for your iGPU — higher offloads more layers
n_ctx = 65536          # context window; 65536+ recommended for agentic tasks
n_threads = 12         # physical core count (not hyperthreads)
extra_args = []        # e.g. ["--flash-attn", "--cache-type-k", "q8_0", "--jinja"]

[models]
dir = "~/models"
active = "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"
hf_token = ""          # only needed for gated/private HuggingFace models

[proxy]
port = 8443
lan_ip = "192.168.1.x"       # this machine's LAN IP
lan_subnet = "192.168.1.0/24"
api_key = "..."               # generate one (see below)
cert_path = "/etc/ssl/local-llm/cert.pem"
```

**Client** (on a server+client machine, leave `server_url` empty — it defaults to the local server):
```toml
[client]
server_url = ""   # set to "https://192.168.1.x:8443/v1" on a client-only machine
api_key = ""      # remote server's api_key (client-only)
cert_path = ""    # path to the remote server's cert.pem (client-only)
```

**Optional — per-token cost tracking** (used by pi):
```toml
[model_cost]
input = 0.0
output = 0.0
```

Generate a strong API key:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 4 — Download a model

```bash
# See all supported aliases
uv run llm model download --list

# Download the recommended default (~8.5 GB)
uv run llm model download qwen2.5-coder-14b-q4

# Or specify a raw HuggingFace repo + file
uv run llm model download bartowski/Qwen2.5-72B-Instruct-GGUF \
    --file Qwen2.5-72B-Instruct-Q4_K_M.gguf
```

### 5 — Generate the TLS certificate

```bash
uv run llm config gencert
```

This generates a self-signed cert at `/etc/ssl/local-llm/cert.pem` (and `key.pem`) with the correct SubjectAltName for your `lan_ip`. Use `--force` to regenerate if your LAN IP changes.

### 6 — Apply config

```bash
uv run llm config apply
```

This renders all templates and installs everything in one step:

- **opencode** — writes `~/.config/opencode/config.json` (validates against the live schema)
- **pi** — writes `~/.pi/agent/models.json`
- **nginx** — renders the proxy config, copies it to `/etc/nginx/sites-available/llm`, enables the site, and starts or reloads nginx
- **systemd** — renders the service file, installs it to `/etc/systemd/system/`, runs `daemon-reload`, and enables the service

On a client-only machine, the nginx and systemd steps are skipped.

### 7 — Start the server

```bash
uv run llm server start
uv run llm server status
```

`server start` also ensures nginx is running. The server auto-starts on boot via the systemd service installed by `config apply`.

---

## Client setup (remote machine or LXD container)

**On the server**, generate connection instructions:

```bash
uv run llm client setup
```

This prints step-by-step instructions with your real IP, API key, and cert content filled in.

**On the client machine** — you only need env vars and opencode. No Python CLI or `config.toml` required:

```bash
# Install opencode
curl -fsSL https://opencode.ai/install | sh
```

**Step 1 — Trust the TLS cert** (copy the PEM block from `client setup` output):

```bash
mkdir -p ~/.config/opencode
cat > ~/.config/opencode/local-llm.pem << 'EOF'
<paste cert here>
EOF
```

**bash** — add to `~/.bashrc`:
```bash
export NODE_EXTRA_CA_CERTS="$HOME/.config/opencode/local-llm.pem"
```

**fish** — add to `~/.config/fish/config.fish`:
```fish
set -gx NODE_EXTRA_CA_CERTS $HOME/.config/opencode/local-llm.pem
```

**Step 2 — Set connection env vars**:

**bash** — add to `~/.bashrc`:
```bash
export OPENAI_BASE_URL="https://192.168.1.x:8443/v1"
export OPENAI_API_KEY="your-api-key"
```

**fish** — add to `~/.config/fish/config.fish`:
```fish
set -gx OPENAI_BASE_URL "https://192.168.1.x:8443/v1"
set -gx OPENAI_API_KEY "your-api-key"
```

**Step 3 — Test connectivity**:
```bash
curl -s --cacert ~/.config/opencode/local-llm.pem \
  https://192.168.1.x:8443/health \
  -H "Authorization: Bearer your-api-key"
```

---

## Model Recommendations

This machine has 62 GB RAM — larger models than most systems can run.

| Alias | Size | Notes |
|---|---|---|
| `qwen2.5-coder-7b-q8` | ~8 GB | Fastest, good for quick tasks |
| `qwen2.5-coder-14b-q4` | ~8.5 GB | **Default** — best speed/quality balance |
| `qwen2.5-coder-32b-q4` | ~18 GB | Strong coding model |
| `qwen2.5-coder-32b-q8` | ~34 GB | High-precision 32B |
| `qwen2.5-72b-q4` | ~42 GB | Near-frontier quality — fits in 62 GB |
| `qwen3-8b-q8` | ~9 GB | Qwen3 8B — fast, near-lossless |
| `qwen3-14b-q8` | ~16 GB | Qwen3 14B — near-lossless, strong coding |
| `qwen3-32b-q4` | ~20 GB | Qwen3 32B dense — top-tier coding quality |
| `qwen3-30b-moe-q4` | ~19 GB | Qwen3 30B MoE — fast, outperforms QwQ-32B |
| `qwen3.6-35b-moe-q4` | ~21 GB | Qwen3.6 35B MoE — SWE-bench 73%, 262K ctx |
| `qwen3.6-27b-q4` | ~18 GB | Qwen3.6 27B dense — Apr 2026, 262K ctx, multimodal |
| `gemma-4-31b-q4` | ~20 GB | Google Gemma 4 — newest, multimodal |
| `gemma-3-27b-q4` | ~17 GB | Gemma 3 27B — strong all-rounder, multimodal |
| `gemma-3-27b-q8` | ~29 GB | Gemma 3 27B — high precision, multimodal |
| `gemma-3-12b-q4` | ~7 GB | Gemma 3 12B — fast, multimodal |
| `gemma-3-12b-q8` | ~13 GB | Gemma 3 12B — high precision, multimodal |

Switch models without stopping the server:
```bash
uv run llm model switch gemma-4-31b-q4
```

---

## Benchmarking

```bash
# End-to-end API benchmark (tokens/sec via the HTTP API)
uv run llm benchmark run

# Custom prompt and token count
uv run llm benchmark run --prompt "Explain how async/await works in Python" --n-tokens 500

# Include raw llama-bench (no HTTP overhead)
uv run llm benchmark run --raw

# View history
uv run llm benchmark history
uv run llm benchmark history -n 50
```

Results are appended to `logs/benchmark-history.csv` (gitignored).

**Tuning `n_gpu_layers`:** Start at 20, run `benchmark run`, increase by 5, repeat. The Radeon 890M has shared VRAM (from system RAM), so there's no hard ceiling — but past a point there's diminishing returns.

---

## LXD containers

```bash
# Create and configure an LXD container for development
uv run llm lxd create <name>

# Run `make setup` in each craft_dir configured in config.toml
uv run llm lxd setup-crafts <name>
```

Configure mounts and craft project paths in `config.toml` under `[lxd]`:
```toml
[lxd]
craft_dirs = [
    "~/dev/craft/snapcraft",
]

[[lxd.mounts]]
host = "~/dev"

[[lxd.mounts]]
name = "opencode-config"
host = "~/.config/opencode"
```

---

## Auto-start with systemd

`uv run llm config apply` installs and enables the systemd service automatically. To check or control it manually:

```bash
sudo systemctl status llm-server
sudo systemctl restart llm-server
```

---

## All Commands

```
uv run llm server start          Start llama-server (also starts nginx)
uv run llm server stop           Stop llama-server and nginx
uv run llm server restart        Restart llama-server
uv run llm server status         Show running status
uv run llm server logs [-f]      Tail server logs

uv run llm model list            List downloaded models
uv run llm model download <id>   Download a model from HuggingFace
uv run llm model switch <name>   Set active model + restart

uv run llm benchmark run         Run API + optionally raw benchmark
uv run llm benchmark history     Show past benchmark results

uv run llm client setup          Print connection instructions for a remote client

uv run llm config init           Generate config.toml with examples
uv run llm config show           Print current settings (masked) + opencode/pi config
uv run llm config apply          Render templates + install nginx/systemd/opencode/pi configs
uv run llm config gencert        Generate self-signed TLS cert with correct SAN

uv run llm lxd create <name>     Create and configure an LXD container
uv run llm lxd setup-crafts      Run make setup in configured craft directories
```

---

## Security Notes

- `config.toml` is **gitignored** — it contains your API key, LAN IP, and HF token.
- nginx enforces both a Bearer token and a source-IP subnet allowlist.
- TLS (self-signed) encrypts traffic on the LAN.
- The server only listens internally (`127.0.0.1`); nginx handles the LAN exposure.

---

## Development

```bash
# Lint + format
uv run ruff check src/
uv run ruff format src/

# Type check
uv run ty check src/
```
