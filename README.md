# local-llm

Run a local [Qwen](https://huggingface.co/Qwen) model on your machine and expose it securely over your LAN as an OpenAI-compatible API — ready for use with [opencode](https://opencode.ai) or any compatible client.

**Engine:** [llama.cpp](https://github.com/ggerganov/llama.cpp) (`llama-server`) with Vulkan backend for iGPU acceleration
**Proxy:** nginx (HTTPS + Bearer-token auth + LAN IP allowlist)
**CLI:** single `uv run llm` entry point

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
apt install cmake libvulkan-dev glslc spirv-headers spirv-tools nginx
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j$(nproc)
# Install to ~/.local/bin so it's on your PATH
mkdir -p ~/.local/bin
cp build/bin/llama-server build/bin/llama-bench ~/.local/bin/
```

Ensure `~/.local/bin` is on your PATH (add to `~/.bashrc` / `~/.zshrc` if needed):
```bash
export PATH="$HOME/.local/bin:$PATH"
```

Verify:
```bash
llama-server --version
```

### 3 — Create your config

```bash
uv run llm config init
```

This creates `config.toml` (gitignored) with every option pre-commented. Edit it:

```bash
$EDITOR config.toml
```

Key fields to set:

```toml
[server]
llama_server_bin = "~/.local/bin/llama-server"  # or just "llama-server" if ~/.local/bin is on PATH
n_gpu_layers = 20      # tune for your iGPU — higher offloads more layers

[models]
active = "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"

[proxy]
lan_ip     = "192.168.1.x"     # this machine's LAN IP
lan_subnet = "192.168.1.0/24"  # your local subnet
api_key    = "..."             # see below for generating one
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

### 5 — Start the server

```bash
uv run llm server start
uv run llm server status
```

### 6 — Set up nginx proxy

```bash
# Render the nginx config with your real settings
uv run llm config apply

# Generate a self-signed TLS certificate
sudo mkdir -p /etc/ssl/local-llm
sudo openssl req -x509 -newkey rsa:4096 \
    -keyout /etc/ssl/local-llm/key.pem \
    -out /etc/ssl/local-llm/cert.pem \
    -days 3650 -nodes -subj "/CN=local-llm"

# Install and enable the site
sudo cp nginx/llm-proxy.conf /etc/nginx/sites-available/llm
sudo ln -s /etc/nginx/sites-available/llm /etc/nginx/sites-enabled/llm
sudo nginx -t && sudo systemctl reload nginx
```

### 7 — Connect from your LXD container

**On the server**, generate the connection instructions:

```bash
uv run llm client setup
```

This prints the exact `export` commands and config for your specific server IP and API key. Copy and paste the output into the LXD container.

**In the LXD container** — you only need two env vars and opencode. No Python CLI required:

```bash
# Install opencode
curl -fsSL https://opencode.ai/install | sh

# Add to ~/.bashrc (values from `llm client setup` output above)
export OPENAI_BASE_URL="https://192.168.1.x:8443/v1"
export OPENAI_API_KEY="your-api-key"

# Test connectivity
curl -sk https://192.168.1.x:8443/health \
  -H "Authorization: Bearer your-api-key"

# Start coding
opencode
```

> **The `llm` CLI is a server management tool.** The client container does not need this repo, `uv`, or a `config.toml`.

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
| `qwen3-30b-moe-q4` | ~17 GB | MoE architecture, efficient at 30B scale |

Switch models without stopping the server:
```bash
uv run llm model switch Qwen2.5-72B-Instruct-Q4_K_M.gguf
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

## Auto-start with systemd

```bash
uv run llm config apply   # renders systemd/llm-server.service

# Install (replace YOUR_USER)
sed -i 's/%i/YOUR_USER/' systemd/llm-server.service
sudo cp systemd/llm-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llm-server
sudo systemctl status llm-server
```

---

## All Commands

```
uv run llm server start          Start llama-server
uv run llm server stop           Stop llama-server
uv run llm server restart        Restart llama-server
uv run llm server status         Show running status
uv run llm server logs [-f]      Tail server logs

uv run llm model list            List downloaded models
uv run llm model download <id>   Download a model from HuggingFace
uv run llm model switch <name>   Set active model + restart

uv run llm benchmark run         Run API + optionally raw benchmark
uv run llm benchmark history     Show past benchmark results

uv run llm client setup          Print opencode connection instructions

uv run llm config init           Generate config.toml with examples
uv run llm config show           Print current settings (masked)
uv run llm config apply          Render nginx + systemd templates
```

---

## Security Notes

- `config.toml` is **gitignored** — it contains your API key, LAN IP, and HF token.
- nginx enforces both a Bearer token and a source-IP subnet allowlist.
- TLS (self-signed) encrypts traffic on the LAN.
- The LXD container on the client side adds an additional isolation boundary.
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
