#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p -m 700 "$root/.imagine"
export ST_PROXY_CHAT_PORT="${ST_PROXY_CHAT_PORT:-5002}"
export ST_PROXY_IMAGE_PORT="${ST_PROXY_IMAGE_PORT:-8189}"
export ST_PROXY_MCP_PORT="${ST_PROXY_MCP_PORT:-8190}"
export ST_PROXY_MCP_OUTPUT_DIR="${ST_PROXY_MCP_OUTPUT_DIR:-/run/media/veracrypt1/mcp-images}"
export ST_PROXY_COMFY_ENABLED="${ST_PROXY_COMFY_ENABLED:-false}"
export ST_PROXY_COMFY_URL="${ST_PROXY_COMFY_URL:-http://127.0.0.1:8188}"
export ST_PROXY_COORDINATOR_SOCKET="${ST_PROXY_COORDINATOR_SOCKET:-$root/.imagine/gpu.sock}"
export ST_PROXY_IMAGINE_ACCESS_FILE="${ST_PROXY_IMAGINE_ACCESS_FILE:-$root/.imagine/access.json}"
export ST_PROXY_LLM_BACKEND="${ST_PROXY_LLM_BACKEND:-all}"

cargo build --manifest-path "$root/Cargo.toml" --release -p st-proxy --bins
exec "$root/target/release/st-proxy-launch" "$@"
