#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ST_PROXY_CHAT_PORT="${ST_PROXY_CHAT_PORT:-5002}"
export ST_PROXY_IMAGE_PORT="${ST_PROXY_IMAGE_PORT:-8189}"
export ST_PROXY_MCP_PORT="${ST_PROXY_MCP_PORT:-8190}"
export ST_PROXY_COMFY_ENABLED="${ST_PROXY_COMFY_ENABLED:-false}"
export ST_PROXY_COMFY_URL="${ST_PROXY_COMFY_URL:-http://127.0.0.1:8188}"
export ST_PROXY_LLM_BACKEND="${ST_PROXY_LLM_BACKEND:-all}"
# Optional bundled proxy is selected atomically with the matching server/UI.
if [[ -f "$root/comfyui/active-product.sh" ]]; then
    source "$root/comfyui/active-product.sh"
    if [[ -n "${IMAGINE_PRODUCT_PROXY_LAUNCH:-}" ]]; then
        if [[ "${IMAGINE_PRIVATE_STORAGE:-0}" == 1 ]]; then
            [[ "${XDG_RUNTIME_DIR:-}" == /* ]] || { echo "Runtime directory unavailable" >&2; exit 2; }
            export IMAGINE_PRIVATE_STORAGE
            mkdir -p -m 700 -- "$XDG_RUNTIME_DIR/imagine"
            export ST_PROXY_COORDINATOR_SOCKET="$XDG_RUNTIME_DIR/imagine/gpu.sock"
        fi
        export ST_STACK_SOURCE_DIR="$root/proxy"
        exec "$IMAGINE_PRODUCT_PROXY_LAUNCH" "$@"
    fi
fi
mkdir -p -m 700 "$root/.imagine"
export ST_PROXY_MCP_OUTPUT_DIR="${ST_PROXY_MCP_OUTPUT_DIR:-/run/media/veracrypt1/imagine/.mcp-cache}"
export ST_PROXY_COORDINATOR_SOCKET="${ST_PROXY_COORDINATOR_SOCKET:-$root/.imagine/gpu.sock}"

CARGO_TARGET_DIR="$root/comfyui/h3-main-20261020/target-product" cargo build --manifest-path "$root/Cargo.toml" --locked --offline -p st-proxy --bins
exec "$root/comfyui/h3-main-20261020/target-product/debug/st-proxy-launch" "$@"
