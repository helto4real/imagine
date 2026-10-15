# Rust st-proxy

Standalone Rust port of `/home/thhel/git/st-proxy` at `10201aa1bc29f31977009d2c29a436da04eae497`, in the Imagine Cargo workspace. See [PARITY.md](PARITY.md) for evidence and remaining limitations. The original Python checkout and its launchers remain usable. No service is automatically migrated.

## Build and launch

From `/home/thhel/ChatGPT/imagine`:

```sh
cargo build --release --workspace
python3 native/prepare.py --check
python3 native/workflow/build.py          # separate Turbo-compatible native engine
python3 native/workflow/build.py --check
```

Python is used for restoration/build tooling and reference tests, never by the running proxy or Imagine API. The workflow build requires the same local CUDA toolchain, architecture 89 and pinned CUTLASS headers as this machine's accepted build. It restores pinned source archives and patches into `.imagine/workflow-native`; it does not modify `stable-diffusion.cpp`, the two base patches, or the frozen v2 executable. `SD_BUILD_JOBS=2` controls build parallelism.

Start Imagine with coordination enabled (terminal 1):

```sh
cd /home/thhel/ChatGPT/imagine
mkdir -p .imagine/coordinated
IMAGINE_COORDINATOR_SOCKET="$PWD/.imagine/coordinated/gpu.sock" \
  ./target/release/imagine-server --project "$PWD" \
  --data "$PWD/.imagine/coordinated/data" --bind 127.0.0.1:8080
```

Start the proxy (terminal 2); this example uses already running backends:

```sh
cd /home/thhel/ChatGPT/imagine
./start-proxy.sh --proxy-only --llm-backend sc117 \
  --llm-url http://127.0.0.1:48640 --comfy-url http://127.0.0.1:8188 \
  --chat-port 5002 --image-port 8189 \
  --coordinator-socket "$PWD/.imagine/coordinated/gpu.sock" \
  --imagine-url http://127.0.0.1:8080 \
  --imagine-access-file "$PWD/.imagine/coordinated/data/access.json" \
  --mcp-port 8190 \
  --mcp-input-dir /run/media/veracrypt1/mcp-inputs \
  --mcp-output-dir "$PWD/.imagine/coordinated/output"
```

`start-proxy.sh` defaults to chat port **5002**, image port **8189**, and enabled MCP on **8190**, matching `st-stack.zsh`. ComfyUI defaults to upstream port 8188 and KoboldCpp to 5001. MCP output defaults to `/run/media/veracrypt1/mcp-images`, matching the original proxy and Pi container mount. Reference input defaults to `/run/media/veracrypt1/mcp-inputs`. Environment variables and CLI arguments can override these defaults.

`./start-proxy.sh` without arguments incrementally builds the release binaries and runs `proxy/st-stack.zsh` in **all** mode. It starts or reuses KoboldCpp, TabbyAPI, Prism, SC117, Orca, SC117 128k and Orca 128k, waits for readiness, then starts the proxy with the combined `backend/model` catalog that Pi expects. Models are activated on demand through the shared GPU coordinator. Imagine must be started separately first. Ctrl-C or `./start-proxy.sh --stop` stops the stack; `--log` shows backend logs. Set `ST_PROXY_LLM_BACKEND` to select another stack backend.

Proxy CLI options and unambiguous abbreviations configure the supervised stack too: `./start-proxy.sh --llm-b sc117 --chat-port 5002` starts that backend and the proxy. Use `--proxy-only` to connect to independently managed backends without starting them. `--help` lists existing CLI options plus the three integration options. The usual `ST_PROXY_*` variables work; the new ones are `ST_PROXY_COORDINATOR_SOCKET`, `ST_PROXY_IMAGINE_URL`, and `ST_PROXY_IMAGINE_ACCESS_FILE`. `ST_PROXY_MCP_TOKEN` remains environment-only. Tokens are not logged.

Use `--llm-backend koboldcpp|tabbyapi|ollama|prism|sc117|orca|sc117_128k|orca_128k|all` and the corresponding origin options. `all` exposes `backend/model` IDs and requires distinct origins. Kobold router mode retains `--kobold-router-mode`, admin password, bootstrap/cache and default-model behavior. `--model-profiles "$PWD/proxy/config/model-profiles.json"` enables the copied profiles. `--check-backend` validates the selected backend without starting listeners.

`proxy/st-stack.zsh` preserves the original stack's menus, backend choices, colors, cleanup, TTS and SillyTavern options. Build first with `cargo build --release -p st-proxy --bins`, export the three integration variables above, and start Imagine separately before launching this script. Its listener ports match the root launcher; inspect its environment options for other stack settings. It does not own the separate Imagine process. The Kobold bootstrap and Strata log filter are Rust binaries. No other stack launch scripts are replaced.

Coordination is enabled only when Imagine receives `IMAGINE_COORDINATOR_SOCKET`. Every direct API/UI and MCP job then reserves the same FIFO as LLM and Comfy requests. Unavailable coordination fails admission with 503; loss during generation stops the native engine. There is no standalone fallback. An idle image engine may remain resident until the next owner requests the GPU. Starting an independent uncoordinated Imagine instance bypasses this arrangement.

## Desktop UI through the proxy

`./start-ui.sh` defaults to `http://127.0.0.1:8189/imagine`. The image listener
forwards `/imagine/*` to `--imagine-url`, keeping the UI client's bearer token,
request body, idempotency key and streamed responses. Imagine still checks asset
and job ownership and admin rights. Its scheduler reserves the shared GPU ticket;
the forwarding request does not take an additional lease. The root launcher disables ComfyUI by default: no Comfy cleanup or health requests
are made, and Comfy routes return 503 `comfy_disabled`. To use Comfy alongside
Imagine, pass `--comfy-enabled true` (or `ST_PROXY_COMFY_ENABLED=true`); Comfy
routes then use the same image port outside `/imagine`, and confirmed Comfy GPU
release is required during handoffs. An unreachable enabled Comfy is never ignored.

The root `start-server.sh` / `start-proxy.sh` launchers now default to the shared
`.imagine/gpu.sock` and `.imagine/access.json`. With custom `--data`, pass the matching
`--imagine-access-file` to the proxy and `--access` to the UI. The explicit coordinated
example above uses `.imagine/coordinated/data/access.json`; select that file when
starting the UI. No existing services are restarted by these changes.

MCP keeps an admitted job in `recovering` during an Imagine status/result outage.
`get_job` continues to use the same job. Ambiguous submission responses are retried
with the identical request body and idempotency key; no new generation ID is created.
Restart Pi to load its updated `qwen-scene` profile instructions. Its output mount
must match `ST_PROXY_MCP_OUTPUT_DIR` if that variable is overridden.

## Imagine profiles and native compatibility

Imagine owns the [YAML profiles](../profiles/README.md), defaults and override validation. MCP exposes `list_profiles`, `describe_profile`, `register_image`, `generate`, `generate_batch` and `get_job`. Pass `profile` instead of `workflow_id`, and choose `quick-draft` instead of `parameters.mode`. Process-local jobs, retry fingerprints, random or explicit 53-bit seeds, reference registration and consecutive batch ordering remain. MCP calls only authenticated public Imagine endpoints; general Comfy proxy traffic still uses ComfyUI. `nsfw=true` is an explicit LoRA switch. The copied graphs/manifests remain historical fixtures, with their hashes preserved; `--mcp-workflows-dir` no longer controls MCP.

The default v2 model/profile stays on its frozen binary. Only Qwen Turbo selects `.imagine/workflow-native/build/bin/sd-server`, after checking its manifest, base lock, patch and binary hash. The compatibility patch casts the zero-centered RMSNorm weight to F32 before CUDA `scale_bias`; the BF16 Turbo checkpoint otherwise trips the F32 assertion. This is a separate compatibility build, not a change to the accepted v2 inference settings.

Profiles support model/component selection, sampling/CFG/sigmas, LoRAs, references, upscale sizing and color/tone matching. Imagine merges request overrides into profile defaults and validates the result before admission. Encrypted default prompts use the existing unlocked privacy session inside Imagine and stay out of logs. The complete example is [qwen.yaml](../profiles/qwen.yaml).

## Tests

The reference Python virtualenv is needed only for comparison tests:

```sh
cd /home/thhel/ChatGPT/imagine
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --offline -- -D warnings
cargo test --workspace --offline
cargo build --offline -p st-proxy --bins --example oracle
cargo build --offline --example workflow-color
PY=/home/thhel/git/st-proxy/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
"$PY" proxy/tests/differential.py --reference /home/thhel/git/st-proxy
"$PY" proxy/tests/http_parity.py --reference /home/thhel/git/st-proxy --logs \
  --backends koboldcpp tabbyapi ollama prism sc117 orca sc117_128k orca_128k
"$PY" proxy/tests/gateway_parity.py --reference /home/thhel/git/st-proxy
python3 proxy/tests/root_launcher.py
"$PY" proxy/tests/stack_parity.py --reference /home/thhel/git/st-proxy
"$PY" proxy/tests/multi_parity.py --reference /home/thhel/git/st-proxy
"$PY" proxy/tests/failure_parity.py --reference /home/thhel/git/st-proxy
"$PY" proxy/tests/recovery_parity.py --reference /home/thhel/git/st-proxy
"$PY" proxy/tests/terminal_parity.py --reference /home/thhel/git/st-proxy
"$PY" proxy/tests/workflow_numeric.py \
  --aio-reference /home/thhel/git/comfyui-all-on-one-image-generation-node
python3 proxy/tests/check_sources.py --reference /home/thhel/git/st-proxy
```

`mcp_parity.py` is retained as the historical workflow-contract oracle; its exact metadata expectations intentionally differ from the new profile API. Current MCP profile/override, batch, seed, privacy and FIFO behavior is covered by `tests/gpu_coordination.rs` together with `tests/api_lifecycle.rs`.

These tests bind disposable loopback ports and Unix sockets. They never use active backend services. Differential log comparisons normalize timestamps, durations and disposable origins; independent asynchronous streams are compared in their own causal order with no events discarded.

Explicit real GPU checks, using only this run's own services and synthetic prompts:

```sh
cargo build --release --workspace
PYTHONDONTWRITEBYTECODE=1 /home/thhel/git/st-proxy/.venv/bin/python \
  proxy/tests/live_imagine.py --run
PYTHONDONTWRITEBYTECODE=1 /home/thhel/git/st-proxy/.venv/bin/python \
  tests/live-native.py --run
```

For real external backend acceptance, reserve the services/GPU, start the coordinated stack for the selected backend, and supply an explicit synthetic Comfy API graph:

```sh
/home/thhel/git/st-proxy/.venv/bin/python proxy/tests/live_stack.py --run \
  --backend sc117 --model YOUR_EXACT_MODEL_ID \
  --chat-url http://127.0.0.1:5002 --image-url http://127.0.0.1:8189 \
  --imagine-url http://127.0.0.1:8080 \
  --access-file "$PWD/.imagine/coordinated/data/access.json" \
  --comfy-workflow /absolute/path/to/synthetic-api-workflow.json \
  --output /tmp/st-proxy-sc117-live.json
```

Repeat for each configured backend and for `all` with each `backend/model` ID. This script exercises actual load/unload and can affect existing services; it is prepared but has not been run here. It is not proof of overlap exclusion by itself; the guarded synthetic tests cover that invariant.
