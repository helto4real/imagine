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
  --mcp-port 8190 \
  --mcp-output-dir /run/media/veracrypt1/imagine/.mcp-cache
```

`start-proxy.sh` defaults to chat port **5002**, image port **8189**, and enabled MCP on **8190**, matching `st-stack.zsh`. ComfyUI defaults to upstream port 8188 and KoboldCpp to 5001. The root launcher uses private storage and does not inject a legacy fixed MCP output directory. Agents exchange files through key-scoped workspaces in Imagine's MCP root (`/run/media/veracrypt1/mcp-root`, set on the Imagine server with `--mcp-root`), never through host paths; see [MCP-WORKSPACES-DESIGN.md](../docs/MCP-WORKSPACES-DESIGN.md). Environment variables and CLI arguments can override these defaults.

`./start-proxy.sh` without arguments incrementally builds the local `main` binaries through the [shared product builder](../docs/PRODUCT-BUILDS.md) and runs `proxy/st-stack.zsh` in **all** mode. It starts or reuses every available backend in [the backends file](#backends-file), waits for readiness, then starts the proxy with the combined `backend/model` catalog that Pi expects. Models are activated on demand through the shared GPU coordinator. Imagine must be started separately first. Ctrl-C or `./start-proxy.sh --stop` stops the stack; `--log` shows backend logs. Set `ST_PROXY_LLM_BACKEND` to select another stack backend.

Proxy CLI options and unambiguous abbreviations configure the supervised stack too: `./start-proxy.sh --llm-b sc117 --chat-port 5002` starts that backend and the proxy. Use `--proxy-only` to connect to independently managed backends without starting them. `--help` lists existing CLI options plus the integration options. The usual `ST_PROXY_*` variables work; the new ones are `ST_PROXY_COORDINATOR_SOCKET` and `ST_PROXY_IMAGINE_URL`. `ST_PROXY_MCP_TOKEN` remains environment-only. Tokens are not logged.

Use `--llm-backend ID|all` with a backend id from the backends file; `--ID-url URL`, `--backend-url ID=URL` or `ST_PROXY_<ID>_URL` changes a backend's origin. `all` exposes `backend/model` IDs and requires distinct origins. Kobold router mode retains `--kobold-router-mode`, admin password, bootstrap/cache and default-model behavior. `--model-profiles "$PWD/config/proxy/model-profiles.json"` enables the copied profiles. `--check-backend` validates the selected backend without starting listeners.

`proxy/st-stack.zsh` preserves the original stack's menus, backend choices, colors, cleanup, TTS and SillyTavern options. Build first with `cargo build --release -p st-proxy --bins`, export the integration variables above, and start Imagine separately before launching this script. Its listener ports match the root launcher; inspect its environment options for other stack settings. It does not own the separate Imagine process. The Kobold bootstrap and Strata log filter are Rust binaries. No other stack launch scripts are replaced.

Coordination is enabled only when Imagine receives `IMAGINE_COORDINATOR_SOCKET`. Every direct API/UI and MCP job then reserves the same FIFO as LLM and Comfy requests. Unavailable coordination fails admission with 503; loss during generation stops the native engine. There is no standalone fallback. An idle image engine may remain resident until the next owner requests the GPU. Starting an independent uncoordinated Imagine instance bypasses this arrangement.

## Desktop UI through the proxy

`./start-ui.sh` defaults to `http://127.0.0.1:8189/imagine`. The image listener
forwards `/imagine/*` to `--imagine-url`, keeping the request body, session header,
idempotency key and streamed responses. Authorization is stripped on this route.
Imagine has one shared local workspace and requires no token. The proxy rejects
a non-loopback image listener when Imagine coordination is enabled. Its scheduler reserves the shared GPU ticket;
the forwarding request does not take an additional lease. The root launcher disables ComfyUI by default: no Comfy cleanup or health requests
are made, and Comfy routes return 503 `comfy_disabled`. To use Comfy alongside
Imagine, pass `--comfy-enabled true` (or `ST_PROXY_COMFY_ENABLED=true`); Comfy
routes then use the same image port outside `/imagine`, and confirmed Comfy GPU
release is required during handoffs. An unreachable enabled Comfy is never ignored.

The root `start-server.sh` / `start-proxy.sh` launchers default to the shared
`$XDG_RUNTIME_DIR/imagine/gpu.sock`. Custom data paths no longer require credential configuration.
`--imagine-access-file` and `ST_PROXY_IMAGINE_ACCESS_FILE` have been removed.
The UI needs only `--api`. Existing services must be restarted to use the updated
binaries together.

MCP keeps an admitted job in `recovering` during an Imagine status/result outage.
`get_job` continues to use the same job. Ambiguous submission responses are retried
with the identical request body and idempotency key; no new generation ID is created.
Restart Pi to load its updated `qwen-scene` profile instructions. Its output mount
must match `ST_PROXY_MCP_OUTPUT_DIR` if that variable is overridden.

## Imagine profiles and native compatibility

Imagine owns the [YAML profiles](../config/image/README.md), defaults and override validation. MCP exposes `list_profiles`, `describe_profile`, `register_image`, `generate`, `generate_batch`, `get_job` and `confirm_saved`. Pass `profile` instead of `workflow_id`, and choose `quick-draft` instead of `parameters.mode`. Process-local jobs, retry fingerprints, random or explicit 53-bit seeds, reference registration and consecutive batch ordering remain. MCP calls the local public Imagine endpoints without Imagine credentials; general Comfy proxy traffic still uses ComfyUI. `nsfw=true` is an explicit LoRA switch. The copied graphs/manifests remain historical fixtures, with their hashes preserved; `--mcp-workflows-dir` no longer controls MCP.

Video production adds `list_video_profiles`, `describe_video_profile`, `video_capabilities`, `video_project`, `video_register_reference`, `video_storyboard`, `video_run` and `video_assembly` (`src/mcp_video.rs`). They map to Imagine's `/v1/video/*` endpoints and keep no proxy state: project folders are confined to the MCP input directory and references to the MCP roots, a timed-out storyboard start is recovered with `video_storyboard list` and `bundle`, run and assembly admission forward `job_id` as `Idempotency-Key: mcp-<imagine session>-<job_id>`, storyboard bundle images become MCP image content blocks, and errors carry only Imagine's stable `code`, `at` and `detail`.

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
  --comfy-workflow /absolute/path/to/synthetic-api-workflow.json \
  --output /tmp/st-proxy-sc117-live.json
```

Repeat for each configured backend and for `all` with each `backend/model` ID. This script exercises actual load/unload and can affect existing services; it is prepared but has not been run here. It is not proof of overlap exclusion by itself; the guarded synthetic tests cover that invariant.

Image sizing accepts `max_side` from 256 to **4096 pixels**, in multiples of 32.
For `image_upscale`, **3200 is only the default** when omitted, not the maximum.

## Image delivery and temporary storage

Each MCP process creates a fresh `session-<uuid>` child of `ST_PROXY_MCP_OUTPUT_DIR`.
Each job has a private `job-<uuid>` directory inside that session
(default `/run/media/veracrypt1/imagine/.mcp-cache`). Batch items share this directory,
with one `<item-index>.png` per successful item. Results include `image_id`,
`path`, `bytes`, `sha256` and `storage: "temporary"`. These delivery image IDs
are distinct from IDs returned by `register_image`.

The client must copy every successful image to a unique persistent path under the returned `persistent_root` on the verified encrypted volume, outside the MCP cache,
check that the destination is a regular file, and compute its byte size and
SHA-256. Both must match the result metadata. This is a file integrity check:
do not open, render, visually inspect or analyze the image, or judge whether it
matches the prompt. Filename existence alone is insufficient.

After the job finishes, acknowledge verified copies (including subsets of a
partial batch):

```json
{
  "name": "confirm_saved",
  "arguments": {
    "job_id": "teapot-001",
    "images": [{
      "image_id": "<image_id from the result>",
      "saved_path": "/run/media/veracrypt1/imagine/teapot-001.png",
      "bytes": 123456,
      "sha256": "<SHA-256 computed from the destination file>"
    }]
  }
}
```

Replace the example size and digest with values measured from the copied file.
The proxy checks all acknowledgements against its own metadata before deleting
anything. Only confirmed files in that job's staging directory are removed;
the directory is removed when empty. Unconfirmed files remain only for the
current Imagine/proxy session. Identical confirmations are idempotent, including after a lost response.
If filesystem cleanup fails partway through, retry the same confirmation;
`get_job` records which images were already confirmed.
Confirmed images have `storage: "saved"`, `saved_path`, and `path` pointing to the
client's persistent copy, including in same-job generation retries.

The proxy relies on the client's content-integrity check, but also verifies that the
saved path is a regular file under the same trusted media destination. Container-only
paths or unverified mappings are rejected; never copy to plaintext workspace/tmp paths.
Job lookup and references remain session-local. Normal Imagine session loss or proxy
shutdown removes only that process's owned delivery copies and previews. Startup never
scans/deletes old cache entries. Crash leftovers may therefore remain on the encrypted disk.
Generated originals saved by Imagine survive session shutdown. Saved files under the
trusted destination can be registered directly without another input-directory copy.

Before opening the cache, the proxy fetches `/v1/media-destination` from Imagine and
verifies its host, mount, source, filesystem identity and root locally using the shared
media-destination code. Imagine must therefore be available before MCP proxy startup.
`mcp_output_dir` must be a direct child of that trusted root; a legacy override such as
`/run/media/veracrypt1/mcp-images` is rejected without creating it. Root/cache symlinks,
wrong/unmounted/read-only/unwritable/full destinations fail closed, with no fallback.
All cache/media creation uses directory file descriptors pinned to that volume; atomic
partials remain there. Metadata, normalization and HTTP upload/download use RAM.
Preview paths remain available for the MCP contract, so normalized previews are kept
only in this encrypted process cache. Historical plaintext cache/session files are not
inspected, migrated or deleted by this implementation.

## Backends file

`config/proxy/backends.json` (under `--config`; `--backends PATH` or
`ST_PROXY_BACKENDS` selects another file) lists the LLM backends. Adding, removing
or disabling a backend needs no rebuild; restart the stack. Each entry has:

| Field | Meaning |
| --- | --- |
| `id`, `label` | Backend id (lowercase letters, digits, `_`; the `backend/` model ID prefix) and display name |
| `adapter` | Compiled protocol: `koboldcpp`, `tabbyapi`, `ollama`, `llama_router` (Prism) or `strata` |
| `origin` | Loopback origin with an explicit port |
| `enabled` | `false` hides the backend from `all`, the menu and `--llm-backend` (default `true`) |
| `in_all` | `false` keeps an enabled backend out of `all` (Ollama) |
| `requires` | Paths such as weights; when one is missing the backend is skipped with a warning |
| `required` | `true` stops the stack when the backend does not become ready |
| `default_model` | Default preset (`llama_router`) or model (`tabbyapi`) instead of the global option |
| `launch` | `dir`, `command` (`{port}` is the origin's port), `match` (lowercase process text), `all_args` (added in `all` mode), `terminal_filter` |

In `all` mode the stack skips backends whose `requires` are missing and leaves out a
backend that does not become ready within `ST_STACK_START_TIMEOUT`, unless it is
`required`; the proxy then starts with `--exclude-backends`. `ST_PROXY_EXCLUDE_BACKENDS=a,b`
leaves backends out explicitly. `st-proxy --print-backends table` shows each backend
as available, disabled or skipped. Only one KoboldCpp backend can be in `all`. Model
profiles name `backend/model`; a profile for a backend that is not in the file is
skipped with a warning. `ST_STACK_<ID>_DIR` and `ST_STACK_<ID>_COMMAND` still
override one backend's launch.

## SC117 Swift 1.5

`swift` (32768 context, port 48648) and `swift_128k` (131072 context, port 48650)
use the Strata adapter, lazy startup and confirmed GPU handoffs. Their native ID is
`sc117-swift-1.5-qwen3.8-flash-next-gsq-rco-abliterated-iq3_xxs`;
all-mode IDs add `swift/` or `swift_128k/`.

The pinned installation is in `~/git/qwen-next-server`; the 128k launcher is in
`~/git/local-model-runtimes/models/qwen-next/swift-128k`. Weights and native pack
are in `~/models/strata-sc117-swift-1.5-qwen3.8-flash-next-gsq-rco-iq3_xxs`.
The exact-model sampling profile follows the Swift model card; explicit caller
settings win. The runtime defaults to high (Qwen xhigh), with existing context,
memory and output limits. Passive catalog requests do not load the model.

Run the focused synthetic adapter tests with `cargo test --release --locked
--offline -p st-proxy --test swift`. Prepare the updated proxy through
`python3 tools/product_build.py prepare proxy`; an existing process keeps its
old binary until explicitly restarted through `start-proxy.sh`.
