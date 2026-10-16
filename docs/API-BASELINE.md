# Imagine API baseline v1

Accepted: 2026-10-14. Revision: `imagine-api-v1`.

This is the implementation contract for the first Rust service. It supersedes
older illustrative API examples in `design-document.md`. The native profile was explicitly advanced to `qwen-image21-1600x896-v2` on
2026-10-15. API revision and request validation remain unchanged. The new profile
adds dynamic VAE strips while preserving precision and memory policy.
On 2026-10-16, Qwen Image 2.1 passes with references additionally adopted the
[hybrid/prefetch baseline](../native/reference-cache/README.md). Frozen v2 remains
the text-only profile and verified fallback; other models retain their own
selection. The [character-sheet operation](CHARACTER-SHEETS.md) composes two
inference passes within one job.

## Scope and architecture

One Rust executable, Axum HTTP, Tokio task/process management, SQLx/SQLite in memory for
jobs and metadata, disposable local files for media. There is no ComfyUI dependency.
Request handling and generation orchestration remain in Rust. Python/Pillow are
used by the optional live acceptance test.

Rust owns validation, shared local assets, a session-local FIFO queue,
idempotency, worker lifecycle and output publication. One job runs at a time.
Only that job reaches a persistent loopback `sd-server`. It is started lazily,
with the pinned binary SHA-256, frozen environment and three model paths from
the existing baseline manifest. Inherited `SD_*`/`GGML_*` overrides are cleared;
profiling injection is rejected. Normal requests reuse the native process and
its process-local CPU conditioning cache of four entries. Idle weight retention
remains off; auto-fit uses the native CUDA scratch accounting without a manual
VRAM margin. Native stdout/stderr are discarded, and image metadata is disabled.

Module responsibilities:

| Module | Responsibility |
| --- | --- |
| `contract` | Typed requests, semantic validation, model metadata/schema |
| `api` | HTTP, session checks, local workspace, SSE |
| `storage` | Session-local SQLite and atomic media files |
| `scheduler` | Sole owner of dispatch, cancellation and unload ordering |
| `engine` | Native process and `/sdcpp/v1` adapter |
| `config` | Loopback binding, single-server lock, project/model paths |

The backend trait is a small testable boundary, not a plugin framework.
The v2 environment remains pinned by its accepted manifest. The Qwen reference
executable is separately pinned by `native/reference-cache/reuse-budget-source-lock.json`;
changing native profiles stops the previous child before launching another.

## Start and local access

```sh
./start-server.sh
# Optional paths/address:
./start-server.sh --bind 127.0.0.1:8081 --data /private/path/imagine-data
```

The script builds the Rust release executable using `Cargo.lock`, then replaces
itself with the server. Subsequent builds are incremental. It does not build the
native engine. Rust/Cargo, the existing frozen native executable and model files
are prerequisites. A new clone does not contain the frozen executable: see the
root README. There is no silent fallback to an unvalidated development build.

Model-root precedence: `QWEN_MODEL_ROOT`, then the existing ignored
`experiments/qwen-image21-sage/local-settings.json`, then project `models/`.
Relative roots resolve against the project directory.

Default address: `127.0.0.1:8080`. Imagine binds only to loopback. The proxy
requires a loopback image listener when Imagine coordination is enabled. There
is no built-in TLS or CORS. The native engine port remains internal and dynamic.

All endpoints work without Authorization, including runtime controls and operation
status. API, UI and MCP share one local workspace. The storage layer retains its
internal owner field, always set to `local` by the API; it is not an authorization
boundary. Idempotency keys share that workspace too. Local processes are trusted.
`access.json` is neither created nor read; existing files are left untouched.
The data directory remains mode 0700. `X-Imagine-Session` still rejects stale
requests after restart, and responses retain `no-store`/`nosniff` headers.

SQLite holds prompts, settings and metadata only in process memory. Internal
media is stored as ordinary files. Startup and shutdown clear history and
temporary media. The server starts without terminal login; API tokens
are not required. See [session privacy](PRIVACY.md) for MCP exports,
crash behavior, credentials and saved-file exclusions. A file lock prevents two
servers from sharing a data directory.

## HTTP contract

| Endpoint | Result |
| --- | --- |
| `GET /v1/models` | Configured model capabilities; no GPU loading |
| `GET /v1/models/{id}` | Defaults, constraints and request JSON Schema |
| `POST /v1/assets` | 201, uploaded image asset metadata |
| `GET /v1/assets/{id}` | Local workspace image bytes |
| `POST /v1/jobs` | 202, session-local ID, status/events URLs |
| `GET /v1/jobs/{id}` | Authoritative state, effective settings, output or error |
| `GET /v1/jobs/{id}/events` | SSE `job` snapshots, closes on terminal state |
| `POST /v1/jobs/{id}/cancel` | 202 cancellation acceptance |
| `GET /v1/runtime` | Paused state, queue count, active flag, native phase/PID |
| `POST /v1/runtime/unload` | 202 unload operation ID/status URL |
| `POST /v1/runtime/resume` | 200, scheduler unpaused |
| `GET /v1/operations/{id}` | Unload operation status |

Requests and responses use JSON except media upload/download and SSE. Uploads
are raw PNG/JPEG/WebP bytes with the matching `Content-Type`, not multipart.
Accepted images are decoded for validation, then original bytes are retained.
Maximum upload: 32 MiB, 8192 pixels per axis and 16,777,216 pixels; at most two
image validations execute concurrently. EXIF orientation is not normalized by
the API; clients should upload images with the intended pixel orientation.

Job objects have `id`, `status`, timestamps in milliseconds, `effective`,
`output`, `error`, and `execution_to_saved_ms`. The latter includes model loading
when required and output publication, but excludes queue wait and client
transfer. It is also present on terminal failures/cancellations as elapsed
execution time, not a claim that an image was saved.

SSE supplies snapshots, not a durable event log. Reconnect using GET status or
resubscribe; a terminal job emits its final snapshot. Both GET status and SSE
include the active job's phase and numeric `progress`, or null when unavailable.
Phases include `verifying_profile`, `loading_model`, `preparing_inputs`,
`inference`, `sampling`, `finishing_image` and `saving_output`.

Example: `{"completed":12,"total":40,"elapsed_seconds":10.8,
"remaining_seconds":25.2,"updated_at_ms":1791980000000}`. These are sampling
steps, not a percentage of the entire generation. ETA uses the mean of observed
native step durations. Loading, input preparation, final VAE decoding and saving
are excluded. `updated_at_ms` is the observation's Unix timestamp, so clients can
derive an estimated sampling finish time without shifting it on every poll.

With previews disabled, the frozen native HTTP endpoint has no step counters.
The adapter reads the hash-pinned executable's `pretty_progress` stdout records
(`s/it` or `it/s`), rejecting weight-loading byte counters and unrelated output.
Only sequential steps matching the current request qualify. This depends on the
current single-pass/batch-one profile with VAE tiling disabled; revisit the adapter
if that profile or the native output format changes. Raw output is bounded in
memory, discarded, never saved or forwarded to clients. The native binary and
inference settings are unchanged; previews remain disabled. Progress is scoped
to the active public job ID and cleared at terminal states. Unknown progress is
never replaced by a timer estimate. HTTP responses have `Cache-Control: no-store`.

Errors use `{"error":{"code":"..."}}`. Typical errors: 404 `asset_not_found`/`job_not_found`, 409
`runtime_suspended`/`idempotency_conflict`/`unload_in_progress`, 422
`invalid_request` or a semantic validation code. Oversized bodies may use Axum's
413 rejection. Unknown request fields and unsupported settings are rejected,
never silently forwarded to native CLI arguments. Worker errors have safe
codes such as `model_weights_missing`, `frozen_binary_unavailable`,
`frozen_binary_hash_mismatch`, `native_startup_failed`, `generation_timeout`,
`native_generation_failed`, and `output_save_failed`; raw engine diagnostics,
paths and prompt contents are not returned as errors.

## Generation and reference editing

```json
{
  "model": "qwen-image-2.1-int8",
  "operation": "image_edit",
  "prompt": "Place the woman in a Scandinavian harbor. Preserve her identity.",
  "negative_prompt": "artifacts, blurry",
  "width": 1600,
  "height": 896,
  "seed": 42,
  "sampling": {
    "steps": 40,
    "cfg_scale": 2.5,
    "sampler": "er_sde",
    "scheduler": "beta"
  },
  "inputs": {"reference_images": ["asset_REPLACE_WITH_UPLOADED_ID"]},
  "model_options": {
    "qwen_image_2_1": {"reference_max_pixels_per_image": 1000000}
  },
  "output": {"format": "png", "count": 1},
  "client_metadata": {"scene_id": "harbor"}
}
```

For `text_to_image`, omit `inputs` or use an empty reference list. Minimum
request: model, operation and non-empty prompt. Defaults are shown above;
negative prompt defaults to empty and an omitted/null seed resolves once to a
random unsigned 32-bit seed. Explicit seed 0 is valid. Defaults are persisted in
`effective`, together with API/profile revisions. Client metadata is an opaque
JSON object (maximum serialized size 4096 bytes), with no effect on inference.

Current validated profile: **1600×896 only**, PNG, one result, ER-SDE/beta.
Steps may be 1–100, CFG 0–20. The benchmark remains 40 steps/CFG 2.5/seed 42;
allowing other sampling values is not a universal pixel-identity claim. Each
prompt is limited to 32,768 UTF-8 bytes. LoRAs, masks, first/last frames and
other samplers are rejected. Output dimensions now accept multiples of 32, each
edge 128–4096 with the longest edge at least 256. This supports the UI's AIO ratio
and max-side controls; it does not expand the frozen validation evidence.
Model metadata separates allowed `dimensions` from `validated_resolutions`.
All sizes use the existing frozen runtime configuration, including its VAE
optimizations; only 1600×896 has the recorded acceptance. New sizes may differ
numerically or exceed available memory. There are
no public CUDA flags, local paths or arbitrary native options.

Qwen reference editing accepts **1–10 ordered image asset IDs**. Model support
is distinct from local validation: one reference has been tested; ten references
have not been benchmarked on this 16 GB card. Resources depend on total reference
tokens. Multiple references may fail for memory reasons and are never silently
dropped or reduced in count.

`reference_max_pixels_per_image` defaults to 1,000,000; allowed configured values
are 262,144–4,000,000. The server uses AIO's shrink-only geometry: proportionally
resize oversized inputs with Lanczos3, then center-crop each axis down to a
multiple of 32. Small inputs are never enlarged. Each reference has its own
budget, independent of output area and reference count. The API uses literal
pixels; AIO's MP control uses 1,048,576 pixels per MP, while Imagine's uses
1,000,000. A 1600×896 source at 1,000,000 becomes **1312×736 / 965,632 pixels**.
Extreme reference aspect ratios producing zero or >8192 aligned dimensions are
rejected. The effective job contains ordered `reference_preprocessing` records
with source and calculated VAE dimensions, attributed to
`aio_shrink_center_crop_v1`. Preprocessed PNGs go to the unchanged native engine
with `resize_before_vae=false`, so vision and VAE receive the same geometry.
The Rust Lanczos3 implementation is not claimed pixel-identical to ComfyUI/Pillow.
Original uploads are preserved. Historical native-resize timing/quality results
do not validate this new preprocessing path.
The 1 MP default is provisional evidence from the portrait/edit comparison,
not a guarantee of unchanged likeness for all subjects.

The common `video` object is reserved as `{ "frames": 56, "fps": 24,
"audio": true }`; video operation names are in the typed contract. The
`minimax-h3` model is listed with `available:false`, `execution_enabled:false`
and its planned capabilities, including the 17*k+5 frame grid and 24 fps.
**No video jobs execute in v1.** H3 and its model-specific options will only be
enabled after separate native validation. Future H3 first/last-frame and
reference-image modes must retain their mutual-exclusion rules.

## Idempotency, persistence and lifecycle

Send `Idempotency-Key` (1–128 characters) on job creation. It is scoped to the
client. Repeating the same typed request returns the original ID, even while
paused or after a terminal result; a different request under that key is 409.
Defaults are normalized before fingerprinting. An omitted seed and an explicitly
supplied seed are distinct requests, even if the original resolved to that seed.
No key means each submission is a new job. There is no implicit retry of a
failed or interrupted generation.

States: `queued`, `running`, `cancelling`, `succeeded`, `failed`, `cancelled`,
`interrupted`. Upload/output files are written to temporary files, synced and
atomically renamed; asset metadata is committed before a job can become
`succeeded`. A crash between file publication and the final job update may leave
an orphan media file; it cannot publish a false success. Startup removes
old media without opening it. Accepted entries and parameters live in memory.

Restart creates an empty, unpaused queue and new session ID. Jobs, operations,
assets and idempotency keys do not survive a session.
Ctrl-C/SIGTERM stops the owned native process, waits for exit and marks the
active job interrupted. Linux parent-death signaling and `kill_on_drop` prevent
an owned native worker surviving server death. This is Linux-targeted process
management, matching the current CUDA host.

A native failure fails the current job and stops/reaps that worker. The next
queued job can start a fresh worker. The adapter waits up to 300 seconds for
startup and 3600 seconds per generation (including startup); individual native
HTTP calls have a 30-second timeout. A failure to confirm worker shutdown never
produces a successful unload; the scheduler/server stops instead.

## Full unload and resume

```json
{"active_job":"wait"}
```

Or `{"active_job":"cancel"}`. Runtime controls apply to the shared local queue:

1. Persist scheduler pause before acknowledging unload. No further dispatch.
2. `wait` finishes only the active job. It does not drain the queue.
3. `cancel` requests cancellation and terminates the active worker. The native
   engine has no cooperative mid-generation cancel at this revision.
4. Stop/reap the owned process, clearing weights, CUDA context and conditioning
   cache. Report operation `succeeded` only after confirmed exit.
5. Remain paused, preserving queued jobs. New job creation returns 409.

A queued job can still be cancelled while paused. Cancellation or unload
accepted before a generation result is published takes precedence over that
result. A cancellation arriving after success returns `job_already_finished`.
During a pending unload, another unload or resume returns `unload_in_progress`.
`resume` unpauses; loading happens lazily for the next queued/new request.

Only this server's native process is stopped. Other applications' VRAM is not
released or reported as ours. `unloaded` means the owned CUDA process has exited,
not that total GPU memory use is zero.

## Validation

```sh
cargo fmt --all --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
./start-server.sh --help
# Explicit real GPU acceptance; requires idle GPU, native files and Pillow:
python3 tests/live-native.py --run
```

The Rust tests use a controlled fake engine for queue/unload ordering,
idempotency, shared local API access, input validation, cancellation, restart isolation,
worker failure recovery and terminal SSE. They are not GPU performance tests.
The live test uses only synthetic frozen prompts and its own generated reference,
checks the frozen RGBA hash before and after unload/resume, performs a 1 MP edit,
and verifies both unload policies and native PID exit. Results remain in ignored
`.imagine/test-runs/`. See `API-VALIDATION.md` for the completed local run.
