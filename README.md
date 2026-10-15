# Imagine

A local image/video generation backend with a Rust API and no ComfyUI runtime
dependency. The Rust service runs Qwen Image 2.1 through our frozen, optimized
**stable-diffusion.cpp** engine. MiniMax H3 is declared but not enabled yet.

Start with [the design](design-document.md), the
[frozen 1600×896 profile](experiments/qwen-image21-sage/BASELINE.md) and the
[experiment decision register](experiments/qwen-image21-sage/TEST-DECISIONS.md).

The standalone [Rust st-proxy](proxy/README.md) adds shared LLM/ComfyUI/Imagine
GPU coordination and MCP generation through Imagine YAML profiles. Its [parity report](proxy/PARITY.md)
separates synthetic comparisons, real GPU evidence and remaining differences.

## Start the API

On this configured machine:

```sh
./start-server.sh
```

This incrementally builds the Rust release server and starts it at
`http://127.0.0.1:8080`. It loads the native engine only when a job runs.
The launcher uses the proxy GPU queue at `.imagine/gpu.sock`; start
`./start-proxy.sh` with your backend options before submitting jobs.
The server starts without password or YubiKey login. API bearer tokens remain
required. Prompts and job history stay in memory; temporary images are ordinary
files removed at session end. See [session privacy](docs/PRIVACY.md).
Ctrl-C clears history and stops the server and its worker. Optional arguments:
`--bind 127.0.0.1:8081 --data /private/path/imagine-data`.

First start creates `.imagine/access.json` with a client token and a separate
admin token. Local credentials, jobs and media are ignored by Git. Use Rust/Cargo
and the frozen native executable/model files described below. A fresh clone must
restore the frozen binary; the API does not substitute a different build.

With `curl` and `jq`, from the project root:

```sh
TOKEN=$(jq -r '.clients.local' .imagine/access.json)
curl -sS -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8080/v1/models
curl -sS -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: first-lighthouse' \
  -d '{"model":"qwen-image-2.1-int8","operation":"text_to_image","prompt":"A lighthouse on a rocky coast at sunrise, realistic photography","seed":42}' \
  http://127.0.0.1:8080/v1/jobs
```

Follow the returned `status_url` or `events_url` with the same bearer token.
When `status` is `succeeded`, download `/v1/assets/{id}` using the ID in
`output.assets[0].id`. Defaults are 1600×896, 40 ER-SDE/beta steps, CFG 2.5 and PNG.
Editing accepts ordered reference asset IDs, with a default 1 MP resize target
per image. Other resolutions and H3 execution are deliberately unavailable in v1.

To release all model VRAM owned by this service and pause the queue:

```sh
ADMIN=$(jq -r '.admin_token' .imagine/access.json)
curl -sS -H "Authorization: Bearer $ADMIN" -H 'Content-Type: application/json' \
  -d '{"active_job":"wait"}' http://127.0.0.1:8080/v1/runtime/unload
# Follow the returned operation status URL; then resume when wanted:
curl -sS -X POST -H "Authorization: Bearer $ADMIN" http://127.0.0.1:8080/v1/runtime/resume
```

Use `active_job:"cancel"` to stop an active generation immediately. Read the
[frozen API contract](docs/API-BASELINE.md) for ownership, reference limits,
restart behavior, schema and examples. [Validation](docs/API-VALIDATION.md)
separates simulated lifecycle tests from real native/GPU checks.

## Desktop UI

With Imagine and the proxy running in separate terminals:

```sh
./start-ui.sh
```

The native Rust app uses the Helto design system. Enter a prompt, optionally
upload and reorder reference images, choose **Spara i → Välj mapp**, then
**Generera**. It uses the API's standard settings and saves each PNG under a
unique filename in your chosen folder. The folder choice persists across starts.

Default output: `.imagine/images`; preferences: `.imagine/ui-settings.json`.
The app connects through `http://127.0.0.1:8189/imagine` and reads the local
client token from `.imagine/access.json`. `start-server.sh` and `start-proxy.sh`
use the same `.imagine/gpu.sock` by default. ComfyUI is not required by these
launchers. MCP images use `/run/media/veracrypt1/mcp-images`, matching Pi.
No direct-API fallback is used.
See [UI setup and behavior](docs/UI.md) for custom API paths and recovery.

## Build the native engine

The tested Linux build uses CMake, Ninja, GCC/G++, CUDA at `/opt/cuda`, and an
RTX 4070 Ti SUPER (SM89). CUDA 13.4 was used for the accepted measurements.
Python 3 is used for setup and benchmark tooling, not native inference.

```sh
python3 native/prepare.py
./build-inference.sh
./stable-diffusion.cpp/build-cuda/bin/sd-server --help
```

The setup script downloads pinned upstream revisions into `stable-diffusion.cpp/`
and applies the patches in `native/patches/`, including the modified ggml kernels.
It never overwrites an existing checkout. An interrupted initial setup may leave
a partial checkout; inspect it before removing it and retrying.
The exact CUTLASS header snapshot used in the experiments is included under
`third_party/cutlass/`; it does not depend on a local Python package cache.

```sh
python3 native/prepare.py --check
```

This verifies upstream revisions, the 29 recorded native source hashes, patch
checksums and 822 CUTLASS headers. It does not certify a new toolchain's generated
binary or image output. Adapt the explicit compiler/CUDA paths and architecture
in `build-inference.sh` for other machines; those machines are outside the
measured baseline scope.

The native checkout is ignored by the outer Git repository. **When changing it,
export both the stable-diffusion.cpp and ggml diffs, including new files, into
`native/patches/` and update the lock/checksums before committing.** Committing
only the outer repository will not capture new unexported native edits.

## Frozen local baseline

The independently preserved, hash-checked baseline executable is intentionally
not distributed through Git. The repository contains its identity, settings,
synthetic prompts and compact measurement evidence. A fresh clone can build the
engine, but the strict frozen launcher requires the exact archived executable;
it does not silently accept a rebuilt development binary as the frozen baseline.

For a machine with that executable and the model files:

```sh
export QWEN_MODEL_ROOT=/path/to/models
./experiments/qwen-image21-sage/run.sh --dry-run
./experiments/qwen-image21-sage/run.sh --serve
# Or, after stopping the server, replay one first request and two warm requests:
./experiments/qwen-image21-sage/run.sh --run
```

Pillow is needed for the replay client. Model files are not supplied; their
relative filenames are listed in the baseline document. An optional ignored
`experiments/qwen-image21-sage/local-settings.json` can contain
`{"model_root": "/path/to/models"}`; the environment variable takes precedence.
Native development builds can use the same flags in `baseline-environment.json`,
but must be validated separately before being called the frozen baseline.

## Measurements and cleanup

Historical v1 saves 590.625 MiB of VAE workspace versus the preceding
control, with 12/12 reference images pixel-identical. Its measured warm request
to PNG mean was 39.219 s. This is not a fresh simultaneous comparison with
ComfyUI and not a universal bit-identity guarantee.

All experiment reports and decision explanations are retained. The compact
accepted evidence is in `experiments/qwen-image21-sage/evidence/1600x896-v1/`.
Old generated images, raw logs, traces and redundant build artifacts were
removed. Historical `runs/` paths in reports are provenance identifiers, not
downloadable assets. Archived one-off harnesses may need those removed inputs
and the original ComfyUI/Kitchen environment; they are not standalone test suites.
The current baseline launcher and its offline checker use the preserved files.

No model weights, private ComfyUI data, credentials, generated images or frozen
binaries belong in Git. The only published prompts are the synthetic benchmark
prompts. Personal filesystem roots have been replaced with portable paths or
placeholders in reports and archived harnesses.

## Third-party notices

The engine and ggml patches are based on the MIT-licensed upstream projects;
their notices are retained in `native/licenses/`. CUTLASS has its own license
and per-file notices in `third_party/cutlass/`. Downloaded upstream dependencies
retain their respective licenses. These notices do not license model weights.

Current native baseline: `qwen-image21-1600x896-v2` (accepted 2026-10-15).
The Rust service and experiment launcher select v2; v1 is retained as control.
The strip algorithm is dynamic; 1600×896 identifies the validated profile.
See [current baseline](experiments/qwen-image21-sage/BASELINE.md).

Generation defaults now live in [YAML profiles](profiles/README.md). See the complete
[Qwen example](profiles/qwen.yaml); API clients select `profile` and send only overrides.
