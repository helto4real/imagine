# Rust inference backend

`rust` is Imagine's default backend. Start it with:

```sh
./start-server.sh
```

The launcher builds the main-checkout server and the allowlisted image/H3
adapters over the pinned Rust core, writes a verified combined runtime manifest
and starts the server. See [product builds](PRODUCT-BUILDS.md) for the source
boundary, build profile, update procedure and explicit rollback.
This uses the already prepared local dependencies; it does not install
packages, download models, launch Python inference or rebuild CUDA libraries. Startup
validates paths/hashes. The first scheduled generation launches the worker
after acquiring the existing shared GPU reservation.

Alternatives remain explicit:

```sh
./start-server.sh --backend native   # preserved stable-diffusion.cpp engine
./start-server.sh --backend comfy    # original Comfy core Python worker
```

There is no automatic backend, sampler or scheduler fallback. Changing backend
requires restarting Imagine. This change does not restart an existing server,
replace another running service or change the proxy's GPU coordination policy.

## Current model coverage

| Model | Product path | Sampling |
| --- | --- | --- |
| Qwen Image 2.1 INT8 ConvRot | Text-to-image, ordered reference editing, LoRA, existing two-pass character sheets | Euler / ER-SDE; simple / beta; explicit sigmas |
| Qwen Image 2.1 BF16 | Same Qwen product paths, with the ordinary BF16 DiT | Same sampler/scheduler bindings as INT8 |
| Qwen Image 2.1 Turbo BF16 | Text-to-image, ordered reference editing, LoRA | Same sampler bindings; shipped Turbo profile retains its eight manual steps and CFG 1 |
| MOSS Image SFT INT8 ConvRot | Text-to-image | Euler; normal / sgm_uniform; profile defaults 50 steps, CFG 3, shift 5 |

MOSS appears in the existing model selector and has a separate settings group.
Its Qwen3.5 encoder and Flux.2 VAE cannot be mixed with Qwen Image's components.
The existing model library discovers their headers in the usual recursive Comfy
model directories. Only these selected MOSS filenames are exposed initially:

- `moss_image_sft_int8_convrot.safetensors`
- `qwen3.5_9b_int8_convrot.safetensors`
- `flux2-vae.safetensors`

For a profile-based API request, post `{"profile":"moss","prompt":"A blue
butterfly on a yellow flower"}` to `/v1/jobs`. The UI uses the existing model
selector and component library.

The source-controlled `config/image/profiles/moss.yaml` gives the local component paths.
A library selection can resolve another nested location with the same supported
filenames. MOSS references, LoRA, layer skipping, MTP decode and other variants
remain unavailable and fail explicitly. Ordinary Qwen BF16 is selectable as
`qwen-image-2.1-bf16` when `qwen_image_2.1_bf16.safetensors` is discovered. It
shares the Qwen settings/LoRA group and the existing INT8 Qwen3-VL encoder and
BF16 Qwen VAE. The new product selection requires the Rust backend; other
backends reject it before queue admission. No existing profile is changed.
Full sampler/scheduler/model parity is still later work.

The bounded BF16 HTTP/reference control matches the original ComfyUI pixels
exactly with automatic offload and no LoRA. Fresh BF16/Turbo LoRA controls also
match all numerical boundaries, pixels and observed component/DiT lifecycle
states under automatic offload; see the
[Stage 2 follow-up](../design/stage-2-upgraded-followup.md). Earlier failed pairs
with different live free-memory inputs remain recorded separately. These
selected controls do not establish all workload/pressure combinations.

Unsupported `res_multistep` selections fail before queue admission on `rust`.
Its metadata advertises only its implemented choices. Native/Comfy keep their
own supported choices. Switching models restores that model's sampling defaults;
Qwen LoRA selections survive switching to MOSS and back.

## Execution and lifetime

The Rust server supervises one private Rust executable through bounded binary
pipes. It reuses the existing Comfy adapter's transport and process lifecycle,
not its Python implementation. The worker calls the same reusable Rust model,
sampling, Kitchen, AIMDO and memory-management components as the validated
inference controls. No Comfy graph, UI, node executor or Python inference is
involved. LibTorch and native CUDA libraries remain required.

Prompts and raw reference/output pixels travel only through private pipes.
The worker writes no prompts, model intermediates, generated media or diagnostic
traces to files. Only kernel/compiler caches persist. Images returned through
the API use the existing session storage and cleanup. Worker stderr is suppressed
and errors are static allowlisted codes. All selected weights/LoRAs are loaded
from the job's frozen library selection; changed file stamps fail the job.
Original static CPU/GPU offload is automatic, with shared model ownership and
host pins. Components unload between phases; base weights are not retained
between jobs by this initial adapter. Kernel/runtime state can remain in the
worker until unload or cancellation.

Cancel, unload, coordinator loss, failed generation and shutdown use the same
scheduler stop-and-wait path. The child is killed/reaped before releasing the
GPU reservation, and dies if the server dies. No new queue, GPU service or
worker concurrency is introduced. Model execution remains single-threaded in
one child. CPU tensor threads retain the selected controlled value of two;
numerical/kernel and autotuning policies are unchanged.

## Local dependency deployment

`prepare-rust-backend.sh` uses `native/product/build-lock.json` and the same
build path as normal server startup. It builds the server and both workers
without launching services or models. The lock records prepared library paths,
hashes, compiler versions and headers. Missing or changed artifacts stop the
build. Legacy `IMAGINE_RUST_DEPENDENCIES` and `IMAGINE_RUST_SITE` overrides are
no longer used by this preparation path.

The completed artifact directory contains `runtime.json` and `build.json`;
the script prints its path. Normal startup passes this versioned manifest
explicitly, rather than selecting `comfyui/runtime-current.json`.
For an independently prepared, caller-managed deployment, skip the launcher's
default native preparation and select its manifest explicitly:

```sh
./start-server.sh --backend rust --rust-runtime /absolute/path/runtime.json
```

`--comfy-root` and `--comfy-python` apply only to `--backend comfy`.
The selected reference is ComfyUI `446ef4e005fc7134d53ce6869ebd77d2134cd39c`,
Kitchen 0.2.38. Runtime deployment is not a rolling reference-upgrade policy.

## Validation

See [the integration report](../design/rust-backend-server-integration.md) for
synthetic versus trained evidence and remaining limits. Ordinary model tests
do not load trained weights. The explicit `inference/tools/check_server_backend.py`
uses a newly created isolated project, selected assets and synthetic inputs.
It exercises the real default server, HTTP validation/admission, model-library
selection, queue, progress, worker, PNG download and shutdown cleanup. Its
comparisons use decoded pixel equality, not PNG compression bytes or tolerances.
