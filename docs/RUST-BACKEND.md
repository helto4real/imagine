# Rust inference backend

`rust` is Imagine's default backend. Start it with:

```sh
./start-server.sh
```

The launcher incrementally builds the independent Rust worker and release
server, writes the verified local runtime manifest and starts the server.
This uses the already prepared local dependencies; it does not install
packages, download models, launch Python or rebuild CUDA libraries. Startup
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

The source-controlled `profiles/moss.yaml` gives the local component paths.
A library selection can resolve another nested location with the same supported
filenames. MOSS references, LoRA, layer skipping, MTP decode and other variants
remain unavailable and fail explicitly. General BF16 Qwen checkpoints remain
supported by the inference development controls; this adapter retains Imagine's
existing public INT8 and official Turbo model IDs. It adds no standalone BF16
product selection. Full sampler/scheduler/model parity is still later work.

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

On this machine, `prepare-rust-backend.sh` defaults to the prepared
`comfyui/moss-20261018` dependency/build root and the installed Comfy environment's
LibTorch shared libraries. It uses Cargo offline. Override these build locations
with `IMAGINE_RUST_DEPENDENCIES` and `IMAGINE_RUST_SITE` when deliberately
preparing another local deployment. Missing artifacts fail with a path; there
is no automatic dependency installation. The worker currently uses the selected
Cargo development build, not a claimed tuned production benchmark executable.

The ignored `comfyui/runtime-current.json` records the worker/native-library
hashes, compiler profile and approved launch environment. The server checks it
at construction and before launching a new worker. For an independently
prepared deployment, skip the launcher's default preparation and select its
manifest explicitly:

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
