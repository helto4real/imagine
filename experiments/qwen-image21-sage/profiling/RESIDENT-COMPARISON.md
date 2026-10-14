# Resident sd-server with prompt cache versus headless AIO/Kitchen

Date: 2026-10-14 (Europe/Stockholm); run directory timestamps use UTC.
RTX 4070 Ti SUPER 16 GB. No inference implementation or defaults are changed.

## Measured result

**Prompt-cache reuse works and preserves each engine’s reference image.**

| Six warm requests per engine | Native resident server | Headless AIO/Kitchen |
| --- | ---: | ---: |
| Request → saved PNG, mean | 40.216 s | 38.814 s |
| Request → saved PNG, median | 39.384 s | 38.708 s |
| Sampling, mean | 38.197 s | 37.412 s |
| Conditioning, mean | <0.01 s | 0.001 s |
| VAE decode, mean | 1.767 s | 0.678 s |

Native request-to-PNG range: **38.863–43.168 s**.

Comfy request-to-PNG range: **37.922–39.491 s**.

Native minus Comfy mean latency: **1.402 s** (3.61%). Median difference: **0.675 s**. All samples are retained, including the native A request at 43.168 s. These are local results with shared-GPU variation, not a guaranteed engine ranking.

| Series / engine | First request → PNG | Process startup → first PNG |
| --- | ---: | ---: |
| A / native | 41.832 s | 42.242 s |
| A / comfy | 44.099 s | 48.653 s |
| B / comfy | 42.038 s | 44.296 s |
| B / native | 40.897 s | 41.205 s |

First-image times are individual observations, excluded from the warm means.

| Series / engine | Warm PNG samples | Warm sampling samples |
| --- | --- | --- |
| A / native | 39.029, 38.863, 43.168 s | 36.970, 36.720, 41.180 s |
| A / comfy | 39.387, 37.922, 39.491 s | 37.978, 36.567, 38.016 s |
| B / native | 41.469, 39.089, 39.679 s | 39.490, 37.090, 37.730 s |
| B / comfy | 38.712, 38.705, 38.665 s | 37.365, 37.359, 37.187 s |

All eight native images and eight Comfy images were decoded and checked against their respective references. The six warm requests in each engine have two prompt-cache hits and no text encoding. Every image still performs 80 diffusion forwards. No inference source change or rebuild was needed.

## What is being tested

The existing `sd-server` creates one `sd_ctx_t` and reuses it for successive
requests. Its existing `--conditioning-cache-size 4` stores positive and negative
conditioning in the context's bounded RAM cache. `SDCondition` owns CPU tensors;
no conditioning tensors are serialized or written to disk. This tests the
accepted native configuration through the actual localhost server API.

The configuration keeps Sage2++, direct V on, split K off, F32 gate/up,
BF16 attention/residual and all accepted fusions. Gate/up–SwiGLU uses per-call
packing; `GGML_CUDA_GATE_UP_DIRECT=0`. Automatic memory fitting remains enabled
without a manual margin. The frozen server binary is
`runs/gate-up-direct-build/sd-server`, SHA256
`2a13601caa0ad9cb552c19869d5fac36379f3f2d5960cbf7dd94d75c9a730022`.

Resident context does **not** mean every model weight and compute graph remains
pinned in VRAM. `ExecutionScope` calls `end_runners()` after each generation;
runner cleanup releases workspaces and evicts compute-backend parameters. The
automatic residency manager also makes room for the VAE during a request.
Warm native logs still report diffusion tensor loading/transfers. This experiment
keeps that existing policy and tests process/context and conditioning reuse.
Further persistent graph/weight-residency changes would be a separate experiment.

ComfyUI runs the installed AIO Qwen adapter headlessly, using the existing
privacy-preserving harness. The process and CPU prompt-encoding cache persist
for each series. AIO still invokes its model loaders each request and ComfyUI
manages GPU residency. No ComfyUI HTTP server or private workflow is read.

## Protocol

Same physical model files and approved synthetic positive/negative prompts,
1600×896, seed 42, 40 ER_SDE/beta steps, CFG 2.5, batch one, no LoRA, reference
image, second pass or live previews. Native uses 11 CPU threads and CPU RNG;
Comfy uses its previously measured default CPU-thread setting and Kitchen INT8
attention. No precision, sampler or attention changes are introduced to align
images between engines.

Each series contains a first image in a new process and three warm requests.
The first native series has a substantial timing spread, so a second series per
engine is run in reverse order. All samples, including the slow native request,
remain in the results. Order: native A, Comfy A, Comfy B, native B. Engines never
run model jobs concurrently. Each process exits before the next one starts.
First images are reported separately from the six warm samples per engine.
Desktop GPU use and unlocked clocks remain sources of variation; the block
order is counterbalanced, but this is not per-image alternation or a confidence
interval. OS file caches are not flushed: first-image numbers are process-start
measurements, not physical-storage cold-start guarantees.

Native request latency starts immediately before POST `/sdcpp/v1/img_gen` and
ends after fetching, base64-decoding and writing/closing the PNG file. The
native async job is polled every 50 ms; polling and HTTP/JSON/PNG transport are
included in latency. Preview is explicitly `none`; the compatibility API's
implicit per-step previews are avoided. The server runs on loopback only.

Comfy latency starts at the adapter generation call and ends after PIL PNG save.
It includes tensor-to-image conversion and PNG encoding/writing but no HTTP
transport. Both endpoints represent a successfully saved PNG, not a disk-fsync
promise. Thus the comparison aligns process/cache lifecycles and file-save
endpoints, while retaining a small transport difference against native.
Startup-to-first-PNG is also recorded: native from server process launch, Comfy
from the harness's early startup timer (after its initial standard-library
imports, before torch/Comfy imports). Those startup boundaries are close but not
identical.

The native conditioning timer is logged at 0.01-second precision. A displayed
0.00 seconds means below its display precision, not literally zero work.
Sampling timers retain each engine's setup/loading work within that phase.
No CUPTI or torch profiler runs during these ordinary timing series.

## Validation and privacy

Every native image must match the native RGB reference
`c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c`.
Every Comfy image must match its established RGB reference
`526a2535b24c7517b4787ef77870692a1972ed07058075633972b87b8c62a276`.
These are different references; cache reuse is checked within each engine,
not as a claim of cross-engine pixel identity.

Native warm requests require two conditioning cache hits, no text-encoder graph
calls and 80 single-segment diffusion forwards. Comfy warm requests require two
conditioning cache hits, zero encodes, 80 diffusion forwards and 40 sampling
steps. No final image or diffusion-step skipping substitutes for generation.

Only approved synthetic inputs, selected source files and benchmark-created
logs/images are used. Private ComfyUI logs, history, queues, configuration and
user workflows remain excluded. Native PNG metadata is disabled; prompt-cache
contents are not persisted. Source and binary hashes are checked. The harness
shuts down its own server and monitors; GPU/loopback execution uses approved
sandbox escalation. No dependencies, system services, commits or pushes are
created. LoRA/prompt-change cache invalidation is not separately exercised here.

## Artifacts

- [Resident comparison driver](compare-resident.py).
- [First native/Comfy series](../runs/20261013T215723Z-resident-plrhkv3m/).

- [Reverse-order series and combined results](../runs/20261013T220330Z-resident-reverse-qsx728m8/).
- [Reverse-order driver](repeat-resident-reverse.py).
- [Saved-result validator](summarize-resident.py).
