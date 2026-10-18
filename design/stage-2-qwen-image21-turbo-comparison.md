# Stage 2 — Qwen Image 2.1 Turbo, 1600×896

2026-10-18. **Completed: three exact matched Turbo pairs at 1600×896**, using
the designed eight-step explicit sigma profile and original static synchronous
offload. Earlier INT8 and BF16 controls remain unchanged. Stage 2 remains open.

## S2-17 — explicit Turbo sampling profile

Tomas requested the same Turbo comparison according to the agreed design,
with offload when needed and documented timing/image quality against original
ComfyUI. The design's explicit Turbo workload is **eight Euler steps, CFG 1**,
with Float32 sigmas `[1, .978453, .954180, .926626, .895080, .845148, .704534,
.414568, 0]`. This matches the current source profile
`profiles/quick-draft.yaml` and the source-side custom-sigma contract in
`workers/comfy/runtime.py` / `workers/comfy/test_worker.py`. It is an explicit
test configuration, not a universal schedule inferred from a checkpoint name.

The separate profile is
`inference/profiles/qwen21-1600x896-8steps-turbo-offload.json`. It uses local
`qwen_image_2.1_turbo_bf16.safetensors`, the same INT8 ConvRot Qwen3-VL text
encoder and BF16 VAE as the preceding controlled comparisons, the same
synthetic butterfly prompt, seed 17, batch 1, 1600×896, source shift 0.69 and
no references/LoRA. The current product source profile also selects this INT8
text encoder; this is not an all-BF16 text-encoder acceptance claim.

The original CFG=1 shortcut discards the negative branch in
`comfy.samplers.sampling_function`. The controlled driver retains its existing
positive-only conditioning path. The product's zero-negative flag therefore
does not introduce a second denoising branch for this test. General negative
encoding/guidance remains separate coverage; it must not be claimed from this
CFG=1 workload.

The previously verified Linux/Nvidia static synchronous memory policy is
selected. Rust owns placement, transfer and registration; Python is only a
separate oracle. Compiler/async offload remain disabled, PyTorch attention,
Kitchen startup dispatch, BF16 compute and default cache settings remain
unchanged. No autotuning or numerical tolerance is changed.

The local Turbo file has **265 BF16 tensors**, 14,230,280,616 total file bytes,
and no `.comfy_quant` tensors. Its constructor matches the existing BF16 DiT
binding. No weight conversion/quantization is substituted.

## Narrow implementation change and validation

The development driver and two host guards register the new profile. Explicit
sigma values are converted to CPU Float32, exported and moved to the execution
device. The Python oracle passes the same CPU Float32 tensor through original
`comfy.sample.sample(..., sigmas=...)`; original `CFGGuider.outer_sample`
moves it to the execution device. The explicit schedule is not shifted again
or replaced by the default simple schedule. Existing profiles retain their
default-schedule path.

The strict comparison now additionally checks every explicit sigma byte,
alongside all conditioning, latent, processed Float32 RGBA and PNG pixels.
All checks remain exact and require finite values. Every image must contain
exactly eight callbacks. The new sigma boundary is conditional on an explicitly
registered custom schedule; existing three-boundary comparisons are preserved.

All-target Clippy/formatting and Python/shell syntax checks pass. A complete
hash comparison against the BF16 checkpoint confirms unchanged model, loader,
offload, sampler arithmetic, attention, VAE, native and fixture implementations.
Only the driver and three host tools differ, plus the new profile. The earlier
18 focused/18 Python-free checks, including the later three offload retests,
are reused for those unchanged bodies. Model hashes are verified before timed
runs, warming the OS model cache. No unsafe meta-oracle rerun is needed.

The new series uses three fresh-process pairs in Python→Rust, Rust→Python,
Python→Rust order, with one GPU process at a time. Timing uses the preceding
monotonic/CUDA-synchronized convention, including per-component load/unload
and intermediate export in pipeline wall time. Build, launcher, preflight,
interpreter imports and final image serialization are excluded. Both engines
use 23 CPU threads and the frozen optimized Torch/Kitchen libraries; Rust is
the existing debug build. GPU clocks and automatic placement can vary.

## Accepted numerical and timing results

All three pairs pass strict byte equality for Float32 conditioning
`[1,20,4096]`, sigmas `[9]`, latent `[1,64,56,100]` and processed RGBA
`[1,896,1600,4]`, plus exact decoded PNG pixels. All values are finite and all
unequal counts, maximum error, MAE and RMSE are zero. Each Float32 boundary
has one hash across all six runs; PNG file hashes repeat within each encoder
and pairwise decoded-pixel equality establishes identical pixels across all
six images. Every run has exactly eight callbacks. The conditioning hash also
matches the retained INT8 forty-step control. Both engines used 23 CPU threads.

Seconds, median and observed range across three fresh processes per engine:

| Phase | Rust median | Python median | Rust range | Python range |
| --- | ---: | ---: | ---: | ---: |
| Text load | 1.044 | 1.263 | 1.026–7.917 | 1.255–3.386 |
| Text encoding | 0.317 | 0.504 | 0.312–0.322 | 0.501–0.566 |
| DiT load/initial placement | 2.740 | 2.801 | 2.500–5.246 | 2.716–8.924 |
| Eight-step sampling/preparation | 10.181 | 10.314 | 10.160–10.277 | 10.271–10.436 |
| VAE load | 0.805 | 0.921 | 0.419–1.110 | 0.222–0.997 |
| VAE decode/pixel processing | 0.665 | 0.675 | 0.662–0.738 | 0.663–0.685 |
| Measured pipeline | 19.394 | 18.552 | 15.726–23.276 | 17.292–23.376 |

The sampling median is **0.133 s / 1.29% shorter in Rust**. The measured pipeline
median is **0.843 s / 4.54% longer in Rust**. The total ranges overlap widely,
and three processes with variable loading do not establish a reliable overall
winner. Phase medians are not additive. All measurements are retained; no
slower run was discarded or repeated to improve the result.

| Matched pair / order | Rust pipeline | Python pipeline |
| --- | ---: | ---: |
| 1, Python → Rust | 19.394 | 23.376 |
| 2, Rust → Python | 23.276 | 18.552 |
| 3, Python → Rust | 15.726 | 17.292 |

The first Python DiT load took 8.924 s; Rust run 2 text load took 7.917 s,
compared with approximately 2.5–2.8 s DiT and 1.0–1.3 s text loads in the
fastest processes. Hashing warmed the OS cache before measurement but does not
guarantee continued page residency. The evidence localizes the observed
variation to load/housekeeping phases; it does not independently prove disk
cache eviction, system RAM pressure or another process as the cause. GPU clocks
were not fixed. No tuning or extra retries followed these observations.

At sampling, retained parameter inventories are **11.565 GiB CUDA + 1.688 GiB
CPU/pinned in all Rust runs**, and **11.472–11.628 GiB CUDA + 1.625–1.781 GiB
CPU/pinned in Python**. The saved module sets and reported counters show that
both follow the same source policy with their own actual memory counters.
These inventories are not physical peak RAM/VRAM measurements. No forced layer
count, shared artificial budget, async policy or autotuning change was used.

For context, the preserved same-resolution controls have these medians:

| Workload | Rust sampling | Python sampling | Rust pipeline | Python pipeline |
| --- | ---: | ---: | ---: | ---: |
| Standard INT8 ConvRot, 40 Euler/simple steps, resident | 22.341 | 22.629 | 27.081 | 27.250 |
| Standard BF16 DiT, 40 Euler/simple steps, offload | 49.833 | 51.562 | 55.033 | 58.082 |
| Turbo BF16 DiT, 8 explicit-sigma Euler steps, offload | 10.181 | 10.314 | 19.394 | 18.552 |

The Turbo workload changes checkpoint, step count and schedule, and some
controls differ in residency. This contextual table does not isolate precision,
step cost, distillation quality or Rust language overhead.

## Image quality

Rust and ComfyUI produce **identical pixels**, so there is no quality difference
between engines on this test. The Turbo image resolves butterfly wing scales,
veins and fine pale fringes despite only eight steps. Compared with the retained
standard forty-step images, the butterfly is larger in the frame, the red flower
has a dense cone-shaped center, and the background is darker olive with a
smoother appearance. The standard image has longer open petals and a lighter,
more mottled background. These are visibly different compositions from different
weights/schedules; one synthetic prompt and seed cannot establish a general
quality ranking or prove equal detail on other subjects.

| Turbo BF16, 8 steps, Rust | Standard BF16, 40 steps, Rust |
| --- | --- |
| ![Turbo image](stage-2-qwen-image21-turbo-comparison-evidence/rust-image.png) | ![Standard BF16 control](stage-2-qwen-image21-bf16-offload-evidence/rust-image.png) |

[The Turbo Python image](stage-2-qwen-image21-turbo-comparison-evidence/python-image.png)
has exactly the same pixels as the Turbo Rust image. The preserved
[standard INT8 forty-step image](stage-2-qwen-image21-1600-40steps-comparison-evidence/rust-image.png)
also remains available for comparison.

## Final verification and retained evidence

All **372 implementation/profile/resource/fixture inputs**, **275 original
source inputs**, all four Stage 0 locks and all 32 original CUDA objects verify.
The reference checks 273 imported ComfyUI files against the frozen manifest.
Complete SHA256 hashes of all three selected model files are retained.
The exact source delta from the preceding BF16 acceptance is the new profile
and four driver/host files; model/offload/kernel arithmetic remains unchanged.
Clippy/formatting and syntax checks pass; the unchanged numerical suite is reused
as described above. All three measured Rust processes pass Python-free isolation.
GPU sandbox escalation was approved; no automatic approval was rejected.

The [checkpoint](stage-2-qwen-image21-turbo-comparison-checkpoint.json) hashes
all files in the [evidence directory](stage-2-qwen-image21-turbo-comparison-evidence/):
three strict comparisons, six result/memory/log records, three isolation reports,
the two PNGs, explicit sigma bytes, timing/memory summary, model hashes, source
snapshots, validation and input-delta records. Large transient Float32 exports
stay in isolated development storage; their exact hashes are in the comparisons.
Earlier BF16/INT8 evidence, including excluded attempts, is preserved. No commit
or push was requested for this continuation; these changes remain uncommitted.

## Reproduction and scope

With the pinned Stage 1/2 dependency environment restored and enough RAM/VRAM
available for one model process, run the retained development-only entrypoint
with a fresh output prefix:

```sh
IMAGINE_PARITY_ROOT=/path/to/isolated-development-root \
IMAGINE_REFERENCE_SITE=/path/to/frozen/site-packages \
bash inference/tools/benchmark_qwen_first_image.sh \
  /path/to/frozen/ComfyUI /path/to/authorized/models fresh-turbo-prefix \
  "$PWD/inference/profiles/qwen21-1600x896-8steps-turbo-offload.json"
```

The trained benchmark is opt-in and is not part of ordinary tests. Its Rust
process has no Python interpreter, ComfyUI source tree or network access. The
three selected checkpoints are mounted read-only; output and caches are confined
to the task's isolated development roots. No production prompts, histories,
logs or images are used. No product API/UI, ordinary Dreamchat configuration,
frozen native baseline, services or unrelated GPU processes are changed.

This selected Turbo T2I acceptance does not complete Stage 2: full family
loading/detection and errors, all-BF16 text encoder, editing/vision/reference
conditioning, LoRA, remaining samplers/guiders and general dynamic/shared memory
policies remain explicit coverage gaps. No new parity exception, numerical
tolerance, autotuning rule or source behavior change is introduced.
