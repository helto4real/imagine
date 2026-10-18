# Stage 2 — approved ComfyUI memory policy for the BF16 comparison

2026-10-18. The repaired allocation-free placement oracle, selected Rust offload
binding and **three exact BF16 DiT comparison pairs** are complete for this
workload. Full Stage 2 remains in progress. The preceding full-residency BF16
attempt and successful INT8 controls remain preserved.

## S2-14 — explicit approval

Tomas answered the concrete memory-profile question:

> Ja, använd ComfyUI:s minnesstyrning i båda motorerna

This authorizes replacing the comparison's full-component GPU-residency
requirement with the frozen ComfyUI source's memory-budgeted CPU/GPU placement
in both engines. It does not authorize approximate numerics, alternate kernels,
changed autotuning, quantizing the BF16 DiT or changing image settings. The
profile remains 1600×896, 40 Euler/simple steps, seed 17, CFG 1, shift 0.69,
the same synthetic prompt, INT8 ConvRot text encoder and BF16 VAE. Async
offload remains disabled; the selected original path is synchronous static
ModelPatcher loading/casting. Dynamic vbars remain separate pending coverage.

Tomas subsequently extended this permission: “För framtida tester, du kan
alltid använda avlastning om det behövs, bara det är rust versionen som
används”. Future isolated tests may select necessary source-equivalent offload
without requesting approval again. The Rust engine must own and execute that
offload; invoking Python as its implementation is not authorized. Separate
Python oracle processes remain reference validation only. This permission does
not waive exact parity, authorize new approximations or alter production
services. Record the selected policy with each test so timings remain interpretable.

The implementation must preserve the original Linux reservations, Nvidia
weight ratio, sampling memory estimate, module ordering, strict fit comparison,
offload-buffer reservation, host registration and per-operation weight lifetime.
Physical counters may differ between separate processes; the policy is shared,
not a fixed placement list copied from the other backend.

The new workload uses its own explicit profile. Previous profiles, generated
outputs and timing evidence must not be overwritten. The prior resident INT8
pipeline timing is context, not a controlled measurement of precision alone
when the BF16 workload requires transfers.

## S2-15 — repair the development oracle before resuming

Two earlier unbounded placement-oracle attempts were interrupted when the app
crashed. The second log ends at the simulated 11 GiB placement budget; that
number is an input, not an observed memory peak. Source inspection identifies
an error in the oracle: even though construction used `meta`, original
`ModelPatcher.load` calls `_zero_init_parameter` outside that context. The
original helper's `device=None` then allocates real CPU zeros for full-sized
model parameters. Repeated constructors and reference cycles exacerbate RAM
pressure. No kernel OOM record was inspected, so an OS OOM kill is not claimed
as independently verified. No Rust image generation was running in those tests.

The repaired oracle retains a meta default throughout constructor/load/unload,
and a Torch dispatch guard rejects real-device input tensors and explicit real
allocation devices **before dispatch**. An intentional huge CPU allocation is
rejected by that guard. Every placement and partial-unload case checks that
parameters remain meta. Transfers, registration and patch writes are stubbed;
the original selection algorithms themselves execute unchanged. Cycles are
collected between cases and before interpreter teardown.

Per-process hard limits are installed before importing Torch: 2 GiB data,
8 GiB address space, 60 seconds CPU; the explicit invocation also uses a
90-second wall timeout. These limits never affect the parent/app or trained
inference. Single-thread BLAS/OpenMP settings apply only to this metadata
oracle, avoiding excessive import-time thread address reservations under the
limits. Earlier bounded import attempts terminated or were stopped; they
produce no accepted fixture. A later fake-object diagnostic attribute error
was also corrected before accepting the fixture.

The completed oracle records **330 modules, 10 initial placements, 40 partial
unloads and 24 budget calculations**. Peak process RSS is **795,404 KiB
(approximately 777 MiB)**, with all model parameters still meta. Its fixture,
module schema and executable reference script are saved under `inference/`
for later ComfyUI upgrades. This confirms the safety and output of this run;
the hard limits and allocation guard remain mandatory for future invocations.

## Implemented selected Rust binding

The new static BF16 DiT owner ports the Linux 400 MiB reservation, minimum
inference memory, Nvidia weight ratio, Qwen sampling estimate, descending load
order, strict fit threshold and ascending partial-unload order. It loads
constructor-checked BF16 parameters on CPU, retains selected modules on CUDA,
registers CPU storage with the original `cudaHostRegister(..., 1)` policy,
and releases registrations after synchronized use. Plain Linear transfers use
the original Preserve `empty_like` → nonblocking copy → dtype-cast order on
the current stream; no asynchronous offload streams are enabled.

Three focused tests pass: exact original placement/unload/budget decisions;
small host registration/copy/unregister lifecycle; and 12 exact synthetic
CUDA forwards with all-CPU/mixed CPU/GPU BF16 parameters, including 2/32 blocks,
split/fused MLP and reference conditioning. Initial test-only CUDA initialization
and metadata-placement mistakes were corrected without weakening assertions.

The selected owner intentionally rejects alternate constructors, quantized
DiT state and extra attention metadata outside this binding. Those remain
explicit missing integration coverage, not substituted execution. The existing
resident INT8 text/VAE path is retained when those components fit; the Python
reference uses the same permitted memory setting. This is not complete general
ComfyUI memory-manager, dynamic-vbar, LoRA or multi-model support.

## S2-16 — preserve reload lifecycle and original accounting

The first offload measurement series produced an exact first image pair, but
source review identified an incorrect Rust pre-reload eviction. Original
`load_models_gpu` removes same-model/clone registrations with `detach(false)`
before calling `free_memory`. Weights stay in place, and an isolated single
model is therefore absent from the eviction list. Rust now follows that order;
it checks allocator-cache pressure and uses the original load budget, with
partial unload only when original `partially_load` selects its negative-budget
branch. It does not invent a pre-eviction based on the model's offloaded size.

Original `load` also keeps previously patched GPU objects when a later fit
calculation skips their module. Its recomputed loaded-byte counter and actual
retained-object set must remain distinct; Rust preserves both original results
instead of silently correcting this accounting behavior.

The guarded meta oracle was extended with **60 original partial-reload cases**
and a two-call registration-lifecycle check. The latter observes an empty
registered-model list in both `free_memory` calls and exactly one
`detach(unpatch_all=False)`. All 134 placement/unload/reload/budget cases pass;
the expanded run's peak RSS is **799,872 KiB (approximately 781 MiB)**. The
three focused offload tests and their three separate Python-free invocations
pass after this correction. The other 15 successful focused/isolation results
are reused because their numerical implementation and inputs are unchanged.
All-target Clippy also passes again.

The initial series `qwen21-1600-40steps-bf16-offload-comparison-01` was stopped
by terminating only its own test process tree. Its data is retained for
diagnosis; none of its times enter the accepted benchmark. A completely new
three-pair series, suffix `comparison-02`, measures the corrected implementation.
No source/design exception or autotuning change was introduced.

## Measurement contract

The corrected series alternates Python→Rust, Rust→Python, Python→Rust, using
fresh processes and the same three selected model files for each image. The
comparison profile is
`inference/profiles/qwen21-1600x896-40steps-bf16-offload.json`.
The original registered profiles/defaults and earlier successful controls are
unchanged. The new profile changes the BF16 full-residency profile only in
descriptive purpose and `force_full_load_per_component=false`.

The strict gate checks all Float32 conditioning, latent and processed RGBA
bytes, finite values and decoded PNG pixels. Required shapes are `[1,20,4096]`,
`[1,64,56,100]` and `[1,896,1600,4]`; every image has exactly 40 callbacks.
Rust runs inside its existing network/PID/mount isolation without an available
Python interpreter, ComfyUI installation or production runtime. Python runs
the frozen original loaders, model, conditioning, full sampler and VAE with
the existing file/network audit and original-source hash verification.

Timing uses the preceding monotonic/CUDA-synchronized convention. Sampling
includes original sampling-time model preparation. Pipeline wall time includes
component loading, inference, unload, intermediate export and housekeeping;
launcher/build/preflight/import time and final image serialization are excluded.
Model hashes were verified and the OS model cache warmed before measurements.
Both engines use 23 CPU threads, the existing debug Rust executable and the
same optimized frozen Torch/Kitchen libraries. Clocks are not fixed.

The policy uses actual memory counters for each process; resident module sets
need not be identical between runs/backends. Saved memory records retain both
the original loaded-byte counter and selected/retained module names. That
counter is not a physical peak-VRAM measurement and can differ from the sum of
retained objects after a reload. No forced common budget, layer count, cache
threshold, precision or autotuning setting is introduced to equalize timings.

## Accepted results

All three corrected pairs pass exact Float32 conditioning, latent, processed
RGBA and decoded PNG comparison. Each Float32 boundary has one common hash
across all six completed images. All values are finite; unequal counts,
maximum error, MAE and RMSE are zero. The conditioning hash also matches the
preceding INT8 forty-step control, confirming unchanged text conditioning.
All six images have the requested 1600×896 dimensions and 40 sampling callbacks.

Seconds, median and observed range from three fresh processes per engine:

| Phase | Rust median | Python median | Rust range | Python range |
| --- | ---: | ---: | ---: | ---: |
| Text load | 0.983 | 1.248 | 0.972–1.009 | 1.217–1.331 |
| Text encoding | 0.321 | 0.489 | 0.314–0.334 | 0.487–0.502 |
| DiT load/initial placement | 2.527 | 2.549 | 2.189–2.672 | 2.470–2.634 |
| Forty-step sampling/preparation | **49.833** | **51.562** | **49.333–50.738** | **51.422–52.628** |
| VAE load | 0.308 | 0.209 | 0.262–0.333 | 0.176–1.058 |
| VAE decode/pixel processing | 0.775 | 0.672 | 0.666–0.807 | 0.666–0.822 |
| Measured pipeline | **55.033** | **58.082** | **54.898–56.336** | **57.259–58.811** |

The measured Rust pipeline median is **3.049 s / 5.25% shorter**, and its
sampling median is **1.729 s / 3.35% shorter**, for this sample. Medians are
not additive. Automatic placement, transfers, loading/housekeeping and unlocked
clocks all contribute; this is not an isolated measurement of language overhead
or kernel performance. No tuning was performed after these results.

At sampling, sums of retained-module parameter bytes are **11.784–11.909 GiB
for Rust** and **11.409–12.034 GiB for Python**. CPU weights/pinned bytes are
respectively **1.344–1.469 GiB** and **1.219–1.844 GiB**. These are tensor/module
inventories, not physical peak VRAM. Python run 3 reports a 12.003 GiB loaded-byte
counter while retaining 12.034 GiB of module parameters, demonstrating the
original reload accounting distinction described above. The independent
processes follow the same policy with their actual free-memory counters;
different retained amounts prevent attributing the timing gap solely to Rust.

For context, the preserved resident INT8 forty-step medians were 22.341/22.629 s
sampling and 27.081/27.250 s pipeline (Rust/Python). The BF16 workload here is
slower and requires offload. This comparison changes both DiT representation
and residency/transfer behavior; it does not isolate the cost of BF16 arithmetic
or quantify an offload-only slowdown.

## Image quality

Rust and the original Python reference produce **identical pixels**, so this
workload shows no quality difference between engines. The BF16 image has
clearly resolved butterfly wing veins/scales, fine edge hairs and flower-petal
texture. The preserved INT8 forty-step image has the same overall composition
and similarly clear detail, with small local differences. Visual inspection of
this one prompt/seed does not establish a general quality advantage for BF16.

| BF16 DiT, Rust | Preserved INT8 DiT, Rust |
| --- | --- |
| ![BF16 image](stage-2-qwen-image21-bf16-offload-evidence/rust-image.png) | ![INT8 control](stage-2-qwen-image21-1600-40steps-comparison-evidence/rust-image.png) |

[The BF16 Python image](stage-2-qwen-image21-bf16-offload-evidence/python-image.png)
has exactly the same pixels as the BF16 Rust image.

## Final verification and retained evidence

- All-target Clippy/formatting, Python AST and shell syntax checks pass.
- The full focused Qwen suite and its 18 separate Python-free invocations pass.
  After the reload correction, its three affected offload tests and three
  separate Python-free invocations pass again; unchanged numerical tests are
  reused rather than described as newly rerun.
- All **371** implementation/fixture/resource/profile inputs and **275**
  original source inputs verify. The full reference verifies 273 imported
  ComfyUI files against the Stage 0 manifest. All four Stage 0 locks and all
  32 original CUDA objects remain unchanged; complete model hashes verify.
- The final three-pair series passes the strict gate without an allowed
  numerical tolerance. Every measured Rust image passes Python-free isolation.
- The safe meta oracle retains 134 algorithm cases, the same-model lifecycle
  check and allocation-guard/process-limit evidence for future source upgrades.

The [evidence directory](stage-2-qwen-image21-bf16-offload-evidence/),
[summary](stage-2-qwen-image21-bf16-offload-evidence/summary.json) and
[checkpoint](stage-2-qwen-image21-bf16-offload-checkpoint.json) retain the accepted
timings, exact metrics, source snapshots, per-run policies/logs/isolation and
both synthetic images. The excluded initial series has a separately labelled
diagnostic record. Raw tensors remain in this task's isolated runtime.

No production runtime, cloud service, Computer Use, service replacement or
other user's process inventory was used. Only our own stalled import/test
process tree was stopped. Sandbox escalations were approved; there was no
automatic approval rejection. No commit or push was requested for this slice.

This validates **BF16 DiT + INT8 ConvRot text encoder + BF16 VAE** for the
recorded workload. All-BF16 text encoding, Turbo, editing, LoRA, broader shapes
and general/dynamic/multi-model memory behavior remain separate acceptance
requirements. Rust owns the offload implementation; Python is an independent
oracle only. No new source/design exception or autotuning change was needed.

## Reproduction

```sh
export IMAGINE_PARITY_ROOT=/tmp/imagine-rust-comfy-parity-20261017
export IMAGINE_REFERENCE_SITE=/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages
timeout 90s bash inference/tools/launch_reference.sh \
  inference/tools/reference_qwen_memory.py \
  --site "$IMAGINE_REFERENCE_SITE" --comfy /home/thhel/git/ComfyUI \
  --own-root "$IMAGINE_PARITY_ROOT/runtime/NEW_META_NAME" \
  --output "$IMAGINE_PARITY_ROOT/runtime/NEW_META_NAME.json" \
  --vae-dim 96 --vae-dec-dim 144
bash inference/tools/benchmark_qwen_first_image.sh \
  /home/thhel/git/ComfyUI /mnt/media/gemensam/AI/ComfyUI/models NEW_RUN_NAME \
  /home/thhel/ChatGPT/imagine/inference/profiles/qwen21-1600x896-40steps-bf16-offload.json
```

Use fresh names. The first command is allocation-guarded metadata validation;
the second explicitly loads trained weights and generates images. Neither is
part of ordinary product startup or automatic background work.
