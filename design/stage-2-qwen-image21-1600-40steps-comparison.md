# Stage 2 — 1600×896 with 40 steps

2026-10-18. Tomas explicitly requested the same Rust/ComfyUI comparison at
1600×896 with 40 steps, all other settings retained, and documented results.
The preceding [four-step comparison](stage-2-qwen-image21-1600-comparison.md)
remains the control. Full Stage 2 remains in progress.

## S2-12 — change step count alone

The separate [40-step profile](../inference/profiles/qwen21-1600x896-40steps-int8.json)
differs from the four-step profile only in `steps` and the descriptive `purpose`.
Both previous profiles and all their evidence remain unchanged.

The synthetic prompt remains “A blue butterfly resting on a red flower, soft
daylight.”, seed 17, batch 1, 1600×896 RGBA. The sampler remains **Euler**, scheduler
**simple**, CFG **1**, source FLUX shift **0.69**, with no LoRA or references.
The exact same INT8 ConvRot DiT/Qwen3-VL weights and BF16 VAE are used. BF16
compute, PyTorch attention, default prefix-cache dtype/auto thresholds, native
allocator, 23 CPU threads, static patchers, disabled Comfy compiler/async offload,
full component residency and unload between components are retained. Kitchen
Triton stays disabled as the frozen ComfyUI startup requires without opt-in.
No kernel, autotuning, precision, cache or memory policy changed.

Only the development driver's registration/result label and the two host-side
registration guards were extended to recognize the new profile. They continue
to reject unregistered configurations. Hash comparison against the preceding
checkpoint confirms all model/tensor/attention/sampling/VAE/native implementations
are unchanged. The default development profile is still the original 512×512
four-step test.

The frozen native baseline and test decisions were read before execution. The
separate stable-diffusion.cpp baseline is untouched and is not used as a
substitute for the requested direct ComfyUI Python comparison.

## Measurement and acceptance

Three fresh-process pairs alternate Python→Rust, Rust→Python, Python→Rust. Both
sides use the original matching source path; Rust executes with no Python or
ComfyUI runtime available inside its isolated process. Python executes the
frozen original loaders, conditioning, model, sampler and VAE headlessly, with
source-hash verification and read/network audit. All prompts, outputs, caches
and logs belong to this isolated development task; ordinary production runtime
is not an input.

The strict gate compares all Float32 context, latent and processed RGBA bytes,
finite values, and decoded PNG pixels. The expected shapes remain `[1,20,4096]`,
`[1,64,56,100]`, `[1,896,1600,4]`. No tolerance or visual-only acceptance replaces
the exact criterion.

Times use the same monotonic clock and CUDA synchronization at phase/callback
boundaries as earlier runs. Pipeline wall time includes component loading,
inference, unload, intermediate export and housekeeping, excluding launcher,
build, preflight, interpreter imports and final PNG/raw-image serialization.
OS model caches are warm; each image uses a fresh process. This is the existing
debug Rust development build with optimized frozen LibTorch/Kitchen libraries.
Clocks are not fixed.

## Test-storage recovery

The initial Python image completed. The initial Rust process completed all 40
steps and VAE processing, then failed while writing its Float32 pixel file with
OS error 122, **disk quota exceeded**. The file was partial and there was no
completed Rust result. This is a failed artifact write, not a numerical parity
result. Neither member of that initial pair is included in measured results.

The managed command sandbox also hit the temporary-storage quota while creating
its own mount target. Aggregate capacity and only this task's directories were
inspected; no production data or process inventory was read.

All isolated runtime entries were copied from
`/tmp/imagine-rust-comfy-parity-20261017/runtime` to the ignored task-only directory
`/home/thhel/ChatGPT/imagine/.rust-comfy-runtime/runtime`. Before removing the
old copy, **all 16,274 file/link entries were verified**, including each regular
file's complete hash and each symlink's unchanged absolute target. The old
runtime path is now a symlink, preserving all existing paths and evidence.
Source, models, native objects, compiler configuration and cached contents were
not changed. The source/profile/fixture snapshot still verifies after migration.
No model memory, offload, tiling, quantization or autotuning policy was changed.

The complete 40-step measurement restarts with a fresh run prefix. Both backends
write to the same new storage. A fresh three-pair **four-step control is also
repeated on this storage**, because the changed output medium is concrete new
evidence justifying repetition. The original four-step results remain preserved.
The new paired four-step control, rather than the old tmpfs pipeline times, is
used for the step-count timing comparison. Pipeline timing includes intermediate
export, so that storage distinction must remain explicit.

## Forty-step results

All **three complete matched pairs pass exact comparison** at the context,
latent, Float32 RGBA and decoded PNG boundaries. Each Float32 boundary has a
single common hash across all six runs. All values are finite; unequal counts,
maximum error, MAE and RMSE are zero. The same 24,698,880 Float32 bytes per image
are checked as in the four-step geometry test. This exact result applies to
this prompt, seed, profile, locked implementation and GPU; it does not complete
the model-family acceptance matrix.

Seconds, median and observed minimum–maximum from three fresh processes per
backend:

| Phase | Rust median | Python median | Rust range | Python range |
| --- | ---: | ---: | ---: | ---: |
| Text load/residency | 1.199 | 1.466 | 1.158–1.239 | 1.390–1.477 |
| Text encoding | 0.325 | 0.509 | 0.309–0.334 | 0.499–0.514 |
| DiT load/residency | 0.994 | 0.992 | 0.919–1.022 | 0.956–0.996 |
| Forty-step sampling | **22.341** | **22.629** | 22.316–22.555 | 22.612–22.764 |
| VAE load/residency | 0.919 | 0.257 | 0.428–1.413 | 0.192–0.999 |
| VAE decode/pixel processing | 0.685 | 0.654 | 0.662–0.703 | 0.644–0.654 |
| Measured pipeline | **27.081** | **27.250** | **26.355–27.805** | **26.896–27.829** |

Rust's median pipeline is approximately **0.170 s / 0.62% shorter** in this
sample; its sampling median is approximately **0.287 s / 1.27% shorter**. The
pipeline ranges overlap substantially and VAE loading varies markedly. Treat
overall performance as effectively similar in this experiment, rather than a
general claim that Rust is faster. Medians are not additive, and no kernel-level
cause or release-build speedup has been established. No tuning followed the
measurements.

## Fresh four-step control on the same storage

The repeated four-step control also passes all three exact pairs, with one common
context/latent/pixel hash per boundary across all six control images. Its outputs
match the preserved pre-migration four-step outputs byte for byte. The completed
pre-migration Python forty-step output also matches the new forty-step output
hashes. Migration changed the output storage, not the generated values.

The following step-count comparison uses the two fresh datasets on the same
storage. Values are medians of three fresh processes per backend:

| Steps | Rust sampling | Python sampling | Rust pipeline | Python pipeline |
| --- | ---: | ---: | ---: | ---: |
| 4 | 2.345 s | 2.400 s | 6.645 s | 6.874 s |
| 40 | 22.341 s | 22.629 s | 27.081 s | 27.250 s |

Ten times as many steps costs about 9.5 times as much sampling time and about
four times as much measured pipeline time on this workload. The pipeline also
includes loading, conditioning, VAE and housekeeping, so it does not scale only
with step count.

The fresh control's small Rust/Python pipeline difference has the opposite
direction from the preceding tmpfs run set. That reinforces the limited strength
of small timing differences in these samples. The known storage change is not
an isolated attribution of the timing change: cache/load/system variation and
unlocked clocks remain. The old control's measured values are preserved and
are not substituted into this same-storage table.

## Image quality

At 40 steps the butterfly has clearly resolved wing edges, veins and fine
surface detail, and the red flower has distinct petal texture and central
structure. The large blur and ghosted wing appearance of the four-step image
are largely absent. This is a qualitative visual observation of one synthetic
prompt, not a formal perceptual-quality benchmark.

Rust and Python produce that improved image **pixel for pixel**, so this case
shows no quality loss from the independent engine. No prompt, seed, guidance,
sampler, scheduler, shift, weight or precision change was used to improve it.

| Four steps, preserved image | Forty steps, Rust |
| --- | --- |
| ![Four-step control](stage-2-qwen-image21-1600-comparison-evidence/rust-image.png) | ![Forty-step image](stage-2-qwen-image21-1600-40steps-comparison-evidence/rust-image.png) |

[The forty-step Python image](stage-2-qwen-image21-1600-40steps-comparison-evidence/python-image.png)
has identical pixels. The report's evidence directory retains both synthetic
PNGs, per-run timing/metric/source records, isolation reports and registered
profiles. Raw activations remain in the isolated runtime directory.

## Verification and limits

- All-target Clippy with warnings denied, formatting and Python/shell syntax
  checks pass. The original inference implementation and previous profiles are
  unchanged; no component arithmetic was modified or component suite rerun.
- Three complete 40-step pairs and three fresh four-step pairs pass strict
  Float32/PNG comparison. All six measured Rust images pass Python-free
  isolation, and callback records have exactly the requested 40 or 4 steps.
- The original implementation/profile/fixture snapshot verifies after storage
  recovery and after measurements. The full reference's imported ComfyUI files
  are checked against the frozen original source hashes.
- Complete hashes of all three model files, all four Stage 0 locks and all 32
  original CUDA objects are rechecked against the preceding control.
- The initial failed output-write attempt is retained as a failure record and
  excluded from successful-run counts. No numerical mismatch was accepted.

The [summary](stage-2-qwen-image21-1600-40steps-comparison-evidence/summary.json)
and [checkpoint](stage-2-qwen-image21-1600-40steps-comparison-checkpoint.json)
record the exact evidence. No production runtime, cloud service, Computer Use,
process inventory, service interruption, product integration, commit or push
was involved. Sandbox escalations were approved; the quota failure was an
environmental error, not an automatic approval rejection.

Peak VRAM and kernel-level performance attribution were not measured. This
single prompt/seed and static resident profile do not establish all workloads,
other GPUs, BF16/Turbo, editing, LoRA, general memory policy or full Stage 2
acceptance. No new source/design departure or autotuning change was needed.

## Reproduction

```sh
export IMAGINE_PARITY_ROOT=/tmp/imagine-rust-comfy-parity-20261017
export IMAGINE_REFERENCE_SITE=/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages
bash inference/tools/benchmark_qwen_first_image.sh \
  /home/thhel/git/ComfyUI /mnt/media/gemensam/AI/ComfyUI/models NEW_RUN_NAME \
  /home/thhel/ChatGPT/imagine/inference/profiles/qwen21-1600x896-40steps-int8.json
```

Use a fresh run name. The existing arguments/defaults preserve both four-step
controls. This script explicitly generates trained images and is not invoked
by ordinary component tests.
