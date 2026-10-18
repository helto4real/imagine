# Stage 2 — Qwen Image 2.1

Started 2026-10-18. Status: Stage 2 is **in progress**. Resident DiT, language,
tokenizer, image VAE, initial parameter loading and selected sampling services
are implemented. [First-image preparation](stage-2-qwen-image21-first-image.md)
records the earlier readiness gate. The subsequent
[trained comparison](stage-2-qwen-image21-first-comparison.md) records actual
INT8 ConvRot image generation, exact numerical checks and matched timing runs.
The subsequent [1600×896 comparison](stage-2-qwen-image21-1600-comparison.md)
also passes three exact pairs with only geometry changed; the observed pipeline
gap shrinks from 7.85% to 3.08%, with essentially equal sampling time.
The [40-step continuation](stage-2-qwen-image21-1600-40steps-comparison.md)
passes three exact 1600×896 pairs with improved visual detail and about 27 s
pipeline time for either backend. It also repeats the four-step control after
an isolated test-storage migration; all prior evidence remains preserved.
The subsequent [BF16 DiT attempt](stage-2-qwen-image21-1600-40steps-bf16-comparison.md)
retains the same INT8 text encoder and other settings. Its original Python
reference runs out of GPU memory during full-residency DiT loading, before
sampling. Rust's isolated constructor/header preflight passes; there is no
completed BF16 image or matched pair in that initial checkpoint. A change to
the retained full-residency setting was presented for approval at that point.
The [approved offload continuation](stage-2-qwen-image21-bf16-offload.md)
subsequently repairs the metadata oracle, implements original static synchronous
offload in Rust and passes three exact BF16 DiT pairs at 1600×896/40 steps.
Measured pipeline medians are 55.033 s Rust / 58.082 s Python, with automatic
placement differences recorded. Text stays INT8 ConvRot and VAE stays BF16;
this does not establish all-BF16 encoder or complete Stage 2 acceptance.
The [Turbo continuation](stage-2-qwen-image21-turbo-comparison.md) now passes
three exact pairs using the designed eight-step custom-sigma profile at
1600×896, Turbo BF16 DiT, unchanged INT8 text encoder/BF16 VAE and the same
static offload policy. Sampling medians are 10.181 s Rust / 10.314 s Python;
pipeline medians are 19.394 / 18.552 s with substantial load-time variation.
All three selected DiT variants now have isolated trained T2I evidence; general
family/encoder/editing/LoRA/memory acceptance remains incomplete.

The [vision/editing integration](stage-2-qwen-image21-editing-integration.md)
adds the complete vision tower, multimodal conditioning, original Pillow C
binding, VAE reference inputs and reference-conditioned sampling. Five selected
320×224 functional pairs match exactly, followed by final shared-VAE two-reference
revalidation. Twenty-four focused regressions pass; this remains bounded
acceptance, with general editing/memory/precision and LoRA work still pending.

A subsequent [1MP portrait/reference test](stage-2-qwen-image21-portrait-1mpx.md)
uses the full newly generated 1024×1024 input and 40 editing steps. All six tensor
boundaries and PNG pixels match, with identical original CPU cache selection.
It adds the selected static pinned-host/two-buffer binding and 31 source cases;
25 focused checks pass. Both engines share the observed excessive contrast in
this image; general editing-quality acceptance remains open.

The [seed-only rerun](stage-2-qwen-image21-portrait-seed3061383787.md) at
seed 3061383787 removes the severe artifacts to the same extent in both exact
outputs. Tomas accepted it as a new [development baseline](stage-2-qwen-image21-editing-baseline.md)
on 2026-10-18 and requested preservation in Git. This is selected-workload
acceptance, not complete editing coverage or a demonstrated seed-handling bug.

The [LoRA continuation](stage-2-qwen-image21-lora.md) implements original adapter
math, model/text aliases, stacking, INT8 ConvRot requantization and selected
static patch/offload restoration. INT8 model, text and ordered-stack functional
pairs match exactly. Automatic BF16/Turbo pairs differ where patched weights
take different resident/lazy paths; replaying original placements gives exact
diagnostic pairs. Fresh automatic acceptance awaits confirmed free GPU memory.
No numerical criterion or automatic policy was relaxed. General LoRA lifecycle
and Stage 2 acceptance remain incomplete.

Tomas first requested a stop before generation; that gate was reached. He then
authorized: “Ja gör det, spännande, jämför tider och kvalitet mot comfyui
pythonversionen”. This permits the isolated first trained runs and matching
Python references. It changes no parity, quantization, autotuning or isolation
requirement. Ask only for a departure from the frozen ComfyUI implementation
or agreed design. Full Stage 2 and all three variants remain incomplete.

Continuation: [the complete resident DiT computation](stage-2-qwen-image21-dit.md)
is now implemented, including transformer blocks, FunControl, cache policy and
hook boundaries. The initial component results below remain historical evidence;
the continuation records current coverage and pending shared-runtime integration.

Tomas requested preservation of the existing foundation before this work.
Commit `059c875` preserves the implementation, integration tests, source locks,
decision ledger and the durable Stage 1 synthetic oracle archive. It contains
419 scoped files; unrelated existing Imagine/API/UI/native/profile edits were
excluded. No push was requested or performed. See
[the preservation report](stage-1-preservation.md) and
[fixture restoration instructions](../inference/fixtures/README.md).

## Required variants and acceptance

All three requested variants remain mandatory: **INT8 ConvRot, BF16 and Turbo**.
They share the frozen `comfy/ldm/qwen_image21/model.py` architecture. Turbo
checkpoint weights and sampling settings are a separate acceptance dimension;
a BF16 component fixture named Turbo does not establish Turbo model support.
No checkpoint filename or shape-only detection will silently select a new
sampler, sigma schedule, CFG value or precision policy.

The frozen default DiT has 64 input/output channels, 32 blocks, 32 attention
heads of dimension 128, context dimension 4096, MLP ratio 3 and RoPE axes
16/56/56. Logical-shape detection also preserves the original split/fused MLP
branches and detected dimensions. The family registry allows BF16/FP32
inference and specifies continuous-flow multiplier 1 and shift 0.69. Latent
normalization uses 64 channels and downscale 16.

The Qwen 2.1 text path uses **Qwen3-VL 8B**. Its last hidden state precedes the
final RMSNorm. Token processing removes the system turn, handles image spans
and emits image slots; the explicit `keep_vision` option must remain supported.
The old Qwen-Image architecture/encoder cannot substitute for this path.

The existing Imagine source includes an explicit eight-step Turbo profile,
CFG 1 and sigmas `[1, .978453, .954180, .926626, .895080, .845148, .704534,
.414568, 0]` (`src/comfy_engine.rs` and `workers/comfy/test_worker.py`). These
are existing source-level profile requests to preserve in later sampler
integration, not evidence of a universal ComfyUI Turbo default. Those unrelated
files were neither edited nor included in the preservation commit. Diffusion,
text encoder and VAE precision remain independent settings subject to the
original supported combinations. Turbo remains compatible with every original
applicable precision/quantization combination; the initial Turbo-labelled BF16
fixture is not a restriction on that requirement.

## Implemented initial slice

`inference/src/models/qwen_image21/` now contains:

- The Qwen 2.1 detection branch over logical shape descriptors, including
  prefixes, contiguous block counting and split/fused MLP selection.
- Zero-centered RMSNorm and resident text projection with GELU(tanh), using
  ordinary weights or real Kitchen quantized tensor storage.
- Text/reference/target sequence assembly, image slot handling, causal text
  masks, centered reference coordinates and source RoPE tensor layout.
- Exact prefix cache key construction; this is not the cache lifecycle itself.
- Shared modulation row splitting, training/inference normalization branches,
  Kitchen AdaLN and in-place gated residuals.
- Source timestep rounding, sinusoidal embeddings and latent process-in/out.
  The 128 mean/std constants preserve the original Python binary64 values;
  a source-hash-checked generator makes them reproducible.

These are component APIs inside the inference crate. There is no new product
API or replacement of the ordinary Dreamchat/ComfyUI service.

## Source-compatible decisions and findings

**S2-01 — retain the three variants.** The requested modes are recorded as
required in [the coverage matrix](stage-2-qwen-image21-coverage.json). A pending
component is not an omitted requirement. The Stage 2 exit gate remains the
first complete Qwen 2.1 path, including editing, LoRA, precision and memory
behavior from the approved design.

**S2-02 — preserve actual linear decomposition.** A first differential run
found an INT8 image-projection mismatch. The original `torch.nn.Linear` path
on the transposed, non-contiguous rank-3 image input executes
`t → expand → view → expand → view → bmm → _unsafe_view`. Kitchen's
unhandled `expand` dequantizes the logically transposed weight. A contiguous
input instead executes an INT8 `mm`, producing different output bytes.
The new component preserves the original decomposition and dequantization
point. It does not force contiguous inputs to obtain an INT8 fast path.
The oracle records both original dispatch traces for each device/variant.
This follows existing source behavior and requires no deviation approval.

The fixture supplies resident `torch.nn` operations to the original projection
constructor. Full ComfyUI `comfy.ops` casting, fused activations, loading and
patching are separate pending boundaries; these component results must not be
presented as their validation.

**S2-03 — retain exact statistics and policy.** Latent statistics are generated
as exact `f64::from_bits` constants. This avoids treating a trained decimal
coefficient as an approximate mathematical constant. No coefficient, numeric
tolerance, kernel-selection rule or autotuning policy was changed. Existing
approved Stage 1 exceptions remain scoped to their original cases; none was
extended to these new numerical tests.

## Verification scope

The checked-in oracle `inference/fixtures/qwen-image21-core.json` contains
**192 initialized synthetic cases**: 180 numerical/sequence component cases
and 12 logical-shape detector cases. Each case uses the unchanged original
Python class/function/method definitions extracted from the frozen source,
real pinned Torch and Kitchen, and operands created for this test only.
The generator verifies source hashes before capture. No model directory,
checkpoint weights, production prompts, histories, images or logs were read.

The numerical comparison checks exact bytes, shape, dtype, strides, device
and gradient flags. Post-call arguments are checked, and gated residuals must
retain input storage identity. FP32/BF16/FP16 arithmetic helpers, CPU/CUDA,
training/inference normalization branches, prefixes 0/3/7, empty/single/multiple
row splits, odd/even timestep widths, both default floating dtypes for latent
constants and both explicit FP64 RoPE policies are exercised. Sequence cases
cover no references, two slots, implicit slots and repeated slots for resident
BF16, Turbo-labelled BF16 and actual INT8 ConvRot weights on CPU/CUDA.
The sequence fixtures use deliberately small initialized matrices, not trained
Qwen weights. Detection covers 1/3/32 blocks, both prefixes and both MLP forms;
full detector family precedence and checkpoint parsing are not tested here.

Run the retained focused verification entrypoint using the restored Stage 1
build/dependency environment:

```sh
IMAGINE_PARITY_ROOT=/path/to/isolated-development-root \
IMAGINE_REFERENCE_SITE=/path/to/frozen/site-packages \
bash inference/tools/check_qwen_image21.sh
```

It checks formatting, Clippy, the focused differential tests and each new test
inside the existing Python-free, network-isolated ELF-only runtime. The
interpreter is a host-side development launcher only. Reference recapture is
separate and uses `launch_reference.sh` plus
`reference_qwen_image21_core.py --site ... --comfy ... --output ...` against
an explicitly verified source lock. An upstream upgrade must retain old
fixtures and identify source deltas before replacing expected values.

Formatting, Clippy with warnings denied, both focused Rust tests and both
individual Python-free isolation runs passed. The latent constant generator
verified all 128 exact values and the shell entrypoint passed syntax checking.
The ELF dependency inspection printed existing non-executable shared-library
mode warnings; dependency resolution and both isolated runs nevertheless
completed successfully.

Final execution evidence is stored in
[the checkpoint](stage-2-qwen-image21-checkpoint.json) and
[validation artifacts](stage-2-qwen-image21-evidence/). The earlier full Stage 1
155/146-test results are reused for its unchanged implementation; this report
does not claim a new full unfiltered run. Complete-model loading, generated
images, model VRAM/latency and production acceptance remain untested.

## Remaining Stage 2 work

The coverage matrix keeps the remaining continuation explicit: general loader
detection/conversion and error behavior; complete vision/editing/reference
conditioning; remaining sampler/scheduler/guider and CFG combinations; LoRA
merge/requantization/restoration; dynamic compiler/vbars, prefetch/offload,
host cache/pinned-memory and OOM/tiled policies; and trained end-to-end
differential acceptance for all three variants. Initial resident loading,
language/tokenizer, image VAE and selected sampling are now verified at the
component boundary. The first isolated INT8 image run is prepared and deliberately
unrun, as recorded in [the readiness report](stage-2-qwen-image21-first-image.md).
This is not full Stage 2/model-support acceptance.

No departure from ComfyUI or the agreed design was introduced. GPU and isolated
runtime checks required sandbox escalation; no escalation was rejected.
