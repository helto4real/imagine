# Stage 2 — first-image preparation

Historical readiness checkpoint, 2026-10-18: **ready; stopped before generation**.
Tomas subsequently authorized the first image and comparison with Python.
The [first trained comparison](stage-2-qwen-image21-first-comparison.md)
supersedes this report's stopping boundary and unrun-generation statements.
The component/load-only results below remain the evidence for the earlier gate.

## Current instructions and selected files

Tomas requested continuation until the isolated new engine is ready for its
first image generation, then a stop **before** that generation. He selected
INT8 ConvRot first and supplied the model root
`/mnt/media/gemensam/AI/ComfyUI/models`. The initially suggested project-local
`models` directory does not exist; no directory was silently substituted.

The selected files are the profile's diffusion, Qwen3-VL and VAE checkpoints:

- `diffusion_models/qwen-image2.1/qwen_image_2.1_int8_convrot.safetensors`
- `text_encoders/qwen-image2.1/qwen3vl_8b_int8_convrot.safetensors`
- `vae/qwen-image2.1/qwen_image_2.1_vae_bf16.safetensors`

Their headers match all 265 DiT, 750 text/vision/head and 238 VAE parameter
shapes from the corresponding original constructors after the source text-key
mapping. The VAE widths are **96/144**, detected from the checkpoint rather
than assumed from Wan's generic constructor defaults. Its latent channels
remain 64, output channels RGBA 4, spatial factor 16, patch size 1 and temporal
kernel 1. The text file's `model.visual.*` and `lm_head.*` prefixes are mapped
exactly as `sd.load_text_encoder_state_dicts` does.

## Explicit initial test configuration

The checked-in [profile](../inference/profiles/qwen21-first-int8.json) is an
isolated development test selection, not engine defaults, a replacement for
the frozen native 1600×896 baseline, or a restriction on the parity target.
It requests 512×512, batch 1, seed 17, four Euler steps, simple schedule,
source Qwen21 FLUX shift **0.69**, CFG 1, BF16 compute, PyTorch attention,
and no LoRA/references. The fixed prompt is synthetic source material.

Source-supported settings are selected explicitly: static patchers, disabled
Comfy compiler/allocation graphs, disabled asynchronous offload, full component
residency and explicit unload/cache release between text, DiT and VAE phases.
The matching reference driver must request the same settings and sequencing.
These settings do not change Kitchen/Triton autotuning or kernel numerics.
The isolated process uses LibTorch's native caching allocator with no inherited
allocator override. A reference started through ComfyUI's main launcher must
also select its native allocator rather than the optional cudaMallocAsync mode.

The concrete static execution service has no dynamic vbars or allocation
graphs because this profile explicitly disables them. Same-device GPU cache
selection uses the source free-memory formula and original strict 4×/2×
thresholds. The source default cache dtype remains unchanged. Host-cache
pin-budget/registration/staging, CPU cgroup memory and dynamic services remain
explicit development gaps; they are not represented by invented successes or
different thresholds. A workload reaching such a gap fails explicitly.

The full original text checkpoint is prepared, including visual and LM-head
parameters unused by T2I, plus the nonpersistent vision rotary buffer. The
initial driver does not discard these weights to reduce residency.

The frozen `model_management.py` startup also enables reduced precision in
math-SDPA on this Torch version. The driver binds that setting explicitly,
enables the three source PyTorch SDP bodies, and keeps cuDNN benchmark disabled
for the empty `--fast` feature set. This is source compatibility, not a newly
chosen numerical approximation. Existing callers' settings are not changed
by the general SDPA wrapper; only the explicit standalone startup applies it.

## Implemented and differentially checked components

All fixtures below use initialized synthetic data. Exact tensor bytes, types,
strides, metadata, policies and applicable errors/aliases are compared; hashes
stand for equality of complete tensor bytes, not tolerances.

| Boundary | Retained source evidence |
| --- | --- |
| AIMDO safetensors | 50 dtype/order/offset/empty/error/storage-owner cases; mapped views survive loader/ELF drops |
| Quantized module loading | 76 source Linear/module cases covering INT8 ConvRot, FP8, MXFP8, NVFP4 and grouped formats/policy |
| Distinct embedding loading | 32 original loads: FP8/INT8 scales, default scale, top-level ConvRot, aliasing and original freeze order |
| Full resident parameter preparation | 10 whole tiny-model original state export/load/move cases for DiT, language and VAE on CPU/CUDA |
| Tokenizer | 547 BPE/template/padding/image-placeholder cases; separate exhaustive scalar NFC/contextual-regex checks |
| Shared SDPA | 576 small/large, CPU/CUDA, F32/BF16, GQA/mask/backend-state cases with both math-reduction settings |
| Qwen3-VL language conditioning | 75 complete/core forwards, including 36 layers, INT8 ConvRot, BF16, scaled FP8, weighting, GQA and pre-final normalization |
| Image VAE | 18 original encode/decode calls plus five original row-strip cases; CPU/CUDA F32/BF16/FP16 and RGBA processing |
| Sampling | 60 original schedules, 16 CONST/CFG cases, eight CPU-noise/RNG states, 12 complete Euler histories and four full-area CFG1 aggregations |
| Concrete resident services | Reuses original DiT/cache outputs with source-selected preferences, real CUDA memory counters, cancellation and live-storage-preserving cache release |

The VAE preserves 3D convolution for singleton time, `nearest-exact`, temporal
shortcut padding/duplication, `F.normalize` RMS behavior and its explicit +0,
the fixed `2**24` strip threshold, halos, and in-place additive shortcuts.
There is no Conv2d substitute for the source Conv3d branch.

Sampling keeps Torch generators and exact CPU noise/state, Float32 sigma
construction, terminal zero, solver/callback ordering and churn branches.
The source CFG=1 shortcut still performs `_calc_cond_batch` accumulation and
the ordinary CFG arithmetic. Those operations are retained, including signed
zeros and the `1e-37` count initializer. They are not optimized to a direct
return of the model output.

## Launcher and stop boundary

[The standalone Rust driver](../inference/src/bin/qwen21-first-image.rs)
defaults to preflight. Only its explicit `--run` branch reaches a trained
forward. `load-check` maps and prepares the chosen weights, loads each complete
component onto CUDA, constructs the relevant typed model, releases it, and
hashes the source files. It performs no text prefill, DiT step or VAE decode.

[The launcher](../inference/tools/launch_qwen21.sh) uses the existing audited
ELF/compiler allowlist and an isolated network/PID/mount namespace. It adds
only the three explicitly selected checkpoint files read-only and its own
development output directory. No Python interpreter, ordinary ComfyUI tree,
Dreamchat runtime or existing media is exposed inside the process. Python is
a host-side development launcher/reference dependency only.

For the authorized readiness check:

```sh
export IMAGINE_PARITY_ROOT=/tmp/imagine-rust-comfy-parity-20261017
export IMAGINE_REFERENCE_SITE=/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages
bash inference/tools/launch_qwen21.sh /mnt/media/gemensam/AI/ComfyUI/models load-check
```

The first-image command changes only the final word to `run`. The earlier
preparation task stopped here; its successor has explicit authorization to run.
It writes RGBA Float32 tensor
evidence and a four-channel PNG into the isolated new `first-image` directory.
Existing outputs are not overwritten. PNG compression identity with graph/UI
save nodes is outside inference; the underlying numerical pixel tensor remains
the comparison artifact.

## Remaining parity work

Full Stage 2 still requires trained end-to-end comparisons for INT8 ConvRot,
BF16 and Turbo; complete vision/editing/reference processing; all remaining
samplers/schedulers/guiders and conditioning combinations; LoRA/adapter
patch/requantize/restore; general checkpoint detection/conversion and malformed
input behavior; dynamic memory/vbars/compiler/prefetch, host cache transfer and
OOM/tiled recovery. None has been removed from the design. Turbo does not
receive guessed sampler settings from its filename.

No new departure from the frozen source/design has been accepted or requested.
The initial preservation commit remains `059c875`; this continuation remains
uncommitted. No push, production integration, service interruption, GPU process
inventory or Computer Use occurred.

## Final readiness evidence

The [checkpoint](stage-2-qwen-image21-first-image-checkpoint.json) and
[retained evidence](stage-2-qwen-image21-first-image-evidence/) record:

- Formatting and all-target Clippy with warnings denied passed.
- All **15 focused tests passed together**, then all **15 passed separately
  without a Python interpreter**, with network isolation. These are focused
  Stage 2 checks, not a rerun of all historical Stage 1 tests.
- The three selected real components passed a separate Python-free **load-only
  CUDA preflight**: 751 text parameters/buffers, 265 DiT parameters and 238 VAE
  parameters. No trained forward was invoked. The first-image directory does
  not exist.
- All **359** frozen implementation/fixture/resource inputs and **48** original
  source inputs remained unchanged through validation. All four Stage 0 locks
  and all **32 original CUDA object hashes** still match the Stage 1 freeze.
- Fixtures retain **1,970 distinct reference records**, including the earlier
  481 core/DiT/control/cache/interruption records. The concrete resident service
  additionally replays 102 original forwards, including 66 cache forwards;
  those reused results are not counted twice. Exhaustive Unicode aggregates
  separately cover 1,112,064 scalar NFC and 1,112,064 contextual-regex checks.

| Selected asset | Bytes | SHA-256 |
| --- | ---: | --- |
| DiT INT8 ConvRot | 7,256,783,064 | `cb74113cb03faecd79611b01fd7fd642f0aa60d6f0b95086abee214d75eaa57d` |
| Qwen3-VL INT8 ConvRot | 9,350,798,360 | `8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f` |
| Qwen21 VAE BF16 | 675,509,688 | `bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9` |

The preflight's memory arrays are `[device_free, device_total, active_bytes,
reserved_bytes]`. They describe loading/residency, **not** measured inference
peak memory, latency, image quality or trained numerical parity. A first full
run is still required to validate that workload.

Model assets were read only under Tomas's explicitly supplied root. No ordinary
Dreamchat/ComfyUI runtime data or GPU process inventory was read. GPU and mount
isolation checks required approved sandbox escalation; no escalation was
rejected. No first image, trained text prefill, trained DiT step or trained VAE
decode has been run in either candidate or reference.

For future ComfyUI upgrades, retain the fixtures, source schemas, generators,
tests and checkpoint manifests. Regenerate oracles against a newly explicit
source lock and keep old evidence as the comparison control. Rerun load-only
preflight if any model asset changes; matching filenames do not extend these
hashes/results to new weights. `tools/stage2_snapshot.py --verify` checks the
retained `verified-inputs.json`; it does not advance a reference or run a model.

## Recorded compatibility decisions

- **S2-04:** follow the user's first-image stop gate and selected INT8 files.
  The explicit smoke profile is a source-supported test configuration; it does
  not replace engine defaults or waive later GPU memory/compiler obligations.
- **S2-05:** keep the separate source embedding loader, its scale defaults and
  top-level ConvRot handling. Do not reuse Linear's different metadata rules.
- **S2-06:** retain all unused visual/head parameters and the original vision
  buffer in resident T2I preparation. Source constructor shapes are generated
  on meta tensors and frozen, rather than deduced from filenames.
- **S2-07:** bind Comfy's math-SDPA startup flag. Keep caller state separate
  from the shared wrapper and test both settings. No autotuning policy changes.
- **S2-08:** the first cold standalone attempt queried CUDA allocator counters
  before LibTorch's lazy CUDA initialization. Device-property queries alone
  do not initialize the allocator. The C ABI now performs the equivalent of
  Python's `_lazy_init` before memory statistics/cache release; the separately
  isolated runtime test includes this cold-start boundary. This is a binding
  correction, not a different memory threshold or source algorithm.
