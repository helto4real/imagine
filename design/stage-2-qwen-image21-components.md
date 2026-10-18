# Stage 2 — shared encoder/VAE ownership and image VAE policy

Historical continuation snapshot, retained with the completed
[Stage 2 baseline](stage-2-qwen-image21-ready.md). Counts and incomplete-status
wording below refer to this earlier checkpoint.

Work in progress, 2026-10-18. This report does **not** declare Stage 2 complete.
The preservation commit requested before this continuation is `ac8594c`
(`Preserve Qwen LoRA and shared-memory parity baseline`). No push was made.
Unrelated product/UI/native changes were excluded from that commit.

## Requirements and decisions

The approved design and frozen ComfyUI revision
`ba994879b88d0ebd9950b1e5c1e923d928f216ba` remain the contract. The user authorized
finishing Stage 2, including the three requested Qwen variants; only departures
from ComfyUI or the design require new approval. The existing authorization for
isolated development artifacts and selected local models applies. No ordinary
Dreamchat runtime data, services or GPU process list was accessed.

- **S2-COMP-01:** reuse the physical static owner, patch-selection identities,
  pin budget and `memory::Manager` for the encoder and VAE. Constructor buffers
  and parameters outside `_load_list` move with the complete module, without
  being invented leaf weights. This generalizes the existing DiT implementation.
- **S2-COMP-02:** inventory the complete original text wrapper, including its
  FP32 `logit_scale` parameter (four bytes). Its rotary `inv_freq` is a
  nonpersistent FP32 buffer: moved by `Module.to`, excluded from `module_size`.
  Preserve parameter versus buffer `requires_grad` behavior on device moves.
- **S2-COMP-03:** preserve `CLIP.__init__`'s
  `set_model_compute_dtype(torch.float32)` and resulting forced casts. This is
  independent of BF16 or INT8 checkpoint storage. It changes original load-list
  workspace estimates and must not be replaced by the DiT's BF16 estimate.
- **S2-COMP-04:** preserve the image VAE's outer policy: original memory estimates,
  batch splitting, CPU output assembly, input/output transforms, and the three
  overlapping tiling passes in their different encode/decode orders. Discard
  temporaries from the failed attempt before clearing the cache and retrying.
  Ordinary errors and cancellation must escape rather than trigger tiling.
- **S2-COMP-05:** retain native c10 allocation/accelerator exception categories
  and accelerator error codes at both FFI boundaries. The original `torch-sys`
  string-only catch loses information required by ComfyUI's `is_oom`. Vendor
  the exact already-locked `torch-sys` 0.26.0 and apply a recorded binding-only
  patch. Tensor operations, original Kitchen bodies, CUDA flags and autotuning
  are unchanged. Fault injection is a dev-dependency feature and is absent from
  the normal inference executable. No Python runtime is added.

These implement source behavior; none is an approved numerical or policy
exception. No new departure has been requested or accepted.

## Source and synthetic validation

The source oracles hash the frozen files before executing original definitions.
No full trained model is constructed by an ordinary Cargo regression test.

- `reference_qwen_vae_policy.py`: 32 original tiler cases across CPU/CUDA,
  Float32/BF16 inputs and Float32/Float64 default allocation dtypes; 32 original
  image VAE lifecycle cases, including regular batching, load/call failures,
  partially completed output, three-pass retry and non-OOM error propagation.
- `reference_qwen_components.py`: four original cast-dtype policies and twenty
  small physical load/unload transitions, including a nonpersistent buffer and
  a persistent nonleaf parameter.
- `reference_qwen_oom.py`: six original `is_oom`/`raise_non_oom` cases with explicit
  exception categories and accelerator codes. The native tests construct the
  corresponding c10 exceptions through the actual C ABI and boxed ATen path.
  They do not request an oversized allocation or exhaust RAM/VRAM.
- Constructor/module inventories are generated on `meta`, with controlled empty
  model directories. The full BF16 text wrapper has 575 load-list modules and
  `module_size=17,534,247,396`; the selected 96/144 VAE has 150 modules and
  `module_size=675,480,808`. Quantized storage sizes are recomputed from actual
  prepared state, rather than these unquantized inventory sizes.

The first focused regression run passed **38 tests in 163.87 seconds**. Subsequent
focused component/boxed-exception checks passed **2 tests in 12.99 seconds**.
The final focused suite passed **40 tests in 182.88 seconds**; all **40 tests**
then passed individually in Python-free isolation. Formatting, offline locked
build/Clippy, vendor reverse-patch restoration and the four unchanged Stage 0
locks are verified. The normal executable contains no development fault-injection
operator. See [the checkpoint](stage-2-qwen-image21-components-checkpoint.json)
and [retained evidence](stage-2-qwen-image21-components-evidence/).

## Trained validation

Three sequential trained comparisons passed with automatic memory policy and
no placement replay. All compared Float32 boundary tensors are finite and byte
identical; decoded PNG pixels are identical. All five observed component phases
per case match every recorded field: model size, loaded bytes, buffer bytes,
resident modules, lazy-patch state, forced-cast state, pin counters, buffer
placement and live coordinator registrations.

| Existing profile | Rust pipeline / sampling, s | ComfyUI pipeline / sampling, s |
| --- | ---: | ---: |
| INT8 ConvRot + LoRA, portrait editing, 1600×1600, 40 steps | 105.027 / 86.643 | 101.616 / 87.015 |
| BF16 + LoRA, editing, 320×224, four steps | 13.442 / 0.761 | 17.237 / 0.825 |
| Turbo BF16 + LoRA, editing, 320×224, eight prescribed sigmas | 27.859 / 1.355 | 8.940 / 1.406 |

These are individual functional comparisons, not a steady-state performance
benchmark. Pipeline timings include component unloads and development exports;
I/O/initialization costs vary. The INT8 metadata-export correction reran Rust
under `qwen-components-20261018-02-int8-rust` against the already successful,
unchanged Python control `qwen-components-20261018-01-int8-python`. The BF16 and
Turbo pairs use `qwen-components-20261018-02-small-{bf16,turbo}-{rust,python}`.

The initial INT8 pair already had six exact numerical boundaries and identical
PNG pixels, but correctly **failed** the newly strict component check: the Rust
metadata renderer reported 430 rather than 575 forced modules, excluding plain
norm modules on which the source also sets `comfy_force_cast_weights`. The
renderer was corrected; computation, placement and acceptance criteria were
unchanged. The initial failure and corrected comparison are retained separately.

All three controls retain the INT8 Qwen3-VL encoder and BF16 VAE. The real INT8
text wrapper size is **9,350,657,364 bytes**, including `logit_scale`; all 575
modules are resident in these runs. The selected VAE size is **675,480,808 bytes**,
with all 150 modules resident. Full unload returns registrations and weight
pin counters to zero. These comparisons are not an all-BF16 encoder or complete
Stage 2 acceptance claim.

## Remaining work

Follow the existing remaining-work list in `stage-2-qwen-image21.md`; do not
interpret this continuation as removing loader/conversion, precision, editing,
guidance/sampler, patch/hook, dynamic-memory or broader lifecycle requirements.
The general engine/product interface remains separate from this explicit
isolated comparison driver. Existing app/API behavior has not been changed.
