# Stage 2 — Qwen Image 2.1 vertical-path acceptance

Status, 2026-10-18: **the selected Stage 2 Qwen vertical path is complete** under
the exit condition in master-design Section 15. INT8 ConvRot, BF16 and Turbo can
run through conditioning, editing/LoRA, sampling and VAE output in Rust. Overall
ComfyUI coverage is partial; this is not production-release acceptance.

This report preserves the accepted old-reference checkpoint. The subsequent
[upgraded-reference follow-up](stage-2-upgraded-followup.md) completes ordinary
BF16 server selection and fresh strict BF16/Turbo automatic LoRA comparisons.
Its new results do not relabel the old-reference evidence below.

## Contract and preservation

The requested initial commit is **`ac8594c`**, `Preserve Qwen LoRA and
shared-memory parity baseline`. The completed implementation, tests, source
fixtures, dependency sources, evidence and documentation are preserved by the
commit containing this report, `Preserve completed Qwen Stage 2 parity baseline`.
No push or product-backend replacement was performed. Unrelated working-tree
changes are outside this baseline.

The reference remains ComfyUI
`ba994879b88d0ebd9950b1e5c1e923d928f216ba`, the frozen local packages and the
four unchanged Stage 0 locks. Runtime inference uses Rust, LibTorch and the
standalone Comfy Kitchen build. Reference Python is a development oracle.
The isolated inference namespace has no Python interpreter or usable ComfyUI
installation. No stable-diffusion.cpp execution is involved.

## Selected Qwen capabilities

| Capability | Implementation and evidence |
| --- | --- |
| INT8 ConvRot, BF16 and Turbo DiT | Complete Qwen DiT, original quantized layouts and Kitchen dispatch; trained T2I and editing comparisons retained in the variant reports |
| Text and image conditioning | Original tokenizer/Unicode, Qwen3-VL language and vision towers, multimodal positions/deepstack, slots, filtering and keep-vision; INT8 and actual BF16 encoder storage |
| Reference editing | Original image resize/alpha/normalization and VAE encoding; no-reference-VAE, one-reference, two-reference and accepted portrait cases |
| LoRA | Original selected model/text mappings, ordered stacks, patch math, INT8 ConvRot requantization, restoration and static shared clones |
| Sampling | Original selected Euler/simple and prescribed Turbo sigmas; full-area CFG and ER-SDE/beta; exact noise/generator and callback histories |
| Memory and ownership | Original static synchronous load/offload, common weak model registry, physical owners and pins for DiT/text/VAE, sampler buffer, prefix cache and host staging |
| Image VAE policy | Regular batch sizing, typed OOM handling and retry, original three-aspect tiling, explicit tiled encode/decode, original inference context and dtype order |
| Precision | Independent component construction/freezing; complete small FP16/FP32/BF16 controls; trained requested DiT variants with INT8 and BF16 text storage and BF16 VAE |
| Failure and lifetime | Bounded source controls for cancellation, callbacks, load/copy/patch failures, weak-owner release, partial eviction, OOM classification and cleanup |

The reusable execution boundary is
`models::qwen_image21::QwenDenoiser`, together with the existing component,
checkpoint, memory, patch, preprocessing and VAE policy APIs. Development
profile allowlists and artifact exports stay in `qwen21-first-image` and its
launch tooling. Native model/kernel safety preconditions remain explicit.
Run commands are in [the inference README](../inference/README.md).

## Validation

The unfiltered pre-context primitive continuation passed **191 tests in
1265.01 s**, with zero failures. The latest focused Qwen source passed **46
tests in 196.84 s**, with zero failures. All **46** then passed individually in
Python-free isolation. Two library unit tests, formatting, locked offline
all-target Clippy, the normal executable build, vendor reverse-patch restoration
and exact SciPy/Boost source verification also pass.

Twelve continuation comparisons are preserved in the
[final machine-readable checkpoint](stage-2-qwen-image21-precision-sampling-checkpoint.json),
with 472 evidence files and 628 implementation/fixture hashes. Earlier trained
variant/editing/LoRA/T2I checkpoints remain retained independently. The latest
five end-to-end controls use the final runtime source; earlier unchanged-path
results are reused with the affected-component and final isolation checks.

| Final trained control | Exact tensor boundaries, plus PNG pixels | Rust / original pipeline, s |
| --- | ---: | ---: |
| Explicit tiled encode/decode, CFG/ER-SDE/beta + LoRA | 7 | 24.294 / 5.474 |
| 1600×1600 portrait, 40 Euler steps + LoRA, automatic offload | 6 | 107.517 / 106.582 |
| Two reference images | 8 | 7.155 / 7.225 |
| Keep-vision without reference VAE | 5 | 7.735 / 8.518 |
| Text-to-image without references | 3 | 4.776 / 4.023 |

All compared tensors are finite and byte-identical. The tiled and large
automatic-memory pairs also match all five measured component states. The other
three retain their existing full-residency profiles; no unrecorded physical
state equality is inferred. The large portrait's image hash equals the previous
accepted shared-component control. Sampling is 86.732 s Rust / 86.883 s original.
The image's accepted appearance is preserved; this is not a new general quality
study. Cold loading/setup accounts for much of the single small-run variation.

Ordinary regression tests use bounded synthetic tensors. Full model inventories
use `meta` constructors; no ordinary test constructs full trained weights or
attempts to exhaust RAM. Trained model runs are explicit, sequential, isolated
development commands. The actual BF16 encoder launcher checks aggregate host
RAM before loading its checkpoint. Only selected local models and own test
artifacts are read. No ordinary Dreamchat history/media, GPU process list or
other process control was used. GPU/namespace validation required sandbox
escalation; no escalation was rejected.

The [precision/sampling report](stage-2-qwen-image21-precision-sampling.md)
retains the new cases, decisions and strict comparison failures. In particular,
two BF16-encoder pairs have exact numerical results but different physical
reload placements after differing measured free-memory inputs. Their strict
live-state failures remain recorded. Allocation-free execution of the original
575-module policy reproduces each engine's actual states on its own recorded
inputs. This verifies the policy; it does not establish equal allocator
histories or physical peak VRAM. The underlying small free-memory difference
has not been fully attributed. No threshold or live placement is overridden.

## Decisions and retained limits

All changes in this continuation implement frozen source behavior. No numerical,
algorithmic, quantization or autotuning exception was introduced. Source-context
guards, typed native exceptions, default dtype, parameter freezing, casting,
load ordering and solver edge cases are recorded in the component and
precision/sampling reports. Earlier approved exceptions retain their original
scope in the decision ledger. Performance is reported separately from parity;
single functional timings are not steady-state benchmarks.

The master design's Section 15 defines Stage 2 as the **first complete selected
Qwen vertical path**, with partial overall coverage. Earlier continuation lists
mixed selected-path blockers with unfinished general contracts. The following
requirements remain visible and required for the full project; they are not
removed or declared supported by this milestone:

| Retained requirement | Master delivery milestone |
| --- | --- |
| Complete loader detection/conversion/error, conditioning, sampler/scheduler/guider registry and general patching contracts | Stage 3 |
| Arbitrary LoRA config/bypass/shape-changing, hook/object/attachment clone semantics and their interactions | Stage 3 shared contracts; complete model matrix in Stage 4 |
| Dynamic compiler/vbars ownership, dynamic pin/RAM eviction and graph RAM callbacks | Stage 3 loading/memory contracts and Stage 5 lifecycle acceptance |
| Broader variants/modalities, trained reuse/pressure/failure combinations and the complete required hardware/model matrix | Stages 4–5 |
| Imagine API expressiveness, GPU coordinator/worker interruption and shutdown, packaging and production integration | Stage 5 |

The selected test profiles explicitly use the original static patcher,
synchronous offload and PyTorch attention, with the Comfy compiler and Kitchen
Triton startup option disabled. The foundation separately validates Kitchen's
registered backends and standalone compiler. These profile choices are not
promoted to new native engine defaults or a claim that dynamic/default ComfyUI
execution has full model acceptance. No public API, UI, service, default
profile or frozen native comparison executable is changed by this milestone.

Complete ComfyUI parity and production-release acceptance remain open under
Sections 14–15 of the [master design](rust-comfy-parity-engine.md).

No Stage 2 selected comparison, current regression or isolation gate remains
pending in this checkpoint. No new deviation approval is required. Future
reference upgrades must update provenance deliberately and rerun the saved
source oracles, synthetic/isolation checks and relevant trained controls.
The preservation inventory excludes generated Python bytecode caches; this
corrects the earlier 636-entry inventory's eight cache entries without changing
inference, fixtures, numerical gates or any retained test result.
The staged whitespace check reports original whitespace in the hash-verified
vendored SciPy/Boost/torch-sys files and retained verbatim test logs. Owned
source/documentation files pass; the original evidence and upstream bytes are
preserved rather than reformatted.
