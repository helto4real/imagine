# Stage 2 — Qwen sampler-buffer lifecycle and accounting

2026-10-18. Tomas authorized implementation and validation of the sampler
buffer's original lifecycle and memory accounting. Frozen reference:
ComfyUI `ba994879b88d0ebd9950b1e5c1e923d928f216ba`; source/dependency locks,
Kitchen and autotuning policy remain unchanged. Work is isolated development,
with no ordinary Dreamchat data or production service changes.

## Source contract and implementation

`ModelSamplingFlux.set_parameters` registers 10,000 Float32 sigmas for Qwen21,
40,000 bytes. `FluxSampling` already calculates the exact values; the new owner
retains that actual tensor through the model lifetime, rather than constructing
and dropping a separate temporary just for schedule selection. No numerical
algorithm, precision, seed, schedule or test profile changes.

Original `ModelPatcher.model_size()` caches `module_size(model)` including the
registered buffer. `_load_list()` inventories eligible leaf modules and does
not include the buffer-only sampling module. These are distinct source values:

- Initial full-load eligibility uses a **strict greater-than** comparison with
  the cached full model size, including the actual sampler tensor's bytes.
- Full `load()` moves the whole model, including the buffer, then sets the
  loaded counter to the cached full size.
- Partial load moves selected leaf weights; the sampler keeps its device.
  Its loaded counter comes from the leaf load calculation.
- Partial unload moves selected leaf weights and subtracts their bytes. It
  leaves the sampler on its previous device, even after every leaf is unloaded.
- `partially_load()` skips positive reload when the original low-VRAM flag is
  false and some bytes are loaded; equal/full-looking leaf residency alone
  must not trigger buffer migration or silently normalize the counter.
- Full `unpatch_model(offload_device, unpatch_weights=True)` restores backups,
  removes lazy functions and pins, moves weights and sampler to CPU, and resets
  the loaded/offload counters. The cached full size remains available for reuse.

The selected `StaticOffload` owner now preserves these branches and owns a
`FluxSampling`. Its production-shaped constructor and bounded test constructor
share the same implementation. `load_for*` retains the existing aggregate
memory inputs and budget policy, then enters the original low-VRAM/full-size
state logic. On a load failure it attempts the original detach-style cleanup.
No manual VRAM margin, competing workload or placement override is introduced.

The standalone driver uses the owned buffer for schedule selection and keeps
it alive through sampling. This also applies when Turbo supplies a separate
custom step-sigma table: that table does not replace the registered model
buffer. Full-residency and automatically managed paths both move and release
the buffer. The managed path records initial, sampling and unloaded states.
Explicit old placement-replay diagnostics now require sampler-device and
low-VRAM state; absent state is rejected rather than guessed.

One additional source requirement appeared in the bounded tests:
`Module._apply` preserves an ordinary Parameter's `requires_grad` flag when
moving it inside `no_grad`. A small module-movement helper preserves that flag;
backup `tensor.to` retains its distinct original semantics. The reference
fixtures were not relaxed. Two initial test failures (the missing movement flag,
and a test-side CPU copy dropping that flag) are retained separately.

## Validation

`reference_qwen_sampler_buffer.py` executes the hash-checked original sampler,
ModelPatcher and module-size methods with three initialized 16×256 weights.
There is no real model constructor. Six source sequences cover 28 transitions:
full/unload/reload, partial-to-equal-size reload, retained GPU buffer after all
leaves unload, and initial budget one byte below/equal/above full model size.

The Rust regression checks actual devices, cached/loaded/offload byte counts,
low-VRAM state, leaf residency, buffer bytes and values, unchanged parameter
bytes/gradient flags, and exact schedules at 4 and 40 steps. It passes both
normally and in the Python-free namespace. In particular, the tests retain
source behavior where a later leaf-only reload recomputes the counter without
the still-GPU sampler. No artificial always-add-40,000 adjustment is used.

The first trained reference attempt produced matching loaded counts and GPU
buffer placement at initial load and sampling, but its final observer was
missing: deleting the Python model before `release()` let ComfyUI's weak
registration disappear before the detach callback could run. The strict gate
correctly rejected the incomplete lifecycle evidence. The harness now invokes
original `release()` with the owner alive, records the original detach result,
then drops the owner and performs its normal final cleanup. Only the Python
oracle needs rerunning; Rust inference and its saved artifacts are unchanged.
No missing observation is filled from synthetic expectations.

## Trained validation

The corrected reference is `qwen-sampler-buffer-1600-20261018-01-python-02`,
compared with unchanged Rust artifacts from `qwen-sampler-buffer-1600-20261018-01-rust`.
This retains the existing 1600×1600, 40-step INT8 ConvRot portrait-editing
profile, seed 3061383787, Lenovo LoRA strength 1/CLIP 0, automatic memory
policy and original prefix-cache policy. No recorded placement replay is used.

The strict gate passes all six finite, byte-exact tensor boundaries and exact
PNG pixels. It additionally compares the actual registered sampler bytes and
metadata through all three phases:

| Phase | Rust loaded bytes | ComfyUI loaded bytes | Sampler device |
| --- | ---: | ---: | --- |
| Initial DiT load | 7,256,748,608 | 7,256,748,608 | CUDA |
| Sampling | 7,256,748,608 | 7,256,748,608 | CUDA |
| Full unload | 0 | 0 | CPU |

Both cache the same full model size, 7,256,748,608 bytes, including the actual
40,000-byte tensor. The sampler hash is unchanged in all six exports:
`77fdc7101e679fbcdf3aeffd40c0b9afb98fd4b49f14bf7d25e2379f807b4266`.
Leaf placements/low-VRAM state match. The earlier 40,000-byte full-load
accounting gap is closed for this trained path. The image's Float32 hash also
matches the preserved pre-change 1600 portrait exactly, independently of the
new reference pair.

Single fresh-process observations: Rust pipeline 112.639 s, sampling 94.698 s;
corrected Python pipeline 105.965 s, sampling 90.446 s. The two independent
Python attempts give differing timings (first sampling 95.059 s). These are
validation observations, not a controlled speed comparison or an optimization
claim. Pipeline timing includes unload/intermediate exports and excludes
launcher/build/preflight/final PNG export. New buffer exports and explicit
unload observation add harness work; do not treat prior pipeline differences
as an isolated effect of this change.

The trained case retains all DiT leaf weights on GPU and offloads the 4.910 GiB
prefix cache to CPU. Real trained partial-weight offload is not newly claimed;
the bounded source tests cover full/partial buffer transitions independently.
No fresh BF16/Turbo image generation is claimed in this slice; they use the same
buffer owner, and existing computation tests remain in the focused suite.
These source counters are not measurements of identical global allocator peaks.

## Checks and preservation

All **31 focused Qwen tests pass**, zero failures, in 171.94 seconds on the
final inference source. This includes existing BF16/INT8/Turbo component,
conditioning, sampler, LoRA and offload regressions plus the new lifecycle test.
One new exact test also passes separately in Python-free isolation; this is not
a new 31-test isolated-suite claim. No browser/product-runtime check was run.
Locked/offline build, all-target
Clippy with warnings denied, formatting, three Python AST checks and affected
shell syntax checks pass. New bounded normal and Python-free lifecycle tests
pass. All three complete model hashes match the previous control; original
source hashes and Stage 0 locks remain unchanged.

The evidence directory preserves both successful images, comparison, metadata,
all six sampler-buffer exports, namespace proof, earlier missing-observer
failure, failed/passing bounded-test logs and before-change source snapshots.
The old 1600 comparison and its source/evidence hashes remain historical and
are not overwritten. The new source fixture and normal/isolation entrypoint
retain the 28 original transitions for future ComfyUI upgrades.

Replay with a fresh prefix using the command in the preserved
[1600 report](stage-2-qwen-image21-lora-1600.md). The current harness now observes
original unload while the owner is alive and enforces its sampler state too.
GPU and namespace operations used approved sandbox escalation; none were
rejected. No commit or push was requested or made.

## Scope

This completes the selected Qwen sampler-buffer ownership/accounting slice
with the checks above. General model-clone ownership, arbitrary
object/bypass patches, other wrapper buffers/models and broader OOM/tiling
acceptance remain separate Stage 2 work. No departure from the frozen source
or agreed design has been adopted. This does not complete Stage 2.
