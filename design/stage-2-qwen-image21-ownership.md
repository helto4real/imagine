# Stage 2 — shared model ownership and static memory coordination

2026-10-18. Work authorized by Tomas for model ownership and shared memory
management. Reference remains ComfyUI
`ba994879b88d0ebd9950b1e5c1e923d928f216ba`. This is isolated development;
ordinary Dreamchat data and production services are outside the task.

## Source-compatible implementation decisions

**S2-OWNER-01 — separate patcher identity from physical storage.** The shared
coordinator has weak registrations, following `LoadedModel`. Retaining a Rust
handle retains that patcher; creating a patcher clone gives it a separate
identity, a strong parent and the same physical parameters, backups, pins and
registered sampler. Patch configurations copy their ordered patch lists and
retain their original tensor storage. Parent fallback follows the weak parent
when the registered child disappears. A dead patcher with a still-live storage
token remains distinguishable from destroyed model storage; the registry does
not acquire strong ownership to hide that state. Arbitrary hook/attachment/
object-patch clone semantics are not newly claimed.

**S2-OWNER-02 — retain the original coordinator policy.**
`inference/src/memory.rs` ports the Linux NORMAL_VRAM, smart-memory path:
order-preserving request/additional-model deduplication and reversal,
same-model/clone `detach(false)` before freeing, per-device aggregate requests,
the original 1.1 loading estimate, minimum/reserved memory, partial versus full
unload, re-registration and allocator-cache threshold. Free candidates sort by
negative offloaded bytes, logical reference retention, full size and list index.
Reference-retention fixtures supply explicit source counts; Rust exercises the
corresponding live handle counts. Allocator observations remain runtime inputs,
not constants substituted into the actual driver. The source's smart-memory
`device=None` behavior and strict comparisons are retained.

**S2-OWNER-03 — one static pin-registration budget.** Static model weights and
prefix caches now share actual registration accounting. Alias/double-pin,
same-pointer wrong-size unpin, cap rejection and successful release follow the
original rules. This closes the previous cache binding gap when another static
owner has pins. Original `models_for_pin_eviction` only selects dynamic models:
static registrations must not become invented eviction candidates. With only
static owners, unmet RAM/registration budgets therefore return false. General
dynamic pin tiers and graph RAM-release callbacks remain separate bindings.
The shared registry retains allocation handles until successful unregister;
normal owners explicitly unregister, and destruction provides error cleanup.

**S2-OWNER-04 — preserve forced-patch behavior.** A changed patch selection (or
forced patching) restores the shared original weights and adjusts the requested
extra-memory amount by the original loaded-byte delta. The initial forced
offloaded patch uses the original CPU Float32 compute path. A later regular
load patches the current weight again even when an original backup already
exists; it retains that original backup. This unintuitive source behavior is
preserved rather than replaced with an idempotent merge. A forced CPU patch's
empty lazy-function list survives an intermediate low-memory reload. Switching
patch sets with a still-negative adjusted budget enters original `load`, not
the unrelated negative-budget partial-unload shortcut.

**S2-OWNER-05 — wire the selected Qwen execution path.** The automatic static
DiT driver uses the shared manager for initial loading, sampling reload and
full release. Its prefix cache uses the same pin registry. Existing explicit
placement replay remains diagnostic only. The normal run uses live aggregate
CUDA/free-reserved-active inputs, and verifies that final DiT release leaves no
registrations or pins. Component arithmetic, seeds, profiles, precision,
Kitchen selection and autotuning are unchanged.

## Validation

All **35 focused Qwen regressions pass**, zero failures, in 167.95 seconds.
The four new ownership tests also pass separately in Python-free isolation.
This is a fresh four-test isolation result, not a fresh 35-test isolated-suite
claim. Formatting, locked/offline builds, all-target Clippy with warnings
rejected, both new Python AST checks, affected shell syntax and scoped
whitespace checks pass. A fresh no-Torch coordinator capture is byte-identical
to its saved fixture.

The pure coordinator fixture contains **18 sequences / 23 actions**. The
actual-tensor fixture contains **22 static clone/LoRA states** (BF16 and INT8
ConvRot), plus **seven two-model pressure/reload states**. The separate actual
CUDA-registration test covers shared cap exhaustion, duplicate aliases,
wrong-size unregister and final owner release. Initial syntax/format check
logs and the failed exact forced-patch comparison are retained. The latter
exposed missing source re-patching of an already CPU-patched weight; expected
numerical results were not relaxed.

The coordinator oracle executes hash-checked original `LoadedModel`,
`load_models_gpu`, `free_memory` and cleanup functions with bounded explicit
patcher/device inputs. It imports no Torch and allocates no weights. The tensor
oracle separately executes original static `ModelPatcher.__init__`, `clone`,
load/unload/patch methods and the original coordinator using initialized
16×256 weights. It cannot construct the trained model. Its two-model pressure
case uses small actual tensors; controlled memory observations avoid pressuring
real system RAM/VRAM. CPU/GPU devices, counters, low-memory flags, patch
functions, exact parameter bytes, sampler bytes and registry identities are
checked separately.

The standalone engine's Python-free namespace and trained model/image
comparisons remain separate evidence from these bounded tests.

## Fresh trained comparisons

Three fresh automatic pairs pass the existing strict comparator, without
placement replay. Both engines use the same registered profiles, local
checkpoints, seed, LoRA selection and original policy. All six Float32
boundaries are finite and byte-exact; Turbo additionally matches its explicit
step-sigma table. PNG pixels match exactly. Each pair applies the same 192 DiT
patch targets, with INT8 ConvRot Qwen3-VL and BF16 VAE. This is not all-BF16
encoder acceptance.

| DiT variant | Output / steps | Rust / Python pipeline, s | Rust / Python sampling, s |
| --- | --- | --- | --- |
| INT8 ConvRot + LoRA | 1600×1600 / 40 | 103.323 / 102.252 | 87.154 / 86.795 |
| BF16 + LoRA | 320×224 / 4 | 12.096 / 17.559 | 0.790 / 0.817 |
| Turbo BF16 + LoRA | 320×224 / 8 explicit sigmas | 11.221 / 8.252 | 1.369 / 1.390 |

These are one functional pair each, with variable first-load times. They are
not a performance ranking or evidence of an optimization. Pipeline includes
unloading/intermediate exports and excludes launcher/build/preflight/final PNG
export. The six INT8 boundary hashes also equal the preserved pre-change
1600 sampler-buffer control. No prior benchmark/reference was overwritten.

All three pairs have identical resident-module sets, lazy-state inventories,
loaded counters, cached model sizes and sampler bytes/device in initial,
sampling and final-unloaded observations:

| Variant | Loaded bytes, initial and sampling (both) | Sampler during sampling | Final loaded bytes (both) |
| --- | ---: | --- | ---: |
| INT8 ConvRot | 7,256,748,608 | CUDA | 0 |
| BF16 | 12,653,191,168 | CPU | 0 |
| Turbo BF16 | 12,653,191,168 | CPU | 0 |

INT8 retains all 330 DiT leaf modules on GPU and uses the original
5,271,715,840-byte pinned CPU prefix cache with two staging buffers. BF16 and
Turbo each retain 283 modules and offload 47, with 1,577,058,304 actual pinned
weight bytes; their 40,000-byte registered sampler remains on CPU. Their
163,053,568-byte prefix cache stays on GPU. These are actual trained partial
weight-offload checks at the small geometry; they are not new large BF16/Turbo
editing-quality runs. Older BF16 counter differences are retained as historical
evidence; these fresh matching observations do not retroactively prove every
previous allocator/counter state correct.

The Rust manager records one registered model at initial load/sampling and
zero after full release in all three pairs. Total pin registration returns to
zero as well. Loaded counters, actual pins, physical module residency and
allocator peaks remain distinct; no identical peak-VRAM claim is made.

## Preservation and replay

The [evidence directory](stage-2-qwen-image21-ownership-evidence/) retains all
three strict comparisons, both images and sampler exports, run metadata,
Python-free namespace proofs, test logs, source snapshots/diffs and replay
scripts. Full trained boundary tensors remain in the explicitly named isolated
runtime roots; their hashes are retained in the comparisons. The checkpoint
hashes source inputs, binaries and saved evidence. Frozen Stage 0 locks and
reference sources are unchanged. Full checkpoint hashes are recorded separately
in `verified-model-hashes.json`, including continuity with the previous INT8
control. No ordinary Dreamchat data or service was read or changed.

Use the normal `inference/tools/check_qwen_image21.sh` entrypoint to retain the
new tests for future ComfyUI updates. Explicit trained replay uses the evidence
`replay-trained.sh FRESH_PREFIX` script: large INT8 portrait first, then the
existing small BF16/Turbo profiles. It never replaces the saved earlier runs.
GPU/namespace validation required approved sandbox escalation; no escalation
was rejected. No commit or push was requested or made.

## Remaining scope

This is the shared coordinator and selected static Qwen binding, not full Stage
2 acceptance. Dynamic ModelPatcher/vbars/compiler ownership and its pin/RAM
eviction callbacks, arbitrary hook/object/bypass/shape-changing patch clones,
and general encoder/VAE shared-manager binding and OOM/tiling remain open.
The current trained harness still uses component-scoped text/VAE lifetimes;
small multi-owner tensor tests do not establish simultaneous trained DiT,
encoder and VAE residency. Broader precision, sampling and end-to-end model
acceptance retain their existing gates. No policy exception, relaxed numerical
criterion or changed autotuning is authorized or introduced by this work.
