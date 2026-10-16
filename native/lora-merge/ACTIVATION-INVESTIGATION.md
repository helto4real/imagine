# First LoRA activation: measured cause, 2026-10-16

The main cost is the native loader's progress-polling sleep, invoked separately
for each LoRA operand and original scale. **576 small loads spend 5.836 seconds
inside the sleep call; 5.617 seconds occur after their workers have finished,
with no concurrent recorded GPU activity.** Rebuilding operand readers/catalogs
adds another 0.732 seconds. No inference optimization was changed.

## Scope and measurements

The workload is the same Qwen Image 2.1 INT8/ConvRot portrait, 1600×896, 40
ER-SDE/beta steps, CFG 1, seed 42, batch one, no references, only
`lenovo_qwen21.safetensors` at strength 1. Existing precision, flags, memory policy,
thread counts, sampler and conditioning policy remain unchanged.

Two separate native diagnostic builds add fixed-name monotonic timing intervals
only. The second measures existing sleeps, workers and joins without changing
their behavior. The parent executable and source files are unchanged. A separate
Comfy/AIO harness wraps methods in memory; installed Comfy/AIO source is unchanged.
New timers add no CUDA synchronization. Ordinary logs and generated images are
discarded; only authorized timing/kernel/copy counters were read.

| New diagnostic series | First after enabling LoRA | Subsequent warm mean | First-minus-warm cost |
| --- | ---: | ---: | ---: |
| Native | 27.466 s | 20.982 s | 6.484 s |
| ComfyUI headless/AIO | 20.460 s | 20.195 s | 0.265 s |

The new total first-request gap is **7.006 s**, versus **6.206 s** in the earlier
comparison. Warm engine time also differs by 0.787 s in this new series.
Subtracting each engine's own warm time leaves an **excess activation cost of
6.219 s**. These are distinct metrics; the historical 6.206-second gap is not
silently relabeled or treated as an invariant. The initial broad native probe
was somewhat slower overall, including its measured GPU activity, so this report
uses the detailed second probe for attribution.

## Native first-use breakdown

192 affected matrices each cause three separate loader calls: up operand,
down operand, and original scale. The full calls take **6.687 s**:

| Component within those loads | Wall time |
| --- | ---: |
| Existing polling sleep: 576 calls | 5.836 s |
| Rebuilding the 384 operand readers/catalogs | 0.732 s |
| Remaining loader/dispatch work outside those intervals | 0.118 s |

Workers' actual file-open/read/convert work totals **0.188 s**, overlapping the
sleep intervals; it must not be added to that table. Joining the workers takes
only **0.000535 s**. Of the 5.836 seconds asleep, **5.617 s** are after all workers
for that load have finished. Intersecting these idle tails with the complete
CUPTI kernel/copy timeline finds **zero overlapping GPU activity**.

The whole native merge scope takes **7.254 s**, including the loads above,
**0.500 s** in the complete CUDA helper (allocation, transfers, GEMM, arithmetic,
requantization, synchronization and cleanup), and about **0.067 s** in other
merge administration. These scopes are nested; do not add the 6.687-second load
total to the 7.254-second merge total. A small 0.0046-second merge bookkeeping
scope lies inside conditioning; merge outside conditioning is 7.250 s.

Conditioning independently takes **0.310 s** at first LoRA activation and about
0.00006 s on the following requests. It is a smaller contributor here, not the
source of the multi-second loader cost.

Warm native requests prepare 28 matrices and therefore make **84** small loads.
Their sleep time is 0.850 / 0.852 s, of which 0.839 / 0.833 s is after worker
completion with no GPU activity. Reader rebuilding takes about 0.117 s, and the
whole merge scope about 1.080 s. The same loader behavior therefore also explains
much of the remaining warm LoRA overhead.

## Source-grounded cause

In the unchanged candidate source:

- `src/model_manager.cpp:2087` calls `load_merge_operand` separately for up/down;
  the same merge loop separately reloads the original scale before CUDA merging.
- `src/model/adapter/lora.hpp:104–120` constructs a one-tensor load for each
  operand through `load_file_tensors`.
- `src/model_loader_files.cpp:310–335` constructs a fresh reader and rebuilds
  its catalog for each such call.
- `src/model_loader.cpp:1347–1365` polls worker progress. Line **1360** sleeps
  **10 ms for <=4 tensors**, otherwise 200 ms. Only the progress *printing* is
  guarded by `log_progress`; the polling/sleep still executes when it is false.
  A worker usually finishes during that sleep, but the caller does not resume
  until the sleep expires. The following `join()` would already ensure completion.

These source locations refer to `.imagine/lora-merge/source/`, whose hashes still
match the published whole-matrix candidate. The wait is not introduced by the
whole-matrix CUDA arithmetic; that integration exposes the general loader's
per-call cost by invoking it hundreds of times for small inputs.

## What differs in the tested Comfy path

`nodes.py:LoraLoader.load_lora` obtains the LoRA tensor dictionary together using
`comfy.utils.load_torch_file`. `comfy.sd.load_lora_for_models` maps those loaded
tensors into patches. `LoRAAdapter.calculate_weight` uses these operands directly
when `ModelPatcher.patch_weight_to_device` prepares weights; it does not invoke
our one-tensor loader/progress polling for each operand and scale.

In this new Comfy run, the first request spends **0.0348 s** attaching patches.
All **503** `patch_weight_to_device` calls together take **0.8048 s**, including
**192** actual LoRA calculations. This is host-observed method time, with queued
GPU work possible; it is not an isolated GPU-kernel total. GPU loading overall
is 1.356 s and includes patching, so those totals must not be added. Complete
sampling/conditioning boundaries retain the existing harness synchronization.

The tested AIO path also reuses text encoding when LoRA leaves CLIP patches
unchanged. See `services/lora_application.py:_clip_patch_signature` and
`services/qwen_encoding_cache.py:resource_context/CachedCLIP`. Its first-LoRA
conditioning time is 0.0196 s in this run. Native currently clears conditioning
cache on any LoRA configuration change at `src/pipeline/diffusion_engine.cpp:1834`.
This cache comparison is specific to the tested Comfy/AIO path, not a claim about
all Comfy entry points.

## Implication and limits

The first implementation target should be the unnecessary progress-polling wait
for small/non-progress loads, retaining worker joining, error propagation and
source validation. Reader/catalog reuse or batched operand loading is the next
separate source of overhead. Conditional conditioning reuse is a smaller issue
for this workload and must depend on actual text-encoder patch changes.

**None of these changes was implemented or benchmarked.** The measured idle
intervals establish wasted waiting in this execution; they are not a promise
that removing it will improve total latency by exactly the same amount. Other
LoRAs, file systems, shapes, concurrent workloads and GPU conditions were not
investigated. Timings vary between series; the phase attribution is more useful
than treating an earlier total gap as exact to the millisecond.

All eight requests in both native probes and in the Comfy probe succeeded.
Comfy executed 40 steps each time and passed the existing precision assertions.
Native inference-fusion counts remain unchanged, and all CUPTI checkpoints
report zero dropped records. Diagnostic scripts/builds and source hashes pass
focused checks. Approved sandbox escalation was used for GPU runs; there was
no automatic approval-review rejection. No commit or push was made.

## Evidence and reproduction

[activation-investigation.json](activation-investigation.json) contains the
numeric summaries, source/binary provenance and comparison arithmetic.

- Broad native phase probe: `runs/lora-cfg1-28avm4fq`.
- Detailed native wait/worker probe: `runs/lora-cfg1-o0f9f_hf`.
- Comfy method timing probe: `runs/comfy-lora-cfg1-mvdjgtgn`.

These paths are under `experiments/qwen-image21-sage/`. Each has authorized
`phases.tsv` / `activation-summary.json` counters beside its CUPTI trace.
No ordinary runtime logs or images are used as evidence.

The unchanged parent binary SHA-256 is
`2a9adef375ab60b01a82c6e28b9af1835b1f94d66dac2c877df0173facdc12de`.
Frozen v2 and the previous tiled control remain untouched.

`profiling/build-lora-activation-diagnostic.py --loader-waits` creates separate
instrumented objects and a separate executable/manifest under
`.imagine/lora-merge/activation-waits-diagnostic/`, linked against the unchanged
candidate libraries. Run it through the same native comparison harness, with
that manifest and `IMAGINE_LORA_PHASE_TRACE` pointing to a unique timing-only
file. `profiling/run-lora-activation-comfy.py` uses the same arguments as the
existing headless comparison. `profiling/summarize-lora-activation.py RUN PHASES`
reduces only the fixed-name intervals and numeric request boundaries. The build
alone does not execute a GPU workload.
