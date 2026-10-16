# Qwen reference cache

Accepted reference baseline: **2026-10-16**, for the documented Qwen Image 2.1
workloads. See [retained optimization material](REPRODUCIBILITY.md) for local
controls, test inputs and cleanup policy.

The normal Rust API enables hybrid placement and asynchronous prefetch for
Qwen Image 2.1 inference passes that actually contain references. This covers
ordinary images with one or several references and character-sheet passes.
See [corrected-budget acceptance](ACCEPTANCE-REUSE.md). Text-only passes use
frozen v2; Turbo and other model families retain their own selection.

`IMAGINE_QWEN_REFERENCE_CACHE=off` selects v2. Unset or `auto` enables the
reference profile. The local executable at `native/reference-cache/bin/sd-server`
must match `reuse-budget-source-lock.json`; an absent or mismatched binary
falls back to v2. An inference failure is not automatically retried. Profile
changes stop and wait for the previous native process before loading the next.
The release server has been rebuilt; an already running server needs a restart.

Native engine defaults remain GPU cache and prefetch off. Direct native callers
opt in with `--model-args` containing
`qwen_image_2_1_prefix_cache_device=hybrid,qwen_image_2_1_prefix_cache_prefetch=true`.
The `cpu` cache mode also remains available for controlled comparisons.

The following RAM/hybrid/prefetch measurements describe earlier candidate
stages; their default-off statements are historical. The active implementation
adds the branch and reuse-budget corrections at the end of this document.

The cache keeps positive and negative prefixes separately in process-local host
memory, using the backend's pinned host allocator when available. Storage dtype
is unchanged. Each graph-cut segment uploads only its own cache inputs into the
reusable GPU workspace. Transfers are synchronous; ComfyUI-style asynchronous
prefetch and double buffering are described in the optional follow-up below.
Normal completion, failure and runner cleanup release the host cache. No cache
tensors are written to disk.

Validation covers a single CUDA GPU with BF16 prefix caching. Multi-GPU cache
streaming and other devices are not validated. RAM mode can be slower when the
prefix and model already fit in VRAM.

## Restore and build

The reference profile is an overlay chain on the published base in
`native/upstream-lock.json`. Keep the base patches, workflow compatibility lock,
and frozen v1/v2 artifacts unchanged; the reference profile is selected separately. The current development
checkout has the RAM, hybrid, prefetch, branch-count and reuse-budget overlays applied.

From a restored base checkout, apply once:

```sh
git -C stable-diffusion.cpp apply --check ../native/reference-cache/ram-cache.patch
git -C stable-diffusion.cpp apply ../native/reference-cache/ram-cache.patch
git -C stable-diffusion.cpp apply --check ../native/reference-cache/hybrid-cache.patch
git -C stable-diffusion.cpp apply ../native/reference-cache/hybrid-cache.patch
git -C stable-diffusion.cpp apply --check ../native/reference-cache/prefetch.patch
git -C stable-diffusion.cpp apply ../native/reference-cache/prefetch.patch
git -C stable-diffusion.cpp apply --check ../native/reference-cache/branches.patch
git -C stable-diffusion.cpp apply ../native/reference-cache/branches.patch
git -C stable-diffusion.cpp apply --check ../native/reference-cache/reuse-budget.patch
git -C stable-diffusion.cpp apply ../native/reference-cache/reuse-budget.patch
cmake --build stable-diffusion.cpp/build-cuda --target sd-server --parallel 2
```

`source-lock.json` records the base lock, overlay, before/after source hashes,
and tested RAM executable. `hybrid-source-lock.json` pins the additional overlay,
its parent lock, final sources and tested hybrid executable. A clean archive of the pinned upstream plus the published
base patch and both overlays restored all four changed files byte-for-byte.
The base-only `native/prepare.py --check` deliberately does not accept an
overlaid development checkout; verify the overlay with its source lock instead.
`branches-source-lock.json` and `reuse-budget-source-lock.json` pin the last
two overlays. The full chain has been restored and verified against eleven
files. Install the accepted executable at `native/reference-cache/bin/sd-server`;
that directory is ignored by Git. A rebuild with a different executable hash
requires separate validation and an updated pin before the API accepts it.
Do not modify frozen v2's source manifest to describe this reference profile.

## Matched character-sheet test

40 ER-SDE/beta steps, CFG 2.5, identical seeds and prompts, portrait 704×864,
turnaround 832×864, final composition 1536×864. References were preprocessed
with the same 1,000,000-pixel cap per image. Both native runs used the same API
job path and scoped diagnostics; only the candidate binary/cache mode changed.

| Measurement | v2 GPU cache | RAM candidate | ComfyUI/AIO reference |
| --- | ---: | ---: | ---: |
| Total to saved output, seconds | 210.839 | 111.635 | 70.289 |
| Portrait sampling, seconds | 21.407 | 39.064 | 22.144 |
| Turnaround sampling, seconds | 173.180 | 56.580 | 31.411 |
| Turnaround weight-load events, steps 2–40 | 1493 | 0 | not measured |
| Summed newly loaded weight buffers, steps 2–40 | ~290 GiB | 0 | not measured |

All three candidate PNG files are byte-identical to the v2 CFG 2.5 control.
No cache fallback occurred. Both branches remained in RAM throughout sampling.
Synthetic CUDA checks passed exact BF16 copies for GPU and host storage,
separate branch contents, successful commit, failed-graph rollback and cleanup.
Frozen v2's executable hash is unchanged. All test processes stopped.

These are single-run measurements, not a general speed guarantee. ComfyUI uses
its own scheduler/noise implementations and asynchronous host-cache transfers.
The candidate removes the measured weight-reload bottleneck but is not yet as
fast as ComfyUI, and it regresses the portrait that already fit in VRAM.


## Historical hybrid placement

Hybrid mode chooses a fixed cache budget at the first store graph of each pass.
It uses the existing capacity checker with all diffusion parameters protected,
workspace, CUDA scratch, graph-cut buffers, K/V staging, the existing 512 MiB
runtime reserve and an additional 256 MiB planning margin. Unknown device memory
selects RAM; multi-device execution is rejected for this experimental mode.

The planner tries complete GPU placement first. Otherwise it estimates the peak
segmented workspace, counting only already produced cuts and the current outputs
that coexist. It searches the remaining capacity in 1 MiB increments. Half the
budget is reserved for each of two CFG branches; this is conservative when CFG
uses only one branch. Each branch packs whole layer K/V pairs into its share,
with remaining pairs in host RAM. The first prefix determines the budget; a
longer second prefix still cannot exceed its reserved share. Placement never
changes during successful sampling and is discarded at pass cleanup.

GPU tensors remain direct graph inputs. Only host tensors are staged, one segment
at a time, with the existing synchronous transfers. BF16 data and sampling math
are unchanged. This is an opt-in candidate, not a new default profile.

Synthetic CUDA assertions cover zero, partial and full GPU placement, indivisible
K/V pairs, two-branch budget accounting, exact BF16 bytes, rollback and per-pass
reset. The matched live result is recorded below. Tests do not establish timing
or memory guarantees for other models, shapes, devices, concurrent GPU workloads
or cancellation during live sampling.


### Matched hybrid result

The corrected hybrid run used the same API, settings, references and seeds as
both controls above. All three PNGs are byte-identical to both controls.

| Measurement | v2 GPU cache | RAM candidate | Hybrid candidate |
| --- | ---: | ---: | ---: |
| Total to saved output, seconds | 210.839 | 111.635 | 89.479 |
| Portrait sampling, seconds | 21.407 | 39.064 | 21.939 |
| Turnaround sampling, seconds | 173.180 | 56.580 | 51.240 |
| Portrait cache in VRAM / RAM, MiB | 3925.50 / 0 | 0 / 3925.50 | 3925.50 / 0 |
| Turnaround cache in VRAM / RAM, MiB | 9144.50 / 0 | 0 / 9144.50 | 3143.42 / 6001.08 |
| Turnaround weight-load events, steps 2–40 | 1493 | 0 | 0 |

Hybrid has zero weight-load/release events in steps 2–40 of either pass, no
cache fallback, 78 monolithic steady portrait forwards and 78 segmented steady
turnaround forwards. The result is one measurement per candidate, not a general
speed guarantee. It remains slower than the earlier 70.289-second ComfyUI run;
async host prefetch is still not implemented.

The first implementation trial overcounted not-yet-produced graph cuts in its
workspace estimate: it placed only 1531.23 MiB of the portrait in VRAM and all
turnaround cache in RAM. Its images also matched. That estimate was corrected
before the final run above; the initial executable and patch remain isolated in
the local test artifacts. Final placement counts only produced/current live cuts.

Live evidence: `character-sheet-green-dress-jewelry-cfg25-hybrid-cache-20261016-082120-de3a59`
under `test-generations`, job `job_68d16ae0518146ad83abb3e5f6f82820`.
Build, CUDA assertions, overlay restoration, source/binary hashes and diff checks
passed. GPU/API test execution used approved sandbox escalation. Both test API
servers and their native/wrapper processes stopped. The frozen v2 and previous
RAM executables remain unchanged. No default profile promotion, commit or push.


## Asynchronous prefetch candidate

Add `qwen_image_2_1_prefix_cache_prefetch=true` to the model args, alongside
`qwen_image_2_1_prefix_cache_device=hybrid`. Prefetch is off by default. The
normal Rust API still uses frozen v2. The independently preserved synchronous
hybrid executable remains the comparison control.

Two persistent GPU buffers each hold one segment's host K/V inputs. A separate
backend instance on the same CUDA device owns the transfer stream. The next
host group is queued before the current group's compute. Ready events gate
reads on the compute stream; consumed events gate reuse on the transfer stream.
GPU-resident cache entries stay direct inputs. No cache dtype or math changes.

Staging buffers are reused across positive/negative forwards and released at
pass cleanup. Their bytes are tracked as runtime residency, and hybrid planning
reserves two K/V pairs rather than one. Allocation admission protects all model
weights. Unsupported backends/events, unsafe cross-segment input reuse or failed
admission retain synchronous loading. Numeric diagnostics distinguish active
prefetch from fallback.

Graph cleanup synchronizes both streams before clearing graph pointers; pass
cleanup also releases staging before host cache tensors. Synthetic CUDA tests
with assertions enabled verified exact BF16 copies across seven groups and two
slots, distinct forwards, persistent reuse, rejected admission and interruption
with the next host copy still queued. No live cancellation test was run.

`prefetch.patch` adds `cache_prefetch.h/.cpp` and updates the runner and Qwen
model-argument parsing. `prefetch-source-lock.json` pins the hybrid parent lock,
all five changed/new source files and the tested executable. Clean restoration
from pinned upstream through base + RAM + hybrid + prefetch matched all files.
No ggml, CUDA kernel, frozen profile or base publication lock was modified.


### Matched prefetch result

Same isolated API workload, seeds, references, 40 ER-SDE/beta steps and CFG 2.5.
All three PNGs are byte-identical to v2, RAM and synchronous hybrid controls.

| Measurement | Synchronous hybrid | Hybrid + prefetch | Earlier ComfyUI/AIO |
| --- | ---: | ---: | ---: |
| Total to saved output, seconds | 89.479 | 79.754 | 70.289 |
| Portrait sampling, seconds | 21.939 | 21.884 | 22.144 |
| Turnaround sampling, seconds | 51.240 | 41.846 | 31.411 |
| Turnaround weight-load events, steps 2–40 | 0 | 0 | not measured |

Prefetch ran in all 78 steady turnaround forwards: 39 positive forwards with
22 host groups and 39 negative forwards with 21. No prefetch or prefix-cache
fallback occurred. Both passes have zero weight-load/release events after the
first step. Two staging buffers total 290.81 MiB and are reused across forwards.
Portrait cache remains entirely in VRAM (3925.50 MiB); turnaround placement is
2998.02 MiB VRAM + 6146.48 MiB RAM. Placement differs slightly from synchronous
hybrid because prefetch reserves additional staging memory and capacity reflects
available VRAM at admission. This is an end-to-end candidate comparison, not a
fixed-placement isolated transfer benchmark or hardware overlap trace.

Total time fell 9.725 seconds (10.9%); the gap to the earlier ComfyUI measurement
fell from 19.190 to 9.465 seconds. These are single-run measurements and do not
establish a general speed guarantee. ComfyUI has different scheduling/noise
implementations; cross-engine image identity is not claimed.

Live evidence: `character-sheet-green-dress-jewelry-cfg25-prefetch-20261016-083004-a3c16b`
under `test-generations`, job `job_9e67494e18e542a08022a3909585a1b7`.
Build, synthetic CUDA assertions, complete patch restoration, source/binary hashes
and diff checks passed. GPU/API execution used approved sandbox escalation.
The isolated API server, wrapper and native process stopped. Frozen v2 and the
synchronous hybrid executable are unchanged. No default promotion, commit or push.


## Branch-aware budget and reuse correction

The current candidate reserves cache space for the guidance branches actually
used by the complete sampling schedule, including forced negative guidance.
The sampling skip predicates and cache planner share the same calculation.
CFG 1 therefore reserves one branch rather than two. No sampling math changes.

Once every planned branch has been stored, the runner rechecks the budget once
against the smaller steady reuse graph. It first checks capacity without
discarding the current workspace, reserving model weights, measured segmented
workspace, K/V staging,
the existing runtime reserve and 256 MiB additional headroom. If the remaining
host cache fits completely, it releases the oversized store workspace, rechecks
physical capacity, copies the cache to VRAM and rebuilds the graph. Old
host tensors remain alive until the old graph is freed. Allocation failure
retains the original cache; insufficient capacity retains hybrid plus prefetch.
There is no per-step cache movement or repeated budget search.

`branches-source-lock.json` and `reuse-budget-source-lock.json` extend the
restoration chain. The complete chain restores eleven files with matching
hashes, including both prefetch source files. Frozen v2, the base publication
patches and the separate Turbo/workflow compatibility profile remain intact.
