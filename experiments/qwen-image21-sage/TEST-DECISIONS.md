# Qwen experiment decisions — read before proposing more optimization

Consolidated 2026-10-14, after Tomas accepted
[qwen-image21-1600x896-v1](BASELINE.md). This is the current decision register;
individual reports preserve the original chronology and detailed evidence.
No additional optimization is scheduled by this document. The current decision
is to freeze the reference profile and proceed with backend/API integration.

Publication cleanup preserved these reports and the compact accepted evidence
under `evidence/1600x896-v1/`. Old raw logs, images and redundant builds were
removed. Historical `runs/` references are provenance identifiers. The current
baseline launcher/checker no longer depends on those removed artifacts.

## How future work must use these results

- Do not rediscover or rerun a completed experiment merely because it appears
  plausible. Read its linked report and the frozen flags first.
- Reopen a rejected experiment only with a stated material change: relevant
  source/kernel/driver/hardware change, new workload or resolution, reproducible
  regression, or an explicitly requested investigation. State what new evidence
  makes the old conclusion insufficient. This is not a permanent ban on testing.
- Use the frozen executable and profile as the control. Preserve rejected
  results and failures; do not silently enable dormant flags or rebuild over v1.
- Keep synthetic timing, complete-graph memory, CUPTI attribution and actual
  image timing separate. An isolated speedup or smaller tensor did not always
  improve the full model. Allocator peaks moved repeatedly between operations.
- Compare contemporaneous pairs. GPU clocks were unlocked, desktop use remained
  present and small timing differences varied. Do not sum gains from separate
  stages of this chronology or claim that every accepted change is bit-exact
  relative to the original F32 engine.
- Bit identity means the stated comparison and inputs. The BF16 precision
  decisions established a new native image reference; ComfyUI has its own
  different reference. Current saved reference pixels are exact, while some
  small synthetic VAE shapes still fail.

## Setup, diagnosis and comparison experiments

| Investigation | Outcome and decision | Evidence |
| --- | --- | --- |
| Initial model/runtime setup and source build | Selected local Qwen INT8 ConvRot diffusion/encoder and Qwen VAE load successfully on SM89/CUDA 13.4. Native C++ is the inference engine; ComfyUI is only a comparison environment. H3 remains research-only. | [Design and setup](../../design-document.md) |
| Initial 1 GiB versus 512 MiB manual margin | Sampling stayed about 71 s; the second run also added the negative prompt, so this was not an isolated margin benchmark. Neither margin is the current setting. | Design: first/second Qwen smoke tests |
| Manual Comfy image comparison, with and without LoRA | Early 47/42 s UI observations were not controlled: model loading/LoRA differed; the selected no-LoRA PNG also recorded a BF16 encoder whereas native used INT8. Do not reuse these as an identical-input benchmark or expect matching pixels. | Design: weight-residency and Comfy source comparison |
| Are all diffusion weights resident during sampling? | Debug run staged 6920.55 MiB once; all 80 forwards were one segment with no repeated weight reloads within sampling. Low sampled VRAM did not establish offloading as the 71 s bottleneck. VAE decoding caused the later eviction/peak. | Design: weight-residency diagnosis |
| Is Sage really active? | CUPTI verified 2560 Sage2++ core calls, INT8 Q/K and FP8 V. The 64 masked-prefix Flash calls are an expected limited fallback. Do not propose globally inactive Sage as an unexplained cause again. | Design: direct kernel verification; [GPU profile](profiling/RESULTS.md) |
| F16 prefix-cache model argument | Saved about 276 MiB of sampled sampling-phase usage, changed pixels and showed no meaningful speed gain. Did not change the main F32 activation path. Not selected; later BF16 attention supplies its own BF16 caches. | Design: F16 prefix-cache experiment |
| First complete diffusion profile | INT8 GEMM ~21.08 GPU s, separate dequantization ~9.21 s, copies ~8.49 s, elementwise ~11.29 s; Sage core ~5.71 s. Established memory traffic/output stages as major targets, rather than assuming attention alone was responsible. | [RESULTS](profiling/RESULTS.md) |
| Matched ConvRot/INT8 chain versus Kitchen | Kitchen's fused output epilogue removed separate accumulator/dequantization traffic; smaller quantization differences also existed. Tested projection, gate/up and down, including reverse-order repeat. Motivated native fusions, not a Kitchen runtime dependency. | [INT8-KITCHEN](profiling/INT8-KITCHEN.md) |
| Actual headless AIO/Comfy/Kitchen path | Instrumented the installed adapter/core without a ComfyUI server. Kept private logs/history/workflows out. Established phase and kernel attribution on approved prompts/models; source inspection alone was not treated as runtime evidence. | [Headless results](comfy-headless/RESULTS.md) |
| Source-matched attention chain | Compared projections, Q/K norm/RoPE, prefix concatenation, attention and output. Native Sage and Kitchen use different precision/quantization; Kitchen also performs output-layout work. Motivated reduced traffic and BF16 investigation, not an unsupported claim of identical kernels. | [ATTENTION-CHAIN](profiling/ATTENTION-CHAIN.md) |
| Refreshed Comfy comparison after fusions | Native/Comfy had different remaining memory/precision/fusion paths; fresh-process and cached timings must not be mixed. Later resident comparison supersedes these as the main end-to-end reference. | [COMFY-CURRENT](profiling/COMFY-CURRENT.md) |
| Persistent runtime plus conditioning cache | Accepted. Last matched warm comparison: native 40.216 s mean / 39.384 median versus Comfy 38.814 / 38.708 s. Both skip warm text encoding. Native includes HTTP/polling/PNG transport; Comfy headless does not. References differ across engines. | [RESIDENT-COMPARISON](profiling/RESIDENT-COMPARISON.md) |
| Portrait followed by a single-reference environment edit | Completed with the unchanged frozen executable/flags. Very strong visual likeness in this one pair; 40.569 s portrait and 138.934 s edit to saved PNG. Actual VAE/vision reference encoding verified. Edit used 34 diffusion segments and larger BF16 prefix caches. This is a smoke test, not a new frozen editing baseline or universal quality guarantee. | [REFERENCE-EDIT](profiling/REFERENCE-EDIT.md) |
| Same edit with a 1 MP reference budget | Native `vae_input_max_pixels=1000000` produced 1344×736 reference encoding. Portrait/reference input hashes and all other settings matched. Edit fell from 138.934 to 47.401 s; 79/80 forwards used one segment, with smaller BF16 prefix caches. Very strong visual likeness remained in this pair; different output pixels. Single-run follow-up, not a universal quality or timing guarantee; frozen baseline unchanged. | [REFERENCE-EDIT follow-up](profiling/REFERENCE-EDIT.md#follow-up-1-megapixel-reference-budget) |

## Accepted diffusion and memory-planning choices

All are included in the frozen binary/profile. Several early implementations
are subsumed by later fusions; keeping their eligibility/fallback switches is
intentional. A switch being on does not mean its kernel always executes when a
later fusion replaces it.

| Experiment / selected setting | Why retained | Evidence |
| --- | --- | --- |
| Direct packed gate/up SwiGLU graph | Removed two copied halves; copy time dropped ~8.19→3.35 GPU s and compute workspace 1547.55→1274.27 MiB. Reference image preserved. This is different from the later rejected direct weight-read experiment. | [DIRECT-SWIGLU](profiling/DIRECT-SWIGLU.md) |
| Fused INT8 gate/up GEMM + F32 output | Removed separate INT32-output dequantization work. Full chain and model improved; bit and dispatch checks passed. Uses native CUTLASS code, not Python/Kitchen execution. | [FUSED-INT8](profiling/FUSED-INT8.md) |
| Fused 4096 projections | Full image improved; keep the >=512-row eligibility threshold. Small prefix shapes retain the original route because forcing the fused route there was not beneficial. | [PROJECTION-FUSION](profiling/PROJECTION-FUSION.md) |
| 11 CPU threads | Tested 4/8/11/16 over varied-order blocks. 11 was fastest overall and most stable; more threads did not help. Keep 11 on this hardware. | [CPU-THREADS](profiling/CPU-THREADS.md) |
| 512 CUDA threads for K=4096 ConvRot | Tested 256/512/1024; 512 selected, K=12288 remains on its existing width-specific path. CPU threads and CUDA block size are separate settings. | [CONVROT-THREADS](profiling/CONVROT-THREADS.md) |
| Packed SwiGLU + ConvRot | Removed the separate SwiGLU stage; affected GPU work ~5.19→2.86 s, reference and lifetime checks passed. Later gate/up epilogue further subsumes eligible work. | [SWIGLU-CONVROT](profiling/SWIGLU-CONVROT.md) |
| INT8 down-projection output fusion | Synthetic path passed; original full-image attempt was blocked by VAE memory admission, not proof the fusion was wrong. After scratch accounting fix, three pairs improved start-to-PNG 60.889→60.347 s with matching images. | [Original test](profiling/DOWN-PROJECTION.md), [completed test](profiling/DOWN-AUTO.md) |
| Auto-fit with actual CUDA scratch accounting | Fixed the measured missing cuBLAS-pool growth in admission. Reference succeeds without manual margin; internal safety reserve remains. Do not reintroduce -0.5/-1 as the baseline workaround. | [Diagnosis](profiling/VAE-MEMORY.md), [AUTO-SCRATCH](profiling/AUTO-SCRATCH.md) |
| Fused RMSNorm + RoPE | Paired image mean 58.109→56.289 s; F32-reference-preserving arithmetic and real dispatch checked. Later paired Q/K path handles the selected BF16 case. | [RMS-ROPE](profiling/RMS-ROPE.md) |
| LayerNorm/modulation + gated residual fusions | Mean start-to-PNG 56.282→53.400→50.617 s in that experiment; reduced large elementwise passes while preserving the then-current precision reference. | [MODULATION](profiling/MODULATION.md) |
| BF16 residual stream | Standalone gain ~0.945 s; initially retained as an optional precision change, not immediately selected. Later combined test justified enabling it with BF16 attention. Pixels differ from the original F32 reference. | [BF16-RESIDUAL](profiling/BF16-RESIDUAL.md), [combined test](profiling/BF16-COMBINED.md) |
| BF16 attention chain and caches | Direct BF16 projection/cache/preparation reduced traffic; real kernel dispatch and stable per-variant images verified. Not an all-BF16 attention core: Sage still uses INT8 Q/K, FP8 V. Accepted together with BF16 residual and established the newer native reference. | [BF16-ATTENTION](profiling/BF16-ATTENTION.md), [BF16-COMBINED](profiling/BF16-COMBINED.md) |
| BF16 norm/modulation + ConvRot | Reference-preserving fusion; paired start-to-PNG 47.351→46.361 s. | [NORM-CONVROT](profiling/NORM-CONVROT.md) |
| MLP down-projection + gated residual | Removed F32 intermediate output traffic while preserving explicit rounding into BF16 residual. Improved full-image timing and preserved current reference. | [DOWN-RESIDUAL](profiling/DOWN-RESIDUAL.md) |
| Attention projection + scale + gated residual | Removed ~1.535 GPU s of targeted work; full-image pairs favored it, but larger wall-time drift is not a guaranteed saving. Same reference image. | [ATTENTION-RESIDUAL](profiling/ATTENTION-RESIDUAL.md) |
| Paired Q/K RMSNorm/RoPE | Shared launch for Q/K; image sampling 40.117→38.770 s in its paired test, reference preserved. | [PAIRED-RMS-ROPE](profiling/PAIRED-RMS-ROPE.md) |
| Direct split V prefix reads, K off | Removes V concatenation; mean start-to-PNG 43.740→43.170 s, reference preserved. Additional K is a separate rejected choice below. | [SAGE-SPLIT-KV](profiling/SAGE-SPLIT-KV.md) |
| Fused F32 gate/up + SwiGLU, packing retained | Includes packing cost and still wins all three full-image pairs: 43.095→41.911 s mean to PNG. Reference preserved; keep this rather than the direct-read variant. | [GATE-UP-SWIGLU](profiling/GATE-UP-SWIGLU.md) |

## Rejected, superseded or conditionally excluded choices

| Experiment / final setting | Why not in the baseline | Evidence / reopening boundary |
| --- | --- | --- |
| ConvRot fused absmax scan | No useful gain: full-model quantization 3.407→3.419 GPU s. Patch restored out of active source. Do not try the unsupported archived flag in the current binary. | [CONVROT-AMAX](profiling/CONVROT-AMAX.md); reopen for a changed quantizer/workload, not the same K=4096/512 case |
| Gate/up CUTLASS tile/pipeline sweep | Six candidates plus reverse-order repeats. Smaller tiles/two stages slower; four-stage gains inconsistent. Keep 128×256×64, three stages. Archived tuning selector is not a current supported flag. | [GATE-UP-TUNING](profiling/GATE-UP-TUNING.md) |
| Force projection fusion on short shapes | Small sequences regressed; keep the tested fallback threshold instead of assuming all GEMMs benefit. | [PROJECTION-FUSION](profiling/PROJECTION-FUSION.md) |
| Old F32 V direct-cast flag = 0 | Tested and bit-exact on the F32 attention path, but later BF16 attention reads V directly itself. Superseded, not an extra additive optimization for the current path. | [ATTENTION-CHAIN](profiling/ATTENTION-CHAIN.md), [BF16-COMBINED](profiling/BF16-COMBINED.md) |
| Vectorized F32 SwiGLU/ConvRot = 0 | Only ~0.006 ms isolated advantage; full-model targeted time 2.763→2.782 GPU s. No useful transferred speedup, despite bit identity. Did not proceed to a new ordinary image timing series. | [SWIGLU-F32](profiling/SWIGLU-F32.md) |
| BF16 gate/up intermediate = 0 | Image changes: ~92.3% of pixels differ, SSIM ~0.952. Mean PNG gain only 0.358 s; only one of three paired PNG times favored BF16, profile gain unstable. Keep F32 gate/up. | [BF16-GATE-UP](profiling/BF16-GATE-UP.md) |
| Split K prefix reads = 0 | Targeted copy reduction did not produce consistent extra end-to-end benefit over V alone; K mean/quantization became slower, V+K median was worse. Keep V alone. | [SAGE-SPLIT-KV](profiling/SAGE-SPLIT-KV.md) |
| Direct gate/up weight reads without packing = 0 | Isolated chain 3.1–3.4% faster but full images showed no gain: mean to PNG 41.872→41.914 s, same pixels. Keep packing; no allocator saving was demonstrated. | [GATE-UP-DIRECT](profiling/GATE-UP-DIRECT.md) |
| Idle retention = 0, before VAE strips | Same 5911.70 MiB reload in both modes; only 0.135 s mean gain, inconsistent by series. No demonstrated residency benefit. | [RETAIN-IDLE](profiling/RETAIN-IDLE.md) |
| Idle retention = 0, after original convolution strips | Same 2946.44 MiB / three-block reload and 0.40 s loading. Observed 0.853 s mean advantage mostly sampling variation, not improved retention. User chose off. No retention-on comparison was run with the final three-way combination; its retention-off reload stayed unchanged. | [STRIP-RETENTION](profiling/STRIP-RETENTION.md); revisit only with evidence that placement can preserve more blocks |
| Fused VAE upscale + scale + im2col = 0 | Production synthetic im2col exact and locally saved 1575 MiB, but complete loaded graph grew to 12606.86 MiB; admission fell back to tiles and image changed. Stopped at image identity gate. | [VAE-UPSCALE-IM2COL](profiling/VAE-UPSCALE-IM2COL.md) |
| Old metadata-only VAE probe | Not an optimization. Wrong parameter shapes forced a different convolution path and misleading memory prediction. Corrected probe matches loaded graph/offsets. Never reuse the old probe to claim full-VAE savings. | [VAE-LOADED-LAYOUT](profiling/VAE-LOADED-LAYOUT.md) |
| PAD-scale / resampling-output reuse = off | All eight corrected-probe combinations failed to beat the 10632.62 MiB control. Fusion+PAD candidate was 11026.37 MiB; live images exact but warm time 39.595→41.258 s and VAE 1.715→1.768 s. Not selected. | [VAE-BUFFER-REUSE](profiling/VAE-BUFFER-REUSE.md) |
| Manual VRAM margins as a permanent fix | Useful diagnostic controls before scratch accounting, now superseded. Auto with real scratch accounting is the accepted path. Do not equate the old manual margin with the still-existing internal reserve. | [VAE-MEMORY](profiling/VAE-MEMORY.md), [AUTO-SCRATCH](profiling/AUTO-SCRATCH.md) |

## VAE strip sequence: rejected alone, accepted together

These rows must be read as a sequence. A negative standalone result does not
contradict the final accepted combination.

| Experiment | Full-graph result and decision | Evidence |
| --- | --- | --- |
| Original two large convolution strips | **Accepted.** VAE workspace 10632.62→8663.87 MiB, reload 5911.70→2946.44 MiB, exact reference images. Warm VAE ~1.713→1.693 s. Keep original precision and strip input packing. | [VAE-STRIP-CONVOLUTIONS](profiling/VAE-STRIP-CONVOLUTIONS.md) |
| Whole final-stage first residual block alone | Isolated workspace 8275.39→3144.27 MiB and production synthetic output exact, but complete VAE remained 8663.87 MiB. No image run followed; not useful alone. Later enabled in the final combination. | [VAE-BLOCK-STRIPS](profiling/VAE-BLOCK-STRIPS.md) |
| DupUp3D shortcut with direct addition alone | Isolated workspace 5512.50→2531.25 MiB, synthetic cases exact. Complete arena instead grew to 8669.36 MiB. Metadata-only strip sizes 16/32/64 rows produced the same arena; only implemented 32-row path got numerical validation. No image run. | [VAE-SHORTCUT-STRIPS](profiling/VAE-SHORTCUT-STRIPS.md) |
| Stage-3 convolution alone / with shortcut | Alone 9057.62 MiB; shortcut+stage-3 8669.36 MiB, still worse than control. Production convolution exact and locally 3150 MiB smaller, but no image test because complete-memory gate failed. | [VAE-SHORTCUT-STAGE3](profiling/VAE-SHORTCUT-STAGE3.md) |
| All three plus original convolution strips | **Accepted for 1600×896.** Arena 8663.87→8073.24 MiB (590.625 MiB saved); 12/12 RGBA images exact. Warm PNG 39.074→39.219 s, VAE 1.700→1.720 s, same three-block reload. Chosen for memory headroom, not speed. | [VAE-THREE-STRIPS](profiling/VAE-THREE-STRIPS.md) |
| Additional stage-2 800×448, 576→576 convolution strips | **Tested candidate; not promoted into v1.** With all frozen flags, arena 8073.24→7879.11 MiB (194.13 MiB saved); production component and 12/12 RGBA images exact. Warm PNG mean 38.627→38.756 s, median 38.689→38.522 s; VAE 1.6725→1.6825 s. Same 2946.44 MiB / three-block reload. Useful memory headroom, no demonstrated speed or residency gain. Candidate patch/binary preserved separately, native source restored. | [VAE-STAGE2-STRIPS](profiling/VAE-STAGE2-STRIPS.md); repeat only for changed workload, implementation or residency evidence |

Numerical boundary: production component shapes pass, but small synthetic VAE
shapes do not all match. Original strip convolution failures were localized to
GEMM after im2col matched, not halo/SET assembly. The stage-3 65×129 test had
2,288,498 differing floats (max 6.375); whole-block 65×129 had 1,173,302 (max
0.0234375). Their exact internal causes were not newly isolated. Do not suppress
these failures or extrapolate saved reference-pixel identity to all shapes.

## Dynamic stage-2 follow-up (completed)

The fixed-size guard was removed in a separate candidate. Production component
and all eight images at 1600×896 and 4096×2304 are bitidentical per resolution.
At 4096 the full-graph prediction drops 53145.00→51856.03 MiB, but both use 49
VAE tiles in practice: actual arena 3321.56→3242.32 MiB, warm VAE
32.19→32.28 s. Total warm time worsens 508.18→671.94 s with much more segmented
diffusion/weight streaming; the cause of different admission is not isolated.
Not promoted. Do not propose this as an untested idea or claim it accelerates
4096 generation. See [VAE-STAGE2-DYNAMIC](profiling/VAE-STAGE2-DYNAMIC.md).

## ComfyUI 4096 reference (completed)

Headless AIO/Kitchen, same synthetic prompts/models/seed and 4096×2304,
40 ER-SDE/beta steps, CFG 2.5: first/repeat 473.87/465.64 s, repeat sampling
450.97 s and VAE 7.47 s. Two images are bitidentical within ComfyUI, not to
native. The repeat prompt cache expired (AIO TTL 300 s). No Comfy spatial VAE
fallback was logged; internal allocation warnings recovered. Native frozen
repeat was 508.18 s with 32.19 s VAE. One repeat each is not a controlled speed
sweep. See [reference report](comfy-headless/REFERENCE-4096.md) before proposing
another 4096 comparison or attributing the gap to a particular implementation.

## Discussed or researched, not completed optimization tests

- **FlashAttention replacement:** user explicitly deferred it. The observed
  masked-prefix Flash fallback is not a whole-model Flash-versus-Sage benchmark.
- **Kitchen in the text encoder:** source path discussed; it was not selected
  for implementation. Warm cached comparisons execute no text encoder anyway.
- **MiniMax H3, alternative quantizations, LoRAs, broader editing scenarios,
  other resolutions and GPUs:** research or future compatibility work. The
  single-reference portrait-to-environment smoke test above is the only editing
  case measured so far; it does not validate these other workloads.
- Public RTX 4070 and AMD reports used different weights/backends/conditions.
  They motivated local profiling; they are not proof of our precise bottleneck.

## Freeze validation

This freeze changes launch/configuration/documentation, not native inference.
Reuse the completed 12-image comparison; do not rerun GPU benchmarks solely to
re-establish the same binary/flags. Verify profile hashes, generated commands,
flag selection and launcher behavior locally. A change to native computation,
model assets or workload warrants separately scoped runtime validation.

## Dynamic whole-block/upscale strips (completed 2026-10-15)

A separate `SD_VAE_DYNAMIC_BLOCK_STRIPS=1` candidate extends tensor-budget-based
strips to more whole residual blocks and complete upscale/convolution sequences,
preserving precision and all frozen switches. Leaf destinations initially grew
the arena; allocating destinations after the first computed strip fixed that.
Thirteen synthetic comparisons pass byte identity and retained-input checks.

Full non-tiled graph arena: 1600, 8073.24→6373.44 MiB; 4096,
53145.00→36279.00 MiB. At 1600 all four images match v1, warm PNG
39.210→39.007 s, VAE 1.73→1.72 s, reload 2946.44→1008.83 MiB.
At 4096 the candidate fits 9 larger tiles instead of 49; warm VAE
32.69→23.39 s and PNG 499.518→480.405 s, with sampling also 9.82 s faster.
Actual tiled arena increases 3321.56→10498.50 MiB because tile size increases.
One warm sample per mode is not a statistically established total speed gain.

**4096 cross-variant pixels are not bitidentical**: 44.2843% of RGBA pixels
change, RGB mean absolute error 0.196635/255, max channel error 15. Repeats
within each mode are exact. Different outer tiling is a confounder; no
same-outer-tile comparison isolates its share. Not promoted; active native
source restored, candidate patch/binary preserved separately. Read
[VAE-DYNAMIC-BLOCKS](profiling/VAE-DYNAMIC-BLOCKS.md) before proposing it again.

## v2 acceptance — 2026-10-15

Tomas accepted the dynamic-block candidate as `qwen-image21-1600x896-v2` for
1600×896. The verified binary/flags are copied and pinned independently; v1 is
preserved. This supersedes the prior "not promoted" decision for the 1600 scope
only. The algorithm remains dynamic. Same-grid 4096 investigation is separate
from the default profile and does not set a fixed outer tile size in launchers.

## 4096 identical-outer-grid control — completed 2026-10-15

Frozen v1 versus v2, initial plus warm image each, explicitly 49 tiles of
1024×576 at overlap 0.5. Actual geometry and absence of fallback are asserted.
All four RGBA images are bitidentical and match the v1 reference. Internal
v2 strips remain dynamic. Actual VAE arena 3321.56→2476.69 MiB (844.87 MiB /
25.4% saved); warm VAE 33.05→32.00 s, initial 33.52→32.95 s. One warm sample
per mode does not establish a speed gain. Warm full image 498.072→507.611 s
because sampling/staging also differ; do not attribute that to VAE or claim
improved 4096 residency. Same-grid exactness supports outer tiling as the cause
of the earlier auto-grid pixel difference for this workload. Do not repeat
this as an untested hypothesis. No fixed outer tile setting is adopted into
v2 defaults. See [VAE-FIXED-TILES-4096](profiling/VAE-FIXED-TILES-4096.md).

## Reference-cache acceptance (2026-10-16)

The separate RAM/hybrid/prefetch candidate passed image equality and live API
cancellation/recovery. An ordinary three-reference image improved from 205.457
to 57.331 s with identical PNG bytes. However, the reference sheet at CFG 1
regressed from 41.271 to 46.025 s: hybrid reserved two branches and streamed
cache unnecessarily for a single computed branch. **Do not promote yet; v2
remains the default.** See [acceptance report](../../native/reference-cache/ACCEPTANCE.md)
for scope, preserved controls and the proposed branch-aware follow-up. That
proposal is not authorization to implement or run it.


## 2026-10-16 — corrected reference-cache budget accepted

Supersedes the CFG 1 promotion block above. Counting the actual guidance
branches was insufficient (43.667 vs 39.888 s); the runner now also rechecks
cache capacity once against the smaller steady reuse graph. It promotes the
remaining host cache only when the entire cache fits, preserving model,
workspace, staging and safety reserves. Rejected promotion retains the warm
workspace and existing hybrid placement.

Final CFG 1: 39.620 s vs v2 39.888 s, identical PNGs. CFG 2.5: 80.843 s with
all pinned PNG hashes matching; zero steady weight loads and 78 turnaround
prefetch forwards. Three-reference ordinary-image sampling: 45.836 s vs prior
prefetch 45.394 s, identical PNG. Cancellation while prefetch was active and
immediate recovery passed; recovery completed in 81.393 s.

Intermittent low automatic startup limits produced all-RAM/synchronous fallback
in separate trials. Waiting for process exit alone did not eliminate that
variation. Final cancellation/multi-reference benchmarks also stabilized free
GPU memory before native launch; that test-only wait is included in their raw
totals and is not a production memory requirement. Native auto limits remain
unchanged; individual measurements do not establish guarantees under competing
GPU allocations. See the full evidence and limitations in
[native/reference-cache/ACCEPTANCE-REUSE.md](../../native/reference-cache/ACCEPTANCE-REUSE.md).

Enable the verified reference executable for all actual Qwen Image 2.1
reference passes, independent of public operation. Keep v2 for text-only,
explicit `IMAGINE_QWEN_REFERENCE_CACHE=off`, and missing/hash-mismatched candidate
fallback. Other model families and Turbo keep their existing selection.
Frozen v1/v2, base publication patches and workflow compatibility remain intact.
