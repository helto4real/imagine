# ConvRot CUDA block-size experiment — 2026-10-13

The 512-thread variant is the preferred **opt-in experimental reference** for
K=4096: mean sampling fell from **58.83 to 58.42 seconds (0.7%)** in three
full-generation comparisons. Mean start-to-saved-PNG fell from **63.71 to
63.33 seconds (0.6%)**. The default remains 1024; the improvement is modest
and does not explain the wider gap to ComfyUI.

## Full-generation results

Means of three uninstrumented runs per variant, with CPU threads fixed at 11:

| ConvRot CUDA threads | Sampling | Sampling min–max | Start to saved PNG | Process wall |
| --- | ---: | --- | ---: | ---: |
| 1024 control | 58.83 s | 58.81–58.84 s | 63.71 s | 63.82 s |
| **512** | **58.42 s** | **58.31–58.62 s** | **63.33 s** | **63.48 s** |
| 256 | 58.55 s | 58.39–58.76 s | 63.41 s | 63.54 s |

Block-by-block sampling times:

| CUDA threads | Block 1 | Block 2 | Block 3 |
| --- | ---: | ---: | ---: |
| 1024 | 58.84 s | 58.81 s | 58.84 s |
| 512 | 58.33 s | 58.62 s | 58.31 s |
| 256 | 58.49 s | 58.76 s | 58.39 s |

Both smaller blocks beat their controls in all three blocks; 512 also beat
256 in every block. The mean 512-vs-256 sampling difference is only **0.127 s**
and is small relative to their run-to-run variation. The result supports using
512 for the next local experiment, not a universal ranking on other GPUs or
workloads. The 0.41 s reduction is the measured whole-image sampling gain;
the much larger percentage improvement in the quantizer must not be presented
as a whole-image speedup.

Mean conditioning times are 1.947/1.967/1.927 s for 1024/512/256; mean VAE
times are 2.027/2.030/2.027 s. These show no material end-to-end gain outside
sampling in this run. Warm filesystem cache and the shared desktop GPU mean
these results must be compared with fresh controls, not with a fastest image
from an earlier experiment.

## Full-diffusion GPU attribution

Separate CUPTI runs, excluded from the timing averages above:

| Requested K=4096 block size | ConvRot GPU time | Observed launches |
| --- | ---: | --- |
| 1024 | 3.8518 s | 10,240 at 1024 threads |
| 512 | 3.2411 s | 7,680 at 512, 2,560 at 1024 |
| 256 | 3.2639 s | 7,680 at 256, 2,560 at 1024 |

The 512 variant reduces this stage by **0.611 s (15.9%)** in the separate
profiles. Of its 3.2411 s, 1.6321 s is in the new 512-thread launches and
1.6090 s in the unchanged 1024-thread down-projection quantizer. For 256,
the corresponding durations are 1.6547 and 1.6092 s. This confirms a real
improvement in the targeted GPU work; single profiles on an unlocked shared
GPU do not determine the exact total latency saving. Use the uninstrumented
0.41 s sampling difference for the measured whole-image result.

All three profiles have 80 forwards, 2,560 Sage cores, 10,240 fused ordinary
projections, 2,560 fused gate/up operations, the expected 2,560 remaining
separate GEMM/dequantization pairs, zero dropped records, and zero activities
crossing the selected phase boundaries. Actual CUDA block dimensions and
kernel specializations are verified, including the unchanged K=12288 path.

All **13 full images** (nine timing images, one warmup, three profiles) match
the original RGB reference. Timing-run validation also confirms unchanged
non-thread parameters, prompt bytes, binary hash, placement, compute-buffer
sizes, and 11 CPU threads. Both new block sizes passed focused CUDA memcheck
and racecheck, and Python syntax and GGML whitespace checks passed. The GPU
returned to desktop usage (1669 MiB, 0% utilization) after completion.

## Implementation

The native `quantize_rowwise_i8_convrot_cuda` kernel is templated on block size.
Its rotation loop now uses `block_threads / 64` groups per wave instead of a
hardcoded 16. The max reduction already uses `blockDim.x / 32` participating
warps. Arithmetic, scaling, rounding, saturation, zero-row handling, padding,
and shared-memory layout remain unchanged. All three compiled variants use
28 registers per thread and 128 bytes of static shared memory; K=4096 also uses
16 KiB of dynamic shared memory. Resource counts come from `cuobjdump`, not a
hardware-counter or achieved-occupancy measurement.

`GGML_CUDA_CONVROT_THREADS=256` or `512` selects the experimental variant for
K=4096 only. Unset, `1024`, or unrecognized values use 1024. The value is read
once per process. K=12288 and other widths retain the 1024-thread implementation;
the existing insufficient-shared-memory fallback is unchanged. Selection is by
tensor width, so eligible text-encoder operations can also use the new variant.

No absmax/rotation fusion, SwiGLU fusion, Kitchen numerical semantics, CPU
affinity, or CUDA clock settings are changed in this experiment. Both existing
INT8 GEMM fusion flags remain enabled. Full-image tests use `--threads 11`.

## Synthetic correctness and timing

Thirteen cases cover K=4096 with 1, 3, 16, 57, 230, 511, 512, 513, 5600,
5657, and 5830 rows; a 5600-row gate/up projection; and a K=12288 down-projection
control. The test uses the actual GGML quantizer and Linear graph. Inputs include
normal F32 random values, zero rows, 1e-20 and subnormal magnitudes, large finite
values, an impulse, alternating signs, an outlier, and inverse-rotated target
values at and near quantization half steps. Channel scales are random F32.

All three new variants match the preserved pre-change executable bit for bit:
**238,100,480 output F32 values**, **169,287,680 quantized INT8 values**, and
**30,130 row scales** per variant. Outputs and scales are checked for finiteness.
The standalone quantization graph also matches the full Linear graph.

Each measurement uses 20 warmups and seven synchronized batches of 30 calls.
Batch medians below include GGML dispatch and synchronization, and exclude input
creation, tensor loading, and output checking. Separate five-call CUPTI traces
verify actual kernel names and block sizes, with zero dropped events and no
events crossing the measured boundaries. K=12288 always launches 1024 threads.

| Shape | Metric | 1024 threads | 512 threads | 256 threads |
| --- | --- | ---: | ---: | ---: |
| M=5600, K=N=4096 | Quantization only | 0.2601 ms | 0.1916 ms | 0.1938 ms |
| M=5600, K=N=4096 | Full Linear | 0.9240 ms | 0.8906 ms | 0.8686 ms |
| M=5657, K=N=4096 | Full Linear | 0.9592 ms | 0.8855 ms | 0.8786 ms |
| M=5830, K=N=4096 | Full Linear | 0.9889 ms | 0.9387 ms | 0.9228 ms |
| M=5600, K=4096, N=24576 | Full gate/up Linear | 4.2477 ms | 4.2386 ms | 4.1923 ms |
| M=5600, K=12288, N=4096 | Unchanged down control | 2.7581 ms | 2.8069 ms | 2.7458 ms |

The unchanged control's variation illustrates measurement noise. Small row
counts are mostly dispatch-bound; 256 threads has a small kernel-time penalty
at 1–57 rows. The 512 variant is marginally faster for isolated large-row
quantization, while 256 is faster in the tested whole Linear chains. This is
why both candidates advance to full-generation testing rather than selecting
one from the quantizer alone.

CUDA Compute Sanitizer **memcheck and racecheck** each passed for both 256 and
512 threads: zero errors, and zero race warnings/hazards. These checks use a
17-row K=4096 quantization graph with 20 warmups and one measured invocation,
including a zero row and small values. They are focused checks of shared-memory
rotation and row-tail handling, not exhaustive verification on every GPU.

## Full-generation method

One excluded 1024-thread warmup precedes three measured blocks:
`1024,512,256`; `256,512,1024`; and `512,1024,256`.
All runs use the same new executable, fixed 11 CPU threads, 1600x896, 40
ER-SDE/Beta steps, CFG 2.5, seed 42, CPU RNG, F32 prefix cache, SageAttention2++,
both GEMM fusion flags, the saved prompts, and a 512 MiB auto-fit margin.

Every run starts a new process with warm filesystem cache. The desktop GPU is
shared and clocks are unlocked. Start-to-save is measured from immediately
before CLI launch to receipt of its successful PNG-save log; process wall also
includes exit. Conditioning, sampling, and VAE durations come from engine logs
and include their relevant lazy loading. Full images are compared by decoded
RGB SHA-256 against the unchanged reference. Timing runs do not load CUPTI.

## Reproduction and evidence

```sh
python experiments/qwen-image21-sage/profiling/compare-convrot-threads.py
python experiments/qwen-image21-sage/profiling/compare-convrot-images.py
```

Sources, the pre-change binary/source snapshot, new binaries, build log,
isolated patch, resource counts, and sanitizer logs are in
`../runs/convrot-threads-build/`. The CLI and server both built successfully;
the server request path is not live-tested.

- Synthetic results: `../runs/20261013T095446Z-convrot-threads-xlu857p4/`.
- Full-image comparison: `../runs/20261013T095713Z-convrot-images-m_3l0onq/`.
- Full profiles and block counts: `../runs/convrot-threads-build/full-profile-comparison.json`.
- Profile 1024: `../runs/20261013T100806Z-activity-90km660l/`.
- Profile 512: `../runs/20261013T100909Z-activity-ykkf3ot5/`.
- Profile 256: `../runs/20261013T101012Z-activity-wufkrgvk/`.
- CLI SHA-256: `343105859bf1c32028762085a5a46afcdcc4c515c5517c9f97f03c4489d8dbd3`.
- RGB reference SHA-256: `253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2`.

GPU runs required approved sandbox escalation. No commit or push was made.
