# Fused ConvRot absolute maximum — 2026-10-13

**Not adopted: no useful speedup was demonstrated for K=4096 at 512 CUDA
threads.** The active engine source is restored byte for byte to the previous
block-size experiment. Its 512-thread opt-in remains the reference. The
experimental patch, binaries, tests, and measurements are retained for review.

## Change tested

Accumulate each thread's absolute maximum from the four register values
produced by the final radix-4 rotation stage, across both group waves. This
removes the subsequent scan of the row in shared memory. The warp/block max
reduction, scale computation, rounding, saturation, zero handling, barriers,
and stored rotated values are unchanged.

The experiment is selected only for K=4096 and 512 threads by
`GGML_CUDA_CONVROT_FUSED_AMAX=1`. That flag exists **only in the preserved
experimental binary**, not in the restored active engine. Other widths and
block sizes retain their old behavior. No SwiGLU fusion or Kitchen numerical
semantics are introduced. Both compiled variants use 28 registers per thread,
128 bytes of static shared memory, and 16 KiB dynamic shared memory at K=4096.

## Synthetic results

The same 13 cases as the block-size experiment cover rows 1, 3, 16, 57, 230,
511, 512, 513, 5600, 5657, and 5830 at K=N=4096, a full-size gate/up projection,
and an unchanged K=12288 down-projection control. Inputs include normal F32
values, zero rows, tiny/subnormal values, large finite values, impulses,
alternating signs, outliers, and values at or near quantization half steps.

Both new variants (fusion off/on) match the preserved 512-thread executable
bit for bit across **238,100,480 Linear output F32 values**, **169,287,680
quantized INT8 values**, and **30,130 scales** per variant. Outputs/scales are
finite. The quantization-only graph also matches the full Linear graph.

Timings use 20 warmups and seven synchronized batches of 30 calls. These are
batch medians, including GGML dispatch/synchronization but excluding setup:

| Shape | Metric | Fusion off | Fusion on |
| --- | --- | ---: | ---: |
| M=5600, K=N=4096 | Quantization only | 0.190034 ms | 0.189883 ms |
| M=5657, K=N=4096 | Quantization only | 0.193739 ms | 0.193743 ms |
| M=5830, K=N=4096 | Quantization only | 0.199093 ms | 0.198565 ms |
| M=5600, K=N=4096 | Full Linear | 0.849826 ms | 0.848954 ms |
| M=5657, K=N=4096 | Full Linear | 0.859503 ms | 0.855057 ms |
| M=5830, K=N=4096 | Full Linear | 0.901715 ms | 0.899405 ms |
| M=5600, K=4096, N=24576 | Full gate/up Linear | 4.100378 ms | 4.097646 ms |
| M=5600, K=12288, N=4096 | Unchanged down control | 2.836957 ms | 2.774405 ms |

Smaller rows show slight kernel-time reductions. The gate/up case's isolated
quantizer reported 0.218720/0.189708 ms, unlike the essentially unchanged same-K
projection cases. Its separate five-call GPU trace also varied substantially.
The unchanged down control varied as well. We therefore do not interpret that
isolated gate/up difference as a repeatable gain. Full Linear differences are
small, and the full-model attribution below does not support promoting the
change. The complete original/off/on results remain in `summary.json`.

Every separate synthetic CUPTI trace verifies the expected block size and
fused/non-fused template specialization, with zero dropped records and no
events crossing the timing boundaries. The down control remains non-fused.

## Full-diffusion profiles

One complete Qwen generation per variant, profiled separately from the
synthetic timing batches, uses the same experimental executable, 11 CPU
threads, 512 CUDA threads for K=4096, both GEMM fusion flags, SageAttention2++,
1600x896, 40 ER-SDE/Beta steps, CFG 2.5, seed 42, CPU RNG, F32 prefix cache,
512 MiB auto-fit margin, and the unchanged reference prompts/models.

| GPU activity | Fusion off | Fusion on |
| --- | ---: | ---: |
| K=4096 quantizer, 7,680 launches | 1.721022 s | 1.738111 s |
| K=12288 control, 2,560 launches | 1.685661 s | 1.681135 s |
| Total ConvRot / activation quantization | **3.406683 s** | **3.419246 s** |

This is effectively unchanged, not evidence of a useful optimization. A
single profile pair on a shared GPU with unlocked clocks does not establish
a statistically significant regression either. No repeated uninstrumented
full-image latency benchmark was run: the isolated and full-model stage
measurements did not justify advancing the candidate.

Both full images are pixel-identical to the existing RGB reference. Profiles
confirm all 80 forwards, 2,560 Sage cores, expected fused GEMM counts, 7,680
selected K=4096 launches, and 2,560 unchanged K=12288 launches. The on profile
contains exactly 7,680 fused-amax kernels; the off profile contains none.
Neither profile has dropped events or boundary-crossing activities. Placement
and compute-buffer sizes match between runs.

## Validation and disposition

CUDA memcheck and racecheck on the new 17-row K=4096 quantizer report zero
errors, warnings, and race hazards. The test includes a zero row and tiny
values, 20 warmups, and one measured invocation. The CLI and server build;
the server request path is not live-tested. Python syntax and GGML whitespace
checks pass. GPU execution required approved sandbox escalation.

The engine source was restored only after checking that it still matched our
experimental snapshot, preserving all pre-existing changes. The candidate is
retained as an isolated patch and frozen executable rather than added to the
active runtime. No commit or push was made.

After rebuilding the restored source, a separate uninstrumented reference
generation also produced exactly the same RGB pixels. Its run is
`../runs/20261013T105012Z-control-s03v2qiv/`, and the rebuilt CLI SHA-256 is
`b648e696636762fd4c351a27672a31d827991b86d213cfe727d9fb3363070074`.
Source restoration is byte-identical; the fresh build has a different binary
hash from the earlier preserved CLI. This last image verifies restoration,
not a repeated latency benchmark. Details are in `restoration.json`.

## Evidence and reproduction

- Build, before/after source, isolated `experiment.patch`, frozen binaries,
  resource counts, sanitizer logs, and restoration log:
  `../runs/convrot-amax-build/`.
- Complete synthetic results:
  `../runs/20261013T104257Z-convrot-amax-hffsvs8e/`.
- Full profile comparison:
  `../runs/convrot-amax-build/full-profile-comparison.json`.
- Off profile: `../runs/20261013T104500Z-activity-l3zust7a/`.
- On profile: `../runs/20261013T104605Z-activity-6q3irjwl/`.
- Experimental CLI SHA-256:
  `c235277800686ca42551ae7509ea037c02a5626c58c7831df09eece70be91789`.
- RGB SHA-256:
  `253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2`.

The synthetic driver deliberately uses the frozen before/after binaries:

```sh
python experiments/qwen-image21-sage/profiling/compare-convrot-amax.py
```

For one full experimental replay, use the preserved binary explicitly; change
`FUSED_AMAX` to `0` for the control. These commands require a free GPU.

```sh
GGML_CUDA_FUSED_INT8_GATE_UP=1 GGML_CUDA_FUSED_INT8_PROJECTION=1 \
GGML_CUDA_CONVROT_THREADS=512 GGML_CUDA_CONVROT_FUSED_AMAX=1 \
python experiments/qwen-image21-sage/profiling/run.py --threads 11 \
  --binary experiments/qwen-image21-sage/runs/convrot-amax-build/sd-cli
```
