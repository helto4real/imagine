# Qwen Image 2.1: F32 SwiGLU/ConvRot kernel experiment

Date: 2026-10-13. RTX 4070 Ti SUPER, CUDA 13.4, SM89.

## Result and decision

**No useful runtime speedup is demonstrated.** The tested candidate preserves
INT8 values, F32 row scales and the reference image exactly, but its tiny isolated
advantage does not survive the full-model profile. Keep the original kernel in
the accepted configuration. The prototype remains available behind the
**default-off** `GGML_CUDA_SWIGLU_CONVROT_VEC=1` switch.

| Measurement | Original | Vector candidate |
| --- | ---: | ---: |
| Isolated 5600-row kernel, 24 alternating pairs | 1.022730 ms | 1.016804 ms |
| Full-model SwiGLU/ConvRot GPU time, 2560 calls | 2.762655 s | 2.781533 s |
| Total selected diffusion GPU activity | 36.969080 s | 37.334061 s |
| Instrumented sampling timer | 38.87 s | 39.42 s |

The isolated mean improvement is **0.005926 ms / 0.58%**, about 15 ms over
2560 calls if it transferred unchanged. The candidate wins 17 of 24 pairs;
paired differences range from −0.007711 to +0.016927 ms, with a standard
deviation of 0.006708 ms. The target GPU time in the model instead increases
by **0.018879 seconds / 0.68%**. These small differences do not establish a
practical improvement. The model timers above include CUPTI instrumentation;
there is no new ordinary wall-time benchmark or claim of an overall regression.

Both generated images are bit-identical to the established native RGB reference:
`c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c`.
This is not a claim of cross-engine identity with ComfyUI.

## What was tested

The original fused kernel uses 1024 threads, shared-memory radix-4 ConvRot
stages and repeated block synchronizations. The isolated screening compares it
with nine alternatives:

- The original implementation with 512 and 256 threads.
- A register/warp implementation with 128, 256, 512 and 1024 threads.
- A vector implementation with 128, 256 and 512 threads, loading `float4`
  gate/up values and writing four packed INT8 outputs per store.

The new implementations keep the same SwiGLU calculation, F32 input/output of
each rotation stage, ordering of additions/subtractions, 1/16 normalization,
row maximum, scale division and round-to-nearest INT8 conversion. Neither BF16
intermediates nor algebraic reassociation were introduced. The original kernel
was extracted verbatim for the standalone reference.

The first register/thread screening shows no consistent benefit. In a second
rotated-order screening the vector-256 candidate is slightly faster at all three
representative row counts (5600, 5657, 5830). A separate 24-pair comparison
therefore tests that fixed candidate before the conditional image experiment.
Each timed side contains 100 launches measured using CUDA events; profiling is
disabled in those timings.

At 5600 rows, the kernel must read 550,502,400 bytes of packed F32 gate/up input
and write 68,812,800 INT8 bytes plus 22,400 scale bytes. Those roughly **619 MB
per call** are unchanged by the variants. Similar performance despite fewer
synchronizations is consistent with memory traffic limiting this case, but no
hardware bandwidth counters were collected, so this is an interpretation rather
than a proven bottleneck. Halving the intermediate with BF16 remains a separate
precision/quality experiment and was not performed.

## Integration and correctness

Only `ggml/src/ggml-cuda/ggml-cuda.cu` changes in the inference source. The
vector-256 path reuses the existing SwiGLU/ConvRot fusion eligibility and
lifetime checks. Additional alignment checks require 16-byte input and 4-byte
output alignment; unsupported cases retain the previous implementation. The
packed output layout, row padding, scales, graph and model precision are
unchanged. No public API, dependency or default profile changes.

- Fourteen isolated cases compare **214,705,564 INT8/scale bytes per candidate**
  exactly for all nine alternatives. Cases cover representative and small row
  counts, 256/768/4096/12288 channels, random values, signed zeros, tiny values,
  large finite values and outliers. Row scales remain finite.
- Ten graph/allocator cases compare **211,455,136 INT8, padding and scale
  bytes**, plus retained/reused F32 results. They exercise partial row padding,
  externally retained/reused SwiGLU results, swapped gate/up, unaligned input
  and unsupported width. CUPTI verifies exactly five eligible vector calls,
  with zero dropped records.
- A small standalone case passes Compute Sanitizer memcheck and racecheck for
  all candidates: zero errors, warnings or hazards. An eligible graph case also
  passes memcheck. This is not full-model sanitizer coverage.
- CLI and server builds pass. Runtime execution uses only the CLI. A checker
  indentation warning was fixed without changing its logic, and its rebuild is
  warning-free. Python syntax and repository whitespace checks pass.

All previous accepted flags are **on in both image runs**, including both BF16
flags, norm/ConvRot, MLP down/residual, attention output/residual and paired Q/K.
Only the vector flag changes. The saved synthetic prompts, model paths,
1600×896, seed 42, 40 ER_SDE/beta steps, CFG 2.5, CPU RNG, Sage2++, 11 CPU
threads, batch one and absence of LoRA/reference/second pass are preserved.
Normalized commands, prompt hashes, binary hashes and environments are checked.

Each profile passes the existing strict analyzer: 80 single-segment forwards,
all expected previous fused dispatches, zero dropped records and zero boundary
crossings. The candidate has 2560 vector calls and zero original fused
SwiGLU/ConvRot calls; the baseline has the reverse. Auto-fit completes without
manual margin. Diffusion compute allocation remains 1167.68 MiB. Whole-GPU
sampled peaks are 15316 and 15317 MiB respectively; these include desktop use.

## Evidence and boundaries

- [Builds, exact source snapshots, scoped patch, synthetic results and validation](../runs/swiglu-f32-build/)
- [Original-kernel model profile](../runs/20261013T195943Z-activity-8v4u8bj0/)
- [Vector-kernel model profile](../runs/20261013T200027Z-activity-kipfq2c9/)
- [Standalone harness](check-swiglu-f32.cu), [candidate kernels](swiglu-f32-kernel.cuh), [graph checker](check-swiglu-f32-graph.cpp).

Frozen CLI SHA256:
`d7569757f905b3a36677aff1c87617f8d83763b0baae2d669e39be8ec518740b`.
Frozen server SHA256:
`10ac406497db1248097549c80d2d87e5db81ac110eb33fed990491d68d8963da`.

The other 32 selected native/Comfy/AIO/Kitchen source hashes remain unchanged.
GPU tests successfully used sandbox escalation. Only approved models, synthetic
prompts, our own benchmark artifacts and selected source were used; private
ComfyUI logs/history/queues/workflows/configuration were excluded. No server
runtime, UI test, commit or push was performed.
