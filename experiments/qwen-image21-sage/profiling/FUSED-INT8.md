# Native fused INT8 gate/up experiment — 2026-10-13

Two uninstrumented A/B pairs saved 2.00 and 3.02 seconds of sampling
(3.0% and 4.6%). Mean sampling time fell from 65.595 to 63.085 seconds,
about 3.8%. All five generated images, including the separately profiled
run, match the earlier reference pixel for pixel. The optimization is useful,
but does not close the full ComfyUI performance gap.

The experimental path combines INT8 GEMM and F32 output scaling in one native
CUTLASS kernel. It targets only the bias-free Qwen gate/up shape on SM89:
K=4096, N=24576, ConvRot group size 256. Native activation rotation and
quantization, F32 output, direct SwiGLU, attention, sampling, and model files
are unchanged. Unsupported cases retain cuBLAS plus separate dequantization.

This is an opt-in experiment. Build with
`GGML_CUDA_INT8_GATE_UP_CUTLASS_PATH` pointing to CUTLASS headers and set
`GGML_CUDA_FUSED_INT8_GATE_UP=1` in the process environment. Unset or `0`
retains the previous path. The executable has no Kitchen or Python runtime
dependency. The local headers identify themselves as CUTLASS 4.5.0; an
unchanged, hashed copy is archived under `runs/cutlass-4.5.0/`.

## Implementation

`ggml/src/ggml-cuda/int8-gate-up.cu` uses an SM80 INT8 Tensor Core kernel on
SM89, a 128x256x64 threadblock, 64x64x64 warps, and three mainloop stages.
The visitor epilogue converts INT32 accumulators to F32, multiplies by the
activation row scale and then the weight channel scale, and writes F32.
That preserves the fallback's arithmetic order. The full INT32 output
buffer and its separate read/scale/write pass disappear for this shape.
At 5600 rows that buffer would be 525 MiB; this is an avoided allocation,
not a promise that whole-process VRAM falls by that amount.

The existing fallback is selected before any fused launch if the experiment
is disabled, the device/shape is unsupported, bias is present, or CUTLASS
rejects the arguments. A failure after selecting the fused launch is reported
as an error. No inference request/API schema or installed ComfyUI files changed.

## Synthetic validation and timing

`compare-fused.py` runs the actual GGML graph with fusion off/on in separate
processes, with identical normally distributed F32 inputs, random INT8 weights,
non-power-of-two per-channel scales, and a zero input row. Seed: 20261013.
Each timing uses 20 warmups and seven synchronized batches of 30 calls.
Loading, transfers, and output validation are excluded. Separate five-call
CUPTI repeats verify dispatch and have zero dropped records and no events
crossing timing boundaries.

| Rows / case | Fallback median | Fused median |
| --- | ---: | ---: |
| 1, zero row | 0.1860 ms | 0.1758 ms |
| 57 | 0.2006 ms | 0.1896 ms |
| 230 | 0.2884 ms | 0.2318 ms |
| 5600 | 5.9560 ms | 4.2328 ms |
| 5830 | 5.9795 ms | 4.2917 ms |
| 5657 | 5.8280 ms | 4.1932 ms |
| 5600, unsupported K=N=4096 | 1.1873 ms | 1.1899 ms, fallback verified |

All **449,945,600 F32 output values** are finite and their complete byte
streams match by SHA-256. All quantized input values and row scales also
match. This includes odd prefix lengths, not just aligned 5600-row inputs.
The one-row case checks zero-row behavior; it is not a general one-row
numerical test. Full temporary tensors are deleted after hashing; inputs
are reproducible from the script and recorded input hashes.

The steady gate/up trace measures ConvRot at 0.290/0.292 ms, old GEMM at
3.534 ms plus 1.842 ms dequantization, and fused GEMM/output at 3.837 ms.
Its whole-chain median is 28.9% shorter. Some timed batches were slower
(fallback 5.693–7.274 ms, fused 4.177–5.242 ms); this is a shared desktop
GPU with unlocked clocks. Full-generation results are the relevant outcome.

## Full generation

The saved reference uses 1600x896, 40 ER-SDE/Beta steps, CFG 2.5, seed 42,
F32 prefix cache, SageAttention2++, and a 512 MiB auto-fit margin. The same
binary is used for off/on comparison, with only the fusion environment flag
changed. Model loading and VAE decode are included in process wall time.

| Run | Sampling | Engine generation | Process to exit |
| --- | ---: | ---: | ---: |
| Fusion on, first | 64.21 s | 68.25 s | 69.328 s |
| Fusion off, control | 66.21 s | 70.15 s | 71.247 s |
| Fusion on, repeat | 61.96 s | 65.93 s | 67.007 s |
| Fusion off, repeat | 64.98 s | 69.21 s | 70.324 s |

Paired sampling reductions are 2.00 and 3.02 seconds. Mean process wall time
falls from 70.786 to 68.167 seconds (3.7%). With only two runs per variant
and visible variation on the shared GPU, these are local observations rather
than a stable throughput guarantee. The first fused run is included, not
discarded as an outlier. The profiled run is excluded from these averages.

All test images, including the profiled run, match the earlier reference pixel for pixel, decoded RGB SHA-256
`253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2`.

The separate full-model CUPTI trace confirms 80 forwards, **2560 fused
gate/up launches**, 12800 remaining separate GEMM/dequantization pairs,
2560 Sage core launches, zero dropped records, and no boundary crossings.
Fused gate/up occupies **9.812 s** of GPU time. The earlier direct-SwiGLU
profile spent **9.370 s GEMM + 4.900 s dequantization = 14.270 s** on the
same gate/up shapes. This is a 4.458 s reduction for that path in those
two profiles. The rest of ConvRot is unchanged (3.935 vs 3.891 s).
Those profiles were taken at different times on a shared GPU; they explain
the mechanism but do not replace an uninstrumented total-time comparison.

## Reproduction and artifacts

See [build and run commands](README.md#experimental-fused-gateup).
Paths below are relative to `experiments/qwen-image21-sage/runs/`:

- `20261013T085614Z-fused-4l0w0s_a/`: full-output synthetic A/B checks,
  input/output hashes, timing samples, per-case traces, and source snapshots.
- `20261013T085658Z-control-q9h6xcn8/`: first fused image.
- `20261013T085818Z-control-0le00yl1/`: fusion-off control image.
- `20261013T085945Z-activity-pxub23nc/`: full fused GPU activity profile.
- `20261013T090128Z-control-492rqbct/`: uninstrumented fused repeat.
- `20261013T090328Z-control-xrokd_fo/`: uninstrumented fusion-off repeat.
- `fused-int8-memcheck/`: Compute Sanitizer memcheck, 57-row gate/up,
  20 warmups plus one measured call, **zero errors**; method and input hashes retained.
- `fused-int8-baseline/`: preserved pre-experiment CLI and Linear binary.
- `fused-int8-build/`: tested fused CLI, source additions, tracked GGML diff,
  configure/build logs. CLI SHA-256:
  `b488096e70f51ec1479876c60abd0f6197d0b0315008994ff2d849968cf578fe`.
- `cutlass-4.5.0/`: archived headers and per-file SHA-256 manifest.

Both CLI and server built successfully. The server's request path was not
live-tested. The optional translation unit also compiled with the feature
disabled and without CUTLASS include paths; a separate complete build with
the CMake option disabled was not run. Python syntax and whitespace checks passed. GPU access escalations
were approved. No commit or push was made. The optimization remains disabled
by default while retaining the previously accepted direct-SwiGLU change.
