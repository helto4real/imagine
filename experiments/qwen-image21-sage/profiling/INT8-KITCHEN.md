# Matched INT8 ConvRot Linear comparison — 2026-10-13

Kitchen's combined INT8 GEMM/output path substantially outperforms the current
GGML chain for the ordinary projection and combined gate/up projection, even
with **F32 input and output**. For these two shapes its forced separate-output
fallback is close to GGML. This supports targeting the output epilogue and
shape-specific GEMM choice, rather than changing all activations to BF16 first.
No inference source was changed in this experiment; the prior direct-SwiGLU
optimization remains in place.

## Method

All tests use 5,600 rows, matching the image-token count at 1600 x 896:

| Case | Input channels K | Output channels N |
| --- | ---: | ---: |
| Ordinary projection | 4,096 | 4,096 |
| Combined gate/up | 4,096 | 24,576 |
| Down projection | 12,288 | 4,096 |

For each case, all variants read identical synthetic F32 activations, INT8
weights already expressed in the rotated basis, and F32 weight scales from
the same temporary files. SHA-256 hashes are retained in the manifest. Input
values are random integer multiples of 1/64 in [-125/64,125/64], exactly
representable in both F32 and BF16. The BF16 variant verifies the entire
converted input against the original. Weight scale is 1/128, group size 256,
with no bias, LoRA, residual, or activation before the Linear call.

The native benchmark calls the actual bundled GGML CUDA graph operations
`ggml_quantize_i8_convrot` and `ggml_mul_mat_i8_tensorwise`. Kitchen calls its
installed public `int8_linear` API under `use_backend('cuda')`. The fallback
control disables Kitchen's CUTLASS output path only inside that test process.
No installed package or ComfyUI source files are modified.

Each uninstrumented measurement warms up 20 calls, then measures seven batches
of 30 calls, synchronizing before the first batch and at each batch end.
Reported times are medians of batch means, including normal host dispatch and
per-call allocations, but excluding model/input loading, initial transfers,
graph construction, warmup, and validation reads. Weights and inputs stay on
the GPU. GGML graph storage is preallocated; Kitchen uses its normal warmed
PyTorch allocator. No CUDA Graph replay is enabled.

Separate CUPTI runs use 20 warmups and five measured calls. Untimed wrapper
hooks record Kitchen's actual low-level dispatch and return status; these
hooks do not run during timing. GPU timestamps confirm the selected kernels.
All 15 traced variants have zero dropped records and no activity crossing
the selected measurement boundaries.

## Whole-chain timings

Milliseconds per call; all columns include rotation/quantization, GEMM, and output:

| Case | GGML F32→F32 | Kitchen F32→F32 | Kitchen F32→BF16 | Kitchen BF16→BF16 | Kitchen fallback F32→F32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Projection | 1.2255 | **0.8701** | 0.8364 | 0.7827 | 1.1920 |
| Gate/up | 5.7880 | **4.1753** | 3.9870 | 3.8915 | 5.7189 |
| Down, first run | 2.7204 | 2.7235 | 2.4626 | 2.3847 | 2.6973 |
| Down, reverse-order repeat | 2.7701 | 2.4625 | 2.4878 | 2.3841 | 2.6759 |

At matched F32 input/output, projection time is **29.0% shorter** and gate/up
time **27.9% shorter** with Kitchen. Within Kitchen, the combined-output path
is about **27% shorter** than its forced fallback for both shapes, using the
same libraries and quantizer. This comparison changes GEMM implementation and
epilogue together, so it does not isolate epilogue arithmetic alone.

The down F32 result is less stable: the first batch-mean range was
2.496–2.768 ms, while the reverse-order repeat median was 2.4625 ms. Across
the two paired run medians the gain is approximately 0–11%; do not claim a
fixed down-projection speedup from this evidence. BF16→BF16 down medians were
stable at about 2.384 ms. Full per-batch samples are in `summary.json`.

## Where GPU time goes

These are stage times from separate CUPTI repeats, not components of the
uninstrumented medians above. Values are milliseconds per call:

| Case / F32 route | ConvRot + quantization | GEMM | Separate dequantization | GEMM + fused output |
| --- | ---: | ---: | ---: | ---: |
| Projection, GGML | 0.279 | 0.603 | 0.309 | — |
| Projection, Kitchen | 0.209 | — | — | 0.639 |
| Projection, Kitchen fallback | 0.219 | 0.617 | 0.295 | — |
| Gate/up, GGML | 0.352 | 3.635 | **2.006** | — |
| Gate/up, Kitchen | 0.211 | — | — | **3.977** |
| Gate/up, Kitchen fallback | 0.219 | 3.637 | 1.845 | — |
| Down, GGML | 0.658 | 1.748 | 0.367 | — |
| Down, Kitchen | 0.592 | — | — | 1.848 |
| Down, Kitchen fallback | 0.651 | 1.807 | 0.275 | — |

For gate/up, native GEMM itself is faster than Kitchen's combined GEMM/output
kernel, but the additional **2 ms output pass** more than removes that advantage.
Kitchen's quantizer is also faster, but explains a smaller part of the gate/up
difference. The trace confirms Kitchen's `cutlass_int8_dequant` returns true
and runs `DefaultGemmWithVisitor` kernels. The fallback instead calls
`cublas_gemm_int8` followed by `dequantize_int8_linear`. GGML and Kitchen's
fallback use the same named `cutlass_80_tensorop_i16832gemm_s8_256x128_64x3_tn_align16`
GEMM family selected by their respective cuBLAS versions.

## Numerics

Validation reads the first eight complete rows of quantized activations,
their scales, and Linear output for each case. It is a sampled comparison,
not an exhaustive check of every output element or an image-quality test.

- Kitchen F32→F32 and its forced fallback match GGML **bit for bit** in all
  sampled INT8 values, row scales, and F32 outputs. Across three shapes this
  covers 163,840 quantized values, 24 scales, and 262,144 outputs per variant.
- Keeping F32 input but writing BF16 leaves sampled quantized inputs unchanged;
  relative L2 output difference is about **0.165%**, consistent with changed
  output precision in this test.
- Kitchen BF16 input changes **8.16–9.03%** of sampled quantized values despite
  exactly matching input values before rotation. Relative L2 output difference
  is **0.84–0.91%**. Thus this route changes intermediate numerical behavior;
  its benefit cannot be attributed solely to storing the same result in BF16.
  The exact internal rounding mechanism was not traced here.
- All profiled samples match their corresponding uninstrumented samples.

These specially controlled synthetic inputs and power-of-two scales do not
guarantee bit-identical results for arbitrary model activations and scales.

## Environment and limits

- GPU: RTX 4070 Ti SUPER, SM89; shared desktop GPU, no clock locking or power changes.
- Native: existing pinned GGML build, CUDA runtime **13.4.92**, cuBLAS **13.8.0.4**.
- Kitchen **0.2.37**, PyTorch **2.13.0+cu130**, packaged CUDA runtime **13.0.96**,
  cuBLAS **13.1.1.3**. Loaded library paths were captured from `/proc/self/maps`.
- Different library versions and host frameworks remain confounders for the
  cross-engine comparison. Kitchen's combined/fallback comparison controls
  those factors within the same installed environment.
- This is a warmed single-Linear benchmark. It does not reproduce full-model
  cache pressure, Q/K/V quantization reuse, prefix creation, attention, or
  Kitchen's fused SwiGLU/residual opportunities. Do not multiply these timings
  into a promised full-generation speedup or claim the complete ComfyUI gap
  is explained.

**Recommended next implementation experiment:** a shape-targeted fused INT8
output path for gate/up, initially retaining F32 output and the existing
cuBLAS path as a fallback. Compare numerical errors on actual activations and
full generation before adoption. Kernel selection matters; use the down
results as a reason to avoid assuming one fused kernel wins for every shape.
This implementation was not attempted in this task.

## Artifacts

Under `experiments/qwen-image21-sage/runs/`:

- `20261013T083551Z-linear-r0k4nzdl/`: complete five-variant/three-shape suite,
  separate CUPTI traces, dispatch records, source snapshots, binary, input
  hashes, timings, sampled tensors, telemetry, `summary.json`, and `kernels.csv`.
- `20261013T083827Z-linear-endcf6kr/`: reversed-order down timing repeat.

Synthetic full inputs were temporary and removed after each case; their seed,
generation method, and hashes remain. No model weights or private prompts
were used. The benchmark build and Python syntax checks passed. GPU access
escalations were approved. No commit, push, or inference change occurred.
