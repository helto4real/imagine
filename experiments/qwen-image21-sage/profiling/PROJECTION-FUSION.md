# Native fused 4096 projection experiment — 2026-10-13

Two full-generation A/B pairs reduced mean sampling from **61.525 to
59.065 seconds (4.0%)**, in addition to the existing gate/up optimization.
Mean process wall time fell from **66.513 to 64.071 seconds (3.7%)**.
All five images, including a separate profiled run, are pixel-identical
to the original reference.

This experiment extends the existing F32 INT8 gate/up kernel to bias-free
K=N=4096 ConvRot projections on SM89. Gate/up remains enabled in both
full-generation variants; the new projection flag is the only A/B change.
Rotation, activation quantization, F32 output, direct SwiGLU, SageAttention2++,
model files, prompt, and sampler configuration remain unchanged.

## Implementation and dispatch

The existing CUTLASS 128x256x64 kernel and scaling epilogue are reused.
The internal helper is now named `ggml_cuda_int8_fused_linear`; the original
source filenames and CMake option are retained for build compatibility.

- `GGML_CUDA_FUSED_INT8_GATE_UP=1`: existing K=4096/N=24576 path.
- `GGML_CUDA_FUSED_INT8_PROJECTION=1`: new K=N=4096 path, **rows >= 512**.
- Both are opt-in, independent, and default off.
- Shorter projections, down projection (K=12288), biased operations, and
  unsupported cases retain the existing cuBLAS/dequantization fallback.

The row floor follows a measured crossover. With unrestricted fusion,
256-row projections took 0.0596 ms with fallback and 0.0689 ms fused;
512 rows took 0.1134 vs 0.0836 ms. The 1024- and 2048-row tests also favored
fusion. The final code therefore avoids the measured short-sequence
regression. This is a conservative measured floor, not an exhaustive search
for the fastest threshold at every row count or on other GPUs.

## Full-output synthetic checks

`compare-fused.py --projection` executes the actual GGML graph with gate/up
held on and projection fusion off/on. Inputs are normal random F32 values,
random INT8 weights, and non-power-of-two per-channel F32 scales, with seed
20261013 and a zero input row. Loading and validation are outside timing.
Each variant uses 20 warmups plus seven synchronized batches of 30 calls.

| Case / rows | Projection off | Projection on |
| --- | ---: | ---: |
| Zero row / 1 | 0.0241 ms | 0.0230 ms, fallback |
| Tail / 3 | 0.0231 ms | 0.0232 ms, fallback |
| Prefix / 57 | 0.0235 ms | 0.0235 ms, fallback |
| Prefix / 230 | 0.0593 ms | 0.0594 ms, fallback |
| Below threshold / 511 | 0.1143 ms | 0.1330 ms, fallback |
| Threshold / 512 | 0.1130 ms | 0.0835 ms |
| Image / 5600 | 1.1932 ms | **0.9184 ms** |
| Positive prefix + image / 5830 | 1.2695 ms | 0.9655 ms |
| Negative prefix + image / 5657 | 1.1968 ms | 0.9241 ms |
| Down control / 5600, K=12288 | 2.6903 ms | 2.6975 ms, fallback |
| Gate/up control / 5600, N=24576 | 4.1681 ms | 4.1698 ms, fused in both |

Every output is checked for finiteness and complete output bytes are compared
by SHA-256: **235,933,696 F32 values**, including **75,370,496** from the new
projection shape. All match bit for bit, as do all quantized inputs and row
scales. The 1-row case checks zero-row behavior; other cases include nonzero
inputs. Full synthetic files are temporary; hashes and reproduction sources
remain. The unchanged kernel dispatch in control cases is separately verified,
so timing variation such as the 511-row case is not a different algorithm.

Separate five-call CUPTI repeats confirm the expected fused/fallback dispatch
for every case, with zero dropped records and no boundary-crossing events.
At 5600 rows, old GEMM plus dequantization costs 0.603 + 0.309 ms; the fused
kernel costs 0.630 ms. ConvRot is unchanged in code, but measured 0.276/0.342 ms
in those separate short traces. Whole-chain median time is **23.0% shorter**.
These are local measurements on a shared desktop GPU with unlocked clocks.

## Full-generation results

The saved reference remains 1600x896, 40 ER-SDE/Beta steps, CFG 2.5,
seed 42, F32 prefix cache, SageAttention2++, and 512 MiB auto-fit margin.
The same new executable is used for both variants. Gate/up is on throughout;
only `GGML_CUDA_FUSED_INT8_PROJECTION` changes. The order is control/on,
then on/control, followed by a separately profiled both-on run.

| Run | Sampling | Process wall time |
| --- | ---: | ---: |
| Gate/up-only control 1 | 61.39 s | 66.456 s |
| Gate/up + projections 1 | 58.45 s | 63.527 s |
| Gate/up + projections 2 | 59.68 s | 64.616 s |
| Gate/up-only control 2 | 61.66 s | 66.571 s |

The pairs save 2.94 s (4.8%) and 1.98 s (3.2%) of sampling, respectively.
The averages above exclude profiling and include both timed runs per variant.
Process time covers startup, loading, sampling, VAE decode, image saving,
and process exit. These are two local pairs on a shared GPU, not a throughput
guarantee. The five images all have decoded RGB SHA-256
`253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2`.

The complete profile verifies 80 forwards, **10240 fused ordinary projections**,
2560 fused gate/up operations, 2560 remaining separate GEMM/dequantization
pairs, and 2560 Sage cores. There are zero dropped records and no activity
crossing the selected boundaries. Profiling run sampling time is 58.84 s.

| GPU activity | Earlier gate/up-only profile | Both fusions |
| --- | ---: | ---: |
| Separate INT8 GEMM | 10.937 s | 4.532 s |
| Separate INT32→F32 dequantization | 4.008 s | 0.764 s |
| New fused ordinary projections | — | 6.737 s |
| Sum of those three rows | 14.945 s | 12.033 s |
| Existing fused gate/up | 9.812 s | 9.864 s |
| Native ConvRot + quantization | 3.891 s | 3.912 s |

The new kernel removes 10240 separate output passes. Remaining separate
dequantization is the down projection. The older profile was recorded earlier
on the same shared GPU; stage attribution supports the mechanism, while the
fresh uninstrumented pairs above establish the observed total-time gain.

Compute Sanitizer memcheck on a **513-row** projection (an incomplete final
tile just above the dispatch threshold) reports **zero errors** after 20
warmups and one measured call. This is a focused memory check, not a claim
of exhaustive coverage for every GPU, shape, bias, or LoRA configuration.

## Artifacts and reproduction

See [build and run commands](README.md#experimental-fused-4096-projections).
Paths are relative to `experiments/qwen-image21-sage/runs/`:

- `20261013T090940Z-fused-2dcrxxje/`: initial unrestricted fusion test.
- `20261013T091032Z-fused-yhm6ky3r/`: 256/512/1024/2048-row crossover test.
- `20261013T091130Z-fused-7qicvru4/`: final thresholded implementation,
  all synthetic checks, dispatch traces, hashes, and timings.
- `projection-fusion-baseline/`: preserved pre-experiment gate/up binary/source.
- `projection-fusion-build/`: final tested binary, sources, build log,
  experiment driver and aggregate full-generation results.
- `20261013T091214Z-control-n6doo8iz/`: first gate/up-only control.
- `20261013T091322Z-control-_cly238e/`: first both-on image.
- `20261013T091426Z-control-lnpl1rj3/`: second both-on image.
- `20261013T091531Z-control-l4tuhjz1/`: second gate/up-only control.
- `20261013T091638Z-activity-4vrccm33/`: complete both-on GPU trace.
- `projection-fusion-memcheck/`: sanitizer log, command, method, and input hashes.

Tested CLI SHA-256:
`33aa286e9ff42e4b6e61af134bed5a1fa64b1d7d98ce32d27ea486ff7d225805`.
Both CLI and server built successfully; the server request path was not
live-tested. Python syntax and changed-code whitespace checks passed.
GPU access escalations were approved. No commit or push was made.
