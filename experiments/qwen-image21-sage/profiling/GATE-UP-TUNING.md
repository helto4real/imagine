# Gate/up CUTLASS configuration experiment — 2026-10-13

**No new configuration adopted.** The existing 128×256×64 tile with three
pipeline stages remains the reference. Smaller tested tiles and two stages
were slower. Four stages showed small, inconsistent changes that did not
establish a useful gain when the closest candidates were repeated in reverse
order. No candidate was advanced to full-image A/B tests.

## Scope

Only the existing fused INT8 gate/up path (K=4096, N=24576, SM89) selected the
experimental configurations. Ordinary projections retained the existing
128×256×64, three-stage kernel. The INT8 Tensor Core instruction shape
16×8×32, warp shape 64×64×64, F32 output, rounding, row/channel scale order,
ConvRot, and previously accepted fusions were unchanged.

The temporary startup flag was `GGML_CUDA_GATE_UP_CONFIG`, with six choices:

- `128x256x64-s3`: original configuration.
- `128x128x64-s3`, `64x256x64-s3`, `64x128x64-s3`: alternative tile sizes.
- `128x256x64-s2`, `128x256x64-s4`: alternative pipeline depths.

All six variants compiled and executed. This flag exists only in the preserved
experimental binary. It is not supported by the restored active engine.
This was a bounded configuration comparison, not an exhaustive CUTLASS search;
warp shapes, instruction shapes, precision, and attention were not changed.

## Method and validation

The native harness runs the actual GGML ConvRot → INT8 GEMM → F32 output graph,
with M=5600, 5657, and 5830 matching steady image tokens and the reference's
two prefix-plus-image sizes. All variants read identical synthetic F32 input,
INT8 weights, and non-power-of-two channel scales. A zero row, tiny values,
large finite values, and an outlier are included. Models are not loaded.

Each variant performs 20 warmups and seven synchronized batches of 30 calls.
Times below are medians of batch means, including GGML dispatch and excluding
loading, setup, validation, and initial transfers. Each shape also runs a
harness linked to the pre-experiment engine. Variant order rotates between
shapes; the second sweep reverses the selected configuration order.

All quantized values, row scales, and F32 output values match the preserved
original bit for bit. Each configuration checked across the three shapes
covers **419,930,112 output F32 values**, **69,988,352 INT8 values**, and
**17,087 scales**. F32 outputs and scales are finite. Comparisons include the
new executable's default configuration, not just candidate-to-candidate checks.

Separate CUPTI runs use 20 warmups plus five measured calls. They verify the
actual tile, pipeline depth, grid, block size, and five fused GEMM launches
without fallback. Every trace has zero dropped records and no activities
crossing timing boundaries; traced outputs also match the timed outputs.

Hardware/build: RTX 4070 Ti SUPER (SM89), CUDA 13.4.92, local CUTLASS 4.5.0,
Release build. GPU clocks remain unlocked and the desktop shares the GPU.
K=4096 ConvRot stays at 512 CUDA threads; existing fusion flags stay enabled.
The harness is a single Linear benchmark, not a full FFN or image benchmark.

## First sweep

Whole-chain milliseconds per call:

| Configuration | M=5600 | M=5657 | M=5830 |
| --- | ---: | ---: | ---: |
| Preserved original executable | 4.1666 | 4.1927 | 4.2906 |
| **128×256×64, 3 stages** | **4.1831** | **4.1878** | **4.2704** |
| 128×128×64, 3 stages | 4.2580 | 4.3170 | 4.3387 |
| 64×256×64, 3 stages | 4.5909 | 4.6389 | 4.7591 |
| 64×128×64, 3 stages | 5.0945 | 5.2201 | 5.4038 |
| 128×256×64, 2 stages | 4.6241 | 4.3571 | 4.4094 |
| 128×256×64, 4 stages | 4.1138 | 4.2029 | 4.3717 |

The four-stage candidate was 1.66% faster at M=5600, but slower at both prefix
sizes. The smaller tiles and two-stage variant did not improve whole-chain
latency in this sweep. Separate kernel timings varied too and are preserved
in each case's summary; they are not substituted for whole-chain timings.

## Reverse-order confirmation

| Configuration | M=5600 | M=5657 | M=5830 |
| --- | ---: | ---: | ---: |
| Preserved original executable | 4.1527 | 4.1781 | 4.2819 |
| **128×256×64, 3 stages** | **4.1587** | **4.1841** | **4.3710** |
| 128×128×64, 3 stages | 4.1478 | 4.3166 | 4.3206 |
| 128×256×64, 4 stages | 4.1504 | 4.1927 | 4.2690 |

At the dominant M=5600 size the four-stage advantage shrank to **0.20%**,
only about 0.008 ms per call. At M=5830 it changed from a 2.37% loss to a
2.34% gain against the new executable's baseline, whose own timing shifted
from 4.2704 to 4.3710 ms; the preserved original measured 4.2819 ms on that
repeat. The smaller near-candidate likewise alternated between slight gains
and losses. These observations do not establish a repeatable useful gain.
They do not prove the configurations are mathematically equal in performance.

All compiled variants use 226–242 registers per thread, with no stack/local
spills reported by cuobjdump. This is compiler resource evidence, not measured
occupancy or a hardware-counter diagnosis of the limiting resource. Dynamic
shared-memory usage is not represented by cuobjdump's static SHARED:0 field.

## Disposition and limits

The experiment is retained as frozen source, patch, binaries, input hashes,
full-output comparisons, telemetry, and separate kernel traces. The active
helper source is restored byte for byte to its pre-experiment contents;
previous source files for SwiGLU/ConvRot, the original ConvRot tuning, and
Qwen's packed SwiGLU remain unchanged.

The existing 59.61 s mean start-to-PNG reference remains the latest accepted
image measurement; it was **not remeasured in this experiment**. No candidate
passed the performance gate for image testing, so no candidate image latency
or image-quality result is claimed. The rebuilt reference is checked separately
with one image to validate restoration; that is not a timing comparison. No sanitizer test was added for these
rejected configurations. Both experimental and restored CLI/server targets built; server
requests were not live-tested. GPU execution required approved sandbox
escalation. Existing unused-parameter warnings from the CUTLASS headers remain non-fatal.
No commit or push was performed.

The restored CLI's control image matches the established RGB reference exactly
(SHA-256 `253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2`).
Its new executable SHA-256 is
`142ccc876d5f0c5e3c265f60030338ac82c558e4a9bdaec646f760a4d03da4d7`.
This single restoration check is excluded from performance comparisons.

## Artifacts and reproduction

Paths relative to `experiments/qwen-image21-sage/runs/`:

- `20261013T112433Z-gate-up-configs-a30zheef/`: first six-configuration sweep.
- `20261013T112558Z-gate-up-configs-yr38dq0q/`: reverse-order confirmation.
- `20261013T112939Z-control-52fhxur8/`: restored-runtime control image.
- `gate-up-tuning-build/`: frozen before/experimental source and executables,
  `candidate.patch`, executable hashes, build logs, and compiler resources.

The benchmark script uses those frozen experimental files, even after the
active engine is restored:

```sh
python experiments/qwen-image21-sage/profiling/compare-gate-up-configs.py
python experiments/qwen-image21-sage/profiling/compare-gate-up-configs.py \
  --configs 128x256x64-s3 128x128x64-s3 128x256x64-s4 --reverse
```
