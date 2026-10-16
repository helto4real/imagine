# Qwen INT8 LoRA fusion attribution — 2026-10-16

The user requested the same CFG 1 benchmark, with no optimization changes, to
identify lost fusions and their contribution to the slowdown. The user explicitly
approved reading only newly collected CUDA kernel names/counts/timings and copy
measurements. No server logs, submitted prompt records, job history, images or
image metadata were read. Native output and response bodies went to `/dev/null`.

## Controlled run

Same inputs as [the unprofiled comparison](LORA-CFG1.md): the source-defined
adult-woman portrait, only `lenovo_qwen21.safetensors` at strength 1, Qwen INT8,
CFG 1, seed 42, 1600×896, 40 ER-SDE/beta steps, no references. Same frozen v2
executable (SHA-256 `9fd49b0e5cb3426043ebd85271ad167555551ac65252590cde5ac25bed44bbb0`),
all 29 frozen optimization settings, auto-fit and conditioning cache. One owned
server process handled all eight requests; no existing service was stopped.

Only a diagnostic CUPTI preload was added. It records concurrent GPU activities,
without per-launch synchronization, and uses CLOCK_MONOTONIC like the client
request boundaries. An explicit post-response checkpoint flushed the profile
before terminating the owned server. Dropped records: **0**. Every request
returned HTTP 200. The diagnostic library/harness changed; native source,
production configuration and frozen binaries did not.

| Request | Client seconds |
| --- | ---: |
| Without, first | 23.758 |
| Without, warm 1 | 20.470 |
| Without, warm 2 | 20.482 |
| With, first | 45.319 |
| With, warm 1 | 44.879 |
| With, warm 2 | 44.705 |
| Removed, first return | 20.893 |
| Removed, warm return | 20.705 |

Warm means **20.476 → 44.792 s**, **+24.316 s / 2.188×**. The preceding
unprofiled series was 19.926 → 44.493 s (+24.567 s / 2.233×). Profiled wall times
were approximately 2.8% and 0.7% higher respectively; this cross-run difference
includes normal variation, so it is not an isolated profiler-overhead estimate.

## Fusions actually lost

Counts are per warm request and identical across each mode's two repeats.

| CUDA path | Without LoRA | With LoRA | Replacement/observation |
| --- | ---: | ---: | --- |
| INT8 gate/up + SwiGLU | 1,280 | 0 | F32 INT8 output, LoRA addition, standalone SwiGLU |
| INT8 Q/K/V projection + BF16 output | 3,840 | 0 | F32 output followed by LoRA and separate BF16 conversion |
| INT8 attention output + scale + gated residual | 1,248 | 0 | Separate output scale and gated residual |
| INT8 MLP down + gated residual | 1,248 | 0 | Separate gated residual |
| BF16 norm + modulation + ConvRot | 2,496 | 0 | Norm/modulation remains fused; ConvRot is separate |
| BF16 attention input cast/scale + ConvRot | 1,280 | 0 | BF16→F32, scale, F32 ConvRot |
| Paired Q/K RMSNorm + RoPE | 1,280 | 1,280 | Preserved |
| Sage attention core | 1,280 | 1,280 | Preserved |

32 layers × 40 steps gives 1,280 layer visits. Some first-step prefix paths
already lack residual/norm fusions, explaining 1,248 = 32 × 39, rather than
1,280. Ordinary fused INT8 GEMM/dequantization remains enabled: 7,680 F32-output
projection calls replace the more specialized output forms. LoRA does not
simply disable every optimized kernel or Sage attention.

The model's Linear path adds LoRA's output with `ggml_add_inplace`. This breaks
`MUL_MAT → GLU`, `MUL_MAT → CPY`, and the projection→residual patterns required
by the specialized fusions. LoRA also consumes the original floating-point
input, preventing the norm/ConvRot and cast/scale/ConvRot chains from eliminating
that input. The shared SwiGLU output is needed by LoRA's down projection, so the
alternative SwiGLU→ConvRot fusion cannot eliminate it either.

## Time attribution

Mean differences of summed GPU kernel durations, with disjoint categories:

| Observed extra work | Extra GPU seconds |
| --- | ---: |
| Replacement work for lost fusions, net | **4.691** |
| LoRA output additions: +7,680 calls | 6.718 |
| LoRA strength/linear scaling: +12,800 calls | 5.301 |
| FP16↔FP32 conversions: +15,360 in each direction | 5.829 |
| Additional GEMMs: +15,360 calls | 1.848 |
| Other existing kernels, including Sage | −0.204 |
| **Total extra kernel duration** | **24.183** |

LoRA-side work totals **19.697 GPU s**. Of the +24.316 s wall difference,
approximately **19%** corresponds to the net fusion-replacement bucket and
**81%** to LoRA-side work. Small changes in existing kernels and host gaps account
for the remaining balance. This is source-grounded trace accounting, **not an
ablation or a guarantee of the speedup from restoring a fusion**. A redesigned
fusion could also absorb some LoRA-side work; these buckets do not bound its
potential benefit.

The 4.691 s fusion-replacement bucket includes:

- All specialized/ordinary INT8 projection kernels plus gate/up packing, net
  **−0.239 s**. Disappeared fused GEMM times are subtracted, not counted as losses.
- Separate SwiGLU **+1.808 s** and gated residual **+0.819 s**.
- BF16 output/input conversions **+1.254 s**.
- Norm/modulation/ConvRot replacement chains, net **+0.283 s**.
- Attention base input/output scales **+0.767 s**.

The two attention scale chains are identified by per-stream execution order:
BF16→F32 / scale / F32 ConvRot, and INT8 GEMM / output scale. This agrees with
the unchanged source. Counts are 0→1,280 input scales and 32→1,280 output scales;
the remaining additional scales are exactly 12,800. Without this separation,
counting every extra scale as LoRA work would understate fusion replacement.
The trace has no graph-node labels: classification uses kernel signatures,
launch counts, sequence and source, not direct per-node instrumentation.

The summed kernel durations were **19.211 → 43.394 s**. The union of all measured
GPU activity intervals, counting overlaps once, was **19.539 → 43.732 s**.
This closely tracks client wall time while distinguishing GPU activity from
host/launch/HTTP gaps.

Warm transfer counts and bytes are identical in both modes:

| Transfer | Without / with |
| --- | ---: |
| Host→device | 397 calls / 1,767.236 MiB |
| Device→host | 41 calls / 76.563 MiB |
| Device→device | 6,770 calls / 46,613.855 MiB |

Combined copy-duration difference is about **+0.009 s**. The measured slowdown
is not increased warm host/device weight traffic. This does not measure peak
VRAM. The small 76 MiB LoRA still requires operations on full activations: at
5,600 image tokens, the F32 24,576-wide gate/up output is **525 MiB**. Repeated
conversion, scaling and addition over such outputs explains why file size is a
poor predictor of execution overhead.

## Source and reproducibility

Relevant sources:

- `src/model/common/ggml_block.hpp`: INT8 Linear and adapter output sum.
- `src/model/adapter/lora.hpp`: down/up projections, output scaling and addition.
- `src/model/diffusion/qwen_image_2_1.hpp`: fusion dependencies and shape guards.
- `src/core/ggml_extend.cpp`: attention input/output scale handling.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: fusion matching and FP16/F32 GEMM conversion.
- `ggml/src/ggml-cuda/int8-gate-up.cu`: specialized INT8 epilogues.

CUDA fusion/epilogue, Linear and scale source hashes match the frozen v2
manifest. LoRA and parent Qwen attention sources match the pinned upstream.
The current Qwen model file also contains newer reference-cache work; the v2
version was independently restored in `/tmp` from the pinned upstream plus the
published patch and verified against the frozen hash before attribution.

Build `activity.cpp` as described in [README](README.md), adding `-pthread`.
Run `compare-lora-cfg1.py` with the same `--model-root`/`--lora`, `--run`, and
`--trace-library /absolute/path/activity.so`. The optional tracer requires an
explicit exception to the project's runtime-data policy. Then run
`summarize-lora-fusions.py RUN_DIRECTORY`. Do not use the historical replay
script, which reads other runtime sources.

Local approved trace: `runs/lora-cfg1-mt0yajgf/`, ignored by Git. Compact numeric
results are preserved in [LORA-FUSION-COUNTERS.json](LORA-FUSION-COUNTERS.json).
Validation: tracer built with `-Wall -Wextra`; Python syntax and numerical
interval/classification checks passed; live trace completed with zero dropped
records and all eight HTTP responses successful. Images were not inspected.
The GPU subprocess used approved sandbox escalation. Report and diagnostic
tooling changes are uncommitted; no push occurred.
