# Current optimized runtime versus headless AIO/Kitchen

Date: 2026-10-13. RTX 4070 Ti SUPER 16 GB. This is a new measurement of the
combined native optimizations, including both BF16 flags, norm/ConvRot fusion
and MLP down/residual fusion. No inference source or default was changed.

## Main result

| Ordinary measurement | Optimized native CLI | Headless AIO/Kitchen |
| --- | ---: | ---: |
| Sampling, three repeat measurements | **40.600 s** | **37.459 s** |
| Sampling range | 39.900–41.570 s | 37.125–38.035 s |
| Start of each request/process → saved PNG, mean | 45.381 s | 38.821 s |
| Prompt conditioning, mean | 1.890 s | 0.000923 s |
| VAE decoding, mean | 2.007 s | 0.658 s |

The measured sampling gap is **3.141 seconds**, or 8.4% relative to ComfyUI.
These are phase timers around each engine's sampler, including its associated
setup/loading; they are not pure kernel times. The full PNG columns deliberately
describe different lifecycles: every native CLI image starts a fresh process,
whereas AIO retains its process and prompt-encoding cache. AIO still invokes its
model loaders on every request. Native persistent-server performance is untested.

AIO's first image took 43.473 s from request start to PNG, or **46.756 s** from
the harness startup timer through PNG. Its first conditioning took 3.196 s and
first VAE decode 1.228 s. The native mean includes a fresh process on every image
but warm filesystem caches. No OS cache flush or physical-disk fsync was used;
“saved” means successful PNG save. Do not compare the 45.381/38.821 gap as a pure
diffusion-engine difference.

## Protocol and reproducibility

- Same approved synthetic positive/negative prompts, physical model files,
  1600×896, seed 42, 40 ER_SDE/beta steps, CFG 2.5, batch one, no LoRA,
  reference image, second pass or compile. Prompt hashes are verified equal.
- Native: 11 CPU threads, CPU RNG, Sage2++, automatic memory fit with no manual
  VRAM margin. All accepted experimental flags remain on, including both BF16
  flags and both latest fusions. Direct V cast remains off; separate ConvRot
  uses 512 threads. The large fused SwiGLU/ConvRot kernel uses 1024 threads.
- Comfy: 23 CPU threads, Kitchen INT8 attention, Comfy sampling schedule,
  automatic memory/cache device, default cache dtype, privacy mode on,
  zero-negative conditioning off. Observed prefix caches are BF16 on CUDA.
- Order: excluded native warm-up; native sample 1; one Comfy process containing
  first image, three repeats and a separate profiled fifth image; native samples
  2 and 3; separate native profile. No concurrent model workloads were launched.
- Native ordinary sampling: 41.57, 39.90, 40.33 s. Comfy repeat sampling:
  38.034649, 37.217593, 37.124634 s. Profiled images are excluded from means.
- Desktop GPU activity remains present; clocks are unlocked. Three ordinary
  samples and one profile per engine establish local evidence, not a confidence
  interval or guaranteed gain. Instrumentation can perturb GPU timings.

Frozen native CLI SHA256:
`b523eb7c3952ac0cb7bbdb7feed372f98e0e3b7eb568d5fa09dc74e334393a30`.
Native build: CUDA 13.4/SM89. Comfy PyTorch: 2.13.0+cu130, CUDA 13.0;
Kitchen: 0.2.37. Comfy revision `08ff3c11b3a85eb07c2c06f21cb378ac2e086840`;
AIO revision `52ecbf87dda674f51e1e0a798f8f56d0d37d303d`.
Selected local sources, rather than revision identifiers alone, were hashed:
all 33 selected native/Comfy/AIO/Kitchen files are unchanged during this comparison.

## GPU comparison with matching operation boundaries

The new native profile contains **38.852 s** of actual diffusion GPU activities;
Comfy contains **35.998 s**, a 2.854 s difference. Native CUPTI reports zero
dropped records and zero phase-boundary crossings. Comfy GPU activities are
assigned through CUDA launch correlations to CPU scopes; none are unmapped
within diffusion. Torch's exported trace provides no equivalent dropped-record
counter. GPU annotations are never counted as additional GPU work.

Both paths execute 80 diffusion forwards and 2,560 target Sage calls. Comfy runs
2,560 separate target blocks plus 64 small prefix blocks. Native instead runs
64 joint prefix/target blocks on the first two forwards, then **2,496** target
blocks using its cache. Consequently the table compares **GPU milliseconds per
5600-token target block**, excluding prefix construction. It does not pretend
that the full forward graphs have identical work.

| Exclusive chain | Native ms/block | Kitchen ms/block | Native minus Kitchen, scaled to 2496 blocks |
| --- | ---: | ---: | ---: |
| Attention norm + quantization + Q/K/V projections | 2.227 | 2.531 | **−0.759 s** |
| Q/K norm/RoPE + layout + attention preparation/core | 4.016 | 3.779 | +0.593 s |
| Attention output quantization/projection + scale/gated residual | 1.500 | 1.031 | **+1.172 s** |
| MLP norm + quantization + gate/up projection | 4.276 | 4.007 | +0.670 s |
| MLP SwiGLU/ConvRot + down projection + gated residual | 2.943 | 2.657 | +0.713 s |

These normalized deltas are diagnostic, not predicted wall-time savings. Chain
assignment is exclusive. Native Q/K/V GEMMs are interleaved with RoPE/copies,
so identity/shape checks assign them to the first chain instead of using a
single contiguous timing interval. First-two-forward work and small operations
outside the selected blocks remain separately accounted for: 1.508 s native,
0.146 s Comfy. The different exclusion sets make those residual totals unsuitable
for direct comparison.

### Where the remaining time goes

1. **Attention output is the strongest precision-preserving candidate.** In
   2,496 native target blocks its GEMM takes 1.672 s, output ConvRot 0.512 s,
   standalone output scaling 0.714 s and BF16 gated residual 0.828 s, plus small
   gate/setup operations. The scale and residual kernels alone cost **1.543 s**.
   Kitchen's output GEMM takes 0.639 ms/block and gated residual 0.235 ms/block;
   native uses 0.670 + 0.286 + 0.332 ms/block for GEMM, scaling and residual.
   Our MLP fusion has not removed this attention path.
2. **MLP activation traffic remains different.** Profile metadata confirms
   Kitchen gate/up outputs `[1, 5600, 24576]` BF16 and feeds that directly into
   its fused SwiGLU/ConvRot/down path. Native emits F32 gate/up and its quantizer
   reads F32. That intermediate is 525 MiB versus 262.5 MiB. Native gate/up
   takes 3.944 ms/block versus 3.697 ms in Kitchen. SwiGLU/ConvRot takes
   **1.074 ms/block versus 0.638 ms**, an approximately 1.09-GPU-second
   difference at 2,496 blocks. Dtype, quantizer implementation and layout all
   differ; this does not isolate BF16 as the cause. Native fused down/residual
   already takes 1.865 ms/block versus Kitchen's separate down + residual
   at about 2.018 ms/block.
3. **Q/K RMSNorm/RoPE could be paired.** Native launches separate Q and K
   kernels: 4,992 calls, 1.496 s in these target blocks (0.599 ms/block).
   Kitchen's paired call launches one kernel per block: 0.331 ms/block.
   The normalized difference is about 0.671 GPU seconds. Merely sharing a
   launch does not guarantee that speedup; memory access and implementation
   also matter.
4. **Keep the successful pieces.** Native attention normalization/quantization
   is shared across Q/K/V, giving the first chain a measured advantage. Native
   Sage core is also faster here: 2.249 versus 2.456 ms/block. It uses FP8 V
   where Kitchen uses INT8 V, so these are different numerical paths. Native
   layout/conversion and preparation partly erase that core advantage. Changing
   attention algorithm is not the recommended next experiment.

The native profile's diffusion GPU span contains roughly 1.019 s without GPU
activity, versus 0.204 s in Comfy's span. Boundaries/instrumentation differ;
this is not a measurement of Python/C++ overhead or proof that all of it is
recoverable. The source comparison gives more focused targets first.

## Ranked next experiments — proposals, not implemented

**1. Fuse attention output projection → ×32 → gated residual.** Keep all
current flags on. Native `ggml_ext_linear_i8_tensorwise` restores output scale
after the GEMM because the input path uses 1/32. The new epilogue must retain
the existing F32 rounding sequence, including that explicit ×32 operation,
before the final BF16 store. Do not combine scaling factors algebraically or
introduce an FMA that changes rounding. Use the existing MLP down/residual
fusion as a structural starting point, with its alias/liveness/fallback guards.
The 1.543 s of separate work is a candidate budget, not a promised saving.

First validate synthetic shape/boundary/retained-output cases against the
unchanged separate path, then verify full reference RGB identity, expected
dispatch and a paired ordinary timing comparison. A smaller scale-plus-gated
kernel is a fallback if the GEMM fusion is materially more complex.

**2. Pair Q/K RMSNorm/RoPE while preserving their individual arithmetic.**
This is the next candidate if exact native pixel preservation remains the
priority. Benchmark representative BF16 tensors before a model graph change;
check outputs bitwise and profile the entire QKV/attention preparation chain,
because reordering Q/K GEMMs can affect locality and scheduling.

**3. Isolate the MLP SwiGLU/ConvRot gap, then consider BF16 gate/up.** Start
with synthetic matching shapes and identical F32 input to distinguish quantizer
implementation from dtype. Separately test native BF16 gate/up plus a BF16
SwiGLU/ConvRot input path. That would change rounding before nonlinear
activation/quantization and must be treated as a quality/precision experiment,
with image deltas and visual assessment rather than an assumed bit-identity
requirement. Do not silently substitute BF16 to make the first fusion faster.

Outside diffusion, a resident worker/conditioning cache and VAE profiling
could reduce request latency; the current 1.890 s encoding and 2.007 s VAE
phases justify separate work after the diffusion slice.

Source anchors: native
[scaled INT8 linear](../../../stable-diffusion.cpp/src/core/ggml_extend.cpp),
[attention scale setup](../../../stable-diffusion.cpp/src/model/diffusion/qwen_image.hpp),
[Qwen graph](../../../stable-diffusion.cpp/src/model/diffusion/qwen_image_2_1.hpp),
and [CUDA epilogues](../../../stable-diffusion.cpp/ggml/src/ggml-cuda/int8-gate-up.cu).
Installed reference:
[Qwen block/MLP/paired RMSRoPE](/path/to/ComfyUI/comfy/ldm/qwen_image21/model.py:63),
[linear_input_act](/path/to/ComfyUI/comfy/ops.py:1018),
[Kitchen INT8 dtype selection](/path/to/ComfyUI/.venv/lib/python3.12/site-packages/comfy_kitchen/backends/cuda/__init__.py:2035).
Kitchen's `convrot64` name describes its 64-lane implementation, not a change
of model rotation group size: both paths here use ConvRot256 weights.

## Correctness, memory and privacy

All five native images (warm-up, three ordinary, one profile) reproduce native
reference RGB hash `c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c`.
All five Comfy images reproduce its reference RGB hash
`526a2535b24c7517b4787ef77870692a1972ed07058075633972b87b8c62a276`.
These hashes differ: there is **no cross-engine pixel-identity claim**.

Native auto-fit keeps all 6920.55 MiB diffusion weights on CUDA, with 1179.54
MiB diffusion compute allocation and single-segment execution throughout.
Text encoder and VAE weight backing remain on CPU. VAE compute is 10632.62 MiB.
No OOM/manual margin was required. Sampled whole-GPU peaks for native ordinary
runs are 15308–15362 MiB; the entire Comfy session peaks at 15581 MiB.
These include desktop activity and one-second sampling can miss peaks.
Comfy's separate Torch allocated peaks are 12298.17 MiB first image and
9318.12 MiB repeats; reserved memory differs. Do not equate those counters
with total model weights or whole-GPU usage.

Only our benchmark logs/artifacts and selected source were inspected. Private
ComfyUI logs, queues, history, workflows and user configuration were excluded.
The Comfy config/results/trace/log contain neither literal benchmark prompt;
all five PNGs have empty metadata. Tensor metadata contains only dtype, shape
and stride; no tensor values were saved. The native harness keeps its existing
approved synthetic prompt copies. No messages or uploads were sent.

GPU execution required sandbox escalation and completed successfully. No build
was needed; no engine changes, new engine tests, server run, UI acceptance,
commit or push occurred. The four changed/new harness and analyzer files pass
AST parsing; both profiles pass their strict dispatch/accounting checks.

## Evidence

- [Driver, flags, ordinary summary, hashes and matched chains](../runs/current-comfy-comparison/)
- [Fresh Comfy run and trace](../runs/20261013T184654Z-comfy-1fkgzu9r/)
- [Fresh native profile](../runs/20261013T185153Z-activity-ol1uvaxi/)
- Native ordinary runs: `20261013T184607Z-control-4cst8vw4`,
  `20261013T185022Z-control-5nmfmqn4`, `20261013T185107Z-control-sgmdl9io`.
- [Matched-chain analyzer](matched-chains.py),
  [Comfy analyzer](../comfy-headless/analyze.py),
  [comparison driver](../comfy-headless/compare-current.py).
