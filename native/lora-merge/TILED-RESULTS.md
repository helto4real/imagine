> Historical 64-row candidate, superseded by [whole-matrix results](RESULTS.md).

# Qwen INT8 / ConvRot LoRA merge — 2026-10-16

The opt-in native merge reduces warm LoRA request time from **38.041 to
21.672 seconds (43.0%)** on the matched workload. ComfyUI headless takes
20.820 seconds; native is 0.852 seconds / 4.1% slower. The flag remains off
by default and the production engine selection is unchanged.

## Matched workload and measurements

Qwen Image 2.1 INT8/ConvRot diffusion and text encoder, BF16 VAE file,
1600×896, 40 ER-SDE/beta steps, CFG 1, seed 42, batch one, no reference images,
empty negative prompt. The source-defined portrait prompt describes an adult
woman in a blue shirt. The sole LoRA is `lenovo_qwen21.safetensors`, strength 1.
All frozen v2 optimization switches remain; both native modes also enable the
previous `GGML_CUDA_FUSED_LORA_OUTPUT=1` implementation.

Each series uses eight requests: three without LoRA, three with LoRA, then two
without it again. Warm means use the second and third requests in each initial
group. Both final native series use the exact same executable. Comfy, native
merge-on, and final native merge-off run sequentially with the GPU idle gate.
All three use the same CUPTI kernel/copy profiler; no dropped records are reported.

| Path | Without LoRA, warm mean | With LoRA, warm mean | First request after enabling LoRA |
| --- | ---: | ---: | ---: |
| ComfyUI headless AIO/Kitchen | 21.063 s | 20.820 s | 21.282 s |
| Native, merge off | 19.916 s | 38.041 s | 38.579 s |
| Native, merge on | 19.886 s | 21.672 s | 32.381 s |

Native merge-on warm LoRA samples are 21.6729 and 21.6712 seconds; merge-off
samples are 38.0336 and 38.0477 seconds. After removing LoRA, final warm requests
return to 19.920 and 19.904 seconds respectively. All requests complete with HTTP
200. Two warm samples per mode establish this local result, not a performance
guarantee across workloads or machines.

Comfy timing includes PNG encoding in memory; native timing includes its HTTP
response and base64 handling. Native retains its CPU RNG and inference pipeline;
matching seed and sampler names does not imply matching pixels. Neither generated
images nor ordinary model/server logs were inspected.

## Verified Comfy precision path

The tested headless entry point selects ordinary `ModelPatcher`, not the dynamic
patcher selected by other Comfy entry points. Runtime assertions confirm FP16
unrotated base weights, FP32 LoRA matrix product/intermediate arithmetic, FP16
delta/addition, and FP16 input to stochastic INT8/ConvRot requantization with
recalculated scales and group size 256. The quantizer seed comes from CRC32 of
the full Comfy weight key. Ordered patches are applied before quantizing once.

Source locations are Comfy `model_patcher.py:patch_weight_to_device`,
`weight_adapter/lora.py:LoRAAdapter.calculate_weight`, and
`ops.py:mixed_precision_ops.Linear.set_weight`, followed by Kitchen's INT8 layout
and CUDA quantizer. The installed versions used here are Comfy revision
`ba994879b88d0ebd9950b1e5c1e923d928f216ba`, AIO revision
`52ecbf87dda674f51e1e0a798f8f56d0d37d303d` with its existing local changes,
Torch 2.13.0+cu130 and Kitchen 0.2.37. Installed Comfy/AIO source was not modified.

In each measured Comfy LoRA request, CUDA records show 192 FP16 ConvRot
dequantizations, 192 FP32 matrix products and 192 stochastic FP16 ConvRot
requantizations. Their combined kernel time is about 0.13 seconds. This is the
sum of these selected kernels, not the total wall cost of applying LoRA.

## Native fusion and preparation evidence

Representative inference kernel calls per warm image:

| Kernel family | No LoRA, merge on | LoRA, merge off | LoRA, merge on |
| --- | ---: | ---: | ---: |
| INT8 gate/up + SwiGLU | 1,280 | 0 | 1,280 |
| INT8 attention + residual | 1,248 | 0 | 1,248 |
| INT8 down + residual | 1,248 | 0 | 1,248 |
| INT8 BF16 output | 3,840 | 0 | 3,840 |
| Norm/modulation + ConvRot | 2,496 | 0 | 2,496 |
| Runtime fused LoRA output | 0 | 7,680 | 0 |

Native warm kernel time falls from about 36.701 to 18.753 seconds with LoRA;
the merge-on no-LoRA mean is 18.657 seconds. The union of measured GPU activity
is about 19.287 seconds with merging versus 18.961 without LoRA. Wall time still
has roughly 1.79 seconds of LoRA overhead; the activity difference accounts for
about 0.33 seconds. The remaining gap is outside recorded CUDA activity and is
not a separately measured CPU attribution.

The tiled merge prepares 22,528 row tiles on first use and 3,392 per warm image
as evicted weights need preparation again. Warm additional H2D/D2H copy time is
about 0.229 seconds and merge helper kernels about 0.075 seconds. First use is
still substantially more expensive than Comfy; no further tile, memory-policy,
or precision optimization is introduced in this change.

## Correctness checks and scope

- Eight synthetic comparisons against installed Comfy/Kitchen match INT8 bytes
  and FP32 scales exactly: positive, negative, zero and small strengths, ordered
  adapters, gate/up halves, width 24,576 and rank 256. Dyadic operands isolate
  rounding stages from different FP32 GEMM algorithms; arbitrary operands are
  not guaranteed bitidentical across different GEMM tiling.
- Fifteen synthetic ModelManager lifecycle cases pass for ordinary linear,
  separate gate/up halves and full fused gate/up adapters across CPU, GPU and
  disk placement, including independent eviction, reuse, strength changes,
  restoring strength and removing LoRA.
- The first real merge attempt returned HTTP 500. An explicitly authorized,
  isolated diagnostic run recorded only fixed rejection codes and source lines.
  The shape rejection at the then-current `model_manager.cpp:2079` was reproduced
  with a synthetic full gate/up adapter. The implementation now accepts both
  full-matrix and half-matrix adapters. No diagnostic instrumentation is in the
  final candidate.
- An exploratory group-64 synthetic case differed in 484 INT8 values. The
  candidate explicitly rejects group 64; the actual checked workload uses 256.
- Full native build, Python syntax checks, patch restoration, source hashes,
  frozen binary verification and focused diff checks pass. GPU execution used
  approved sandbox escalation. No approval-review rejection occurred.

Scope is Qwen Image 2.1 on one CUDA GPU, ordinary linear LoRAs of rank at most
256, ConvRot group 256 and input width up to 24,576. Unsupported adapters fail
the opted-in request. Temporary GPU workspace is reserved through the existing
memory manager (128 MiB); no full floating-point model copy is kept. Peak VRAM
was not measured. Image quality, cross-engine pixel identity, other GPUs,
other resolutions, reference editing and concurrent production requests are
not validated by this test.

## Provenance and reproduction

The requested checkpoint commit was made first: `ec712a1`. The new merge
implementation and this evidence remain uncommitted at completion.

Final native SHA-256:
`c48eb730ffd8f5be2ea73c41ffca18de836e669b19a57649d91236e42eef3aa5`.
Source lock SHA-256:
`a666846e13e179beee1af43a265ec5020f37ce8f41e97b45eea71ab4de2d9d6b`.
Frozen v2 remains:
`9fd49b0e5cb3426043ebd85271ad167555551ac65252590cde5ac25bed44bbb0`.

`build.py` restores the pinned base patches, the LoRA-output parent overlay,
and this complete overlay including the new CUDA source/license/notice. It
verifies their hashes before building a separate candidate. It does not alter
the existing ignored native checkout or overwrite accepted executables.

Numeric evidence is in [comparison-results.json](comparison-results.json) and
[synthetic-results.json](synthetic-results.json). Local authorized timing/kernel
traces are under `experiments/qwen-image21-sage/runs/`:

- Comfy: `comfy-lora-cfg1-veb6za2v`.
- Final native merge on: `lora-cfg1-4llj23t2`.
- Final native merge off: `lora-cfg1-c0a1lgld`.

Use the commands in [README.md](README.md). Keep all model/workload arguments
fixed and toggle only `--lora-merge 0` / `--lora-merge 1` for the native pair.
`--trace-library` enables only the separately authorized CUDA measurements.
