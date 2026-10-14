# Auto-fit CUDA scratch accounting — 2026-10-13

The active runtime now includes the upcoming graph's cuBLAS temporary pool
requirement in auto-fit capacity checks. It accounts for existing physical pool
capacity before requesting additional headroom. No explicit `--max-vram`
override is needed for the tested Qwen reference.

This implements the fix motivated by [the VAE diagnosis](VAE-MEMORY.md).
The accepted INT8, ConvRot, SageAttention2++, and FP16-output computation paths
are unchanged. The experimental down-projection fusion remains disabled.

## Implementation

- CUDA exposes an optional `ggml_backend_cuda_graph_pool_memory` query through
  the existing backend proc-address interface. The core wrapper returns zero
  contributions when the extension/configuration is unsupported.
- For floating-point matmuls, it bounds the cuBLAS input conversions, temporary
  output, and batched pointer arrays from tensor dimensions and types. The
  execution and estimator share precision and output-type selection, including
  `GGML_CUDA_CUBLAS_COMPUTE_TYPE` and per-node precision settings.
- Simultaneously live buffers within one operation are summed and aligned to
  128 bytes. Serial operations reuse the pool: the graph takes the maximum,
  not the sum of all operations. Required pool growth is rounded to the device's
  VMM allocation granularity.
- Current physical pool residency and additional growth travel with each
  `DeviceMemoryRequest`. The free-device check adds **only growth** to pending
  allocations, missing weights, and the existing 512 MiB safety reserve.
  The configured-budget check includes resident pool capacity plus growth once,
  alongside graph/runtime memory and weights. Existing graph tensors are not
  charged again as scratch.
- Requests are rebuilt after weight preparation and workspace allocation.
  Auto-fit can therefore evict more weights before approving execution, using
  the existing eviction mechanism.

For this VAE, the bound is **788 MiB**, with **92 MiB already resident**, giving
**696 MiB additional physical growth**. All image runs below report those exact
values. The runtime releases six weight blocks before decoding, versus five in
previous failed default-auto diagnostics. The final weight-release line after
image generation is cleanup and is not included in that count.

## Coverage and limits

This is a targeted bound for floating-point cuBLAS scratch on the serial-stream
CUDA/VMM path. Floating matmuls that dispatch to another kernel may be
conservatively overestimated. Other kernels (including INT8/attention scratch),
cuBLAS-private workspace, and external activity still rely on the existing
safety reserve and driver-reported free memory. This is not a complete scratch
planner for every CUDA operator.

Non-VMM pools and `GGML_CUDA_GRAPH_OPT=1` concurrent-stream execution retain the
previous fallback behavior; their pool requirements are not newly estimated.
The local acceptance scope is one SM89 GPU, the stated Qwen model, and 1600×896.
Multi-GPU placement, other resolutions/models, adversarial external allocation
pressure, and persistent server requests are not validated here. A desktop
allocation can still race a capacity check.

## Verification

CLI and server build successfully. A first compile exposed the local ggml graph
accessors' non-const signatures; the query signature was corrected and the
subsequent build passed. Both repositories pass `git diff --check`.

`test-cuda-pool-memory.cpp` queries large graph metadata without allocating its
large tensors. It checks the 788 MiB VAE bound, serial reuse, BF16 direct output,
F32 input conversion, mixed-type inputs, and batched pointer arrays. It then
executes a small real CUDA GEMM twice, checks every output, and verifies that the
allocated 6 MiB pool removes subsequent growth and reduces the VAE estimate to
782 MiB. The unsupported concurrent-stream configuration returns no estimate.
All assertions pass on the local RTX 4070 Ti SUPER.

Three of three complete reference runs succeed without `--max-vram`, and all
three produce the exact established RGB SHA-256:
`253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2`.

| Run | Start to saved PNG | Sampling | VAE | Initial / sampled peak GPU MiB |
| --- | ---: | ---: | ---: | ---: |
| `20261013T132008Z-control-yx7y5lej` | 61.390 s | 56.41 s | 2.05 s | 1389 / 14580 |
| `20261013T132129Z-control-inqqzahk` | 60.693 s | 55.83 s | 2.05 s | 1375 / 14581 |
| `20261013T132243Z-control-oa1u6ztq` | 59.965 s | 55.15 s | 1.97 s | 1391 / 14541 |

Mean start-to-save: **60.683 s**; mean sampling:
**55.80 s**. These validate functionality, not a
controlled speed comparison with the older 59.61 s baseline. The initial
desktop VRAM load here is lower than the previous OOM diagnostic's 1890 MiB.

Active CLI SHA-256:
`1fd07ca30dd47221627ae941bc574b1ba39aae27a62783a3e78db7bc4309af26`.

Each run starts a
fresh process with warm filesystem cache and saves an RGB image. Start-to-save
uses parent receipt of the successful-save log; it is not a filesystem `fsync`
measurement or persistent-server latency. VRAM is sampled once per second and
includes the desktop, so it is not an exact process high-water mark.

## Reproduction and artifacts

```sh
GGML_CUDA_FUSED_INT8_GATE_UP=1 GGML_CUDA_FUSED_INT8_PROJECTION=1 \
GGML_CUDA_CONVROT_THREADS=512 GGML_CUDA_FUSED_SWIGLU_CONVROT=1 \
python3 experiments/qwen-image21-sage/profiling/run.py --threads 11 --default-vram
```

`GGML_CUDA_CONVROT_THREADS=512` is a CUDA thread count, not a VRAM margin.
The replay retains 1600×896, 40 ER-SDE/Beta steps, CFG 2.5, seed 42, CPU RNG,
SageAttention2++, F32 prefix, no LoRA, and the frozen positive/negative prompts.
`--default-vram` removes the historical replay command's explicit `--max-vram`.
Historical commands and comparison scripts retain their original configurations.

`runs/auto-scratch-build/` preserves before/after source, the scoped patch,
source/binary hashes, CLI/server binaries, both build logs, the local synthetic
executable and its output, and `results.json` with per-run evidence. Full runs
include commands, environment, logs, binary hashes, timing, GPU telemetry, and
saved PNGs. GPU execution used approved sandbox escalation. No commit or push
was made.
