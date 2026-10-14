# VAE scratch-memory diagnosis — 2026-10-13

**Cause reproduced and measured:** the VAE graph fits the memory manager's
estimate, but a cuBLAS FP16 output buffer is allocated separately from that
graph. The CUDA scratch pool needs **696 MiB of additional physical capacity**
during decoding; the manager checks only a fixed **512 MiB** scratch reserve.
It stops evicting weights too early in the failing run. This reproduces on the
accepted engine without the experimental down-projection fusion.

A separate diagnostic run with a **1024 MiB CLI margin** evicts one more model
block, completes decoding, and produces the exact reference image. No memory
policy, numerical path, model, sampler, or attention implementation was changed.
This is a diagnosis and one successful workaround check, not a permanent fix
or a new performance baseline.

**Later implementation:** [Auto-fit CUDA scratch accounting](AUTO-SCRATCH.md)
adds the measured missing pool-growth accounting and validates the reference
without a manual margin. The diagnostic evidence below remains historical.

## Instrumentation and method

An isolated build adds opt-in `SD_MEMORY_TRACE=1` metadata logging at:

- `GGMLRunner::memory_requests` and segment execution: runner identity, measured
  graph bytes, reusable buffers, pending allocations, and configured limit.
- `ModelManager::check_capacity`: free device memory, tracked weights/runtime,
  required device/budget bytes, the fixed reserve, and the actual fit decision.
- CUDA VMM pool growth: current capacity/usage, allocation request, rounded
  physical growth, driver-reported free VRAM before/after, CUDA result code,
  and cuBLAS source/output tensor shapes and types.

The existing parameter-release log identifies evictions. Pool logs go to stderr
and are flushed before and after `cuMemCreate`, so the failing allocation is
retained even when the existing error handler aborts. There is no new retry,
allocation fallback, synchronization, or changed eviction rule. Metadata logging
and memory queries can affect timing; these are not performance measurements.
No tensor contents or model weights are read into the diagnostic or saved.

Both runs use the same frozen diagnostic executable, 1600×896, 40 ER-SDE/Beta
steps, CFG 2.5, seed 42, the existing positive/negative prompts, SageAttention2++,
11 CPU threads, and all previously accepted fusion flags. Down fusion is absent
from this build. Neither run uses CUPTI. No unrelated application is stopped.
The desktop shares the GPU; clocks and free VRAM can change during execution.

## What auto approves, and what CUDA actually needs

All values below are MiB (1024² bytes), from allocation/capacity events rather
than the once-per-second GPU utilization monitor.

| Measurement | Default auto-fit | Explicit 1024 MiB margin |
| --- | ---: | ---: |
| GPU parameter blocks evicted before VAE execution | 5 | 6 |
| Tracked weights remaining after VAE staging | 2484.524 | 1491.660 |
| Measured VAE runtime/graph allocation | 10632.617 | 10632.617 |
| Free VRAM at final VAE preflight | 636.500 | 1629.250 |
| Internal ModelManager scratch reserve | 512.000 | 512.000 |
| Fit decision | accepted | accepted |
| Scratch-pool capacity on entering VAE | 92.000 | 92.000 |
| Required final scratch-pool capacity | 788.000 | 788.000 |
| Additional physical capacity needed | 696.000 | 696.000 |
| Outcome | CUDA OOM during VAE | image saved, reference-identical |

In the failed run, five evictions leave 2001.713 MiB of tracked weights before
VAE weights are staged. Adding the 482.811 MiB VAE weights yields 2484.524 MiB.
The final preflight sees zero pending graph allocations and zero missing
weights, so `required_device_bytes` becomes just the fixed 512 MiB reserve.
There is 636.5 MiB free, and the graph also satisfies the configured budget.
`fits()` therefore returns true and no sixth block is evicted.

Actual scratch growth in the failing VAE is:

| Node | Temporary request | Pool before | New physical allocation | Free immediately before allocation |
| --- | ---: | ---: | ---: | ---: |
| `node_532` | 196.875 | 92 | 106 | 634.500 |
| `node_715` | 393.750 | 198 | 196 | 528.250 |
| `node_894` | 787.500 | 394 | 394 | 332.250 |

Pool granularity is 2 MiB. Temporary storage is reused between these operations;
the three request sizes must **not** be added together. Physical capacity grows
from 92 → 198 → 394 → 788 MiB. The last 394 MiB allocation has only 332.25 MiB
available, a **61.75 MiB shortfall**, and returns CUDA status 2 (out of memory).
The same three expansions all succeed in the 1024-MiB-margin control, with
931.5 MiB free immediately after the last expansion. There is no later pool
growth during that successful VAE execution.

The initial pool already exists and is already excluded from free VRAM. The
missing reservation is therefore based on its **696 MiB additional growth**,
not a second charge for all 788 MiB. This exceeds the fixed reserve by 184 MiB.
Small changes in available VRAM and coarse, roughly 1 GiB weight-block evictions
explain why the old margin can sometimes complete a run and sometimes fail.
The exact effect of other processes is not isolated, but the allocation
shortfall itself is directly measured.

## The allocation is an FP16 output temporary

The failing operation is the decoder's
`first_stage_model.decoder.upsamples.3.upsamples.3.resample.1.weight`
convolution lowered to cuBLAS matrix multiplication:

| Tensor | Type | Shape in GGML order | Size |
| --- | --- | --- | ---: |
| Lowered convolution input | F16 | 2592 × 1433600 × 1 × 1 | 7087.5 MiB |
| Reshaped weight | F16 | 2592 × 288 × 1 × 1 | 1.424 MiB |
| Graph output | F32 | 1433600 × 288 × 1 × 1 | 1575 MiB |
| Backend temporary output | F16 | 1433600 × 288 | 787.5 MiB |

In the active reference source,
[`ggml_cuda_mul_mat_cublas_impl`](../../../stable-diffusion.cpp/ggml/src/ggml-cuda/ggml-cuda.cu)
allocates `dst_temp.alloc(ne_dst)` for the FP16 path on this GPU, runs GEMM into
it, then converts to the graph's F32 destination. That temporary allocation is
outside graph buffer accounting. The large lowered input and F32 graph output
are already in the graph estimate; they are **not** extra allocations to add
again to the measured 10632.617 MiB graph allocation.

[`ModelManager::check_capacity`](../../../stable-diffusion.cpp/src/model_manager.cpp)
explicitly substitutes a fixed 512 MiB allowance for backend scratch/pipelines.
This is separate from the CLI margin used by auto-fit to cap managed buffers.
Default auto-fit also reserves 512 MiB, as confirmed by the earlier experiment;
removing `--max-vram -0.5` does not remove the internal scratch-accounting gap.

## Validation and disposition

- Default auto-fit reproduces the VAE OOM on the accepted reference computation.
- The same diagnostic build with a 1024 MiB CLI margin completes one full image.
- Successful image: 1600×896 RGB SHA-256
  `253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2`, exactly the
  established reference. Diagnostic start-to-save time is 61.035 s; VAE is
  2.05 s. Neither number replaces an uninstrumented benchmark.
- GPU telemetry starts at 1890/1860 MiB and samples peaks of 15663/15055 MiB for
  default/1024 margin. These include the desktop and are not exact high-water
  marks of the process.
- Engine source is restored byte-for-byte after collecting evidence. CLI and
  server are rebuilt; the diagnostic patch and executable are preserved.
- No server request, post-restore image, additional sanitizer run, or wider
  GPU/model matrix is tested. No claim of general stability follows from one
  successful larger-margin run.
- GPU execution used approved sandbox escalation. No commit or push is made.

## Next change to consider

The evidence supports making auto-fit account for backend temporary storage
before it stops evicting model weights: estimate the upcoming graph/segment's
scratch requirement and the extra physical pool growth still needed, then include
that in capacity checks. Existing pool capacity must not be double-counted.
This should leave safety headroom for unrelated GPU activity as well.

For this exact configuration, a 1024 MiB explicit CLI margin is a demonstrated
workaround. A permanent blanket increase to that value is not implemented and
would not establish correctness for other resolutions, operators, or models.
Changing the FP16-output numerical path to avoid its temporary is also a
separate optimization, not part of this diagnosis.

## Artifacts and reproduction

Relative to `experiments/qwen-image21-sage/runs/`:

- `vae-memory-build/`: before/instrumented source, `instrumentation.patch`,
  frozen binaries/hashes, build logs, `comparison.json`, and restoration audit.
- `20261013T130535Z-control-zl8h2y41/`: failed default-auto-fit reproduction.
- `20261013T130751Z-control-a2eltua9/`: successful 1024-MiB-margin control.

Each run includes command, environment, logs, timestamped log receipt, GPU
telemetry, `memory-events.json`, and `memory-summary.json`. The diagnostic CLI
SHA-256 is `47dc374cf66910de71f35b9cca1ba0be51dbea6c31f7c61b452ec418ea9e5859`.
Only that frozen diagnostic build supports `SD_MEMORY_TRACE`; the active source
has been restored.

```sh
SD_MEMORY_TRACE=1 GGML_CUDA_FUSED_INT8_GATE_UP=1 \
GGML_CUDA_FUSED_INT8_PROJECTION=1 GGML_CUDA_CONVROT_THREADS=512 \
GGML_CUDA_FUSED_SWIGLU_CONVROT=1 \
python experiments/qwen-image21-sage/profiling/run.py --default-vram \
  --binary experiments/qwen-image21-sage/runs/vae-memory-build/sd-cli --threads 11
```

For the separate headroom control, replace `--default-vram` with
`--vram-margin-mib 1024`. These options are mutually exclusive. Analyze the
printed run directory with:

```sh
python experiments/qwen-image21-sage/profiling/analyze-vae-memory.py RUN_DIRECTORY
```
