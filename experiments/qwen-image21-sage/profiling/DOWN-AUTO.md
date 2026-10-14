# Down-projection image comparison with corrected auto-fit — 2026-10-13

This resumes the [down-projection experiment](DOWN-PROJECTION.md) after the
[CUDA scratch accounting fix](AUTO-SCRATCH.md). Both variants include that fix
and run with `--auto-fit on` and no explicit `--max-vram`.

## Method and scope

The candidate restores exactly the previously validated helper, extending the
fused INT8 GEMM/F32 output path to K=12288, N=4096, at least 4096 rows, on SM89.
The source audit confirms that all other snapshotted engine files remain
unchanged, including the memory fix. The existing numerical, eligibility, and
focused sanitizer checks remain applicable to the identical helper; they were
not rerun. This task checks complete images and actual full-model dispatch.

One excluded fusion-off warmup precedes three measured pairs, in orders
**off/on, on/off, off/on**. Every image starts a fresh process with warm
filesystem cache. Both variants use the same frozen executable and all accepted
optimizations; only `GGML_CUDA_FUSED_INT8_DOWN=0/1` changes. The comparison
checks binary hashes, normalized commands, prompt bytes, logged environment,
11 CPU threads, output size, and exact reference RGB hash. No unrelated GPU
process is stopped and GPU clocks are not locked.

Generation stays at 1600×896, 40 ER-SDE/Beta steps, CFG 2.5, seed 42, CPU RNG,
SageAttention2++, F32 prefix, and no LoRA. Start-to-save includes loading and
is measured at receipt of the successful PNG-save log, not a disk `fsync`.
Sampling includes lazy diffusion loading. These are not persistent-server
request timings. A separate CUPTI run verifies dispatch and is excluded from
latency means.

CLI and server build successfully; no server request is live-tested. GPU
execution uses approved sandbox escalation. No dependency is installed or
updated, no unrelated code is changed, and no commit or push is made.

## Full-image results

All seven unprofiled runs (warmup plus six measurements) completed without OOM
and produced the established reference RGB SHA-256:
`253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2`.

| Mean, three runs per variant | Fusion off | Fusion on | Reduction |
| --- | ---: | ---: | ---: |
| Sampling | 56.157 s | 55.603 s | 0.553 s / 0.99% |
| Start to saved PNG | 60.889 s | 60.347 s | 0.542 s / 0.89% |

| Pair and order | PNG time off | PNG time on | Time saved |
| --- | ---: | ---: | ---: |
| 1 (off/on) | 60.737 s | 60.442 s | 0.295 s |
| 2 (on/off) | 60.898 s | 60.357 s | 0.541 s |
| 3 (off/on) | 61.032 s | 60.241 s | 0.791 s |

Every pair favors fusion, including reversed order. The local image benefit is
small (about 0.9%), not the 4.7–6.5% isolated Linear-chain benefit. Three pairs
on an unlocked shared GPU do not establish a universal performance guarantee.
Use these fresh controls rather than comparing to the historical 59.61 s mean.

Mean conditioning is 1.850/1.853 s off/on; VAE is 2.027/2.037 s. The saving is
in sampling, not a faster encoder or decoder.

## Auto-fit and memory

Before VAE, the current CUDA pool is 92 MiB with down fusion off and 44 MiB
with it on. Auto therefore reserves 696 MiB versus 744 MiB of additional
physical growth, reaching the same 788 MiB total requirement. This confirms
that the new accounting responds to actual pool residency across both paths.
The existing internal reserve remains; neither variant uses a manual CLI margin.

Successful runs evict either five or six parameter blocks before VAE depending
on live capacity. Sampled whole-GPU peaks range from 14483 to 15509 MiB;
starting desktop usage is 1294–1326 MiB. Telemetry includes the desktop and is
sampled once per second, not an exact process high-water mark. A 48 MiB smaller
pre-VAE pool does not establish a lower complete-image VRAM peak.

## Dispatch verification

The separate fusion-on CUPTI run
`20261013T133554Z-activity-he73vero/` completes and produces identical reference
pixels. Its 80 diffusion forwards contain 2560 fused gate/up calls, 12800 fused
ordinary-projection/down calls (10240 + 2560), and 2560 Sage core calls. There
are **zero remaining separate INT8 GEMM/dequantization pairs** in that window.
The trace has zero dropped records and zero phase-boundary-crossing events.
This verifies execution of the intended path; the combined projection group
is not reported as a separate down-only duration. The profiled run is excluded
from timing means, making eight successful reference-identical images in total.

## Disposition

Retain the candidate as an **opt-in**:
`GGML_CUDA_FUSED_INT8_DOWN=1`. It is a small consistent local gain with exact
reference pixels. The default remains off; no launcher defaults are changed.
All previously accepted optimizations and the auto-scratch fix remain intact.

## Artifacts and reproduction

Relative to `experiments/qwen-image21-sage/runs/`:

- `down-auto-build/`: candidate/helper, before-source snapshots, scoped patch,
  source audit, CLI/server binaries and hashes, build logs, frozen drivers,
  `image-results.json`, and comparison/profile logs.
- `20261013T132816Z-down-images-vraxr1kz/`: fixed plan, seven per-run results,
  aggregate statistics, and replay logs pointing to individual image runs.

Each individual image run preserves prompts, command, environment, binary hash,
image, timing log, GPU telemetry, and exact-pixel validation.

```sh
python3 experiments/qwen-image21-sage/runs/down-auto-build/compare-down-images.py --default-vram
```

The driver freezes the tested candidate and enables gate/up, ordinary
projection, SwiGLU/ConvRot, 512 CUDA threads, and 11 CPU threads, toggling only
down fusion. The original `profiling/compare-down-images.py` still reproduces
the historical pre-fix candidate; use the new driver for this comparison.

Candidate CLI SHA-256:
`c1f949f2f552325a206093e8dd0f3e9bb03d396a52b904e142b95c053a3e0965`.
