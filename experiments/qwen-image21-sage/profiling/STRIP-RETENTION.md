# Idle retention with strip convolution enabled

Date: 2026-10-14. Follow-up to [VAE strip convolutions](VAE-STRIP-CONVOLUTIONS.md).

## Result

Retention does not reduce weight reloads further on this reference workload.
Both modes reload **2946.44 MiB, 199 tensors in three blocks, in 0.40 s** per
warm request. All 12 images are bit-identical to the established native reference,
including alpha. All accepted optimization flags remain unchanged.

| Four warm requests per mode | Retention off | Retention on |
| --- | ---: | ---: |
| Request → closed PNG write, mean | 39.707 s | 38.854 s |
| Request → closed PNG write, median | 39.743 s | 38.571 s |
| Request → closed PNG write, range | 38.952–40.390 s | 38.423–39.850 s |
| Sampling, mean | 37.625 s | 36.793 s |
| VAE, mean | 1.838 s | 1.825 s |
| GPU parameter buffers reloaded | 2946.44 MiB | 2946.44 MiB |
| Tensor loading | 0.40 s | 0.40 s |
| VAE graph workspace | 8663.87 MiB | 8663.87 MiB |
| Whole-GPU idle memory, mean | 6112.25 MiB | 6122.75 MiB |
| Sampled whole-GPU peak, A / B | 15324 / 15277 MiB | 15309 / 15273 MiB |

The observed end-to-end difference is **0.853 s (2.15%)** in favor of retention.
Of that, 0.833 s is in sampling, while transfer volume and reported transfer time
are unchanged. The sample is small, timing ranges overlap, and this is a shared
desktop GPU. Preserve the measured difference, but do not treat it as proof of a
repeatable speedup or of improved weight residency. No outliers were removed.

| Series | First request → PNG | Process start → first PNG | Warm request → PNG |
| --- | ---: | ---: | --- |
| Off A | 43.471 s | 43.776 s | 39.314, 40.390 s |
| On A | 40.926 s | 41.234 s | 38.462, 38.680 s |
| On B | 40.958 s | 41.264 s | 38.423, 39.850 s |
| Off B | 41.313 s | 41.620 s | 40.171, 38.952 s |

These first requests are fresh server processes, not guaranteed cold filesystem
cache measurements. Their initial GPU diffusion parameter load is 6920.55 MiB.

## What survives between images

The retention branch is exercised twice per image. It reports 1167.68 MiB of
reclaimable diffusion workspace after sampling and 0.00 MiB after VAE decoding.
During VAE admission, the first-image logs release the same three diffusion
blocks as the control: approximately 1008.83, 944.74 and 992.86 MiB (individual
rounded values). The next request prepares 2946.44 MiB of GPU parameter buffers.

Initial diffusion load minus warm reload implies approximately **3974.11 MiB**
of diffusion weights survive in both modes. This is inferred from loading logs,
not from the whole-GPU idle counter. The VAE arena still needs 8663.87 MiB plus
other allocations, so the current run does not keep all diffusion weights in
VRAM through decoding. Prompt encodings are reused from the existing CPU cache.

The flag avoids explicit runner-end eviction; it does not pin weights against
memory pressure. Also, flag-off cleanup does not necessarily remove every GPU
weight: the model manager evicts eligible storage/staging blocks. Thus retained
weights with the flag off are consistent with the actual implementation.

**Decision:** leave `SD_QWEN21_RETAIN_IDLE` off by default. The strip-convolution
change provides the previously measured reduction from 5911.70 to 2946.44 MiB
of warm reloads; enabling retention adds no further reduction here. This test
does not promote either experimental flag or change inference code. Strip
convolution remains an opt-in validated for this reference; the previously
documented small-shape numerical failures remain unresolved.

## Protocol and validation

Same frozen `vae-strip-build/sd-server`, order off A / on A / on B / off B,
three images per fresh server. Warm statistics exclude the first image of each
series. Both modes use `SD_VAE_STRIP_CONV=1`, prior rejected VAE fusion and
PAD-scale reuse off, and all accepted diffusion flags. Only
`SD_QWEN21_RETAIN_IDLE` varies. F32 gate/up, BF16 attention/residual, accepted
fusions, split V and unsplit K are preserved.

Reference: 1600×896, seed 42, 40 ER_SDE/beta steps, CFG 2.5, no LoRA,
approved synthetic prompt pair, native CPU RNG, 11 threads, Sage2++, automatic
fit with no manual margin. Every image executes 80 diffusion forwards and two
strip-convolution builders, with no VAE tiled fallback. Every warm request hits
both prompt-cache entries and executes no text encoder graph.

The timer begins before the HTTP request and ends after response decoding and
closed local PNG write, without fsync. Server startup is separate in warm
statistics. Weight loading, diffusion, VAE, transport and PNG handling are
included. Device telemetry is sampled once per second and includes desktop
usage and CUDA pools; it is not a precise allocator peak.

Final validation passed for all 12 decoded RGBA images, accepted flags, prompt
hashes, frozen binary hash, and native source hashes against the build snapshot
and before/after each run. RGBA SHA-256:
`473bfa4ab0c2230258a168afbe07523bd6bc7d81a4a2daf458500b6df036132d`.
RGB SHA-256:
`c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c`.
This is identity with the native reference, not a claim of ComfyUI pixel identity.

Only approved synthetic inputs and selected model files were used. No private
ComfyUI logs, workflows, history or configuration were read; no conditioning
tensors were persisted. No ComfyUI rerun, source rebuild, commit or push.
GPU/loopback execution used approved sandbox escalation. All four server and
monitor processes completed cleanup; no test server remained after completion.

## Artifacts

- [Counterbalanced driver](compare-strip-retention.py).
- [Validator](validate-strip-retention.py).
- [Request harness](compare-resident.py).
- [Full experiment, timings, validation and per-series paths](../runs/20261014T154706Z-strip-retention-25k4roa3/).
- [Frozen native build](../runs/vae-strip-build/).
