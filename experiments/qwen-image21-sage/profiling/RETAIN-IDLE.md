# Reclaimable Qwen diffusion residency between server requests

Date: 2026-10-14 (Europe/Stockholm). Experimental, default off.

## Measured results

| Six warm requests per mode | Control | Retain idle |
| --- | ---: | ---: |
| Request → PNG, mean | 38.937 s | 38.802 s |
| Request → PNG, median | 38.706 s | 38.644 s |
| Sampling, mean | 36.935 s | 36.792 s |
| VAE, mean | 1.757 s | 1.762 s |
| Whole-GPU memory after request, mean | 3630.500 MiB | 3663.333 MiB |
| GPU parameter buffers reloaded per warm request | 5911.700 MiB | 5911.700 MiB |
| Tensor loading per warm request | 0.800 s | 0.800 s |

| Series | First request → PNG | Warm request → PNG samples | Warm sampling samples |
| --- | ---: | --- | --- |
| control_A | 42.376 s | 39.065, 39.953, 38.783 s | 37.090, 37.750, 36.830 s |
| retain_A | 42.684 s | 39.628, 38.609, 38.628 s | 37.640, 36.600, 36.610 s |
| retain_B | 40.834 s | 38.663, 38.633, 38.654 s | 36.640, 36.630, 36.630 s |
| control_B | 40.869 s | 38.600, 38.629, 38.595 s | 36.630, 36.660, 36.650 s |

Observed mean request-to-PNG difference: **0.135 s (0.35%)**. All samples retained.

All 16 images match the native reference. All warm requests hit both prompt-cache entries with no text encoding and still execute 80 diffusion forwards. Source and binary hashes are stable.

Whole-device telemetry peaks: `{"control_A": 14845.0, "retain_A": 14843.0, "retain_B": 14769.0, "control_B": 14769.0}` MiB. These include desktop and CUDA pools, not just model weights.

Changed-prompt, LoRA-change, model-replacement and cancellation paths were not exercised live in this experiment. The flag remains off by default.

## Question and implementation

Can the resident server avoid unconditional diffusion weight eviction and improve
request-to-saved-PNG latency without changing the reference image?

`SD_QWEN21_RETAIN_IDLE=1` makes the Qwen Image 2.1 sampling runner enter an idle
state instead of unconditionally releasing its GPU weights and workspace.
Prompt-dependent prefix caches are still cleared. The existing residency manager
can reclaim the workspace and unpinned weights when another stage needs memory.
At the request boundary any remaining diffusion workspace is released and its
runtime owner is removed; surviving diffusion weights remain tracked by the
model manager. This preserves the loader's no-active-runner update contract.
It does not retain a complete executable graph between requests.

An unchanged empty LoRA configuration can reuse idle weights. A nonempty or
changed LoRA configuration takes the existing full cleanup path. Model reset
and destruction still perform full cleanup, including weights retained while
idle. Only the Qwen Image 2.1 diffusion runner opts in; text encoder and VAE
cleanup is unchanged. No public C API, memory fitting limits, precision,
attention, VAE tiling or numerical kernels change.

This is deliberately bounded by the current 16 GiB hardware: the diffusion
weights occupy about 6921 MiB, while the unchanged untiled VAE workspace is
10633 MiB before VAE weights, CUDA pools and desktop allocations. Full diffusion
residency throughout decoding cannot fit. The useful measurement is how many
weights can survive pressure and how much transfer time this saves. Keeping
all weights would require reducing VAE memory or more VRAM, a separate change.

## Decision

Keep `SD_QWEN21_RETAIN_IDLE` off. The observed mean difference is 0.135 s
(0.35%), with only 0.062 s median difference. Series A favors retention by
0.312 s, while series B favors control by 0.042 s. This small shared-GPU sample
does not establish a useful speedup, and the intended mechanism—avoiding weight
reloads—does not occur. Both modes reload 5911.70 MiB in 0.80 s per warm image.

The next useful residency investigation is the unchanged VAE's 10632.62 MiB
workspace. Reducing that workspace could permit more diffusion weights to
survive decoding and also targets the previously measured VAE latency gap.
That investigation has not been implemented or run here. No configuration is
promoted to the accepted baseline by this experiment.

## Memory outcome

The retention branch is exercised: after sampling the diffusion runner reports
1167.68 MiB of retained, reclaimable workspace. At the end of VAE decoding it
reports 0.00 MiB. The VAE has reclaimed that workspace and evicted 5911.70 MiB
of diffusion parameter buffers. The next request reloads those same buffers,
just as the control does. Roughly 1009 MiB of the 6921 MiB diffusion parameters
survive in both modes. The observed retention policy therefore does not achieve
full residency on this reference workload.

## Protocol

Same rebuilt binary with the flag off versus on. All accepted optimization flags
remain enabled: F32 gate/up, packing, BF16 attention/residual, direct V, split K
off and accepted fusions. Reference: 1600×896, seed 42, 40 ER_SDE/beta steps,
CFG 2.5, no LoRA, native CPU RNG, 11 threads, Sage2++, auto fit and the same
approved synthetic positive/negative prompt pair. Existing CPU prompt cache
capacity is four; warm requests require two hits and zero text-encoder calls.

Order: control A, retention A, retention B, control B. Each starts a new server
process and generates a first image plus three warm repeats. First images are
excluded from warm statistics. Engines never overlap. No profiling overhead is
added. Timing includes native HTTP, polling, image transport and local PNG save
without fsync. GPU telemetry includes the desktop. Per-request idle memory is
sampled after the timed image save and correctness checks.

Every image must match the established native RGB SHA256
`c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c`
and execute all 80 diffusion forwards. Recorded source and binary hashes must
remain stable during each series. No private ComfyUI logs, history, workflow or
configuration are read; no conditioning tensors are persisted. GPU and loopback
execution use approved sandbox escalation. Test servers and monitors stop after
each series. No commit or push is performed.

The earlier ComfyUI numbers are historical context, not a concurrent rerun:
38.814 s mean / 38.708 s median warm request-to-PNG, six warm samples. Compare
this experiment primarily with its same-binary native controls.

## Artifacts

- [Counterbalanced driver](compare-retain-idle.py).
- [Native request harness](compare-resident.py).
- [Build, prior source snapshots and experiment-only patch](../runs/retain-idle-build/).

- [Timing, image and memory validation artifacts](../runs/20261013T221628Z-retain-idle-q62nyzt4/).
