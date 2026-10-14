# Qwen Image 2.1: combined BF16 attention and residual experiment

Date: 2026-10-13. RTX 4070 Ti SUPER, CUDA 13.4, SM89.

## Question and scope

Does the earlier BF16 residual-stream gain survive when combined with the
BF16 attention chain? This experiment enables two existing switches together;
no C++/CUDA source, kernel, binary, default, dependency or API is changed.

| Setting | Baseline | Candidate |
| --- | --- | --- |
| `SD_QWEN21_BF16_ATTENTION` | `1` | `1` |
| `SD_QWEN21_BF16_RESIDUAL` | `0` | `1` |
| `SD_SAGE_DIRECT_V_CAST` | `0` | `0` |

Both variants keep prior INT8 gate/up, Q/K/V/output/down projection, 512-thread
ConvRot, SwiGLU/ConvRot, RMSNorm/RoPE, norm/modulation and gated-residual fusions.
The old direct-V flag is unnecessary for the new direct BF16 V path.

The existing final CLI is
`../runs/bf16-attention-build/final/sd-cli`, SHA256
`b8d6582e991aaf0033d40032bdff3de4fceea98811a142e3462ecda370398243`.
It matches the current build. Both variants use this same frozen executable.

Attention stores Q/K/V, normalized/rotated Q/K and prefix caches in BF16.
The candidate additionally stores the residual stream in BF16 during the 78
prefix-reuse forwards. The two prefix-building forwards retain F32 residuals.
Norm/modulation reads BF16 residuals and outputs F32 for INT8 ConvRot; gated
residual reads the F32 branch and writes BF16. Sensitive arithmetic remains
F32. The MLP activation path stays as before. Native Sage2++ still uses INT8
Q/K, FP8 V and its internal FP16 output before conversion to BF16.

## Validation and protocol

The combined full-model CUPTI profile is
`../runs/20261013T172644Z-activity-fqw4byk1/`. The existing analyzer accepts both
BF16 flags together and passes its strict assertions:

- 80 diffusion forwards and 2,560 native Sage2++ core calls.
- 7,680 BF16 Q/K/V projection calls and 5,120 BF16 RMSNorm/RoPE calls.
- 2,560 direct BF16 ConvRot output-quantization calls.
- 4,992 BF16 norm/modulation and 4,992 BF16 gated-residual calls, with no F32
  versions of those fused kernels in the diffusion window.
- Previous INT8 GEMM fusion counts remain valid.
- Zero dropped CUPTI records and no events crossing selected phase boundaries.
- Both prefix caches are logged as BF16 (230 and 57 tokens).

No numerical kernel code changed, so previously successful isolated numerical
and small sanitizer checks were not repeated. This turn validates full-model
composition, actual dispatch and image repeatability. It does not add full-model
sanitizer coverage or establish equivalence to Kitchen.

Unprofiled timing uses one excluded attention-only warm-up and three pairs
ordered baseline/candidate, candidate/baseline, baseline/candidate. Only the
residual flag varies. The driver checks binary SHA256, prompt hashes, normalized
CLI/environment equality, 11 CPU threads, auto-fit without `--max-vram`, image
size and the expected per-variant pixel hash.

Fixed settings: 1600×896, seed 42, 40 ER_SDE/beta steps, CFG 2.5, CPU RNG,
Sage2++, no LoRA/reference/second pass and the established synthetic prompt and
negative prompt. Each image starts a fresh process, with warm OS file caches.
Start-to-PNG includes model loading, conditioning, sampling, VAE and saving;
the engine's sampling phase includes lazy diffusion loading. The save endpoint
is the successful save log, not a disk fsync. Desktop use remains present.
Profile timings are excluded from ordinary means.

## Profile comparison

The new attention-only profile is
`../runs/20261013T173517Z-activity-0fa9rhll/`. It passes the strict attention-only
fusion counts, has no dropped records or crossing events, and reproduces the
established attention-only pixels. Both profile images use the same frozen
binary and are excluded from the ordinary timing means.

| Summed GPU kernel time | Attention only | Combined |
| --- | ---: | ---: |
| Norm/modulation, 4,992 calls | 1.517 s | 1.118 s |
| Gated residual, 4,992 calls | 2.243 s | 1.590 s |
| Attention RMSNorm/RoPE, 5,120 calls | 1.521 s | 1.549 s |
| All diffusion GPU activities | 40.919 s | 40.686 s |

The targeted norm/residual operations total 1.053 fewer GPU seconds, but the
net GPU-time difference in these two single traces is only 0.233 seconds:
other kernel timings vary between captures. Do not add only favorable groups
or substitute profile totals for the paired ordinary-run result. Dispatch and
the target operations support the intended mechanism; these single captures
are not a controlled per-kernel latency estimate across repeated pairs.

## Image behavior

Candidate RGB SHA256:
`c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c`.
Attention-only reference:
`b07e26e2c6ea38bb8c69df093d7f4f27c0190784d35b647fbd7463746903e8a8`.

| RGB difference | Attention-only reference | Original native F32 reference |
| --- | ---: | ---: |
| SSIM | 0.883433 | 0.955252 |
| MAE, 0–255 | 7.9520 | 3.6824 |
| RMSE, 0–255 | 20.9689 | 10.0982 |
| Pixels changed in at least one channel | 96.14% | 92.86% |

The combined image was visually inspected: the harbor/cafe scene, woman, cup
and sign remain coherent, with altered local details. It happens to be closer
to the original F32 image by these metrics than attention-only was. This one
sample is not evidence of generally better quality or improved ComfyUI parity;
rounding effects do not accumulate monotonically. Neither comparison is pixel
identity. Raw metrics are in `../runs/bf16-combined-check/image-metrics.json`.

## Timing and memory

| Mean of three ordinary images | Attention only | Attention + residual |
| --- | ---: | ---: |
| Sampling | 42.770 s | 41.837 s |
| Process start → saved PNG | 47.486 s | 46.582 s |
| Conditioning | 1.863 s | 1.880 s |
| VAE decode | 1.993 s | 2.000 s |

The combination saves **0.933 s (2.18%)** in sampling and **0.904 s (1.90%)**
from process start to saved PNG. Sampling pair savings are **0.72, 0.54 and
1.54 seconds**; start-to-PNG savings are **0.695, 0.445 and 1.572 seconds**.
All three pairs favor the combination, including reversed order, but the size
of the gain varies. Sampling ranges are 42.30–43.25 s baseline and 41.71–42.04 s
combined. This is a modest local improvement, not a precise universal saving.

The earlier attention-only mean was 41.867 s. The new baseline is slower despite
matching the same image and configuration. Use the current paired comparison
to attribute this change's gain; comparing the candidate directly with the old
mean would obscure the changed timing conditions. The cause of the drift was
not isolated. Desktop GPU use remains present and clocks are not locked.

All seven ordinary images reproduce their variant reference pixels exactly,
and the three combined images match the separate combined profile image.
Raw results and the replay plan are in
`../runs/20261013T172814Z-bf16-combined-images-1nstdf31/`.

The planned diffusion compute allocation is **1179.54 MiB in both variants**.
All ordinary runs use 80 single-segment diffusion forwards, 6920.55 MiB of
diffusion weights on GPU and a 10632.62 MiB VAE compute allocation. Auto-fit
succeeds without a manual margin or an OOM. The combined flag narrows residual
storage during reuse, while prefix-building residuals remain F32; it does not
lower the reported maximum allocation.

Whole-GPU peaks sampled at 1 Hz are 15215, 15288 and 15212 MiB for the baseline,
and 15228, 14632 and 15256 MiB combined. These include the desktop and other
pipeline phases; the single lower sample is not a repeatable reduction.
No reliable whole-generation peak-VRAM saving is established.

**Decision:** use the combination as a candidate reference for further tests,
keeping both flags opt-in. The earlier residual-only saving broadly survives,
but the repeatability/quality coverage remains one prompt and seed. No new
kernel optimization is included in this experiment.

## Limits and artifacts

Only experiment scripts and documentation are added/updated. Existing engine
sources and binaries remain unchanged and both BF16 switches remain opt-in.
No build was needed, and no commit/push, Computer Use or subagents were used.
GPU access required approved sandbox escalation.

The selected model files and previously approved synthetic benchmark prompt
files are reused. Existing private ComfyUI logs/history/queues/workflows were
not inspected. Other prompts/seeds, models, GPU architectures, batch sizes,
LoRAs, image editing, server/API/UI behavior and full-model sanitizers were not
tested. ComfyUI was not rerun. Results apply to this fixed reference workload.

The comparison script is `compare-bf16-combined-images.py`; image metrics use
`measure-bf16-combined-image.py`. Source hashes, profile summary, image metrics
and timing driver output are retained in `../runs/bf16-combined-check/`.
