# Qwen Image 2.1: paired BF16 Q/K RMSNorm and RoPE

Date: 2026-10-13. RTX 4070 Ti SUPER, CUDA 13.4.92, SM89.

## Result

The paired kernel preserves the native reference image exactly. In the full
model profile, Q/K RMSNorm/RoPE takes **1.556 → 1.094 GPU seconds**, saving
**0.462 seconds / 29.7%** and **2,560 kernel launches**.

Three ordinary, alternating pairs with every earlier optimization enabled give:

| Measurement | Separate Q/K | Paired Q/K | Mean saving |
| --- | ---: | ---: | ---: |
| Sampling | 40.117 s | 38.770 s | 1.347 s / 3.36% |
| Process start to saved PNG | 44.889 s | 43.526 s | 1.363 s |

Individual sampling savings are **0.38, 1.83 and 1.83 seconds**. The independent
profile pair has the opposite whole-window result: total selected GPU activity
is **37.842 → 38.527 seconds**, with other work collectively taking 1.147 seconds
longer. Thus the directly measured improvement is in Q/K itself; the whole-image
mean is encouraging but its entire 1.347-second gain cannot be attributed to the
fusion. Do not compare these absolute times directly with older ComfyUI runs.

All nine generated images—seven ordinary runs including one excluded warmup,
and two profiled runs—match the established native RGB hash:
`c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c`.
This establishes equality for this reference, not pixel parity with ComfyUI or
bit identity for every possible input.

## Implementation

`GGML_CUDA_FUSED_QK_RMS_ROPE=1` enables the experiment; it remains **off by
default**. It operates on the existing BF16 attention path. No new public GGML
operation, model representation, precision change or dependency update is added.

One 128-thread block processes the same head/token of Q and K. It retains two
independent reductions in the original order, separate epsilon and learned norm
weights, explicit F32 multiply/add rounding and final BF16 stores. The two
reductions use disjoint shared scratch. The RoPE matrix is loaded once for both
outputs.

When enabled, Qwen expands both projections before two adjacent norm/RoPE
chains. Existing dependency views keep both inputs and norm weights alive past
both outputs. CUDA fuses the ten-node pattern only when types, shapes, shared
position tensor, intermediate uses and non-overlapping input/output storage
permit it. Unsupported or externally retained/reused intermediates preserve
fallback execution. The off path keeps the previous graph order.

Only four inference source files change: the Qwen model header, `ggml-cuda.cu`,
`rope.cu` and `rope.cuh`. The other 29 selected native/Comfy/AIO/Kitchen source
hashes are unchanged. Earlier local work is retained in the source snapshots and
scoped patch.

## Synthetic and build validation

- The standalone candidate and exact extracted previous kernel compare
  **298,107,904 BF16 bytes** across twelve cases. Cases include 1/17/57/230/
  5600/5657/5830 tokens, one and 32 heads, multiple batches, distinct Q/K weights
  and epsilons, zeros/signed zeros, and tiny/large finite inputs. Outputs are
  finite and bit-identical.
- Eight alternating CUDA-event batches at 5,600 tokens average **0.5654 →
  0.3766 ms per Q/K pair**, a 33.4% reduction. The larger two token counts show
  similar improvements. This was measured before integration.
- Twelve graph/allocator cases compare **287,470,080 BF16 output bytes** exactly,
  plus retained/reused F32 intermediates. CUPTI verifies **seven** eligible
  paired calls and zero dropped records. Other cases exercise different position
  tensors and retained/reused cast, norm, multiply and RoPE intermediates.
  Cross-input/output overlap guards are checked explicitly.
- Standalone and graph versions each pass a small eligible Compute Sanitizer
  memcheck and racecheck case: zero errors, warnings or hazards. These are
  synthetic safety checks, not full-model sanitizer coverage.
- CLI and server builds pass. Only the CLI is executed. Python syntax and both
  native repositories' `git diff --check` pass.

Frozen binaries are in [the build evidence directory](../runs/paired-rms-rope-build/).
CLI SHA256: `dd6a151a9708b8b32367fec130bbde2fb83a1a1f3e9f1bd0b5eb70e7915b9c3b`.
Server SHA256: `4924e46d0ab858449eb515b325f587bdce23233c8134a260343d8e0640c1950f`.

## Full-model protocol and profile

The saved synthetic prompts, approved INT8 ConvRot checkpoints, 1600×896,
seed 42, 40 ER_SDE/beta steps, CFG 2.5, CPU RNG, Sage2++, 11 CPU threads and
batch one are unchanged. There is no LoRA, reference image or second pass.
Auto-fit uses no manual VRAM margin. Both BF16 flags, norm/ConvRot, MLP
down/residual, attention output/residual and all other accepted fusions remain
on. Only the paired-Q/K flag changes. Binary, normalized command, environment
and prompt hashes are checked across ordinary and profiled variants.

Strict profiles verify 80 single-segment forwards, 2,560 Sage calls, 7,680 BF16
Q/K/V projections, 4,992 norm/ConvRot calls and 2,496 calls each of fused MLP and
attention residual projections. The candidate replaces 5,120 separate
RMSNorm/RoPE calls with 2,560 paired calls; no separate target calls remain.
Both accepted traces have zero dropped records and zero boundary crossings.

The entire-window activity count falls by 2,557 rather than 2,560. Three small
initial kernels (RMSNorm, scale and multiply, **13.184 microseconds** total)
finish before the baseline parent-log boundary, while their counterparts are
inside the candidate window. The target Q/K counts are exact; strict validation
has not been relaxed. See `boundary-evidence.json` for timestamps.

Diffusion weights remain 6920.55 MiB on CUDA. Reordering the graph changes its
compute buffer from **1179.54 to 1167.68 MiB**. VAE compute remains 10632.62 MiB.
All runs complete without OOM or manual margin. Ordinary whole-GPU sampled peaks
are 15255–15390 MiB off and 15336–15393 MiB on, including desktop use and
one-second sampling; no reduction in whole-GPU peak is established.

## Evidence and scope

- [Builds, snapshots, scoped patch, sanitizers, synthetic results and validation](../runs/paired-rms-rope-build/)
- [Alternating ordinary image pairs](../runs/20261013T193638Z-paired-rms-rope-images-m6iotkfx/)
- [Paired profile](../runs/20261013T194205Z-activity-r_b5xbbs/)
- [Separate profile](../runs/20261013T194252Z-activity-ywwf4a2b/)
- [Standalone checker](check-paired-rms-rope.cu), [graph checker](check-paired-rms-rope-graph.cpp), [image driver](compare-paired-rms-rope-images.py).

GPU execution successfully used sandbox escalation. Only approved models,
synthetic prompts, our benchmark artifacts and selected source files were used.
Private ComfyUI logs, history, queues, workflows and configuration were excluded.
No commit, push, default-profile change, server runtime or UI test was performed.
