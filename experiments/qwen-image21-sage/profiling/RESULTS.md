# Diffusion profile — 2026-10-13

The largest measured target is the INT8 Linear chain and its intermediate
memory traffic. Sage attention is working and is a smaller share of this run.
No inference source, build settings, or launcher defaults were changed.

Subsequent experiment: [direct SwiGLU removes the gate/up copies](DIRECT-SWIGLU.md).
That follow-up changes one Qwen model header and records new before/after runs;
the measurements below describe the original engine.

## Runs and validation

All paths below are relative to `experiments/qwen-image21-sage/runs/`:

- Profile: `20261013T080946Z-activity-33vmsv3q/`.
- Uninstrumented control: `20261013T081123Z-control-0b9ut73p/`.
- Original debug reference: `20261013T074331Z-debug-7nor7cre/`.

| Measurement | CUPTI profile | Control |
| --- | ---: | ---: |
| Process wall time | 79.06 s | 76.04 s |
| Engine generation | 77.98 s | 74.92 s |
| Sampling, including lazy diffusion loading | 73.93 s | 70.93 s |

The observed sampling difference is +3.00 s (+4.2%). This is one paired
comparison; instrumentation and ordinary clock/load variation are not isolated
statistically. Do not subtract that difference uniformly from individual kernels.

Both runs exited successfully and produced 1600 x 896 images with RGB pixels
identical to the original reference. Both prompt files were byte-identical, and
arguments matched after normalizing copied prompt/output paths. Decoded RGB SHA-256:
`253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2`.
The profile directory contains `validation.json` and the raw evidence.

## Diffusion-only GPU breakdown

The graph window was **72.839 s**, after model loading and before VAE decoding.
It contains 80 forwards, 148,741 GPU activity records, zero dropped records,
and no records crossing its timestamp boundaries. Summed GPU activity and its
interval union both equal **71.777 s**: the captured operations did not overlap.
Kernels account for 63.255 s and memory operations for 8.523 s. The remaining
1.062 s of the window is uncovered by these GPU activities; it is not a direct
measurement of CPU sampler time and includes launch gaps and scheduling.

| Category | GPU seconds | Share of summed GPU activity |
| --- | ---: | ---: |
| INT8 matrix multiplication | 21.082 | 29.37% |
| Gating and elementwise operations | 11.286 | 15.72% |
| INT32 accumulator dequantization to F32 | 9.209 | 12.83% |
| Device-to-device copies | 8.489 | 11.83% |
| Normalization and RoPE | 5.768 | 8.04% |
| Sage attention core | 5.714 | 7.96% |
| Linear ConvRot activation quantization | 4.043 | 5.63% |
| Layout, concatenation, and dtype kernels | 4.034 | 5.62% |
| Sage preparation and quantization | 2.069 | 2.88% |
| Other kernels and memory operations | 0.084 | 0.12% |

Generic V conversion and attention-output conversion are in the layout/dtype
category, so core plus preparation is not the entire attention pipeline.
The 64 existing masked-prefix FlashAttention calls total only 0.00154 s; this
was the unchanged Sage configuration, not a FlashAttention comparison experiment.

## What the trace establishes

**Linear operations:** 15,360 INT8 GEMMs and 15,360 separate dequantization
kernels ran, together with 10,240 ConvRot quantization kernels. These three
groups total **34.333 s (47.83%)** of GPU activity. The GEMM symbol is
`cutlass_80_tensorop_i16832gemm_s8_256x128_64x3_tn_align16`, selected internally
by cuBLAS. Thus "switch to CUTLASS" is not a sufficient optimization hypothesis.
The relevant difference from Kitchen's proposed route is a fused output
epilogue, intermediate layout/dtype, and kernel suitability for these shapes.
The local [INT8 implementation](<stable-diffusion.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1697>)
still explicitly allocates INT32 output and launches a separate F32 dequantizer.

**MLP output is a particularly large target:** during the 78 forwards after
prefix creation, dequantization with grid `(96,5600,1)` accounts for **4.900 s**
over 2,496 calls. Source maps that grid to 24,576 output channels and 5,600
tokens, matching the combined gate/up projection. The corresponding
24,576-wide GEMM launch shape totals **9.411 s**. The gated SiLU kernel itself
takes another **3.746 s** across all forwards. These are measured costs,
not estimated savings from fusion.

**Copies are within GPU memory:** 27,604 device-to-device operations move
2,370,416,918,528 bytes cumulatively (2.37 TB of copied payload, not simultaneous
VRAM occupancy or a complete count of hardware memory traffic). Copies of
275,251,200 bytes account for **4.943 s**, and copies of 91,750,400 bytes for
**3.345 s**. Those sizes equal F32 tensors of 5,600 x 12,288 and 5,600 x 4,096.
By comparison, host-to-device and device-to-host transfers within this window
total only **0.0334 s**. This reinforces the distinction between repeated
activation copies and moving model weights between RAM and VRAM.

Source suggests concrete places to investigate: the Qwen MLP uses
[`ggml_ext_chunk` for gate/up](<stable-diffusion.cpp/src/model/diffusion/qwen_image_2_1.hpp:293>),
and modulation uses slices, whose default contiguous conversion materializes
views in [the helpers](<stable-diffusion.cpp/src/core/ggml_extend.cpp:72>).
The 4,992 large copies in the steady phase match two gate/up halves per
32-layer forward over 78 forwards. This attribution is supported by sizes,
counts, and source, but individual copy call stacks were **not** captured.
Do not assume every copy is redundant or safely removable.

**Attention is genuine Sage2++:** all 2,560 core launches use the expected
INT8 Q/K, FP8 V kernel with FP16 instruction accumulation. Attention tuning
could help, but cannot by itself explain or recover the entire observed gap
to the user's ComfyUI queue times.

## Next focused experiment

Prioritize the MLP/Linear path. Confirm the large-copy call sites and downstream
stride requirements, then benchmark a representative gate/up + activation +
ConvRot + down-projection chain, including all conversions, against a fused
alternative. A fused INT8 output epilogue also targets a
measured 9.209 s dequantization stage, but zero-cost removal is not realistic
and changing the GEMM can change its own runtime and numerical behavior.

No ComfyUI GPU profile or Kitchen comparison was run here. The experiment
identifies costs in our engine; it does not prove which optimization explains
ComfyUI's advantage. Keep the existing engine path as a reference before any
implementation change, then verify numerical output and full-generation timing.

## Reproducibility and limits

`activity.cpp`, `run.py`, and the compiled tracer are copied into the profile
run directory; `command.sh`, `trace-environment.json`, `generation.log`,
`log-timestamps.jsonl`, `activity.tsv`, GPU telemetry, and summary CSV/JSON files
remain there. The tracer uses installed CUDA 13.4 record layouts and no new
packages. CUPTI reports execution time, not memory throughput counters or
occupancy. Timings describe this GPU, model, prompt pair, and resolution.
GPU access required sandbox escalation, which was approved for both runs.
