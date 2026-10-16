# Opt-in LoRA output fusion — 2026-10-16

Implemented the user-approved combined FP16→FP32 / scale / add output step.
Warm Qwen CFG 1 wall time with the selected LoRA fell from **44.611 to 39.892 s**:
**4.720 s saved, 10.579% lower latency, 1.118× throughput** in this profiled test.
The improvement does not eliminate LoRA overhead: without LoRA the same candidate
was approximately 20.3–20.5 s.

The earlier profiling work was committed first as `1e5740d`. Implementation,
verification tools and this report remain uncommitted. No push occurred.

## Implementation and precision

`GGML_CUDA_FUSED_LORA_OUTPUT=1` opts in. Unset/0 retains the old path; the global
fusion-disable switch also takes precedence. The application still selects its
accepted engines; this candidate has not replaced any production/frozen binary.

The separate candidate is restored from frozen v2 sources plus the small overlay
under [native/lora-output](../../../native/lora-output/README.md). Only two native
files change: the LoRA adapter marks its output sums, and the CUDA backend matches
`MUL_MAT → SCALE [→ SCALE] → ADD`. It calls the existing cuBLAS implementation
unchanged and consumes its FP16 temporary output directly in the new kernel.

- FP16 GEMM compute/output rounding stays intact.
- Every original SCALE is executed in order, with the same separately rounded
  F32 multiply/add as the existing CUDA `scale_f32` build.
- The final base+LoRA addition has a separate F32 rounding.
- Attention's output rescale is not algebraically combined with the LoRA strength.
- Multiple adapters retain their original accumulation order.
- Original graph/scratch allocations and all other optimization switches remain.

The initial matcher is limited to the measured Ada/SM89 FP16-cuBLAS route,
contiguous unbatched operands, rank ≤256 and width/sequence ≥512. Shared
intermediates, incompatible precision/shape, concurrent streams, foreign device
buffers and unsafe aliases fall back. No API/UI surface or production default
was added.

## Numerical checks: synthetic data

13 graph fixtures, run with the flag off/on, produced **594,411,008 identical
output bytes**, SHA-256:
`9b4a7dc9ee56f8d3638e1c9e5e4301a336e397eac914446c4e65986556cb017b`.

Cases include positive, negative, zero and tiny strengths; attention rescaling;
three successive adapter sums; separate output with nonzero bias; unmarked,
retained-intermediate, forced-F32, short-sequence and large-rank fallbacks; and
an actual-size 24,576 × 5,600 gate/up output (525 MiB). Inputs intended for
post-execution checks were explicitly retained. The first fixture revision had
omitted that lifetime declaration and failed its control assertion; this was
corrected before the final off/on comparison.

CUPTI confirms **0 fused calls off / 10 on**, exactly the eligible fixture cases,
and zero dropped records. Thus equal bytes are not merely an inactive flag.
The final candidate was retested after its concurrency/device fallback guards.
These are synthetic tensor checks, not an end-to-end image-identity comparison.

## Live model comparison

Both final modes used **the same candidate executable**, SHA-256:
`8982e808790507113edb287c7754b115f1cb4f31b00fc1cd7339013abc070f9e`.
Frozen v2 remains
`9fd49b0e5cb3426043ebd85271ad167555551ac65252590cde5ac25bed44bbb0`.
A preliminary control from an earlier guard revision is excluded from these
results. The final enabled series ran first, followed by the final disabled
series, each in its own owned server process.

The original eight-request sequence is unchanged in each mode. Same Qwen Image
2.1 INT8 ConvRot diffusion/encoder, frozen VAE, source-defined adult-woman portrait,
only `lenovo_qwen21.safetensors` at strength 1, CFG 1, seed 42, 1600×896,
40 ER-SDE/beta steps, CPU RNG, empty negative prompt, no references. All 29 frozen
optimization settings, auto-fit and conditioning cache capacity 4 are unchanged.
The only inference switch varied is `GGML_CUDA_FUSED_LORA_OUTPUT`.

| Request | Flag off, s | Flag on, s |
| --- | ---: | ---: |
| Without LoRA, first | 23.852 | 23.661 |
| Without LoRA, warm 1 | 20.525 | 20.441 |
| Without LoRA, warm 2 | 20.558 | 20.152 |
| With LoRA, first | 45.447 | 39.459 |
| With LoRA, warm 1 | 44.553 | 40.549 |
| With LoRA, warm 2 | 44.669 | 39.234 |
| LoRA removed, first return | 20.926 | 20.755 |
| LoRA removed, warm return | 20.472 | 20.968 |
| **With LoRA, warm mean** | **44.611** | **39.892** |

All 16 requests returned HTTP 200; both traces have zero dropped records. CUPTI
was enabled identically in both modes. These are client wall times, including
image encoding and response transport. Two warm samples per mode establish a
local measurement, not a universal performance guarantee or an unprofiled timing.
The roughly 0.245 s no-LoRA mean difference is not claimed as an optimization gain.

## Kernel evidence

Per warm request, counts are stable across both repeats:

| Operation | Flag off calls | Flag on calls |
| --- | ---: | ---: |
| New fused LoRA output | 0 | **7,680** |
| Separate F16→F32 conversion | 15,665 | 7,985 |
| Separate scale (excluding attention base scales) | 16,156 | 7,196 |
| Separate addition | 8,157 | 477 |
| F32→F16 conversion | 15,360 | 15,360 |
| Other GEMMs, including LoRA | 15,750 | 15,750 |
| Sage attention core | 1,280 | 1,280 |
| Paired Q/K RMSNorm+RoPE | 1,280 | 1,280 |

The change replaces **7,680 conversions, 8,960 scales and 7,680 additions** with
7,680 fused calls: six LoRA projections × 32 layers × 40 steps. The extra 1,280
absorbed scales are the attention LoRA up-projection's output rescale. The
attention base scales and other existing fused/unfused paths remain as before.

Summed GPU time in the affected conversion/scale/add groups, including their
replacement kernel, falls **15.582 → 10.733 s** (−4.848 s). The new kernel itself
costs 10.064 GPU s; the removed operations' durations cannot all be counted as
saved time. Total kernel time falls **43.207 → 38.529 s** (−4.678 s), closely
matching the **4.720 s** wall improvement. The union of measured GPU activity
intervals falls 43.537 → 38.851 s.

Warm host→device, device→host and device→device copy counts and bytes are exactly
the same in both modes. This improvement is not a change to offloading policy.
Peak VRAM and generated-image equality were not measured.

## Reproduction and validation

- [Build/flag instructions](../../../native/lora-output/README.md).
- [Candidate source lock](../../../native/lora-output/source-lock.json) and complete
  overlay on the unchanged published base patches.
- [Synthetic evidence](../../../native/lora-output/synthetic-results.json).
- [Compact numeric model results](../../../native/lora-output/results.json).
- Tools: `test-lora-output.py/.cpp`, `compare-lora-cfg1.py` and
  `summarize-lora-fusions.py` in this directory.

Final local traces: `runs/lora-cfg1-2qo8sxe8/` (on) and
`runs/lora-cfg1-wp3nkzzx/` (off), ignored by Git. Native logs and image response
bodies were discarded; no existing job history, prompts, images or runtime
metadata were read. The user-approved exception covered the new kernel/copy
measurements only. Python is build/test tooling; inference is native C++/CUDA.

Full native build, clean patch restoration, source/frozen-binary hashes,
synthetic numerical/dispatch checks, Python syntax, matched live runs and
`git diff --check` pass. Existing CPU enum/CUTLASS warnings remain; no build error
remains. No broader Rust/UI checks were run because those paths did not change.
CPU, other GPU architectures and multi-GPU execution were not live-tested.
Git/GPU and owned-build-control sandbox escalations were approved. No existing
user service was stopped; each owned benchmark server was stopped afterwards.
