# Silent tensor loader wait fix — 2026-10-16

Skipping progress polling when progress display is disabled reduces the first
LoRA request **27.488 → 21.829 s**, saving **5.659 s / 20.6%**. Warm LoRA requests
improve **21.019 → 20.190 s**, saving **0.829 s / 3.9%**. A separate instrumented
run confirms zero polling sleeps in all 576 first-use LoRA operand/scale loads
and all 84 warm loads. Worker execution, joining and source validation remain.

## Change and scope

The only changed native file is `src/model_loader.cpp`. The progress loop now
uses `while (log_progress)` instead of `while (true)`; workers are still joined
unconditionally before checking failure and validating the source files again.
The progress-enabled path keeps its existing behavior. This avoids an unnecessary
10 ms polling sleep for each small silent load, while waiting for actual work
through `join()`. It does not skip work or validation.

The fix applies to all silent loads in this separate experimental binary,
whether the INT8 merge flag is on or off. It does not change precision, rounding,
CUDA kernels, reader construction, conditioning caching, VRAM policy or engine
selection. `SD_QWEN21_MERGE_LORA_INT8` remains default-off. The frozen production
control and the ignored primary native checkout are unchanged.

## Matched workload and wall times

Qwen Image 2.1 INT8/ConvRot, the same encoder and BF16 VAE, 1600×896, 40 ER-SDE /
beta steps, CFG 1, seed 42, one image, no references, and the source-defined adult
woman portrait. Only `lenovo_qwen21.safetensors`, strength 1. All frozen v2 native
switches, CPU RNG, thread count and memory settings remain unchanged.

| Path | No LoRA, warm mean | LoRA, warm mean | First LoRA request |
| --- | ---: | ---: | ---: |
| Native before, preserved control | 19.905 s | 21.019 s | 27.488 s |
| Native after, final binary | 19.892 s | 20.190 s | 21.829 s |
| Latest Comfy timing-only diagnostic, reused | 20.282 s | 20.195 s | 20.460 s |
| Earlier Comfy kernel/copy-only control, reused | 21.063 s | 20.820 s | 21.282 s |

The final native series contains eight requests: three without LoRA, three
with it, then two without again. Both warm LoRA samples are 20.190 s; the final
warm request after removal is 19.878 s. All HTTP statuses are 200. Warm LoRA costs
0.298 s / 1.5% over this run's own no-LoRA mean.

Neither Comfy control was rerun in this slice. The latest Comfy control has
additional host timers; the earlier one has only kernel/copy profiling, like
the final native series. Against them, the remaining first-request gap is
respectively **1.369 s** and **0.547 s**. Treat these as local comparisons with
run-to-run variation, not simultaneous measurements or a universal speed claim.
Native's first-minus-warm overhead is 1.639 s versus the latest Comfy's 0.265 s:
the extra activation gap is now **1.374 s**. Native HTTP/base64 and Comfy in-memory
PNG handling, RNG and other pipeline differences still apply.

## Independent timing-only confirmation

The separate post-fix diagnostic completes the same eight requests, measuring
21.842 s first / 20.166 s warm. It is not used for the headline native wall times.
No build overlaps either measured series.

| Phase | Previous diagnostic | Post-fix diagnostic |
| --- | ---: | ---: |
| Polling sleep in first LoRA loads | 5.836 s / 576 sleeps | 0 s / 0 sleeps |
| Sleep after workers finish, without concurrent GPU work | 5.617 s | 0 s |
| Full first merge scope | 7.254 s | 1.609 s |
| Full warm merge scope | about 1.080 s | 0.245 / 0.251 s |

The remaining first-load operands/scales take 0.979 s, including 0.661 s in
repeated reader/catalog construction. Their workers take 0.167 s; joining now
waits for actual workers and totals 0.226 s. These are nested/overlapping scopes,
so do not add them. The complete CUDA helper scope is 0.565 s across 192 matrices;
conditioning is 0.310 s. Reader/cache policy and these operations were not changed.

All inference fusion counts remain equal to the no-LoRA path on all 16 requests:
1,280 INT8 gate/up+SwiGLU, 1,248 attention-residual, 1,248 down-residual, 3,840 BF16
output, and 2,496 norm/modulation-ConvRot calls. There are no runtime LoRA output
kernels. Both CUPTI checkpoints report zero dropped records. First preparation
still covers 192 matrices, and each warm request covers 28.

## Verification and provenance

- Full native build and clean patch replay/source hash verification pass.
- Fifteen synthetic CPU/GPU/disk lifecycle cases pass: reuse, independent
  eviction, strength changes, restoration and removal, including gate/up forms.
- All pre-existing candidate source hashes are identical to the preserved
  control. The complete overlay and lock now also include `model_loader.cpp`.
  The previously successful numerical precision suite is reused because all
  precision implementation files are byte-identical; it was not rerun.
- All 16 real requests succeed; no generated image, ordinary log or submitted
  prompt was inspected. Image quality and pixel identity were not tested.
- GPU execution used approved sandbox escalation; no automatic approval-review
  rejection occurred. No new commit or push was made.

Final binary SHA-256:
`f5ab9ed7e583d2d85fad917efa0338ade07da3629b249b12ebf7b95f2ec2d116`.
Source-lock SHA-256:
`990b8e4c4e4a4dbe9f9dfe35dd61d662814ecbe686e6baff0b89503208ef3de7`.
Frozen v2 SHA-256:
`9fd49b0e5cb3426043ebd85271ad167555551ac65252590cde5ac25bed44bbb0`.

Final run: `lora-cfg1-pr448kaa`. Separate diagnostic: `lora-cfg1-hhh6_7o9`.
The previous whole-matrix binary, loader source, patch, lock, manifest, reports
and diagnostic executable are preserved in `.imagine/lora-merge/whole-matrix-control/`.
The original investigation remains in [ACTIVATION-INVESTIGATION.md](ACTIVATION-INVESTIGATION.md).
Numeric results, phase counts, exact provenance and script hashes are in
[loader-wait-results.json](loader-wait-results.json). Rebuild and matched native
benchmark commands are in [README.md](README.md).
