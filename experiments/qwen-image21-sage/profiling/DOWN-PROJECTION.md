# INT8 down-projection and F32 dequantization fusion — 2026-10-13

**Historical experiment:** the initial image comparison was blocked by VAE
memory instability and the candidate was archived. The subsequent
[comparison with corrected auto-fit](DOWN-AUTO.md) completes all three pairs,
verifies actual dispatch, and retains the candidate as an opt-in. The evidence
below describes the original pre-fix builds and failures.

## Scope

This experiment extends the existing CUTLASS fused Linear path to Qwen's
K=12288, N=4096 down-projection, behind `GGML_CUDA_FUSED_INT8_DOWN=1`.
The final candidate requires at least 4096 rows and an Ada (SM89) device.
The caller still requires ConvRot group size 256, channel scales, contiguous
supported tensors, and no bias; unsupported cases retain the original path.
The default remains disabled. No public API, Rust code, attention, weights,
quantization rules, F32 output precision, or existing fusion flags change.

The same 128×256×64 CUTLASS tile, 64×64×64 warp, three pipeline stages, and
16×8×32 INT8 Tensor Core instructions are used. The epilogue converts the
INT32 accumulator to F32, multiplies by the input row scale, then by the
weight channel scale in the same order as the existing separate kernel.
It avoids writing an INT32 intermediate to global memory and reading it back.

Only `ggml/src/ggml-cuda/int8-gate-up.cu` changes in the engine relative to the
accepted reference. Source hashes check that previous optimizations remain
unchanged. CLI and server are rebuilt; server requests are not live-tested.

## Synthetic method

The existing native harness executes the actual GGML ConvRot → INT8 GEMM →
F32 output graph. It uses synthetic F32 inputs, INT8 weights, and non-power-of-two
channel scales, including zero, tiny, large finite, and outlier input rows.
It compares the preserved pre-experiment executable with candidate fusion off
and on. All output F32 values, quantized values, and row scales are hashed;
F32 outputs and scales must be finite. Large temporary arrays are deleted.
No model activations are written to disk.

Each measurement uses 20 warmups and seven synchronized batches of 30 calls;
reported milliseconds are medians of batch means. Variant order rotates across
shapes and is reversed on repetition. Separate five-call CUPTI traces verify
actual fused/fallback dispatch and output hashes, with zero dropped records
and no events crossing timing boundaries. The harness measures one Linear,
not a complete FFN or an image.

The desktop shares the GPU and clocks are unlocked. Differences between runs
of the unchanged fallback show substantial timing variability; isolated kernel
trace times are not substituted for the unprofiled whole-chain measurements.

## Initial sequence-length guard

An initial candidate used the existing projection's 512-row floor. For the
three image-related sizes, M=5600/5657/5830, it improved whole-chain time in both
orders. The reverse-order repetition measured 3.0190 → 2.8780 ms,
3.0513 → 2.9048 ms, and 3.3030 → 3.0886 ms respectively (4.7–6.5%).

However, M=513 regressed in both sweeps: 0.3995 → 0.4933 ms and
0.3421 → 0.3988 ms. M=512 and M=1024 improved on repetition. This is sufficient
reason not to inherit a general 512-row guard for down-projection. The final
candidate conservatively requires 4096 rows and tests both that boundary and
its row tail. This is a bounded eligibility policy, not a proven optimal cutoff
for every sequence length.

The initial short-shape sweep also covered M=1/3/57/230/511 and unchanged
4096-wide projection/gate-up controls. All outputs were bit-identical.
Initial binaries, source, patch and hashes remain under
`runs/down-projection-build/initial-512-floor/`; later tests use the final build.

## Evidence locations

Relative to `experiments/qwen-image21-sage/runs/`:

- `down-projection-build/`: frozen reference and candidate, source diff, build
  log, hashes, sanitizer and full-image profiling drivers.
- `20261013T124150Z-down-projection-qaib28er/`: first image-shape microbenchmark.
- `20261013T124233Z-down-projection-tkls4hz1/`: reverse-order image-shape repeat.
- `20261013T124324Z-down-projection-8ieas40a/`: initial short shapes and controls.
- `20261013T124420Z-down-projection-e_809pg5/`: repeat around the initial floor.
- `20261013T124534Z-down-projection-27w7onz0/`: final eligibility boundary and
  image-shape validation.

GPU execution requires sandbox escalation. No dependencies are installed or
updated, and no commit or push is performed.

## Final candidate: isolated validation

All six cases in the final-build sweep are bit-identical to the preserved
original and the same candidate with fusion disabled. Each variant checks
120,320,000 F32 output values, 360,960,000 INT8 values, and 29,375 row scales.

| Rows | Preserved original (ms) | Fusion off (ms) | Fusion on (ms) |
| ---: | ---: | ---: | ---: |
| 4095 (fallback) | 2.5748 | 2.5542 | 2.2739 |
| 4096 | 2.3593 | 2.4718 | 1.9895 |
| 4097 | 2.1653 | 2.3942 | 1.9715 |
| 5600 | 3.4156 | 3.4674 | 2.9116 |
| 5657 | 3.1387 | 3.0926 | 2.9055 |
| 5830 | 3.2664 | 3.2963 | 3.1035 |

The 4095-row case executes the **same fallback** in all three variants; its
apparent timing improvement illustrates shared-GPU variability, not a fusion
benefit. The three large image sizes improved in both earlier sweep orders
and in the final build. Image measurements, rather than these percentages,
determine the end-to-end benefit.

Compute Sanitizer memcheck reports **0 errors** and racecheck reports
**0 errors / 0 warnings** for the first fused CUTLASS launch at M=4097,
K=12288, N=4096, including the row tail. Only that kernel launch is instrumented;
this is not a sanitizer run over the complete model. Logs and the exact filter
are saved under `down-projection-build/sanitizer-{memcheck,racecheck}/`.

## Reproduction

The scripts pin or copy the frozen final candidate. GPU tests run sequentially.
Existing optimization flags stay enabled, with 11 CPU threads and 512 CUDA
threads for K=4096 ConvRot. K=12288 ConvRot remains at 1024 CUDA threads.

```sh
python experiments/qwen-image21-sage/profiling/compare-down-projection.py \
  --rows 4095 4096 4097 5600 5657 5830 --reverse
python experiments/qwen-image21-sage/runs/down-projection-build/sanitize.py
python experiments/qwen-image21-sage/runs/down-projection-build/profile-images.py
python experiments/qwen-image21-sage/profiling/compare-down-images.py
```

The sanitizer driver creates fixed log directories and is intended to be run
once per saved build; preserve or rename its existing result directories before
repeating. The other drivers create separate result directories automatically.

Final candidate CLI SHA-256:
`1be5b3fb3a5e677fa6e02ff16cfd2add4c7259533f1e5006b2a81f8d82daf597`.
Preserved reference CLI SHA-256:
`142ccc876d5f0c5e3c265f60030338ac82c558e4a9bdaec646f760a4d03da4d7`.

## Full-model issue at the saved 512 MiB margin

The candidate with fusion disabled completed a profiled reference image in
61.49 s process time (56.08 s sampling), with the exact established RGB hash.
The analysis counted 80 diffusion forwards, 2560 Sage kernels, 2560 fused
gate/up kernels, 10240 fused ordinary projections, and 2560 separate down GEMM
and dequantization kernels. The trace had zero dropped or boundary-crossing
records. This is one profiled control, not a new timing baseline.

With down fusion enabled, **two profiled attempts and one unprofiled attempt**
completed sampling but aborted with CUDA out-of-memory during VAE decoding.
Sampling log times were 55.66 s, 55.55 s, and 55.39 s. No output image was saved.
These incomplete runs cannot establish an image speedup or image equivalence.
Aborted traces did not flush complete CUPTI evidence and are not analyzed as
valid full-model profiles. The planned three unprofiled image pairs were initially deferred after these
failures, pending a successful complete run.

The successful off run evicted six GPU parameter blocks before VAE execution;
each failed on run evicted five. The failing allocation is the CUDA VMM scratch
pool's `cuMemCreate`, called from `ggml_cuda_mul_mat_cublas` during VAE. It is
not a CUTLASS down-projection launch failure. Source inspection shows that
`ModelManager::check_capacity` includes a fixed 512 MiB reserve for backend
scratch/pipelines, which are not included in graph measurements. A change in
scratch residency crossing the coarse eviction threshold is a plausible
explanation, but the exact allocation sizes and causal chain have not been
instrumented. Shared desktop VRAM also varied; no unrelated process was stopped.

Evidence:

- `20261013T124659Z-activity-7y6stqvn/`: successful fusion-off profile/image.
- `20261013T124801Z-activity-7pmqe0fp/`: first fusion-on VAE failure.
- `20261013T125025Z-activity-ddj8ocz7/`: same-settings profiled reproduction.
- `20261013T125217Z-control-0ja1dvbf/`: unprofiled reproduction.
- `down-projection-build/vae-investigation.json`: exit status, sampling logs,
  and pre-VAE eviction evidence.

### Default auto-fit follow-up

At the user's request, `run.py --default-vram` removes the saved
`--max-vram -0.5` argument while retaining `--auto-fit on`. Historical replay
commands remain unchanged unless this option is specified. CLI default
`max_vram` is `0`, and `backend_fit.cpp::enumerate_gpu_devices` also subtracts
512 MiB when this default is used. The planner then propagates that budget to
the runners. Thus omission does not request a larger reserve or remove all
internal 512 MiB reserves. The separate ModelManager scratch reserve remains
unchanged as well.


The first default-auto-fit candidate run succeeded:
`20261013T125535Z-control-q18juv0g/`, 55.51 s sampling, **60.5003 s to saved PNG**,
60.6589 s process wall time. Its 1600×896 RGB pixels match the established
reference exactly. It evicted five parameter blocks before VAE, whose reported
compute buffer remained 10632.62 MB.

The planner reported 14062 MiB free and a 13550 MiB budget, confirming that
auto-fit still reserved 512 MiB. The sampled desktop baseline was 1713 MiB,
compared with 1989 MiB at the first failed on run. The success therefore does
**not** establish that removing the argument fixed the problem: available
memory changed too. Peak sampled whole-GPU usage was 15760 MiB.

The subsequent image comparison uses the same default-auto-fit configuration
for both off and on, and is separate from the previous 512-MiB-explicit reference:

```sh
python experiments/qwen-image21-sage/profiling/compare-down-images.py --default-vram
```


The default-auto-fit A/B driver stopped during its first warmup, **with down
fusion disabled**, before any measured pair. That run,
`20261013T125745Z-control-b4wcvug5/`, completed sampling in 56.33 s and then hit
the same VAE `cuMemCreate` out-of-memory error. It reported 14065 MiB free,
13553 MiB budget, a 1710 MiB initial desktop baseline, and five pre-VAE
parameter-block evictions. Similar initial free memory therefore does not
ensure success. The plan and aborted driver log remain under
`20261013T125745Z-down-images-26kgzwkg/`.

This reproduction corrects the initial suspicion of a fusion-specific
regression: the unfused path can fail too. The precise relationship between
scratch-pool residency, eviction thresholds, and live free VRAM remains
unresolved. No unrelated application was stopped, and neither the internal
safety margin nor the memory manager was modified.

## Disposition and next diagnostic

The active helper was restored byte-for-byte to its pre-experiment source.
All other engine source hashes are unchanged. Both active CLI and server were
rebuilt successfully; `restore-validation.json` records the source checks and
binary hashes. The experimental down-fusion flag is supported **only by the
preserved candidate**, not the restored active engine.

The rebuilt active CLI hash is
`191d818a5885064121563fdb175896c2f8e4a23489a3052a7fe180d32cb50e19`.
No separate image was run through that restored executable after rebuilding;
the same candidate with fusion off exercised the reference computation, but
its last auto-fit image failed in VAE. Server requests were not live-tested.
The earlier 59.61 s accepted image reference has not been replaced.

Two complete images from this experiment matched the established RGB SHA-256
`253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2`:
one profiled off image and one unprofiled default-auto-fit on image. They are
not a matched timing pair. There is **no measured mean image speedup** for this
candidate, and three-pair image validation remains incomplete.

The next useful diagnostic is to record the VAE's actual pending CUDA scratch
allocation, existing pool capacity, live free VRAM, and parameter evictions
at the failing allocation. This should precede a memory-manager change or
another down-projection image comparison. Merely omitting `--max-vram` does
not change the planner's default 512 MiB reserve. A larger explicit budget
margin could be a separate diagnostic workaround, but was not tested here.

## Follow-up: VAE allocation measured

The subsequent [VAE memory diagnosis](VAE-MEMORY.md) directly measured the
scratch allocation and its shortfall on the accepted engine. A 787.5 MiB FP16
output temporary requires the existing pool to grow by 696 MiB, beyond the
manager's 512 MiB allowance. One separate 1024 MiB margin control succeeds with
six parameter-block evictions and identical pixels. No permanent memory fix
or resumed down-projection timing comparison has been made.
