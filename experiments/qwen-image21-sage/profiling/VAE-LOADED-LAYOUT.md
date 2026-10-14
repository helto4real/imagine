# Loaded VAE graph layout: the discrepancy is fully accounted for

Date: 2026-10-14. Diagnostic investigation only; no inference, allocator policy,
memory-fitting or optimization behavior changed.

## Finding

The earlier metadata probe selected the wrong convolution construction path.
Its dummy `decoder.conv1.weight` entry had `ne[2]=64`, forcing `is_2D=true`.
The real checkpoint's registered tensor is `[3,3,1,73728]`, so the runner chooses
`CausalConv3d`. Its singleton temporal kernel is then lowered to a 2D convolution,
with **explicit spatial padding before im2col**. The two paths have different
graph topology and allocation lifetimes even though both process one image.

The old probe's control workspace happened to equal the live control exactly.
That was insufficient validation of the probe. Its fused graph underestimated
the loaded graph by **1,657,262,592 bytes / 1580.488770 MiB**.

A corrected probe, populated with the 128 parameter shapes captured from the
loaded graph, now reproduces both live graphs **byte for byte in the metadata
dump**, including operation parameters, shapes, strides, flags and dependencies.
All **513 control / 511 fused allocation entries**, including buffer/chunk,
offset and size, also match exactly. No checkpoint values are needed to reproduce
the allocation discrepancy. There are **zero unexplained admission bytes**.

| Graph planner result | Control | Fusion |
| --- | ---: | ---: |
| Earlier probe, forced 2D path | 10632.617432 MiB | 11026.367432 MiB |
| Loaded graph | 10632.617432 MiB | 12606.856201 MiB |
| Corrected probe using captured shapes | 10632.617432 MiB | 12606.856201 MiB |

The actual fusion regression is **2070139392 bytes / 1974.238770 MiB** versus
the accepted control, rather than the old probe's 393.75 MiB regression.
These are planned graph allocations. The fused full-image request is rejected
before allocating/executing that graph; tiled fallback is a separate graph.

## Why removing a buffer increases the arena

The allocator chooses reusable free regions based on tensor lifetimes and
contiguous available space. Removing the upscale intermediate changes those
regions; the smaller local tensor set does not guarantee a smaller arena.

The following offsets are measured within the CUDA graph arena, not absolute
GPU addresses. Their sizes are in MiB (1024² bytes):

| Allocation | Control offset | Fused offset | Size |
| --- | ---: | ---: | ---: |
| Resampling im2col | 3545.117432 | 1970.117432 | 7087.500000 |
| Resampling GEMM output | 788.867432 | 9057.617432 | 1575.000000 |
| Following residual convolution's scaled padded input | 1576.367432 | 3938.867432 | 1580.488770 |
| Following residual convolution's im2col | 3156.856201 | 5519.356201 | 7087.500000 |

At the fused resampling im2col, the high-water mark does fall from 10632.617432
to 9057.617432 MiB. However, its 1575 MiB GEMM output no longer fits into the
same reusable region and is placed at the end, restoring the previous peak.
The changed placement propagates into the following residual convolution,
`decoder.upsamples.4.upsamples.0.residual.2`.

That convolution explicitly pads its F32 input from `[1600,896,1,288]` to
`[1602,898,1,288]`, requiring 1580.488770 MiB. In the fused layout, the subsequent
7087.5 MiB im2col begins at 5519.356201 MiB, raising the arena to
**12606.856201 MiB**. In the control, the same im2col ends at 10244.356201 MiB,
within the earlier high-water mark.

The old probe uses `Conv2dBut3d` with permutations/contiguous copies and im2col
padding parameters instead of the checkpoint's `CausalConv3d` path with an
explicit PAD tensor. Its aggregate operation counts differ by 80 PERMUTE and
80 CONT nodes versus 120 RESHAPE and 36 PAD nodes; the net node-count difference
is only four. The apparent closeness in node counts and equal control peak
masked a materially different allocation plan.

## Exact admission arithmetic

The diagnostic captured every component of the live fused request:

| Component | Bytes | MiB |
| --- | ---: | ---: |
| Planned VAE graph | 13219246848 | 12606.856201 |
| VAE weights requiring device staging | 506264448 | 482.811401 |
| Additional CUDA scratch-pool growth | 780140544 | 744.000000 |
| Existing internal safety reserve | 536870912 | 512.000000 |
| **Required device capacity** | **15042522752** | **14345.667603** |

The CUDA pool already holds 44 MiB, which is already excluded from driver-reported
free memory. It must not be charged again as new device demand. The separate
managed-budget calculation includes that resident pool and excludes the device
safety reserve:

`12606.856201 + 482.811401 + 44 + 744 = 13877.667603 MiB`.

Both totals reproduce the earlier live failure exactly. The 788 MiB pool estimate
from an initially empty metadata-probe context is consistent with live execution:
44 MiB already resident plus 744 MiB additional growth. The earlier mismatch was
not an unaccounted CUDA workspace or an incorrect safety-margin calculation.
No manual margin has been added or changed.

## Source path

- `src/model/vae/wan_vae.hpp`: `WanVAERunner` chooses `is_2D` from
  `decoder.conv1.weight.ne[2] > 3`; the old probe forced this branch.
- The same file's `CausalConv3d::init_params` recognizes one-frame-deep exported
  kernels; `forward` explicitly pads, reshapes and calls `ggml_ext_conv_2d` with
  padding zero when the kernel/input temporal dimensions are one.
- The resampling block still uses `Conv2d` and is the same targeted fusion in
  both graphs. The divergent construction of surrounding convolutions changes
  later buffer reuse.
- `ggml/src/ggml-alloc.c`: best-fit free-block selection and parent/view reuse
  account for the measured placement. No change to either policy was made.
- `src/core/ggml_runner.cpp` and `src/model_manager.cpp`: graph demand, staged
  weights, additional CUDA pool growth and the internal reserve produce the
  admission totals above.

## Method and validation

An isolated diagnostic executable was linked from copies of the runner,
model manager and allocator sources, using the original compilation flags and
all other existing objects. Instrumentation records only graph metadata,
allocation offsets/sizes and memory-request components. The ordinary source
files, build archives and executable were left unchanged. The diagnostic patch,
commands, copied sources and binary hashes are retained with the artifacts.

The live diagnostic used all accepted optimizations, F32 gate/up, Sage2++,
automatic memory fitting without a manual margin, the approved synthetic prompts,
1600×896, seed 42, 40 ER_SDE/beta steps and CFG 2.5. Retention remained off.
The same diagnostic executable ran control and fusion. Two control images
matched native RGB SHA256 `c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c`;
the first fused image again hit the full-image admission failure, retried nine
spatial tiles and matched the earlier fallback hash
`a80c12191b007ab9f1141f4dd5d20d387cba9282aa52c0364817b7d7218ba9bf`.
Its expected reference-image assertion stopped further requests. All images ran
80 diffusion forwards. Diagnostic timings are not new performance results.

The old and corrected probes ran after both servers had stopped. Their graph
measurements allocate no tensor buffers and execute no model kernels. The corrected
probe's control/fused TSV hashes exactly equal the corresponding loaded dumps:

- Control: `7dbf17fe6f1ecbfd9db4e00d41f80fcad826cd1e7d9ca8502abbf8171a49ed6d`.
- Fusion: `02e55a042663b49a399a8fcb9352cb1c915bb66a788e46e84c6ef4ce55c14519`.

Builds, metadata/allocation equality, arithmetic assertions, PNG hashes and
tracked source/binary hashes were validated. Instrumentation did not inspect or
persist tensor contents or prompt encodings. Private ComfyUI logs, history,
workflows and configuration were excluded.
GPU/loopback diagnostics used approved sandbox escalation. Test servers and
monitors exited; no commit or push was performed. Other checkpoints, resolutions,
LoRAs, tiling policies or allocator strategies were not tested.

## Disposition

Leave the fusion off. The discrepancy is resolved, and the existing admission
check correctly rejects the requested layout under the observed limits. A next
optimization experiment can now use the corrected probe as an exact allocation
reference for this graph and target buffer reuse around the resampling GEMM and
following padded convolution. Such an optimization has **not** been implemented
or tested in this investigation.

## Artifacts

- [Diagnostic build script](build-vae-layout-diagnostic.py).
- [Live diagnostic driver](trace-vae-layout.py).
- [Validation and arithmetic analysis](analyze-vae-layout.py).
- [Source copies, instrumentation patch, probe sources and build commands](../runs/vae-layout-diagnostic-build/).
- [Loaded graph dumps, corrected/old probes, parameter-shape fixture and analysis](../runs/20261014T143229Z-vae-layout-ggy122ez/).
