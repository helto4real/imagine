# Stage 1 convolution and GroupNorm/padding backend bindings

Date: 2026-10-17. **Eight more bodies connected: 89 total, 46 remaining.**
Full verification passes **81 crate tests / 73 Python-free primitive functions**,
including **1,434 new records**, for **17,096 ordinary records** and unchanged
**3,572 controlled combinations**. The full unfiltered verification is recorded
in [the evidence](stage-1-evidence.json). Stage 1 remains incomplete. The preceding checkpoint is
[RoPE/RMS-RoPE](stage-1-rope-bindings.md).

## Original execution behavior

[`backend_conv.rs`](../inference/src/backend_conv.rs) connects eager and CUDA
`fp16_conv3d`, `fp16_conv3d_out`, `group_norm_silu_pad3d` and
`group_norm_silu_pad3d_out`. It calls the original compiled Kitchen kernels and
the pinned ATen operations; no replacement kernel/backend is selected.

The eager convolution preserves conv3d, optional residual addition and final
channels-last-3d materialization. CUDA preserves the original FP16/channel/
filter/stride/bias/residual gate, channel counts below eight zero-padded to eight,
NDHWC view strides, alignment checks, optional out-buffer validation and the
original CUTLASS return-or-decline decision. A declined allocation is never read.
Fallback uses the original caller inputs. The allocating CUDA fallback converts
the convolution before residual addition; eager converts after addition, and
the CUDA out variant adds before copying. These distinctions are retained.

GroupNorm remains per frame, with the original weight/bias conversion order,
optional SiLU, spatial reflection or zero border and zero frames in front.
`zero_pad=False` means reflection, not unwritten padding. The eager path keeps
its Python group-norm batch-size check, ATen normalization and final layout;
a nothing-to-do call still returns a clone rather than aliasing the input.

CUDA retains the source channel/group/grid/reflection/alignment gates, original
short-circuit division errors, prepared affine parameters and source eager
fallback. Its supported branch uses the original partial/finalized statistics
and fused apply kernels, workspace size and current stream. Affine statistics
and every live output vector, including front and border padding, were audited
in `group_norm_pad3d.cu` before accepting results. Comparisons are against the
same source backend; source eager and CUDA arithmetic are not assumed identical.

The original zero-vector cache is keyed by length, device and dtype. Empty
placeholders reuse the existing CUDA empty-tensor cache. Caller out buffers are
written directly only where the source allows packed NDHWC/alignment; other
buffers receive the original temporary-and-copy path. `_out` returns None;
CUDA's optional `out` on the allocating GroupNorm entry returns the caller's
original object on both direct and copy paths.

## Exception ABI correction

A focused negative case found that the standalone generic native call erased
`std::invalid_argument`, which Nanobind translates to `ValueError`. The bridge
now transports that class, and the corresponding `std::out_of_range` /
`IndexError` class, separately from RuntimeError. The original C++ wrapper still
performs the same checks at the same point; no checks are moved into Rust and
no invalid kernel is launched. Only exception transport changes. No kernel,
generated original wrapper, compiler flag or algorithm changes.

The first focused run passed convolution but rejected the class mismatch in
GroupNorm's invalid affine-size case. The final focused run passes both tests;
the previous failure is superseded, not accepted or weakened. The prior bridge,
fixture and successful isolation report are retained in the isolated development
artifacts. The final full-suite run rechecks previous native boundaries too.

## Python-free dependency closure

All 81 ordinary crate tests passed. The first unfiltered isolated pass then
aborted at convolution because the initial DT_NEEDED-only exposure omitted
cuDNN's dynamically loaded components (`Cannot load symbol cudnnGetVersion`).
That failed launcher is retained and is not reported as a successful full run.

The isolation launcher now adds only the cuDNN component ELF objects and NVRTC
builtins already present, fingerprinted and version-locked in Stage 0. Each
package file is checked against its existing lock hash before mounting. Their
transitive system ELF files are resolved and mounted individually, including
zlib; the final report records system hashes as well. The isolated library path
includes the pinned cuDNN/CUDA library directories. No package directory, Python
interpreter, ComfyUI source tree, cloud access, alternate library version or new
installation is exposed. This restores the reference's existing dependency
availability; it is not an algorithm or dependency-version change.

The unchanged successful ordinary crate phase is reused. Only the complete
unfiltered isolation phase is repeated after this packaging correction; the
two raw logs and phase outcomes are recorded separately in aggregate evidence.

## Differential coverage

The [source capture](../inference/tools/reference_conv_bindings.py) verifies all
52 locked Kitchen source hashes and calls actual resolved registered functions.

| Category | Cases | Exact outputs | Original errors/selection failures |
| --- | ---: | ---: | ---: |
| Convolution | 351 | 245 | 106 |
| GroupNorm/SiLU/padding | 1,083 | 882 | 201 |
| Total | 1,434 | 1,127 | 307 |

CPU/CUDA eager and CUDA native are exercised across FP32/FP16/BF16, standard and
channels-last layouts, frame-offset windows and deliberately misaligned owned
views. Convolution includes pixel channels, supported channels, non-native
channel/filter counts, stride one/two, batch one/two, 1x1x1/3x3x3 kernels,
bias/residual, invalid stride/filter/channel/out shape, and normal/out entries.

GroupNorm includes native and fallback channel/group shapes, mixed FP32 affine
parameters, absent affine/bias, all norm/SiLU combinations, both padding modes,
front padding, zero padding, nothing-to-do clone, caller out, invalid pad/group/
affine/out shapes and the source's differing tiny-group behavior. Every one of
the eight additional backend-operation entries has actual successful source
numerical cases.

Tests compare exact bytes, dtype/device, shape/strides, requires-grad metadata,
returned object identity, aliases, all post-call arguments and selected backend.
They also compare the **actual native-wrapper call sequence**, preventing a
numerically matching alternate route from passing. Shared output tests compare
the entire initialized backing allocation, including untouched frame halos.
Input storage offsets and their fully initialized backing allocations are
reconstructed so misalignment/fallback behavior is not lost during restoration.

The complete verification comprises successful fmt, Clippy with warnings
denied and all 81 crate tests, followed by the successful unfiltered isolation
rerun of all 73 primitive functions without Python, ComfyUI or network. All four Stage 0 locks, original 32 CUDA objects, generated original
wrappers, dependencies, RNG and autotuning remain unchanged. Existing upstream
CUDAEvent interference-size compiler warnings remain visible.

The additions bring ordinary records from 15,662 to **17,096**, leaving approved
controlled coverage unchanged at **3,572 unique combinations**. No new
unwritten-output criterion, numerical tolerance or deviation approval is used.

## Limits and authority

**46 backend bodies remain: 11 eager, seven CUDA and 28 Triton.** Remaining
layout/signature/availability/standalone Triton/ownership/packaging work remains
required by the design. This matrix does not establish arbitrary sparse,
autograd, out-buffer alias races or every native geometry. No new support gate,
scope exclusion or alternate algorithm is adopted.

Only own isolated synthetic development data is used. No ordinary Dreamchat
runtime data, model/checkpoint, product integration, cloud service, GPU process
listing, service stop, browser/manual verification, commit or push is involved.
Authorized local GPU/build/namespace commands use sandbox escalation; no automatic
approval rejection occurred. Necessary faithful bindings and the exception
transport correction follow Tomas's existing authority clarification and need no
additional deviation approval.
