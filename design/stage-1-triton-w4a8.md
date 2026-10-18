# Stage 1: original Triton grouped W4A8/W6A8 path

Status: the integrated Rust frontend, compiler, registered wrapper and separate
Python-free comparisons pass. Stage 1 is not complete.

The frozen path first validates packed geometry and operands. A correction
tensor selects the original eager W4A8 body. Otherwise the original Triton
unpacker constructs an INT8 weight, then calls the same Triton INT8 wrapper and
two autotuners used by direct INT8 calls. This source-selected eager branch is
required parity and is not a substitute for the accelerated no-correction path.

The Rust unpacker retains both nibble and six-bit upper planes, per-group scale
loads, optional codebook gathers, libdevice round-to-nearest, saturation and
masked paired output stores. Source locations and helper evaluation order are
preserved. No saved IR or specialization binary is an engine input.

All 24 source/optimized-TTIR cases match in a Python-free namespace, covering
both bit widths, both codebook specializations, dynamic/constexpr-one/wide scalar
profiles and both pointer-alignment states. All four selected complete compiler
paths match original TTGIR, LLVM IR, PTX and CUDA binary bytes. Compiler-only
six-bit/codebook combinations do not imply that original wrapper validation
accepts that combination.

All 42 ordinary records and 66 observed boundaries match, including exact output
bytes, errors, ordered arguments, metadata, strides, gradient flags, aliases and
object identities. Successful ordinary calls repeat exactly. Unwritten
output/codebook storage is never read. Both default INT8 autotuners retain the
original state transitions and scoped timing/winner criterion.

The four complete binaries also match through the integrated on-demand compiler;
memory-cache identity and fresh disk-cache restoration pass. Both W4A8 test
functions pass separately in Python-free GPU isolation. The affected INT8 wrapper
also passes again after the shared ATen identity correction below.

The boundary comparison exposed a real shared binding gap: ordinary ATen returns
lost existing tensor object identity. `THPVariable_Wrap` reuses an object for the
same TensorImpl; the Rust boxed ABI now interns returned TensorImpl identities
against recursive inputs and other outputs. Shared storage with a different
TensorImpl remains a distinct object. Eight independent original CPU records
cover contiguous/to, views, detach/alias, splitting and repeated list entries;
all pass in Python-free isolation. Object attributes survive identity returns.
This implements original behavior and adopts no deviation.

The physical master now holds **35,165 records**. Four focused isolation reports
cover the two W4A8 functions, ATen identity and the INT8 wrapper regression. The
compiler namespace contains eight verified compiler-only files at reference
paths. This is not a new full checkpoint; the latest full result remains MXFP8
117 crate / 109 unfiltered primitive functions.

Owned evidence is under `/tmp/imagine-rust-comfy-parity-20261017/runtime`:
`stage1-triton-w4a8-draft-full.log`, `triton-w4a8-frontend-isolation.json`,
`stage1-triton-w4a8-wrapper-reference.log` and `triton-w4a8-bindings.json`.
Integrated evidence adds `stage1-triton-w4a8-integrated-full.log`,
`stage1-triton-w4a8-registered.log`, `stage1-triton-w4a8-wrapper-focused.log`,
`stage1-triton-integer-isolation.log` and four `integer-*-isolation.json` reports.
No new approval or deviation is required by the implemented source behavior.

The subsequent combined [126/118 full checkpoint](stage-1-triton-integer-checkpoint.json) verifies the integrated source, including the shared ATen identity correction. This remains a partial Stage 1 result.
