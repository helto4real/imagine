# Stage 1: original Triton NVFP4 paths

Status: integrated source/frontend, wrappers, repeated compiler failures and
all three focused Python-free test functions pass. No
successful NVFP4 GPU computation or new approval criterion is claimed.

The frozen quantizer packs FP4 with
`cvt.rn.satfinite.e2m1x2.f32`; the inverse uses
`cvt.rn.f16x2.e2m1x2`. The current SM89 target rejects these instructions in the
original `ptxas`. The inverse backend registration additionally requires SM100.
The port must preserve the original registered selection and direct-call error
paths. It must not introduce an eager replacement or capability guard where
the original wrapper does not have one.

The typed Rust transcription retains the original loops, FP32 scale arithmetic,
FP8 scale rounding, blocked scale addressing, 2D one-hot even/odd selection,
sum reductions, nibble order, inline assembly strings, masks and stores.
Unsigned integer widths and shape-aware helper signatures extend the shared
expression mechanism. Kernel-specific control flow remains with each kernel.

## Verified source/compiler evidence

- 108 quantizer specializations: original raw source IR and optimized TTIR are
  byte-exact across three input precisions, 1/2/4 blocks per program, both nibble
  orders, dynamic/one/wide scalars and both pointer-alignment states.
- 36 inverse specializations: the same exact source/TTIR comparison covers
  three output precisions, both nibble orders, scalar profiles and alignment.
- All 18 selected quantizer and six inverse compilations reproduce original
  TTGIR, LLVM IR and PTX bytes through the standalone original compiler.
- All 24 actual assembler failures reproduce the complete original diagnostic
  byte for byte. Each test binds the candidate-generated PTX to the captured
  original temporary input name inside a private tmpfs, then executes the
  original assembler. No message fields are erased, sorted or normalized.
- Those compiler namespaces expose no Python runtime or GPU device. Saved IR
  and PTX are comparison artifacts; they are not Rust frontend execution input.

The draft is built under the test-owned
`/tmp/imagine-rust-comfy-parity-20261017/nvfp4-development` tree while the current
engine's full MXFP8 validation remains immutable. Checked-in engine installation,
wrapper/error/state tests and later full isolation are separate obligations.

Evidence under the same root: `runtime/triton-nvfp4-frontend-isolation.json`,
`runtime/triton-dequant-nvfp4-frontend-isolation.json`,
`runtime/stage1-triton-nvfp4-draft-full.log` and
`runtime/stage1-triton-dequant-nvfp4-draft-full.log`.

The shared-expression draft also passes complete regressions for MXFP8 (54
source/TTIR cases, nine binaries), RoPE (2,160 cases, 72 binaries) and RMS-RoPE
(1,296 cases, 54 binaries). Separate reports end in
`-frontend-nvfp4-draft-regression-isolation.json`; these are explicitly draft
compiler comparisons, not a full engine checkpoint.

## Integrated wrapper verification

The two registered bodies now connect explicitly through
`Runtime::bind_triton_nvfp4`. All three focused crate test functions pass with
all-target Clippy. The integrated frontend also repeats the 144 exact source/TTIR
comparisons and 24 complete source-to-PTX/error comparisons successfully.

There are 14 ordinary wrapper rejection records: eight early errors and six
actual assembler errors, including zero-width and padded-width paths. Each of
those six compiler failures is repeated twice. The original wrapper-generated
PTX is captured without replacing any compiler input; the candidate reproduces
its entire PTX text. Diagnostics retain exact flags, native error text and line
numbers, with both input/output filename parameters bound to the actual owned
work files in the paired comparison. Both original and candidate raw messages
are retained. Compilation failure publishes no metadata/group/cubin and the
next call retries with a new owned work file. The original diagnostic/PTX
printing is retained, without acting on the diagnostic's request to contact
any external project.

The separate 102 pre-run-boundary records compare initialized input/scale bytes,
output metadata without reading uninitialized output, all argument values,
strides, dtypes, identities, aliases and compile-time arguments. They do not
claim a successful FP4 kernel execution. This exposed and fixed the shared
frozen hook boundary's missing `debug` and `instrumentation_mode` keywords;
these now match `JITFunction.run` before hook invocation. No arithmetic or
compile/autotuning policy changed.

All three focused Python-free runtime tests pass separately under exact test
filters. Each isolation exposes the same seven verified compiler-only files at
reference paths, with no Python runtime or model/production data. The launcher
initially copied the prior unfiltered report after passing filtered runs; that
copy bug was fixed, the final correct filtered report was retained, and only
the two missing report-producing runs were repeated successfully. The last full engine
checkpoint remains [MXFP8](stage-1-triton-mxfp8-checkpoint.json), 117 crate / 109
unfiltered primitive functions. A full NVFP4 checkpoint has not been claimed.

Connected registered bodies are now 132/135 with all explicit compiler bindings;
INT8 linear, W4A8 linear and neighborhood attention remain, alongside the general
compiler/state/lifetime/restoration obligations. Oversized zero-width allocation
and non-boolean `hi_first` specializations remain explicit argument gaps. The
source captures device/stream before pre-run hooks; all wrapper capture positions
must also be audited as part of the remaining state work. No scope is removed.

The subsequent combined [126/118 full checkpoint](stage-1-triton-integer-checkpoint.json) verifies the integrated source, including the shared ATen identity correction. This remains a partial Stage 1 result.

## Dynamic constexpr continuation

The Rust wrapper now retains the original `hi_first` value rather than requiring
a bool. None, bool and integer branches retain their original branch behavior,
source AST text and separate Python-equality JIT memory identity. Other types
reach the original `UnsupportedLanguageConstruct`, with source-derived excerpts
and caret positions. List/dictionary hashability is checked before compilation.

A new original CPU oracle has 78 cases. All 78 explicit AST hashes match; 42
complete source/TTIR comparisons and their full TTGIR, LLVM, PTX and SM89
assembler errors match in a Python-free, GPU-free, network-isolated namespace.
The other 36 source compiler exceptions match exactly. The report is
`runtime/triton-nvfp4-literals-isolation.json` in the isolated development root.
No successful FP4 GPU operation is claimed on SM89. Wrapper differential
validation, regenerated-oracle comparison and final integration remain pending.

The additional executed oracle has 90 repeated cases across all three original
backends. Eager retains ordinary Python truthiness, CUDA retains nanobind's
strict bool boundary and exact signature/invoked-type diagnostics, and Triton
retains its original constexpr rules and generated PTX failures. All 90 pass,
including original CUDA decode successes. Eager and CUDA callbacks preserve
where their respective packing/native conversion evaluates the flag. Evidence:
`runtime/stage1-dynamic-nvfp4-literals-focused.log`. No new tolerance is used.

The previously failing SASS dump additionally passes both ordinary and focused
Python-free isolation with the separately locked local nvdisasm dependency.
Its retained report is `runtime/stage1-sass-dependency-isolation.json`.
