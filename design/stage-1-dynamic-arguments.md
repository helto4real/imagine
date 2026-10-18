# Stage 1: dynamic arguments and constexpr identity

This continuation implements frozen source behavior within the authorized full
Stage 1. No tolerance, kernel algorithm, reference, dependency substitution or
autotuning-policy change is introduced. Stage 1 remains incomplete until its
remaining obligations and final integrated isolation/restoration checks pass.

## Verified focused comparisons

The INT8 default-dtype matrix grows from 16 to 24 original cases, all passing
twice: all four floating default dtypes, tensorwise/per-channel weight scaling,
and absent/FP32/FP64 bias. The original Triton constraint omits bias, so FP64 bias
must remain accepted even for lower-precision output. FP64 conversion is local
to INT8 output/bias metadata, rather than silently enabling unsupported types in
all other frontend families.

56 original wrapper cases pass twice across warm and cold cache sequences.
All 33 observed actual AST/compiler keys match candidate cache directories.
Eight original SM89 NVFP4 assembler failures compare their complete generated
PTX and every diagnostic byte, with only the paired temporary input/output file
names bound to their respective environments. The matrix additionally compares
integer/None causal flags, unsupported conditional objects, boolean/float kernel
sizes, tensor constexpr iteration, structured/pointer scales and each backend's
distinct explicit-None NVFP4 decode behavior. Unsupported-language errors retain
their original class, source excerpt, caret and cause.

The original JIT memory cache uses Python value equality, independently of its
ASTSource digest. True, 1 and 1.0 can share a constexpr specialization. A float
kernel-size call following a boolean call can therefore succeed through warm
reuse, while the same call cold raises the original loop-bound CompilationError.
Rust now preserves both identities: Python-equivalent memory lookup first,
unchanged AST/compiler keys and metadata on a cold miss. No failed compilation
publishes a kernel. This behavior is reference compatibility, not a new cache
optimization or relaxation.

Six additional original boolean-constexpr source cases pass raw IR, optimized
TTIR, TTGIR, LLVM, PTX and complete byte-exact cubin comparison in Python-free,
network-isolated execution without GPU access. 18 separate cold float-constexpr
compiler errors match exactly, including FP32 and FP64 loop-bound classification. They do not count as successful compiler images.

For None NVFP4 output type, eager retains the original LUT/scale dtypes and ATen
multiply. Its native NotImplementedError subtype is preserved through the
existing audited boxed ATen bridge. CUDA retains output allocation under the
current default dtype before its original KeyError(None). Triton retains its
actual output dtype and original SM89 assembler rejection, including FP64 default.

An initial optional jit_cache_hook observer caused Triton's own JSON serializer
to reject tensor constexprs before compilation. That capture was rejected and
was never merged into the master evidence. The final observer wraps the actual
_do_compile call without enabling that optional hook or changing its arguments.
The earlier uninstrumented 28-case run remains separate corroborating evidence.

## Seed and selector state

136 original seed and partial-RoPE selector cases pass focused comparison,
including 66 exact outputs and 70 exact failures. Each successful result repeats
exactly; failures and original input preservation are checked on both calls.
The earlier 82-case subset remains separately verified.

The original paths deliberately differ. Eager accepts only integer objects at
Generator.manual_seed after scaling, while disabled seeds avoid generator
construction. CUDA computes its stochastic flag separately from int(seed or 0),
then applies nanobind's boolean/unsigned conversion. A positive fraction can
therefore enable rounding with seed zero; scalar Tensor flags reach an original
native TypeError. Native signature text is generated from the verified original
module. Float and floating-Tensor conversion retains Python's NaN/infinity and
uint64 boundary behavior. Source-compatible late callbacks preserve evaluation
order without changing RNG calculations or policy. Triton keeps eager delegation
for positive seeds and its own JIT for disabled seeds. Partial RoPE retains
original truthiness, scalar Tensor comparison and eager fallback routing.

39 additional NA3D host/cold-literal cases pass twice: 18 defined outputs and
21 original errors. They preserve the actual next-power bit-twiddle, signed
windows, scalar Tensor arithmetic/iteration, float-shift native exception type,
and cold None/string/tuple source errors. Host dimension comparisons avoid a
lossy integer-to-float comparison. The source does not inspect unknown storage.

## Restoration and integration

The standalone native restoration passes with all 32 original CUDA object hashes
and four Stage 0 locks unchanged. The bridge library remains byte-identical to
the previous complete checkpoint. The whole compiler profile restores byte for
byte. Twelve source-derived Rust files regenerate exactly, including all 135
Python signatures, native signature text, compiler environment names and typed
program transcriptions. A stale generated INT8 header comment was corrected in
the generator; no runtime program calculation changed.

The full 141-function crate regression and subsequent unfiltered 133-function
Python-free primitive run are in progress. The final integrated result has not
yet been claimed. Further applicable source argument/configuration obligations
remain tracked in the Stage 1 completion ledger.

Owned evidence is under
`/tmp/imagine-rust-comfy-parity-20261017/runtime`:
`stage1-triton-double-bias-focused.log`,
`dynamic-arguments-warm-cold-keys-original.json`,
`stage1-dynamic-arguments-warm-cold-focused.log`,
`triton-na-dynamic-frontend-isolation.json`,
`stage1-na-dynamic-frontend-isolation.log` and
`dynamic-seed-state-expanded-original.json`,
`stage1-dynamic-arguments-expanded-focused.log`,
`stage1-native-restoration-proof.json`,
`stage1-compiler-profile-restoration-proof.json` and
`generated-source-restoration.json`.

The prior complete integrated checkpoint remains 126 crate / 118 unfiltered
Python-free primitive functions. New focused evidence does not replace it.
No production data, model execution, GPU process inventory or service change
was involved. No new deviation approval is pending.
