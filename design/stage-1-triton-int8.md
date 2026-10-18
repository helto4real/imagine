# Stage 1: original Triton INT8 compiler and wrapper

Status: source/compiler, all original candidate comparisons and the registered
wrapper comparison pass. The two original default autotuners are connected.
All three functions also pass separately in Python-free GPU isolation. Stage 1
is not complete.

The Rust frontend transcribes both original matmul kernels, including grouped
tile ordering, modulo row/column addressing, pointer strides, carried loop
pointers and accumulators, integer dot, FP32 scaling, optional bias and masked
output stores. The original helper call order, constexpr void returns, loop
argument locations and public function locations are retained. FP8 and unsigned
FP4 source handling remain separate from the new signed INT8 pointer type.

Original `zeros` and `cdiv` helpers are lowered from their frozen definitions.
Loop argument locations were independently queried from the original native IR:
`k_idx` uses the range expression, while carried pointers/accumulators retain
nested names at their original producer positions. Saved IR is not an engine
execution input.

## Verified comparisons

- 432 raw source/optimized TTIR and AST-key cases cover both scale branches, all
  six original configurations, bias on/off, three output precisions, dynamic,
  constexpr-one and wide scalar profiles, and both alignment states.
- All 72 selected complete compilations reproduce original TTGIR, LLVM IR, PTX
  and CUDA binary bytes in a Python-free namespace without GPU access.
- The integrated generic compiler executes all **216 existing original
  candidate samples** from 36 input cases. Every CUDA binary and output payload
  matches the original, along with warps, stages and shared-memory metadata.
  Memory-cache identity and fresh compiler/disk-cache restoration pass for both
  scale paths. These inputs reuse earlier synthetic evidence and are not counted
  again as new physical fixture records.
- The integrated frontend repeats the complete 432/72 comparison. The shared
  legacy pointer-load extraction also preserves all 144 NVFP4 source cases and
  all 24 original source-to-PTX/assembler-error comparisons.
- All 201 original pre-run boundaries match: operand payloads where defined,
  metadata, strides, gradient flags, identity/alias relationships, ordered
  keyword arguments and compile-time defaults. Allocated unwritten outputs are
  compared only by metadata.
- All 51 ordinary wrapper records match, including two calls for each successful
  case. They cover both scale branches, three precisions, bias, activation,
  rotation, residuals, explicit None RMS epsilon and early argument errors.
  Before/after state checks preserve both independent autotuners' attribute
  presence, argument mapping, cache size and retained timing-table size. Actual
  independently measured winners/timings retain only the earlier D-05 criterion.

Compiler configuration now belongs to each original candidate: 8 warps/3 stages
for the first, and 4 warps/4 stages for the other five. Original options hashes,
pass graphs and metadata templates feed each specialization's cache digest.
Existing FP8/rowwise/AdaLN/RoPE/MXFP8/NVFP4 calls retain their original default
configuration through the same compiler core.

## Wrapper and autotuner integration being verified

The wrapper preserves input activation before reshape, optional cached Hadamard
rotation, the original rowwise quantizer, output allocation before weight-scale
normalization, scalar/per-channel scale selection, raw bias layout, output
reshape and residual application. It introduces no eager matrix-multiply
substitute. The two original autotuners have separate caches owned by the shared
compiler/module context; W4A8 must reuse these same states when it calls INT8.

The existing verified six-candidate selection and benchmark implementation is
reused, including only the original three caught candidate exception classes,
lexicographic timing-list selection and a separate final launch. Locks cover
short state operations, not GPU benchmarking. The source's keyword-only `nargs`
state is retained: initially absent, an empty mapping during a call, populated
after failure and None after successful completion. This matters for reentrant
or concurrent source-compatible failure behavior. No new numerical criterion
or autotuning policy is introduced; the two recorded D-05 INT8 approvals remain
limited to their existing scope.

The physical master adds 432 specialization, 51 ordinary wrapper and 201
observed-boundary records to the NVFP4 continuation: **35,025 records**.
All three focused crate test functions pass (the wrapper was rerun separately
after correcting the test observer's internal rowwise operation name). This is
not a new full crate/unfiltered engine checkpoint. The last full result remains
[MXFP8, 117/109](stage-1-triton-mxfp8-checkpoint.json).

Owned evidence under `/tmp/imagine-rust-comfy-parity-20261017/runtime` includes
`stage1-triton-int8-draft-full.log`,
`stage1-triton-int8-integrated-int8-frontend.log`,
`stage1-triton-int8-registered-candidates.log`,
`stage1-triton-int8-wrapper-focused.log`,
`triton-int8-frontend-isolation.json` and
`int8-region-argument-locations.json`.

Remaining wrapper/type/error/state obligations are not exclusions. Non-scalar
invalid argument representation, nonnumeric RMS epsilon, non-integer rotation
sizes, sequence-valued non-tensor scales, non-INT8 weight specialization and
broader compiler/default-dtype controls still require source-compatible work.
The unrelated D-04 DLPack exception is not extended to Triton data pointers.

The first focused wrapper invocation lacked GPU device visibility in the command
sandbox; its failure is retained and is not passing evidence. The authorized
local GPU retry succeeded after the test-only observer correction. Automatic
approval review did not reject an action. No numerical tolerance, source change
or policy deviation was adopted.

The subsequent combined [126/118 full checkpoint](stage-1-triton-integer-checkpoint.json) verifies the integrated source, including the shared ATen identity correction. This remains a partial Stage 1 result.
