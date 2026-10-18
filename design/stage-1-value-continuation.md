# Stage 1 source-value continuation

The frozen [148/139 checkpoint](stage-1-value-contracts-checkpoint.json) passed
completely. Its 272 source/tool inputs, executable, fixture, bridge and compiler
profile are preserved independently. This continuation changes source after
that snapshot; the snapshot is not evidence that the new source passed a full
152-test crate /143-test unfiltered isolation run.

At this focused checkpoint the differential master contained 63,149 physical records; subsequent additions are recorded in [final validation](stage-1-final-validation.md).
The following 1,784 new ordinary records are integrated:

| Source obligation | Records | Current verification |
| --- | ---: | --- |
| Structured Sol ranges, comparisons, truthiness and diagnostics | 756 | Focused Rust pass |
| Weight-seed values, delayed integer conversion and native bool rejection | 450 | Focused CPU/GPU pass; two identical calls and unchanged inputs |
| Eager NA values and scales, CPU and CUDA | 560 | Focused CPU/GPU pass; two identical calls, gradients and unchanged inputs |
| Signed-zero JIT equality and nested unhashable error order | 18 | Focused GPU pass, including exact new-load counts |

The previous 13,072-case structured metadata test also passes with the new
predicates. The current frontend additionally passes 180 original source/TTIR
cases and 306 original compilation failures in Python/GPU-free isolation.
Those integer-window compiler cases never launch large GPU neighborhoods.
They are separate compiler proofs, not 486 additional physical master records.

`SourceArgument` retains owned values and a borrowed call context for lazy
validation. Sol now follows the original length/unpacking/comparison order and
formats the actual rejected value. The W4/W6 seed retains its original numeric
or Tensor type through positivity, chunk offset addition and late `int(seed)`.
In particular, positive 0.5 enables stochastic rounding with integer seed zero.
Native callbacks evaluate arguments before strict bool/uint64 conversion, as
the original wrappers do; passing a Tensor flag is not silently accepted.

Dynamic eager NA follows the original window construction, Python scalar and
Tensor arithmetic, dtype inference, identity-sensitive geometry grouping,
mask reduction/host synchronization, stacking and SDPA order. Its already
verified ordinary typed path remains connected. Original ATen `promote_types`
is called directly; no surrogate tensor arithmetic infers dtypes.

Two initial focused failures are retained in
`runtime/stage1-value-continuation-focused.log`: the original c10 dtype return
is unsigned, and a fused native seed comparison must not be coerced to bool
before the following integer conversion. Both are fixed; the corresponding two
focused tests pass in `runtime/stage1-value-continuation-two.log`. Neither was
accepted as a difference or resolved by weakening validation.

The original CUDA objects, dependencies and autotuning policy are unchanged.
At this historical focused checkpoint the native bridge was SHA256
`f87a4e609786c2eaa0bb0fdc8e38c2046dff635e99769af746367b9cade34ce6`.
No new numerical criterion, backend substitution, approval request or approval
review rejection occurred. Source reference captures and test data are local
and synthetic; production Dreamchat data is not read.

At this focused checkpoint Stage 1 continued. Its remaining binding audit included dynamic CUDA
NA scalar conversion, diagnostic Tensor representations, and distinguishing
unreachable development/custom-constructor guards from actual frozen-runtime
obligations. Source-compatible fixes and final restoration/integration remain
authorized. Python warning-filter state is not independently verified; these
results do not claim a complete Python SDK implementation or complete engine
parity. No scope is silently removed.

Current integrated status and the later native bridge are recorded in [final validation](stage-1-final-validation.md).
