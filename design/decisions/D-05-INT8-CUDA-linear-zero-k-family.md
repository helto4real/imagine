# D-05-INT8-CUDA-LINEAR-ZERO-K-FAMILY: remaining identified linear variants

Date: 2026-10-17. Status: **explicitly approved by Tomas; all 3,456 reference/Rust/Python-free combinations verified**.

Tomas replied “ja” to the preceding question approving the consolidated CUDA
int8_linear K=0 criterion and the 3,456-combination matrix, with guards before
every consumer, exact defined results and unchanged autotuning. Recorded before
new dependent probes, dispatcher or expanded observation implementation.

## Source finding and reason for a consolidated decision

The frozen `cuda.int8_linear` has three identified originally unwritten buffers:
FP32 activation scales, INT32 cuBLAS accumulation, and floating linear output.
The bounded earlier decisions cover specific row counts and same input/output
dtypes without activation, bias or residual. They explicitly exclude other
variants; they are not general approval for this public operation's K=0 family.

There are further reachable source paths within that same operation:

- CUDA `__init__.py:2060-2075`: at activated K=0 the fused-activation gate is
  false. `gelu_tanh` is applied eagerly to an empty 2D input, then `input_act`
  becomes None. M=1 non-ConvRot proceeds to its original no-write fast branch.
  No new numerical/allocator probe was used to identify this path.
- `backends/_activations.py:43-69` also defines SwiGLU and RMSNorm before the
  same linear selection. Their original empty-input/error behavior must be
  preserved; this proposal does not assume all combinations succeed.
- CUDA `__init__.py:2082-2085`: input and output floating dtypes can differ;
  the original dtype code selects the output allocation/kernel conversion.
- `:2100-2124`, `:2174-2187`, `:2190-2255`: bias conversion/validation does
  not change the native zero-K no-write returns. Do not synthesize the expected
  mathematical bias result where the reference skips the write.
- `:2088-2094` and `_activations.py:73-86`: `_finish` applies residual addcmul
  eagerly unless fusion succeeded. It therefore **consumes the floating
  unwritten output** on those zero-K routes. A guard only at cuBLAS dequantization
  does not protect this separate ATen consumer.
- M=1, M=2 and later GEMM branch decisions are already source-identified. The
  original native early returns depend on K=0, not the test's particular N=4
  or M=4. Original shape/dtype/scale/bias/residual validation still runs.

These are parameters of the same identified no-write/consumer family. This
proposal groups them in one operation-specific criterion; no prior approval is
silently extended. Frozen source hashes are in the aggregate evidence.

## Approved scope and acceptance

For this frozen **CUDA `int8_linear` only**, initial SM89 platform, contiguous
2D raw input `[M,0]`, INT8 weight `[N,0]`, M>0/N>0, supported FP32/FP16/BF16 input and
output independently, ConvRot false or group256 true, include the original
None/`none`, GELU-tanh, SwiGLU and RMSNorm options, original bias/residual forms,
and original CUTLASS-enabled/disabled setting. Only calls that reach the three
identified unwritten buffers after original validation use this criterion.
Every original rejection/exception remains exact; do not turn errors into
success, assume RMSNorm/SwiGLU succeeds, or add new supported-shape limits.

Preserve allocation and call order, backend choices, bias behavior, activation,
residual arithmetic, native return flags, RNG and autotuning. Do not fill original
buffers in the engine, reject otherwise accepted inputs, force another route,
substitute zero/bias values or change a default. Compare metadata exactly.
Compare all defined arithmetic and final bytes exactly under matching known own
allocation state for every originally unwritten buffer that can affect output.
Independent uncontrolled allocation histories need not yield equal unwritten
bytes or their derived results. No numerical tolerance is proposed.

Before **any CUDA or ATen consumer** may read such a buffer, metadata-only
observers must prove its complete range belongs to freshly primed own storage.
This includes activation scales and INT32 accumulation before dequantization,
and floating no-write output before eager residual addcmul. Refuse unmatched
storage before consumption. Observers must not fill/read raw hidden bytes,
change stream state, alter arithmetic/selection or emit raw addresses. Compare
final written output to the original operation on explicitly known own pattern
values; compare no-write returned bytes only after the complete-range guard.

This criterion covers the identified operation/conditions above; it does not
approve other Kitchen operations, nonzero-K undefined behavior, arbitrary
unwritten inputs, other GPUs, fallback replacement, or ordinary Dreamchat data.
Existing decisions and their historical evidence remain unchanged. The family
scope does not claim numerical validation for dimensions that have not run.

## Approved bounded differential matrix

Authorize a synthetic reference/Rust/Python-free matrix of **3,456 combinations**:

| Axis | Values |
| --- | --- |
| M / N / raw K / ConvRot group | M=1/2/4, N=4, raw K=0, group256 |
| ConvRot | false / true |
| Input activation | None / gelu_tanh / swiglu / rms_norm |
| Input/output dtype | FP32/FP16/BF16 independently: nine pairs |
| Bias | absent / own length-4 output-dtype vector, arange * 0.015625 |
| Residual | absent / own output-dtype 0.375 tensor with length-4 scale 0.5 |
| Original CUTLASS policy | enabled / disabled |
| Own byte pattern | 0x35 / 0x6a |

RMSNorm uses an own empty weight in input dtype and original epsilon default 0.0.
Warm caches only with defined nonzero-K own inputs. Prime at most five own
64-byte buffers per bounded case; all numerical operands outside these primed
ranges are explicitly initialized own synthetic tensors. Apply the complete
consumer guards above, including before eager residual arithmetic. Refuse any
case whose allocation provenance cannot be proved; do not replace it with a
passing golden. Preserve exact original exceptions and treat resource failure
as a failed capture, never an expected exception.

The matrix includes the earlier 48 bounded linear cases; map/reuse earlier
source-identical reference evidence where valid rather than adding duplicate
counts. This record was approved before any expanded case or new dependent observer/dispatcher ran.

## Why approval is required

The latest [cuBLAS approval](D-05-INT8-CUDA-CUBLAS-empty-intermediates.md) states:
“Other shapes, mixed dtypes, activations, bias/residual, unknown storage or other
undefined-storage operations are outside this approval.” The
[design, Section 14.2](../rust-comfy-parity-engine.md) requires “every specific
non-exact acceptance criterion has explicit approval”. Tomas requires a stop for
such a decision. That explicit approval has now been given for the operation/conditions and
matrix above. Further conditions outside this scope still need a separate decision.

## Implementation and verification

All 3,456 approved combinations pass exactly, including guards before residual
consumption. The [continuation report](../stage-1-cuda-int8-linear-family.md)
records the 144 defined RMS-statistic overwrites of subsequently reused scale
storage; expected state models these original writes without changing the
algorithm or acceptance criterion. An additional 1,134 ordinary activation/route
cases pass (1,008 exact outputs, 126 exact errors). Earlier 48 linear cases map
into the matrix and are counted once. Full verification passes 60 crate tests
and 52 Python-free primitive functions: 7,831 ordinary fixtures and 3,504 unique
controlled combinations overall. Original 32 CUDA objects, four Stage 0 locks,
Triton artifacts and autotuning are unchanged; the optional metadata observer
changes the bridge binary. The separate SVDQuant proposal is not approved.
