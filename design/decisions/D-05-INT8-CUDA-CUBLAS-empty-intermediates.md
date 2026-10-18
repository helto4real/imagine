# D-05-INT8-CUDA-CUBLAS-EMPTY-INTERMEDIATES: zero-K consumed buffers

Date: 2026-10-17. Status: **explicitly approved by Tomas; twelve reference/Rust/Python-free cases verified**.

Tomas replied “ja” to the preceding twelve-case cuBLAS approval question.
The approval covers the criterion and exactly twelve cases below, with complete
provenance guards for both consumed intermediate ranges before the kernel reads
them. Exact defined results and original autotuning remain required. Recorded
before dependent probing or observer implementation.

## Concrete next condition

Keep M=4, K=0, N=4, group256, scalar FP32 weight scale, no activation/bias/residual,
same FP32/FP16/BF16 input/output dtype. Select the original supported
`COMFY_KITCHEN_DISABLE_CUTLASS=1` setting. This is an existing reference option,
not a proposed default/tuning change.

- CUDA `__init__.py:1633-1635` records that setting; `:2221` skips CUTLASS.
- `:2150-2167`: the rowwise/ConvRot quantizer returns on K=0, leaving its
  allocated nonempty FP32 activation scale `[4,1]` unwritten.
- `:2269-2276`: the wrapper allocates nonempty INT32 accumulator `[4,4]`
  with `torch.empty` and calls the original cuBLAS binding.
- `dlpack_bindings.cpp:2269` forwards dimensions and pointers.
- `ops/cublas_gemm_int8.cu:176-194`: the original cuBLAS implementation returns
  on K=0 without writing the accumulator.
- CUDA `__init__.py:2280-2290` and `dlpack_bindings.cpp:3485` call the original
  dequantizer with that accumulator and activation scale.
- `ops/int8_linear.cu:1721-1766`: dequantization checks M/N, not K; M=N=4
  launches its vec4 kernel. `:546-574` reads the unwritten INT32 accumulator
  and FP32 scales, multiplies them with the defined weight scale, and writes
  output. Output depends on both prior allocation contents, despite being written.

This source finding reads no buffer contents. The earlier approved default
CUTLASS route returned without consuming its hidden scales; checking only the
returned output range is insufficient for this cuBLAS case.

## Approved criterion and twelve bounded cases

Preserve allocations, original quantizer/cuBLAS/dequantizer calls and arithmetic;
do not fill the two intermediates in the engine, reject K=0 or substitute
mathematical zeros. Compare metadata exactly and final bytes exactly under the
same known controlled own state for **both consumed intermediate buffers**.
Equality under independent uncontrolled allocation histories is not required.
All defined numerics/errors, backend choices, RNG and autotuning remain exact.

Authorize **12** paired reference/Rust/Python-free cases: M=4/K=0/N=4/group256,
ConvRot false/true, same FP32/FP16/BF16 input/output, scalar FP32 weight scale,
no activation/bias/residual, patterns `0x35`/`0x6a`, original CUTLASS-disabled
option. Warm only with defined own nonzero-K operands, including the workspace
cache. Prime only four own 64-byte buffers with the chosen known pattern.

Use metadata-only checks at the native boundary: before dequantization can read
either activation scale or accumulator, require the complete range of **each**
to lie inside freshly primed own storage. Refuse unmatched storage before that
kernel runs. Do not inspect raw hidden bytes or emit raw addresses. Original
intermediate allocations/calls remain unchanged; observers must not add fills,
change route, synchronize different streams or alter arithmetic. Compare the
final written output between reference and Rust and against the original vec4
operation on the corresponding **explicitly known own pattern values**, with
exact overflow/cast behavior. No ordinary Dreamchat memory/data is permitted.

Other shapes, mixed dtypes, activations, bias/residual, unknown storage or other
undefined-storage operations are outside this approval.

## Why approval is required

The [GEMM approval](D-05-INT8-CUDA-GEMM-empty-output.md) explicitly excludes
“forced-cuBLAS and other unwritten storage”. The [design, Section 14.2](../rust-comfy-parity-engine.md)
requires “every specific non-exact acceptance criterion has explicit approval”.
This case additionally needs provenance checks on consumed hidden intermediates,
not merely the returned no-write output. The explicit approval now lifts this bounded stop. Further undefined-storage
conditions outside these cases still require a separate decision before adoption.

## Implemented validation

The twelve reference/Rust/Python-free cases pass with complete guards for both
consumed intermediate ranges before the original dequantizer. Final bytes and
metadata match the explicit known-pattern operation exactly. The Rust observer
also propagates refusal on defined warm operands. No raw hidden bytes or
addresses are inspected/emitted by observers. See the
[continuation](../stage-1-cuda-int8-cublas-empty.md) for aggregate verification and
limits. The separate [remaining-family proposal](D-05-INT8-CUDA-linear-zero-k-family.md)
was subsequently explicitly approved and verified; see the [later family report](../stage-1-cuda-int8-linear-family.md). This does not retroactively broaden this bounded decision.
