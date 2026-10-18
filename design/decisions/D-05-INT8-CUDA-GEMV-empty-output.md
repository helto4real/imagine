# D-05-INT8-CUDA-GEMV-EMPTY-OUTPUT: further zero-K output routes

Date: 2026-10-17. Status: **explicitly approved by Tomas; 18 reference/Rust/Python-free cases verified**.

Tomas answered “Ja” to: “Godkänner du samma kriterium för dessa tre varianter
och de 18 avgränsade testfallen?” Approval is limited to the criterion and
18 cases below. It changes neither defined numerical results nor autotuning.
Recorded before implementing or executing the dependent probe.

## Frozen-source finding

The separately approved [M=1 fast-branch criterion](D-05-INT8-CUDA-M1-empty-output.md)
now has six passing controlled reference/Rust cases. It explicitly excludes
other shapes and ConvRot. The next original `int8_linear` branch reaches the
separate `_C.int8_gemv_dequant` entrypoint with these additional zero-K variants:

| Rows | ConvRot | Why the prior fast branch is skipped |
| --- | --- | --- |
| 1 | True | Its ConvRot M=1 predicate requires K >=256 |
| 2 | False | Its fast branch requires M=1 |
| 2 | True | Its fast branch requires M=1 |

For all three, keep K=0, N=4, group=256, scalar FP32 weight scale, no input
activation, bias or residual. The original registry accepts the metadata.
The contiguous two-dimensional empty input avoids ambiguous reshape inference.

- `cuda/__init__.py:2061-2099`: K=0 does not enter the fused-activation/fast
  ConvRot M=1 branch.
- `:469-476` and `:2150-2152`: the later ConvRot predicate accepts group256/K=0
  on SM89; its original rowwise kernel returns without writing the hidden scale.
- `:2157-2167`: the non-ConvRot rowwise quantizer likewise returns on K=0.
- `:2169-2187`: the GEMV predicate accepts either M=1/K%4==0 or M=2/K%16==0,
  allocates nonempty output `[M,4]` with `torch.empty` and calls GEMV.
- `dlpack_bindings.cpp:3535`: shape/type checks do not reject zero K.
- `ops/int8_linear.cu:1816-1817`: the GEMV launcher returns immediately for K=0,
  without writing output. With no residual the Python wrapper returns it as-is.

Thus these are three further unwritten-output routes. This conclusion uses
frozen source only: no unknown output bytes were inspected or new zero-K
variants executed. Source hashes are in [the aggregate evidence](../stage-1-evidence.json).

## Approved criterion and 18 bounded cases

Extend the same controlled-state acceptance criterion to exactly the three rows
above: same floating input/output dtype (float32, float16, bfloat16), two own
byte patterns (`0x35`, `0x6a`), yielding **18** paired reference/Rust/Python-free
cases. Prime only own small buffers (32 bytes covers the maximum `[2,4]` FP32
output), warm required own caches before priming, and require the complete
returned output range to lie within a just-filled own buffer before any byte
read. Refuse unmatched storage; do not inspect hidden unwritten scales or emit
raw allocation addresses.

Preserve original allocations, route selection, native calls and no-write
behavior. Do not zero-fill, reject the accepted input, substitute another backend
or return mathematical zero/bias values. Output metadata remains exact; output
bytes must match under the same own controlled allocation state. Independent
uncontrolled unwritten bytes need not be equal. All defined numerics, errors,
backend decisions, RNG and autotuning remain exact and unchanged.

This does not cover other row/output counts, groups, input/output dtype mixtures,
activations, bias/residual or any other undefined-storage path.

## Why explicit approval is required

The [approved M=1 decision](D-05-INT8-CUDA-M1-empty-output.md) states that its
approval applies “only to the bounded CUDA M=1, non-ConvRot zero-K case above”.
The [design, Section 14.2](../rust-comfy-parity-engine.md) requires “every specific
non-exact acceptance criterion has explicit approval”. These variants are outside
that bound. That approval has now been given for these 18 cases. Other undefined-storage
conditions still require a separately bounded decision before adoption.

## Implemented validation

All 18 guarded original-reference cases and normal Rust/Python-free cases pass, with exact
metadata and unchanged own known bytes. No hidden scale or unmatched storage
was read. The [GEMV continuation](../stage-1-cuda-int8-gemv.md) records ordinary
coverage, aggregate verification and limits. The next separate
[GEMM criterion](D-05-INT8-CUDA-GEMM-empty-output.md) is pending; this approval
has not been extended to it.
