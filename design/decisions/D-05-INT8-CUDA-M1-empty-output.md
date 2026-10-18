# D-05-INT8-CUDA-M1-EMPTY-OUTPUT: unwritten zero-K linear output

Date: 2026-10-17. Status: **explicitly approved by Tomas; six controlled reference/Rust/Python-free cases verified**.

Tomas explicitly approved: “Ja, jag godkänner kriteriet och de sex avgränsade testfallen med samma exakthetskrav och oförändrad autotuning. Dokumentera beslutet och fortsätt steg 1. Stoppa om något annat kräver mitt uttryckliga godkännande.”
The approval covers the criterion and exactly six cases below; it is recorded
before the new allocator probe or dependent Rust work. Earlier source findings
are not retroactively labelled numerical validation.

## Concrete frozen-source path

The next Stage 1 CUDA INT8-linear path includes a reachable zero-width case:
`x=[1,0]`, INT8 `weight=[4,0]`, scalar FP32 weight scale, floating input/output,
`convrot=False`, no input activation, bias or residual.

1. CUDA `__init__.py:4464` registers `int8_linear` when cuBLASLt is available.
   Its shape rules require input rank >=2 and weight rank 2, without rejecting K=0.
   cuBLASLt is available in the locked initial environment.
2. `int8_linear` (`__init__.py:2035`) preserves the contiguous 2D input and
   passes its inner-dimension assertion. For M=1, non-ConvRot K=0 satisfies
   `k % 4 == 0` and `k <= 2560`, selecting the original M=1 branch.
3. That branch allocates empty INT8 scratch `[1,0]`, FP32 scale `[1,1]` and
   **nonempty output `[1,4]` with `torch.empty`**, then calls `_C.int8_linear_m1`.
4. `dlpack_bindings.cpp:3594` validates the shapes, calls the rowwise quantizer,
   then the GEMV decoder, without adding an empty-width rejection or fill.
5. `ops/int8_linear.cu:1334` returns from rowwise quantization on zero width;
   `:1816` returns from GEMV on K=0. Neither scale nor output is written.
6. With no residual, the Python wrapper returns that unwritten nonempty output.

Thus its bytes depend on prior allocator contents, not only the specified input.
No output bytes or unknown storage have been read to establish this finding.
The source files are verified against the unchanged Stage 0 locks; hashes and
line references are recorded in [the aggregate evidence](../stage-1-evidence.json).
No CUDA M=1 numerical/allocator probe or new dependent port is started before
this approval. Independent FP16 and registry validation may finish.

## Approved bounded criterion and investigation

Preserve the original allocations, native calls and no-write behavior. Do not
zero-fill the result, reject K=0, switch backend or substitute a mathematical
zero/bias result. Compare output metadata exactly. Compare output bytes only
when both reference and Rust runs have matching controlled allocation history
containing this isolated process's own known synthetic bytes.

Authorize six reference/Rust/Python-free cases: CUDA input and output of the
same dtype (float32, float16 or bfloat16), M=1, K=0, N=4, scalar FP32 weight scale,
`convrot=False`, no activation/bias/residual, two own byte patterns (`0x35` and
`0x6a`). Prime only own small buffers with known bytes. Before inspecting the
returned output require its pointer/storage range to match a just-primed own
buffer; refuse to read unmatched storage. Do not inspect the unreturned scale's
contents or any other workload's memory. Allocation identities are used only
for local guards, not emitted as raw addresses.

Uncontrolled independent histories are not required to produce identical
unwritten output bytes. All defined numerical results, errors, backend decisions,
RNG and the original autotuning policy remain exact and unchanged. This approval
applies only to the bounded CUDA M=1, non-ConvRot zero-K case above; other
shapes, fused activations, bias/residual and undefined-buffer cases remain outside.

## Why a new approval was required

The [earlier empty-width decision](D-05-INT8-ConvRot-empty-width.md) and
[two-helper extension](D-05-INT8-ConvRot-empty-variants.md) cover specified
ConvRot **scale tensors**, not this separate nonempty linear **output**.
The [design, Section 14.2](../rust-comfy-parity-engine.md) requires “every specific
non-exact acceptance criterion has explicit approval”. Tomas also explicitly
requires stopping when another such approval is needed. This proposal neither
reopens earlier approvals nor silently extends them.

## Implemented verification

The six original high-level reference cases passed with the two own known byte
patterns. The Rust M=1 route also passed all six normally and without Python, with the same output metadata,
complete guarded byte-range checks and unchanged bytes. No hidden scale bytes
or unmatched storage were read. The ordinary M=1 continuation separately passes
378 exact numerical results, 126 exact expected errors and five source-driven
route exclusions. The full normal/Python-free suite is recorded in the
[M=1 continuation](../stage-1-cuda-int8-m1.md) and aggregate evidence.

This approval remains unchanged. The later separate
[GEMV empty-output proposal](D-05-INT8-CUDA-GEMV-empty-output.md) was separately
approved and verified for its 18 bounded cases. Neither approval covers the
[next GEMM proposal](D-05-INT8-CUDA-GEMM-empty-output.md).
