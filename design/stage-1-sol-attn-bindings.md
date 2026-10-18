# Stage 1 eager Sol-Attn binding

Date: 2026-10-17. **Verified in the full 87 crate / 79 Python-free primitive
suite**, jointly with the following [eager ConvRot W4A4 continuation](stage-1-w4a4-linear-binding.md).
The six Sol-Attn records contain one ordinary exact control and five approved
controlled-intermediate cases; these are accounted separately. Prior checkpoint:
[eager quantization](stage-1-eager-quant-bindings.md). Aggregate [evidence](stage-1-evidence.json).

## Implementation and approved comparison

[`backend_sol.rs`](../inference/src/backend_sol.rs) connects the original eager
`sol_attn` registry body. It retains FP32 pooling and centering, exact token
scores, query-block scatter-add and division, threshold/top-k routing, forced
neighbor/sink blocks, pooled tail, base-2 softmax, denominator block lengths,
output dtype/contiguity, and gated coarse attention with original `addcmul_`.
The original ignored `token_aug` remains ignored. Python top-k rounding uses
round-to-even. No CUDA Sol-Attn or Triton implementation is substituted.

The scatter operation is the original ATen in-place scatter-add, dimension 2.
[`Runtime`](../inference/src/execution.rs) provides an optional per-runtime
isolated-development observer, absent by default. It observes operands before
the original operation and results afterward, before division. No ordinary
execution substitutes a tensor, adds a deterministic reduction, changes dtype,
changes a kernel or alters autotuning.

The [reference harness](../inference/tools/reference_sol_bindings.py) executes
the frozen original eager callable on six owned FP32/SM89 cases. It verifies
the immutable recorded canonical capture hash and source hashes. Before every
scatter it checks exact pre/index/source bytes, shape, strides, dtype, device,
dimension and current stream against the capture; the original scatter runs
once, followed by the approved canonical copy for the five T513 cases only.
The Python method is restored even on failure. The earlier JSON proposal labels
and false copy flags remain unchanged as historical capture provenance.

The [Rust comparison](../inference/tests/support/sol_bindings.rs) exercises the
actual resolved eager registry callable, checks automatic selection, guards the
same operands/metadata/stream, verifies the original scatter has run, and only
then copies the same canonical state for the five approved ragged cases.
All six outputs and post-call inputs must match exact bytes, strides, shape,
dtype and device. Inputs must remain unchanged and outputs must not alias them.
The T256 control is additionally called with the observer removed and must
still be byte-identical.

This proves **controlled-intermediate downstream equality** for default ragged,
no-tail, top-k, key-bias/sink and coarse-gate cases. It does not prove a unique
ordinary FP32 atomic result for these five cases. No tolerance is introduced.
The earlier 3,572 controlled unwritten-storage combinations are a different
category and must not be relabeled as Sol-Attn evidence. Five new controlled
Sol-Attn cases and one ordinary control are accounted separately.

## Scope and remaining work

One additional body is connected: 95 total, 40 remaining (eager 5, CUDA 7,
Triton 28). The six cases are initial verification coverage, not restrictions
on the public source contract. Complete signatures, exceptional dynamic Python
inputs, all dtypes/geometries, block-length/live-row combinations and additional
ordinary numerical controls remain broader Stage 1 obligations. The original
CUDA Sol-Attn INT8 sparse path is distinct and remains unbound.

No model inference or ordinary Dreamchat runtime data is involved. No dependency,
native bridge, original CUDA compilation unit, service or product integration
is changed. Local GPU and namespace validation uses authorized escalation;
no process inspection, service stop, commit or push is performed.

## Verification result

The focused six-case test, initial full 84/76 chain and final integrated full
87/79 chain pass. Both full chains repeat all primitives without Python,
ComfyUI, stable-diffusion.cpp or network access. The original bridge and its
32 CUDA objects, locked ELF closure, four Stage 0 locks and original flags
are unchanged. The original source/canonical hashes and own test artifacts
are recorded in aggregate evidence. No additional approval is pending.


## Additional ordinary eager coverage

Forty-five further ordinary records use CPU q/k/v in FP32/FP16/BF16: 30 exact
outputs and 15 original key-bias exceptions. All use the default runtime with
no scatter observer or canonical replacement. They cover valid INT32 block
lengths, original length clamp/ragged tail, partial live rows, sink queries,
three bool key-bias forms, gated coarse output, explicit scale/tau/no-tail,
ignored valid token augmentation and a top-k 2.5 round-to-even boundary.
Mask rank/head/width/batch/device errors preserve exact source ValueError text.

Six output cases deliberately obtain the original direct eager callable with
`kwargs=None`, as the original registry allows, to exercise FP64 CPU/GPU block
length conversion into CPU FP32. The same six metadata-validated lookups must
retain their original NoCapableBackendError; the registered API is not widened.
Tensor restoration/checks honor each operand's recorded device. Block-length
`.to(device=..., dtype=float32)` remains one original ATen operation, followed
by clamp. Each successful source control also repeats exactly without a hook.
Partial-live-row outputs are initialized by the eager code; no semantics for
unspecified dead query rows or additional CUDA controlled state are invented.

The final integrated chain passes 87 crate / 79 isolated primitive tests.
The isolated ATen exp2 JIT compiles using the frozen local NVRTC dependency
without Python. Clearenv intentionally leaves no HOME/kernel-cache variable;
its cache-disabled warning is retained and does not change kernel arithmetic
or autotuning. No cache dependency or library mount is added to hide it.

The first added ordinary run exposed an incomplete FP64 dtype label in the
metadata error renderer. A single original `torch.float64` mapping restored
the exact source rejection text; the failed attempt and successful rerun are
retained in evidence. Complete invalid-argument/dtype rendering beyond these
cases remains part of the unfinished full argument contract.
