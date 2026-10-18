# Stage 1: registered binding and JIT runtime contracts

This continuation remains part of the authorized entire Stage 1. No new
exception, tolerance, algorithm, autotuner configuration or backend substitution
is adopted. The earlier 126 crate / 118 unfiltered Python-free result is the last
complete snapshot; subsequent focused checks do not replace it.

## Complete constructor and argument binding

`Runtime::frozen` connects all 135 registered eager/CUDA/Triton bodies and shares
one compiler context across the Triton families. The registry reports no missing
body. This constructor is a development API, with remaining contract work tracked
in the completion ledger; it does not integrate with or replace Dreamchat.

All 647 original parameters are positional-or-keyword parameters. A source-derived
Rust signature table validates the original 950 missing, unknown, duplicate and
excess-argument cases before execution. All 950 match exact original TypeError
messages. A discovered one-argument pluralization mismatch was corrected.
The capturer can regenerate/check the table against the frozen Python functions.
Public replacement callables retain their own contracts rather than inheriting
signatures of the bodies they replace.

## Load and launch hooks

Nine original scenarios, twice each, pass focused differential GPU validation:
no hooks, all hooks, exit only, failures at load start/end and launch enter/exit,
CPU-pointer rejection and zero grid. Comparison includes event order, exact
exceptions, metadata contents, cache digest, lazy handle presence and resources.
Addresses are checked for presence and same-process stream identity.

The original native launcher enters its callback before tensor-pointer extraction.
A pointer failure therefore suppresses the exit callback but follows the enter
callback. Load-start failure is retried by launch-metadata initialization while
an enter hook is present. Load-end failure leaves loaded handles available for
the next call. Rust preserves these states. No host state lock is held while
user callbacks execute. No hook is installed by default.

## Explicit ASTSource and JIT keys

The hook digest comparison exposed a source-compatible correction: the actual
JIT `_pack_args` retains empty attribute lists for pointers, integers and every
constexpr. The earlier direct ASTSource matrix intentionally supplied only
nonempty divisibility attributes. These two original paths produce distinct
cache keys, even when they produce identical IR and cubins.

The compiler now represents both paths. Existing direct ASTSource methods retain
their exact keys. Registered wrappers use the JIT path, derived from the same
source signature and original argument names. Neither path reads saved IR or
cubin fixtures during execution. A CPU-only original capturer obtains 7,740 paired
keys through the original `_pack_args` over all frontend matrices, including the
288 separately labeled NA3D compiler-only FP64 probes. All 7,740 paired candidate keys pass exact comparison. INT8, W4A8 and NA3D wrapper regressions and the original cold/warm load/resource-failure regression pass through the corrected paths.

Five wrappers also capture device/stream before pre-run hooks, as JITFunction.run
does: RoPE, RMS-RoPE, MXFP8 and both NVFP4 directions. All three independently observed stream-boundary scenarios (RoPE, RMS-RoPE and MXFP8) pass exact comparison, stopping at the launch-enter callback before GPU work. NVFP4 capture order follows the same source order; its SM89 compilation failure occurs before launch.

Owned evidence under `/tmp/imagine-rust-comfy-parity-20261017/runtime`:
`stage1-signature-binding-focused.log`, `stage1-signature-restoration.log`,
`kernel-hooks-original.json`, `stage1-kernel-hooks-focused.log`,
`triton-jit-keys-original.json`, `stage1-jit-keys-original.log` and `stage1-complete-jit-wiring-focused.log`.
No production runtime material, model weights, GPU process inventory or ordinary
Dreamchat data was read. The [compiler-mode continuation](stage-1-compiler-modes.md) and [INT8 default-dtype continuation](stage-1-int8-default-dtype.md) additionally pass focused comparison. Remaining Stage 1 state/lifetime and restoration work continues.

## Tensor objects at the native launch boundary

Eight additional original cases pass focused comparison with two stateful calls
per case. Each of load-start, load-end, enter and exit hooks replaces the original
tensor's storage, for both a one-dimensional tensor and a reshaped matrix view.
The candidate preserves the original distinction between a shared TensorImpl and
an already-created view: late pointer extraction observes storage changes on the
same object, while the view retains its original storage. Outputs, post-call
inputs and callback events match exactly on both calls.

All eleven registered launch sites now retain owned tensor objects through native
pointer validation and the exit callback. Pointer extraction occurs after load
and enter callbacks, as in the original launcher. This adds no synchronization,
copy, null-pointer rejection or stream-recording policy. Raw driver calls retain
their explicit raw-pointer contract. This evidence covers the source launch
lifetime; it does not claim arbitrary Python plugin compatibility.

Owned evidence: `triton-object-arguments-original.json`,
`stage1-triton-object-arguments-original.log` and
`stage1-triton-object-arguments-focused.log` under the isolated runtime root.
