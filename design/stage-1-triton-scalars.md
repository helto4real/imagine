# Stage 1: numeric and structured normalization scalars

Focused source/compiler and registered-wrapper comparisons pass for original
AdaLN, RMS-AdaLN and RMS-RoPE epsilon specialization. Integer/bool arguments retain
their original runtime type and alignment attribute; integer one remains a
constexpr and has no runtime argument slot. Python floats remain FP32 JIT values.
No blanket conversion of all epsilon arguments to float is used.

Each normalization family adds 180 source/optimized-TTIR cases and 30 complete
compiler comparisons, byte-exact through TTGIR, LLVM IR, PTX and CUDA binary. The
matrices include three precisions, both operation branches, aligned/unaligned
operands, scalar index profiles and five additional epsilon specializations.
These 360 cases extend the two original 1,296-case matrices to 1,476 each.

144 wrapper cases pass focused comparison: 108 valid numerical cases reproduce
exact bytes twice, and 36 failures preserve original error class/message. The
observed cold compiler directories also match actual original compiler digests.
Coverage includes FP32/FP16/BF16, both normalization branches and both RoPE splits,
zero/one/ordinary/wide integers, bools, floats and overflowing-to-FP32 infinity.
Invalid inputs include strings, lists, dictionaries, dtype/device objects, None,
FP32/FP64 tensor pointers and a mixed constexpr/runtime tuple.

The source error adapter retains the original cold boundary: memory/disk-cache
lookup occurs before frontend failure. Static original kernel excerpts and AST
positions reproduce the 12-line source excerpt/caret and cause. They are generated
from hashed source, never from runtime logs or saved IR. Structured signatures
and constant paths retain original nesting. In particular, root constexpr
attributes have an empty entry, while nested tuple constexpr entries are omitted
by the original `_pack_args`; the Rust key builder preserves that distinction.
The earlier 7,740 explicit/JIT key pairs remain exact after this correction.

An initial wrapper capture selected package function aliases instead of module
objects and contained no successful inference calls. It was rejected, kept under
an explicit invalid-import filename and never merged into the master fixture.
The corrected capture verifies callable module symbols and requires all 108
chosen valid numeric cases to succeed before it writes an oracle.

Owned evidence under `/tmp/imagine-rust-comfy-parity-20261017/runtime`:
`stage1-rms-rope-scalar-original.log`, `stage1-rms-rope-scalar-isolation.log`,
`stage1-adaln-scalar-original.log`, `stage1-adaln-scalar-isolation.log`,
`triton-scalar-bindings-with-keys-original.json`,
`stage1-triton-scalar-bindings-with-keys-original.log` and
`stage1-triton-scalar-wrappers-focused.log`.
Both extended 1,476-case AST-key crate checks pass. Final full isolation of the combined source remains pending. Stage 1 continues; no new deviation or autotuning policy is
adopted. Existing D-04 remains limited to the approved null DLPack export path.
