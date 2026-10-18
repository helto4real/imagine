# Stage 1: Sage INT8 attention

Date: 2026-10-18. Full **97 crate tests and 89 unfiltered Python-free primitives pass**,
covering 243 attention records and 30 public packed-state records. Stage 1 remains incomplete.

[kitchen_sage.rs](../inference/src/kitchen_sage.rs) ports the frozen public
INT8 attention and split prequantize/consume functions. It keeps input-validation
order, head padding/cropping, GQA metadata, CTA selection, scale conversion,
mask broadcasting/preparation, all packed tensor layouts, allocation order and
current-stream launches. Regular execution invokes the original fused native
`sage_sdpa` entry; split execution invokes the original quantize and prequantized
entries. No SDPA substitute, FP8 arithmetic or alternate backend is used.

The packed object owns quantized Q/K/V, scales and the original active mask
representation. Temporary floating Q/K/V references remain only inside the
preparation call and are not fields of the returned object. Float32 input keeps
the original BF16 native output followed by a float32 conversion.

The matrix includes FP32/FP16/BF16, head sizes 31/64/80/128/192/256, short/long K,
broadcast Boolean and floating masks, dense masks, regular and split calls,
repeat consumption, input/state errors, negative/zero/nonfinite scale behavior,
public packed-state dtype/CTA/device errors and original zero/negative crop
metadata. Packed payloads, scales, masks, outputs, strides and post-call input
state compare exactly. The 30 public-state cases include 8 outputs and 22 errors;
malformed states used there are rejected before a native read or have valid
backing capacities. Undersized hidden scale storage is not launched.

## Original unwritten prepared-mask padding

Frozen `sage_attention.py` allocates a prepared row of
`ceil(tiles * 129 / 4) * 4` float32 elements. In
`sage_attention/sage_attn_launcher.cu`, `prepare_key_mask_kernel` writes 128
biases per tile and one tile classification value. It does not write the final
0–3 alignment elements. For K=1025, nine tiles produce 1,164 elements: indices
0–1160 are written, and the last three retain prior allocator contents.
The numerical attention result is not a license to read those unknown bytes.

Tomas's later [authority clarification, Section 1](rust-comfy-parity-engine.md)
explicitly permits metadata-only provenance guards and exact known-own-state
comparisons for original no-write behavior; prior bounded matrices are historical
coverage, not new capability limits. This continuation uses that authorization,
without requesting another permission for faithful compatibility work.

Eight controlled cases use FP16/BF16, regular/split calls and patterns 101.25 /
-202.5. Immediately before the original prepared-mask allocation, development
hooks allocate and fill an own `(1,1,1164)` float32 buffer, drop it, and prove that
the complete original allocation reuses that same owned range. A mismatch fails
before the original preparation kernel or any consumer/byte inspection. Hooks
retain metadata only. The original kernel then writes its own defined region;
all complete packed bytes, including the known trailing pattern, compare exactly.
No original output is filled or repaired after allocation. Ordinary adapters
install no observer, and their no-write padding behavior is unchanged. No unknown
allocator bytes become reference goldens. Ordinary K=1536 prepared masks have no
unwritten alignment tail and are compared without controlled allocation state.

This is source-compatible known-state validation, not a new numerical relaxation
or parity departure. The separate D-04 null-export exception remains the only new
approved behavioral departure in this continuation. Kernel arithmetic, dispatch,
RNG and autotuning are unchanged.

Own logs under `/tmp/imagine-rust-comfy-parity-20261017/runtime`:
`stage1-sage-reference.log`, `stage1-sage-build.log`, `stage1-sage-focused.log`
and `stage1-sage-full.log`. No ordinary runtime data, model/product execution,
GPU process inspection, service change, commit or push is involved.

Remaining Stage 1 work includes Sol chunked production/allocation context,
general Triton compilation and all 28 bindings, broader dynamic/FFI and
asynchronous-lifetime behavior, and final restoration/isolation packaging.

The full fixture has 24,025 physical records: 24,002 ordinary, five approved
Sol records, ten approved null-export records and eight source-compatible known
mask-padding cases. The prior 3,572 controlled combinations remain separate.
The four Stage 0 locks and 32 original CUDA object hashes were rechecked unchanged.
See [checkpoint](stage-1-sage-checkpoint.json).
