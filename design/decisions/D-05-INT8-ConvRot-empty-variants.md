# D-05-INT8-CONVROT-EMPTY-VARIANTS: two further empty return paths

Date: 2026-10-17. Status: **explicitly approved by Tomas; bounded validation authorized**.

The earlier [empty-width decision](D-05-INT8-ConvRot-empty-width.md) is approved
and verified. Its executed scope comprises the public INT8 weight quantizer and
direct `quantize_int8_rowwise_convrot`, group 256, seed 0, float32/float16/bfloat16
and one/four rows. This record does not revoke or reopen that approval.

## Source finding

Two additional directly callable original CUDA helpers allocate nonempty row
scales with `torch.empty`, then return without writing them when width is zero:

| Original Python helper | Wrapper allocation | Original native return |
| --- | --- | --- |
| `quantize_int8_convrot_staged` | Empty rotated/partial/payload buffers; float32 scale `[M,1]` | `ops/int8_linear.cu:1536` checks zero rows/columns before quantization |
| `quantize_int8_rowwise_convrot64` | Empty INT8 payload; float32 scale `[M,1]` | `ops/int8_linear.cu:1624` checks zero rows/columns before quantization |

The frozen Python definitions are at CUDA `__init__.py:1562` and `:1588`
(respectively; the exact definition lines are verified in the aggregate source
record). `dlpack_bindings.cpp:3387` and `:3430` validate the corresponding shapes
and supported floating dtype, without rejecting `[M,0]` for group 256 and no
input activation. The staged wrapper's partial buffer has shape `[M,0]`.

At the time approval was requested this was a **source finding**, with no new
empty-output GPU probe or fixture capture for either helper. Existing
nonempty fixtures already exercise the existing Rust bindings for both.

## Approved bounded criterion

Extend the same approved controlled-state criterion to these **two helpers**:

- CUDA inputs `[1,0]` and `[4,0]`, float32/float16/bfloat16, group 256,
  stochastic rounding 0 and no input activation.
- Preserve the original allocation order, shapes, dtype, device and zero-width
  return. Do not initialize scales, reject the input or choose another backend.
- Compare exact scale bytes only after the returned pointer matches this same
  isolated process's just-released allocation filled with 101.25 or -202.5.
  Refuse every read if the pointer does not match.
- Do not require identical uninitialized scale bytes between independent,
  uncontrolled allocator histories. All defined numerical outputs and source
  errors remain exact. Autotuning policy remains unchanged.

The approved validation adds **24 reference cases**
(two helpers × three dtypes × two row counts × two patterns), then the matching
24 Rust cases in normal and Python-free execution. These would stay separate
from ordinary numerical/error fixtures. No other input activations, groups,
seeds, operations or unknown storage are covered by this extension.

## Approval record

Tomas answered **“ja”** to the immediately preceding question approving the same
criterion for `rowwise64` and `staged`, plus 24 controlled cases. This approves
exactly the bounded scope above. The approval was recorded before the new GPU
probe started. Defined results remain byte-exact; autotuning remains unchanged.

## Why this needed a separate approval

Tomas requires explicit approval for each non-exact criterion. The approved
[design, Section 14.2](../rust-comfy-parity-engine.md) requires “every specific
non-exact acceptance criterion has explicit approval”. The existing decision
names two other calls; treating it as permission for these additional helpers
would silently expand its operation scope.

At the previous stop, the independent W4A4 check completed and was documented.
No empty-variant execution took place before the explicit approval above. The
new 24-case investigation is now authorized; it does not authorize broader
activations, groups, seeds or unknown storage reads.

## Approved result

All **24** new original-reference cases passed. The current Rust empty test
passes all **48** approved combinations (prior 24 plus these 24), in normal and
Python-free execution. Every returned scale pointer matched this process's
known-pattern allocation before it was read. Exact metadata and bytes match;
no kernel, allocation policy or tuning change was needed.

The focused check reuses prior successful numerical-suite evidence for unchanged
inputs. Its own test filter requires exactly one executed passing Rust test,
so an unmatched filter cannot be reported as success. Detailed hashes are in
[`stage-1-evidence.json`](../stage-1-evidence.json).
