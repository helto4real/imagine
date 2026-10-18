# Stage 1: controlled empty ConvRot, Hadamard and packed weight continuation

Date: 2026-10-17. Status: **43 crate tests / 35 Python-free primitive tests passed**.
Stage 1 remains incomplete. Tomas subsequently approved the scoped
[empty-variant extension](decisions/D-05-INT8-ConvRot-empty-variants.md); its
24 reference/Rust cases passed, bringing controlled empty cases to 48.

## Approval and reference boundaries

Tomas answered “Ja” to the immediately preceding two-point question approving
the [original empty-width criterion](decisions/D-05-INT8-ConvRot-empty-width.md)
and the bounded three-dtype/one-and-four-row investigation. That approval was
recorded before GPU execution resumed. It preserves original allocation order
and the absence of scale writes. Independent uncontrolled allocator histories
are not required to produce identical undefined scale bytes.

The expanded original reference probe and Rust both pass **24 controlled-state
cases**: float32/float16/bfloat16, one/four rows, public INT8 weight quantization
and direct rowwise quantization, group 256, seed 0 and the two exact known
patterns 101.25 / -202.5. Rust repeats these in Python-free isolation. Every
scale value read follows a successful match to the pointer of this same
process's just-released known-pattern allocation. Unknown unwritten storage is
never inspected. These cases are separate from numerical/error fixtures.

All defined numerical outputs remain byte-exact. The original INT8/Triton
autotuning candidates, timing procedure, selection and caches are unchanged.
The earlier automatic-review rejection remains historical evidence; the later
explicit user approval authorized the bounded work, with no workaround used.

## Implemented and compared

| Mechanism | Cases | Exact successes | Expected source errors |
| --- | ---: | ---: | ---: |
| Hadamard construction/cache | 60 | 24 | 36 |
| Eager weight/activation rotation helpers | 252 | 78 | 174 |
| High-level CUDA INT8 ConvRot weight quantization | 126 | 108 | 18 |
| CUDA INT8 ConvRot weight dequantization | 81 | 51 | 30 |
| Signed/unsigned row-major INT4 storage codecs | 36 | 28 | 8 |
| Native packed-INT4 to INT8 weight preparation | 6 | 3 | 3 |
| CUDA ConvRot W4A4 weight quantization | 135 | 72 | 63 |
| CUDA ConvRot W4A4 weight dequantization | 81 | 33 | 48 |

These **777** additions bring ordinary tensor numerical/error fixtures from
1,449 to **2,226**. Successes compare exact payload/scale bytes and relevant
shape/stride/dtype/device/alias metadata. Expected errors compare original class
and full message; binding-owned C++ stack locations are excluded from the
message comparison. Resource failures cannot become golden expected errors.

`int8_utils.rs` retains the H4 coefficients, Kronecker order, scalar
normalization, dtype/device conversion, matmul/reshape ordering and cached
storage aliasing. The exercised cache boundary uses canonical typed CPU/CUDA
devices. Generic Python device-string spellings and arbitrary dynamic argument
handling remain pending; no production API has been narrowed.

`convrot.rs` now implements the original high-level INT8 weight selection and
native/fallback weight decode. Its hardware decision reads the same ATen-cached
CUDA capability/default/opt-in shared-memory properties as Python, caches by
device and retains the strict `<` memory comparison and branch order. SM89
ordinary cases execute rowwise64, staged and eager rotation; the separately
approved empty cases execute rowwise. Other hardware is not claimed tested.

`convrot_w4a4.rs` retains the original quant-group-64 validation, source shape
errors, INT4-specific shared-memory calculation, original native quantizer and
nonfused Hadamard-plus-quantizer path. Scales retain the original flattened
layout. Native decode and the signed-nibble eager fallback use original order
and dtype conversion. The prepared-INT8 path calls the original unpack kernel.
The signed storage codec preserves the full -8 through 7 nibble range; it does
not route through NVFP4 or a different quantizer.

Native symbol-presence conditions match this frozen complete 69-entrypoint
build. Packaging and initialization of other library variants remain pending.
Full W4A4 linear/layout math, W4A8/W6A8 codebook/weight/linear paths and the
remaining layout/dispatcher/JIT obligations are not covered by these results.

## Build and validation record

The source-matched native library was rebuilt for the ATen device-property ABI.
All original CUDA sources, SM89 target, compiler and arithmetic flags remain
unchanged. The build now mounts the same pinned manylinux system directories
read-only under a temporary Bubblewrap root, avoiding a redundant Podman VFS
writable-layer copy. Native sources and transitive headers were hash-checked.

An initial mount setup failed before compilation because `/src` did not exist
on the read-only root. It was corrected by using a temporary root while binding
the original image directories read-only. The successful build and its original
third-party alignment warnings are retained in the evidence.

Two Cargo link attempts failed with signal 7; `/tmp` subsequently reported its
user quota exhausted, including shell/sandbox temporary-file creation. Only
this task's disposable incremental cache was removed. The own Cargo target was
then moved onto the workspace disk at ignored `inference/target/`, preserving
its logical `/tmp/imagine-rust-comfy-parity-20261017/target` path via a symlink.
Reference sources, numerical fixtures, logs, tuning caches and compiler flags
were retained. The later complete chains pass; failed attempts remain failed.

Completed checks: locked/offline Cargo suite (**43** tests), all **35** primitive
functions repeated without Python/ComfyUI/network, formatting and Clippy with
warnings denied. The new W4A4 chain reuses unchanged native/Triton artifacts.
The prior standalone compiler's six exact complete-cubin cases are separate,
unchanged evidence, not a new general JIT claim.

No checkpoint/model inference, product integration or browser/manual validation
was performed. No ordinary Dreamchat data, GPU process list or other workload
was inspected; no service was stopped. No commit or push was made.

Detailed file/artifact hashes, logs, counts, source-only pending finding and
approval records are in [`stage-1-evidence.json`](stage-1-evidence.json).
