# Stage 1 AWQ W4A16 primitive continuation

Date: 2026-10-17. Status: all **400 differential cases**, **68 crate tests** and
**60 Python-free primitive functions** pass. Stage 1 and full engine parity
remain incomplete.

This continuation implements the frozen eager and initial-SM89 CUDA AWQ direct
helpers in [`awq.rs`](../inference/src/awq.rs). It follows the existing
[authority clarification](rust-comfy-parity-engine.md): implementation needed for
source compatibility proceeds without another approval. No deviation, numerical
tolerance, different default or new dependency is adopted.

## Executable source behavior

The eager helper retains flattening, divisibility and packed-width validation,
unsigned nibble unpacking, group dequantization in `wscales.dtype`, ATen matmul,
out-of-place bias addition and restoration of the input's leading dimensions.
Its scale/zero transposes precede the dequantization arithmetic, as in the source.
The existing unsigned codec has exactly the same int32 unpack/stack/reshape/int8
order as the eager AWQ helper and is reused.

The CUDA helper retains the original **M > 256** ATen dequant/matmul fallback.
The source docstring says 512, but the executable `_AWQ_W4A16_MMA_M_LIMIT` is 256;
the implementation follows the constant. For M <= 256 it calls the unchanged
`awq_w4a16` native entrypoint with the current ATen stream. On the initial SM89
platform the native launcher uses the original naive kernel at M <= 8 and the
original MMA kernel at M > 8. The input becomes contiguous before conversion to
the scale dtype. Packed weights are viewed as uint8; no additional weight,
scale or zero contiguity copies are added. Native operand capacities/layouts
remain an explicit unsafe caller contract, as in the existing low-level bridge.

CUDA bias uses the original in-place `add_`, preserving output dtype. Eager bias
uses the original out-of-place addition and can promote FP16/BF16 results to
FP32. The two backend algorithms are compared against their own original
implementations; eager is not substituted for the native kernel and their
outputs are not required to equal each other where the original paths differ.
The large-M CUDA helper retains its distinct int8-before-stack nibble conversion,
view semantics and operation order rather than being replaced by the eager body.

The default group remains 64. Scalar input, malformed weight rank, zero group,
non-divisible K, mismatched packed K and zero-K reshape errors retain original
class/message and evaluation order. In particular the CUDA wrapper reads weight
dimension zero before checking divisibility, whereas eager checks divisibility
first. The large-M CUDA helper's later tuple-unpack check is also preserved.
No extra validation changes these original errors into an engine-specific error.

## Exact differential evidence

The final matrix has **368 byte-exact results and 32 exact reference errors**:

| Backend | Results | Errors |
| --- | ---: | ---: |
| CPU eager | 123 | 11 |
| CUDA eager | 123 | 11 |
| CUDA native wrapper and ATen fallback | 122 | 10 |

The ordinary cross product covers independent FP16/BF16 activation and scale
dtypes, absent/same-kind/FP32 bias, contiguous and strided inputs, and these
M/N/K/group geometries: 1/7/64/16, 8/17/128/32, 9/17/128/64,
256/16/64/64 and 257/7/64/16. Rank-one and rank-three inputs test restoration
of leading dimensions. Non-power-of-two N tests native output tails.
The M=8/9 and M=256/257 pairs exercise both executable tier boundaries.
Synthetic activation, scale and zero values also exercise FP16/BF16 rounding.

Additional cases cover eager FP32, genuinely strided packed weights and
multi-group scales/zeros, and the corresponding large-M CUDA fallback. Error
cases include simultaneous failures to verify original precedence. The source
naive native kernel also has a defined M=1, group=-64 behavior: its negative
group count skips all reads and stores FP32 zero converted to the output dtype.
That initialized, fully written output is matched exactly, while eager's negative
group raises its original invalid-view error. This is not an unwritten-output
case or a new guarded-allocation acceptance rule.

For every successful result, the test compares all output bytes, dtype, device,
shape and strides. For every rejection, it compares the full original exception
class/message, removing only the language-binding C++ stacktrace suffix already
handled by the existing test helper. All preflight errors record no native call.
The CUDA successes trace **97 native calls and 25 ATen fallbacks**; the Rust
metadata-only observation agrees with the original native-call trace. The
observer is absent in ordinary execution and changes no stream or allocation.
Internal naive/MMA selection is determined by the unchanged source launcher;
the native subkernel is not separately instrumented.

Fixture payloads use lossless hex and every byte is compared. Original fixture
categories and prior complete fixture bytes are retained separately. The
capture aborts resource failures instead of treating OOM as an expected error.

## Validation and remaining work

Focused tests, fmt and Clippy with warnings denied pass. The complete locked/
offline suite passes **68 crate tests** (1 CPU, 3 policy, 60 primitive, 4 registry),
followed by an unfiltered repeat of **all 60 primitive functions without Python,
ComfyUI or network**. The current aggregate is **11,889 ordinary numerical/error/
selection records** (11,489 prior + 400 AWQ), with **3,572 unique controlled
combinations** unchanged. [The evidence file](stage-1-evidence.json) records exact
source/artifact hashes and logs and retains the prior 67/59 successful snapshot.
No original native/CUDA/Triton source, binary, build flag, RNG, autotuner, cache
key, candidate or tuning policy changes. The existing compiled AWQ native
entrypoint is used without rebuilding the C++ bridge. Four Stage 0 locks remain
the reference; no model weights or ordinary runtime data are read.

This is a typed direct-helper port. AWQ quantized-layout dequantization,
transpose/state export, mixed layout dispatch and generic executable backend
binding remain separate Stage 1 obligations. The original AWQ layout explicitly
raises `NotImplementedError` from `quantize()` and requires offline calibration;
parity must preserve that behavior instead of inventing an online quantizer.
The general requirement to reproduce ComfyUI quantization remains unchanged for
operations the reference actually implements. No checkpoint loading, model
inference, Dreamchat/product integration, browser/manual acceptance, cloud
inference, process inspection or service stop was performed. Local GPU/build/
namespace commands use the existing authorized sandbox escalation; no automatic
approval rejection occurred in this continuation. No commit or push was made.


Subsequent work: the [AWQ layout continuation](stage-1-awq-layout.md) implements
and verifies the typed metadata/decode/transpose/export and operator handler
bodies that were pending at this direct-helper checkpoint. Its evidence and
remaining global binding obligations are separate from the historical 400-case
primitive result above.
