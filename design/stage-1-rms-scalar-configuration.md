# Stage 1 RMS scalar conversion and Kitchen configuration

This continuation implements the frozen source without new approvals or changed
numerical criteria. All evidence uses initialized own inputs in the isolated
engine workspace; no ordinary Dreamchat runtime data is used.

The original INT8 input activation uses Torch's optional-double parser in eager,
Triton and unfused CUDA execution. It accepts None and non-gradient 0D tensors,
while rejecting higher-dimensional or gradient tensors. The fused CUDA path
instead calls Python float: one-element tensors and gradient tensors are
numerically accepted, as are numeric strings. Its None, string and complex
conversion errors differ. A shared source-derived fused eligibility predicate
preserves which conversion applies; fusion and autotuning stay unchanged.

All 174 executed original cases match, including all result bytes and exception
class/messages. The synthetic matrix includes three backends, fused/unfused
geometry, optional defaults, bool/integer/uint64/scalar inputs, numeric/invalid
strings, scalar/vector/gradient tensors, containers, dtypes/devices and complex
scalars. Evidence: `runtime/stage1-int8-rms-epsilon-extended-focused.log`.

The new item<double> bridge initially missed a standard C++ overflow exception
for complex conversion. That failed synthetic run is retained as
`stage1-int8-rms-epsilon-uncaught-std-exception-rejected.log`; it is not passing
evidence. The corrected FFI catches c10, standard and unknown exceptions before
they cross into Rust. The final 174-case run passes the original RuntimeError.
Python warning-filter/emission state has not been independently validated by
this numerical/exception matrix; no claim about all Python SDK diagnostics is
made from these cases.

Kitchen's original import-time chunk-size expression now parses Python decimal
syntax, Unicode digits, signs/underscores, surrounding space, surrogateescaped
invalid environment bytes and the frozen 4300-digit limit. Bad values raise the
original import error rather than becoming a later binding gap. Large positive
settings retain the same consumed min(tensor_rows, setting): signed int64 tensor
row dimensions allow that effective bound to saturate at i64::MAX without
changing any chunk. Zero/negative values clamp to one as in the original.

All 34 source-verified CPU configuration cases pass, including exact error
classes/messages, four flag interpretations and consumed chunk boundaries.
The original unbounded Python module attribute is not exposed as a Python
reflection API by this Rust inference interface. Evidence:
`runtime/stage1-kitchen-configuration-focused.log`.

The bridge hash is now
`f87a4e609786c2eaa0bb0fdc8e38c2046dff635e99769af746367b9cade34ce6`.
All 32 original CUDA object hashes and four Stage 0 lock hashes remain unchanged;
`runtime/stage1-native-uint-scalar-proof.json` records the verification.
Full unfiltered integration and remaining source-backed argument audits are
still required before declaring Stage 1 complete.
