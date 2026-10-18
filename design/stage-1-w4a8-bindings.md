# Stage 1 W4A8/W6A8 bindings

The frozen eager and CUDA quantize, dequantize and linear bodies are connected.
The differential test passes **1,171 synthetic records**: 1,099 defined
outputs and 72 original errors. Each record checks complete payload bytes,
dtype, dimensions, strides, post-call inputs, aliases, object identities,
explicit backend trace and automatic selection. The full checkpoint passes
**90 crate tests / 82 unfiltered Python-free primitives**. See the
[checkpoint evidence](stage-1-eager-cuda-checkpoint.json).

The implementation keeps the original packed four/six-bit planes, INT8-grid
rounding/clamping, two ALS iterations, fixed/heavy-tail fitted codebooks,
grid-aware scale neighbours, stored scale dtype and seeded stochastic
assignment. CUDA retains fused, staged and chunked quantization, GEMV/chunked
linear selection, the L2-size rule, activation-quantizer selection, CUTLASS
return handling, eager fallback and correction/bias operation order.

The binding-only ATen bridge now exposes its cached `cudaDeviceProp::l2CacheSize`
(the field backing Python's `L2_cache_size`) and generator-backed `at::randint`.
No original Kitchen kernel or arithmetic is changed. The existing ATen CPU/CUDA
generators supply the source's local seeded sample/rounding streams; there is no
production Python dependency. `serde_json`, already locked and used by tests,
is also linked normally for the existing native Sol workspace-plan protocol.

## Retained reference failure

The extra large CUDA chunk test with `(M,N,K)=(9,4103,256)` faulted inside the
original backend with `CUDA error: misaligned address`. A fresh subprocess
isolated it from the two successful M=512 cases. The acceptance matrix uses
N=4104 for that chunk-boundary case and does not turn an asynchronous accelerator
fault into an expected error fixture. The N=4103 case remains unvalidated; no
public shape gate, kernel repair, fallback, tolerance or failure-timing promise
was added. The failed diagnostic is retained in the own development root at
`runtime/stage1-w4a8-diagnostic.log`.

Tomas's whole-Stage-1 authority covers this source-faithful work. No new numerical
criterion, reference update, autotuning policy or parity departure is adopted.
Earlier scoped approvals remain bounded by their ledger entries.
