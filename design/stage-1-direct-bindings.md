# Stage 1: direct Flash-decode and DeltaNet helpers

Date: 2026-10-18. Full **94 crate tests and 86 unfiltered Python-free primitives
pass**. Stage 1 remains incomplete.

[kitchen_direct.rs](../inference/src/kitchen_direct.rs) implements the frozen
public Flash-attention decode, gated DeltaNet decode and DeltaNet convolution
step wrappers. These functions bypass the backend registry in the original as
well; the 107 registered eager/CUDA body count is unchanged.

The port retains Flash's query/head reshaping, exact split-efficiency heuristic,
SM80 availability gate, original scratch shapes and empty scratch aliases. It
retains DeltaNet's SM75 and opt-in shared-memory checks, per-device property
cache, dtype/device/contiguity preparation, optional snapshots, in-place state
updates and rejected-launch errors. Current-device lookup uses ATen's original
CUDA device state. The existing original native wrappers and CUDA objects are
invoked; no alternate attention or recurrence algorithm is substituted.

The 631 synthetic source records contain:

| Surface | Exact outputs / booleans | Original errors |
| --- | ---: | ---: |
| Flash availability | 3 | 0 |
| DeltaNet availability | 60 | 0 |
| Flash split heuristic | 80 | 100 |
| Flash decode | 12 | 52 |
| DeltaNet convolution step | 39 | 105 |
| Gated DeltaNet decode | 30 | 150 |

All initialized arguments and optional snapshots are compared after each call.
The typed Rust helpers in this checkpoint cover ordinary tensors. This does not
claim complete arbitrary tensor-subclass or nanobind conversion-error coverage.
Source captures use CPU and the initial SM89 GPU. Head shape/shared-memory
rejections are retained; a rejected shape is not silently implemented by a
fallback. There is no new numerical criterion, tuning change or approval request.

The fixture now contains **23,744 physical records**, including the preceding
five approved Sol records and ten approved null-export records; ordinary records
number 23,729. The separate earlier 3,572 controlled unwritten-storage combinations
are unchanged. See [checkpoint](stage-1-direct-checkpoint.json).

Own evidence logs under `/tmp/imagine-rust-comfy-parity-20261017/runtime`:
`stage1-direct-reference.log`, `stage1-direct-build.log`,
`stage1-direct-native-build.log`, `stage1-direct-focused.log` and
`stage1-direct-full.log`. No ordinary runtime data, model run, process-list
inspection, service change, commit or push was involved. Namespace/GPU commands
used authorized sandbox escalation; no automatic approval rejection occurred.

Remaining direct Kitchen work includes Sage attention, Sol's chunked producer
and its allocation-context/cache lifecycle. General Triton and other Stage 1
completion obligations remain in the [ledger](stage-1-completion.md).
