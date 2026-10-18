# D-05-REGISTRY-DEVICE-SET-ORDER: unordered allowed-device diagnostics

Date: 2026-10-17. Status: **explicitly approved by Tomas; implemented and verified for the recorded metadata scope**.

Tomas explicitly approved: “Ja, jag godkänner fast sortering även av enhetsnamnen med samma exakthetskrav och oförändrad autotuning. Dokumentera beslutet och fortsätt steg 1. Stoppa om något annat kräver mitt uttryckliga godkännande.”
This lifts the separate device-set approval stop before implementation.

## Concrete reference evidence

The frozen `constraints.validate_param` renders `set(allowed)` at
`constraints.py:205`. The actual frozen Triton registrations for `adaln`,
`rms_adaln` and `quantize_per_tensor_fp8` allow `cuda` and `xpu`. Each rejects
an own synthetic CPU float32 tensor before executing a kernel.

Six fresh CPU-only reference processes (hash seeds 0, 1, 2, each with two own
import-allocation contexts) produced both messages for all three registrations:

```text
device cpu not in {'cuda', 'xpu'}
device cpu not in {'xpu', 'cuda'}
```

All 18 observations retained the same failed parameter and allowed-device
membership. The original validators and device sets were unchanged. This is a
text observation, not numerical validation or acceptance of a new criterion.
The report is `runtime/device-diagnostics-processes.json` under the isolated
development root; hashes are in [the aggregate evidence](../stage-1-evidence.json).
GPU visibility was disabled. No tensor values, ordinary runtime data, process
lists or services were inspected.

## Approved criterion

Only for allowed-device set literals in frozen `constraints.validate_param`
failure reasons, render the exact original quoted device names in fixed
lexicographic order. Retain identical members, cardinality, braces and
separators. The two examples above would both be compared as:

```text
device cpu not in {'cuda', 'xpu'}
```

All surrounding text, rejected device, failed parameter, validation order,
exception class, backend choice and fallback must remain exact. Preserve every
numerical result and the original autotuning policy. This does not change the
allowed devices or authorize any other diagnostic normalization.

## Why a separate approval was required

The approved [type-set decision](D-05-registry-type-set-order.md) explicitly says:
“Device sets, other unordered diagnostics, value rendering, exception classes
and all numerical results remain outside this criterion.”
The [design, Section 14.2](../rust-comfy-parity-engine.md) requires “every specific
non-exact acceptance criterion has explicit approval”. This is a distinct set
literal outside that approval. Before this approval, Stage 1 stopped without adopting or implementing
this criterion. The earlier partial checks are not retroactively claimed as
complete diagnostic parity. Implementation and verification results follow below.

## Implementation and verification

Rust now renders the exact frozen device names with the approved sorted order,
including singleton/empty-set syntax. The unchanged original validator supplies
both raw messages and approved expected strings; comparison logic verifies the
full original message before applying only this presentation change.
All 2,891 device failure strings are compared in the 20,864-case metadata suite.
The 440 existing selection cases now compare complete public error messages,
and 153 additional state/override cases check exact ordered failure details.
The focused normal and Python-free passes are recorded in the
[registry continuation](../stage-1-registry-diagnostics.md).
