# Stage 1 SM89 signed INT4 native/chunked continuation

Date: 2026-10-17. Status: **67 crate tests / 59 Python-free primitive functions pass**.
Stage 1 and full model/engine parity remain incomplete.

Tomas explicitly approved the [remaining INT4 criterion](decisions/D-05-INT4-remaining-zero-k-output-family.md)
with “Ja”, before dependent implementation/probes. He then clarified that work
needed to match ComfyUI requires no additional approval; only actual departures
do. This authority clarification is recorded in the [main design](rust-comfy-parity-engine.md).
Prior matrices remain historical exercised coverage, not new implementation
support limits. No numerical tolerance or changed inference algorithm is adopted.

## Implemented typed direct-helper selection

`int4_linear::linear` now retains the original direct CUDA helper's initial
SM89 selection. The shared prefix validates rank/equal packed width, allocates
the original outer output, converts/counts FP32 scales, prepares the original
aligned output-kind bias/empty operand and obtains the active stream.

With force-INT8 off, the original legacy preference selects the native kernel
or tries CUTLASS; its returned boolean controls original fallback. With the
original force-INT8 option on, unpacking is followed by packed-weight GEMV at
M<=8 or the original chunked native wrapper at M>8. Chunking keeps the original
N<=2,560 shortcut and otherwise uses the original clamped chunk-size setting
(default 4,096). The decoded-weight and INT32 workspaces are allocated in source
order and retained through return/observation, as is the unused outer output.
The existing shared cuBLAS workspace/capability/empty/vector helpers are reused.
No new backend, eager replacement, output initialization or shape restriction
is introduced. The previous selected small-M entrypoints use this same body.

The original native C++ seven-candidate INT4 autotuners, caches, timing code,
geometry, arithmetic and return flags are reused unchanged. Their internal
measured winners are not exposed or compared by this report; there is no waiver
of defined output equality. Native CUTLASS refusal-to-legacy code is preserved,
but this numerical matrix contains successful CUTLASS returns rather than an
actual source-native false-return case. Do not claim that branch was exercised.
Generic kwargs/backend binding and other GPU architectures remain pending;
this is a typed primitive implementation on the initial SM89 platform.

## Original device metadata

The C ABI device-properties result gains only `multiprocessor_count`, read
through the same ATen-cached CUDA properties used by Python. Rust uses it in the
original integer legacy-preference formula; it is not a hard-coded card value.
The reference and Rust both report **66 SMs** on the selected RTX 4070 Ti SUPER.
Existing fields and native kernel bodies are unchanged. The shared library was
rebuilt for this metadata ABI change; its hash changes and the prior binary is
retained. All **32 original CUDA object files remain byte-identical**.

## Defined numerical and selection evidence

**110 defined-input outputs** match all bytes, dtype, device, shape and stride,
as well as every recorded native call/boolean return and chunk argument:
55 forced chunked routes, 36 native legacy routes and 19 successful native
CUTLASS routes. The matrix covers three output kinds independently of three
scale kinds, bias absent/present, native/forced policy, original CUTLASS-disable
option, the original chunk-size override and a real multi-chunk tail.
Geometries include M/N/K 9/4/64, 17/20/128, 9/2,576/64 with chunk512,
1/32,768/64, and 33/128/128. Two additional initialized FP16 cases cover
1024/2560/64 forced chunking eligible for the original internal strided-CUTLASS
branch, and 128/4096/64 native selection. The internal strided return flag is
not separately exposed; no per-candidate numerical coverage claim is inferred.

**100 predicate cases** compare the actual original legacy preference over
five M, four N and five K values using the same hardware SM count. These are
selection metadata evidence, not extra CUDA numerical outputs.

All **six defined legacy K=0 controls** match the original fully written
zero/bias output exactly, and their traces call only the original legacy INT4
kernel. Scales are own initialized ones; bias cases use the ordinary synthetic
arange*0.015625-0.25 vector. This is ordinary defined-result validation, not
controlled-unwritten-output acceptance. No inferred bias/zero replacement is
used in the engine.

## All 24 approved controlled no-write cases

The original reference and focused Rust checks pass all 24 combinations:
forced chunked M=9/N=4/K=0, or native CUTLASS M=1/N=32,768/K=0; FP32/FP16/BF16;
bias absent/present; patterns 0x35/0x6a. Scales are own initialized FP32 ones;
optional output-kind bias is arange*0.015625. Warmup uses only initialized K=64
operands on the same original policy, preserving CUTLASS acceptance/refusal.
Warmup and zero-K native traces are compared as well.

Forced chunked cases prime three own 144-byte buffers, within the approved
256-byte maximum. Native CUTLASS cases prime two buffers of exactly the output
size: 65,536 bytes for FP16/BF16 or 131,072 for FP32. Before inspection/return,
metadata-only observers prove each complete output range belongs to freshly
primed own storage. Every returned byte equals that known pattern exactly.
The chunked launch returns before consuming its unused workspaces; those bytes
are never inspected. Native CUTLASS returns true without writing at K=0.

No unknown storage is consumed/read, raw address is logged, original output is
filled/repaired, or stream/algorithm/default changed. Ordinary calls install no
observer. The native legacy controls remain fully defined and are not subjected
to this no-write criterion. The registered floating ConvRot wrapper's earlier
reshape[-1,0] rejection remains distinct; no model-level K=0 success is claimed.

## Full verification and limits

The full locked/offline chain passes fmt, Clippy with warnings denied,
**67 crate tests** (1 CPU, 3 policy, 59 primitive, 4 registry) and an unfiltered
repeat of **all 59 primitive functions without Python, ComfyUI or network**.
All four INT4 functions pass, including the previous 360 ordinary and 36
controlled small-M cases after sharing the dispatcher. Current totals are
**11,489 ordinary numerical/error/selection records** (11,273 prior + 110
outputs + six defined controls + 100 predicates), and **3,572 unique controlled
combinations** (3,548 prior + 24). The [evidence](stage-1-evidence.json) retains
the earlier 65/57 snapshot, exact artifacts and source hashes. Fourteen scoped
D-05 records are approved and verified; no actual deviation approval is pending. Previous fixture categories and
the complete previous fixture bytes are retained. Hex payload encoding for the
larger own synthetic outputs is lossless; comparison still checks every byte.

Four Stage 0 locks, original CUDA sources/objects and compiler flags, Triton/AOT
artifacts, RNG and tuning policy are unchanged. Native compilation retains its
previous GCC CUDAEvent alignment warnings; Rust Clippy uses warnings denied.
No new dependency, reference advance or numerical waiver occurred.

Only own isolated synthetic development data was used. No model/checkpoint,
ordinary Dreamchat runtime data, process list, service stop, browser/manual
acceptance, product integration, cloud inference or later implementation stage
was accessed/run. GPU/build/namespace commands used authorized sandbox escalation;
no new automatic approval rejection occurred.

Stage 1 still needs generic executable backend binding, broader layout/operator
and remaining Kitchen/quantizer coverage, general Triton specialization/compiler
integration, and packaging. Successful tests here prove the exercised primitive
subset, not complete Stage 1 or full model parity. Source-faithful continuation
needs no per-branch approval under Tomas's clarification; a real proposed
behavioral/algorithmic/default/architectural deviation still requires approval.
