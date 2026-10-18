# D-05-SOL-ATTN-EAGER-FP32-SCATTER — approved controlled-intermediate criterion

Date: 2026-10-17. **APPROVED by Tomas, replying “Ja” to the six-case controlled-intermediate question.**
Approval recorded before dependent implementation or validation. No numerical
tolerance is adopted. At approval time no Sol-Attn engine body was installed. The completed implementation checkpoint at approval was
[83 crate / 75 Python-free tests](../stage-1-eager-quant-bindings.md), with
94 connected bodies and 41 explicit gaps.

## Why a decision is needed

The [binding design, Section 2.1](../rust-comfy-parity-engine.md#21-functional-algorithmic-and-numerical-parity)
requires: “If an exact comparison cannot hold because of reference
nondeterminism or a backend limitation, document the cause and propose a bounded,
operation-specific acceptance criterion for Tomas's approval.” Existing fourteen
D-05 approvals do not cover this reduction or permit a general tolerance.

The frozen **original eager Sol-Attn**, without any Rust implementation involved,
produces different FP32 bits for identical initialized CUDA inputs at shape
`(B,T,H,D) = (1,513,1,128)`. The source retains its registered head-dimension-128
requirement. `block_len=None` means all actual rows are live; this finding is not
an unwritten-output or dead-row case.

An initial 17-call ordinary repeat produced three complete output hashes, with
maximum absolute difference from the first output `5.960464477539063e-08`.
Three other ordinary geometries remained bit-identical in that sample. These
observations are diagnostic findings, not accepted comparison bounds.

The first observed divergent operation is
`comfy_kitchen/backends/eager/sol_attn.py:199`:

```python
colmean.scatter_add_(2, qblk.view(1, 1, t, 1).expand(b, h, t, n), s_blk)
```

Across 32 traced original calls, the pre-call zero tensor, index and source
values have identical hashes and metadata, while the post-scatter matrix has
32 distinct hashes. The complete function has two output hashes in that run.
The acquired pinned PyTorch CUDA `ScatterGatherKernel.cu` uses atomic addition
for the reduction. Its Git blob is verified against the unchanged Stage 0
acquisition lock; this is not inferred solely from a general nondeterminism note.
Exact measured deltas and artifact hashes are in [aggregate evidence](../stage-1-evidence.json).

The diagnostic observer calls the original method, copies only owned synthetic
intermediates and restores the original Python method. It introduces diagnostic
synchronization; it changes no kernel, flag, reduction algorithm or operand.
Ordinary uninstrumented repeats independently demonstrate the differing final
outputs. An initially absent isolated JIT-cache directory was created at the
already configured own cache path; the traced variability remains with a working
cache. Deterministic algorithms remain false and FP32 matmul precision highest.

## Approved criterion

Tomas approved **only the following controlled-intermediate validation criterion** for
original eager Sol-Attn, FP32 on CUDA/SM89, in the six cases below:

1. Ordinary engine execution retains original `aten::scatter_add_`, operand
   construction, reduction dimension, ordering of surrounding operations, default
   flags and autotuning. No deterministic replacement, dtype change, alternate
   backend or new production default is introduced.
2. In isolated comparison tests, execute the original reduction first. Verify
   its operator, dimension, operand bytes, shapes, strides, dtypes, device and
   current stream against the source contract. Abort the fixture if these differ.
3. Before the first consumer, **only in the comparison harness**, copy the same
   explicitly recorded owned canonical post-scatter tensor into reference and
   Rust. The canonical bytes come from a genuine original call on these exact
   own inputs. They are initialized, hash-recorded and structurally checked.
   The five ragged variants share identical pre/index/src fingerprints, so the
   same recorded `(1,1,9,9)` state applies to each.
4. Require byte-exact downstream outputs and deterministic metadata, unchanged
   inputs and original dispatch. Keep the `(1,256,1,128)` ordinary case as an
   additional full-path byte-exact control without state substitution. If another
   independent divergent operation appears, stop rather than extend this rule.
5. Label the results **controlled-intermediate validation**. Approval permits not requiring independent raw atomic outputs/default full-path outputs
   to be bit-identical for the five ragged cases. Those raw results must never be
   reported as ordinary deterministic numerical parity. No floating tolerance,
   measured-delta bound or global numerical exception is authorized by this rule.

**This changes the validation criterion for one identified source
nondeterminism.** It does not establish a unique bit pattern for ordinary raw
FP32 atomic execution. Implementation/algorithm parity and the full remaining
Stage 1 obligations continue to apply.

## Six approved cases

All use distinct owned FP32 q/k/v on `cuda:0`, head dimension 128, one head,
`scale=None`, `tau=1`, `sink_q=None`, `block_len=None` and `token_aug=0`.
Unlisted optional arguments keep the original defaults.

| Case | q/k/v shape | Additional arguments | Approved check |
| --- | --- | --- | --- |
| Ordinary control | `(1,256,1,128)` | defaults | Full-path byte equality; no substitution |
| Ragged default | `(1,513,1,128)` | defaults | Controlled post-scatter state; exact downstream |
| No pooled tail | `(1,513,1,128)` | `tail=False` | Same controlled state; exact downstream |
| Top-k | `(1,513,1,128)` | `topk_ratio=0.25` | Same controlled state; exact downstream |
| Key bias / sink | `(1,513,1,128)` | FP32 key bias 0.125 for keys 0–63, zero otherwise; `sink_blocks=[0,1]` | Same controlled state; exact downstream |
| Coarse branch | `(1,513,1,128)` | FP32 `coarse_gate` filled with 0.25, matching q shape | Same controlled state; exact downstream |

q/k/v values are formed independently from FP32 CUDA arange with offsets 0/17/31:
`(((arange(numel) + offset) % 97) * 0.03123 - 1.5032).reshape(shape)`.
The selected original callable and finite output were verified for every case.
Canonical states and exact pre/index/src fingerprints are prepared, with
`controlled_state_criterion_applied=False` and `canonical_copy_applied=False`
for every captured case. No comparison test had replaced the state at that proposal checkpoint.

These six cases are bounded approved verification coverage, not new public input
restrictions. No waiver is requested for another dtype/backend, additional
nondeterministic operations, unsafe accesses or changes to output semantics.
Additional source-faithful implementation and ordinary exact cases remain
within the existing authority; extending this numerical criterion needs its
own applicable authorization.

## Artifacts and current stop

The aggregate evidence records both standalone probe scripts/launchers, raw
logs, repeat/trace reports, source hashes, pinned Torch Git blob and canonical
state hashes under the isolated development root. Probe calls are diagnostic
source-only executions and are **not** added to the 18,250 accepted ordinary
fixture records or 3,572 controlled combinations. The six original calls verify
proposal feasibility; they do not validate the unimplemented Rust Sol-Attn path.

Only own initialized synthetic data is involved. No model/checkpoint, normal
Dreamchat runtime data, cloud service, GPU process listing, service stop,
production integration, commit or push is involved. Authorized local GPU commands
used escalation; there was no automatic approval rejection.

**Approval received; continue the faithful binding and six-case validation. Stop
only if another independent divergence requires extending this criterion.**


Prepared [canonical fixture](fixtures/sol-attn-eager-fp32-scatter-proposed.json)
SHA256: `da1f41413837c8e19d4bfffa204a678f7b6ed552a8aa73f56f7d645480c0aee4`. It retains its proposal labels as immutable capture provenance. The approved
comparison harness now consumes its exact canonical bytes and operand fingerprints. The last trace measured maximum differences
from its first result of `7.62939453125e-06`
at the reduction and `2.9802322387695312e-08`
at the final output. These measurements are not tolerance thresholds.

## Approval record

Tomas explicitly answered “Ja” to the six-case controlled-intermediate question.
Five ragged cases may share the recorded post-scatter state in comparison tests;
the ordinary T256 control uses no replacement. Original scatter, exact operand
checks, byte-exact downstream results and unchanged autotuning remain mandatory.
The proposed JSON below is preserved byte-for-byte as capture provenance; its
proposal labels and false copy flags describe the earlier source-only capture.
They do not supersede this approval and are not a claim that validation has run.

## Completed verification

The approved six cases pass in reference/Rust and the Python-free full suite;
see [the binding report](../stage-1-sol-attn-bindings.md). Original scatter is
retained, exact operands/metadata/current stream are guarded before and after
it, and the canonical state is copied only after the original operation for
the five ragged tests. The ordinary control additionally passes without an
observer. No further criterion or tolerance was adopted. Final integrated
verification passes 87 crate tests / 79 isolated primitives after the separate
source-faithful eager ConvRot W4A4 wrapper addition. Raw atomic/default ragged
full-path bitwise determinism is still not claimed.
