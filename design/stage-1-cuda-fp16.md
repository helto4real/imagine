# Stage 1 CUDA FP16 linear continuation

Date: 2026-10-17. Status: **400-case numerical parity passed; the snapshot at this continuation passes
53 Rust tests and 45 Python-free primitive functions; Stage 1 remains incomplete**.

`inference/src/cuda_linear.rs` ports the frozen CUDA `fp16_linear` wrapper using
original compiled CUTLASS entrypoints. It preserves input reshape/contiguity,
16-byte pointer alignment gates, weight conversion, aligned bias/scale vectors,
empty-placeholder caching, fused residual acceptance, native decline followed
by LibTorch `linear`, residual addcmul order and final batch-shape restoration.
No kernel, native binary, numerical tolerance, RNG or autotuning policy changes.
No unimplemented backend is registered as executable.

The synthetic oracle covers **400** cases: **364** successful byte-exact results
and **36** exact missing-residual-scale ValueErrors. Inputs cover CPU/CUDA,
float32/float16/bfloat16, rank 2/3, plain/bias/residual/both, misaligned input,
weight and vectors, noncontiguous weight, double residual, empty rows and zero K.
The floating dtypes/CPU cases also exercise the directly callable helper's
fallback. They do not broaden the unchanged CUDA registration's FP16-only policy.

The original native heuristic declines shapes with fewer than 96 ordinary tiles.
Small fixtures alone therefore cannot demonstrate nonempty CUTLASS execution.
Four additional `[6144,32]` by `[512,32]` FP16 cases reach that unchanged boundary
for plain, bias, residual and both. All four reference results differ from eager
bytes, so replacing native execution with fallback would fail strict comparison.
The oracle records 12 accepted native calls: four nonempty GEMMs and eight empty
returns. The original Python callable is wrapped only to record each call name
and acceptance flag, then restored in `finally`; kernel bodies/arguments stay
unchanged. Large payloads use lossless hex, still compared byte for byte.

Reproduce reference capture in the authorized own environment using `-S -B` and
explicit frozen site path with `tools/reference_cuda_linear.py --site … --output …`.
The canonical primitive capture also includes the new category. Previous 3,448
ordinary fixtures are preserved in a separate own snapshot and remain unchanged
as structured records. Current ordinary fixture count is **3,848**; the **48**
controlled empty ConvRot cases remain separate.

The pending [CUDA INT8 M=1 zero-K output decision](decisions/D-05-INT8-CUDA-M1-empty-output.md)
was found by source inspection while selecting the next linear path. No probe
reads its unwritten output, no new criterion is adopted and no dependent CUDA
INT8 high-level port is started before that approval.

No models/checkpoints, ordinary runtime data, process lists, service changes,
browser/manual acceptance or product integration were involved. GPU tests and
Python-free isolation use authorized sandbox escalation. Detailed source/input
hashes and final aggregate results are in [stage-1-evidence.json](stage-1-evidence.json).

The aggregate launcher also passes formatting and Clippy (`--tests -- -D warnings`).
All previous 3,448 ordinary fixtures and 48 controlled empty cases were rerun,
along with all 400 new FP16 cases. No automatic approval rejection occurred.

The later [CUDA INT8 M=1 continuation](stage-1-cuda-int8-m1.md) is now the latest
full aggregate snapshot; this document retains the earlier FP16-specific counts.
