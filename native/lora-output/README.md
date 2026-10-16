# Opt-in LoRA output fusion

Separate candidate on the frozen Qwen v2 source. The accepted v1/v2 binaries,
reference-cache engine and application engine selection remain unchanged.

Measured CFG 1 warm LoRA latency: **44.611 → 39.892 s (−10.579%)** with identical
synthetic output bytes. See the [full result](../../experiments/qwen-image21-sage/profiling/LORA-OUTPUT.md).

`GGML_CUDA_FUSED_LORA_OUTPUT=1` enables the CUDA fusion; unset or `0` preserves
the existing path. `GGML_CUDA_DISABLE_FUSION=1` also disables it. The initial
scope is Ada/SM89, contiguous unbatched FP16 LoRA up-projections using the
existing FP16-output cuBLAS route, rank at most 256, width/sequence at least 512.
Concurrent streams, foreign/split device buffers, unsupported shapes, precision choices, shared intermediates and unsafe aliases
fall back to the ordinary graph. No new public API or UI setting is introduced.

The matcher requires a LoRA-marked `MUL_MAT → SCALE [→ SCALE] → ADD` chain.
It calls the same cuBLAS implementation with the same FP16 output buffer,
compute type and GEMM parameters. Only the final FP16→F32 conversion changes:
one kernel converts, applies each original SCALE in order and adds the base
output. Each SCALE uses a separately rounded F32 fused multiply/add, matching
`scale_f32` under the existing CUDA fast-math build; the final addition has a
separate F32 rounding. The attention rescale is not combined algebraically with
the LoRA multiplier. No FP16 GEMM rounding is removed, and LoRA additions remain
ordered when several adapters apply. Existing graph scratch reservations remain.

## Restore and build

```sh
python3 native/lora-output/build.py
python3 native/lora-output/build.py --verify-only
```

`SD_BUILD_JOBS` controls build concurrency. The script uses local pinned Git
objects and the two complete published base patches, then applies
`lora-output.patch`. It verifies clean restoration, unchanged frozen hashes and
the two modified source hashes in `source-lock.json`. Source/build/binary output
lives under ignored `.imagine/lora-output/`. The existing development checkout
is not edited. This overlay is separate to preserve existing accepted patch
chains and source locks.

## Checks and matched benchmark

```sh
python3 experiments/qwen-image21-sage/profiling/test-lora-output.py --run
```

This checks synthetic graph results byte-for-byte with the flag off/on, including
negative/zero strengths, subnormal values, attention rescaling, repeated adapter
sums, out-of-place output, retained intermediate outputs and fallback cases.
The large fixture matches the 24,576 × 5,600 gate/up output. It reads synthetic
fixtures only; it does not inspect generated images or private model runtime data.

Use `compare-lora-cfg1.py` with its existing model-root and LoRA arguments plus:

```sh
--candidate-manifest .imagine/lora-output/manifest.json --lora-output 0 --run
--candidate-manifest .imagine/lora-output/manifest.json --lora-output 1 --run
```

Each option set runs the original eight-request sequence. All 29 baseline
optimization flags, source-defined prompt, seed, CFG 1, geometry and model paths
are held constant. An optional `--trace-library` records the specifically
user-authorized kernel/copy measurements, with server logs and image responses
discarded. Python is build/test tooling only; inference remains native C++/CUDA.
