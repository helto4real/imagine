# Experimental Qwen INT8 / ConvRot LoRA merge

Validated experimental candidate on the frozen v2 + LoRA-output overlay.
`SD_QWEN21_MERGE_LORA_INT8=1` requests prepared INT8 weights. Default is off.
No application engine selection or frozen executable is changed.

The verified precision contract is the tested ComfyUI ordinary ModelPatcher path:
FP16 unrotated base weight, FP32 LoRA matrix product and strength scaling,
FP16 delta and addition, then FP16-aware stochastic INT8/ConvRot requantization
with recalculated row scales and the CRC32 of the original Comfy weight key.
Actual-path assertions and CUDA measurements confirm this contract. Different
GEMM shapes may still round differently; this is not a pixel-identity claim.

Patches are ordered and quantized once after all selected LoRAs. Gate/up halves
retain their offsets. Current configuration invalidation restores source weights;
no cumulative patching across strength changes. Preparation follows the verified
Comfy strategy: a whole FP16 weight matrix, a full FP32 product per ordered patch,
and one row-fused ConvRot/requantization pass. GPU-resident INT8 weights stay on
the GPU. Host weights use one bulk transfer in each direction. There is no
64-row loop or tile fallback, and no full floating-point model copy is kept.

The existing memory manager reserves the exact shape-dependent CUDA scratch
plus 32 MiB cuBLAS headroom, in addition to its existing safety margin. The
largest observed scratch allocation for this workload is 577.844 MiB; its
reservation is 609.844 MiB. The buffer is released after each whole matrix.
There is no Python inference dependency.

Silent tensor loads now wait directly for their worker threads to finish.
The progress polling loop runs only when progress display is enabled. Worker
joining, failure handling and before/after source validation are unchanged.
This shared loader change applies to all silent loads in the experimental binary,
independently of the merge flag; the accepted production binary is unchanged.

Initial scope: Qwen Image 2.1, a single CUDA GPU, ordinary rank <=256 linear
LoRAs on INT8 ConvRot matrices, group 256, input width <=24576. Unsupported
adapters fail the opted-in request rather than silently dropping their effects.
Other models and the default path keep the existing implementation.

The complete native server and both synthetic test helpers compile. The changed
C++ integration also passes syntax checks, and clean replay verifies
base/parent/new patches plus frozen source/binary hashes. An incremental rebuild
confirmed that the final source lock matches the built candidate.
Eight synthetic group-256 precision cases pass exact INT8 and scale comparison
against installed Comfy/Kitchen. Fifteen ModelManager lifecycle cases pass across
CPU, GPU and disk placement, including independent weight/scale eviction, fused
gate/up mapping (separate halves and full fused matrices), strength changes and removal. Group 64 differed numerically in
an exploratory synthetic case (484 INT8 values) and is explicitly rejected;
the actual model's checked Comfy path uses group 256.

The real Comfy run passes the FP16/FP32/requantization assertions. The current
loader-wait result is in [LOADER-WAIT-RESULTS.md](LOADER-WAIT-RESULTS.md), with
numeric evidence in `loader-wait-results.json`. The earlier whole-matrix and
tiled results remain in [RESULTS.md](RESULTS.md) and `comparison-results.json`.
All inference fusions remain enabled when using prepared INT8 weights.

The initial HTTP 500 was a shape check that assumed gate/up LoRAs always address
two halves. A synthetic full-matrix LoRA reproduced it; both layouts now pass.
Read [LOADER-WAIT-RESULTS.md](LOADER-WAIT-RESULTS.md) for the latest measurements,
and [RESULTS.md](RESULTS.md) for precision, memory and scope limitations.
`comparison-results.json` and `synthetic-results.json` contain numeric evidence.
The candidate remains opt-in; production engine selection is unchanged.

## Reproduce after freeing the GPU

```sh
python3 native/lora-merge/build.py
python3 native/lora-merge/build.py --verify-only
```

Run `profiling/test-lora-int8-merge.py` with the installed Comfy Python and
`--comfy-root` to build the independent synthetic reference check. `--run`
explicitly starts GPU validation. It compares actual installed Comfy LoRA
application and Kitchen quantization against the native helper. Dyadic input
matrices isolate the staged rounding/quantizer contract from different cuBLAS
tiling; this does not establish end-to-end pixel identity.

`profiling/test-lora-int8-lifecycle.py` builds a native ModelManager test using
synthetic safetensors. `--run` tests reuse, eviction, changing strength, restoring
the initial strength, and removing LoRA without accumulating patches.
Both scripts are under `experiments/qwen-image21-sage/`.

For the matched eight-request native test, use the existing `compare-lora-cfg1.py`
model and LoRA arguments plus:

```sh
--candidate-manifest .imagine/lora-merge/manifest.json --lora-output 1 --lora-merge 0 --run
--candidate-manifest .imagine/lora-merge/manifest.json --lora-output 1 --lora-merge 1 --run
```

Comfy's matching harness now accepts `--trace-library` for the authorized
kernel/copy profiler and `--verify-merge` to assert the actual precision
contract. Model runtime logs and generated images remain discarded.
For a separately authorized allocation-size measurement, the native harness
accepts `--trace-memory` with `--trace-library`. The profiler emits only allocation
sizes, kinds, operation codes and timestamps into `memory.tsv`; it does not emit
addresses, variable names or tensor data. `profiling/summarize-lora-memory.py`
summarizes these counters. It deliberately does not infer physical peak VRAM
from allocation/release balances, because the backend also uses virtual memory.

Kernel math derived from Comfy Kitchen retains its Apache-2.0 copyright,
license and notice inside the complete patch. Installed ComfyUI is neither
modified nor required by the native inference executable.
