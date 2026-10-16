# Qwen diffusion activity profiling

**Current baseline:** [1600×896 v1](../BASELINE.md). Before proposing or repeating
experiments, read [the completed test decisions](../TEST-DECISIONS.md). The
commands below are historical diagnostic replays, not current baseline launchers.

This diagnostic replays the saved F32-prefix reference with unchanged model files,
prompts, 1600 x 896 resolution, seed 42, 40 ER-SDE/Beta steps, CFG 2.5,
SageAttention2++, and a 512 MiB auto-fit margin. It does not modify the engine.

**Current auto-fit validation:** [CUDA scratch accounting fix](AUTO-SCRATCH.md)
uses `--default-vram --threads 11` with the accepted optimization flags and no
explicit `--max-vram`. The saved replay default below remains historical.

`activity.cpp` is an optional process-local `LD_PRELOAD` library built against the
installed CUDA 13.4 CUPTI headers. It records **concurrent** kernel execution,
memcpy, and memset activities, without synchronizing each launch. Timestamp
normalization uses `CLOCK_MONOTONIC`, matching Python's log-receipt timestamps.
It flushes records on process exit and records the dropped-record count.

From the project root, build and run:

```bash
mkdir -p experiments/qwen-image21-sage/runs/activity-profiler
g++ -std=c++17 -O2 -Wall -Wextra -shared -fPIC -pthread \
  experiments/qwen-image21-sage/profiling/activity.cpp \
  -I/opt/cuda/targets/x86_64-linux/include \
  -L/opt/cuda/targets/x86_64-linux/lib \
  -Wl,-rpath,/opt/cuda/targets/x86_64-linux/lib -lcupti -ldl \
  -o experiments/qwen-image21-sage/runs/activity-profiler/activity.so
python3 experiments/qwen-image21-sage/profiling/run.py \
  --trace-library experiments/qwen-image21-sage/runs/activity-profiler/activity.so
```

Run `run.py` without `--trace-library` for the control. Each invocation performs a
full GPU generation and saves a fresh ignored run directory. It requires the
existing debug reference directory, the model files, and available GPU memory.
No model process is stopped or unloaded by the script.

`--binary PATH` selects a preserved reference executable. Each new run records
its executable SHA-256. With tracing enabled, `--copy-stacks` additionally
captures one host stack for each distinct device-to-device copy payload size
above 80 MiB, together with the CUDA API's 2D pitches and dimensions. This
diagnostic mode adds runtime callback overhead; use an uninstrumented pair
for generation timing. Symbolize executable-relative frame offsets with
`addr2line -f -C -e EXECUTABLE OFFSET...` against that exact executable.

Pass the printed profile directory to `analyze.py`. The analyzer requires all 80
diffusion forwards, zero dropped records, valid timestamps, no GPU activity
crossing the selected boundaries, 2,560 Sage core launches, and matching counts
of 15,360 INT8 GEMM/dequantization launches. It produces `profile-summary.json`,
`kernels.csv`, and `kernel-shapes.csv`.

The selected window starts at the first `qwen_image_2_1 executing segment` log
and ends at `sampling completed`. It excludes the text encoder, lazy diffusion
weight loading, and VAE decode, but includes both prefix creation and all 40
sampling steps. Log boundaries are approximate parent pipe receipt times, not
engine-instrumented ranges. GPU durations themselves come from CUPTI. Summed
activity duration and the interval union are reported separately to reveal
overlap; percentages use summed GPU activity, including copies, as denominator.
Sage-specific quantizers are grouped with Sage preparation, not Linear quantization.
Generic format/layout kernels remain a separate category.

See [the measured results](RESULTS.md). This is timing attribution, not a
hardware-counter study, a ComfyUI profile, or a promise that any operation can
be eliminated without changing numerical results.

`check-swiglu.cpp` is a local CUDA correctness experiment comparing the old
copied-halves graph with packed `ggml_swiglu`. It checks every output bit and
independently checks the formula, gate order, and row stride at sampled positions.
The actual Qwen intermediate width (12,288) is tested with 1, 57, 230, and 5,600
rows. Build against this checkout's GGML libraries, then run only with the GPU
available; it uses synthetic tensors and does not load model weights.

## Matched Kitchen Linear benchmark

[Measured results and limitations](INT8-KITCHEN.md) cover the whole ConvRot,
INT8 GEMM, and output chain. Build the native harness from the project root:

```bash
mkdir -p experiments/qwen-image21-sage/runs/linear-benchmark
g++ -std=c++17 -O2 -Wall -Wextra \
  experiments/qwen-image21-sage/profiling/linear.cpp \
  -Istable-diffusion.cpp/ggml/include -Wl,--start-group \
  stable-diffusion.cpp/build-cuda/ggml/src/libggml.a \
  stable-diffusion.cpp/build-cuda/ggml/src/libggml-base.a \
  stable-diffusion.cpp/build-cuda/ggml/src/libggml-cpu.a \
  stable-diffusion.cpp/build-cuda/ggml/src/ggml-cuda/libggml-cuda.a \
  -Wl,--end-group -L/opt/cuda/targets/x86_64-linux/lib \
  -Wl,-rpath,/opt/cuda/targets/x86_64-linux/lib \
  -lcudart -lcublas -lcublasLt -lcuda -lnccl -lgomp -lpthread -ldl \
  -o experiments/qwen-image21-sage/runs/linear-benchmark/linear
/path/to/ComfyUI/.venv/bin/python \
  experiments/qwen-image21-sage/profiling/compare-linear.py
```

Build the CUPTI tracer above first. The driver runs variants sequentially and
uses the existing Kitchen environment, without installing dependencies. Each
case's full synthetic inputs are shared across variants and removed afterward;
hashes and sampled outputs remain. Add `--case down --timing-only --reverse-order`
to reproduce the ordering check. Summarize a printed result directory with
`analyze-linear.py RUN_DIRECTORY` using the same Python interpreter.

## Experimental fused gate/up

The optional native CUTLASS path combines INT8 GEMM with F32 output scaling.
It retains native ConvRot quantization, and applies only to bias-free
K=4096/N=24576 Linear operations with group size 256 on SM89. All other
operations retain the existing path. It has no Python or Kitchen runtime
dependency. Both build-time inclusion and runtime opt-in are required.

From the project root, using the locally archived CUTLASS 4.5.0 headers:

```bash
cmake -S stable-diffusion.cpp -B stable-diffusion.cpp/build-cuda \
  -DGGML_CUDA_INT8_GATE_UP_CUTLASS_PATH="$PWD/experiments/qwen-image21-sage/runs/cutlass-4.5.0/include"
cmake --build stable-diffusion.cpp/build-cuda --target sd-cli sd-server --parallel 2
```

Rebuild the `linear` harness above after rebuilding GGML. Then run
`compare-fused.py` with the existing ComfyUI Python interpreter for synthetic
A/B checks. It checks every F32 output and all quantized inputs/scales by
SHA-256, separately checks output finiteness, and verifies actual dispatch
using CUPTI. Temporary full tensors are removed; hashes and sampled trace
outputs remain. Cases cover 1, 57, 230, 5600, 5657, and 5830 rows plus an
unsupported projection shape that must retain the fallback.

For full generation:

```bash
GGML_CUDA_FUSED_INT8_GATE_UP=1 python3 experiments/qwen-image21-sage/profiling/run.py
GGML_CUDA_FUSED_INT8_GATE_UP=0 python3 experiments/qwen-image21-sage/profiling/run.py
GGML_CUDA_FUSED_INT8_GATE_UP=1 python3 experiments/qwen-image21-sage/profiling/run.py \
  --trace-library experiments/qwen-image21-sage/runs/activity-profiler/activity.so
python3 experiments/qwen-image21-sage/profiling/analyze.py RUN_DIRECTORY --fused-gate-up
```

The last command requires exactly 2,560 fused gate/up kernels and 12,800
remaining separate GEMM/dequantization pairs, alongside the original 2,560
Sage cores and zero dropped trace records. An unset runtime flag leaves the
fusion disabled. Passing an empty CUTLASS CMake path removes it from the build.
See [the experiment results](FUSED-INT8.md).

## Experimental fused 4096 projections

The same optional build now also supports bias-free K=N=4096 ConvRot
projections on SM89. Enable them independently with
`GGML_CUDA_FUSED_INT8_PROJECTION=1`. Sequences shorter than 512 rows retain
cuBLAS because the initial crossover tests measured regressions there.
The down projection (K=12288) retains its previous implementation.

After building and relinking the Linear harness above:

```bash
/path/to/ComfyUI/.venv/bin/python \
  experiments/qwen-image21-sage/profiling/compare-fused.py --projection
GGML_CUDA_FUSED_INT8_GATE_UP=1 GGML_CUDA_FUSED_INT8_PROJECTION=0 \
  python3 experiments/qwen-image21-sage/profiling/run.py
GGML_CUDA_FUSED_INT8_GATE_UP=1 GGML_CUDA_FUSED_INT8_PROJECTION=1 \
  python3 experiments/qwen-image21-sage/profiling/run.py
```

The synthetic suite verifies the 511/512-row boundary, full image and prefix
shapes, the unchanged gate/up path, and down fallback. Add `--rows 256 512
1024 2048` for a focused crossover rerun. Full-generation comparison holds
gate/up on in both variants; only the projection flag changes. Both flags
default to off. The historical gate/up build-option name is retained.

For a both-on trace, append the `--trace-library` option shown above and run
`analyze.py RUN_DIRECTORY --fused-gate-up --fused-projection`. It requires
exactly 2560 fused gate/up kernels, 10240 fused ordinary projection kernels,
and 2560 remaining separate GEMM/dequantization pairs, as well as the original
attention and trace-completeness checks. The fixed tile's column grid
distinguishes ordinary projections from gate/up in this build.

See [the projection experiment](PROJECTION-FUSION.md).

## CPU thread reference

`run.py --threads N` sets the CLI computation/loading thread count for one
reference replay. It does not change CUDA kernel block sizes or CPU affinity.

```sh
python experiments/qwen-image21-sage/profiling/compare-cpu-threads.py
```

The comparison uses the preserved projection-fusion binary with both INT8
fusion flags enabled. It runs one excluded warmup followed by three blocks
containing 4, 8, 11, and 16 threads, with varied order. Every image is checked
against the decoded RGB reference hash. Commands, logs, GPU monitoring,
per-phase timings, and the parent-observed time to the successful image-save
log are retained under a new `runs/*-cpu-threads-*` directory.
Model loading is lazy and overlaps the logged phase totals; do not add loader
times to conditioning, sampling, or VAE times. This measures fresh CLI processes
with warm filesystem cache, not requests to a persistent model server.

The [measured CPU-thread comparison](CPU-THREADS.md) selects explicit
`--threads 11` for subsequent reference replays. This matches the previous
automatic setting; the runner and engine defaults remain unchanged.

## ConvRot CUDA block-size comparison

`GGML_CUDA_CONVROT_THREADS=256` or `512` selects that block size for native
K=4096 ConvRot activation quantization. Unset, `1024`, and unrecognized values
use 1024 threads. Other widths and the shared-memory fallback are unchanged.
This is separate from the CPU `--threads` setting.

Build the engine and rebuild `linear.cpp` with the command above, writing the
harness to `runs/convrot-threads-build/linear`. The experiment drivers use that
directory's preserved before/after binaries and verify the full-image binary
hash; they intentionally do not silently adopt a later build.

```sh
python experiments/qwen-image21-sage/profiling/compare-convrot-threads.py
python experiments/qwen-image21-sage/profiling/compare-convrot-images.py
```

See [the ConvRot block-size report](CONVROT-THREADS.md). The selected experimental
reference uses 512 CUDA threads for K=4096, with 11 CPU threads and both existing
INT8 GEMM fusions enabled:

```sh
GGML_CUDA_FUSED_INT8_GATE_UP=1 GGML_CUDA_FUSED_INT8_PROJECTION=1 \
GGML_CUDA_CONVROT_THREADS=512 \
python experiments/qwen-image21-sage/profiling/run.py --threads 11 \
  --binary experiments/qwen-image21-sage/runs/convrot-threads-build/sd-cli
```

## ConvRot absmax fusion (not adopted)

The [absmax experiment](CONVROT-AMAX.md) removed the separate shared-memory
maximum scan at K=4096/512 threads but demonstrated no useful speedup.
The active engine was restored to the block-size reference. The isolated
patch and frozen binary remain in `runs/convrot-amax-build/`; its experimental
`GGML_CUDA_CONVROT_FUSED_AMAX` flag is not supported by the active engine.

```sh
python experiments/qwen-image21-sage/profiling/compare-convrot-amax.py
```

## Packed SwiGLU / ConvRot fusion

The [SwiGLU/ConvRot report](SWIGLU-CONVROT.md) documents the retained opt-in
`GGML_CUDA_FUSED_SWIGLU_CONVROT=1`, its input-lifetime dependency and safe
fallbacks, numerical/sanitizer checks, full-model profile, and three image
comparison pairs. The final executable is preserved in
`runs/swiglu-convrot-build/sd-cli`. CPU threads stay at 11 and K=4096 ConvRot
stays at 512 CUDA threads; the fused K=12288 kernel uses 1024 threads.

```sh
python experiments/qwen-image21-sage/profiling/compare-swiglu-images.py
```

The image driver pins the preserved executable hash. The report also provides
a single-image replay command. Defaults are unchanged; set the flag before
starting the CLI/server process.

## Gate/up CUTLASS configuration sweep (no new variant adopted)

The [gate/up tuning report](GATE-UP-TUNING.md) compares six tile/pipeline
configurations and repeats the closest candidates in reverse order. No useful
repeatable gain was established. The active helper retains the original
128×256×64, three-stage configuration; the experimental
`GGML_CUDA_GATE_UP_CONFIG` flag is only supported by the frozen executable in
`runs/gate-up-tuning-build/`. The report contains reproduction commands.


## Down-projection fusion (retained as opt-in)

The [resumed full-image comparison](DOWN-AUTO.md), with corrected CUDA scratch
accounting and no manual margin, completes three A/B pairs. Mean start-to-PNG
falls from 60.889 to 60.347 s (0.9%); all three pairs favor fusion. Seven
unprofiled images and one separately profiled image match the reference exactly.
The trace confirms replacement of the last 2560 separate INT8 GEMM/output pairs.

The active candidate supports `GGML_CUDA_FUSED_INT8_DOWN=1`, disabled by default.
Use `runs/down-auto-build/compare-down-images.py --default-vram` for the frozen
new comparison. The [original report](DOWN-PROJECTION.md) and original driver
retain the historical pre-fix candidate, microbenchmarks, and failed image runs.

## VAE memory accounting diagnosis

The [VAE memory report](VAE-MEMORY.md) identifies the actual out-of-graph FP16
output temporary and scratch-pool growth behind the intermittent OOM. The
accepted engine reproduces the failure; a separate 1024 MiB margin control
completes a reference-identical image after evicting one additional block.
No memory policy was changed. The active engine is restored; the optional
metadata logger exists only in `runs/vae-memory-build/sd-cli`.

`run.py --vram-margin-mib 1024` selects the separate headroom diagnostic and
cannot be combined with `--default-vram`. The report gives full reproduction
commands and `analyze-vae-memory.py` usage.
