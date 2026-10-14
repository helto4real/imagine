# CPU thread comparison — 2026-10-13

**Use explicit `--threads 11` for the next ConvRot comparison.** It had the
lowest mean start-to-save time and the smallest sampling-time spread among
the four tested settings. This preserves the previous effective thread count;
it is not a new speedup over the previous configuration.

## Results

Means of three measured runs per setting, in seconds; warmup excluded:

| CPU threads | Conditioning | Sampling | Sampling min–max | VAE | Start to saved PNG | Process wall |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 4 | 1.98 | 58.47 | 57.69–59.02 | 2.14 | 63.52 | 63.64 |
| 8 | 1.95 | 59.89 | 59.20–60.51 | 2.10 | 64.88 | 65.03 |
| **11** | **1.94** | **57.59** | **57.49–57.72** | 2.19 | **62.65** | **62.78** |
| 16 | 1.99 | 59.18 | 58.70–59.76 | **1.99** | 64.07 | 64.20 |

Mean loader durations, already included in the phase times above:

| CPU threads | Text-encoder tensors | Diffusion tensors | VAE tensors |
| --- | ---: | ---: | ---: |
| 4 | 0.40 | 0.80 | 0.33 |
| 8 | 0.41 | 0.87 | 0.20 |
| 11 | 0.40 | 0.81 | 0.20 |
| 16 | 0.41 | 0.88 | 0.20 |

Sampling times in block order:

| CPU threads | Block 1 | Block 2 | Block 3 |
| --- | ---: | ---: | ---: |
| 4 | 57.69 | 59.02 | 58.69 |
| 8 | 59.97 | 59.20 | 60.51 |
| 11 | 57.72 | 57.49 | 57.57 |
| 16 | 58.70 | 59.08 | 59.76 |

The 4-thread setting essentially tied 11 in the first block, but not in the
other two. The 8- and 16-thread settings were slower than 11 in every block.
This is a practical local reference choice, not proof that 11 is universally
optimal. Three samples per setting on a shared desktop do not isolate why
the differences occur. We did not test CPU affinity, every possible thread
count, cold disk cache, or a persistent server.

There is no observed gain from increasing the previous 11-thread setting.
The next experiment should keep `--threads 11` fixed and test the ConvRot
CUDA block size independently, with fresh controls in that experiment.

## Validation

All **13** full generations, including warmup, completed and produced identical
1600x896 RGB pixels. All logged 80 diffusion forwards. Post-run checks verified
the same executable hash, prompt bytes, non-thread generation parameters,
fusion environment, model placement, and compute-buffer sizes (independent of
log ordering). GPU monitoring reported no errors, and GPU memory returned to
1663 MiB of desktop usage after completion.

Only the experiment runner gained a thread argument and a comparison driver;
the engine and GPU kernels were not edited or rebuilt. Python syntax checks
passed. GPU access required approved sandbox escalation. No commit or push.

## Scope and method

Compare explicit `--threads 4`, `8`, `11`, and `16` using the preserved
projection-fusion executable. The previous automatic setting resolved to 11.
No engine source, CUDA launch configuration, affinity, clock limits, or other
running applications are changed for this experiment.

The machine reports Intel Core Ultra 9 285K, with 23 online CPUs and CPU 5
offline. Linux schedules the workers across the allowed CPUs; this experiment
does not distinguish performance and efficiency cores or pin threads.

One complete 11-thread generation warms filesystem cache and is excluded
from timing statistics. Three measured blocks use these orders:

1. 11, 4, 8, 16
2. 16, 8, 4, 11
3. 8, 11, 16, 4

Each image starts a fresh CLI process. This is warm-filesystem-cache loading,
not a persistent server keeping models and conditioning loaded. There is no
CUPTI instrumentation in these runs. One-second `nvidia-smi` monitoring is the
same for every run; the desktop GPU remains shared and clocks are not locked.

All runs retain the reference prompt and negative prompt, 1600x896, seed 42,
CPU RNG, 40 ER-SDE/Beta steps, CFG 2.5, F32 prefix cache, SageAttention2++,
512 MiB auto-fit margin, and both experimental INT8 fusion flags enabled.

## What the thread setting controls

- Model-loader workers, including host-side reading and conversion:
  `src/model_loader.cpp:1064` and `:1125`.
- The engine's CPU tensor executor, used by host tensor operations:
  `src/pipeline/diffusion_engine.cpp:950` and `src/core/parallel.cpp:134`.
  Small operations can use fewer workers because of the grain-size threshold.
- GGML CPU backends used within graph execution:
  `src/core/ggml_runner.cpp:762`.

It does not change CUDA block sizes or the number of threads inside the
ConvRot, GEMM, or Sage kernels. CPU RNG is implemented separately in
`src/core/rng_mt19937.hpp`, without this worker-count setting.

## Timing definitions

- **Start to saved image:** parent monotonic time immediately before launching
  the CLI through receipt of the successful PNG-save log. This includes loading,
  inference, and PNG writing. Log receipt is an approximate boundary; this is
  not a measurement of `fsync` or durable physical-media flush.
- **Process wall:** the same start through process exit.
- **Conditioning, sampling, VAE:** durations printed by the engine. They include
  relevant lazy loading/staging and setup, rather than pure GPU compute alone.
- **Tensor loading:** the three loader durations for text encoder, diffusion,
  and VAE. These are already inside the corresponding phase timings; they must
  not be added a second time.

## Reproduction and evidence

```sh
python experiments/qwen-image21-sage/profiling/compare-cpu-threads.py
```

The driver verifies the preserved executable SHA-256 before starting:
`33aa286e9ff42e4b6e61af134bed5a1fa64b1d7d98ce32d27ea486ff7d225805`.
It checks every image's decoded RGB bytes against reference SHA-256
`253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2`.

Evidence directory:
`../runs/20261013T093457Z-cpu-threads-e3mudidm/`.
It contains the plan, source snapshots, machine information, per-run records,
and the final statistical summary. Individual replay directories retain exact
commands, prompts, binary hashes, logs with parent timestamps, GPU monitoring,
images, and pixel validation.
