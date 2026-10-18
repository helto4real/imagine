# Stage 2 — first trained INT8 ConvRot comparison

2026-10-18. The first isolated Rust image has been generated and compared with
the frozen local ComfyUI Python implementation. **Full Stage 2 remains in
progress.** BF16, Turbo, editing, LoRA and the general memory/algorithm acceptance
matrix are still required by the design.

## Authorization and decisions

**S2-09 — first trained comparison authorized.** Tomas resumed after the earlier
readiness stop: “Ja gör det, spännande, jämför tider och kvalitet mot comfyui
pythonversionen”. Runs use the already selected model root and synthetic prompt
in the [explicit smoke profile](../inference/profiles/qwen21-first-int8.json).
No ordinary Dreamchat/ComfyUI runtime or existing media is an input. Nothing is
connected to production, and no service or unrelated process is stopped.

**S2-10 — reproduce ComfyUI application startup as well as raw Kitchen.** The
first comparison found a real parity bug. Raw Kitchen enables its Triton
registry backend, but frozen `comfy/quant_ops.py` disables it unless the caller
explicitly opts in. The initial Rust driver had not applied that application
startup rule. CUDA split-half RoPE supports FP16/BF16, while this language path
uses FP32; consequently Rust selected Triton and ComfyUI selected eager.

An observational diagnostic established exact embeddings, frequency matrices,
first input norm, Q/K/V projections and Q/K norms. The first unequal values
appeared at split-half RoPE: Q maximum absolute difference `9.5367431640625e-7`,
K `1.9073486328125e-6`. The difference propagated through the text context and
sampling into the image. The original failed comparison is retained rather
than counted as successful evidence.

The correction explicitly disables Triton in this standalone profile's startup,
as the original application does. It does **not** disable general Triton support
in the Rust runtime, modify kernels, alter autotuning, or relax exactness.
The profile now records this source-supported startup selection explicitly.
This was source compatibility work and required no departure approval.

The language regression oracle now includes both raw Kitchen and ComfyUI's
no-opt-in startup policy: **53 complete conditioning calls and 30 core forwards**
over ten groups (83 total, eight more than the previous 75). Existing source
norm/quantization/attention arithmetic is retained.

## Comparison contract and measurement

The complete original Python loaders, text wrapper, model wrapper, KSampler,
Euler implementation and VAE wrapper execute headlessly, without a graph,
server or custom-node startup. A read audit blocks non-frozen ComfyUI files,
unselected model assets, unrelated home data and Python network connections.
Every imported original ComfyUI source file is checked against the Stage 0
manifest. The original AIMDO safetensors reader loads the same three assets as
Rust. All directories and caches belong to this isolated development root.

Both sides use 512×512 RGBA, batch 1, seed 17, four Euler/simple steps, CFG 1,
source FLUX shift 0.69, BF16 compute, PyTorch attention, Kitchen Triton disabled,
default prefix-cache dtype/auto budget, static patchers, disabled Comfy compiler
and async offload, full component residency, and unload between components.
The prompt is “A blue butterfly resting on a red flower, soft daylight.”
The models and their hashes are recorded in the earlier
[load-only checkpoint](stage-2-qwen-image21-first-image-checkpoint.json).

Three matched pairs alternate Python→Rust, Rust→Python, Python→Rust. Each image
uses a fresh process and full component loading. OS model-file caches are already
warm; these are **not cold-disk or resident steady-state measurements**. No GPU
clocks, autotuning, dispatch thresholds or native settings are overridden.
Rust uses the current debug development build and the same optimized LibTorch
and Kitchen libraries; Python uses the locked interpreter and packages.

Phase timing uses a monotonic wall clock with CUDA synchronization at phase
boundaries and at the original per-step callback boundary. Load times include
mapping, preparation and full GPU residency. Sampling includes noise/schedule,
the solver, final latent normalization and transfer to CPU. Pipeline wall time
runs from tokenization/component loading to processed pixels and component
release, including intermediate exports and housekeeping, excluding launcher,
compilation, source/ELF preflight, interpreter imports and final PNG/raw-image
serialization. It must not be presented as application request latency.

## Numerical and visual quality

After the startup correction, the first complete comparison has **zero differing
bytes** in all three exported Float32 boundaries: text context `[1,20,4096]`,
post-normalization latent `[1,64,32,32]` and pixels `[1,512,512,4]`. All values
are finite; maximum absolute error, MAE and RMSE are zero. Decoded PNG RGBA8
pixels also match exactly. PNG container/compression byte identity is outside
the inference contract.

The generated image contains the requested blue butterfly and red flower. It
has soft detail, grain and imperfect wing/flower structure at four steps. Since
the final pixels are identical, this comparison shows **no quality loss relative
to Python on this exact case**. It does not establish production quality, prompt
coverage, higher-step quality, or parity of other variants/settings.

All **three matched pairs passed**, and each boundary has one common hash
across **all six images**, so the outputs also reproduce across process
restarts. The retained comparison checks all 4,784,128 Float32 bytes per image
across these three boundaries, plus the decoded PNG pixels.

![Rust synthetic smoke-test image](stage-2-qwen-image21-first-comparison-evidence/rust-image.png)

[The Python image](stage-2-qwen-image21-first-comparison-evidence/reference/image.png)
has identical pixels. [All metrics](stage-2-qwen-image21-first-comparison-evidence/summary.json)
and per-pair reports are retained beside the images.

## Measured times

Seconds; median and observed minimum–maximum from three fresh processes per
backend. Do not sum medians to construct a pipeline time: the pipeline is timed
independently and also includes exports, unload and housekeeping.

| Phase | Rust median | Python median | Rust range | Python range |
| --- | ---: | ---: | ---: | ---: |
| Text load and residency | 1.175 | 1.537 | 1.149–1.247 | 1.312–1.591 |
| Text encoding | 0.326 | 0.510 | 0.315–0.362 | 0.509–0.521 |
| DiT load and residency | 0.928 | 0.957 | 0.895–1.149 | 0.915–0.990 |
| Four-step sampling | 0.737 | 0.485 | 0.730–0.762 | 0.471–0.491 |
| VAE load and residency | 0.674 | 0.201 | 0.424–1.287 | 0.187–0.509 |
| VAE decode and pixel processing | 0.177 | 0.158 | 0.175–0.212 | 0.147–0.160 |
| Measured pipeline | **4.902** | **4.545** | **4.244–5.248** | **4.143–4.694** |

The measured pipeline median is about **7.9% longer in Rust** in this small
sample. Its text encoding is faster, while sampling and VAE decoding are slower.
VAE loading varies substantially and the pipeline ranges overlap. These runs
are a first controlled comparison, not evidence of a general performance
ranking, production request latency or release-build throughput. A cause for
the differences has not been established by kernel profiling. No optimization
was enabled in response to the measurements.

Both sides use 23 CPU threads, Torch `2.13.0+cu130`, the native caching allocator
and reduced math-SDPA precision enabled by original ComfyUI startup. Native
kernel objects and the source autotuning policy are unchanged.

## Reproduction and retained tests

The explicitly invoked
[benchmark runner](../inference/tools/benchmark_qwen_first_image.sh) creates three
matched pairs. It refuses an existing report directory. The Rust launcher accepts
a fresh third-argument run name and never overwrites an image directory.

```sh
export IMAGINE_PARITY_ROOT=/tmp/imagine-rust-comfy-parity-20261017
export IMAGINE_REFERENCE_SITE=/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages
bash inference/tools/benchmark_qwen_first_image.sh \
  /home/thhel/git/ComfyUI /mnt/media/gemensam/AI/ComfyUI/models NEW_RUN_NAME
```

[The comparator](../inference/tools/compare_qwen_first_image.py) saves complete
boundary hashes, finite-value checks, unequal counts, error metrics and timings,
and fails by default on any numerical mismatch. `--allow-mismatch` is explicitly
diagnostic. Model files and raw activations stay in the isolated runtime root;
compact metrics, source hashes and the two synthetic example images are retained
with this report for future upgrades.

The `reference/` evidence directory also retains the three exact Float32 binary
outputs and original Python result metadata. It can be passed directly as the
comparator's `--python` directory when checking a fresh matching Rust run.

The separate language diagnostic uses observations after original operations;
normal execution has no tensor observer. A diagnostic Python attempt retained
the text module through bound observation methods and exhausted VRAM when it
continued to DiT loading. The diagnostic is now text-only and exits before
another component loads. It contributes no timing/quality success evidence,
and no memory policy, process or service was changed to accommodate it. An
earlier import-only audit failure was fixed by allowing the pinned Python
standard-library code. Neither failed attempt is counted as a completed image.
The text-only diagnostic was then rerun successfully with orderly unload and
cleanup before exit; no OOM or interpreter-shutdown errors remained.

## Validation and remaining work

- Formatting, shell syntax and all-target Clippy with warnings denied passed.
- The extended language test passed normally and separately without Python:
  53 complete conditioning calls plus 30 core forwards.
- The resident attention/cache/cancellation replay passed after the observer
  addition. Full trained Rust runs independently passed Python-free isolation.
- All three strict trained comparisons passed; mismatch rejection is checked
  against the preserved initial failing comparison.
- Imported ComfyUI source hashes, implementation/profile inputs, Stage 0 locks
  and original CUDA object hashes are retained in the final checkpoint.

The earlier complete 15-test Stage 2 focused suite and 155/146 full Stage 1
checks remain historical evidence for their documented inputs. This change
reruns the affected language/resident tests, Clippy and the trained comparisons;
it does not claim a new full unfiltered suite run.

The [checkpoint](stage-2-qwen-image21-first-comparison-checkpoint.json) records
the exact evidence inventory and checks. The initial measured-input snapshot
was taken before the benchmark. Only `inference/README.md` documentation changed
while those runs executed; executable source, profile, fixtures, libraries and
locks did not change. The final verified snapshot includes that documentation
update and all 273 original ComfyUI files imported by the full reference.
After measurement, only cleanup in the diagnostic-only `--trace` exit branch
changed. The measured normal branch is unchanged, and the exact measured
reference script is retained and hash-checked against the measurement snapshot.
The final snapshot verifies 362 implementation/fixture/resource inputs and
275 original source inputs, including those 273 ComfyUI imports.

No new departure from the design or frozen implementation was requested. All
sandbox escalations were approved; no approval was rejected. No production
integration, cloud service, service interruption, Computer Use, commit or push
occurred. The existing preservation commit remains `059c875`; Stage 2 work is
still uncommitted. Completing this first comparison does not complete Stage 2.
