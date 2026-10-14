# Headless AIO / ComfyUI / Kitchen reference — 2026-10-13

The installed AIO Qwen adapter runs successfully without a ComfyUI HTTP server.
Four complete images reproduce identical pixels within this engine. The largest
remaining difference from our optimized sd.cpp path is normalization,
elementwise work and tensor layout/copies, rather than INT8 GEMM itself.

## Inputs and execution

Run: `../runs/20261013T135141Z-comfy-3_ly1kh7/`.

- RTX 4070 Ti SUPER, installed ComfyUI Python environment, Torch 2.13.0+cu130,
  Kitchen 0.2.37. ComfyUI revision `08ff3c11b3a85eb07c2c06f21cb378ac2e086840`;
  AIO revision `52ecbf87dda674f51e1e0a798f8f56d0d37d303d`.
- Exact selected Qwen Image 2.1 INT8 ConvRot diffusion and Qwen3-VL INT8 ConvRot
  encoder files, plus BF16 VAE, under the supplied model root. The logical
  `clip/` encoder name resolves to the physical `text_encoders/` file.
- 1600×896, aspect ratio 16:9, max side 1600, multiple 32; 40 ER-SDE/Beta steps,
  CFG 2.5, fixed seed 42, batch 1. Original test prompt files from
  `20261013T074331Z-debug-7nor7cre` are read directly, with hashes in `config.json`.
- Kitchen INT8 attention; memory policy auto; prefix cache device auto and dtype
  default; Comfy sampling schedule; torch compile off. Zero negative
  conditioning false. No LoRA, reference image, second pass, PID capture or
  color/brightness/contrast processing. Screenshot settings are recorded in
  `config.json` and effective settings in `results.json`.
- Calls AIO `adapter.generate` once per image, through its actual preparation,
  encoding cache, native Comfy sampler and VAE decoder. Model loaders are called
  on every image. One process and its encoding cache are retained. This is not
  a persistent preloaded-model API benchmark.
- No graph executor, HTTP server, browser, live preview or UI timing. PNG saving
  uses Pillow without prompt/workflow metadata. Save timing ends when PNG writing
  returns, without a disk `fsync`.

The selected source files and measured harness snapshot are fingerprinted in
`source-hashes.json`; Git revisions alone do not identify local modifications.
No installed ComfyUI/AIO source or dependency was changed. GPU execution used
approved sandbox escalation. No commit or push was made.

## Ordinary execution times

All values are seconds. Phase timers synchronize CUDA at coarse boundaries.
Sampling includes lazy model transfer/loading as in our sd.cpp timing. Individual
loader times are nested inside preparation and must not be added to it again.

| Phase | First image | Repeat 1 | Repeat 2 |
| --- | ---: | ---: | ---: |
| Prepare models | 0.564 | 0.255 | 0.247 |
| Conditioning | 3.942 | 0.001 | 0.001 |
| Sampling | 38.745 | 37.152 | 37.430 |
| VAE decode | 1.130 | 0.666 | 0.653 |
| Per-image start to saved PNG | 44.833 | 38.526 | 38.782 |

First process start to saved PNG is **47.233 s**, including imports/bootstrap.
This is not a cold-filesystem test. Repeat mean is **38.654 s**, with **37.291 s**
sampling. Both prompt encodings are reused on the repeats; the first image has
two encoding misses. Repeat timing must therefore not be directly compared with
fresh-process sd.cpp total time as a pure diffusion-speed measurement.

The current [sd.cpp baseline](../profiling/DOWN-AUTO.md), with all accepted
fusions enabled, averages **55.603 s sampling / 60.347 s process-to-PNG** across
three fresh processes with warm filesystem cache. Its conditioning averages
1.853 s and VAE 2.037 s. Comfy uses 23 Torch CPU threads here; the separately
tuned native reference uses 11. Libraries, precision and process lifecycle also
differ, so this is a comparison of the actual selected execution paths, not a
controlled experiment that varies only one backend setting.

The fourth image is profiled separately: 36.458 s sampling and 39.386 s to PNG,
including profiler shutdown before saving. It is excluded from ordinary timing
means. Its sampling is 2.2% faster than the two ordinary repeats, illustrating
run/clock variation; the difference must not be interpreted as negative profiler
overhead. GPU clocks were not locked. The manual UI's earlier 42 s is context,
not another sample in these measurements.

## Actual GPU work in diffusion

Comfy's Chrome trace is joined through CUDA launch correlation IDs to nested
CPU `bench::diffusion_forward` scopes. Only actual kernel, memcpy and memset
events are summed. GPU annotations are excluded to avoid double counting.
All 83,186 GPU activities in the complete trace map to launch records; 74,090
belong to the **80 diffusion forwards**. There are no unmatched GPU activities.
The native reference is the earlier CUPTI capture
`20261013T133554Z-activity-he73vero`, also covering 80 forwards, excluding encoder,
decoder and initial diffusion weight loading. Native reported zero dropped
records; this Torch trace does not provide an equivalent explicit drop counter.

These are **summed GPU activity seconds**, not independent wall-clock stages.
Categories are exclusive. Normalize/modulate/residual operations are grouped
together because Kitchen fuses operations that occupy separate native groups.

| GPU work | Optimized sd.cpp | Comfy / Kitchen | sd.cpp minus Comfy |
| --- | ---: | ---: | ---: |
| INT8 GEMM + output dequantization | 21.525 | 20.247 | 1.278 |
| Linear ConvRot / activation quantization | 4.460 | 3.593 | 0.868 |
| Normalization, RoPE, gating / elementwise | 12.912 | 2.637 | 10.274 |
| Layout / dtype / device copies | 7.122 | 1.182 | 5.940 |
| Sage attention core | 5.585 | 6.138 | -0.553 |
| Sage preparation / quantization | 1.988 | 1.276 | 0.712 |
| Masked prefix attention | 0.002 | 0.001 | <0.001 |
| Host transfers | 0.033 | <0.001 | 0.033 |
| Other / memset | 0.049 | 0.059 | -0.010 |
| **Total GPU activity** | **53.675** | **35.133** | **18.542** |

Comfy GPU union time is 35.133 s across a 35.402 s first-to-last activity span;
overlap is negligible. Native union time is 53.675 s within its 54.726 s measured
diffusion graph window. Window boundaries differ slightly: launch ancestry and
first/last activities for Comfy versus graph log boundaries for native. Neither
window is a complete image request time.

Normalization/elementwise plus layout/copies account for **16.214 s, or 87.4%**
of the aggregate GPU-time difference. This is an attribution of observed work,
not a guaranteed recoverable speedup. Different precision, tensor layouts,
fusion boundaries and amount of prefix work contribute to those costs.

## Verified dispatch and precision

- 40 sampling callbacks and 80 model forwards in every image. CFG batching is
  not responsible for this speed difference.
- 15,744 successful Kitchen CUTLASS INT8/dequant calls, matched by 15,744 GPU
  kernels. No instrumented fallback call is observed. There are also 15,744
  ConvRot64 kernels. Native has 15,360 fused GEMM kernels and 10,240 activation
  quantizers: it reuses some quantized inputs, and the prefix computation layout
  differs. Kernel counts must not be interpreted as identical per-call work.
- Kitchen really executes 2,560 full attention kernels plus 64 masked-prefix
  kernels. Its full attention uses INT8 Q/K/V and BF16 output; native uses
  SageAttention2++ with FP8 V. Native's attention core is already faster here;
  the overall gap does not indicate missing native Sage activation.
- Observed diffusion inputs and cached prefix tensors are **BF16**. Actual cache
  placement is `cuda:0`, with two initial misses and 78 hits. The native reference
  uses F32 activations/prefix. An INT8 weight file does not imply matching
  activation precision between engines.
- Kitchen fused Q/K RMSNorm + RoPE: **0.827 s**, 2,624 kernels. Native separate
  RMSNorm and RoPE were approximately **2.283 + 1.683 = 3.966 s**.
- Kitchen fused AdaLN: **0.706 s**; gated residual `addcmul_`: **1.087 s**.
  The local Comfy source directly calls `ck.rms_rope`, `ck.adaln`, and in-place
  `addcmul_` in `comfy/ldm/qwen_image21/model.py`. The trace confirms those paths.

## Image and memory checks

All four PNGs are 1600×896 and share RGB SHA-256
`526a2535b24c7517b4787ef77870692a1972ed07058075633972b87b8c62a276`.
They differ from native's established hash, as expected for the different
attention/precision paths. This validates within-engine repeatability, not
cross-engine pixel identity or visual quality acceptance.

Validation passed for both Python files' syntax, all four PNG dimensions and
recomputed RGB hashes, empty PNG metadata, complete GPU launch correlation and
exclusive accounting totals. A literal-content check also found neither test
prompt in the new trace, configuration, results or execution log. The maintained
harness's post-export cleanup change was syntax-checked; the four live runs used
the preserved snapshot before that change.

First-image Torch peaks are 12,860.9 MiB allocated / 13,688 MiB reserved. Repeat
allocated peak is 9,318.1 MiB; reserved is 13,688 then 10,018 MiB. Sampled whole
GPU usage ranges 1,833–15,590 MiB over the process, including desktop usage.
Whole-GPU readings and Torch allocator counters have different scope. These
numbers do not prove that all three models remain resident together; the
sampling log reports the diffusion model fully loaded at approximately 6,921 MiB.

## Next experiment, proposed only

Proceed with the previously proposed **Q/K RMSNorm + RoPE fusion** in sd.cpp.
Its current separate kernels cost about 3.97 s and Kitchen demonstrates a fused
execution path. Start with native's existing precision and a narrow opt-in
implementation, validating numerical behavior and full-image timing against
the frozen reference. Kitchen's BF16 result is not a promise that a native F32
fusion reaches 0.83 s or preserves identical pixels without validation.

After that, separately consider fused normalization/modulation and gated
residuals, then eliminate redundant layout/copies. BF16 activations are a larger
precision/quality experiment and should not be bundled into the first fusion.
Further INT8 GEMM tuning is lower priority given this profile. No additional
native optimization was implemented in this task.

## Privacy, artifacts and reproduction

The harness uses installed source/dependencies, explicitly selected model files
and the existing benchmark prompt files. It does not ingest existing ComfyUI
logs, history, queues, private workflows or private configuration files. AIO
route registration and `extra_model_paths.yaml` are bypassed. Model inference
is local; Hugging Face/Transformers are set to offline mode.

New run artifacts contain timings, code/model paths, hashes, tensor dtype/shape
metadata, profiler kernel names and the requested generated images. Prompt
text, workflow metadata and tensor values are not written into the new logs,
trace or PNG metadata. Encoding tensors remain process-local CPU memory and
disappear when the process exits. Compiler caches are directed into the run.
This is a scoped benchmark, not an OS-level file-access or network audit.

```sh
PYTHONDONTWRITEBYTECODE=1 /path/to/ComfyUI/.venv/bin/python \
  experiments/qwen-image21-sage/comfy-headless/run.py --runs 4 --profile-index 3
python experiments/qwen-image21-sage/comfy-headless/analyze.py \
  experiments/qwen-image21-sage/runs/20261013T135141Z-comfy-3_ly1kh7
```

For another run, replace the analysis directory with the printed run directory.
Use `--probe` to check imports and path/settings resolution without loading
weights or generating images. Each run creates its own directory. Existing
artifacts are not overwritten by the harness.

Primary evidence in the run: `config.json`, `results.json`, `harness.py`,
`source-hashes.json`, `trace-3.json`, `diffusion-profile.json`,
`diffusion-kernels.csv`, `gpu.csv`, and `image-0.png` through `image-3.png`.
`benchmark-retry.log` in this directory is this test's own execution log.
An initial harness attempt failed before model loading on a missing progress
callback method; that was corrected before the four completed images.

The measured harness also exported `kernels-3.json` using FunctionEvent parents.
That diagnostic export is **not used**: CUDA events have empty CPU ancestry and
include overlapping GPU annotations. The maintained harness removes that export;
the analysis above uses actual Chrome-trace activities and launch correlations.
The saved `harness.py` remains the exact measured version. Fine-grained `phases`
arrays in `results.json` are host-call timings, which can include implicit
waits; they are not GPU kernel durations.
