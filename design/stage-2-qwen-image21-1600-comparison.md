# Stage 2 — 1600×896 resolution comparison

2026-10-18. Tomas requested a larger 1600-wide 16:9 image to investigate whether
the Rust/Python differences grow with resolution. This is a new authorized
workload, preserving the earlier [512×512 control](stage-2-qwen-image21-first-comparison.md).
Full Stage 2 remains in progress.

## S2-11 — change geometry alone

The separate [profile](../inference/profiles/qwen21-1600x896-int8.json) changes
only width, height and the descriptive purpose field from the previous profile.
1600×896 is the existing comparison geometry aligned to the model's latent grid;
mathematical 16:9 at width 1600 would be 900 pixels high. This near-16:9 selection
was stated before execution. The 512×512 profile and its evidence are unchanged.

The same synthetic butterfly prompt, seed 17, four Euler/simple steps, CFG 1,
source shift 0.69, INT8 ConvRot DiT and Qwen3-VL files, BF16 VAE, PyTorch attention,
Triton-disabled ComfyUI startup and static full-component lifecycle apply to both
backends. No LoRA, reference image, different precision or new optimization is
introduced. Source VAE `STRIP_ELEMS=2**24`, halos, convolution operations, cache
thresholds and autotuning remain unchanged. Model assets are the three previously
authorized files under `/mnt/media/gemensam/AI/ComfyUI/models`.

Only the development driver's registered-profile selection and result label,
the launch/reference/benchmark profile arguments and the comparator's resolution
description needed extension. Both drivers still reject unregistered profiles.
The default remains the preserved 512×512 test. No model arithmetic, memory
policy or inference-library implementation changed for this resolution test.

Before execution, the frozen native baseline and test decisions were read as
required. That separate stable-diffusion.cpp control, its executable and its
accepted settings are untouched. Its historical performance numbers are not
substituted for the current Rust/ComfyUI comparison.

## Measurement and quality contract

Three fresh-process pairs alternate Python→Rust, Rust→Python, Python→Rust.
The same monotonic, CUDA-synchronized phase boundaries as the 512×512 comparison
apply. Pipeline wall time includes component loading, inference, unload and
intermediate export, and excludes launcher/build/preflight/imports and final
PNG/raw-image serialization. OS model caches are already warm. The Rust build
is the same debug development configuration linked to the optimized frozen
LibTorch/Kitchen libraries. This is not resident throughput or application
request latency; GPU clocks are not fixed.

The strict comparison checks all Float32 bytes of context, latent and processed
RGBA pixels, plus decoded PNG pixels. The larger image still uses **four steps**:
this isolates geometry, and does not establish a production quality preset.
Timing comparisons with 512×512 use its retained earlier three-pair medians,
which were measured earlier in the same development session rather than
interleaved with these larger runs.

## Results

All **three matched pairs pass exact comparison**, and each boundary has a
single common hash across all six runs. Maximum error, MAE and RMSE are zero;
all values are finite. The shapes are context `[1,20,4096]`, post-normalization
latent `[1,64,56,100]` and RGBA pixels `[1,896,1600,4]`. This checks 24,698,880
Float32 bytes per image across those boundaries. All decoded PNG pixels match.
No tolerance was introduced or numerical mismatch accepted.

The image has **5.46875×** as many pixels as the 512×512 control. Text context
is unchanged across resolutions; latent/noise shape and image composition change
with geometry, so cross-resolution image identity is not an acceptance target.

Seconds, median with observed minimum–maximum over three runs per backend:

| Phase | Rust median | Python median | Rust range | Python range |
| --- | ---: | ---: | ---: | ---: |
| Text load/residency | 1.190 | 1.468 | 1.171–1.219 | 1.396–1.484 |
| Text encoding | 0.332 | 0.502 | 0.312–0.337 | 0.500–0.543 |
| DiT load/residency | 0.924 | 0.955 | 0.919–0.956 | 0.943–1.227 |
| Four-step sampling | **2.351** | **2.355** | 2.336–2.361 | 2.349–2.595 |
| VAE load/residency | 0.875 | 0.228 | 0.241–0.993 | 0.192–0.264 |
| VAE decode/pixel processing | 0.663 | 0.638 | 0.656–0.674 | 0.634–0.638 |
| Measured pipeline | **6.933** | **6.726** | **6.241–7.053** | **6.640–7.304** |

The independently timed pipeline includes housekeeping and exports; do not sum
phase medians to reconstruct it. VAE load time is particularly variable, and
the pipeline ranges overlap.

| Measured resolution | Rust pipeline | Python pipeline | Rust extra time |
| --- | ---: | ---: | ---: |
| Preserved 512×512 control | 4.902 s | 4.545 s | +0.357 s / +7.85% |
| New 1600×896 test | 6.933 s | 6.726 s | +0.207 s / +3.08% |

For this four-step workload, the observed gap **shrinks** in both seconds and
percentage. Sampling, previously 0.737 versus 0.485 s, is now effectively tied
at 2.351 versus 2.355 s. The 3.5 ms difference is not evidence of a meaningful
Rust sampling advantage. These small samples and unlocked clocks do not
establish a general scaling law or a performance ranking for 40-step workloads.
No kernel-level cause has been profiled and no tuning followed these results.

The larger image contains the blue butterfly form and red flower, with strong
grain, soft detail and ghosted wing structure at four steps. Both engines produce
those same features **pixel for pixel**. This proves no backend quality loss on
this case, rather than satisfactory production image quality.

![Rust 1600×896 synthetic benchmark image](stage-2-qwen-image21-1600-comparison-evidence/rust-image.png)

[The Python output](stage-2-qwen-image21-1600-comparison-evidence/python-image.png)
has identical pixels. [The summary and samples](stage-2-qwen-image21-1600-comparison-evidence/summary.json)
retain the exact numerical and timing evidence. Raw activations remain in the
isolated development runtime; source, registered profiles, strict comparison
tools, boundary hashes and both synthetic PNGs are retained for later upgrades.

## Validation and limits

- Formatting, all-target Clippy with warnings denied and shell/Python syntax
  checks pass. No model implementation changed, so the earlier component suite
  was not rerun solely for profile registration.
- All three trained Rust runs pass Python-free isolation. The default strict
  comparator passes all three pairs, with matching profile objects.
- The original 512 profile and all model/tensor/attention/sampler/VAE/native
  implementations match the earlier snapshot. Only six driver/tool files and
  the separate added profile changed before measurement.
- The source/profile/fixture snapshot is verified before and after measurement;
  imported original ComfyUI files are checked against Stage 0 source hashes.
- All three complete model-file SHA256 hashes, four source locks and 32 original
  CUDA object hashes are checked against the preserved comparison control.

Neither process ran out of memory or required different offload, tiling, cache
threshold or precision settings. This establishes successful execution on the
available memory for these cases, not matching peak-VRAM usage; peak VRAM was
not instrumented. There was no process inventory, service interruption, cloud
service, Computer Use, production integration, commit or push. All required
sandbox escalations were approved; no auto-review rejection occurred.

The [checkpoint](stage-2-qwen-image21-1600-comparison-checkpoint.json) records
hashes, checks and retained evidence. Stage 2, BF16/Turbo, editing, LoRA and
general memory/algorithm acceptance remain incomplete. No new source/design
departure or autotuning change was needed.

## Reproduction

```sh
export IMAGINE_PARITY_ROOT=/tmp/imagine-rust-comfy-parity-20261017
export IMAGINE_REFERENCE_SITE=/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages
bash inference/tools/benchmark_qwen_first_image.sh \
  /home/thhel/git/ComfyUI /mnt/media/gemensam/AI/ComfyUI/models NEW_RUN_NAME \
  /home/thhel/ChatGPT/imagine/inference/profiles/qwen21-1600x896-int8.json
```

Run names must be fresh. The original three-argument command retains the
512×512 default. Reference imports use frozen original ComfyUI functions and
the existing read/network audit; Rust runs without Python/ComfyUI availability
inside its isolated process. No ordinary Dreamchat/ComfyUI runtime is an input.
