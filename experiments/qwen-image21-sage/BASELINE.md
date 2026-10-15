# Current frozen Qwen Image 2.1 baseline: v2

Accepted by Tomas on 2026-10-15: **qwen-image21-1600x896-v2**.
The internal strip algorithm is dynamic and has no resolution whitelist.
1600×896 describes the accepted validation scope, not a restriction in the
native algorithm. v1 remains an immutable comparison control below.

Use `./experiments/qwen-image21-sage/run.sh --dry-run`, `--serve` or `--run`.
The Rust API also selects v2; its API revision and request validation are unchanged.
All original flags, prompts, precision, auto memory policy and idle-retention
choice remain; the only added switch is `SD_VAE_DYNAMIC_BLOCK_STRIPS=1`.
Native engine defaults remain separate from the frozen profile.

The independently copied executable is
`baselines/1600x896-v2/sd-server`, SHA-256:
`9fd49b0e5cb3426043ebd85271ad167555551ac65252590cde5ac25bed44bbb0`.
The manifest pins command, 29 switches, prompt copies and source hashes.
Preserve both v1 and v2 executables during cleanup; neither binary ships in Git.
The complete native patches/lock now restore the v2 source, with the new flag
still off in engine defaults. Full patch restoration and hashes are verified.

| Matched 1600 warm metric (one sample per mode) | v1 | v2 |
| --- | ---: | ---: |
| VAE arena, MiB | 8073.24 | 6373.44 |
| Prepared GPU weights, MiB | 2946.44 | 1008.83 |
| Start to saved PNG, s | 39.210 | 39.007 |
| VAE, s | 1.73 | 1.72 |

v2 is selected for memory savings, not an established speed gain. Four images
(two per mode) match the v1 RGBA reference exactly. Thirteen synthetic checks
pass. This does not prove all prompts, editing paths, shapes or GPUs are exact.
The wider-scope historical v1 evidence remains valid for its original binary.

At 4096 auto tiling changes from 49 to 9 tiles; the candidate is faster but its
pixels differ from v1. A completed same-outer-grid diagnostic gives four bitidentical 4096 images,
844.87 MiB less VAE workspace and warm VAE 33.05→32.00 s. This does not
introduce a fixed tile size into v2. See [4096 control](profiling/VAE-FIXED-TILES-4096.md). See
[dynamic-block report](profiling/VAE-DYNAMIC-BLOCKS.md),
[v2 evidence](evidence/1600x896-v2/) and [test decisions](TEST-DECISIONS.md).
No new model run is needed solely to freeze this already tested executable.

## Preserved v1 history

# Frozen Qwen Image 2.1 baseline

Accepted by Tomas on 2026-10-14: **qwen-image21-1600x896-v1**.
This is the preserved historical v1 baseline; v2 above is current. Historical reports and their default-off
statements describe earlier decisions; this profile now explicitly enables the
accepted combination. Engine-wide defaults remain unchanged.

## Reuse

From the project root:

```sh
./experiments/qwen-image21-sage/run.sh --dry-run
./experiments/qwen-image21-sage/run.sh --serve
```

Dry-run is the default: verify the frozen files and print settings without using
the GPU. Serve starts the frozen native process on `127.0.0.1:1234`; stop it with
Ctrl-C. `QWEN_PORT` changes only the serve port. Requests use the existing
`/sdcpp/v1/img_gen` API. Keep requests within the validated profile below;
the native API still accepts other settings and the launcher is not an API
validation layer.

To replay the reference, stop the separately started server first, then run:

```sh
./experiments/qwen-image21-sage/run.sh --run
```

This now generates **three images in one process**, rather than the historical
single CLI image: one initial request and two warm repeats. It verifies native
reference RGB pixels, two warm prompt-cache hits and 80 diffusion forwards,
records request-to-closed-PNG-write timing, then stops its server. Results are
written to a fresh `experiments/qwen-image21-sage/runs/*-resident-*` directory.
It does not stop other GPU applications. The replay uses Python and Pillow as a
local test client; native inference has no Python or ComfyUI dependency.

`QWEN_MODEL_ROOT` may relocate the same model files. An ignored
`local-settings.json` beside the launcher can set `model_root`; the environment
variable takes precedence. Otherwise the default is `models` under the current
working directory. Model contents are not
hashed by this launcher; replacing a weight file invalidates the model-level
validation even if its filename is unchanged. The fixed reference seed is 42;
non-42 `QWEN_SEED` overrides are rejected. To retain reproducibility, the
reference prompts are frozen copies, not the editable top-level prompt files.
`--serve` lets API clients provide their own prompts, but the exact-image claim
applies only to the recorded reference prompt/seed.

## Profile

| Setting | Frozen choice |
| --- | --- |
| Hardware acceptance scope | RTX 4070 Ti SUPER, 16 GB, CUDA 13.4, SM89 |
| Diffusion | `diffusion_models/qwen-image2.1/qwen_image_2.1_int8_convrot.safetensors` |
| Text encoder | `text_encoders/qwen-image2.1/qwen3vl_8b_int8_convrot.safetensors` |
| VAE file | `vae/qwen-image2.1/qwen_image_2.1_vae_bf16.safetensors` |
| Canvas | **1600×896**, not 1600×900 |
| Sampling | 40 steps, ER-SDE, beta, CFG 2.5, seed 42, CPU RNG, batch 1 |
| Extra conditioning | No LoRA, reference image or second pass; frozen positive and negative prompts |
| Attention | SageAttention2++; accepted split V on, split K off |
| Diffusion precision | BF16 attention/cache and residual; F32 gate/up, with packing |
| VAE precision | Existing F32 activation/F16 convolution path; file name does not imply all-BF16 execution |
| Threads | 11 CPU threads; 512 CUDA threads for the accepted K=4096 ConvRot path |
| Memory | Auto-fit including CUDA scratch accounting; **no manual `--max-vram` margin** |
| Lifetime | Persistent server, CPU conditioning cache capacity 4; extra idle-retention flag off |
| VAE memory reductions | Existing convolution strips plus shortcut, stage-3 and whole residual-block strips |
| Metadata | Image metadata disabled; benchmark prompts only; no private ComfyUI data |

All 28 explicit runtime switches are in
[baseline-environment.json](baselines/1600x896-v1/baseline-environment.json).
The launcher clears inherited `SD_*`/`GGML_*` switches first and rejects
profiling injection, so old experimental environment settings do not leak in.
The internal runtime safety reserve is unchanged; removing the manual 512 MiB
margin does not remove that internal reserve.

## Frozen artifacts and evidence

[manifest.json](baselines/1600x896-v1/manifest.json) pins the environment,
command template, prompt copies and source-evidence files by SHA-256.
`baselines/1600x896-v1/sd-server` is a separate byte-identical copy of the tested
binary, not a link to the mutable build directory. The binary is ignored by Git.

Binary SHA-256:
`4e33830f7eba36573cba9785249641ae8c727be5e6a506e9c7ef0ee74beea2e3`.

Preserve this local binary when cleaning run/build artifacts. The redundant
original build directory was removed during publication cleanup. If absent,
restore an independently archived executable with the exact hash above; Git
does not distribute it. The launcher fails on a missing or different executable.
A rebuild from a later
source/dependency state is a new candidate, not automatically baseline v1.
`after-hashes.json` records selected native sources, not a full
reproducible toolchain archive. CUDA/driver/shared-library changes can still
change behavior despite an identical executable. `build-inference.sh` builds
the development tree; it deliberately does not overwrite this frozen copy.

Accepted live evidence: four counterbalanced series, 12 images, all decoded RGBA
pixels identical to the native reference. Four warm samples per mode:

| Metric | Previous control | Frozen combination |
| --- | ---: | ---: |
| Request to saved PNG | 39.074 s | 39.219 s |
| VAE time | 1.700 s | 1.720 s |
| VAE workspace | 8663.87 MiB | **8073.24 MiB** |
| Diffusion weights reloaded | 2946.44 MiB / 3 blocks | 2946.44 MiB / 3 blocks |

Compact plans, results and allocation validation are preserved under
[`evidence/1600x896-v1/`](evidence/1600x896-v1/). Publication cleanup removed old
raw images/logs/builds and sanitized filesystem paths; it did not change the
binary, model settings, prompts, numerical measurements or native source hashes.

The combination is selected for **590.625 MiB less workspace**, not a measured
speedup. Earlier convolution strips already reduced workspace from 10632.62 to
8663.87 MiB and reloads from 5911.70 to 2946.44 MiB. Do not add measured speedups
from different historical comparisons.

Native RGBA reference:
`473bfa4ab0c2230258a168afbe07523bd6bc7d81a4a2daf458500b6df036132d`.
The last matched resident comparison was native 40.216 s versus headless
AIO/Kitchen 38.814 s. Comparing the new 39.219 s result with that older Comfy
number is indicative only, not a new controlled 0.405-second result. The two
engines' reference images differ.

**Scope limit:** component tests at some small/odd dimensions fail bit identity.
The accepted 1600×896 reference images pass; that does not prove every prompt,
intermediate tensor, resolution, editing/LoRA path or other GPU is identical.
Do not silently promote these VAE flags to all model/resolution profiles.

See [all test decisions](TEST-DECISIONS.md), the
[latest comparison](profiling/VAE-THREE-STRIPS.md) and
[resident Comfy comparison](profiling/RESIDENT-COMPARISON.md).

## Validation of this freeze

Shell syntax, dry-run and offline checks passed. The checker compares the actual
prepared server/replay commands, flags, request and prompt hashes with the
accepted live-run plan; it also checks inherited-toggle isolation and rejects
invalid overrides. Process launch and port reservation were intercepted, so no
GPU or network execution was needed. Selected native sources and the binary
are unchanged; the prior 12-image live evidence is reused. No escalation, native
rebuild, commit or push occurred during the freeze. Re-run the offline check with:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 experiments/qwen-image21-sage/profiling/check-frozen-baseline.py
```
