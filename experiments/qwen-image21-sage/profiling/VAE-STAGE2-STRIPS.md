# Stage-2 VAE convolution strips

Date: 2026-10-14. Control: frozen `qwen-image21-1600x896-v1`.

## Scope and implementation

The candidate adds `SD_VAE_STRIP_STAGE2=1` for exactly
`first_stage_model.decoder.upsamples.2.upsamples.3.resample.1`, with input
800×448 and 576→576 channels. It reuses the existing 64-row convolution
implementation, including halo handling, F32 scaling by 1/128, F16 im2col and
matrix multiplication, output scaling and bias. It adds no CUDA kernel or
precision change. Other sizes and names retain the existing path, and weight
adapters still take their original early-return path.

All 28 accepted switches remain enabled or disabled as specified by the frozen
profile. In particular, gate/up stays F32, split V stays on, split K stays off,
and the accepted VAE strips stay on. Idle retention, fused upscale/im2col and
PAD-scale reuse remain off. No manual VRAM margin is introduced.

The candidate executable SHA-256 is
`80227a5a72b8972e773d0b2ce7179aa092463e6a741553ab0f963673a219d155`.
The frozen control executable is unchanged. The isolated source delta and
hashes are preserved in [`../evidence/vae-stage2-strips/`](../evidence/vae-stage2-strips/).

## Numerical and full-memory gates

| Metric | Control | Candidate |
| --- | ---: | ---: |
| Isolated convolution arena | 4533.822510 MiB | 2405.990479 MiB |
| Complete VAE graph arena | 8073.242432 MiB | 7879.111816 MiB |
| Complete VAE graph nodes | 2265 | 2332 |
| Additional stage-2 strip writes | 0 | 7 |
| Separate CUDA pool growth estimate | 394 MiB | 394 MiB |

The complete arena saving is **194.130616 MiB (2.40%)**. The much larger
isolated saving does not translate directly to the complete graph. These are
workspace measurements, not total device memory or resident model weights.
Neither graph needs a scheduler fallback.

The production component test compares all **825753600 output bytes**, including
the convolution's scaling and bias, using deterministic synthetic input and
weights. Output is bitidentical and the retained input is unchanged. Small
1×1, 5×33 and 65×129 cases and an unrelated layer name verify that the new
eligibility guard leaves their original path unchanged. They do not validate
strip arithmetic at arbitrary sizes.

The full-graph probe parses only the real checkpoint header, constructs the
actual WAN VAE runner, and registers parameter names **before** building the
graph. It loads no model weight values and performs no image inference. The
control must reproduce both 8073.242432 MiB and 2265 nodes before image tests
are allowed.

An initial missing prefix separator caused a metadata lookup failure. A second
probe omitted parameter-name registration before graph construction and missed
the accepted stage-3 dispatch, producing invalid memory numbers. Both were
corrected before the successful gate. The measurements above use only the
corrected probe; earlier probe output is not evidence of a regression or gain.

## Image comparison protocol

Four independent server sessions in control/candidate/candidate/control order,
each with one initial image and two warm repeats: 12 images total, four warm
samples per mode. Use the frozen synthetic positive/negative prompts, seed 42,
1600×896, 40 ER-SDE/beta steps, CFG 2.5, Sage2++, CPU RNG and 11 CPU threads.
No LoRA, reference image or second pass. Warm requests reuse both conditioning
entries in process-local CPU RAM.

Time to saved image starts before HTTP submission and ends after the PNG file
write closes; it excludes process startup and fsync. Compare decoded RGBA
bytes against the frozen native reference, not against ComfyUI. Each request
must execute 80 diffusion forwards without a VAE spatial-tiling retry.

Only benchmark inputs, model assets and logs created by this experiment are
used. No private ComfyUI logs, history or settings are read. GPU and local
loopback access required sandbox escalation for the authorized image tests.

## Results and decision

All **12/12 decoded RGBA images are bitidentical** to the frozen native
reference, SHA-256
`473bfa4ab0c2230258a168afbe07523bd6bc7d81a4a2daf458500b6df036132d`.
All loaded-graph measurements agree with the corrected memory probe. All warm
requests hit both conditioning entries and execute no text encoder. The new
convolution dispatch occurs once per candidate decode; no spatial retry occurs.

| Warm metric, four images per mode | Frozen control | Candidate |
| --- | ---: | ---: |
| Request to saved PNG, mean | 38.6268 s | 38.7560 s |
| Request to saved PNG, median | 38.6893 s | 38.5219 s |
| Request to saved PNG, range | 38.188–38.940 s | 38.165–39.816 s |
| Sampling, mean | 36.7175 s | 36.8325 s |
| VAE decode, mean | 1.6725 s | 1.6825 s |
| Full VAE arena | 8073.24 MiB | 7879.11 MiB |
| Diffusion weights reloaded | 2946.44 MiB / 3 blocks | 2946.44 MiB / 3 blocks |
| Tensor loading, included in phases | 0.40 s | 0.40 s |

Initial request times were 41.859/41.542 s for the control and 41.703/41.745 s
for the candidate. These are excluded from warm statistics. One-second
whole-device peak samples were 14966/14877 MiB for the control and
14762/14673 MiB for the candidate. They include other applications and can miss
brief peaks; the full-graph arena is the reliable allocation comparison.

**Conclusion:** a reproducible 194 MiB workspace saving with preserved reference
pixels, but no demonstrated speed improvement or additional resident diffusion
block. The mean PNG time is 0.129 s higher while the median is 0.167 s lower;
that mixed result and the observed ranges do not support a speed claim. VAE
time differs by just 0.010 s in the mean.

This is a reasonable optional memory optimization for the tested format. It
does not justify another idle-retention test by itself: the observed reload
remains identical. Do not repeat this candidate as an untested idea, generalize
its identity guarantee to other prompts/shapes/GPUs, or compare these timings
with old ComfyUI timings as a new matched comparison. No ComfyUI benchmark was
run in this experiment.

The frozen profile is **not promoted or edited**. The experiment patch and
candidate executable are preserved separately, and the active native source
is restored to the pinned baseline after testing. The candidate remains at
`runs/vae-stage2-strips-build/sd-server`; the 12-image coordinator is
`runs/20261014T202635Z-vae-stage2-mkiwzvuc`. Compact results, actual request plans,
source/binary hashes and validation are preserved under
[`../evidence/vae-stage2-strips/`](../evidence/vae-stage2-strips/), without raw
images, prompts or device logs in the tracked evidence.

## Reproduction

The experiment uses the existing configured CUDA build and Python/Pillow test
client. Start from the pinned native checkout, with no other native changes.
The build helper checks selected source hashes before compiling and copies the
candidate separately; it never overwrites the frozen executable.

```sh
python3 native/prepare.py --check
git -C stable-diffusion.cpp apply ../experiments/qwen-image21-sage/evidence/vae-stage2-strips/experiment.patch
python3 experiments/qwen-image21-sage/profiling/build-vae-stage2-checks.py
python3 experiments/qwen-image21-sage/profiling/check-vae-stage2.py --model-root "$QWEN_MODEL_ROOT"
python3 experiments/qwen-image21-sage/profiling/compare-vae-stage2.py --model-root "$QWEN_MODEL_ROOT"
```

The memory and numerical gates must pass before the image coordinator starts.
The scripts are experiment tools, not the production launcher. Leave no other
generation running during the timing series. Restore only this delta after the
experiment, preserving the separate candidate executable and evidence:

```sh
git -C stable-diffusion.cpp apply --reverse ../experiments/qwen-image21-sage/evidence/vae-stage2-strips/experiment.patch
cmake --build stable-diffusion.cpp/build-cuda --target sd-server --parallel 2
python3 native/prepare.py --check
```

A rebuilt executable may have a different hash with another toolchain. The
helper records its actual hash and the gates must be rerun; it is not silently
treated as either the measured candidate or frozen baseline.

## Verification boundary

The candidate build helper was executed successfully and reproduced the exact
measured candidate executable hash. Patch application and reversal reproduced
both recorded header hashes in an isolated directory. After restoring the
active source, `native/prepare.py --check` and the frozen-baseline offline
checker passed. Python syntax and preserved JSON checks passed. No new CUDA
kernel was added; compute-sanitizer was not run. The twelve images establish
saved-pixel identity for this reference case, not identity of every VAE
intermediate or all possible inputs. No commit or push was performed.
