# Qwen21 — seed-only 1MP portrait editing comparison

2026-10-18. Tomas requested a new random seed in both Rust and ComfyUI to
investigate the poor appearance of the previous portrait edit. The source
portrait and the seed-17 result are preserved unchanged. This is an isolated
development comparison, with no production runtime data or service changes.

## Controlled change

One `secrets.randbelow(2**32)` draw selected **3061383787**, frozen before either
engine ran. The new registered profile differs from the previous editing
profile in **exactly one field: seed 17 → 3061383787**. This reruns the garden
reference edit; it does not regenerate the input portrait.

The reference is the same full 1024×1024 portrait tensor, with no resize/crop,
and SHA256 `4256a0499cd68d2cfe5413e6ecff48da73ad420e9fcd261ff979c06515ca2fbf`.
Output remains 1024×1024, batch 1, 40 Euler/simple steps, CFG 1, shift 0.69,
INT8 ConvRot DiT and Qwen3-VL encoder, BF16 VAE/compute, PyTorch attention,
no LoRA and empty negative prompt. Full prompt and source configuration are in
`inference/profiles/qwen21-editing-portrait-1mpx-int8-seed3061383787.json`.

The original automatic prefix-cache policy still selects CPU/pinned storage:
2,176,319,488 cached/pinned bytes and two GPU staging buffers, identical in
both engines. Component loading/unloading, precision, quantization, compiler
and autotuning policies remain unchanged. No parity/design exception was needed.

## Result

The newly generated images match **pixel for pixel**. All six tensor boundaries
also match exactly, with finite values and zero unequal values/bytes: positive
context, negative context, reference vision image, reference VAE latent, final
sampled latent and Float32 RGBA. Conditioning metadata matches.

The four conditioning/reference tensor hashes are also unchanged from seed 17;
the final latent and output image hashes change. This confirms that the seed
change reaches sampling while the conditioning input remains identical.

| Measured phase, seconds | Rust | ComfyUI Python |
| --- | ---: | ---: |
| Reference VAE load + encode | 0.687 | 0.785 |
| Text/vision encoding | 2.739 | 3.091 |
| Sampling | 23.506 | 23.782 |
| VAE decode | 0.433 | 0.429 |
| Pipeline including unload/intermediate export | **31.543** | **32.383** |

One sequential fresh-process pair, Rust followed by Python, with the OS model
cache already warm. These are the existing driver pipeline timing boundaries;
launcher checks/build and final PNG export are excluded. This is not a
statistically established speed advantage or request-to-PNG benchmark.

## Visual assessment

Compared with seed 17, the new result has substantially more natural skin,
hair and foliage. The severe coarse detail and exaggerated local contrast
are absent to the same extent. The garden edit and portrait composition remain
recognizable. Both engines produce this same improvement.

This supports seed sensitivity for this exact workload. It does not establish
a seed-handling bug, explain the model's internal cause, or guarantee good
quality across seeds. No CFG, sampler, precision or other tuning was applied.

| Previous seed 17 | New seed 3061383787 |
| --- | --- |
| ![Previous edit](stage-2-qwen-image21-portrait-1mpx-evidence/edit-rust.png) | ![New edit](stage-2-qwen-image21-portrait-seed3061383787-evidence/rust.png) |

## Validation and preservation

Locked/offline Rust build, formatting, all-target Clippy with warnings denied,
Python AST and shell syntax checks pass. The fresh trained pair passes the
strict numerical gate and Rust runs in the Python-free namespace. Three full
model SHA256 hashes match the preserved baseline, as do all four Stage 0 locks
and 32 original CUDA objects. The earlier 25-test regression result is historical;
it was not rerun for this profile-only registration/test-launcher change.

The evidence retains both PNGs/results, conditioning and cache observations,
logs, namespace proof, seed-only comparison and verification metadata. Initial
artifact-export attempts used an incompatible host Python and then an incorrect
object-directory path; these failed before the final verification completed.
Using the frozen Python and discovered object path completed the checks. Neither
generation failed and no model rerun was needed for artifact export.

GPU/namespace execution used approved sandbox escalation; no automatic review
rejected an action. No Computer Use, process inventory, production service
access, commit or push occurred. New files and registry/launcher edits remain
uncommitted. Stage 2 remains incomplete.

Replay with a fresh prefix:

```sh
IMAGINE_PARITY_ROOT=/tmp/imagine-rust-comfy-parity-20261017 \
IMAGINE_REFERENCE_SITE=/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages \
bash inference/tools/check_qwen_portrait_editing.sh \
  /home/thhel/git/ComfyUI /mnt/media/gemensam/AI/ComfyUI/models \
  FRESH_PREFIX \
  /home/thhel/ChatGPT/imagine/inference/profiles/qwen21-editing-portrait-1mpx-int8-seed3061383787.json
```

See the [checkpoint](stage-2-qwen-image21-portrait-seed3061383787-checkpoint.json)
and [retained evidence](stage-2-qwen-image21-portrait-seed3061383787-evidence/).
