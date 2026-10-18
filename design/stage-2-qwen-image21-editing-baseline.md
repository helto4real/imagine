# Accepted Qwen Image 2.1 reference-editing development baseline

Accepted by Tomas on **2026-10-18** after reviewing the new seed comparison:
“Jag har varit med om detta förr, så bra, då kan vi godkänna detta. Gör en
commit detta är en bra ny baslinje”. This document and its supporting source,
tests and evidence are preserved together in the commit containing this file.
The earlier T2I baseline `de955e5` remains intact.

## Frozen selected workload

| Setting | Accepted value |
| --- | --- |
| Profile | `inference/profiles/qwen21-editing-portrait-1mpx-int8-seed3061383787.json` |
| Reference | Preserved generated 1024×1024 adult portrait, no resize/crop |
| Output | 1024×1024 RGBA, batch 1 |
| DiT / text encoder | Qwen Image 2.1 / Qwen3-VL 8B, both INT8 ConvRot |
| VAE / compute | Qwen Image 2.1 BF16 / BF16 |
| Sampling | 40 steps, Euler, simple, CFG 1, shift 0.69 |
| Seed | **3061383787**, drawn once then frozen |
| Conditioning | Frozen garden-edit prompt, empty negative, no LoRA, reference vision + VAE |
| Attention / autotuning | Original PyTorch startup path / unchanged |
| Component memory | Original static policy, full load per component, unload between components |
| Prefix cache | Original auto policy; observed 2,176,319,488 pinned CPU bytes and two GPU staging buffers |

The profile pins checkpoint names and the reference tensor hash. Full model
hashes, original source locks, executable/dependency identity and namespace
proof are retained in the evidence. Model weights and compiled dependencies
remain external local assets; Git does not distribute them.

## Accepted evidence and limits

The [seed-only report](stage-2-qwen-image21-portrait-seed3061383787.md) and
[checkpoint](stage-2-qwen-image21-portrait-seed3061383787-checkpoint.json)
preserve one fresh-process matched pair: all six tensor boundaries are finite
and byte-exact, conditioning metadata matches and decoded PNG pixels match.
The new image has substantially more natural skin/hair than seed 17. The
previous [seed-17 result](stage-2-qwen-image21-portrait-1mpx.md) remains a
historical control; only seed changed, and all four conditioning/reference
tensor hashes are identical across seeds. No seed-handling bug or general
quality guarantee is established.

Rust pipeline time was 31.543 s and ComfyUI Python 32.383 s; sampling was
23.506 / 23.782 s. These existing pipeline boundaries include component
unloads/intermediate exports and exclude launcher/build/final PNG export.
One pair is not an established performance advantage.

Preserved implementation includes vision/DeepStack and multimodal conditioning,
original Pillow C source/adapter/license, VAE reference handling and selected
static host-cache budget/pinning/staging. The saved five new fixture files,
source oracles, Rust regression tests and isolated launchers remain available
for future ComfyUI upgrades. Bounded constructors avoid allocating real model
weights in the synthetic oracle tests.

Validation evidence consists of the earlier 25 focused Qwen regressions,
24 historical standalone isolations plus the new standalone host-cache test,
and the fresh trained seed pair in Python-free Rust isolation. Locked/offline
build, formatting, all-target warnings-as-errors Clippy, Python AST and shell
checks passed on the seed-pair source. Only documentation changes follow that
validation in this preservation commit; no new model run is required solely
to record acceptance.

All 173 retained evidence-file hashes verify, as do the reference input and
original Pillow sources. Eight raw historical test logs contain final blank
lines flagged by the whole staged whitespace check; those byte-pinned logs
are preserved unchanged. Source/documentation/script whitespace checks pass.

Stage 2 and product integration remain incomplete. General model-pin eviction,
broader precision/editing/CFG coverage, OOM/tiling and LoRA remain separate work.
No runtime service, production engine defaults, frozen native C++ experiment or
autotuning policy is changed by this baseline. No parity exception was adopted.

## Reuse and upgrades

Run the replay command in the seed-only report with a fresh output prefix and
the matching local dependencies/models. Ordinary tests do not generate images.
Retain this commit, the seed-17 control and the earlier T2I baseline before
upgrading ComfyUI. Follow [the regression-bundle upgrade procedure](../inference/fixtures/README.md):
inventory the source delta, extend the preserved source/fixture tests, record
new-reference evidence separately and require explicit approval for departures
from the source/design. Do not overwrite old expected results to make a new
reference pass.

Earlier reports/checkpoints describe their state at capture time, including
their then-uncommitted status. This accepted baseline supersedes those Git
status notes without rewriting historical evidence or numerical results.
