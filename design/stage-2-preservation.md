# Qwen Image 2.1 trained T2I baseline

2026-10-18. Tomas requested a preservation commit before continuing the agreed
Qwen completion plan. This baseline builds on Stage 1 commit `059c875`.

## Preserved scope

The independent `inference/` crate, original-source oracle generators,
synthetic fixtures, tokenizer/schema resources, isolated drivers and registered
comparison profiles are preserved with every Stage 2 report/checkpoint and
its durable evidence. The existing Stage 0 locks and Stage 1 archive remain
unchanged and available for future ComfyUI upgrades.

Selected trained text-to-image paths now pass exact original-ComfyUI comparison:

- INT8 ConvRot DiT: 512×512/four steps, 1600×896/four and forty steps, plus
  the separately labeled repeat control after test-storage migration.
- BF16 DiT: 1600×896/forty steps, with original static synchronous CPU/GPU
  memory management owned by Rust.
- Turbo BF16 DiT: 1600×896/eight explicitly specified sigma steps, with the
  same offload binding.

All use the selected INT8 ConvRot Qwen3-VL encoder and BF16 VAE. Each accepted
series has three exact matched pairs. These results do not establish all-BF16
encoder support, complete model-family support or product integration.

The current accepted checkpoint is
[Turbo](stage-2-qwen-image21-turbo-comparison-checkpoint.json), with its
[report](stage-2-qwen-image21-turbo-comparison.md) and
[coverage matrix](stage-2-qwen-image21-coverage.json). Original BF16 OOM and the
excluded initial offload series remain separately documented; their timings
are not accepted controls. Historical checkpoints describe their capture-time
commit status and input hashes; they are not rewritten by this preservation.

No trained weights, build caches, dependency distributions or production
runtime data belong to the commit. The saved images/logs are exclusively this
isolated synthetic test's explicitly authorized evidence. Unrelated Imagine
API/UI, existing backend, native engine, product profile and prototype changes
are excluded. No backend is activated or replaced and no push is authorized.

## Validation reused and checked

The latest Turbo implementation snapshot verifies all 372 recorded
implementation/fixture/resource/profile inputs and 275 original source inputs.
Its four original source locks and 32 CUDA object hashes remain exact. The
37 durable evidence files verify against the accepted checkpoint. All three
Rust isolation reports pass with the same executable, no interpreter or
ComfyUI installation, and an isolated network namespace.

The accepted 18 focused and 18 Python-free component tests are reused for
unchanged bodies, including the subsequent three affected offload retests.
Latest driver changes have passing all-target Clippy, Rust formatting,
Python/shell syntax and the three new trained Turbo pairs. Preservation does
not launch another model run or recapture original fixtures. The bounded
allocation-free metadata oracle remains mandatory; unsafe historical attempts
must never be repeated.

## Authorized continuation order

1. Reference images and editing: original preprocessing, Qwen3-VL vision/image
   spans, ordering/positions and complete editing conditioning, with exact
   synthetic and trained comparisons.
2. LoRA/patch lifecycle: original arithmetic/order, quantization and exact
   restoration across adapter/model changes.
3. Remaining Qwen conditioning/sampling, loader/precision and memory/lifecycle
   contracts from the coverage matrix, with source-faithful integration tests.
4. Product integration follows the design's separate integration milestone.
   Any required public API behavior extension must be made concrete for approval
   before adoption; do not replace the running backend automatically.

The continuation retains exact defined results, the approved scoped exception
ledger, unchanged autotuning and Rust-owned inference/offload. Source-faithful
compatibility work is authorized; stop only for an actual departure from
ComfyUI or the agreed design. A partial path must not be labeled full parity.

## Staged review

The reviewed baseline contains 345 changed/new files: 89 inference files,
21 design/report files and 235 evidence files. No unrelated product paths or
trained model weights are staged. `git diff --cached --check` identifies six
trailing blank lines in preserved historical synthetic test logs (three BF16
offload logs and three first-comparison logs). These exact hashed evidence bytes
are deliberately retained; that whitespace check is not described as passing.
No code-formatting or numerical validation failure is waived.
