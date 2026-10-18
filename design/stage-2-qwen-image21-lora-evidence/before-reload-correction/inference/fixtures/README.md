# Versioned inference regression oracles

`stage-1/` preserves the executed synthetic Stage 1 inputs, expected outputs,
errors, metadata, compiler IR and selected original AOT binaries. It contains no
model weights, production prompts, user images, chat history or production logs.
The retained test log belongs to this isolated synthetic suite.

The manifest pins every uncompressed member, each compressed part, the joined
archive, the reference lock and the final verified checkpoint. Compression parts
are at most 64 MiB. Keep the complete baseline when extending coverage; do not
silently regenerate it against an upgraded ComfyUI installation.

From the repository root:

```bash
python3 -S -B inference/tools/regression_archive.py verify \
  --bundle inference/fixtures/stage-1
python3 -S -B inference/tools/regression_archive.py restore \
  --bundle inference/fixtures/stage-1 --root /path/to/isolated-development-root
```

Restore verifies all data, rejects unsafe archive paths and refuses to replace
a different existing artifact. It does not install dependencies, run a model,
load archived AOT libraries or change the ordinary ComfyUI/Dreamchat environment.
The runtime compiler profile preserves its original paths for provenance;
relocated dependencies require the existing profile preparation tool and a new
explicitly labeled validation run. Relocation is not the old checkpoint.

With the locked dependencies prepared as documented in `inference/README.md`:

```bash
IMAGINE_PARITY_ROOT=/path/to/isolated-development-root \
IMAGINE_REFERENCE_SITE=/path/to/pinned/site-packages \
bash inference/tools/check_saved.sh
```

This runs formatting, warnings-as-errors Clippy, the complete Rust suite and
unfiltered Python-free primitive isolation against the saved data. The broader
`tools/check.sh` also rebuilds native/compiler components and recaptures original
oracles using the selected pinned source. It must not be pointed at a newer
unreviewed reference and presented as validation of the old baseline.

For a future approved reference upgrade:

1. Retain this bundle, its source archives, decision ledger and dependency locks.
2. Inventory the source/dependency delta; map each changed behavior to its Rust
   implementation and regression cases. Keep every unaffected old test.
3. Capture original new-reference cases in a separate isolated root and bundle;
   record reference identities and explicit numerical/algorithmic approvals.
4. Run old and new suites as separately labeled evidence. A changed expected
   value requires source evidence or an explicitly approved departure.
5. Publish a new checkpoint and bundle without overwriting historical evidence.

The capture tool accepts `--checkpoint` for a corresponding approved checkpoint
and verifies its implementation and executed input hashes. The full LibTorch,
CUDA, LLVM/toolchain distributions and build caches are external pinned local
build dependencies, not Git test fixtures or runtime fallbacks.

The Qwen LoRA continuation preserves `qwen-image21-lora.json`,
`qwen-image21-lora-weight.json` and `qwen-image21-lora-lifecycle.json` with their
three `reference_lora*.py` source oracles. They contain bounded initialized
operands, not real Qwen weights. `qwen21-lora/` contains a deterministic tiny
rank-2 text adapter and its generator metadata; trained model adapters remain
external hash-checked local assets. See the [LoRA report](../../design/stage-2-qwen-image21-lora.md)
for passing, failed and diagnostic evidence and replay instructions.
