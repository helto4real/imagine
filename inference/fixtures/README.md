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

`qwen-image21-lora-reload.json` and `reference_lora_reload.py` add three bounded
CPU sequences (FP32, FP16, BF16), each with three initialized 16x256 weights.
They preserve four original load/reload/unload states per sequence, including
resident weights with lazy patches, appended patch functions and signed byte
accounting. No full model constructor or trained weight is used by this oracle.

`qwen-image21-lora-linear.json` and `reference_lora_linear.py` retain twelve
CPU/CUDA and FP32/FP16/BF16 cases with two real linear forwards per case,
including one/two lazy patch passes on both weight and bias. They verify the
original copy-before-functions rule and unchanged stored parameters. The reload
fixture additionally contains a mixed-cost patch/move trace from original code.

## Selected INT8/ConvRot offload regression

`qwen-image21-lora-int8-offload.json` and
`tools/reference_lora_int8_offload.py` preserve original static module-byte
accounting, packed-storage sizes and repeated lazy Linear/SwiGLU
results. All weights are initialized 16×256 synthetic tensors; no full model
constructor is used. The new test runs in the ordinary focused Qwen suite
and its Python-free isolation list. Keep earlier fixtures unchanged when
extending this coverage. The separately requested trained 1600-pixel workload
is documented in [its report](../../design/stage-2-qwen-image21-lora-1600.md).

## Sampler wrapper-buffer lifecycle

`qwen-image21-sampler-buffer.json` preserves six original sequences and 28
transitions from `tools/reference_qwen_sampler_buffer.py`. Only three 16×256
weights and the 10,000-element sigma buffer are constructed. Strict full-size
thresholds, retained buffer placement, cached versus loaded counters and
CPU/GPU/CPU reuse are checked in the normal and Python-free Qwen test entrypoint.
The [continuation report](../../design/stage-2-qwen-image21-sampler-buffer.md)
distinguishes these bounded fixtures from trained-model validation.

## Shared model ownership and static coordinator

`model-management.json` captures 18 original coordinator sequences and 23
policy actions. Its generator imports no Torch and allocates no weights;
reference counts, device capacity and backend responses are explicit inputs.
`qwen-image21-shared-ownership.json` separately retains 22 actual static
clone/LoRA states over BF16 and INT8 ConvRot, plus seven two-model pressure and
reload states. Each model has three initialized 16×256 weights and a small
registered sampler. It never constructs the trained Qwen model.

Regenerate explicitly with `tools/reference_model_management.py` and
`tools/reference_qwen_ownership.py` against the locked source. Ordinary tests
consume the preserved fixtures; the four new cases are also registered in
`tools/check_qwen_image21.sh` for Python-free execution. See the
[ownership report](../../design/stage-2-qwen-image21-ownership.md) for the
source contract, trained validation and remaining coverage.
