# Architecture adapters and checkpoint admission

## Boundaries

`shared/model_spec.rs` is the product registry shared by the server and image
worker. It describes architecture, representation, QKV layout, execution
profiles, component requirements and legacy aliases. It has no storage, queue
or tensor dependency. Checkpoint file IDs and component associations remain
owned by the catalog. Profiles are saved per file ID in `execution_profiles`.

Diffusion discovery reads bounded safetensors headers, regardless of filename
or extension. The first version admits the existing 32-block Qwen Image 2.1 and
42-block MOSS configurations. It distinguishes unknown architecture, unsupported
configuration, unsupported representation and invalid QKV layout. Header admission
is not full validation: component loading still validates all required weights
and quantization metadata. No checkpoint or LoRA file is rewritten.

The legacy Turbo filename is only a migration hint for a compatible Qwen BF16
checkpoint. Other new checkpoints default to standard. Explicit sampling values
are retained. The optional profile selector is displayed only when the checkpoint
has more than one implemented profile. Existing model IDs remain API aliases;
MOSS adds `moss-image-sft-bf16`, without registrations for individual fine-tunes.

`inference/src/components` owns explicit schemas, inventory, buffers, static
ownership, pinning and offload mechanics. Model modules retain their own schemas,
sampling buffers, normalization, memory formulas and orchestration. Qwen exports
compatibility aliases. H3 keeps its separate terminal release path, propagating
unpin failures without copying doomed weights to CPU. Its dynamic encoder and
memory budgets are unchanged.

`patching/binding.rs` shares LoRA reading, parser calls, component binding and
internal reports. Qwen retains its mapping and repeated parser calls for text
and DiT, in the original order. Reports merge consumed/unmatched keys and parsed/
accepted targets across the permitted components. Unmatched or entirely unmatched
files are not new hard errors; existing parser/patch errors still apply. No new
logging or public diagnostic channel was introduced.

MOSS owns its generic and projection aliases. Fused QKV binds split Q/K/V adapters
to logical row ranges; separate projections bind directly. BF16 and INT8 use the
same binding and existing patch arithmetic. Separate QKV changes constructor and
inventory metadata only; tensors are neither split nor fused during loading.
MOSS uses diffusion LoRAs only, with the existing Qwen3.5 INT8 encoder, Flux.2 VAE,
sampling defaults and preview projection.

## Locked delivery

The existing core patch is applied first, followed by:

1. `native/product/patches/components.patch`: mechanical component extraction.
2. `native/product/patches/model-lora.patch`: shared LoRA binding and MOSS layouts.

Both hashes are in `native/product/build-lock.json`. The only added product
allowlist entry is `shared/model_spec.rs`; generic mechanics remain in the core.
The patches are based on restored locked source, not the current main core.
In particular, unrelated later module declarations in main's `inference/src/lib.rs`
are not brought into the lock. The locked lib gains only the components export.
Product overlays that intersect any patched path are rejected before restoration.
Image protocol 1 and H3 protocol 16 are unchanged.

The refactor changes structure only. There are no changed kernels, arithmetic,
cast boundaries, quantization algorithms, RNG, offload decisions, memory budgets,
streams or synchronization calls. New admission, profiles, MOSS BF16 and MOSS
LoRA are separate feature additions. Diagnostic strings inside extracted mechanics
retain their legacy names to preserve error behavior.

## Verification

Initial verification used source-owned synthetic fixtures and tiny tensors.
`tools/test_model_adapter_extraction.py` compares original function bodies across
the moved modules and guards H3 terminal release. Existing native primitive tests
check exact tensor bytes, quantized payloads, shapes, strides, patch order,
restoration, lifecycle and memory decisions against the same locked fixtures.
No reference tolerance was changed.

CPU checks include server contracts/catalog/profile persistence, API lifecycle,
synthetic worker transport for MOSS INT8/BF16 with LoRAs, headless UI model/LoRA
selection, the dedicated `model_adapters_cpu` target and H3 product regressions.
Build checks cover source restoration, patch hashes, product/core collision
rejection and `./build-product.sh` without service startup.

Synthetic CUDA uses the existing `inference/tools/gpu-reservation` runner and
locked image libraries. Run only the selected offload/model-management/LoRA/MOSS
primitive filters and the dedicated `model_adapters_cuda` target, single-threaded;
do not run the whole primitive suite. The dedicated target covers MOSS BF16 QKV
row isolation and the actual H3 terminal release function with tiny storage.

The initial checks did not cover trained inference. The subsequent authorized
GPU acceptance below used controlled synthetic inputs and existing model files;
application runtime history and existing generated media were not used. Image
quality, peak VRAM and individual community checkpoints remain unverified. No
installation, CUDA library rebuild, stable-diffusion.cpp change or service restart
was performed.

### Recorded acceptance — 2026-10-24

| Check | Result |
| --- | --- |
| Server library | 161 passed, one pre-existing live-model test ignored; final catalog slice 15 passed, including the added filename-hint regression |
| API lifecycle / synthetic worker transport | 29 / 5 passed |
| Headless UI model and LoRA selection | 15 passed |
| Dedicated native CPU binding / H3 product / loader diagnostics | 4 / 17 / 1 passed; two explicit H3 GPU/oracle tests ignored |
| Frozen baseline selected primitive cases | 15 passed |
| Final restored locked core, same selected cases and exact-byte fixtures | 15 passed |
| Final restored core, dedicated synthetic CUDA | 2 passed |
| Component extraction guards / product-build regressions | 2 / 13 passed |
| Exact restoration | 17 affected core files and 40 product files matched |
| `./build-product.sh` | Server, image/H3 workers, UI and proxy built successfully; no services started |
| Rust formatting / `git diff --check` / patch hashes | Passed |

One final CUDA attempt hit the reservation deadline; its manifest confirmed the
child was reaped before the finish message. The final run with a larger reservation
time margin passed unchanged numerical assertions. Existing PyTorch pinning API
deprecation warnings occurred in baseline and candidate checks.

Test escalation was used for the repository-mandated synthetic fixture directory,
local test sockets and the coordinated synthetic CUDA executions. No private
runtime sources were inspected.

### Trained GPU acceptance — 2026-10-24

The user approved the refactoring after 33 controlled trained-model runs. All
before/after comparisons were bitidentical for Qwen INT8 with and without LoRA,
Qwen BF16 (1600×896, 40 steps, actual offload), Qwen Turbo BF16, MOSS SFT INT8
and H3 video/audio. Baseline repeats were also bitidentical. Registered placement,
lifecycle and phase/step decisions matched; this was not a peak-VRAM measurement.

Twenty of 21 checks passed. The extra assumption that a zero-strength MOSS LoRA
equals no LoRA failed: the existing low-level INT8 patch path still requantizes
weights. Additional Qwen runs confirmed that the old implementation already
behaves this way and that refactoring preserves its output exactly. This is a
separate product behavior issue, not an accepted numerical refactoring difference.

MOSS BF16 trained inference remains unverified because the checkpoint was absent.
MOSS INT8 with synthetic Q/K/V LoRA and an unmatched key completed successfully;
nonzero strength affected the output and repeated runs were bitidentical.

The local, removable evidence folder is `target/model-adapters-gpu-20261024/`,
including `REPORT.md`, byte comparisons, source hashes and reservation receipts.
Its outputs are not tracked. All GPU children were reaped before reservation
release. These finite cases do not establish parity for every input or checkpoint.
