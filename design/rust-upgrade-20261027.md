# Rust backend upgrade to ComfyUI v0.43.0 and Kitchen 0.2.40

Approved by Tomas on 2026-10-27: port the selected upstream changes made since the
current reference (ComfyUI `446ef4e`, Kitchen 0.2.38) into the Rust backend, test
a candidate worker beside production, and move the product build lock only after
it passes. The new target is ComfyUI `426f5eea` (v0.43.0) with Kitchen 0.2.40.

| # | Change | Upstream |
| --- | --- | --- |
| 1 | MOSS image editing with references (Moss-Image-Edit format first) | `60aade0b`, `9fe7ef76` |
| 2 | Flux2 VAE fused strip decode; Kitchen library rebuilt from 0.2.40 | `d07dff45`, Kitchen `881ac9f` |
| 3 | MOSS LoRA keys without the `diffusion_model.` prefix, fused q/k/v slices | `60aade0b` |
| 4 | Qwen prefix-cache key includes the batch size | `503d785f` |
| 5 | MOSS lower peak memory | `067f8533` |
| 6 | MOSS color/depth control adapter | `e3bee7ff` |
| 7 | MOSS NTK RoPE scaling | `b9b290e4` |
| 8 | ConvRot INT8 shared-memory estimate matches the kernel's launch rule | Kitchen `4f28c98` |

ai-toolkit edit formats need a Rust Qwen3.5 vision encoder and come later.
ComfyUI reference values come from ComfyUI's Python code run directly, never a
live ComfyUI server.

Acceptance limits: paths without an intended change give identical pixels, are at
most about 2% slower and use no more peak VRAM; changed paths are pixel-exact with
the new ComfyUI.

## Phase 0: frozen control and baseline

The production server release `server-t7x112x1` (main `53db93e`, build-lock
`core_revision` `b30b480`, reference `446ef4e`) is preserved as the package
`comfyui/products/rust-control-20261027-53db93e`. The launcher's own package check
verifies it, and `./start-server.sh --product-package <that path>` runs it, so the
old build stays available for comparisons and rollback after promotion.

The edit LoRA is `loras/moss/Moss-Image-Edit/adapter_model.safetensors` from
[chinoll/Moss-Image-Edit](https://huggingface.co/chinoll/Moss-Image-Edit), SHA-256
`17a48b8c605dcffdedb92a2b887d18cf316505be90802717669fb6556098dcdf`. Its project
code is GPLv3; the weights state no licence.

`tools/rust_bench.py` drives fixed synthetic workloads with fixed seeds through the
proxy MCP and records client wall time, peak GPU memory (device and worker
processes, sampled by `nvidia-smi`), peak worker RSS and output hashes. Raw runs
and images stay in `comfyui/bench/` (ignored). It reads only its own jobs.

Control baseline, 2026-10-27, RTX 4070 Ti SUPER, control package served by the
running stack, 1 cold and 3 warm runs per workload
(`comfyui/bench/control-20261027T155021Z`):

| Workload | Cold s | Warm median s | Worker GPU peak MiB | Device GPU peak MiB | Worker RSS peak MiB | Output SHA-256 (all runs) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `moss-t2i` 1024×1024 | 34.2* | 33.2 | 9,954 | 12,568 | 11,133 | `dcd97a33f5a2…` |
| `moss-sheet` 1536×864 | 41.2 | 40.2 | 9,972 | 12,822 | 11,174 | `a567409951c3…` |
| `qwen-t2i` | 68.8 | 52.7 | 9,514 | 12,382 | 11,268 | `2508f027cb1a…` |
| `qwen-edit`, one reference | 73.9 | 73.4 | 9,962 | 12,642 | 12,512 | `fe53fbe59120…` |

\* MOSS was already resident from a preceding smoke run.

Every workload was deterministic across its runs. Open observation: `qwen-edit`
worker RSS rose 10,929 → 11,931 → 12,227 → 12,512 MiB over four identical runs;
check whether the candidate behaves the same (reference host cache or growth).
H3 is not in this baseline; it shares the Kitchen library, so an H3 baseline is
recorded before item 2.

## Phase 1: new reference

The main checkout `/home/thhel/git/ComfyUI` was fast-forwarded to `426f5eea` and
`pip install -r requirements.txt` ran in its `.venv`. That changed only the
frontend (1.57.4), workflow templates (0.11.97) and embedded docs (0.5.15);
Torch 2.13.0, Kitchen 0.2.40 and comfy-aimdo 0.6.0 were already installed.
`pip check` reports conflicts only in packages outside ComfyUI's requirements
(`inference-gpu`, `inference-models`, `onnx`), which the install did not touch.

[`reference-20261027`](reference-20261027/README.md) locks the new reference with
copies of the previous capture tools: 45,357 source fingerprints, the static
dependency closure, native linkage (only Kitchen and Aimdo objects changed) and
the source acquisition lock with Kitchen `v0.2.40` and Aimdo `v0.6.0` checkouts
under `comfyui/rust-upgrade-20261027/sources`. Both verify. `reference-current.json`
now selects it for development tools; product builds do not read it and still
pin the Rust core and Kitchen 0.2.38 bridge. The previous reference and its
expected results stay unchanged in `reference-20261018`.

Recapturing expected values happens per item in Phase 2, from ComfyUI's Python
code run directly: unchanged functions keep their existing fixtures.

## Phase 2: ports

### Development root

`comfyui/rust-upgrade-20261027` is the isolated root for new-reference captures
and the Python-free suite; `source comfyui/rust-upgrade-20261027/env.sh` sets the
same variables as `inference/tools/check_saved.sh` plus the Aimdo and Pillow
libraries. Several inputs of the previous root `comfyui/moss-20261018` had been
links into `/tmp` and are gone (Cargo home, Triton and INT8 AOT probes, Triton
standalone cubins). The new root restores them from the verified `stage-1` bundle
(19,204 files), takes copy-on-write copies of the product Cargo home and the old
build cache, links the intact native inputs (CUDA headers, Kitchen 0.2.38 bridge,
Triton driver, Aimdo, Pillow) and uses the relocated Triton compiler profile from
`moss-20261018` (the bundle's copy keeps its original `/tmp` paths). Fixtures from
the new reference go to `inference/fixtures/reference-20261027/`; the previous
fixtures stay unchanged.

### Item 4: Qwen prefix-cache key

`tools/reference_qwen_image21_core.py` run against `426f5eea` changes only the
prefix-cache key in its 192 cases: in all 24 sequence cases the new key is the old
key with the batch size (2) prepended. The other outputs, including every INT8
projection under Kitchen 0.2.40, and the dispatch traces are identical. Rust's
`prefix_cache_key` now puts the batch size first; the component test checks it
exactly against the new fixture and, with the batch column removed, against the
previous one. All 46 Qwen Image 2.1 tests pass.

The previous root's fixtures under `runtime/` had drifted: its `primitives.json`
is the Kitchen 0.2.38 merge (the bundle's copy predates it) and the bundle's
`backend-contract.json` predates `fixtures/kitchen-0238-backend-contract.json`.
The new root links the upgraded `primitives.json` from `moss-20261018` and the
versioned 0.2.38 backend contract. Two H3 examples no longer compiled against the
current H3 API (`SecondStage` upscale/model fields, motion `capture` and `Window`
arguments), which stopped `cargo test` before any test ran; both are updated.

Triton cubins embed the modification time of the Kitchen source they compile, in
their DWARF line table (`ptxas -lineinfo` reads it from the source path). Installing
Kitchen 0.2.40 on 2026-10-24 rewrote unchanged Triton sources with new times, so
cubin hashes captured earlier no longer match although every stage up to PTX is
byte-identical and the pinned `ptxas` is unchanged. The original Triton run today
produces exactly the Rust compiler's cubins. Recaptured from the new reference:
the W4A8 frontend (`fixtures/reference-20261027/kitchen-0240-w4a8-frontend.json`,
differing only in four cubin hashes and the recorded source time) and the INT8 AOT
probe (`build_int8_aot.py`, 225 cubins, no candidate mismatch).

### Item 8: ConvRot INT8 shared-memory estimate

`convrot::fused_shared_memory_bytes` copied Kitchen 0.2.38's Python estimate, which
did not follow the kernel launcher's block-size rule; Kitchen 0.2.39 corrected the
Python copy. Rust now follows the launcher rule, checked against
`tools/reference_kitchen_convrot_shared_memory.py`, which evaluates Kitchen 0.2.40's
own function over 75 row/width cases. On the RTX 4070 Ti SUPER the old and new
estimates select the same kernels.

Approved by Tomas: cubin comparisons ignore the source-time field. The test helper
`tests/support/cubin_source_times.rs` writes the times a capture compiled with into
the line table of a freshly compiled cubin (same field width) and then compares the
hash exactly, so later Kitchen reinstalls no longer break the compiler-mode,
environment, NA3D and W4A8 tests. The stage-1 capture times come from the bundle's
recorded `compiler_source_files` (`fixtures/reference-20261027/triton-capture-source-times.json`),
W4A8's from its own capture. With this and the dev-root fixes the complete inference
suite passes: 502 tests, 0 failures, 14 ignored.

### Item 3: MOSS LoRA keys

`tools/reference_moss_lora_keys.py` runs `426f5eea`'s `model_lora_keys_unet` on
shape-only split and fused q/k/v layouts (`fixtures/reference-20261027/moss-lora-keys.json`).
`moss_key_map` now also maps bare module paths (`layers.0.proj.o`) and bare names of
fused q/k/v slices (`layers.0.proj.q`); the test compares the whole alias map,
including order, with dict semantics. Our checkpoint uses fused q/k/v, and every
module of the Moss-Image-Edit LoRA (bare `layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}`)
now resolves; previously none of its names had a mapping.
