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
worker RSS rose 10,929 → 11,931 → 12,227 → 12,512 MiB over four identical runs.
The interleaved runs below did not reproduce it with either build.
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

### Item 5: MOSS peak memory

Following `067f8533`, the MOSS block releases its normed input, q/k/v (including a
fused projection, which lives only in its chunks), the attention output and the
FFN input as soon as each is used; the text blocks release q/k/v and the attention
output before their FFN. Outputs are unchanged (the complete MOSS DiT test is
exact); the peak-memory effect is measured with the candidate worker in Phase 3.

### Item 2: Kitchen 0.2.40 (deferred)

The bridge build toolchain was reassembled in the dev root from verified inputs:
the pinned manylinux image (`comfyui/native-toolchain/manylinux-ca4cca0a`), the
CUDA 13.0 wheels named in the stage-1 evidence (`nvcc`/`crt`/`nvvm` 13.0.88,
`cccl` 13.0.85, `runtime` 13.0.96; headers identical to the verified tree) and the
two pinned PyTorch sources `prepare_kitchen.py` checks. Rebuilt with today's
native sources, the Kitchen 0.2.38 bridge passes the complete suite (503/503),
which validates the toolchain. The Kitchen 0.2.40 bridge (`612fcc11…`) passes all
tests except `group_norm_silu_pad3d`, whose native call gained `stats` and
`row_stride` arguments. Because both workers share the Rust Kitchen bindings, H3
must move to 0.2.40 with the image worker, so an H3 old-vs-new run is needed.
Tomas chose to do MOSS editing (item 1) first; item 2 follows items 6 and 7.

Status 2026-10-28: the work is on branch `kitchen-0240-wip` (`3c1d8e1`), not on
main. Done there and exact against 426f5eea with Kitchen 0.2.40: the fused Flux2
VAE (strips, in-place halving, level buffer; fp16 and bf16), the
`group_norm_silu_pad3d` 0.2.40 contract (`stats`, served rule, row-interleaved
input, overlapping `out`; 1,314 binding cases), `group_norm_stats3d`, eager bodies
for `rms_gated_residual` and the two modulated INT8 projections, and regenerated
signature/contract tables. Against a full 0.2.40 `primitives.json` capture eleven
suite tests still fail: dtype-set order in registry messages differs between the
contract and primitives captures (four tests), Triton cubin hashes carry Kitchen
source times (two), and real 0.2.40 changes not yet ported: the W4A8 layout's
`mma_packed` parameter, new INT8 no-activation rejections, eager
`group_norm_silu_pad3d` argument-binding failures, the structured-metadata corpus
size and the registry capability table. H3 must move with the image worker; a
real H3 video run needs a video profile, whose prompts are private.

### Item 1: MOSS editing — DiT

`tools/reference_moss_dit_edit.py` runs the `426f5eea` MOSS DiT on the same bounded
synthetic models as the previous fixture and adds reference edits
(`fixtures/reference-20261027/moss-dit.json`). Its 72 ordinary forwards, events and
operation counts are identical to the previous reference, so the RoPE rewrite and
the memory changes do not alter outputs. `MossDiT::preprocess_refs` embeds the
reference latents and runs every block at t = 0 (reference i on r = i + 1, f = 0),
recording each block's keys and values; `MossInputs::references` appends them to
every block's attention. `MossEditLayout` selects Moss-Image-Edit's text-start
frames or ai-toolkit's image frame. All 32 edit cases (CPU/CUDA, BF16/INT8,
fused/split q/k/v, both layouts, one and two references) match exactly, including
the precomputed keys/values and lifecycle events.

### Item 1: MOSS editing — product path

The Rust worker accepts up to four MOSS references: it encodes them with the Flux2
VAE after the prompt (centered 16-pixel crop, RGB), counts their tokens once per
condition in the DiT memory estimate as `estimate_memory` does, computes their
keys and values once and gives them to both conditions with the text-start layout.
The `moss-edit` profile takes one to four references and adds the Moss-Image-Edit
LoRA with the author's workflow settings (30 steps, CFG 4, euler/normal, empty
negative prompt); the output follows the first reference's aspect at a 1024 long
side. References use the shared preparation (Lanczos shrink to 1 MP, center crop to
multiples of 32). MOSS's model metadata still reports no references, so the UI does
not offer MOSS editing without the LoRA. The author's workflow also applies a
skip-layer guidance node from its own pack; that is not part of ComfyUI core and is
not ported. Until item 2, the reference encode uses the unfused Flux2 VAE path,
while `426f5eea` encodes through the fused strip path.

Candidate builds: Tomas chose to promote the whole of main. Production's Rust core
is `source_base` `cd0ef53` with nine core files from `b30b480` and three patches;
main already contains all of them (the MOSS LoRA patch's file was later changed by
item 3). A working-copy lock with `source_base`/`core_revision` at the candidate
commit and no core patches builds it; the lock is restored afterwards. Candidates are
frozen as `comfyui/products/rust-candidate-20261027-<commit>`.

Live check on `rust-candidate-20261027-dd90cf7` through the MCP: a MOSS 1024²
image of a woman in a pink dress, then `moss-edit` with the author's example
instruction ("Change only the pink dress to bright red. Keep the character, face,
hair, pose, background and composition unchanged."): the dress turned bright red
with face, pose, accessories, background and composition unchanged, 25.6 s for
30 steps. The first candidate failed because the reference encode reused Qwen's
crop, which pads to four channels; fixed in `dd90cf7`.

### Item 7: MOSS Scale ROPE

`MossInputs` and `MossDenoiserConfig` take ComfyUI's Scale ROPE options
(`MossRope`: `scale_x`, `scale_y`, `shift_t`, `shift_x`, `shift_y`). A row or
column scale `s` sets that axis's theta to `10000 · s^(−d/(d−2))` (NTK-aware); a
zero scale keeps theta and collapses the axis's positions; the shifts move the
image's positions. Text and references keep the trained frequencies and positions,
as in `426f5eea`. MOSS ignores `scale_t`. The defaults reproduce the previous
forward bit for bit (72 forwards and 32 edit cases unchanged); 32 new Scale ROPE
cases (NTK scales; a collapsed, shifted axis; the same with Moss-Image-Edit and
ai-toolkit references) match `426f5eea` exactly. The worker passes the defaults;
no product setting exposes the options yet.

### Item 6: MOSS color/depth control

The adapter is darask0's MOSS-Image color/depth control
(`darask0/MOSS-Image-color-depth-control`, revision `357d7cc1`, Apache-2.0,
`moss-control.safetensors` sha256 `e5c0c3ef…`), stored with its README, training
config and example maps in `models/loras/moss/MOSS-Image-color-depth-control/`. It
replaces `video_embedder` with a `[3072, 384]` weight (noise, color and depth
latents) and adds rank-64 LoRAs.

- Loading (`b751dd4`): `convert_moss_control_lora` turns the widened embedder into
  `set` patches, `calculate_weight` replaces a weight whose shape a `set` changes and
  `calculate_shape` reports it; a lowvram weight whose patches change its shape is
  patched at once, as ModelPatcher does. Fixtures: `moss-control-lora.json`
  (conversion, MOSS key map, set math) and `moss-control-offload.json` (the original
  static clone/manager lifecycle with a widened weight). `reference_lora_weight.py`
  now finds Kitchen's setters by class instead of by line number.
- Sampling: `moss_control_latent` (`MOSSImageControl`), `moss_concat_cond`
  (`MOSSImage.concat_cond`), `moss_control_spans` (the timestep-range split and the
  sampler's float32 gate) and `c_concat` in full-area conditions, concatenated before
  the model's dtype cast. Fixture `moss-control.json`: control latents, concat
  conditions, spans over 49 sigmas including every boundary, and the DiT with a
  widened embedder; all exact. A sigma admitted by no or several spans is reported
  instead of sampled; it occurs only exactly at a boundary.
- Worker: a loaded control adapter without maps gets zero control channels, as
  ComfyUI gives. The product has no input for the maps yet; that needs a decision on
  how a profile takes them.

### Candidate bench (`dd90cf7`)

One sequential run (`comfyui/bench/candidate-dd90cf7-20261027T193907Z`, four runs
per workload, after the control): every image is identical to the control's and
worker GPU peaks are unchanged. Warm medians: `moss-t2i` 31.6 s (−4.6 %),
`moss-sheet` 40.7 s (+1.2 %), `qwen-t2i` 52.7 s (0 %), `qwen-edit` 75.4 s (+2.8 %).

The `qwen-edit` difference was checked with interleaved rounds (control, candidate,
candidate, control, …; server restarted per round, three `qwen-t2i` then three
`qwen-edit` runs; `comfyui/bench/ab-*`, driver
`comfyui/rust-upgrade-20261027/ab/run_ab.sh`). Six rounds completed before Tomas
stopped it as conclusive:

| `qwen-edit` | Rounds | Warm runs (s) | Warm median | Mean of all runs |
| --- | --- | --- | ---: | ---: |
| Control | 1, 4, 5 | 75.3 74.4 · 75.9 74.9 · 75.9 75.9 | 75.6 | 75.5 |
| Candidate | 2, 3, 6 | 74.9 75.4 · 75.8 76.4 · 75.3 75.8 | 75.6 | 75.7 |

`qwen-t2i` warm runs were 52.8–55.8 s with both builds. Images, GPU peaks
(9,512/9,962 MiB) and worker RSS (10.8–11.6 GiB, no growth across runs) match. The
earlier +2.8 % and the control's 73.4 s median were run-to-run variation, not a
regression. The control build cannot start with today's `config` because it
rejects the `moss-edit` profile; the driver parks that profile while a control
server runs.
