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

### Item 2: Kitchen 0.2.40 and the fused Flux2 VAE

The bridge build toolchain was reassembled in the dev root from verified inputs:
the pinned manylinux image (`comfyui/native-toolchain/manylinux-ca4cca0a`), the
CUDA 13.0 wheels named in the stage-1 evidence (`nvcc`/`crt`/`nvvm` 13.0.88,
`cccl` 13.0.85, `runtime` 13.0.96; headers identical to the verified tree) and the
two pinned PyTorch sources `prepare_kitchen.py` checks. Rebuilt with today's
native sources, the Kitchen 0.2.38 bridge passes the complete suite (503/503),
which validates the toolchain. The Kitchen 0.2.40 bridge is `612fcc11…`
(staged as `comfyui/rust-upgrade-20261027/artifacts/612fcc11…/libimagine_kitchen.so`).

- Bindings: `backend_signatures.rs`, `backend_contract/tables.rs` and
  `native_argument_signatures.rs` regenerated from 0.2.40 (60 eager, 60 CUDA and 28
  Triton signatures; new `group_norm_stats3d`, `rms_gated_residual`,
  `int8_linear_modulated`, `int8_linear_rms_modulated`); backend contract fixture
  `fixtures/reference-20261027/kitchen-0240-backend-contract.json`.
  `group_norm_silu_pad3d` follows 0.2.40: per-frame `stats` (eager and CUDA), the new
  served rule (C ≤ 2048 instead of 256 % (C/8)), row-interleaved NHWC input read in
  place (`row_stride`), and an `out` overlapping x written through a copy (Kitchen's
  `_tensors_overlap`); `group_norm_stats3d` is bound for both backends.
  `reference_conv_bindings.py` adds 231 cases (statistics, strips with their frame's
  statistics, interleaved inputs, overlap, stats errors). Eager bodies were added for
  the three new portable references (`rms_gated_residual`, `int8_linear_modulated`,
  `int8_linear_rms_modulated`), with cases in `reference_backend_bindings.py`.
- Corpus: the dev root's `primitives.json` is one full 0.2.40 capture
  (`reference_primitives.py`, 3.5 minutes). The 0.2.38 refreshes the test loader
  layered over the old baseline are dropped; two exceptions stay: the W4A8 frontend
  recapture and `kitchen-0240-triton-source-times.json` (compiler modes and
  environment cache), whose cubins differ from a fresh capture only by the
  reinstalled sources' modification times while every earlier stage is identical.
  Selection errors compare set literals sorted, as the backend contract renders them
  (CPython iterates a set of dtypes in address order). The GEMV capture's trace
  wrapper now passes keyword arguments (Kitchen calls the GEMV with `residual=`), and
  W4A8 layouts carry `mma_packed` (false; Kitchen has had it since 0.2.38).
- Flux2 VAE (d07dff45): on CUDA in fp16/bf16 the encoder and decoder run the fused
  path: GroupNorm statistics per frame, row strips with a one-row halo
  (`STRIP_ELEMS` 2^24), the stride-2 downsample in strips, the in-place C → C/2
  halving with interleaved hidden/output views, and the decoder's one shared level
  buffer. 426f5eea makes fp16 the Flux2 VAE's first working dtype, so the MOSS worker
  now encodes and decodes in fp16, and the decode estimate drops the 4x BatchNorm
  multiplier where the fused path serves. Fixture
  `fixtures/reference-20261027/moss-vae-fused.json` (fp16 and bf16, 2^14 and 2^24
  strip sizes, encode 64×96, decode 4×6 and a batch of two): exact. The old
  446ef4e fixture still passes with the fused path off. MOSS images change by
  design, since ComfyUI now decodes in fp16.
- H3 shares the Rust bindings, so its worker moves to the same bridge. The H3
  component parity tests pass on 0.2.40; the H3 old-versus-new run is below.

The complete suite passes against Kitchen 0.2.40 (511 tests; `b440f58`, merged as
`4f82dee`). Main now needs the dev root's 0.2.40 environment: `env.sh` points at
`kitchen-0240/kitchen-manylinux/libimagine_kitchen.so`, `runtime/primitives.json`
at `primitives-0240-full.json` and `runtime/backend-contract.json` at the 0.2.40
fixture.

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

### Candidate bench (`94d206e`, items 1 and 3–8, Kitchen 0.2.38)

`comfyui/bench/candidate-94d206e-20261027T222312Z`, same order and four runs per
workload as the control. Every image is identical to the control's, worker GPU
peaks are unchanged and all runs are deterministic.

| Workload | Warm median (control) | Warm median (candidate) | Worker RSS peak (control → candidate) |
| --- | ---: | ---: | ---: |
| `moss-t2i` | 33.2 s | 32.2 s | 11,133 → 11,238 MiB |
| `moss-sheet` | 40.2 s | 39.7 s | 11,174 → 11,567 MiB |
| `qwen-t2i` | 52.7 s | 51.3 s | 11,268 → 12,412 MiB |
| `qwen-edit` | 73.4 s | 71.9 s | 12,512 → 13,829 MiB |

Worker RSS again grows across the `qwen-edit` runs when they follow the MOSS and
`qwen-t2i` workloads in one server process (12,204 → 13,829 MiB; the control grew
10,929 → 12,512). The interleaved rounds, which restart the server before Qwen,
showed no growth with either build, so this is an effect of the run order; its
size differs between runs and is not yet explained.

### H3 baseline (Kitchen 0.2.38 bridges, `94d206e`)

Tomas allowed the `normal` video prompt profile for this run. Project "lighthouse
waves" in its own MCP workspace: one 5-second reference-to-video scene (Default
ref2v, 16:9 Draft, no dialogue or music), the storyboard written from the profile's
bundle. Driver and sampler: `comfyui/rust-upgrade-20261027/h3/` (`h3_run.py`,
`runs.jsonl`). Renders with the take's seed are byte-identical
(`cff4dc4f…` MP4 each time).

| Run | Wall time | Worker GPU peak | Worker RSS peak |
| --- | ---: | ---: | ---: |
| First (cold) | 115.8 s | 13,412 MiB | 24,999 MiB |
| Same seed | 88.4 s | 13,316 MiB | 24,980 MiB |
| Same seed | 92.1 s | 13,284 MiB | 24,977 MiB |

The Kitchen 0.2.40 candidate renders the same take seed (`all_same_seeds`) for the
old-versus-new comparison.

### H3 old versus new (Kitchen 0.2.38 `94d206e` and 0.2.40 `72d625d`)

The same take seed renders the byte-identical video on both bridges (`cff4dc4f…`),
with identical worker GPU peaks (13,348 MiB in the interleaved rounds) and RSS
(about 24.4–25.0 GiB). Wall time varies between 88 and 135 s for one build; the
workstation had 14 GiB in swap and a full zram device, so model reads come from
the page cache or the disk run to run. Interleaved rounds (server restarted per
round, two renders each; `h3/ab.sh`):

| Bridge | Renders (s) | Median |
| --- | --- | ---: |
| 0.2.40 | 117.3, 117.7, 115.7, 102.7 | 116.5 |
| 0.2.38 | 117.7, 103.1, 116.8, 134.8 | 117.2 |

Earlier sequential renders: 0.2.40 113.3, 113.2, 108.7 s; 0.2.38 88.4, 92.1,
102.7, 102.6, 118.4 s. No difference is visible beyond this variation.

### Image bench on Kitchen 0.2.40 (`72d625d`)

`comfyui/bench/candidate-72d625d-20261028T062113Z`. Qwen images are identical to the control's; MOSS images change by design
(fp16 fused VAE). Worker GPU peaks are within 10 MiB and device peaks are lower.

| Workload | Control | `94d206e` (0.2.38) | `72d625d` (0.2.40) |
| --- | ---: | ---: | ---: |
| `moss-t2i` warm median | 33.2 s | 32.2 s | 31.7 s |
| `moss-sheet` warm median | 40.2 s | 39.7 s | 40.2 s |
| `qwen-t2i` warm median | 52.7 s | 51.3 s | **72.3 s** |
| `qwen-edit` warm median | 73.4 s | 71.9 s | **94.0 s** |

The Qwen difference is not Kitchen 0.2.40. Profiled on 2026-10-28:

- Kernels: `kitchen_qwen_ops` (own example; INT8 ConvRot linears at the DiT's
  shapes, `rms_rope`, `apply_rope1`, `adaln`, each bridge with its own build,
  alternated twice) times the large calls the same (0.460 vs 0.459 ms at
  5600×3072×3072, 1.80 vs 1.79 ms at 5600×12288×3072); small calls carry about
  10 µs more dispatch, at most about a second per job. Qwen's attention is
  PyTorch's, and the Wan VAE's group norm is faster natively (above).
- Phases: a 1 s timeline of GPU utilization and worker CPU over one job on each
  bridge, three jobs each (`comfyui/rust-upgrade-20261027/profile/timeline.py`):
  both spend about 26 s loading and encoding (text encoder to 11.8 GB, unloaded,
  then the DiT), 42–45 s sampling at 100% and 5 s decoding. Both total 73–76 s.
- Cause: the Qwen worker loads the text encoder and the DiT for every job. Last
  night the weights were still in the page cache and loading took a few seconds
  (51 s jobs); this morning the workstation had 14 GiB in swap and a full zram
  device and loading took about 26 s for either build. The Kitchen 0.2.38 build
  benched today gives `qwen-t2i` 64.3 s and `qwen-edit` 107.7 s warm medians
  (`comfyui/bench/recheck-94d206e-20261028T161853Z`).

A clean old-versus-new image bench needs a quiet machine with free memory, run
interleaved; under today's conditions the two builds are indistinguishable.

## Phase 4: build lock moved (2026-10-28)

Tomas approved moving the product build lock. `native/product/build-lock.json` now
restores the Rust core from main `167bf84` (`source_base` and `core_revision`)
without core patches, since main contains all of them. Image and H3 workers load
the Kitchen 0.2.40 bridge `612fcc11…` from the product artifact store
(`comfyui/h3-main-20261020/artifacts/612fcc11…/libimagine_kitchen.so`), and the
image reference revision is ComfyUI `426f5eea`. Checks: `test_product_build.py`
(15) and `test_restore_native_inputs.py` (5) pass, the start scripts parse, and
`./build-product.sh` prepares server, UI and proxy (release `server-dq9064jo`,
runtime naming only the new bridge). The system compiler and `libcuda` differences
it reports were already present and are recorded, not blocking. Services were not
restarted by this step.
