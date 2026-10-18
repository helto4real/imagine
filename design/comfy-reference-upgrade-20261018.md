# ComfyUI reference upgrade and MOSS-Image

Status: the reference upgrade and selected MOSS text-to-image path are
implemented and validated. All 60 selected Python-free regressions pass.
The trained MOSS 1024×1024/50-step control matches the original numerically
and in observed component-memory policy. Full upgraded ComfyUI/Kitchen
parity remains incomplete; Qwen automatic BF16/Turbo comparisons retain the
memory-placement limitation described below.

## Approved change of reference

Tomas approved preserving the existing baseline, updating his main local
ComfyUI checkout and its requirements, locking the resulting reference, adapting
the Rust engine and validating Qwen before adding MOSS-Image. The reference
upgrade is intentional; source-compatible implementation needs no further
approval. A deviation from the source or agreed design still requires approval.
Inference remains local and Python-free in Rust. Python is only the development
oracle. No graph/UI or Dreamchat integration changes belong to this work.

The preserved code baseline is `c0d291c472b9cd0c20226f5dcf0e870be0e98cc2`.
Its previous ComfyUI revision was
`ba994879b88d0ebd9950b1e5c1e923d928f216ba`; the new development target is
`446ef4e005fc7134d53ce6869ebd77d2134cd39c`.

## Preservation and installation evidence

- Old source and an independent copied environment are retained under
  `comfyui/reference-ba994879-20261018/ComfyUI` (ignored local build storage).
  All 45,339 old source/header fingerprints and 779 distribution ELF hashes
  were checked against the old locks. The old `design/reference-local` locks
  and previous runtime/compiler artifacts were retained.
- `inference/tools/with_frozen_reference.sh` mounts that preserved checkout and
  environment at the original canonical path inside a network-isolated
  namespace. This preserves filenames, hashes and native library paths. An old
  execution-context regression passed through this wrapper after the upgrade.
- The main `/home/thhel/git/ComfyUI` checkout was fast-forwarded to the exact
  new revision and `pip install -r requirements.txt` completed in its existing
  `.venv`. Torch remains 2.13.0+cu130; Kitchen changes 0.2.37 to 0.2.38.
- `pip check` reports the same 16 pre-existing third-party conflicts before
  and after installation. There are no additional conflicts. Unrelated
  packages were not repaired as part of this upgrade.
- The new `design/reference-20261018` inventory records 45,345 source/assets,
  984 node contracts, 107 model classes and 779 distribution ELF objects.
  Node/UI contracts are inventory evidence, not Rust inference scope.
- `design/reference-current.json` explicitly selects the development target.
  The path helper supports a deliberate old-reference override. The initial
  edits to 110 development tools changed reference paths only; AST comparison
  verified equivalence after restoring those paths and removing the helper
  import. Subsequent oracle adaptations are documented separately below.

Machine-readable installation, preservation and path-edit evidence lives in
`comfyui/moss-20261018/`. Those files are development artifacts, not production
runtime data. No ordinary Dreamchat/ComfyUI logs, history, submitted prompts or
media were opened. An earlier bounded original CUDA check exhausted available
VRAM; Torch automatically included neighbouring GPU process metadata in its
exception. No process list was requested and no other process was controlled.
That diagnostic log is excluded from the preserved validation evidence.

## Source changes affecting the selected path

The Qwen DiT remains unchanged. Shared operations now dispatch fused input
activation through quantized layout methods. Kitchen adds SiTU activation code
4, a fused ConvRot M=1 quantizer/GEMV with residual operands, and additional
optional packed/decode operations. The new Python-only registration declaration
is removed by the existing metadata adapter; kernel bodies remain unchanged.
The ndarray metadata adapter adds the exact signed INT8 DLPack type mapping.

Qwen3-VL no longer appends an empty think block when a caller supplies
`thinking=False`. Rust follows this change; the field is retained as an ignored
caller option. Oracle loaders now select the new original cache class names and
the new original layout-dispatch helper. These changes are not approved numeric
tolerances or alternative algorithms.

MOSS requires a separate 42-block DiT, Qwen3.5-9B hybrid attention/DeltaNet
conditioning and the Flux.2 VAE. The existing Qwen3-VL/Wan VAE must not be used
as substitutes. Image conditioning uses source prefill; the newly added
autoregressive decode APIs are not exercised by that path. Full new Kitchen
operation parity is a separate remaining matrix, not implied by a MOSS run.

## Local model selection

The initial experimental target is the locally available SFT INT8 ConvRot DiT
with Qwen3.5-9B INT8 ConvRot text weights and the full Flux.2 VAE. The initial
preflight read only headers and allocated no model-weight tensors. Subsequent
explicit trained comparisons loaded these same selected assets.

| Asset under `/mnt/media/gemensam/AI/ComfyUI/models` | Bytes | Header SHA256 |
| --- | ---: | --- |
| `diffusion_models/moss/moss_image_sft_int8_convrot.safetensors` | 5,413,698,958 | `b823edaa8fbfa67d5594a793a0212e8081922b2e59f7dd25a1a217527f0c9192` |
| `text_encoders/moss/qwen3.5_9b_int8_convrot.safetensors` | 9,748,121,532 | `e2a53fa385cb498aeed4860c090b3e0f343d06d8dd58a657417c68cc45ee181a` |
| `vae/flux.2/flux2-vae.safetensors` | 336,213,556 | `87f9d471e9e5e7cc57f56c43f6901c7d36da62ff88264f46395995b69dd0fe44` |

Header hashes are not full checkpoint hashes. No weight download is needed for
this initial selection. The source model card is
<https://huggingface.co/Comfy-Org/MOSS-Image>.

## Acceptance boundary

The evidence below was produced against the new locked reference. Earlier
accepted Qwen evidence remains attached to the preserved old reference.
Selected trained MOSS text-to-image is validated; this does not establish
all MOSS variants, general product integration or complete upgraded parity.

## Completed bounded validation

The new native Kitchen bridge builds successfully with the original CUDA
sources and has no undefined Python symbols. The positive-K INT8 activation
matrix passes 1,386 cases, including SiTU, the fused ConvRot M=1 route and
GEMV residual arguments. No numerical tolerance was introduced.

The first upgraded Qwen regression run passed all 46 focused tests. Fresh
original-source language captures (53 conditioning calls and 30 core forwards)
and complete DiT captures (209 forwards) retained the prior exact results.
Only 18 explicit `thinking=False` tokenizer cases changed, exactly matching
the removed source suffix; the Unicode oracle was unchanged. Subsequent shared
tokenizer/buffer/VAE-policy additions passed the final isolation run as well.

Six initial MOSS Rust component tests pass with exact bytes and source lifecycle
checks: 72 full bounded DiT forwards; 54 DeltaNet cases; 48 hybrid Qwen3.5 core
forwards; 595 tokenizer cases; 28 complete SDClip conditioning calls; and 18
Flux.2 VAE forwards. The VAE check also preserves the original constructor
gradient flags separately from detached `state_dict` snapshots, and keeps
`bn.num_batches_tracked` as INT64. Source/Rust CPU thread inputs are both two;
this is the established controlled-fixture configuration, not an autotuning
change. Trained validation is recorded separately below.

Full constructor inventories use meta tensors exclusively: MOSS DiT has 378
parameters, the initial Qwen3.5 wrapper without MTP 760 parameters/one buffer, and the
Flux.2 VAE 248 parameters/three buffers. The selected model headers match the
fused DiT, unmerged language MLP, and Flux.2 quant/post-quant convolution paths.
No real full-model weights were allocated for these constructor tests.

## Follow-up validation and diagnostic decisions

The bounded MOSS set now also covers 48 discrete-flow schedules, 18 DiT and
24 VAE memory estimates, 36 injected VAE attention failure/retry cases, and
four Diffusers VAE conversion groups. These use initialized small operands or
meta constructors, never intentional exhaustion of RAM/VRAM. Static MOSS DiT
and text execution bindings reproduce the original component outputs too.
The selected Flux.2 checkpoint is in Diffusers layout; its key and attention
weight-view conversion follows `sd.VAE` and `diffusers_convert` exactly.
The complete selected header set passes the Python-free MOSS preflight.

An AIMDO development-preparation bug was found: every environment preparation
used `copyfile` on the same ELF inode, which could truncate a library already
mapped by a concurrent development run. Preparation now retains an identical
file and uses a verified temporary file plus atomic replacement on refresh.
The regression verifies unchanged inode/mtime and intact old mappings after
replacement. This changes publication safety only, not AIMDO or inference code.
Earlier CUDA errors, process exit 139 and an unsuccessful remote GDB transport
attempt remain diagnostic evidence; none are counted as successful checks.

Against the new reference, the registered trained 320×224 four-step INT8
control passes exact context, negative context, reference preprocessing/latent,
schedule, output latent, Float32 pixels and decoded PNG pixels. All observed
component-memory fields and sampler-buffer lifecycle also match. Its measured
first-process pipeline walls are 24.382 s Rust and 17.402 s Python; this small
control includes loading/export and is not a steady-state benchmark.

BF16 and Turbo both complete after fixing publication. Their automatic runs
have exact text/reference inputs but different DiT leaf residency and unequal
latent/pixel results. Replaying the *recorded original DiT placements*, through
the already existing explicit diagnostic API, makes both numerical comparisons
exact. Their resident/lazy LoRA patch routes therefore cannot be compared as
equal arithmetic inputs when physical memory observations differ. The failed
automatic comparisons are retained. Diagnostic replay is labelled separately;
it does not replace automatic memory services, change defaults, relax numeric
checks or establish identical automatic placement under unequal observations.

Fresh Qwen meta constructors retain identical complete parameter/buffer/module
metadata; only source provenance is refreshed. A text-only metadata oracle
intended for an unquantized BF16 encoder was also tried on the INT8 encoder in
these profiles. That input is outside that oracle's model; its mismatch is not
accepted as policy evidence and its guard/expectations were not weakened.

The initial trained MOSS comparison uses a deliberately bounded 320×224,
four-step Euler/normal control, CFG 3.5, source shift 5 and multiplier 1. Text
storage is BF16/INT8 with original Float32 computation; DiT and VAE computation
are BF16. This is a parity control, not an image-quality preset. The initial
selection has no LoRA, image conditioning or autoregressive decoding. Those
branches are not claimed accepted by a text-to-image control.

The first selected text-weight load additionally exposed `mtp.*` tensors.
ComfyUI detects these and constructs `MOSSImageTEModel(mtp=True)`. The selected
meta inventory is consequently regenerated with the original MTP constructor:
775 parameters (15 auxiliary MTP parameters), plus the same vision rotary
buffer. These weights are loaded, owned and counted; they are not discarded
because prefill leaves them unused. MTP decoding is still not a claimed API.
Checkpoint conversion preserves original dictionary order, including unequal
alias collisions; tests verify both converted bytes and retained storage.

Full checkpoint hashes use the same SHA256 algorithm as the Python oracle.
Only the SHA256 crate is optimized in development builds to avoid spending
minutes hashing multi-gigabyte assets in unoptimized scalar Rust code. Tensor
code, native kernels, numerical policy and autotuning flags are unchanged.

Sandbox escalation was granted for the main checkout update and isolated GPU
namespaces. No automatic approval review rejected an action. No other services
or processes were stopped. The privacy boundary and automatically emitted
Torch exception metadata are described above.

The selected DiT also contains serialized `ComfyAttention` preferences for
Kitchen INT8 attention. PyTorch attention is the global fallback, not an
override of these checkpoint preferences. Source ComfyAttention consumes the
JSON tensor into decoded attributes: Module.to does not move it, while
state_dict serializes fresh default-JSON bytes that contribute to cached model
size. The common prepared-parameter/static-owner path now preserves this
distinction. Seven original CPU cases validate JSON byte accounting (including
Unicode, lists, floats and compact input), ordinary buffers/parameters and
owner snapshots. `rope.bands` is removed only because the original MOSS
`process_unet_state_dict` explicitly removes that legacy field.

The first complete trained MOSS control passes all numerical boundaries,
including every denoised step and decoded PNG pixels. Full checkpoint SHA256:
DiT `b4d1475bc7eeffd7b809e82ea13a1fba1fb85c4e18365af3813722faa5c0fe4f`,
text `32e3787472f708fa5f83624acbf5ef23f7fc89c439e05065ea57cfe5a97f025a`,
VAE `d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5`.
Its pipeline walls are 22.694 s Rust / 11.566 s Python, sampling 1.238 s /
0.483 s. These are first-process development measurements with hash checks,
loading and exports, not a steady-state speed claim. Model sizes and loaded
bytes match for all components. The initial metadata export labels resident
forced-cast flags as lazy passes on the Python side; the next export separates
actual lazy passes from explicit `cast_modules`, using the established Qwen
component comparison convention. The earlier export/result remains preserved.

A separate 1024×1024 example uses 50 Euler/normal steps and CFG 3. These two
settings follow the creator's [sampling table](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT#sampling-settings).
Optional negative-layer skipping stays disabled in both engines, as in the
selected ComfyUI baseline; the same full Flux.2 VAE and BF16 computation remain.
This is a separate image workload, not a change to the small control or to
native defaults. It is not acceptance of the creator's separate Diffusers
pipeline, small decoder or optional layer-skipping configuration.

## Final selected validation

`check_reference_upgrade.sh` completed successfully: 46 Qwen regressions and
14 additional MOSS/shared/kernel checks each ran individually without a Python
interpreter, ComfyUI tree or network. All 60 passed. The AIMDO atomic-publication
regression also passed. `cargo fmt --check` and `cargo clippy --all-targets --
-D warnings` pass. These are selected regressions, not an unfiltered rerun of
every new Kitchen operation. Fresh .38 activation/route captures are archived
with their source hashes; unchanged base fixture sections retain their original
provenance.

The final 1024×1024, 50-step MOSS run passes **all numerical and memory-policy
checks**. Contexts, masks, initial noise, sigmas, every denoised step, final
latent, Float32 RGB and decoded PNG pixels are exact. Cached model sizes,
loaded bytes, module placements, cast flags, buffers and pin accounting match
at the observed phase boundaries. Each component fits entirely on GPU during
its active phase; the coordinator unloads between components. This validates
that lifecycle, not actual partial MOSS layer offload under VRAM pressure.
Generic static partial-offload regressions remain separate evidence.

| Measured phase, seconds | Rust | Original ComfyUI |
| --- | ---: | ---: |
| Text load | 1.187 | 1.565 |
| Text encoding | 1.052 | 0.896 |
| DiT load | 0.644 | 0.670 |
| Sampling, including step exports | 24.265 | 23.927 |
| VAE load | 0.111 | 0.085 |
| VAE decode | 0.285 | 0.309 |
| Pipeline wall, including hashes, exports and unload | 37.733 | 34.298 |

These are single functional pairs from fresh processes, using the development
Rust build. They are not repeated steady-state benchmarks. The final native
run reuses the already successful original Python run with identical profile
and full checkpoint hashes. No original model computation changed between them.
The earlier native run took 52.211 s including unbuffered four-byte artifact
writes; buffered writes reduce that diagnostic overhead without changing any
exported tensor bytes. There is no measured image-quality difference between
the engines because their decoded pixels are identical. The saved image is a
natural macro butterfly/flower example; this one prompt does not establish
general model quality.

Two development export corrections are explicit: resident forced-cast flags
are exported as `cast_modules`, separately from actual lazy passes, and the
text owner's force-cast flag applies to every leaf, including plain norms.
The original full-size numerical run already passed; its metadata comparison
failed until the reporter reflected those existing owner flags. The failed
report is retained. The final comparator requires both numeric and policy
identity; no tolerance or expectation was weakened. Model algorithms, original
autotuning and checkpoint attention preferences are unchanged.

Durable evidence is in [reference-20261018/validation](reference-20261018/validation/README.md):
60 isolation reports, source-input fingerprints, check logs, original and Rust
MOSS results/PNGs, per-boundary SHA256 manifests, automatic Qwen comparisons
(including failures) and explicitly labelled placement diagnostics. Large
checkpoints, tensor dumps and development executables remain in the isolated
local root, outside Git. Tests, fixtures, profiles and oracle source are kept
in the repository for the next reference upgrade. Nothing was committed or
pushed as part of this continuation; unrelated dirty work was preserved.

## Remaining coverage

- Full upgraded Kitchen registry/model parity, including new packed/mesh and
  autoregressive APIs, is not complete.
- Selected MOSS support here is SFT INT8 ConvRot text-to-image. Other checkpoint
  variants, image conditioning, MOSS LoRA, optional negative-layer skipping,
  autoregressive/MTP decode, and real partial MOSS GPU residency are not accepted
  by this control. Loaded auxiliary MTP weights are still owned and counted.
- Qwen INT8 automatic comparison is exact. BF16/Turbo automatic runs still
  differ when physical memory produces different leaf placements and LoRA
  arithmetic routes; only their recorded-placement diagnostics are exact.
  This limitation is not hidden behind the successful MOSS result.
- Ordinary product/API integration and default replacement are outside this
  change. The experimental driver is explicit and defaults to preflight.
