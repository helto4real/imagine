# H3 product worker: bounded integration plan and CPU milestone

Latest: direct product-integration approval succeeded. The completed product
implementation and current verification are described in
`h3-product-worker-product-integration-20261020.md`; this supersedes the
historical approval block recorded below.

Subsequent direct approval on 2026-10-20 lifted the GPU pause and approved the
product integration. Automatic permission review nevertheless rejected the
product-source transfer before execution; that path remains blocked by the
review flow. Independent inference GPU validation is now authorized and passed for the
three explicit worker cases recorded in `h3-product-worker-evidence/gpu-milestone.json`. See `h3-product-worker-evidence/authorization-20261020.md`. The CPU
milestone below is a retained historical snapshot, not the current GPU policy.

## Actual start and ownership

Worktree `/home/thhel/.codex/worktrees/h3-product-worker/imagine`, branch
`h3-product-worker`, exact base/starting HEAD
`40ff91114a6d71782aa1a79a76df6940546570bb`. This is a new managed checkout;
the frozen `be07` checkout and its ignored artifacts are unchanged. No later
main, GeneralSet, safe-globals, registry or D-05 work is imported.

Own mutable files are isolated below `comfyui/h3-product-20261020/` (targets,
Cargo home, compiler/media caches, binaries, manifests and synthetic test data).
Only checksum-verified archives already present in the frozen Cargo lock were
copied from the retained local source cache. There are no downloads or new
dependency versions. See `h3-product-worker-evidence/start.json`.

Writes belong to `inference/**` and this task's design/evidence. Product
`src/**`, `ui/**` and the shared protocol are owned by the product integration
task. Its exact protocol was copied only after explicit coordinated transfer:
`shared/h3_protocol.rs`, SHA-256
`b52cac00b8e88687de3112c419c5ca80fbbf81aa6c643e053ca4a2755b49e5d5`.
That source is unchanged and remains product-owned. No product wiring is done
here. The coordinator reports that its broader product wiring was denied by
automatic permission review; this worker work does not bypass that rejection.

## Fixed slice and authoritative contract

One no-reference job: private user text, exactly the frozen four model hashes,
384×256, 39 frames at 24 fps, four Euler/simple steps, CFG 1, shifts 12/3,
seed 42, stereo FP32 32 kHz and 52,000 samples per channel. No negative prompt,
LoRA, references, profile expansion, alternative precision or algorithm policy.

The delivered protocol fixes these requirements:

- A private pipe frame is u32 BE JSON-header length, header, u32 BE body length,
  body. Header limit 64 KiB, event payload limit 4 MiB, prompt limit 16 KiB;
  a request body is empty. Bounds/unknown fields/version/profile/seed are checked
  before constructing a native runtime. The prompt remains byte-for-byte input
  text and is never argv, environment, a log, tensor/token dump or a phase file.
- Ready identifies the actual current executable SHA-256 and a verified,
  canonically serialized `BTreeMap<absolute dependency path, SHA-256>`, plus the
  frozen core revision, model-lock hash and profile. No invented worker hash.
- Typed loading/encoding/sampling/decode phases and monotonic four-step progress
  expose metadata only. RGB8 HWC frames are indexed 0..38. Float32 clamping,
  multiplication by 255 and ties-to-even rounding use the authoritative
  `export_rgb8` helper. PCM is unmodified finite interleaved stereo f32 LE.
- Complete is emitted only after all 39 frames, 52,000 sample frames and four
  steps. Errors use only the closed enum; no exception, path or prompt text.
  The product owns partial bytes and must discard them on failure/cancellation.
- Scheduler alone owns Ticket. The worker never starts a coordinator, a second
  reservation, the standalone reservation CLI, an HTTP service or GPU children.
  One admitted request runs in one transient native process with sequential
  component owners; the product must kill/reap that process before Ticket finish
  on cancel/lost/shutdown, and reap after Complete before releasing the lease.
  There is no in-band Cancel RPC or persistent loaded model pipeline.

The 16 KiB bound is a transport constraint, not a validated GPU context window.

## Reuse boundary and concrete source findings

The dedicated new entry is `inference/src/bin/imagine-h3-worker.rs`, backed by
`inference/src/h3_product/`. `h3-stage4.rs`, its replay tools, model layers,
loading/sampling algorithms and frozen numerical fixtures remain unchanged.

`server/mod.rs::load` rejects `_quantization_metadata` and discards the common
normalization scope. Its shared attention helper sets low-precision attention
false, whereas the accepted H3 execution sets it true. The image worker also
changes global SDPA/math state and enables inference mode in the server path.
Therefore product H3 does not route through `server::generate` or borrow image
worker defaults by changing the model name.

The new adapter calls the same `component::load_aimdo` normalization and
`h3::checkpoint::prepare_normalized/static_component` owners, preserves legacy
metadata precedence, source dtypes, HostPins, static Manager budgets, the DiT
sigmas buffer and synchronized unload/drop/soft-empty boundaries. All four
locked model identities are checked after request admission and before trained
allocations; accepted per-component full checks remain before their loads.
No model payload is read during CPU validation.

Tokenizer, 50-layer unnormalized text conditioning, dense DiT, common two-stream
noise/schedule/CFG1 Euler path, video/audio VAE forwards and audio scaling remain
the frozen implementations. Kitchen uses `Runtime::frozen_partial`, bound
layouts, disabled Kitchen Triton, two CPU threads and no-grad as in the accepted
worker. Attention is Pytorch, priority wrapper true, precision None,
low-precision attention true. No image-worker SDPA/math/inference-mode override
is added. Existing sampler observation sites emit step metadata; tensor contents
are never copied for progress or saved.

A single transient process now carries CPU conditioning/latents between
components. It is not the former separate-process-per-phase harness. No two
trained component graphs are kept loaded at once. This new process composition
requires its own later GPU/memory/lifecycle acceptance; frozen phase acceptance
alone does not prove it.

## Launch metadata for the independent worker

The binary takes no CLI arguments. Trusted deployment paths are environment
configuration, never request fields:

| Variable | Meaning |
| --- | --- |
| `IMAGINE_H3_DEPENDENCY_MAP` | JSON BTreeMap of absolute dependency paths to real SHA-256 hashes; excludes the worker itself, model files and its own map file |
| `IMAGINE_H3_MODEL_ROOT` | Root containing the four exact profile-relative model files |
| `IMAGINE_KITCHEN_LIBRARY` | Final frozen bridge, required SHA-256 `ab7c807cd5f9aac244f58a302d9500a6ab04b0314b7223404881ce407ed0d4a3` |
| `IMAGINE_AIMDO_LIBRARY` | Locked native AIMDO library, present and verified in the dependency map |
| `LD_LIBRARY_PATH` | Verified native dependency search paths supplied by deployment |

Ready's dependency digest is SHA-256 of `serde_json::to_vec` of the sorted map.
The product must bind its expected Ready to the identical executable and map.
A worker/dependency mismatch fails before request admission or CUDA initialization.
Configuring these values is the product owner's future integration; no product
runtime manifest or defaults are modified here.

## Completed CPU milestone

A separate `inference/tools/h3-product-cpu/` crate imports the exact shared
protocol and the worker's real wire/deployment modules by path. Its locked
transitive graph is a subset of the frozen dependencies and has no tch,
torch-sys, inference-runtime or CUDA linkage. Its fixture is a visibly named
CPU-only executable, not a synthetic mode in the product worker.

Eight tests pass: reject malformed/truncated/oversized/unknown/invalid-seed
requests before the factory; preserve Unicode/whitespace; compute and verify
actual executable/dependency identities; check authoritative export/nonfinite
PCM rejection; handle broken writes without leaking exception text; stream a
complete fake AV result over a real child pipe; reject partial/error/panic
results without Complete or private stderr; kill/reap an admitted CPU child.
All test data is task-owned synthetic data. Tests do not load weights or execute
the actual GPU-linked worker, even for Ready. A first lock-graph assembly needed
Cargo's offline normalization; no dependency version was introduced.

Commands use the isolated CPU environment in the ignored own root. The only
allowed executed tests are this explicit CPU crate, with locked/offline Cargo.
Native worker compilation is CPU build evidence only; the binary is not run.
No broad primitives/inference test target, model loader, CUDA capability query,
GPU generation, graphics preview, product process or existing service is run.

## Verified build and evidence

Final isolated, locked/offline checks passed:

- CPU-only transport/deployment crate: 8 passed, 0 failed, 0 ignored.
- CPU-only crate Clippy, all targets, warnings denied: passed.
- Dedicated native worker Clippy, warnings denied: passed.
- Dedicated native worker dev build: passed. The actual executable was hashed
  and inspected with `readelf`; it was never executed. Its CUDA linkage is not
  GPU validation. The executed CPU fixture links no Torch/CUDA library.

`h3-product-worker-evidence/cpu-milestone.json` records the actual artifact hash,
source hashes, commands, frozen lock subset and copied own CPU compiler/test
outputs. Source inputs and the frozen control checkout were checked unchanged.
The only edited pre-existing tracked file is the module registration in
`inference/src/lib.rs`. All worker changes are uncommitted; no push, merge,
product restart, runtime configuration or deployment is performed.

## GPU acceptance held for separate human clearance

The frozen acceptance remains 415/0/7 and 55 trained exact artifacts for the
standalone baseline. New product-path GPU acceptance is pending:

1. Replay the original own fixture prompt through the new request/worker path
   and compare original-profile conditioning/sampling/final AV tensors separately
   from RGB8/PCM export bytes. Use ephemeral test observers in a future acceptance
   harness, never product tensor dumps or prompt logs.
2. Run one additional explicit own test prompt to prove real request text is
   used; retain the same four models, geometry, seed and numerical policy.
3. Measure the new single-job component sequence, actual peaks and owner release;
   test normal/error/cancel/lost/shutdown and next image-job recovery through the
   common scheduler with one Ticket. Root/product integration needs its own
   authorization after the reported permission rejection.
4. Verify 39 frames at 24 fps and stereo 52,000/32,000 = 1.625 seconds, preview
   and explicit Save. Product media/mux/UI acceptance is separate from worker
   tensor evidence and requires the product owner's authorized work.

Stop the affected path if it requires blocked core, new model/precision/algorithm
settings or a changed shared protocol. No such extension is implemented here.


## Subsequent independent GPU worker milestone — 2026-10-20

Direct human approval lifted the pause and covered the product integration.
Automatic permission review rejected the product-source transfer before it ran;
no product paths changed here. The complete product integration is still blocked
by that review flow, not by missing human consent. No alternate product wiring
or message to another chat was attempted.

The independent inference development supervisor built and passed all-target
warnings-denied Clippy using locked/offline, already frozen dependencies. It
imports the unchanged Ticket implementation and uses the existing canonical
coordinator; no service or coordinator is started or replaced. The actual worker
binary and all worker/core/protocol sources are unchanged from the CPU milestone.

Three actual GPU cases passed sequentially with identical worker/deployment
identities and the exact locked no-reference profile:

- Source profile fixture: four steps, 39 RGB8 frames, 52,000 finite stereo PCM
  sample frames, Complete, normal worker exit 0 and reap before Ticket finish.
- Cancellation after trained text graph loading: kill/reap before finish, no
  Complete or output frames/audio published.
- Own alternate source prompt after cancellation: complete AV and normal exit,
  confirming subsequent H3 recovery and that request text affects video output.

Evidence and exact streaming digests are in
`h3-product-worker-evidence/gpu-milestone.json` and its three case reports.
Media was hashed in memory and not retained. The initial dependency-map
preflight failure is retained separately; it started no worker or Ticket.
Only own deployment paths were normalized; strict identity validation stayed.

This validates actual worker execution, framing and these lifecycle cases.
Intermediate tensor equality, peak GPU memory, real product loss/shutdown,
next image-job recovery, product queue/API/storage, mux, preview and explicit
Save remain unvalidated. Native UI/AV playback and perceptual quality were not
checked. No private runtime data or existing generated media was read; the only
model reads were the four locked files needed for these approved synthetic runs.
All changes remain uncommitted; no push, main merge or product restart occurred.
