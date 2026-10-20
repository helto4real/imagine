# Native AV baseline and H3 handoff

## Authority and safe checkpoint

Tomas approved the shared native AV baseline and subsequent H3 worktree in the
coordination task **Kontrollera ComfyUI Rust-portering**
(`01a15798-d376-776c-b90e-26894d57201c`). Its delivered instruction includes the
scoped approval from the voice task and explicitly selects the INT8-finetuned
text encoder below. This changes priority, not the full parity objective.
General pickle work waits unless needed by these selected safetensors.

The current atomic change is complete and uncommitted. Its numeric-hash test
passes directly and in isolation; warnings-denied all-target Clippy, formatting,
source fingerprints and the focused whitespace check pass. Retained evidence:

| Validation | Result |
| --- | --- |
| Frozen full primitive executable before AV follow-up | 386 tests, zero failures, 672.35 seconds |
| Native ZIP record/storage selection | Three selected tests pass |
| Dense V2/V3 reconstruction selection | Four selected tests pass; 64 constructor and 384 metadata histories |
| Numeric hash selection | 764 integer, 20,480 float and 1,168 complex cases pass |

The later additions have selected acceptance; the 386-test full run is not
misrepresented as a replay of the later source. Detailed hashes and scope are in
[the Stage 3 ledger](stage-3-shared-algorithms.md). All our test, build, Clippy
and source-download commands have finished at this checkpoint. **No GPU job
owned by this task remains active.** No other process/service inventory was
read. No process was killed. No ordinary worker was replaced or restarted.

The truncated CPython API response was not used as a source dependency or
implementation input. Its ignored local copy is explicitly marked partial;
complex scalar/extra pickle extensions have not been started.

## Selected local components

Model root: `/mnt/media/gemensam/AI/ComfyUI/models`.

| Role | Exact relative file |
| --- | --- |
| Ref2VA DiT | `diffusion_models/minimax_h3/minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| Text/vision encoder | `text_encoders/minimax_h3/qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors` |
| Video VAE | `vae/minimax_h3/minimax_h3_video_vae_int8_convrot.safetensors` |
| Audio VAE | `vae/minimax_h3/minimax_h3_audio_vae_fp32.safetensors` |

The finetuned INT8 encoder is its own selected weight set. ComfyUI and Rust
must use that exact same file in a comparison. Earlier AWQ evidence is not
acceptance for it. Initial inspection reads bounded safetensors headers and
quantization metadata only. It must not allocate model-sized tensors or read
ordinary prompts, jobs, reference images or generated media.

## Ownership and smallest proposed contract

This core task owns `inference/`, native bridges, core tests and the associated
design/evidence. Existing unrelated root Rust/UI/native work is preserved.
The video task retains product video orchestration, result/export integration
and its own existing edits. The coordinator settles any overlapping paths
before a shared product file is changed. No UI work starts here.

The initial core contract is internal and model-independent:

* A normalized component state retains ordered source keys, metadata and tensor
  owners, and invokes the existing source-faithful `convert_old_quants` before
  quantization detection/preparation. It must not discard the header metadata
  or broadly remove the product loader's current rejection without a tested
  replacement.
* Existing `PreparedParameters`, Kitchen layouts and model/memory owners carry
  actual quantized payloads, scales and policy. Offload and accounting follow
  the reference; no additional scheduling or autotuning policy is introduced.
* Native AV sampling carries an explicit ordered video/audio stream pair plus
  original latent shapes, masks, dtype/device, audio scale and conditioning
  owners. Pack/unpack preserve view/clone boundaries and source audio scaling.
  Results remain tensors and timing/cancellation state at this boundary;
  container encoding and product assets belong to video integration.
* H3 conditioning carries the actual hidden states, integer modality tags,
  references/keyframes and source payload fields. The Qwen Image chat/norm
  recipe is not reused as an H3 recipe. Source H3 uses raw labels/text, its
  selected 50-layer Qwen3-VL-32B output without final norm, and explicit tags.
* Cancellation/lifecycle must release the complete owner graph and GPU lease,
  including exceptional paths. An interface listing or test stub does not
  establish a trained model implementation.

Core changes are expected in focused checkpoint/component-loading helpers,
`inference/src/checkpoint/state_dict.rs`, model-owner/memory bindings,
shared latent packing/sampling and a dedicated H3 model module. Tests, oracles,
fixtures and evidence stay in `inference/tests`, `inference/tools`,
`inference/fixtures` and `design/`. Existing public exports in
`inference/src/checkpoint.rs` / `models/mod.rs` may need registration.

**Shared product paths require coordination first:**
`inference/src/server/{mod,qwen,moss}.rs`, root `src/engine.rs`,
`src/scheduler.rs`, product model/request/result types, worker selection and
launch/preparation scripts. This plan does not assign those paths to a second
writer. A standalone development worker/test harness can validate the core
contract before product integration.

## Sequence and isolation

1. Record bounded headers/quantization profiles and the exact locked H3 source
   functions. Complete only the shared loading, ownership, conditioning and AV
   sampling pieces required by those components. Use bounded synthetic
   differential tests first; no model-sized reference test allocations.
2. Verify the common base and retain reproducible source, native patch/restore
   proof, fixtures, evidence and dependency locks in the first local baseline
   commit. Inspect its exact staged diff; exclude unrelated work. No push.
3. The coordinator creates/assigns a separate H3 worktree after that baseline.
   Remaining full core parity resumes independently with explicit file/commit
   ownership. H3's limited Stage 4 work implements the selected encoder, DiT,
   video/audio VAE forwards and their verified binding to the common AV base.
4. Run no-reference generation first, then one own synthetic image reference.
   A second H3 baseline requires a real short AV clip, consistent stream timing,
   the synthetic-reference case, and lifecycle/cancel checks. Proposed tiny
   clip parameters are technical probes; quality and memory fit remain unproven
   until measured with the selected trained weights.

Each track needs its own Cargo target, worker executable, writable compiler
caches and runtime manifest. Reuse immutable pinned dependency artifacts only.
The accepted `comfyui/moss-20261018/target` worker and frozen evidence remain
controls. New AV GPU jobs wait for a common reservation protocol shared with
the video task; the coordinator must identify the canonical lock/lease before
either track starts such jobs. No such AV GPU job has been started here.

## Gates still open

This document establishes a safe checkpoint and a concrete proposed boundary.
It does not establish native H3/Ref2VA support, trained memory fit, a successful
AV clip, complete Stage 3, either future baseline commit or an H3 worktree.
No new dependency, numerical tolerance, source algorithm exception or cloud
inference service is authorized by this prioritization.


## Coordination update and initial header evidence

The coordinator has confirmed these file boundaries. The video owner reports
baseline `32fb9a2`, 31 passing video tests, no active jobs and no pending video
edits; the core task verified that HEAD and its video-only publication stat.
No core inference changes were absorbed into that video commit.

The canonical protocol is already `src/coordination.rs::Ticket` with
register/acquire/lost/finish, called from the scheduler and product proxy.
There is no known authorized active socket from this core task. The normal
launcher path `.imagine/gpu.sock` is configuration evidence, not proof of the
active socket. A separate flock does not coordinate with that product protocol.
The coordinator reports that automatic review rejected its attempted live
process-environment inspection; it did not execute. This task will not retry
or indirectly perform that read. **New AV GPU jobs remain held.** CPU/source
work and tests against own temporary mock coordinators continue independently.
The existing `tests/gpu_coordination.rs` harness is reused before creating any
duplicate protocol or changing the shared product files.

[Bounded header evidence](stage-3-evidence/native-av-selected-headers-20261020.json)
confirms all four exact files. It allocates no weight tensors and reads no
trained weight payload. The selected text encoder has both **351** embedded
quantization entries and legacy metadata (350 INT8/ConvRot layers plus the
INT8 embedding); the other components have **200** DiT and **144** video-VAE
quantization entries, with none for the FP32 audio VAE. In the text file,
embedded rows include a per_row flag absent from legacy entries. The loader
must preserve the original converter's precedence and scale-derived behavior;
it must not invent a preference for one representation. These are header and
small quant-metadata facts, not trained inference or memory-fit results.


## Reservation gate resolved; development runner boundary

The coordinator has now supplied Tomas's explicit socket/listener authorization
and verified the active private socket at
`/home/thhel/ChatGPT/imagine/.imagine/gpu.sock` (uid 1000, mode 0600).
The earlier discovery hold is resolved. No process-environment read was retried.
All later tracks must pass this absolute socket to the existing Ticket protocol;
no alternate coordinator or flock is used.

The development runner is a separate crate under
`inference/tools/gpu-reservation/`, with `inference/tools/gpu_reserved_job.rs`
sharing the actual tracked `src/coordination.rs`. Its dependency versions are
already locally available and pinned in its own Cargo.lock. It has no dependency
on the unrelated uncommitted product implementation. The initial root-example
prototype and its newly created root test files were moved into this owned crate;
no existing root/product file was changed by that move.

The runner admits a **direct native ELF GPU worker**, with all GPU work in that
single process. Interpreters, shells, bwrap and other launchers are rejected.
GPU-capable descendants/session changes are unsupported by this small controller;
its documentation is a precondition, not a claim of arbitrary process-tree
supervision. Do not pass test_isolation.py or another wrapper as its worker.
SIGINT and SIGTERM both request cancellation. The acquired Ticket stays alive
until wait confirms that direct worker has exited; finished is sent afterward.
A send has no broker acknowledgement and is recorded as finish_sent.

The prototype's six CPU/fake-listener cases and two real SIGINT/SIGTERM cases
passed, with a known harmless child's disappearance checked before finished.
The standalone crate replay passed all eight cases after the source move,
including both real controller signals. Warnings-denied all-target Clippy passed. The existing product-coordinator fake integration test also passed
(1 test, 0.77 seconds). None contacted the active socket or performed GPU work.

Writable core, reservation and product-test targets/Cargo homes are now under
`comfyui/native-av-20261020/`. Immutable, already pinned registry artifacts may be
shared; new global Cargo cache state is track-local. The validation tools accept
a separate execution/runtime root, and CPU-only mode exposes no NVIDIA devices.
The new component selection passed five tests in that separate CPU-only root:
12 selected-header normalization histories, three actual-source annotation
histories, a tiny AIMDO composition, header diagnostics and the retained AIMDO
storage matrix. No trained weight tensor or trained weight payload was read.
Warnings-denied core Clippy passed. Product loader rejection/defaults remain
untouched pending the explicit AV consumer binding.

The video owner's later UI authorization is isolated in
`/home/thhel/.codex/worktrees/video-timeline-ui/imagine`, branch
`video-timeline-ui`, from `32fb9a2`, according to the coordinator's verified
worktree report. That UI branch is not part of this AV baseline publication.


## H3 CPU preparation in progress

The original-source fixture now covers 24 video-block histories (two actual
frames, BF16/F32/F64 input and F32/F64 global defaults), 39 empty AV latents
(temporal lengths 0 through 12), 48 packed audio scaling histories, and 40
image-tag slice histories. The first two Rust tests passed. The first scaling
run exposed a Rust binding bug: the existing context-slice helper requires a
nonnegative axis; the new caller had passed -1 for the last axis. The caller
now resolves the last axis before invoking that helper. The failed selection
is retained; the repaired implementation must pass again before acceptance.
No source algorithm, tolerance or comparison requirement was changed.

A separate fixture executes the actual SDTokenizer, Qwen3VLSDTokenizer and H3
tokenizer classes from the locked source against local assets. It captures 108
raw-text/reference histories including fixed H3 special IDs, escaped parentheses,
empty fallbacks, audio labels, request order, odd-frame repeat padding and
explicit timestamps. No trained encoder or weights are constructed. The H3
factory reuses the byte-identical Qwen25 assets with its own seven fixed added
tokens; it does not alter Qwen Image's tokenizer or frozen settings. Full
textual-inversion and arbitrary Python argument binding remain separate gates.


The repaired slice reached a second binding issue in the retained tch scalar
wrapper: its temporary Scalar destructor panics while the in-place Long-times-
float dtype exception is still pending. H3 now invokes the same native
`aten::mul_.Scalar` operation through the existing boxed boundary, which safely
collects and returns the original exception. The repaired test compares exact
exception class and message. There is no new error tolerance or source exception.
Both failed selections remain retained separately.

The typed stream wrapper owns an actual shared mutable list: `unbind` returns
that same list, shallow copy creates a new list with the same tensor owners, and
scale=1 returns the same wrapper. A separate source fixture covers 18 such
histories and 72 shared global CPU RNG histories, including per-batch indices,
negative indices and empty-noise behavior. These primitives do not establish
full arbitrary Python argument, exception-frame or object protocol bindings.

The original contiguous H3 payload-assembly block has 576 source histories using
opaque synthetic owners. It preserves tags, keyframe/ref list identity, input
order, exact None-vs-absent rules, false/zero augmentation and seed values, and
the model's actual `audio_scale` decision. Learned text projection, denoise-mask
pooling and PackedLayout remain explicit caller gates. The implementation must
pass this selection before any acceptance claim.


## Accepted H3 preparation selection

[The repaired CPU-only selection](stage-3-evidence/native-av-preparation-cpu-repaired/selection.json)
passes all nine selected tests, including the retained Qwen Image tokenizer
regression. H3 checks cover **24** two-frame patch histories, **39** empty AV
latents, **48** packed scales/errors, **40** tag slices, **108** tokenizer
presentations, **18** nested list/owner histories, **72** global RNG histories
and **576** payload assemblies. Outputs use exact tensor bytes, generator state,
original exceptions, key order and source owner identity as applicable.
Warnings-denied all-target core Clippy passes. The input snapshot verified
**1,111 implementation/fixture inputs plus 57 original source files** before
further implementation. The failed selections are retained as failed.

This selection runs in the separate AV runtime root with NVIDIA devices hidden
and no interpreter or source tree inside the native test sandbox. No model
weights, ordinary prompts/jobs/history or model-sized test tensors are mounted.
It is a shared primitive result; neither trained H3 inference nor complete
Stage 3 is claimed. Next is the composed AV stream/flow boundary and focused
shared owner regression before publishing the common baseline.


## Shared baseline gate passed; trained H3 remains next

The actual `inference/tools/check_native_av.sh` replay completed successfully:
**20** individual Python-free, CPU-only native tests, **8** standalone reservation
tests (including real SIGINT/SIGTERM against own harmless children), formatting
and warnings-denied all-target Clippy for both crates. The exact native candidate
is SHA-256 `b2b98fd54aacd744655795f1760f52433b3a483850ff6b474d6dc7d63790a62f`.
The [fresh selection](stage-3-evidence/native-av-common-baseline-cpu/selection.json)
and retained gate log identify this candidate; older successful full-suite
results retain their own distinct hashes and scopes.

The composed source oracle contributes **18** actual packed/nested latent and
flow histories plus two zero-audio-scale errors. It uses an explicitly synthetic
velocity expression, never a learned denoiser. Both native branches preserve
latent-format multiplication/division by 1.0 and separate audio scaling; they
do not optimize away observable clone/owner boundaries.

The [native patch restoration](stage-3-evidence/native-av-torch-sys-restoration.json)
verifies all 20 original torch-sys files, three patch paths and seven added files
against the verified registry archive. The [generator replay](stage-3-evidence/native-av-generated-restoration.json)
matches 73 source-derived outputs. The common source snapshot covers the retained
inference source, fixtures/resources, vendored native patch and dependency locks,
plus the shared Ticket source and explicit H3 reference modules. External build
toolchain/source-acquisition and native ELF identities retain their existing
separate locks; this is not a claim to vendor the compiler or the Torch wheel.

This is the common CPU-validated AV foundation for the approved handoff. No new
AV GPU job, trained weight load, ordinary runtime-data read, trained clip, synthetic
image-reference run, product AV integration or public-default change occurred.
The separate H3 track must still implement and validate the selected 50-layer
unnormalized encoder, DiT, video/audio VAE forwards, full mask/PackedLayout
conditioning, trained memory/lifecycle/cancel paths and GPU isolation/lease
composition. It must compare identical selected files in both engines. Full
Stage 3's dynamic object/caller and registry interaction gates remain open;
this baseline does not redefine them as finished.

The local publication includes the already accepted shared Stage 3 core, its
original-code generators, synthetic fixtures, failed and passing evidence,
source/native locks and this AV follow-up. Unrelated root scheduler/engine/UI
and older native-experiment changes are excluded. No push is authorized.


Publication review selected **1,904** core/design paths and excluded root
product/UI/older native experiments. Full staged whitespace inspection reports
existing upstream whitespace in the byte-preserved NumPy source header and
nlohmann license, plus a trailing blank line in the retained raw red-phase log.
Those three exact source/evidence files are preserved unchanged; the scoped
application-owned staged whitespace check passes. This is a source preservation
detail, not a runtime error, relaxed numerical comparison or changed source
algorithm. Native patch restoration remains verified.
