# Open parity and acceptance gaps

CHECKPOINT-28, 2026-10-21. Numeric evidence and product policy are separate.
No memory optimization from the independent experiment was integrated here.

## P07 source workflows

Cast/project metadata, library-character hydration, image inventory and Story
dialogue/placement projection are implemented. Two HTTP integration cases and
native q61/q62 verify the source workflow and independent project persistence.
Eighteen pure projection cases match actual Director code. All five profiles /
36 copied source files remain unchanged. These are bounded acceptance results.

* **Image algorithm gap resolved in bounded component acceptance:** source
  capture uses EXIF transpose/RGB, longest side at most1280, Pillow-compatible
  Lanczos and optimized JPEG90. All28 procedural PNG/JPEG/WebP cases match raw
  RGB and JPEG bytes against actual Director functions, including EXIF1–8,
  palette/alpha, 8/16-bit gray/color and CMYK. Existing cjpeg/djpeg memory pipes
  avoid a new Cargo/Python dependency. Broader/corrupt input coverage remains.
* Enhance marker protection/restoration and its user-prompt builder are ported.
  Missing markers retain source and fail the proposal; introduced references
  warn without blocking manual review. Full Story dynamic prompt/protection
  construction and complete provider semantics remain open.
  Cast lookup/fingerprinting now share timeline token selection;64 actual
  Director cases pass. The distinct protection rules and native Reference/I2V
  compilers match421 actual-function cases, plus32 I2V text projections.
  Complete Unicode IGNORECASE/control-whitespace handling, parser warning
  diagnostics and broader invalid shapes remain unaccepted.
* Compiled dialogue preserves exact words in <d> tags with speaker/language and
  voice guidance. Scene summary/ambience/music remain distinct source fields.
  Native Plan Review and admission use the same compiler. This is NOT evidence
  of neural identity/timbre. Cast pictures/voices, Section pictures, timeline audio and reference video
  with paired soundtrack now reach the standard worker and conditioning, with
  bounded actual GPU/UI acceptance in CP24–26. Full boundary/Section manifests
  and complete Python neural/algorithmic parity remain open.
  I2V's description-only cast mode follows Director and is explicitly shown.
  Relationship target/note/contribution/strength/voice-binding details need
  complete project-schema/UI/runtime mapping before generic manifest parity.
* Native library selection, browsing, Save/Replace and independent copies are
  implemented and tested in own instances. Full reference metadata preservation,
  all image-role editing and broader fault/cancel acceptance remain open.
* Provider abort acknowledgement, budgeting, live vision/model behavior and
  broader image formats remain unverified. No model/encoder name allowlists or
  advertised-vision preflight block was added.

| Area | Verified state | Remaining requirement |
| --- | --- | --- |
| Montage frame/audio arithmetic | Actual Director AST functions: 199 CPU cases pass. Ten complete raw projects pass stated tolerances. Nine RGB byte-identical; Continuous Shot float max 1.192093e-7 / quantized max one level. PCM max 2.980233e-8. | Broader sources and kernel/reduction-order coverage; this is not blanket algorithmic acceptance. |
| RGB8 resize | Fourteen cases exact against installed Pillow. Native Rust reproduces coefficient quantization and intermediate byte rounding. | Other modes and broader shapes are unverified. Algorithm reference: [Pillow 12.3.0 Resample.c](https://raw.githubusercontent.com/python-pillow/Pillow/12.3.0/src/libImaging/Resample.c). |
| Source decoding | Sequence output streams with the seam tail. One source clip is decoded into a bounded buffer. | The 2 GiB resident-source bound differs from broader reference/spill behavior. Long/VFR/unknown-frame-count sources need acceptance and a suitable decoding path. |
| VFR clock | A procedural variable-PTS source with average rate 192/11 passes the raw oracle. Like Director, selection/fitting uses decoded frame order and average frame rate. MKV without declared nb_frames also passes. | This does not preserve the source's variable timestamps or cover all unknown-clock/corrupt metadata cases. |
| Per-clip normalization | CP19 removed the extra Peak/loudnorm operation from montage. Actual sequence mix ignores overrides; two raw comparisons with authored Peak/LUFS values pass. Final RMS normalization is tested. | Other workflows' normalization and broader LUFS/true-peak behavior are not covered by this montage result. |
| Export failure | CP19 retains compressed bytes in the frozen plan's session cache. Transaction failure rolls back files/Delivery; retry succeeds without decoder descriptors. Cache releases on success. Native save-failure recovery passes. | Cache is session-only; failed-result recovery across server restart is not promised. |
| Generated transition | Specific missing-runtime error is visible and recovery via explicit Hard Cut is tested. | Real bridge Take/runtime binding is missing. Director's assembler warns and concatenates when no generated bridge exists; Imagine currently fails explicitly. This behavioral difference is not generic parity. |
| Native UI | Import/edit/Undo, transition error, real encoding cancel/retry, lost admission retrieval and save failure/retry pass in isolated instances. | Narrow windows, library and project-copy handling remain. |

## H3 execution and memory policy — independent of UI acceptance

Source audit supplied by the memory task:
`/run/media/veracrypt1/imagine/.codex-h3-memory-20261021-c77f4a9b/evidence/H3-PARITY-AUDIT.md`.
Only that authorized report was read, not its linked runtime logs.

* Rust model owners/coordinator are created and dropped per component/job;
  ComfyUI keeps patchers alive longer. Worker process lifetime also prevents
  reuse across jobs. Reference owner lifetime/unload timing must be addressed
  before a separate disposal optimization.
* DiT's separately calculated minimum is not forwarded into the Rust manager.
  Video/audio VAE budget formulas differ from the reference.
* Static synchronous loading lacks the dynamic VRAM/async-prefetch counterpart.
  Extra global synchronization/cache-clearing and pin-pressure/mmap policy
  differences remain to be resolved or explicitly agreed.
* Full unload returns tensors to CPU in both implementations. The discrepancy
  concerns timing, frequency and ownership, not the mere presence of that copy.
* The recent memory experiment's identical output compares Rust with Rust.
  Earlier bounded Python/Rust numeric tests are distinct evidence. Neither
  proves complete functional and algorithmic parity.

Intent/exception basis is partly unknown. No general parity acceptance is
claimed, and no H3 memory-policy changes were made in this UI checkpoint.
Motion Context Exp remains failed/not finally verified; the specifically denied
three-step diagnostic was not retried.

## Mandatory separate review after production selection

Compare all UI/backend/profile/runtime/export controls with Director. Every
product restriction needs a concrete necessity and the least restrictive
implementation. Wrong model/text-encoder choices must remain possible, with
actual load/run errors; compatible alternatives must not need code changes.
Do not turn source/binary test hashes into runtime gates. Preserve required
format/memory safety, resource integrity and the selected encrypted destination.
This review remains pending; no mass removal is part of this checkpoint.

## CP26 bounded reference-video acceptance

22 focused actual-source cases match Director for complete manifest, prompt,
metadata and soundtrack plan. Two own MP4 decodes match raw RGB/44100-Hz PCM;
six procedural preparation cases match FP32 resized frames, Qwen frames/times
and FP16 VAE values/strides exactly. Correct NCHW Lanczos layout was restored
after the first oracle exposed a stride mismatch despite identical values.

Standard worker v8 receives video+paired audio through native UI, completes
20 steps and exports H264/AAC stereo with correct 124/24 duration. Cold own UI
play/pause with live audio process passes. No new GPU observer was used; this
does not establish full Python DiT parity, learned correspondence, or every
video/codec/canvas/boundary case. FinalMix/Both source projection, four actual raw montage cases and two full
native UI render/Delivery-playback flows now pass in CP27. This is bounded
acceptance; broader source/duration/fault cases remain.
Combo/Influence boundary transport is still missing. No memory-policy change.

## CP27 reference soundtrack and next boundary binding

Four procedural raw projects: RGB exact; PCM FinalMix/Timeline exact, maximum
Both/Combined error2.98023224e−8. Reference/Ignore correctly exclude the video
soundtrack. Own q91/q92 native UI render/select actual Delivery/play with live
audio/pause/save passes39 actions each. Output is H264/AAC stereo44.1kHz,
124/24 frames/fps and5.166667s. Test-only viewport/extra-close issues resolved;
no product behavior change was necessary.

New reference_window::prepare is a CPU-verified component ONLY, not runtime
support: actual Director reference-tail profile/attention matches12 cases
exactly, including paired/unpaired audio and a frame0 guide. Combo/Influence
needs full source Take video/endpoint projection and transport, exact resampled
tail/paired PCM range, source identity/provenance and attention installation
per DiT stage. Native UI methods remain disabled until that path works; no
silent fallback or acceptance claim. Director defaults tail3s→72requested at
24fps, aligned by17k+5 and clamped to source/target. Pictures remain unlimited.

No CP27 GPU reservation was requested. Before the next GPU job, normal Ticket
register/acquire must confirm a new reservation can be obtained; do not read
old rejected logs or substitute another data path. The CP26 exact log read was
denied even after fresh human approval; it is not the older Python-stderr or
Exp-three-step request. No such blocked action was retried here.

## CP28 production selection with explicitly retained gaps

Per Tomas16:24 instruction, production selection happens before broad review
and full final acceptance. Active package: comfyui/products/20261021T162857Z-b1ee7679.
No running production process was restarted and no user data was read/migrated.
Own new/rollback server startup+synthetic upload and normal UI startup pass;
image/API/UI regressions pass. The standard image deployment is unchanged.
This is a reversible code/deployment selection, not blanket parity acceptance.
Separate all-code/UI restriction review follows; do not let this decision
implicitly approve memory/execution, Python, Exp, D10 or remaining P07/P08 gaps.
