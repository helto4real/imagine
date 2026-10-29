# Session authoring drafts

Story/Enhance uses no model queue or GPU reservation. Proposals never start
inference or accept a Take. The five Director profiles retain their byte copies,
resolver precedence and prompt inheritance. Admission freezes prompts, resolved
operation/provider settings, address/model and the private key in memory. Reload
and provider edits affect new proposals only. Keys never enter responses/files.

The native flow supports Outline, sequential scene expansion, editable
review, retry from a failed scene, Append/Replace/Continue, Scene description and
Section prompt enhancement. Closing review retains the draft. Recent proposals
retain earlier local drafts. A changed server session never resumes an operation.
Apply publishes reviewed source and provider/profile provenance in one revision.

CHECKPOINT-20 adds project-owned cast editing, source-image inventory, semantic
image placements, dialogue intent and library-character hydration. The actual
native UI flow and cold source-record review are tested. CHECKPOINT-21 matches
Director preprocessing: EXIF transpose, RGB, longest side at most 1280 (no small
image enlargement), Pillow-compatible Lanczos and optimized JPEG quality 90.
Twenty-eight procedural PNG/JPEG/WebP cases match both raw RGB and JPEG bytes.
The existing `cjpeg`/`djpeg` tools run through memory pipes; no Python runtime or
new Cargo dependency is used. Missing codecs report an operation error.
This component result is not complete Director/provider/runtime parity.

## API

All routes use the existing Imagine session guard. Types live in
`crates/video-contract/src/authoring.rs`.

- `GET /v1/video/authoring/profiles`: reload source-owned catalog.
- `POST /v1/video/projects/{id}/authoring`: capture schema 1 Request and revision.
- `GET /v1/video/authoring/{id}`: session state/progress.
- `POST .../review`: validate edited Outline, expanded Scenes or enhancement.
- `POST .../expand`: schema 1 and `start_index`; preceding successful Scenes stay
  unchanged. Model output cannot change IDs, order, weights or boundaries.
- `POST .../cancel`: cancel HTTP waiting and prevent late result publication.
- `POST .../retry`: retry inventory/initial proposal with captured inputs and
  provider settings; completed inventory notes remain. Scene expansion uses its
  existing explicit retry index.
- `POST .../apply`: schema 1 and `expected_project_revision`; both captured and
  current revisions must match. A stale draft remains available for review.

Request fields `cast`, `images` and `add_dialogue` default to empty/false.
Cast selects stable project reference IDs. Image entries carry a stable ID,
project asset, display label and `setting_reference`, `required_moment`,
`start_frame` or `end_frame` intent. Directed frame pairs follow selection order.
The model assigns semantic beats to Scene/Section ordinals; code owns image IDs,
frame roles and the Director two-turn dialogue limit. Eighteen valid/invalid
projection cases match the actual Director contract implementation, including
its code-owned Scene/Section IDs. The Story messages themselves follow
"Director Story messages" below.

Inventory runs sequentially using the exact profile's `image_analysis` settings
and `story_image_inventory` prompt. Outline/expansion choose the matching
dialogue profile prompts when requested. Drafts expose notes/cast metadata, not
inline image payloads or credentials. Image bytes and provider configuration
are captured in session memory. OpenAI-style image_url and Ollama images payloads
are implemented without a model-name or advertised-vision allowlist.

Section/Scene enhancement captures enabled image bindings, Section source images
and referenced cast images. Video-grounded enhancement reports its missing path
explicitly. It does not silently present a video-bound operation as text-only.

Project `creative` owns reference identities, aliases, descriptions, tags,
enabled images, independent previews, strengths/retention, voice metadata and
Section image/dialogue intent. It is optional for older documents. Edits, Undo,
scene duplication/removal, fingerprints and Save/Open retain these identities.
`POST /v1/video/projects/{id}/cast/from-library` captures library/entry/revision
and prepares independent media copies before runtime registration and project
publication. A failed registration leaves the project unchanged; later library
edits do not update its project copy. The native Project & Cast Library window
supports explicit folder selection, Create/Open/Refresh, title filtering,
revision-bound Save/Replace and independent project/cast copies. It uses the
existing API and never discovers libraries automatically or autosaves them.
Project copies are unsaved and need an explicit project destination. Library
character format currently stores name/alias/description/tags/images/voice;
full reference enable/preview/relationship/strength semantics remain separate.

Source capture and scene fingerprints now share Director timeline alias token
selection rather than substring matching. Sixty-four cases match actual parser
results, including case, prefixes, punctuation and optional strength scope.
CHECKPOINT-22 adds Enhance protection/restoration with the distinct Prompt
Studio lexical contract. Missing provider markers retain the original source
and report a failed proposal; new markers produce a review warning. Retry uses
the same snapshot. The Enhance user-prompt builder matches the source function.

Native H3 I2V (three fields) and Reference (six sections) compilers now produce
dialogue/speaker/language/voice guidance, tags, sound/music channels, timed shots,
definitions/retention and boundary instructions. **421 actual Director cases**
match prompt bytes, cues and metadata; **32 I2V text-reference projections** also
match. These are bounded source comparisons; warning diagnostics, special
Unicode IGNORECASE/control-whitespace corners and broader malformed inputs need
coverage/porting before general algorithmic acceptance.

`GET /v1/video/projects/{id}/scenes/{scene}/compiled` and native **Review** expose
the project revision, effective H3 shape, compiled prompt, source manifest and
concrete unavailable runtime inputs. Queue admission uses the same compiler and
freezes its text and project-owned Reference RGB/PCM resources. No model is loaded
by review. Protocol v7 transports ordered picture/voice inputs to the H3 tokenizer,
video/audio VAEs and reference-block conditioning. I2V's cast remains deliberately
description-only, as in Director, and that mode is visible in review. Reference
video and boundary-reference transport remain open and report their missing paths
explicitly. Standalone timeline audio now uses the same ordered audio transport.

Scene title/summary/ambient/music are retained separately in `creative.scene_prompts`
and editable in the inspector. Story no longer concatenates ambient/music into
description. Save/Open, copy/removal, Undo and fingerprints retain these fields;
display title alone does not change generation inputs. Old projects remain valid.

Edit Outline before expansion. New Scenes use the project model and copy the
selected/first Scene's matching settings slot if available. Missing settings
remain a planner blocker. Continuous Shot creates an explicit Last Frame
FirstFrame relation with audio continuity off. Model/conditioning eligibility
must still be reviewed before Runs. Replace currently fails document integrity
validation when existing Scenes have stored Takes; automatic deletion is absent.

## Director Story messages

The inventory, outline and Scene expansion user messages, and the outline
schema in the system prompt, are Director's own construction for the same
input. `src/video/authoring/director.rs` ports the server side
(`shared/prompt_studio.py` `_inventory_user_prompt`, `_build_story_outline`,
`_story_outline_user_prompt`, `_run_story_scene`/`_story_scene_user_prompt`,
`normalize_story_cast`, the continuation normalizers, `protect_references`, and
`shared/contracts/story_outline.py`'s schema) with Python's JSON spacing, key
order, int/float distinction and float repr. `src/video/authoring/steps.rs`
builds the requests Director's browser would send (`web/timeline/prompt_studio*.js`,
`story_cast.js`): cast records, `storyContinuationAvailability`,
`allocateStoryTimings`/`promptStudioDialogueTiming`, visual beat records and the
per-Scene cast filter.

- The outline carries Director's rows: task, array placement rule, character
  reference rule, the minimal JSON shape with the cast's tags (two Scenes above
  15 s, a `dialogue_intent` example with dialogue), the exact schema, total
  duration, maximum Scene duration, minimum Scene count, the music flag, the
  image inventory and visual beats, then `selected_cast`,
  `continuation_context`, `additional_instruction` and `story`.
- An expansion carries `previous_scene_summary`, `previous_scene_context`,
  `next_scene_summary`, the cast its Scene names, `continuity_cast`,
  `dialogue_timing` with dialogue, `additional_instruction` and the Scene in
  Director's layout. Images are supplied as before.
- References in the story, cast, continuation and Scene blocks are sent as
  `[[DIRECTOR_REF_nnnn]]` markers and restored in the answer, as in Director.
  Reference tags use Director's alias normalization.
- Imagine's images map to Director visuals: a setting reference is a reusable
  `story_reference`, a Required moment a `required_moment`, and start/end frames
  are placed once; each start/end pair is a directed beat `beat_<n>`. Beat keys
  are `required_moment:<id>` and `start_end_pair:beat_<n>`, moments first.
  Imagine keeps no separate user cue, so `user_note` and `cue` are empty.
- Continue sends Director's continuation context (the last Scene, the two
  summaries before it and the characters they name). A last Scene without
  authored text fails `story_continuation_text_missing`; malformed context
  `story_continuation_context_invalid`; over-limit context
  `story_continuation_context_limit`.
- A Section may carry Director's optional `summary`; it is never applied.

Evidence: `tools/director-story-messages-oracle.py` runs Director's Python
builders, with only the provider call captured, on eleven synthetic browser
requests (`tests/fixtures/director-story-messages.json`);
`tools/director-story-browser-oracle.mjs` runs the browser helpers on
continuation, timing, beat, cast and previous-Scene inputs
(`tests/fixtures/director-story-browser.json`). Rust compares both byte for
byte without Python or Node. The adaptive and one-Scene policies, visual-binding
sidecars and Director's response text normalization (Markdown and backtick
removal) are not ported; Imagine keeps its own answer validation.

## Timing and source

Ordinary Director timing uses the 250 ms minimum when possible, weighted
remaining duration and 15 second Scene capacity redistribution. Materialization
uses exact microsecond endpoints and local Section times. This Story policy is
separate from H3 generation limits. Ambient/music source is retained and appended
explicitly to authored description; authoring does not synthesize/mix audio.
Continue captures the last Scene in Director's timeline order (start, end,
id), bounded to Director's 16,000-character fields, 128 Sections and
100,000-character context. Oversized input fails rather than being truncated. Provider responses are bounded.

## Verification and remaining scope

Tests use only own synthetic loopback LLM fixtures, never external credentials,
private projects, production services or GPU. Snapshot/retry, stale Apply, exact
reviewed text, timing, expansion structure and Save/Open have regression tests.
Native UI QA is recorded separately. These make no live LLM quality claim.

P07 remains incomplete: broader dialogue lexical diagnostics, full Reference manifest/runtime projection,
full library reference metadata/editing semantics,
binding editing/acceptance for every visual role, Animate authoring, richer
field selection, catalog editor, IntelliSense/Phrase Composer, provider context
budgeting/ETA and live acceptance remain. H3 I2V applies its scene-start First
Frame. Reference picture roles use reference conditioning; this does not implement
frame-specific keyframe constraints for arbitrary Section times. Cast identity/voice
inputs now reach the actual worker; this does not establish semantic identity/timbre
quality or full Python sampling parity. Cancellation drops
the HTTP wait; keyed provider-side abort acknowledgement is not implemented.
No new GPU/model/provider-live acceptance is claimed by this checkpoint.

## Checkpoint 24: actual cast picture and voice conditioning

Admission freezes project-owned media descriptors/files along with the compiled
manifest. Queued edits cannot change their pixels, PCM or presentation order.
The supervised worker protocol is v7: initial frame and continuity bytes retain
their order, followed by reference RGB and interleaved stereo float32 PCM.
The internal audio transport uses Director's 44.1 kHz decode boundary, then the
actual Comfy sinc/Hann resampling algorithm to the audio VAE's 32 kHz boundary.
Reference images use Director's quality geometry, Pillow Lanczos and replicate
padding. Video VAE input is RGB*2-1 cast to its FP16 dtype before encode.

Nine actual-source CPU component comparisons are byteidentical. An actual
416x224, 124-frame, 20-step Reference generation with an own adult portrait and
synthetic voice waveform passes through the shared exclusive GPU coordinator to
MP4 H.264/yuv420p + AAC stereo32k, duration 5.166667 s. The separately approved
test observer records one encoded image, one encoded voice and two reference
blocks passed to video/audio conditioning. Its decoded output is byteidentical
to the ordinary worker; prepared RGB and resampled PCM also match the Python
component boundary exactly. The ordinary worker uses no observer or latent dump.

Evidence is confined to the encrypted own night root: reference-components-01,
reference-e2e-03, reference-e2e-04 and native cold UI q82. This verifies the
transport and exercised conditioning path, not complete Python Reference DiT
parity, perceived identity/timbre quality, video reference transport or all
Section/boundary/timeline-audio semantics. Those remain separate acceptance work.

## Checkpoint 23: lexical and cast-manifest parity

The prompt compiler and Enhance reference protection use CPython 3.12's
Unicode 15 alphanumeric ranges, Python whitespace and splitlines semantics.
ASCII regex patterns with IGNORECASE also recognize İ, ı, ſ and K, preserving
Director's subsequent alias, verb and context normalization. This is static
algorithm data; no Python dependency or version gate runs in the product.
The opt-in source oracle covers 2,964 Unicode boundary, whitespace, dialogue,
channel and protection cases. The previous 421 prompt and 32 I2V projection
cases remain green.

Project source now preserves `VisualReference.relationship` (marker, note and
target reference ID), `ReferenceImage.contribution`, per-asset
`Voice.reference_settings` (enabled, description, contribution, optional
strength and relationship), `Creative.visual_references_enabled`, and optional
global-prompt enable/position settings. Old projects deserialize with their
previous defaults. Cast editing exposes relationship, image contribution and
voice details without model/encoder allowlists. Global-prompt options currently
have a source/API representation; their complete settings UI remains open.

Reference selection follows Director's attribute-transfer target closure in
project order, including cycles without rejecting them. Frozen input revisions
include selected target metadata and enable/position options. The cast-only
Reference manifest includes picture reuse, per-type labels and native keys,
dialogue/speaker records, presentation order, counts and Director's informational
limit metadata. The `helto_limits` values are metadata, not a new admission gate.
The entire manifest matches the actual Director planner in 64 synthetic cases
after excluding external file paths: Imagine resolves project-owned asset IDs
at admission instead. This evidence does not cover all Section/video/timeline
audio/boundary projections or arbitrary metadata normalization.

The CP23 source evidence alone did not establish actual image/voice conditioning.
CP24 added and exercised that transport; CP25 adds the standalone timeline route
below. Neither replaces full Python sampling/execution parity or the remaining
video/boundary-reference implementation.

## Checkpoint 25: Section picture plus scene-local timeline audio

Standalone timeline references follow Director's shot-local overlap extraction,
source-in adjustment, explicit source-out clamp, fade trimming, per-track
start/id ordering and final-mix copy markers. Enabled generation references
follow cast voices in native presentation order. Their purpose, subject/speaker
target, strength and description enter the actual prompt/manifest. Final-mix
volume/fades affect the copy wording, not reference PCM gain. Source changes
that alter that wording now invalidate the generation fingerprint.

H3 settings retain Director's default `audio_reference_mode` of
`Timeline Audio References`; `Disabled` excludes standalone timeline inputs.
The scene inspector exposes this choice, and the audio inspector can select a
target character. Cast-attached voices are independent of this timeline setting.
The existing frozen-media decoder applies the projected source sample range;
no new native protocol or observer path is needed.

Thirteen focused actual-source comparisons match the entire manifest, prompt
bytes and prompt metadata, including First/Last/Influence Section pictures.
A standard worker is invoked by actual native UI Generate with an own Section
Influence image and a 0.05–0.30s timeline voice reference. It completes all twenty
steps to H.264/AAC MP4,124frames/24fps/duration5.166667s. A separate cold native
UI/server opens a copy of this actual result, decodes/previews it, plays with a
live audio process, pauses, reviews the inputs and saves. This is bounded real
execution/playback acceptance; no CP25 observer or Python model run is claimed.

The real UI trial also found `project_already_open` when reopening a saved folder
already held by the same server. Open now returns that live session snapshot for
the same bound location, preserving its revision/resources and avoiding duplicate
registration. An independent folder with a duplicate project identity still
reports the collision. A regression test fails before and passes after the fix.

Reference-video/paired soundtrack transport, broader Section role/source metadata,
boundary-manifest parity, remaining Story/catalog/provider/library UI and runtime
acceptance remain open. Character-target interaction is implemented but its new
dropdown has no dedicated native interaction acceptance yet. Exp and memory
parity gaps and the separate all-code review remain independent requirements.
