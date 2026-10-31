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
Since the corrections after `7d1de17` (I11) the language model receives these
pixels as lossless PNG (`image/png`, fixed encoder settings, so the same input
gives the same bytes), never JPEG; Director's JPEG step remains only for the
vision source oracle (`vision::director_jpeg`). This applies to inventory,
outline, expansion, Enhance, Animate and the external bundle.
This component result is not complete Director/provider/runtime parity.

## API

All routes use the existing Imagine session guard. Types live in
`crates/video-contract/src/authoring.rs`.

- `GET /v1/video/authoring/profiles`: reload source-owned catalog.
- `POST /v1/video/projects/{id}/authoring`: capture schema 1 Request and revision.
- `POST /v1/video/projects/{id}/authoring/measure`: an outline Request's budget;
  nothing starts (see "Inventory and outline budget").
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
builders, with only the provider call captured, on seventeen synthetic browser
requests (`tests/fixtures/director-story-messages.json`);
`tools/director-story-browser-oracle.mjs` runs the browser helpers on
continuation, timing, beat, cast and previous-Scene inputs
(`tests/fixtures/director-story-browser.json`). Rust compares both byte for
byte without Python or Node. The adaptive and one-Scene policies and
visual-binding sidecars are not ported.

### Answer text normalization

On the provider path the raw answer text goes through Director's own
normalization before it is read, as Director applies it:

- Outline, Scene expansion and Animate: `parse_json_object`. The text is
  stripped; a text that is entirely one code fence (` ``` `, ` ```json `,
  ` ```text ` or ` ```markdown `, any case) is unwrapped and stripped, then a
  text that is entirely one backtick span without inner backticks. It is then
  decoded with Python's `json.loads` rules (NaN and Infinity, repeated keys keep
  the first place and the last value), else the span from its first `{` to its
  last `}` is decoded; the root must be an object. Markers are restored after
  decoding, as in Director. Otherwise `authoring_invalid_json` (Animate
  `animate_response_invalid`).
- Inventory notes and Enhance text: `clean_text_response`, the same unwrapping
  and then every backtick removed and the text stripped, before markers are
  restored. An empty result fails `enhancement_empty_response`.

Nothing else is changed: Director's per-field cleanup after decoding
(`normalize_story_scene`) and its `{"scene": …}` wrapper are not ported.
External submissions have no Director counterpart and are read as before.
Twenty-five answer cases match Director's unwrapped text, decoded object and
cleaned text.

## Story context

`Request.context` (optional) carries a client's conversation and the cast's
current visible state. Without it every request, message and response is
unchanged. `GET /v1/video/authoring/{id}` echoes it in `request`.

```json
"context": {
  "earlier_continuity": [{"role": "user|assistant", "name": "", "text": ""}],
  "latest_moment": [{"role": "assistant", "name": "Ann", "text": "..."}],
  "cast_state": [{"reference_id": "<VisualReference.id in cast>",
    "current_appearance": {"exposure": "not_stated|clothed|partially_clothed|nude",
      "garments": [], "worn_items": [], "temporary_traits": [], "visible_body_state": []},
    "fallback_wardrobe": ""}],
  "protected_facts": []
}
```

- Outline: `brief` must be empty. The story Director receives is the
  conversation, one `[name] text` line per message (the role when the name is
  blank), as `<earlier_continuity>…</earlier_continuity>` (left out when empty),
  a blank line and `<latest_moment>…</latest_moment>`. `latest_moment` must not
  be empty; the rendered story is at most 200,000 characters.
- Cast state and protected facts become Director's Story bridge
  `grounding_context` (`character_id` = `reference_id`, `reference_tag` = the
  reference's normalized tag, `source_digest` = SHA-256 of the context's JSON).
  Cast state also becomes `opening_appearance_context`. Both are normalized and
  limited by Director's own rules: 64 characters, list items trimmed, empty and
  repeated items dropped, at most 12 garments, 12 worn items, 20 temporary
  traits and 20 body-state items of 4,000 characters each and 16,000 per
  character, a 2,000-character fallback wardrobe, 128 protected facts of at most
  16,000 characters, all counted in code points.
- Outline and every expansion carry `<grounding_context>`; the outline also
  carries `<opening_appearance_context>`, the "Opening appearance example" rule
  and `opening_appearance` in the shape and schema. The answer must resolve each
  character's opening appearance; it is validated as Director validates it and
  exposed, references restored, as `Draft.opening_appearance`.
- Expand and Retry rebuild their messages from the captured context, with each
  character's `current_appearance` replaced by its opening appearance
  (`exposure`, `garments`, `worn_items`, `temporary_traits`,
  `visible_body_state`) and normalized again; fallback wardrobe, protected facts
  and `source_digest` stay as captured, as Director's Story bridge client sends
  them. An oracle case matches Director's `story_scene` message for that
  grounding.
- `Review.opening_appearance` replaces it, in the same shape:
  `{"schema_version":"1.0","characters":[{"character_id","reference_tag","source","exposure","garments","worn_items","temporary_traits","visible_body_state"}]}`
  with every cast character once, in order, with its reference tag (as shown
  in the Draft) and Director's limits. Later Expand and Retry use it; Scenes
  already expanded stay until expanded again (`start_index` or `only`). An
  invalid value, or a Draft without an opening appearance, fails
  `story_opening_appearance_invalid` with `at` (for example
  `opening_appearance.characters[0].garments`) and `detail`.
- The external bundle's `slot_values.grounding_context` gives the `captured`
  grounding and the rule for `expansions`.
- Errors: `story_context_invalid` with `at` (`brief`, `context.latest_moment`,
  `context.cast_state[i].reference_id`, `context.cast_state[i].current_appearance.garments`,
  `context.protected_facts[i]`, …) and Director's reason as `detail`; an outline
  answer without a valid opening appearance fails `story_opening_appearance_invalid`.
  Enhance operations do not take a context.

Imagine keeps Director's default Scene policy (minimum Scene count from the
duration) with this context; Director's bridge itself requires its one-Scene
or adaptive policy, which Imagine does not offer. Four oracle cases cover the
grounded outline, its answers, facts-only grounding and the grounded
expansion; ten opening-appearance answer validations match Director's.

## Animate

`operation: "animate"` writes one Scene's description, first Section prompt
and ambient sound from its First frame, with Director's `animate_image` prompt
and operation settings and one model call.

- The Scene (`scene_id`) must have a first Section starting at 0 and an enabled
  First frame image, which the model sees. `duration_seconds` is 1 to 15.
- `context.latest_moment` (at most 10 messages, may be empty) becomes
  `scene_context.messages`; `earlier_continuity` and `protected_facts` must be
  empty. `cast` becomes `active_characters` (`id`, `name`, `appearance`).
  `appearance` is the reference's description, then, with a cast state,
  `Exposure: …`, `Garments: a; b`, `Worn items: …`, `Temporary traits: …`,
  `Visible body state: …` and `Fallback wardrobe: …` lines, each list only
  when not empty.
- `instruction` and `mentions` (`[{text, reference_id}]`, each in `cast`)
  become `user_instruction`; `add_dialogue` and `add_ambient_sound` become
  `audio_request`; the project canvas gives `aspect_ratio` (reduced `W:H`) and
  `orientation` (`Landscape` when width ≥ height).
- The answer must be exactly `{description, prompt, ambient_sound}`, strings,
  empty allowed; anything else fails `animate_response_invalid` and nothing is
  repaired. `Draft.animate` holds it, `Review.animate` edits it, and Apply writes
  the description, the first Section's prompt and
  `creative.scene_prompts[scene].ambient_sound` in one revision. Cancel, Retry
  and revision conflicts behave as for other Drafts.
- Errors: `animate_external_unsupported` (`provider`), `scene_not_found`,
  `animate_scene_invalid`, `animate_first_frame_missing` and
  `animate_first_frame_disabled` (`scene_id`), `animate_context_invalid` with
  `at` and `detail`.

The message is Director's `_run_animate_image` construction; two oracle cases
and five content validations match. Imagine sends the First frame as PNG at
its normalized size (longest side at most 1280), where Director passes the
client's PNG unchanged; the answer goes through Director's `parse_json_object`.

## Inventory and outline budget

Director measures an outline after its separate image inventory. Imagine offers
the same split; without the new fields an outline still runs its own inventory
and every request, message and response is unchanged.

- `operation: "inventory"` runs only the story images' inventory, as the outline
  would, and ends in `review` with `Draft.inventory`. It needs at least one
  story image and is provider-only. Review fails `authoring_not_editable`;
  Apply fails `authoring_not_applicable`. Cancel and Retry behave as for an
  outline's inventory.
- `Request.inventory` (outline, provider only) gives one known note per story
  image, as an inventory Draft wrote them. The outline then analyses no image
  and its message carries those notes. A missing, repeated or unknown note fails
  `authoring_inventory_mismatch`.
- `POST /v1/video/projects/{id}/authoring/measure` takes an outline Request
  whose story images all have known notes (else `authoring_inventory_mismatch`
  at `inventory`) and builds the outline's system and user messages exactly as
  its start would, without sending them:

  ```json
  {"schema_version":1,"operation":"outline","chars":81234,"char_limit":100000,
   "text_bytes":90312,"message_count":2,"context_tokens":32768,
   "max_output_tokens":8000,"context_reserve_tokens":8,"fingerprint":"sha256:…"}
  ```

  `chars` counts the user message, which `char_limit` (`story_context_limit`)
  bounds; `text_bytes` both messages. `max_output_tokens` is the outline's
  resolved `max_tokens`; the reserve is Director's 8 tokens. `context_tokens` is
  read as Director reads it: the provider profile's `{base_url}/models`, the
  entry whose `id` is the profile's model, its `meta.n_ctx`. A provider other
  than `koboldcpp`, no such single entry or no positive `n_ctx` fails
  `authoring_context_window_unknown`; an unreachable provider keeps its
  enhancement error. The fingerprint covers both messages, the transport, the
  provider profile's address and model, the resolved settings, the prompt
  profile and its revision, and the budget. Measuring the same request again
  gives the same result. A Request with `budget_fingerprint`, an external
  provider or another operation fails `invalid_authoring_request`.
- An outline start with `budget_fingerprint` measures again first and fails
  `authoring_budget_changed` (409) when the result differs; nothing starts.
- The authoring limits add `char_limit`.
- `Request.outline` (outline, provider only) starts from a reviewed outline
  instead of the provider's, for example a client's saved Storyboard edited
  again. It needs a known note per story image (`authoring_inventory_mismatch`).
  `Request.opening_appearance` may give its opening appearance for exactly the
  context's cast state (else `story_opening_appearance_invalid`); without it the
  Scenes start from the context's current appearance. The context may leave its
  conversation empty, since no outline is written from it. The outline is
  validated as a review of it is, plus its cast as an outline answer.
  The Draft settles in `outline` at once, with no provider request and no busy
  check; Expand then writes its Scenes through the provider. Measure rejects it
  (`invalid_authoring_request`).

The client decides whether a measurement fits. The native Story view's
Generate outline runs the inventory Draft (when story images are selected),
measures with its notes and starts the outline with the fingerprint only when
`chars` is within `char_limit` and an estimated input (a third of
`text_bytes`, 12 tokens per message and 32, plus 15%) fits within
`context_tokens - max_output_tokens - context_reserve_tokens`; otherwise it
shows both numbers. It shows the measured budget beside the started outline.
Generate again measures again and reuses the known notes. Tests use synthetic
loopback providers only.

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
binding editing/acceptance for every visual role, richer
field selection, catalog editor, IntelliSense/Phrase Composer, provider ETA
and live acceptance remain; the outline's context budget is measured as in
Director, and expansions, like Director's, are bounded only by `char_limit`. H3 I2V applies its scene-start First
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
