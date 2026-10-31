# MCP video production: implementation plan

Status: **Plan. Implementation starts only when explicitly requested.**

Date: 2026-10-24. Companion to [MCP-VIDEO-DESIGN.md](MCP-VIDEO-DESIGN.md).

This plan turns the design into ordered work packages with files, changes,
tests and completion criteria. It is based on source inspection of the current
checkout, including the NSFW prompt profile. Nothing here was run.

## 0. Ground rules for every work package

- **Prompt profiles are served verbatim.** Nothing under
  `config/video/prompt-profiles/` is edited, shortened, sanitized or
  softened. The agent receives exactly the text the local provider receives;
  the only substitution is the existing `{{STORY_OUTLINE_JSON_SCHEMA}}`
  placeholder, as today. A test enforces byte equality (WP3). Agent-specific
  guidance lives in a separate shared file, never inside a profile.
- **No GPU before generation.** Authoring on the MCP path never calls the local
  authoring LLM and never requests the GPU. The complete storyboard — outline,
  scenes, sections, references, models and transitions — is written and
  validated before `video_run start`, which is the only GPU handover.
- **No runtime data** is read during development or validation (`AGENTS.md`).
  Tests use synthetic projects, synthetic media and source-owned temporary
  private roots (the pattern in `src/video/private_project_tests.rs`).
- **No inference, native or lock changes.** All work is in the API server, UI,
  proxy and shared contract crate, which follow `main` in normal startup
  (`docs/PRODUCT-BUILDS.md`). `native/product/build-lock.json` is not touched.
- **Validation per package:** `cargo fmt --all --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`, the package's focused
  tests, then `./build-product.sh` as build-only preparation. No service restart
  or GPU run is part of a package; live acceptance is a separate, explicitly
  authorized step (section 4).
- Existing lint or test failures met along the way are fixed, not skipped.
- **Test environment.** Many tests require explicit synthetic roots inside the
  approved task root from `shared/task_fixtures.rs`, and Unix socket tests need
  short paths:
  `TMPDIR`, `IMAGINE_WORKER_A_FIXTURES`, `IMAGINE_LORA_TEST_ROOT` and
  `IMAGINE_WORKER_A_EVIDENCE`, each an existing short directory under that
  root. Without them about 90 tests fail with fixture or `SUN_LEN` errors; with
  them the baseline workspace suite is green (verified 2026-10-24).

## 1. Design corrections found while planning

| # | Design said | Source shows | Consequence |
| --- | --- | --- | --- |
| C1 | The agent submits the Draft content; the server parses it | The provider path builds each message in code: the expansion message for scene N embeds scene N−1's expanded result (`drafts.rs:545-553`) and scenes with images send the normalized images (`drafts.rs:642-661`). Expansions must keep code-owned scene/section IDs and image fields (`engine.rs:102-125`) | A **context bundle** gives the agent every prompt, rendered context block, schema and image up front; the agent submits everything once; the server fills code-owned fields and parses with the same functions. See B1 |
| C2 | Run requests have no idempotency key; put it in the proxy | Run start and assembly submit **require** `Idempotency-Key` and return `idempotency_conflict` on reuse with a different request (`runs_api.rs:27-33`, `scheduler_video_runs.rs:15-29`, `scheduler_assembly.rs:10-24`) | The proxy forwards the agent's `job_id` as a namespaced key. No proxy fingerprints for video |
| C3 | MCP saves after apply, per scene and after assembly | Bound projects autosave on every edit, authoring apply and take review (`persistence.rs:759-806`); runs accept takes through `review` (`runs.rs:203`); private projects are bound at creation (`project.rs:31-41`) | The MCP checks the snapshot's save status and reports it; it saves explicitly only when status is not saved |
| C4 | NSFW outline guidance is aligned to the 15 s cap | The outline user message already states "Every Scene must fit 15 seconds" from code (`drafts.rs:582`) | **No template edit.** The cap becomes one constant used by the user message, `durations()` and capabilities |
| C5 | Listing shows project titles | `VideoProject` and `ProjectDocument` have no title | Listing shows last-saved time, scene count and the first scene title; a project title is optional later work |
| C6 | Save location follows policy | In private storage mode, private stores must be inside `/run/media/veracrypt1/imagine` (`bundle.rs:296-317`) and a layout-1 project must sit at `<private root>/projects/<id>` (`persistence.rs:962-969`, `api.rs:334`). The default MCP input directory `/run/media/veracrypt1/mcp-inputs` is on the same encrypted volume but outside that root | **Decided: the agent gives the MCP a project path.** The server accepts it when it is inside a configured project root on the verified encrypted volume (WP5). The user opens the result with the existing “Open project…” folder picker |

### B1. Context bundle and one submission

1. **Start** (`video_storyboard start`) creates an authoring Draft with
   `provider: "external"` and returns the **context bundle**:
   - `inventory`: the profile's `story_image_inventory` system prompt and the
     per-image user message, for every story image.
   - `outline`: the outline system prompt (dialogue variant selected by
     `add_dialogue`, schema placeholder filled) and the rendered user message.
     Its `<image_inventory>` block is marked as filled from the agent's own
     inventory notes.
   - `expansion`: the scene-expansion system prompt (dialogue variant) and the
     user message rendered with every fixed block (`<selected_cast>`,
     `<continuity_cast>`, `<additional_instruction>`, `<image_sources>`,
     `<visual_beats>`, visual-context rule), with three marked per-scene slots:
     `<previous_scene_context>` (the agent's own expansion of the previous
     scene; continuation context for scene 1 when `destination: continue`),
     `<next_scene_summary>` (from the agent's outline) and `<scene_json>` (the
     agent's outline scene).
   - `images`: every story image as the same normalized JPEG the provider gets,
     labelled with image ID, usage and label; plus enabled cast identity images,
     labelled as identity references.
   - `profile`: ID, revision and prompt files; `limits` from capabilities.
2. The agent writes everything in one pass: inventory notes, the outline (in the
   outline schema), one expansion per scene (in the `StoryScene` shape), and the
   **production block** (model per scene, continuity per continuous boundary).
3. **Submit** (`video_storyboard submit`) sends it all in one call with
   `dry_run`. The server, using the provider path's functions in order:
   stores the inventory notes, parses and projects the outline (assigning
   `scene_NNN` IDs and image bindings, checking cast and durations), fills each
   expansion's code-owned fields (scene/section IDs, image IDs, frame IDs) from
   the projected outline, validates it with `expansion()` and `validate_cast`,
   then builds the apply plan with the production block and validates it on a
   candidate project. All-or-nothing: errors name `outline`, `expansions[i]` or
   `production.scenes[i]` with a stable code and change nothing. With
   `dry_run: false` and no errors, the plan is applied atomically.
4. The Draft keeps the parsed result, so `video_storyboard get` and the existing
   review endpoint remain available for fixes before a later submit.

The agent's expansion context is equivalent to the provider's: the provider
message's per-scene blocks hold the previous expansion and outline values,
which the agent wrote itself. A test proves that filling the three slots with
those values reproduces the provider's message byte for byte.

## 2. Work packages

Order: WP0 → WP1 → WP2 → WP3 → WP4 → WP5 → WP7 → WP8. WP5 can run in parallel
with WP3–WP4. WP6 is optional (Stage 3).

### WP0 — Update the design document

- Apply C1–C6 and B1 to `docs/MCP-VIDEO-DESIGN.md`.
- Done when the design and this plan agree.

### WP1 — One scene cap, extended capabilities

Files: `src/video/authoring/engine.rs`, `src/video/authoring/drafts.rs`,
`src/video/api.rs`, `src/video/adapters/h3.rs`.

- Add `pub const MAX_SCENE_SECONDS: f64 = 15.;` in `engine.rs`. Replace the
  literal `15.` in `engine.rs:253`, `drafts.rs:289` and `drafts.rs:732`, and
  format the outline user message (`drafts.rs:582`) from the constant. The
  rendered text stays byte-identical.
- Extend `GET /v1/video/capabilities` with an `authoring` object:
  `max_scene_seconds`, `max_scenes` (128), `max_sections_per_scene` (12),
  `supported_boundaries: ["hard_cut","continuous_shot"]`, and a `continuity`
  table derived from `h3::variants()`: per target variant, its continuity
  methods and the `auto` resolution (`i2v` → `last_frame_first_frame`,
  `reference` → `motion_context_exp`, today's Story default).
- Tests: capability assertions in `tests/video_api.rs`; existing `durations`
  tests; a golden test that the outline user message is unchanged.
- Done: one constant, capabilities report it, no behaviour change.

### WP2 — Extract message building and answer parsing

Files: `src/video/authoring/drafts.rs`, new `src/video/authoring/steps.rs`.

Pure refactor; the provider path must behave exactly as before.

- Move the user-message construction out of `Service::generate`
  (`drafts.rs:465-661`) into pure functions in `steps.rs`:
  `inventory_message`, `outline_message`, and `expansion_message` taking the
  per-scene values (`previous`, `next_summary`, `scene_json`) as parameters so
  they can be rendered with real values or with marked slots. Dialogue variants,
  the image-context replacement, `<image_sources>`, `<image_inventory>`,
  `<visual_beats>` and the selected-cast substitution move unchanged.
- Move answer handling (`drafts.rs:698-753`) into `accept_inventory`,
  `accept_outline` and `accept_expansion` on `&mut Draft`, calling the existing
  `outline_with_images`, `expansion`, `validate_cast` and `durations`.
- `generate` becomes: build message → provider call → accept. Cancellation,
  generation counters, restore and progress stay in `generate`.
- Tests: `tests/support/video_authoring.rs` and `tests/video_prompt_profiles.rs`
  pass unchanged; golden tests of each message for a synthetic Draft, committed
  before the refactor.
- Done: provider path identical; messages and parsing callable without a
  provider.

### WP3 — External provider: context bundle and submission

Files: `crates/video-contract/src/authoring.rs`, `src/video/authoring/drafts.rs`,
`src/video/authoring/steps.rs`, `src/video/authoring_api.rs`, new
`config/video/agent-guidance.txt`.

- **Start.** `Service::start` accepts `provider == "external"` (not added to
  `profiles::PROVIDERS`, which drives generation parameters):
  `provider_profile_id` may be empty; no task is spawned; the Draft starts in
  `Phase::Outline` with `outline: None`; `provider_snapshot` records
  `{"profile_id":"external"}` plus the resolved `koboldcpp` generation
  parameters as informational metadata. Strict profile: `fallback: true` from
  `Catalog::snapshot` fails with `profile_not_found`; the provider path keeps
  its fallback. External Drafts are never `busy()`.
  The start response includes the context bundle (B1), built from WP2's
  message functions and `Grounding::capture` (CPU image normalization only).
  Cast identity images come from enabled `VisualReference.images` of the
  selected cast, normalized the same way.
- **Submit.** `POST /v1/video/authoring/{id}/submission` with
  `{schema_version, expected_project_revision, inventory: [{id, note}],
  outline: text, expansions: [text], production, dry_run}`:
  1. inventory notes must cover exactly the story image IDs
     (`authoring_inventory_mismatch`); stored through `accept_inventory`;
  2. `accept_outline(outline)`;
  3. `expansions.len()` equals the outline scene count
     (`story_expansion_count_mismatch`); each expansion is parsed as JSON, its
     code-owned fields (`id`, section `id`s, `image_ids`, `first_frame_id`,
     `last_frame_id`) are set from the projected outline scene, then
     `accept_expansion` runs the unchanged structural checks (weights, section
     count, boundary, dialogue participants);
  4. the Draft is now in Review; WP4 builds and validates the apply plan.
  Any error returns `{error: code, at: "outline" | "expansions[i]" |
  "production.scenes[i]", detail}` and restores the Draft's previous state.
  Size limits match the provider path (`story_context_limit`,
  `story_text_limit`).
- **Shared agent guidance.** `config/video/agent-guidance.txt`
  covers only what profiles do not: the bundle and its marked slots, the
  code-owned fields, the production block, Reference/I2V selection, boundary
  mechanisms, cast identity images, and Develop versus Adapt for open ideas
  (the agent passes a developed or supplied story as `brief`). No content
  restrictions; nothing that contradicts or overrides profile text. Exposed by
  `describe_video_profile`, never sent to the local provider.
- **Contract.** `Draft` gains `#[serde(default, skip_serializing_if =
  "Option::is_none")] production: Option<Production>`; `Production { scenes:
  Vec<ProductionScene> }`, `ProductionScene { model: Variant /* reference | i2v
  */, continuity: Option<ContinuityChoice>, rationale: String }`,
  `ContinuityChoice { method: Auto | ContinuityMethod, include_audio: bool }`.
  `Draft` has no `deny_unknown_fields`, so stored documents still load.
- Tests (new `tests/support/video_authoring_external.rs`, synthetic project and
  images):
  - **Verbatim profile text:** for `nsfw`, the bundle's system prompts equal
    `story_outline.txt`, `story_outline_dialogue.txt`, `story_scene.txt`,
    `story_scene_dialogue.txt` and the inherited inventory prompt with only the
    schema placeholder replaced.
  - **Message equivalence:** filling the expansion slots with the provider's
    values reproduces the provider path's message for every scene; the outline
    message with the agent's notes equals the provider's.
  - **Parse equivalence:** one set of answers submitted externally and returned
    by the fake `/v1/chat/completions` provider yields identical Drafts and
    identical apply plans.
  - Bundle images equal `Grounding::capture` output; cast images are labelled.
  - Errors: unknown profile, malformed outline, wrong expansion count, changed
    weights or section count (`story_expansion_changed_structure`), inventory
    mismatch — each with the right `at` and the Draft unchanged.
- Done: an agent can author a complete storyboard in two calls, with no
  provider and no GPU.

### WP4 — Production block, dry run and apply

Files: `src/video/authoring/engine.rs`, `src/video/authoring_api.rs`,
`src/video/persistence.rs`, `crates/video-contract/src/editing.rs` (if needed).

- In `engine::apply` (`Operation::Outline` branch), when the Draft has a
  production block:
  - Length must equal the scene count (`production_scene_count_mismatch`).
  - `model`: map `reference`/`i2v` to the `ModelSelection` from
    `h3::variants()`; use `ModelChoice::Override` when it differs from
    `base.default_model`, followed by the existing
    `Edit::InheritModelSettings { scene_id, model }` so settings slots match.
  - `continuity` for `continuous_shot` scenes: resolve `auto` with the WP1
    table; reject a method not listed for the target variant
    (`continuity_method_unsupported`); for an `i2v` target, reject an explicit
    first-section `start_frame` (`continuity_first_frame_conflict`).
    `include_audio` defaults to false.
  - `i2v` scenes: reject any binding other than one opening First Frame
    (`i2v_input_role_unsupported`, with scene and section ordinal); require a
    start frame or last-frame continuity (`i2v_start_frame_required`).
  - Without a production block, behaviour is unchanged (`ProjectDefault`,
    `MotionContextExp`).
- Dry run: factor the candidate construction out of
  `ProjectSession::apply_authoring` (`editing::apply`,
  `review::invalidate_changed`, `validate()`), run it on a clone, then
  `execution::preflight` for each new scene. Return `{valid, scenes:
  [{ordinal, model, continuity, planned_seconds, blockers}], warnings}`; commit
  nothing. Used by the WP3 submission and by the existing `apply` endpoint
  (which gains optional `production` and `dry_run`).
- Commit returns the snapshot plus `scene_map` (ordinal → scene and section
  IDs).
- Tests: mixed `reference`/`i2v` storyboard with correct `ModelChoice` and
  settings; every rejection code; dry run leaves the revision unchanged and
  reports the plan that commit applies; provider-path apply without production
  unchanged (existing tests).
- Done: per-scene models and transitions are applied and validated before any
  GPU work.

### WP5 — Agent-selected project location

The agent passes an absolute project path to `video_project create`; the whole
production (project document, outputs, work files) lives there.

Files: `shared/private_storage.rs`, `shared/storage_layout.rs`,
`src/video/bundle.rs`, `src/video/persistence.rs`, `src/video/project.rs`,
`src/video/api.rs`, `src/config.rs` (server configuration).

- **Approved project roots.** New server setting `private_project_roots`
  (list of absolute paths), default `["/run/media/veracrypt1/mcp-inputs"]`, the
  MCP input directory. At startup each root must be inside
  `layout::PRIVATE_MOUNT` on the proven encrypted volume; otherwise it is
  ignored with a startup warning. The private root itself is always approved.
- **One guard.** Add `Root::allows(path)` in `shared/private_storage.rs`: true
  when the proof holds and the canonical path is inside the private root or an
  approved project root. Replace the two `starts_with(root.layout().root())`
  checks in `Store::select_private` and `Store::guard_private`
  (`bundle.rs:301`, `bundle.rs:314`) with it. Paths outside the encrypted
  volume stay `invalid_storage_root`; the encryption policy is unchanged.
- **Layout at a path.** Add `Layout::project_at(path) -> Project` with the same
  subpaths as `Layout::project(id)` (`outputs/images`, `outputs/videos`,
  `outputs/audio`, `work`, `tmp`, `autosave`). Persistence resolves media paths
  for a layout-1 project from its bound root instead of from its ID.
- **Locations.** `POST /v1/video/locations` gains optional `"create": bool`:
  the leaf directory is created when missing, only if its parent exists and,
  in private mode, the path is allowed by `Root::allows`.
- **Create at a path.** New route `POST /v1/video/projects/at-location` with
  `{"schema_version":1,"location_id":"…","project":null|CreateProject}`. In
  private mode `ProjectService::create_private_at(request, root, path)` binds
  the new session there (the directory must be empty) and saves immediately, so
  autosave covers everything after it; otherwise create plus `save_to`. A null
  `project` uses the UI's default project, moved to a shared function that the
  UI's `new_project()` also calls. `POST /v1/video/projects` is unchanged.
- **Open from a path.** In `open_for_private` (`persistence.rs:962-969`) accept
  a layout-1 project whose path is either its `<private root>/projects/<id>`
  directory or inside an approved project root. The UI's existing
  “Open project…” folder picker then opens agent projects with no UI change.
  `save` and `save-as` keep their current rules.
- Tests (synthetic private root and synthetic approved root under a test mount,
  as in `src/video/private_project_tests.rs`):
  - create at an approved path, apply edits, reopen from that path;
  - a path outside approved roots, a non-empty directory, and a relative path
    are refused with stable codes;
  - an approved root outside the encrypted mount is ignored at startup;
  - existing private projects in `imagine/projects/<id>` behave as before.

### WP6 — Optional: private project list in the UI (Stage 3)

Not needed for the agent hand-off, because the agent reports the project path
and the folder picker opens it. Useful later for projects in Imagine's own
private root:

- `GET /v1/video/projects/private` listing `imagine/projects/*` and approved
  roots (metadata only: project ID, saved time, scene count, first scene title,
  duration, live-session flag), and an “Open private project” window in
  `ui/src/video/` following the Helto design system.

### WP7 — Proxy: video tools

Files: `proxy/src/mcp.rs` (dispatch), new `proxy/src/mcp_video.rs`,
`config/mcp/tools.json`, `config/mcp/instructions.txt`, `proxy/src/imagine.rs`.

| Tool | Actions → endpoints |
| --- | --- |
| `list_video_profiles` | `GET /v1/video/authoring/profiles` |
| `describe_video_profile` | new `GET /v1/video/authoring/profiles/{id}` (strict): profile metadata, `shared/agent_video.txt`, authoring limits |
| `video_capabilities` | `GET /v1/video/capabilities` |
| `video_project` | `create` with `path` → proxy checks the path is absolute and inside its MCP roots, then `POST /v1/video/locations` (`create: true`) and `POST /v1/video/projects/at-location`; `get` → `GET /v1/video/projects/{id}` (bounded projection: scenes, sections, take summary, save status); `edit` → `POST …/edits`; `save` → `POST …/save` |
| `video_register_reference` | path checked against the MCP roots (`references::read_image`), `POST /v1/video/assets/link`, `POST /v1/video/projects/{id}/assets`; optional `cast: {name, alias, description}` adds a cast entry |
| `video_storyboard` | `start` → `POST /v1/video/projects/{id}/authoring` (`provider: "external"`), returns the bundle; `submit` → `POST /v1/video/authoring/{id}/submission`; `get` → `GET /v1/video/authoring/{id}`; `review` → `POST …/review` |
| `video_run` | `start` → `POST /v1/video/runs` with `Idempotency-Key: mcp-<session>-<job_id>`; `get` (optional bounded wait) → `GET /v1/video/runs/{id}`; `stop` → `POST …/stop` |
| `video_assembly` | `plan` → `POST /v1/video/projects/{id}/assembly`; `submit` → `POST /v1/video/assemblies/{id}/jobs` with the namespaced key; `get` → `GET /v1/video/assemblies/{id}` |

- **Cast entries:** add `Edit::UpsertVisualReference { reference }` to
  `crates/video-contract/src/editing.rs` (applied in `src/video/editing.rs`), a
  typed, revision-checked edit instead of a proxy read-modify-write.
- **Images:** `tool_result` gains image content blocks
  (`{"type":"image","data":…,"mimeType":"image/jpeg"}`) for the bundle's images.
  Image MCP delivery rules are unchanged.
- **Timeouts:** `Mcp::call` applies 5 s to every operation (`mcp.rs:275`).
  Video tools get a per-action table: reads 5 s; `storyboard start` (image
  normalization) and `submit` 60 s; `run get`/`assembly get` with wait up to
  60 s per call.
- **Session:** existing refresh and epoch handling; no video state in the proxy.
- **Logging:** no prompts, answers, images or paths; stable codes only.
- **Instructions:** video section in `instructions.txt`: workflow order, bundle
  and single submission, production block, no GPU before `video_run start`,
  `job_id` recovery, choosing a new or empty project folder inside the MCP
  roots, the final report (project path first), the separation from the
  image MCP `nsfw` switch, image MCP results registered by `saved_path`.
- Tests: probe-server Rust tests (pattern in `proxy/tests/mcp_recovery.rs`) for
  routing, key namespacing, timeouts, image blocks, error forwarding and session
  reset; catalog schema validity.

### WP8 — End-to-end synthetic test

Files: new `tests/mcp_video.rs` (`st-proxy` is already a dev-dependency of the
server crate).

- Server test harness with a synthetic private root, a synthetic approved
  project root and fake H3 execution (pattern in `tests/video_jobs.rs`), proxy
  against it. Drive: create project at a path in the approved root →
  register two synthetic images (cast, setting) → `storyboard start` → one
  `submit` with inventory notes, outline (hard cut and continuous shot), both
  expansions and a mixed `reference`/`i2v` production block, `dry_run: true`,
  then `dry_run: false` → run → assembly → session restart → reopened from the
  same path through `POST /v1/video/projects/open` with a location.
- Assert: no authoring-provider request and no GPU-coordinator request before
  `video_run start`; scene models and continuity as requested; takes accepted;
  saved after each step without explicit save; repeated run start with the same
  `job_id` returns `reused`; different input returns `idempotency_conflict`;
  session restart clears proxy state and the saved project reopens.

## 3. Later stages

### Stage 2 — explicit continuity and recovery

- Explicit continuity methods, `include_audio`, and capability reporting of
  source compatibility.
- Recovery support for existing run modes; retry budget per project per session.
- Default production budget (design open decision 1).

### Stage 3 — more profiles and capabilities

- `normal`, `creative`, `nsfw_prism`, `nsfw_qwen_next` work once WP3 lands
  (the path is profile-agnostic); add their verbatim-text tests and live checks.
- `blend_seam`, `transition`, video and audio references when supported.
- WP6: private project list and UI window.
- Optional project title if the first-scene title proves insufficient.

## 4. Live acceptance (separately authorized)

1. Prepare with `./build-product.sh`, then restart services deliberately.
2. One short NSFW-profile production through a real MCP client with a hard cut,
   a continuous shot and a mixed `reference`/`i2v` pair; confirm the GPU is not
   requested before `video_run start`.
3. Interrupt the client mid-run and recover with the same `job_id`.
4. Open the result in the UI from the reported path, edit a scene, regenerate
   it, reassemble.
5. Record static/build evidence and live evidence separately.

## 5. Implementation status (2026-10-24)

WP1–WP5, WP7 and WP8 are implemented, uncommitted on `main`. WP6 (private
project list in the UI) remains optional Stage 3 work. Decisions made during
implementation, beyond the text above:

- **Bundle slots** are written `{{SLOT:name}}`. Expansion messages have
  `previous_scene_context`, `next_scene_summary`, `scene_json` and, when story
  images exist, `image_inventory` and `image_context` (the per-scene visual
  rule); the outline has `image_inventory`. `expansion.slot_values` supplies the
  first scene's `previous_scene_context` and both `image_context` texts.
  Images: story images first (`role: story`), then cast identity images
  (`role: cast`, `usage: identity_reference`).
- **System prompts** are the profile files with only the schema placeholder
  substituted; the provider path trims trailing whitespace, so the only other
  difference from the file bytes is the final newline. A test covers all five
  profiles and both dialogue variants.
- **I2V opening frame:** a `start_frame` image paired with itself
  (`pair_with` = its own ID) on an I2V scene's first section becomes the First
  Frame only. The existing image model otherwise always pairs start with end.
- **Extra error codes:** `continuity_requires_continuous_shot`,
  `continuity_endpoint_incompatible` (`include_audio` with last-frame
  continuity), `authoring_not_external`, `authoring_not_generated` (expand or
  retry on an external Draft). `at` also takes `expansions` and
  `production.scenes` for count mismatches.
- **Pending continuity sources** (source scene without an accepted take) are dry
  run warnings (`continuity_source_pending`), not blockers.
- **A successful dry run updates the Draft** (Review phase, parsed result,
  production), never the project. Submission `production` is optional;
  `dry_run` is required.
- **Project roots:** `imagine-server --private-project-root PATH` (repeatable;
  replaces the default `/run/media/veracrypt1/mcp-inputs`). Roots outside the
  verified mount are ignored with a startup warning. `Root::allows` also checks
  that the opened directory is on the proven device without following symlinks.
  `<root>/projects/**` and `<root>/library/**` are refused as agent locations.
- **Outputs** of runs and assembly for a located project publish under its own
  `outputs/` folder (`Store::bind_project_location`).
- **Default project** moved to `shared/video_defaults.rs`; the UI's
  `new_project()` uses it.
- **Snapshot** `persistence.location` reports the bound folder.
- **Proxy:** video tools release the session gate after the session check, so
  long waits don't block other calls. Cast entry IDs are derived from the alias
  (`reference_mcp_<hash>`), so re-registering an alias replaces the entry. The
  attach endpoint does not return the new project asset ID; the proxy
  identifies it as the one new asset bound to the linked session asset.

- **Recovery after timeouts:** `POST /v1/video/projects/at-location` returns
  200 with the live project when a session is already bound at that folder, so
  a retried create succeeds. `GET /v1/video/projects/{id}/authoring` lists a
  project's session Drafts (id, phase, provider, profile; no content) and
  `GET /v1/video/authoring/{id}/bundle` rebuilds an external Draft's bundle;
  the proxy exposes them as `video_storyboard list` and `bundle`.
- **Project folders** are created only inside the MCP input directory (the
  proxy no longer offers the media directory, which is under `projects/`). A
  folder inside another project (`project.json` between it and the approved
  root, or inside an open project's location) is refused with
  `project_location_nested`; prepared `outputs`/`work`/`tmp`/`autosave`
  subfolders must contain only empty folders.
- **Errors with location:** an empty external `prompt_profile_id` is
  `profile_not_found`; a first scene with `continuous_shot` and nothing to
  continue is `story_continuity_source_missing` at `outline`, `scenes[0]`;
  `story_scene_duration_capacity` carries `detail: "minimum_scenes=N"`; other
  story errors from apply are reported at `outline`.
- **Bounded results:** the proxy returns a bounded project projection plus
  `scene_map` for a committing submission instead of the full snapshot.
- `capabilities.project_operations` lists `create_at_location`.
- A review that changes an external Draft's outline clears its production
  block.

### Verification (2026-10-24)

- Workspace: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets
  --offline -- -D warnings` and `cargo test --workspace --offline --no-fail-fast`
  pass (777 tests, 0 failed, 6 ignored) with the section 0 test environment.
- `tests/mcp_video.rs` drives the real proxy against an in-process server with
  fake H3 execution: create (and idempotent retry) → register cast and setting
  images → storyboard start with verbatim system prompts and image blocks →
  error forwarding → dry run → commit with mixed `reference`/`i2v` and
  last-frame continuity → **no LLM, coordinator, H3 or GPU-owner activity before
  `video_run start`** → run with idempotent replay and conflict → assembly →
  outputs under the project folder → saved after every step → session restart
  and reopen from the folder.
- Not covered: real inference/GPU (live acceptance, section 4), the UI folder
  picker itself (covered at its HTTP boundary), interrupting a running run.
- No file under `config/video/prompt-profiles/` changed.

### Live acceptance and follow-up fixes (2026-10-24)

Headless Claude sessions with only the Imagine MCP tools ran the complete
flow on the real GPU: they generated cast images, created projects, authored,
generated and assembled 20-second videos.

- **Run 1** (Quick Draft): succeeded; the agent used I2V for a continuation, which
  inlined full cast descriptions into the I2V prompt. Guidance now defaults to
  `reference`, I2V only for a specific start image; dry runs warn with
  `i2v_cast_identity_text_only`; `video_project create` takes a `canvas` size
  preset; instructions shortened after the client truncated them.
- **Runs 2–3** (Draft): every continuous shot into a `reference` scene failed
  with any motion method (also in the UI). To diagnose it, H3 protocol 17 adds
  bounded failure diagnostics (`stage`, `kind`, exception class, accelerator
  code, up to 8 integers; never native text) from the worker through the job
  API, the UI error message and MCP `video_run get` (`failed_scene`). They
  showed `sampling` / `internal`.
- **Cause and fix:** `motion::Window` ignored keyframe and reference-block rows,
  so its length check failed whenever cast images were present. The window now
  accounts for `[text][keyframes][reference blocks][audio][video]`, sharing
  `reference_window::rows` with the boundary window.
- **Run 4** (Draft): all three scenes including a `motion_context_exp`
  continuous shot with two cast references generated; the join is seamless and
  identities are consistent across scenes.

### Scene presets (2026-10-25)

MCP productions choose a named scene preset per scene instead of a bare
variant. Presets live in `config/video/presets.yaml` (gitignored; tracked
example `config/video/presets.example.yaml`; server flag `--video-presets`)
and are read on each use, so edits apply without a restart. A preset has a
readable `key`, a `variant` (reference or i2v), a `description`, `settings`
overrides (model file and other components, sampling, shifts, reference
quality, stage two) layered on the project's settings for that model, and
`loras` with a `lora_merge` mode. `Default ref2v` and `Default i2v` are built
in and keep today's behaviour. Capabilities, profile descriptions and bundles
list presets (key, variant, description, availability; never file paths);
invalid presets are listed with a reason. The production block takes
`preset`; `model` alone means that variant's default preset. Dry runs block on
missing model or LoRA files. Agents are told to use a non-default preset only
when the user names it. A scene preset's LoRAs add to the project's global
LoRAs unless its `lora_merge` says otherwise (default `Add To Global`).

### Presets in the UI (2026-10-25)

Implemented per [VIDEO-PRESETS-UI-DESIGN.md](VIDEO-PRESETS-UI-DESIGN.md).
`GET /v1/video/presets` returns the full catalog for the local UI (settings,
model and LoRA paths, merge mode, availability, file path and file problem).
`POST /v1/video/projects/{id}/presets/plan` turns a preset into the edit batch
for scenes or the project defaults with the same code as MCP; the UI submits it
through the normal edit path, so it is one undo step and revision-checked.
Scenes record the applied preset (`creative.scene_presets`, key and
fingerprint); snapshots report `preset_states` (`applied` or `modified`). The
scene inspector has a Preset dropdown (all selected scenes, Modified and
Reapply), the Project generation window a Default preset dropdown (mode,
project settings and global LoRAs), and the project menu a Video presets
window with Reload.

### Open private project window (2026-10-25)

WP6 is implemented. `GET /v1/video/projects/private` lists saved projects in
Imagine's private root (`projects/*`) and in the approved project roots (agent
folders), newest first: path, folder, project ID, scene count, duration, first
scene title, takes, deliveries, last saved time, source (`imagine` or `agent`)
and whether a session has it open; unreadable folders are listed with an error
code. Projects open in a session are summarized from memory; others are read
read-only. In private storage mode, "Open project…" shows an "Open private
project" window with this list; Open uses the existing location flow, and
"Browse…" keeps the folder picker as a fallback. Names and titles are concealed
in privacy mode.

### Agent workspaces (2026-10-25)

Agents no longer use host paths. Each task runs in a workspace under
`/run/media/veracrypt1/mcp-root/<date>/<time>-<title>-<id>/`, reached only with
its `workspace_key`; files move through `workspace_*` tools and the proxy's HTTP
file routes, and results name files by ID. Agent project folders are
`videos/<title>` inside the workspace. This supersedes the agent-selected
project folders above (`video_project create` takes a title, not a path) and
removes `confirm_saved`. See [MCP-WORKSPACES-DESIGN.md](MCP-WORKSPACES-DESIGN.md).

### Interactive video mode (2026-10-25)

Agents can plan a video with the user step by step: the `video_interactive` MCP
prompt or plain words switch it on; `video_capabilities` reports `mcp_defaults`
(15 s total by default); characters without images become character sheets
generated in one batch or are described in the prompts, asked after the outline
is accepted. See [MCP-VIDEO-INTERACTIVE-DESIGN.md](MCP-VIDEO-INTERACTIVE-DESIGN.md).

Remaining open points: request-body deserialization errors return only
`invalid_request` without field detail (serde messages can echo user text);
`video_run start` omits `reused` on a first start while assembly returns
`reused: false`.

## 6. Risks

| Risk | Mitigation |
| --- | --- |
| WP2 refactor changes provider prompts | Golden tests committed first; provider tests unchanged |
| Agent context differs from provider messages | Slot-filling equivalence test (WP3) |
| External answers bypass a provider check | Same `accept_*` functions; parse equivalence test |
| Large start response with many images | Images already normalized by `vision::normalize`; one bundle per Draft |
| Agent alters code-owned fields | Server overwrites them from the projected outline; structure checks unchanged |
| `continuous_shot` into `i2v` with a start frame | Rejected in the dry run with a scene-specific code |
| Relaxed private-store guard admits an unencrypted path | One `Root::allows` check; approved roots must be inside the proven encrypted mount; tests for outside paths |
| Media path resolution assumes `projects/<id>` | `Layout::project_at` used everywhere a layout-1 path is built; reopen-from-path test |
