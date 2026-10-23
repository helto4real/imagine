# Director UI groups: implementation and acceptance

Approved mapping: `ui-decision-groups-v1-20261021`, all groups 1–15.
Source: UI-GROUPED-DECISIONS-v1.md, UI-FLOW-DECISIONS.md and
UI-FIELD-LOGIC-MATRIX.md in the authorized controls-review evidence directory.

Latest checkpoint: CP26 completes the bounded G12 A/B vertical flow: Director
default/explicit pair rules (12 source-executed cases), swap, per-side review,
shared clock/loop and one audio source, retained selection state after an explicit
retry, Hide and save/reopen. The native modal and compact Take row pass real
normal/small acceptance;324 registered CPU tests and17 check stages pass on
`checks-57dd0b77`. G12 itself remains incomplete: Attach/Captures and provenance-safe
removal are still missing. Whole-UI final acceptance/publication/storage remain
later. The exact all15-group remainder is `UI-GROUPS-OPEN-26.md` in own evidence.

CP25 closes the known Group8 implementation gaps. Add Scene
uses Director's one-second gap insertion, typed mixed selection has one ordered
primary, modifier drags select without moving, and Section context image preview
uses the actual project resource. Media labels retain cross-type binding order
without invalidating generation for display-only changes. Native acceptance on
the immutable `checks-7fb46fce` package passes142-action and92-action flows at
1100x760 and600x760, including grouped motion/duplicate/delete/Undo, range/Fit,
import/reopen, waveform and Hide.321 registered CPU tests and all17 check stages
pass. Scene label clipping and waveform contrast/text readability were inspected
in actual native screenshots. Common whole-UI acceptance remains open; this is
not full-plan completion, publication, GPU or full Python-pipeline acceptance.
The exact all15-group remainder is `UI-GROUPS-OPEN-25.md` in own encrypted evidence.

## Storage direction superseding copied references (2026-10-21 18:40 UTC)

Reference images, reference videos and voices must remain at their original
locations and be linked. Thumbnails remain memory-only. This overrides the
previous independent-media-copy requirement for groups 1/5. The storage proposal is approved but its implementation specification follows
UI publication; do not implement a general storage change now,
move existing data or inspect user directories. The library-copy code and own
ui05 evidence produced before this direction are historical, unpublished and
not acceptance of the new linked-reference requirement. Continue storage-neutral
Section/prompt/model UI while that contract is pending.

## Final delivery order (approved 2026-10-21 18:53–19:02 UTC)

Complete and verify all approved UI changes and the newly approved F05 save
integrity correction, K1/F11/F12 historical runtime gates, and K2/F13 arbitrary
image steps/CFG ceilings. Recheck UI-BUG-001/002/003 with own fixtures after the
UI work; these are now approved, without expanding denied log/Exp operations.
Keep MP4/H.264/AAC; drag/drop and more Animate features remain deferred.

Then visually inspect the running isolated UI: every affected dialog/panel and
main flow at normal and smaller sizes, including long values and empty/filled
states. Check alignment, field sizes, labels, spacing, clipping, scrolling and
buttons. Use the shared icon-button component for recurring tool actions, with
consistent icons/size/spacing, tooltips, accessibility names and keyboard focus;
retain clear text for primary Generate/Save/Cancel choices. Record actual own
screenshots and measured sizes; build/tests do not replace visual acceptance.

Only after that publish the complete UI candidate using the reversible product
flow, without stopping/restarting user processes. Distinguish installed package
from running processes. No intermediate publication. Storage changes, including
reference copies to original links, follow ONLY AFTER UI publication in a separate
stage with the full storage specification. Do not move/delete existing data.

## Isolation

Baseline and candidate code copies: `comfyui/ui-groups-20261021/{baseline,source}`.
Only code, manifests, source fixtures and prompt templates were copied. Existing
working-tree changes are included in the baseline and are not overwritten.
Build with `python3 tools/ui-groups-cargo.py`; its target is separate from both
production and the earlier night build. Own test data and temporary files stay
in the encrypted night test root, under `ui-groups-01`. No publication or launch
selector changes until the entire UI candidate passes the final checks above. Production remains the scroll-fix package.

## Order and acceptance

Each group needs source/default comparison, focused behavioral regression and
real isolated UI acceptance. Runtime-sensitive groups also need actual worker
or GPU evidence; an enabled control or CPU test alone does not accept them.

| Order | Group | Implementation slice | Acceptance |
|---|---|---|---|
| 1 | 1 | Shared contextual project/library/folder media picker; import at cast, scene, Section, audio and Story | Pick existing and newly imported image/video/audio into the original target; search, cancel, reopen, wrong-type feedback and no accidental duplicate bindings |
| 2 | 2 | Live cast edits with Undo; image cards/default preview, semantic relationships and Section anchor roles | Close/reopen retains edits, Undo restores, independent generation/preview, exact Section scope and Director roles |
| 3 | 3 | Prompt tags, alias/catalog completion, snippets/catalog tools, inline Enhance, project Global prompt | Cursor-aware insertion/completion, source-equivalent expansion, Prefix/Suffix/off with retained text, proposal review before Apply |
| 4 | 6 | Project/scene model and scene type; canvas/quality/Reference Quality/shifts; free component paths | Standard new project, inherited/override settings, exact Director defaults and effective generation inputs; alternative names reach the loader |
| 5 | 7 | Exact seed/sampling/Stage2/LoRAs and unified Generate Run menu | Exact values reach worker, stage/LoRA resolver parity, direct Generate plus all four modes and stop/cancel behavior |
| 6 | 4 | Story phases/progress/provider profiles, old profile normalization, semantic image pairs, remove lifetime quotas | Complete own-fixture outline/expand/review/apply/retry, reversed asset-ID pair, >64 completed proposals without a lifetime block |
| 7 | 5 | Reference-card Save/Update; Projects/Characters/Assets library | Explicit library choice, thumbnail/search/sort/editor/delete, independent project copies, no external-original deletion |
| 8 | 8 | Section operations, ordered selection, keyboard/context commands, range/Fit/display, responsive panels and waveform | Ctrl toggle/Shift range, grouped edit/Undo, text-focus safety, narrow window and waveform alignment |
| 9 | 15 | Check Media/Relink and interval-bounded preparation | ID preserved, affected bindings shown, usable larger source decoded only as needed, no arbitrary admission limit |
| 10 | 12 | Default A/B pair, per-side decisions, manual Takes/Captures and cleanup | Accepted+latest/latest-two defaults, side switch/audio/transport intact, clear provenance and owned-file cleanup |
| 11 | 10 | Optional reviewed-plan Generate | Exact reviewed snapshot runs; stale review requires refresh only for this optional path; direct Generate unchanged |
| 12 | 13 | One-action Final, generate missing with fresh seeds, output filename/format | Internal freeze, exact Director readiness/accept rules, cancel/retry/resume and correct MP4 duration/audio; current MP4 preset retained |
| 13 | 14 | Session reference-cache RAM budget/clear | UI reaches real cache accounting/eviction; no persistent encoding disk cache or unrelated memory changes |
| 14 | 9 | LastFrame/Combo/Influence/Motion/Exp and transitions; Blend default 3 | Variant/image/audio dependencies reach conditioning; numeric component proof and actual generation; Exp diagnostics denial remains separate |
| 15 | 11 | Sampling preview and cadence | Worker preview events rendered during sampling, bounded memory, Hide/focus/reveal preserved |

## Explicitly preserved and open

Keep native Timeline, media shelf/import, direct Generate, clear Stop/Cancel,
Animate shortcut, draft retention/Retry, explicit library selection, independent
project copies, optional frozen review and existing privacy controls. No general
storage redesign during the UI stage. The approved storage direction is a separate
subsequent stage. The three separately reported bugs are now approved for own-fixture
diagnosis after UI changes; group 3 completion is already in scope.

Additional codecs are unspecified: keep MP4/H.264 CRF19/yuv420p/AAC and present
any additional choices before implementation. Earlier denied Exp diagnostics and
log reads are not authorized by these group decisions. Full Python/algorithmic
and memory parity remain separate open requirements; do not claim blanket parity.

Status after implementation slice 2 (UI-GROUPS-CHECKPOINT-02):

- Group 1: project/folder selection and contextual imports retain CP01 evidence.
  The Library picker now lists image/voice resources, searches/previews, reuses an
  already selected resource and reopens the same chosen library. Its independent
  copy proof predates the storage direction and is historical, not linked-reference
  acceptance. Existing storage transport stays in place until the later stage.
- Group 2: six Section image-anchor roles, custom influence/relationship/retention,
  named targets and live draft integration are implemented. Eight actual Director
  manifest-oracle cases pass; own native UI proves role/custom editing, rapid
  navigation, Remove/Undo/Redo and saving. Crop/guide strength, broader image-role
  generation and I2V/runtime acceptance remain open.
- Group 3: project Global prompt enable/Prefix/Suffix retains text while disabled;
  source compilation and own native UI pass. Shared @reference completion at the
  cursor covers Section, scene fields, Global and image-anchor text, with keyboard
  and mouse selection and Unicode. Reference-card Copy/Insert actions use shared
  accessible icon buttons. The real UI found and verified the correction preserving
  Section edits in a Cast commit. Catalog completion, snippets/catalog editor and
  per-field Enhance remain open; this is not whole-group acceptance.
- Group 5: the related library media selector works; standalone Assets, full
  search/sort/editor/delete and card Save/Update still remain.
- F05: own precommit and post-rename fault injection passes. Post-rename Story Apply
  retains the applied source and exposes NotSaved/storage_durability_uncertain in
  the returned snapshot; retry and a later edit retain its provenance. No storage
  layout change and no new product restriction were introduced.
- K1/F11/F12 and K2/F13, and separate UI-BUG-001/002/003 diagnosis after the UI
  changes are approved but not yet finished. Completion above is evidence only
  for its own synthetic fields, not closure of every UI-BUG-002 context.

No entire group is fully accepted. Existing GPU/Exp/Python/memory-parity gaps
remain separate and open. No new GPU work or reservation was used in this slice.

Known layout follow-up for the mandatory final visual pass: at a smaller native
window the top toolbar's two action clusters overlap. Inspect/fix the whole
responsive toolbar as part of group 8 and the final normal/smaller-window review.
Own anchor/privacy screenshots are evidence of that finding, not final visual
acceptance. In ui19 the completion popup and edited Section text disappear on
Focus(false), while the Scene description remains visible in that capture. Treat
this as an unresolved whole-inspector privacy observation to isolate with a focused
own-fixture test before claiming full privacy acceptance. Standardize remaining recurring tools using the shared icon component.

Checkpoint checks: 120 UI + 4 contract + 16 API + 3 persistence tests pass (143).
Server-library and normal/QA UI Clippy pass, as do normal/QA builds and fmt.
Immutable package: `comfyui/ui-groups-20261021/packages/checks-971299bc`,
activated=false. That exact package passes own Section ui16 (49 actions), reference
prompt ui17 (71) and ui19 (77, with tooltip/Hide capture), Global prompt ui18 (44).
All own process pairs are reaped. No whole-UI visual, GPU or Python acceptance is
implied. Technical build failures and earlier UI failures remain in own evidence.

Next: isolate the noted inspector privacy observation, then catalog/snippet and
per-field Enhance flow for group 3, followed by groups 6–7
model/sampling. Continue the remaining matrix, approved technical corrections and
own-fixture bug diagnosis; final visual acceptance precedes complete UI publication.
The linked-reference/storage stage follows that publication, never an interim UI.


## Checkpoint 03: verified additions after checkpoint 02

- UI3: per-field Enhance now carries an optional typed SceneField through the
  existing proposal service. Summary/Description/Ambient sound/Audience-only music
  read and apply their own source, leaving all other project fields intact; old
  requests default to Description. Section uses its existing scoped route. Opening
  a field freezes its target and retains review/Apply; changing target archives the
  previous completed proposal. Target changes do not silently retarget a running
  proposal. Own loopback API proof covers all four scene fields; native proof covers
  the four fields plus Section, editing proposals, Apply and saved provenance.
- UI8: toolbar tool actions use the common accessible 32×32 icon component and
  tooltips; primary flow choices retain text. Below 1000 logical pixels the tool
  group and flow group occupy two rows. Actual egui bounds tests cover 600, 760,
  1000 and 1280 logical pixels, including all 12 controls and pairwise overlap.
- Hide: the prior inspector observation did not recur in the unchanged CP02 package
  (ui21) or with QA focus/render flags (ui20). Both fields are concealed in the new
  screenshots; there is no proven cause for the earlier capture. A new two-field
  regression checks opacity, accessibility and stable post-initial-layout geometry.
  Separately, Cast/Story previously omitted their entire body when concealed. That
  code path changed scroll extent by construction. They now render an invisible
  body with the same geometry and redact the actual unique accessibility subtree;
  CPU geometry/accessibility regression passes. No blanket whole-UI acceptance.
- Native test scrolling now uses raw layout rectangles to bring offscreen controls
  into view with real wheel input. Clicks still require visible fresh controls.
  This fixes the own ui23/ui24 test timeouts, not a product admission rule. A static
  existing test prompt was copied unchanged into the code-only candidate so the
  server's all-target type check can compile its existing lifecycle test.

Catalog/snippets/Phrase Composer/editor/reload and wider completion contexts still
remain for UI3; they have not been replaced by a stub or accepted as complete.
UI6/UI7 model/sampling work has not started in this checkpoint. All other pending
criteria, approved K1/K2 corrections, bug investigation and final visual acceptance
remain. No publication, storage migration, GPU work or denied log/Exp retry occurred.


Checkpoint 03 final verification: 123 UI + 4 contract + 17 API + 3 persistence
regressions pass (147). Server-library and normal/QA UI all-target Clippy pass;
normal/QA builds and fmt pass. Immutable package `checks-338b2180`, activated=false.
That exact package passes native ui26 Enhance (121 actions), ui27 completion (77),
ui28 Section/Undo (49), ui29 Cast/media (48). UI26's four scene fields plus Section
retain exactly five Applied provenance records after save. F05 remains green.

Visually inspected native UI26: all field values after Apply, concealed/revealed
proposal body at unchanged window extent, and toolbar/project-prompt layout at
1522×1680 logical pixels (1.25 ppp) and 761×840 (2.5 ppp). The latter deliberately
uses 2× own-UI zoom on the same physical window; it proves narrower logical layout,
not an independently resized physical window. The final all-dialog normal/smaller
physical-window pass and long-value/empty-state coverage are still outstanding.

The toolbar overlap from CP02 is fixed in this slice. The original isolated
inspector capture is retained as a non-reproduced observation; no unsupported
cause or full privacy acceptance is claimed. Cast/Story's proven early-return
layout issue has a separate tested correction. No access blocker was encountered.

Next concrete implementation: UI3 catalog completion, linked placeholders,
snippets/Phrase Composer, editor/reload using permitted source templates and own
fixtures; do not inspect Director's private config. Then UI6/UI7 model/sampling,
remaining matrix and approved technical/bug corrections. Keep final whole-UI
visual acceptance before complete publication, and storage after publication.

## Checkpoint 04 — catalog vertical flow

UI3 now includes namespace/group/alias completion, linked snippet placeholders,
Tab navigation, a template browser with editable values and prompt insertion,
and YAML/Form catalog editing with explicit folder selection, validation,
atomic Save/Save as, reload and last-good completion. The chosen folder persists
as an optional UI preference. Unsaved per-file drafts remain in RAM across file
switches. Form writes canonical 1.1, including ordinary-text Phrase Composer
authoring: @identity creates/reuses named placeholders and @@ escapes literal @.
Six public Director documentation templates are copied unchanged; private Director
config was not read. No runtime logs or user data were read.

The native flow exposed and fixed a real Form bug: accessing absent lists through
mutable JSON indexing inserted null, making the next catalog save invalid.
The invalid write was rejected, preserving disk contents; a roundtrip regression
now proves absent lists remain absent and 1.0 inline placeholders normalize safely.
The earlier ui30 scroll timeout was a test geometry error, not a product issue.
Vertical scrolling now tests vertical visibility rather than a horizontal margin
that scrolling cannot change. Hide remains a non-reproduced observation from CP03.

Verification: 126 UI + 8 contract + 17 API + 3 persistence tests pass (154),
including F05. Server/normal UI/QA all-target Clippy and builds pass. Final immutable
package: checks-8f7d1bc9, not activated. Eight Python catalog source-oracle cases,
five actual JavaScript completion/expansion queries and five actual JavaScript
Phrase Composer cases pass. Final native ui38 passes 93 actions, including restored
catalog preference, linked/Unicode snippet edits, composer insertion, YAML/Form
save/reload/reopen, ordinary-text composer authoring and escaped @. Native ui33–37
passed the preceding checks-f290cb12 package: catalog 90 actions, restored catalog
88, references 77, Enhance 121 and Section/Undo 49; these are not mislabelled as
tests of the final package's later Composer addition. All own processes are reaped;
no GPU or reservation was used. Original baseline source and production selector
are unchanged. Checkpoint source/support snapshots and cumulative/incremental
patches are saved on the encrypted test root.

UI3 is still partial as a whole. Before accepting it, finish/verify nested and
backward snippet navigation and all prompt contexts, the Composer identity picker
and catalog duplicate operations, stale/reload/folder cancellation behavior through
native UI, and final visual/accessibility acceptance. Native natural-name ordering
is not proven equivalent to JavaScript Intl.Collator for accented labels; YAML
anchor semantics also remain an explicit parser difference. Native explicit folder
selection replaces Director's server-owned config location. Recheck whether the
catalog read's O_NOFOLLOW restriction is necessary; it is not accepted as an extra
user-choice restriction. Source format checks, conflict detection and atomic writes
serve catalog integrity; Director's arbitrary catalog quotas were not copied.

UI1/UI2/UI5/UI8 remain partial at CP03 acceptance; no whole group is newly accepted.
UI6/UI7 model/sampling implementation has not begun. Next: close the listed focused
UI3 gaps, then implement coherent UI6/7 flows and continue the entire approved
matrix, K1/K2 and own-fixture bug checks. Complete the final actual-UI visual pass
before whole-candidate publication; perform the separate storage stage afterward.
Earlier Exp/log denials and algorithmic/memory parity gaps remain unchanged.

## Checkpoint 05 — bounded catalog completion and first UI6/7 vertical flow

Notation: C = implemented with focused behavioral evidence; V = implemented but
focused acceptance remains; M = function/semantic work remains; R = shared final
visual/regression acceptance, independent of functional implementation status.
No entire group is final-accepted merely because its core flow passes.

| Group / criterion | Status and exact remaining acceptance |
|---|---|
| 3 / tags, Global on/off/Prefix/Suffix, per-field Enhance | C from CP02/03; R: repeat the combined final UI run with long/empty fields and keyboard/privacy states. |
| 3 / hierarchical #namespace.group.alias, linked/nested snippets, Tab/Shift-Tab, blur/click handling | C: native ui39/ui43 nested and backward navigation, ui42 references. V: representative Global/scene/anchor contexts together in the final matrix. |
| 3 / ordinary-text Composer, identity suggestions, @@, duplicate file/phrase, Delete/Cancel, YAML/Form save/reload | C: 129 native actions in ui43. Source oracle matches parser, expansion, Composer and duplicate identifiers/aliases/labels. |
| 3 / failures and platform semantics | V: native invalid reload/last-good and OS-folder cancellation. Existing file-save conflict/invalid-write tests pass. Native natural-name collation is not proven Intl.Collator-equivalent for accented labels; YAML anchors are accepted more broadly. These are explicit semantics, not blanket algorithmic parity. R: common final layout/accessibility pass. |
| 6 / project and scene mode, per-model project defaults and scene override/inherit | C: shared typed contract/effective-settings resolver, fingerprint and Undo tests; native project Apply + scene settings inheritance. V: native Reference/I2V switching, valid per-model caches and actual GPU requests in both modes. M: retain malformed raw numeric drafts in RAM when switching project mode; the current valid-value-only transfer does not retain that raw input. |
| 6 / Standard new project, canvas/quality/orientation/FPS/custom dimensions | C: native new project Standard 832x480, seed 0; legacy projects retain dimensions/seed 42; Draft 4:3 portrait reaches CPU request and MP4 at 384x512. V: Source/Native Resolution sizing and non-24-fps semantics against Director; H3 currently rejects non-24-fps generation. R: independent smaller physical window. |
| 6 / Reference Quality | C: selector and stored effective settings. V: assert 1/2/4-MP preparation using larger own synthetic references, then GPU path. |
| 6 / free component files, sigma shifts, scene kinds | M: component selectors plus request/worker transport; video/audio shift transport; Generated/Imported/Extended/Edited/Placeholder intent UI and actual semantics. No placeholder controls silently ignore these inputs. |
| 7 / exact seed, base sampling and manual sigmas | C: seed 9876543210123 and exact FP32 schedule reach the frozen request; parser/draft retention prevents per-keystroke HTTP writes. M: full u64 runtime acceptance (existing 2^53-1 cap remains); V: random/direct/Run seed semantics and mode caches. |
| 7 / Stage 2 and handoff | C: UI + DTO + existing freeze path for two stages; CPU request has 20/euler/simple base, manual [1,.7,.4], Stage2 4/euler/simple with [.4,.2,0], Continue latent. V: actual GPU + Python comparison. M: trained upscaling/factor/model and Director-equivalent validation of ignored controls with manual SIGMAS. |
| 7 / LoRA stacks and unified Generate modes | M: project/scenes/targets Inherit/Add/Replace/Disable and Stage2 rules through runtime; unified four-mode Generate menu. Existing separate Runs and direct Generate remain. V: all four modes' seeds/dependencies/cancel after integration. R: final visual/accessibility pass. |

New `generation` metadata is optional and omitted for legacy documents. Effective
settings prefer the exact scene/model slot, then the project/model slot. Inactive
model slots do not change the active input fingerprint. Inherit removes only the
chosen slot; Undo now restores slot removal as well as additions. New scenes use
project defaults; existing explicit scene settings are preserved. The normal
preparation path uses this same resolver, not a UI-only shadow setting.

UI41 is a CPU synthetic backend, not H3 numerical acceptance: 74 native actions
pass through the normal API, queue, request freeze, Take and MP4 publication. The
own output has 73 frames, 24 fps, H.264/yuv420p/BT709 + AAC/32kHz/stereo, video/container
duration 3.041667 s, audio 3.041656 s. Decoded PCM: 97333 samples/channel, RMS
0.0141114212, peak 0.0216820836. Input metadata records no prompts or raw media.

Final package checks-98fbb6d2 is unactivated. 128 UI + 10 contract + 17 API + 3 F05
persistence tests pass (158), with Clippy and normal/QA builds. Native ui41–44 used
the preceding checks-aa1c4f62 package: generation 74, reference 77, catalog 129,
Section/Undo 49 actions. The final package only further narrows numeric-format
feedback to the generated scene and stops blocking unrelated prompt review, plus
clarifies unfinished-feature copy; its native smoke is the next quick check.
No numerical engine or model code changed. The check process remained accessible
through the reported connection notifications and completed; no duplicate run was
started. Every own fixture process was reaped, no GPU/reservation was used.

Before publication, verify rollback compatibility of saved projects containing
the new additive `generation` field: new code reads legacy documents, but older
deny-unknown-fields readers cannot read the new field. No existing user data was
read or migrated, and no storage-layout stage was started. Do not call an old
binary switch sufficient data rollback without resolving this compatibility point.

Next concrete work: quick native smoke of final package, then UI6 component paths
and shifts through an isolated worker source/build and actual loader; complete
scene-kind/Source/FPS semantics and UI7 LoRA/upscaler/Run requirements. Keep each
functional criterion separate from common final visual/regression acceptance.
UI1/2/5/8 acceptance is unchanged from CP04, and all other approved groups, K1/K2,
own-fixture bugs, Exp permission, P08 and memory/algorithmic parity remain open.
No publication, user-data access, storage migration, GPU work or denied-operation
retry occurred in this checkpoint.

## Checkpoint 06 — compatibility, draft retention and catalog failure acceptance

The exact CP05 final package passes its native smoke (ui45). The new final
package is `checks-d6ab15b1`, unactivated: 130 UI + 10 contract + 17 API +
3 F05 persistence + 1 compatibility + 1 H3 sampling adapter tests pass (162).
Server and normal/QA UI Clippy, builds and fmt pass.

| Criterion | Functional evidence and remaining acceptance |
|---|---|
| UI3 invalid reload/last-good/recovery | C: ui51 (141 real native actions) changes only its newly created YAML fixture, reloads invalid YAML, observes the specific error, completes a real snippet from the last good catalog, restores/reloads valid YAML, then saves/reopens edits. Focused tests cover stale async replies, cancelled selection retaining state and invalid/conflicting writes. V: actual OS folder-dialog Cancel; this is not replaced by the controller cancellation test. R: final all-context visual/accessibility pass. |
| UI6 per-model raw drafts | C: malformed SIGMAS survives Reference→I2V→Reference; Apply shows a format error and leaves the dialog open. Component proof also retains an independent alternative sampler name. ui50 covers the real switch and subsequent successful edit/generation. V: actual H3 requests/GPU in both modes, all valid override combinations. |
| UI7 manual SIGMAS | C: supplied curves override the unused steps/scheduler/denoise controls; adapter regression proves equivalent wire input even with steps=0/unknown scheduler/denoise=0. Canonicalization touches only ignored fields, leaving sampler/FP32 curve unchanged. No worker numerical code changed. V: real GPU/Python acceptance, full seed/Run semantics, trained upscaling and LoRAs remain. |
| Rollback data readability | C: explicit Project→Export for older Imagine creates a separately identified copy, retaining the active full original. Effective inherited settings are materialized. Media labels/library links/image annotations and newer Enhance draft records are omitted only from that copy, with a prior explanation. The unchanged baseline's actual ProjectSession reader opens the copy, including Takes/resources; normal saves retain all metadata. ui50 and ui52 cover the UI/API/copy/old-reader path. This is a deliberately limited export, not transparent old-reader support for the new format or numerical parity with an old engine. |

UI50 passes 119 actions (14.55 s), ui51 141 (12.48 s), ui52 130 (21.59 s),
all on the exact final package. UI52 retains all five applied Enhance records in
the original and two old-format-compatible records in the copy; all reviewed
creative text and media remain. The copy is not made the active project. Failed
copy targets preserve the active document, save status and original manifest
bytes in the focused persistence test. No existing user project was inspected.

The first new native attempt ui46 stopped because the test reopened a dialog at
its previous scroll position, then clicked an offscreen Quality control. ui47
passed after explicit scrolling. ui49 completed its GUI actions but failed its
postcondition: Stage 2 SIGMAS was empty. Waiting for native smooth scrolling to
settle before the test clicks fixes that input timing; ui50's postconditions prove
both exact curves in the frozen request. These failed attempts remain evidence,
not accepted runs. One API test also raced its own service's autosave lock during
reopening; it now shuts down its own writer before asserting disk reopen, matching
the other persistence tests. No product lock or validity requirement was loosened.

Visually inspected own ui47 invalid-SIGMAS/compatibility explanation, ui49's
missing-field failure, ui51 specific reload error and ui52 export confirmation at
1522.4×1680 logical pixels (1903×2100 capture, 1.25 ppp). Readable text and controls
fit these dialogs at that size. This does NOT close the required physically
smaller-window/all-dialog final visual acceptance.

Exact UI6 remaining implementation: free component paths and actual worker
transport, configurable video/audio shifts, scene-kind intent and semantics,
Source/Native sizing and non-24-fps behavior. Reference Quality still needs own
large-reference 1/2/4-MP preparation and GPU evidence. UI7 remaining: full u64
runtime seeds/control-after-generate and Run seeds; per-target project/scene
Inherit/Add/Replace/Disable LoRA resolver and Stage 2 rules; trained upscaler/model/
factor; unified four-mode Generate menu with stop/cancel/dependencies; GPU/Python
Stage 2 acceptance. Preserve invalid raw input rather than substituting defaults.

UI1/2/5/8 retain their earlier scoped evidence and open matrix items; all other
groups remain open without a new acceptance claim. K1/K2, own-fixture bugs001/003,
final visual/regression acceptance and whole-candidate publication still remain.
Legacy continuity mode buttons still need the shared defaults/inheritance resolver.
P08, memory/algorithmic parity and the old Exp/log denials remain separate and
unchanged. There is no new permission blocker. All own workers are reaped; no GPU
lease was taken. Production selector/source and existing data remain unchanged.

Next: integrate component paths/shifts through an isolated worker source/build,
then scene-kind/Source/FPS semantics and the remaining UI7 vertical requirements.
Do not add UI-only controls that silently ignore those inputs. Continue the full
approved matrix; publish only after the required final actual visual acceptance,
and start the separately specified storage stage only after that publication.

## Checkpoint 07 — component files and AV shifts reach the actual worker

UI6 now carries free diffusion/text-encoder/video-VAE/audio-VAE paths through
project/scene settings, request freeze and protocol 9 into the real loader.
Paths may be absolute or relative to the configured model root; neither filenames
nor extensions are whitelisted. Blank retains the deployed legacy default.
New project Reference/I2V model names use Director's actual defaults. The compact
folder buttons use the common icon/tooltip component and asynchronous selection.
Replies are scoped to the original project/editor session and mode; Cancel keeps
the current value. Actual OS picker interaction still needs platform acceptance.

The worker now uses selected video/audio shifts for schedule construction,
component-owned ModelSampling and audio time conversion. Defaults remain 12/3;
the UI's .01–100 bounds come directly from Director's input widgets, while the
transport requires only finite positive values. Stage 2 has independent optional
Reference and I2V file inputs, falling back to the original selected mode's model
before Director LoRAs. Its fallback affects the fingerprint only when used; a
change to an inactive mode's unrelated sampler does not invalidate the scene.
The frozen Take records the resolved model input.

The frozen model-lock digest check and equality against a fixed component-size
inventory were removed from this loader path. The associated RAM precheck also
used that same stale inventory. Actual tensor/header loading and the existing
memory manager remain. This is required for compatible alternative files, not
integration of the separate memory optimization. Broader K1 runtime/declaration
gates and memory-lifetime/algorithmic parity gaps remain open.

The H3 source is isolated at `comfyui/ui-groups-20261021/source/inference`, based
on frozen cd0ef53be69dd539ef0ffdf0cb03f4c1c5835fc7 plus the approved pre-task H3
overlays. Its immutable baseline is `worker-baseline`; targets are `target-worker`
and `target-wire`. No existing native target or production source was overwritten.
Only H3 loader/sampling files and the CPU transport fixtures differ from that
frozen worker baseline, plus one new synthetic CPU shift example.

Closed functional criteria in this slice:

- UI6 component/shift entry and exact transport: ui53 proves all four paths,
  including Unicode, spaces and alternative extensions, plus 9.5/2.25 and an
  explicit Stage 2 file, reach the frozen request. ui54/ui55 repeat the flow with
  the folder buttons/session scoping; all have 167/169 native actions.
- UI6 actual Reference/I2V loading with chosen files: three sequential normal
  scheduler/Ticket GPU jobs run through own UI Generate and the standard worker.
  The inputs are five explicitly known model files, selected through renamed own
  symlinks. This tests free paths/names, not different trained weights or a new
  model architecture. No observer or latent dump was used.
- AV shift math: four CPU cases (12/3, 9.5/2.25, 1/1, .5/2) match unchanged
  ComfyUI ModelSamplingAV/simple scheduler/time_shift_sigma functions byte for
  byte for both schedules, MAE/RMSE/max=0, and exact audio_scale. The same Rust
  configuration factory is used by the worker. Full GPU/Python image/latent
  parity is not inferred from this component proof.

Actual GPU results (416×224, 39 frames, 24 fps, 20 res_multistep/simple, seed42):

| Variant | Output duration | Decoded samples/channel | AAC-decoded RMS / peak | Observed job seconds after submission UI closed |
|---|---:|---:|---:|---:|
| Reference 12/3 | 1.625000 s | 52000 | .0082737132 / .0667522922 | 83.90 |
| Reference 9.5/2.25 | 1.625000 s | 52000 | .0081689585 / .0615814254 | 77.73 |
| I2V 9.5/2.25 | 1.625000 s | 52000 | .0039957108 / .0324112326 | 80.64 |

All are MP4/H.264/yuv420p/BT709 limited-range + AAC/32kHz/stereo. The two Reference
jobs use identical inputs apart from shifts. Their decoded lossy-output RGB8 MAE
is 7.66940695, RMSE 15.19733599, max202; AAC-decoded PCM MAE .002534701,
RMSE .003731515, max .038857374. These are post-export observations, not raw latent/
decoder parity. Three own middle frames were visually inspected and show normal
images without obvious green corruption at that sampled time; not all frames
were visually accepted. Timing is functional-test bookkeeping, not a benchmark.

GPU evidence uses immutable server/QA package `checks-19d4a833` and worker
`worker-dcb7ee2b`. Final UI package is `checks-5cf94884`; only the subsequent
component picker/editor-session handling differs in the UI, and server/worker
transport remains the tested one. Final checks: 131 UI +10 contract +17 API +3 F05
+1 compatibility +1 adapter +11 protocol/sampling/options tests (174), plus 15
CPU worker transport cases. Server/UI/worker Clippy, normal/QA builds and fmt pass.

The final package also passes native ui57 (167 actions, 19.02 s). Its server
binary is byte-identical to the one used for all three GPU cases. Source/target
directories were verified as separate real paths, not redirects to production.
Native ui58 passes 174 actions (22.35 s), with additional upper component rows and
the actual Choose Diffusion model tooltip visually inspected. Folder icons/fields
are aligned and readable in these captures; smaller physical sizing remains open.

Visual limitation: ui55 requested 1100×760 but the compositor kept the same
1522.4×1680 logical /1903×2100 physical surface. It is NOT a smaller-window pass.
The only attempted window-state operation was `hyprctl dispatch setfloating
pid:2570307` targeting our own ui56 child; it returned exit7 and the child/server
were reaped. A read-only `hyprctl version` check returned exit2, `Couldn't set
socket timeout (2)`. No other windows/config were queried or changed and no
alternate IPC/access path was attempted. This is an unavailable normal test
interface, not an automatic-review rejection. Independent UI work can continue;
the final physically smaller-window acceptance remains open.

Remaining UI6: Generated/Imported/Extended/Edited/Placeholder intent and semantics,
Source/Native sizing and non-24-fps behavior, large-reference 1/2/4-MP preparation,
and actual file-dialog choose/cancel/late-reply interaction. Distinct compatible
trained files were not tested. Remaining UI7: direct full-u64 seed/control-after-
generate and legacy normal-job restrictions, unified four Run modes and stop/
cancel/dependencies, LoRA targets/merge modes/Stage2 rules, trained upscaling,
complete runtime sampler/scheduler coverage against Director's registry (the
current worker only dispatches Euler/res_multistep), and actual Stage2 GPU/Python
parity. Preserve inactive Stage2 settings across
disable/re-enable/apply/reopen: current Option-based UI still resets some fields.
Director Run fresh seeds deliberately use MAX_SAFE_SEED; do not mistake that
source behavior for the separate editable-u64 executor seed requirement.

All other group criteria, approved bugs, K1/K2, P08, Exp/log denials and full
algorithmic/memory parity remain open as previously stated. No whole group is
newly final-accepted, no production selector changed and no storage stage began.
The three GPU workers exited before successful job publication; own server/UI
processes and normal Ticket connections are closed. No GPU work remains running.
No new permission denial occurred. Before eventual publication, verify protocol9
in the final integrated launch manifest; original/root protocol8 remains unchanged
until the complete UI source/publication transition.

Next: scene-kind/Source/FPS and remaining Reference Quality semantics, then the
remaining coherent UI7 flows, other groups, bugs and whole-UI visual acceptance.
Continue implementation independently of the unavailable smaller-window test
interface; keep that actual final acceptance requirement visible.

## Checkpoint 08: seed, inactive Stage 2 and Run menu

This is a continuation checkpoint, not whole-UI acceptance or publication.
The immutable package is `checks-110d8856`; worker `worker-70d2b4ba` remains
protocol9. Original/root protocol8 and production selectors are unchanged.

Closed functional criteria in UI7's Timeline path:

- Explicit seeds retain the full u64 range through UI, DTO, frozen request,
  worker and Take provenance. A real Reference GPU job with u64::MAX finishes
  20 steps and produces normal MP4/H.264/AAC, 39 frames, 1.625s and 52000 decoded
  samples/channel. This is functional acceptance, not new full Python parity.
- Fixed/Increment/Decrement/Randomize operate on the next widget seed after
  confirmed queue admission. New projects start at seed0/Randomize; legacy
  metadata without the new option retains its old behavior. Native CPU tests
  prove 42/43 queued ->44 stored, 0/0 ->0, full-u64 Fixed retained, and Randomize
  queues the explicitly entered full-u64 value before changing the next seed.
  ComfyUI's widget automatic range is 2^50; manual input remains u64. Director's
  separate Run fresh-seed allocator uses MAX_SAFE_SEED and is unchanged.
- Inactive Stage 2 keeps every typed field and unfinished raw SIGMAS through
  disable/Apply/reopen/enable. Its preset is excluded from the generation
  fingerprint and frozen worker settings. An inactive invalid draft does not
  block single-stage generation. Older-compatible copies omit this new preset
  only in the separate copy; the original remains complete.
- The shared accessible 32px down-arrow next to Generate opens all four Run
  modes. Actual own UI/CPU flows complete 3/3/3/1 targets for Missing/Same/Fresh/
  Selected-with-dependents. Same seeds match the previous accepted run; Fresh
  seeds differ. The last test has no dependency chain, so dependent ordering,
  stop/cancel/retry and admission-network-fault acceptance remain open.
- Six synthetic actual Rust AV prepare_noise cases (seed0, 2^53+1, u64::MAX;
  Float/Half) match unchanged ComfyUI functions byte for byte in both streams:
  MAE/RMSE/max=0. No models or CUDA runtime are initialized for that comparison.

UI3's invalid raw SIGMAS survives Reference/I2V switching in actual native UI.
Optional older-format export still opens in the unchanged frozen reader for
representable Takes. For an exact seed the old reader cannot represent, only
that optional copy fails, with a specific UI explanation and intact original;
no automatic seed conversion or normal-save restriction was introduced.

The smaller-window access issue is resolved for ordinary own application size
hints: the opt-in QA build can request fixed min/max/inner size without any
compositor command or system change. ui63/ui65/ui67/ui68 actually run at
1100x760 logical, 1375x950 physical, compared with the normal 1522.4x1680 logical
surface. This accepts only the tested generation/settings/Run flows, not all
dialogs or the final whole-UI visual checklist. No hyprctl retry was needed.

The early Stage 2 click failures came from the QA runner proceeding while
wheel animation was still moving. It now observes a stable target rectangle
before scrolling/clicking again; this is test instrumentation, not a product
scroll restriction. An early fourth-Run timeout (ui64) was not reproduced in
the complete small ui65 and normal ui66 flows; its cause remains unproven and
no speculative product change was made. All evidence is retained. A final
attempt to view ui68's own tooltip/menu/component screenshots was aborted by
the user before output, and was not retried or replaced by another image path.
Those final images were generated, but their visual inspection remains open.

Final checks: 133 UI +11 contract +17 API +3 F05 +1 compatibility +1 adapter
+13 protocol/sampling/options =179 tests, plus15 CPU worker transport cases;
server/UI/worker Clippy, fmt and normal/QA/worker builds pass. Final native ui68
passes197 actions in29.17s with long Unicode component paths, actual smaller
window, full-u64 Fixed, Stage 2 retention, save/reopen and specific old-format
failure. GPU timing78.60s is observation after the submitting UI closes, not a
benchmark or complete process duration. AAC-decoded RMS .0238808161,
peak .2252658457. Worker's PID was absent before success publication, own
server/UI workers exited and their normal shared Ticket reservation closed.

Exact remaining UI6 criteria: scene kinds Generated/Imported/Extended/Edited/
Placeholder; Source/Native sizing and FPS behavior; 1/2/4-MP large-reference
preparation; real OS file-dialog choose/cancel/late-reply interaction. Component
paths and shifts retain CP07 CPU/GPU evidence, but distinct compatible trained
weights/architectures were not tested.

Exact remaining UI7 criteria: video-LoRA targets and Inherit/Add/Replace/Disable
resolver, clean Stage 2 model behavior, trained upscaler/factor, complete runtime
sampler/scheduler coverage beyond Euler/res_multistep, actual two-stage GPU/
Python handoff parity, legacy general-video job's fixed42/20 restrictions, and
Run dependency-chain/stop/cancel/retry/network-fault UI acceptance. Pending
admission only advances once in the component logic; full HTTP fault injection
for that new seed control is not claimed.

Other groups1-15, approved K1/K2 and UI bugs, P08, denied Exp/log operations,
whole Python/algorithmic/memory parity and final whole-UI visual acceptance
remain open as separately listed above. No whole group is newly final-accepted.
No storage change, product activation, user service restart, commit or push.
All own tests are stopped; no GPU work is left running. No automatic-review
rejection occurred in this slice. Normal sandbox loopback access required the
ordinary narrow escalation for our own API; no alternate access path was used.

Next: complete the remaining UI6 scene-kind/Source/FPS/reference-quality path,
then UI7 LoRA/runtime sampling/Stage2 and the other approved groups and bugs;
finish the full normal/smaller visual checklist before publication, and only
then begin the separately specified storage stage.

## Checkpoint 09: scene types and independent project/H3 frame rates

This checkpoint continues the complete UI task. Candidate `checks-1409f1ff`
is not published; standard worker remains `worker-70d2b4ba`/protocol9. Original
sources/production selectors and storage layout remain untouched.

Closed functional UI6 criteria in this slice:

- The five actual Director types Generated/Imported/Extended/Edited/Placeholder
  are editable in the selected scene's inspector, including from its Section.
  `changeShotType` only changes type; its clip stays stored. A source-only JS
  oracle proves that for all five Director types. The shared Rust contract and
  native UI prove the same retention, duplicate/remove behavior and JSON roundtrip.
- Selecting an Imported clip uses the existing contextual media picker, source
  probe and edit path. Import, Undo/Redo, switching all five types and then
  generating successfully with a retained inactive clip are tested with our own
  synthetic MP4. Other types can generate; Imported uses its clip and is skipped
  by generation Runs, as in Director. A missing imported clip does not silently
  select a generated Take during Final export.
- Legacy documents with a clip and no explicit type still mean Imported. Normal
  save keeps the new type and inactive clip. An explicitly exported older copy
  omits type labels/inactive clips, preserving the complete original and its
  Takes/media. The unchanged frozen reader opens that copy in ui70.
- Positive project FPS is independent of fixed H3 output24fps, following
  Director's informational conversion behavior. The former project-fps24
  admission rejection is removed; no new frame-rate clamp or fallback is added.
  UI explains the distinction. Native CPU tests use project30 and29.97 (2997/100)
  while the actual frozen H3 request remains24. All four Run modes regress at
  29.97 with full-u64 Fixed and a retained inactive source clip.
- Native Resolution at4:3 portrait produces864x1152 in the actual CPU request.
  Extra High and Standard reach the frozen Reference quality field; prior High
  evidence remains. Existing source-derived geometry fixtures now run in the
  common checks, covering all presets/ratios/orientations/durations. Eighteen
  CPU worker reference-geometry cases exactly match the pure Director function
  with explicit approved1/2/4MP budgets. This does NOT prove resized/padded pixel
  equality or large-reference GPU encoding.

Actual Reference GPU acceptance `fpsgpu01`: project30fps, output39frames24fps,
duration1.625000s, AAC32kHz stereo52000samples/channel. AAC-decoded RMS
.008273713159601341, peak .06675229221582413. Its decoded RGB8 (10,902,528 values)
and decoded float PCM (104,000 values) are byte-identical to our previous Rust
project24fps control: MAE/RMSE/max=0. This is a Rust-to-Rust post-export regression,
not Python/latent parity. Observation after the submission UI closes is68.12s;
not a performance benchmark or full process duration. The standard worker's PID
is absent before success, own UI/server exited and their shared Ticket closed.

The Imported UI test found a real error-reporting gap: the scheduler changed
the existing `imported_scene_not_generatable` error to generic
`invalid_scene_request`. The specific code now reaches the existing UI message;
ui70/ui71 prove the actionable error. This is not a new admission restriction.

Native ui70 passes184 actions at1100x760 logical /1375x950 physical, including
Native/ExtraHigh, type/clip workflow, save/reopen and the older-reader copy.
Native ui71 passes197 actions at normal1522.4x1680 logical /1903x2100 physical,
including29.97, Standard, full-u64 and four completed Run modes (3/3/3/1 targets).
All original source-baseline comparisons must remain clean at the snapshot.

The interrupted CP08 image inspection is now complete for our ui68 tooltip,
settled menu and component-file rows: labels/tooltips are readable and the shared
32px icon is aligned. ui70 scene-type and generation-window images are also
visually inspected at the smaller size. Whole-UI visual acceptance is still open;
these images do not cover every dialog, privacy state or responsive panel.

Final checks:133 UI +12 contract +17 API +3 F05 +1 compatibility +1 adapter
+17 geometry/protocol/sampling/options =184 tests. Server/UI Clippy, fmt and
normal/QA builds pass. Worker runtime source is unchanged; only a synthetic
dimensions example was added/formatted. Prior15 CPU wire and seed-noise proofs
remain applicable. No additional sampler, LoRA, upscaler or memory optimization
is claimed in this checkpoint.

Two normal automatic-review rejections are recorded, not bypassed:

1. `python tools/ui-groups-cargo.py fmt --all; python tools/check-ui-groups.py`
   was rejected because the new geometry verifier read Director's `config.py`.
   That read was removed. It now supplies already approved1/2/4MP budgets to
   the pure function without configuration imports. The safer checks passed.
2. `python tools/build-ui-groups-worker.py reference_quality` was rejected on the
   assumption that this helper still read `SOURCE/config.py`. A read-only code
   check proved its SOURCE is the isolated candidate and the branch only builds
   and executes the pure Rust geometry example. One normal retry with this new
   evidence was approved and passed. No configuration read was retried after
   rejection, no alternate execution/access route or system change was used.

No current permission blocker remains for this slice. Older Exp/log denials
remain untouched. All own test processes are closed and GPU reservation released.

Remaining UI6: actual OS component-file-dialog choose/cancel/late-reply; Reference
pixel-resize/padding parity and actual large-reference GPU paths; final mixed
source/project-FPS montage acceptance. Source ratio is an existing Imagine
Animate/I2V extension, not a missing Director project-ratio option; preserve it
without inventing a new Director control. Distinct compatible trained component
weights/architectures still have not been tested.

Next coherent UI7 work: project/scene LoRA main-target resolver and actual worker
patching (Inherit/Add/Replace/Disable, Stage2 clean-model rules), trained latent
upscaler/factor and runtime sampler bindings, followed by actual Stage2/Run
dependency/stop/cancel/retry acceptance. The required existing upscaler model
`models/latent_upscale_models/h3_upscaler_lms_v0.1_fp32.safetensors` is present;
the installed 3D provider source is identified. There is no missing-weight excuse
or authorization to install/download. Use those existing model/ops owners and
component proof before a complete handoff GPU test; do not add ignored UI fields
or claim support through a fallback.

All other approved group/bug/K1/K2/P08 requirements and whole Python/algorithmic/
memory parity remain open. No whole group is final-accepted. Continue through
the fixed matrix, then full normal/smaller visual acceptance, finished UI
publication without restarting user services, and only then the storage stage.

## Checkpoint 10: trained latent upscaling through Stage 2

This is another implementation checkpoint, not completion/publication. The
isolated candidate now uses wire protocol10, carrying distinct base/final
canvases and the selected trained-upscaler path. Production remains unchanged.

The installed MinimaxH3LatentUpscaler3D provider is ported with the actual
LatentResizer3D evaluation operations: inferred state-dict layout, residual and
temporal blocks, GroupNorm32, scale embedding, trilinear interpolation,
normalization constants, temporal chunk32/overlap/blending, FP16 computation and
CPU output restored to the input dtype. No filename allowlist or whole-model
hash gate is added. The ordinary loader reports missing/incompatible weights.
The selected file can have an alternative filename/extension.

Stage2's factor defaults1.0, matches Director's finite1..4 range and only applies
with Re-noise video prediction. Continue latent retains the editor's factor/file
without activating them. Disabling/reopening Stage2 retains those fields and
unfinished SIGMAS. Active transport explicitly requires Re-noise so it cannot
silently ignore an executable upscale instruction. This is a wire-consistency
check, not a restriction on model names or alternative files.

The handoff scales the clean video prediction and aligned I2V keyframes, then
recreates the video-only CPU noise from the same u64 seed before flow re-noising.
Audio continues from Stage1 at its existing size. Stage2 DiT sizing, decode,
wire result and output metadata use the final canvas. Reference conditioning
blocks keep their own geometry. Exp with upscaling is still missing its original
high-resolution prefix transport and reports a specific error; no old denied
Exp diagnostic was retried and this combination is not accepted.

Component evidence, all new outputs directly on the encrypted test volume:

- Synthetic FP32 CPU model: temporal3 and35 match the AST-selected actual Python
  provider exactly (MAE/RMSE/max0), exercising ordinary and chunked paths.
- Actual installed 1,381,153,616-byte FP32 weight file, GPU/FP16 computation:
  temporal3 and37 produce byte-identical Float32 output to the actual provider
  execute/math path, shape[1,24,T,6,8], factor1.5, all differences0. The temporal37
  case used the final numerical implementation; Rust1.754s/Python2.755s are
  component wall observations, not a whole-H3 performance comparison.
- The oracle selects definitions with AST and never imports the provider module,
  Director configuration or private runtime state. It uses the provider's valid
  no-Comfy-MM cache fallback. Its numerical proof does NOT establish whole
  ComfyUI model-management/lifetime parity or whole-H3 Stage2 Python parity.
- Initial standalone GPU component exited1 without initializing the existing CUDA
  runtime bridge (its ELF linked only CPU Torch libraries). Its runtime console
  and exact exception were not read. With bridge initialization in this example,
  subsequent CUDA probes pass; no broader cause is claimed from the exit code.
  All own component children were reaped and each normal coordinator Ticket closed.

Native ui72 completes152 actions at1100x760 logical/1375x950 physical. It edits
factor1.25/free upscaler path, disables and restores Stage2, saves/reopens, and
verifies a frozen CPU-fixture request with base384x512/final480x640 and exact
selected upscaler. The actual screenshot was visually inspected: factor/file
label, field and shared browse icon align/read correctly at the smaller size.
This is scoped visual evidence, not acceptance of every affected dialog.

The standard Reference GPU worker completes4+4 steps, base416x224/final512x288,
39frames24fps, direct H.264/yuv420p/BT709 limited-range MP4 with AAC32kHz stereo.
Duration1.625s and52000 samples/channel, decoded audio RMS.0066918328485259,
peak.038102127611637115. Observation after submission UI closes73.635s.
`upscalegpu04` proves UI submission, normal worker transport, trained handoff,
second-stage sampling and export. It does not claim full Python/H3 parity.

I2V guide-path acceptance `upscalegpu05` also completes4+4 steps with our own
portrait:416x224 to512x288,39frames24fps,1.625s,52000samples/channel AAC32kHz
stereo. Decoded RMS.003836698970060452, peak.0246221162378788; observation after
submission UI closes75.687s. This proves the aligned-keyframe handoff executes
through the standard worker, not its full Python numerical parity. Both standard
GPU workers exited before their jobs published success; own servers/UI exited0
and shared Tickets were released. Final candidate
package is checks-139d22f9; worker-c35ac619, protocol10, SHA
b3f2bbe29dc876912580b0aae83619eb902b7c11d9c562ce169d82d2bc53183b.
185 tests (133UI+12contract+17API+3F05+1compatibility+1adapter+18shared), fmt,
normal/QA UI and server Clippy/build pass; worker Clippy/build and15 wire tests
pass. Runtime-sensitive acceptance remains separate from those tests.

### All 15 groups: functional slices versus remaining implementation/acceptance

No whole group is final-accepted. This table is a concrete backlog, not fifteen
equally weighted completion points. Existing features outside this UI change
remain preserved. Reference original-links/RAM thumbnails belong to the later
storage stage, not an added publication prerequisite in this stage.

| Group | Functionally implemented and evidenced slices | Missing implementation or specific remaining functional acceptance |
|---|---|---|
| 1 | Contextual project/folder/library picker; direct image/video/audio imports in Cast/Section/scene/audio/Story; target retention, search/cancel/reopen happy paths | Finish wrong-type/missing-library/error/actual OS-dialog cases across contexts; common final visual/privacy pass |
| 2 | Live Cast edits/Undo, semantic relationship/retention/target fields, six Section roles and native Remove/Undo/Redo; eight Director manifest cases | Complete default-image cards/crop/guide strength; demonstrate each role reaching actual H3 conditioning, beyond metadata/UI |
| 3 | Cursor-aware @completion, tags, Global off/Prefix/Suffix, catalog/snippets/Phrase Composer and inline Enhance review paths | Close catalog reload/error cases and all completion contexts; common final visual/privacy/keyboard pass; UI-BUG-002 not globally closed |
| 4 | Existing Story outline/expand/review/draft retention/Retry and exact profile copies; text/provider fixture slices | Director semantic image-pair timing, editable phases/progress/provider/profile UI and legacy normalization; remove lifetime quotas and prove >64 proposals; full coherent fixture flow |
| 5 | Shared library resource selector/search/preview/reopen | Projects/Characters/Assets editor, thumbnails/sort/delete and reference-card Save/Update workflows; protect external originals; storage copying is not extended |
| 6 | Project/scene model inheritance/free component paths/shifts; Standard default, five scene types/clip retention, quality/ratio/orientation/Reference quality and projectFPS independent from H3;18 geometry cases | Reference resized/padded pixel parity and actual large-reference GPU; actual OS file choose/cancel/late reply; mixed-source/projectFPS montage |
| 7 | Full-u64 seed/control-after-generate, manual SIGMAS and retained inactive drafts, four Run modes, Euler/res_multistep, trained upscaler/Re-noise normal vertical | Project/scene LoRA resolver+worker patching; other runtime samplers; complete Stage2 Python parity, high-res Motion/Exp prefix; Run dependency/Stop/Cancel/retry/network cases; legacy video-tab seed/steps restriction |
| 8 | Existing native Timeline and responsive toolbar correction; smaller native dialog/action slices | Remaining Section/context/group operations, ordered Ctrl/Shift selection/Delete/text-focus safety, ZoomToFit/range/display/waveform and collapsible side panels; whole normal/smaller layout acceptance |
| 9 | Earlier LastFrame and normal MotionContext end-to-end; reference-video/audio and partial Combo/Influence paths | Exp remains failed/unaccepted with old diagnostic denial; finish Combo/Influence and transitions/Blend default3; enlarged-context transport, dependent generation and raw audio acceptance |
| 10 | Existing optional frozen-plan review, direct Generate retained | Implement optional Approve and Generate against exactly reviewed snapshot, stale review refresh only on that optional path; actual UI/admission proof |
| 11 | Existing status/progress, new upscaler phase messages | Sampling image/video preview wire/runtime/UI with cadence and bounded memory; actual Hide/reveal/focus behavior |
| 12 | Existing explicit A/B selection/synchronized transport/one audio source | Default pair, side swap and per-side Accept/Reject; Attach as Take/Captures and cleanup/provenance/original protection; actual flows |
| 13 | Existing direct fixed MP4 preset and Final mix/frozen CPU export slices | One-action Final plus fresh-seed missing-generation continuation, output filename; generated transitions, larger source import/cancel/retry/save-error acceptance; no extra codecs |
| 14 | Existing session reference cache | Expose actual RAM-budget accounting/eviction and Clear in UI; no disk encoding cache or unrelated memory changes |
| 15 | Existing source-probe/import paths | Check Media/Relink preserving IDs/affected bindings; bounded interval decode/preprocessing for larger usable sources; component+native acceptance |

Approved fixes outside the group table: F05 injected precommit/postrename save
failures pass; K1 remaining historical normal-path binary/hash gates and K2 image
ceilings still need completion. Dedicated own-fixture UI-BUG-001 server-down
project retention and UI-BUG-003 Generate error diagnosis remain open after UI
work. No user runtime logs are authorized by this checkpoint. Hide observation
remains not reproduced, not a proven product defect.

Separate open requirements: full functional/algorithmic Python parity, audited
model-owner/unload/budget/dynamicVRAM memory gaps and P08 broader montage
acceptance. Byte-identical component output or earlier Rust-to-Rust RGB/PCM
regression is not blanket acceptance of those requirements. No memory optimization
was integrated. All-code review follows complete UI publication, per user order.

Next: finish LoRA main-target resolver and actual worker patching including clean
Stage2 rules, then the remaining runtime sampler/Reference pixel+large-reference
paths and coherent group flows above. Preserve narrow own-data testing, ordinary
GPU reservation, all previous denials, and the final visual/publication/storage
ordering. No intermediate product activation, user-service restart or push.

## Checkpoint 11: project/scene LoRA resolver and real model patching

The H3 main-target LoRA vertical is now connected through authoring, frozen
request, the standard worker's existing PatchOwner/static quantized-weight
lifecycle, generation and Take provenance. Wire11 carries an ordered list of
model-only paths/strengths. The project main stack is shared by Reference/I2V;
scene enabled/Inherit Global/Add To Global/Replace Global/Disable LoRAs follow
the actual Director resolver. Disabled/None/both-zero rows do not execute;
negative strengths and duplicate ordered rows are preserved. H3 retains CLIP
strength metadata but applies only model strength, as Director does. Stage2
clones the original model selection with internal LoRA rows cleared, leaving
text encoder/VAEs unchanged. No new filename/model/hash allowlist was added.

The worker uses actual H3 key aliases (native, diffusion_model and lora_unet),
the existing adapter loader and existing static patch/requantization owner.
Absolute paths remain absolute; relative names resolve under models/loras.
Safetensors is currently the bound file reader. Other checkpoint containers,
Director's inventory/filter/info and fuzzy short-name resolution remain explicit
LoRA follow-ups; do not describe those as implemented. Unknown adapter-key
warnings currently remain in the patch diagnostics owner rather than a new UI
warning stream, another open feedback criterion. A loader/application error has
specific h3_lora_load_failed text rather than a new preflight policy restriction.

Project/Scene LoRA editors use ordinary file selection, unbounded finite model
strength, row enable/Add/Remove/Up/Down and the four scene merge modes. Recurring
tools use the shared32px icon/tooltip/accessibility component. Apply combines
pending Cast/Section/global edits into one SetCreative transaction instead of
overwriting them; a regression covers this. Failed Apply retains its draft and
error, successful server confirmation closes the dialog. A selected file reply
uses a unique per-session/per-row identity that follows reordering and is
discarded after removal/reopening, avoiding cross-target late replies. Actual
OS portal late-reply acceptance is still pending; this code is not that proof.

Eight AST-selected actual Director resolver/normalizer cases pass, including
inactive override modes, ordered duplicate/negative/zero/disabled rows and UI
state. The test compares typed numeric values, not JSON's int/float spelling.
Actual ComfyUI model_lora_keys_unet with a tiny MiniMaxH3 state facade gives
exactly the same four aliases, without importing ComfyUI modules or scanning any
model/config/runtime directory. Three real synthetic safetensors cases bind two
ordered patches at0.75/-1.25 to audio_patch_proj.weight; zero model strength skips
a deliberately nonexistent file. These are CPU alias/loading/ownership proofs,
not new full quantized-LoRA Python numerical acceptance.

Native ui73:162 actions at1100x760logical/1375x950physical, project0.75 plus
scene Add To Global -1.25, Undo/Redo, save/reopen, trained Stage2 factor1.25 and
exact frozen ordered rows. Its actual Scene LoRA screenshot was visually
inspected: labels, row tools, strength/file fields and Apply/Cancel fit and align.
Normal-size ui74 completes210 actions with retained inactive Stage2 and invalid
raw SIGMAS over model switches at1522.40625x1680logical/1903x2100physical. Its
Project LoRA screenshot is actually visually inspected, including the aligned
parent settings dialog. Final ui75 additionally asserts the complete creative
snapshot after Undo and restored merge after Redo, rather than accepting key
events alone; its result is recorded in the encrypted checkpoint report. Whole-UI visual acceptance
and privacy/tooltips across every affected state are still open.

Standard-worker paired GPU proof, same own portrait/seed42/shifts9.5+2.25,
I2V4+4steps, trained upscale416x224 to512x288,39frames24fps,MP4 H.264/yuv420p/
BT709tv/AAC32kHz stereo,1.625s,52000samples/channel:

- loragpu01 uses our newly synthesized rank1 audio_patch_proj LoRA. Decoded PCM
  RMS.003856999441212025,peak.026254674419760704. Observed after submission UI
  closes71.610s.
- loragpu02 uses the same worker without the adapter: RMS.003836698970060452,
  peak.0246221162378788,75.667s. Both its decoded RGB8 and float PCM match our
  checkpoint10 upscalegpu05 control byte for byte, proving the no-LoRA regression.
- Adapter versus no-adapter RGB8 MAE10.8257088077,RMSE15.3435897877,max154,
  correlation.9171472272; AAC-decoded PCM MAE.000968237865449,
  RMSE.00141448559099,max.0140735167079,correlation.9324118538. This establishes
  actual adapter influence through this Rust pipeline, not a quality score or
  full Python-pipeline parity. Raw latents were not dumped by a broadened observer.
- Both own standard workers exited before successful publication; own UI/server
  exited0 and normal shared coordinator Tickets closed. No GPU optimization or
  new model-management strategy was introduced, and no user job was interrupted.

Final UI/server package checks-e4c04366 (wire11), worker-4075e8fc SHA
426cfa26cbaaa81060a16f66e0d7d23b7e58aefa22f5f8aed0ea9821fa72d803.
188 tests pass (134UI+13contract+17API+3F05+1compatibility+1adapter+19shared),
plus15 worker wire tests. Normal/QA UI and server build/Clippy/fmt and worker
build/Clippy/fmt pass. A preexisting clipboard timing test failed once during an
early parallel build, then passed its focused run and all later full suites;
no unrelated clipboard product change was made, and its first failure's cause
is not established. F05 fault-injection remains green.

Explicit older-format export now reports that active LoRA/upscale recipes or
their Take provenance cannot be represented by the frozen older reader; it
does not write an unreadable or semantically changed copy. Normal Save keeps
everything. Inactive editor metadata can be omitted in the expressly
requested older copy. This is a concrete format-compatibility requirement, not
a model-choice restriction. Tests retain the complete original. No historical
user project/file was opened or rewritten.

A brief executor transport failure happened before a UI patch was read/written.
Read-only verification established it had not executed; the ordinary retry then
succeeded. Already-running own tests were checked before any retry: loragpu02
had completed and reaped its workers. No GPU job was duplicated. No new approval
rejection occurred, no runtime console was read, and earlier Exp/log denials
were neither retried nor broadened.

### Remaining work as finishable flows

No whole group is final-accepted. The checkpoint10 all15-group table remains
the full backlog with the following group7 change: project/scene stack resolver,
normal worker patching, clean Stage2 policy, frozen provenance and paired GPU
influence are now evidenced. LoRA inventory/filter/info/short-name support,
broader file formats and UI warning feedback, plus actual OS-file-dialog/late
reply and missing/incompatible-adapter UI errors still need closure.

Continue with complete authoring flows rather than only expanding group7:

1. Story/Enhance and reference-library flow (groups1–5): editable phases,
   semantic ordered image pairs, providers/profiles/legacy normalization,
   >64 proposals without lifetime quota; then Projects/Characters/Assets,
   reference-card Save/Update and the remaining field-near media errors. Use
   local own provider fixtures, not private credentials/external submissions.
2. Timeline/review/final flow (groups8/10/12/13): ordered multi-selection and
   Section commands, waveform/responsive panels; default A/B and per-side
   decisions/Attach/Captures, optional reviewed-plan Generate, one-action Final
   with missing generation, fresh seeds, correct export and cancel/retry.
3. Runtime-sensitive closure (groups6/7/9/11): Reference pixel/large-reference
   parity, remaining samplers/Run dependencies, continuity/transition behavior
   and sampling preview/cadence/Hide. Exp's missing high-resolution prefix
   transport is an implementation gap, separate from the specifically denied
   earlier masked3step diagnostic; do not retry that diagnostic or denied logs.
4. Cache/media-health flow (groups14/15): actual RAM budget/clear and
   CheckMedia/Relink/interval-bounded larger-source preparation. No persistent
   encoding disk cache or storage-stage changes.
5. Approved K1/K2 and own-fixture bugs001/002/003, then coherent image/video
   regression and actual normal/smaller whole-UI visual/icon/privacy acceptance.
   Read the frozen Qwen baseline/experiment decisions before any relevant Qwen
   setting change; this task is not a new optimization experiment.
6. Only the completed, visually accepted UI candidate is published reversibly,
   without restarting user services. Only after publication may the separately
   specified original-link/RAM-thumbnail storage stage start. All-code review
   follows the user-chosen publication order.

Full functional/algorithmic Python parity, the already audited model-owner/
unload/minimum/budget/dynamicVRAM gaps and broader P08 acceptance remain explicit
open requirements. They are not accepted by the new component or Rust-pair
results, and this UI stage did not turn into a performance redesign. Production
and original dirty source baselines remain unchanged; D05 is untouched. No push.

## Checkpoint 12: Story lifecycle and library vertical flows

UI4 now has semantic pair identity (`pair_with`) independent of asset-ID sorting,
explicit end-image selection including reuse/self-pair, scene/Section source
editing, single outer review scroll, provider settings in Story, source profile
file/inheritance and parameter inspection, and phase counts/elapsed/approximate
ETA. The estimate is conditional on observed completed work, not a promise.
Elapsed time freezes at terminal/review states and restarts for Retry/expansion.
The five source v3 profiles remain unchanged. Legacy temperature manifests and
v2 profiles normalize in memory, including provider animation inheritance,
without rewriting the user's source files. This deliberately avoids Director's
write-on-load migration, as the approved recommendation required preservation.

The three fixed lifetime quotas (authoring registry, applied records, inline
Enhance registry) are removed. Existing active-work concurrency remains. An API
flow applies 66 proposals in the same project; it passes. The pure Director
Story projection oracle passes 24 focused cases, including reversed start/end
sorting and a reused image. This is contract parity, not an LLM-quality or H3
pipeline claim. Native Story tests exercise inventory, Outline, close/reopen
retention, a deliberately failing expansion, Retry with completed scenes kept,
review edits, Apply, a separate Cancel/Retry, Section Enhance review/Apply and
saved-project reopening. Both 1620x1080 and 1100x760 pass with own loopback-only
provider fixtures. No external provider or GPU is invoked.

UI5 now includes Projects/Characters/Assets tabs, RAM thumbnails through the
existing cache, search, name/recent ordering, standalone Asset Save/Update,
manifest metadata editing and removal, and copying selected assets to the own
project. The reference card directly saves/updates when a library is selected;
otherwise it opens the explicit library selection. Pending Cast edits commit
first. The resulting library identity/revision is recorded on the reference,
so a later card action updates the intended entry. Missing deleted entries
become new saves. Entry removal never deletes content blobs, external originals
or project-owned copies. API tests prove stale revisions fail without mutation,
reopen persists, and both original/copy media remain readable after removal.
These changes reuse existing bundle storage only; original-link conversion and
all general storage-stage changes remain deferred until UI publication.

Library character authoring is still incomplete: direct entry title/description
editing is available, but creating/editing the complete character (all image,
voice, relationship and per-image flags) entirely inside the library is not yet
implemented. The current character copy transport also predates those extended
fields, so full metadata preservation must be closed before claiming UI5 parity.
Likewise, provider settings are now reachable in Story, but the existing separate
connection-profile/transport choices and preferred-profile persistence still
need final alignment with Director. Do not mark either whole group accepted.

The expanded API suite exposed intermittent `storage_busy` in two unrelated
concurrent save tests. A deterministic component test demonstrates a concrete
lock-lifetime issue: closing the parent's descriptor alone does not release a
flock while an inherited/cloned descriptor remains open. A scoped lock guard
now explicitly unlocks at transaction end; it still rejects a concurrent writer
and cannot unlock the next transaction. F05 precommit/postrename tests remain
green. The initial failures' exact child-process history was not instrumented;
that proposed timing explanation is separate from the proven descriptor issue.
No lock policy, destination boundary or access check was removed.

Normal/QA UI and server build, Clippy and formatting pass. The focused check
suite now has 197 passing tests (134 UI, 13 contract, 19 API, 5 profile, 1 pair,
3 F05, 1 lock, 1 compatibility, 1 adapter, 19 shared H3). Candidate protocol stays
11, and worker source/worker-4075e8fc are unchanged from CP11. This stage does not
repeat H3 GPU or full Python generation; prior component/Rust-pair evidence is
not promoted to full Python functional/algorithmic acceptance.

One scripted label/test edit was rejected by automatic review because an
incorrect string slice would have truncated the remaining library.rs file.
Read-only inspection confirmed the rejected operation did not execute. A narrow
contextual apply_patch preserved the file and was approved; its build/tests
pass. This was a new destructive-edit rejection, unrelated to the older Exp or
runtime-log denials, which were not retried. A pure Python source oracle also
created an unintended bytecode cache in Director's source tree; its exact
recorded file was cleaned through normal approval, and future bytecode writes
are disabled. No Director source file was changed. No runtime console was read.

### All 15 groups: functional state and exact open closure

All rows still await the shared final visual/privacy/keyboard and publication
acceptance. A tested slice is not a whole-group acceptance.

| Group | Functionally implemented and evidenced | Missing implementation or concrete remaining acceptance |
|---|---|---|
| 1 | Contextual Project/Library/Folder picker and direct image/video/audio/Story field import; own happy/cancel/reopen slices | Wrong type, missing library/file, OS-dialog cancellation and late replies across every target; deferred storage contract is not implemented here |
| 2 | Live Cast/Undo, semantic relationships/retention, six Section anchor roles; own Remove/Undo/Redo and manifest oracle | Finish default-image/crop/guide-strength cards and actual H3 conditioning evidence for every role |
| 3 | Cursor-aware completion/tags, Global off/Prefix/Suffix, catalog/snippets/Phrase Composer, inline Enhance review | Close catalog error/reload and all text-context completion/privacy/focus cases; dedicated BUG002 final acceptance |
| 4 | New semantic pairs, phases/editable source/progress, provider/profile inspection, legacy in-memory normalization, >64 proposals; complete own Story and Enhance lifecycle in normal/smaller native UI | Connection-profile/transport consolidation and preferred profile persistence; slow-work ETA visibility, all profile/failure/Continue/Replace paths and shared final visual/privacy acceptance |
| 5 | Projects/Characters/Assets, thumbnails/search/name-recent sort, Asset Save/Update/copy/delete, reference-card direct Save/Update with revision origin; native normal/smaller flows | Full standalone character creation/editor and metadata-preserving image/voice/relationship/per-image copy transport; broad missing-media/storage-error cases and final keyboard/tooltips |
| 6 | Free component paths/shifts, model inheritance, Standard new project, scene types, canvas geometry/Reference Quality/projectFPS; source-oracle and prior GPU slices | Reference pixel/large-reference Python parity, OS choose/cancel/late replies, mixed-source/projectFPS montage |
| 7 | Full-u64 seed/control, SIGMAS/stage drafts, four Run modes, Euler/res_multistep, trained upscaler, project/scene LoRA merge and actual worker patching, clean Stage2/no-adapter regression | LoRA catalog/filter/info/short names/formats/warnings/errors; remaining samplers; Stage2 Python parity and high-res Motion/Exp prefix; Run dependencies/Stop/Cancel/retry/network and legacy video-tab limits |
| 8 | Native Timeline and responsive toolbar slices; normal/smaller own dialog tests | Section/context/group editing, ordered Ctrl/Shift selection/Delete focus safety, ZoomToFit/range/views/waveform/collapsible side panels |
| 9 | Earlier LastFrame/normal MotionContext and reference-video/audio paths | Combo/Influence and transition/Blend default3 closure; Exp is unaccepted, high-res prefix transport missing; older denied diagnostic remains prohibited |
| 10 | Existing optional frozen-plan review; ordinary Generate retained | Optional Approve and Generate against precisely reviewed inputs, stale refresh and actual UI/admission acceptance |
| 11 | Status/progress and upscaler phase messages | Sampling preview transport/runtime/UI, cadence, bounded memory and Hide/reveal/focus acceptance |
| 12 | Explicit A/B with synchronized transport and one audio source | Default pair/swap/per-side decisions, Attach as Take/Captures, cleanup/provenance and external-original protection |
| 13 | Existing direct MP4/H264 CRF19/yuv420p/AAC and frozen Final CPU mix slices | One-action Final with fresh-seed missing generation, output filename; generated transitions, larger sources, cancel/retry/save-error closure; no extra codec choices |
| 14 | Existing session reference cache | Actual RAM-budget accounting/eviction/Clear UI and tests; no disk encoding cache or general memory redesign |
| 15 | Existing source inspect/import paths | CheckMedia/Relink with stable IDs/affected bindings and bounded-interval preparation of larger usable sources |

K1/K2 and own-fixture BUG001/003 remain explicit work after the UI flows; F05
remains tested. All full Python functional/algorithmic and audited memory-owner,
unload/minimum/budget/dynamicVRAM gaps remain open. No memory optimization was
integrated. No skarp service/process/data was inspected, restarted or changed.
No publication, general storage migration, commit or push occurs at this
checkpoint. The next coherent work is the remaining full character-library
editor/metadata path and Story provider binding, then Timeline/review/Final,
with the LoRA catalog/error closure kept on the same finite backlog.

## Checkpoint 13: full character editor and bound Story providers

Groups4/5 now have functionally complete core authoring flows with focused
native acceptance. This is not shared final acceptance of either group or the
whole candidate. The standalone library character editor works without an open
project, imports images/voices through the contextual picker, and preserves
image enabled/preview/contribution fields, reference kind/strength/retention,
relationships, and voice language/guidance/per-file metadata. Save, edit,
reopen and copy back to a project retain these properties. Portable resource
keys are remapped to project asset IDs; multiple logical image records may
share one stored resource. Self-relationships follow the copied reference.
Legacy character documents without the optional rich reference remain readable.
This reuses current library bundles; no storage migration or token dependency
was introduced. A failed Save retains the draft and allows correction/Retry.

Story's API transport is owned by the selected connection profile (Koboldcpp
default, optional Ollama), with an explicit advanced override. Preferred Story
profile persists through ordinary UI preferences. Existing profile JSON without
the new transport field keeps its previous default. Append, Continue and
Replace now pass native flows; slow expansion exposes an actual positive ETA,
failure retains completed work/progress, and Retry/Cancel/Enhance/reopen pass.
The five source Director profiles and the CP12 pure projection oracle are
unchanged. These local provider fixtures do not prove live LLM output quality.

Replace exposed a genuine project validation error: completed Takes were
required to reference an active scene, although Replace intentionally removed
that scene while retaining its output and provenance. Historical Takes may now
reference a removed, nonempty scene ID; resource, frozen-input and provenance
checks remain. Replace clears previous scene/Section settings, scene LoRAs,
scene kinds and audio tracks, while retaining prior Takes and resources exactly.
The persistence regression removes all scenes and reopens all four Take states
without losing their resources/provenance. The native Replace fixture includes
actual prior audio tracks and three Takes. An earlier postcondition mistakenly
assumed an audio asset alone created a track; that fixture assumption was
corrected rather than changing product behavior to satisfy it.

The latest immutable package is `comfyui/ui-groups-20261021/packages/checks-97c2787f`.
All14 check stages pass: 217 tests, server/UI/QA Clippy with warnings denied,
formatting and normal/QA builds. Final native cases on that exact package pass:

| Own case | Flow | Size | Actions |
|---|---|---|---|
| characterui04 | Standalone character, failed Save/correction, image/voice, edit/reopen/copy | 1620x1080 | 112 |
| characterui05 | Same character flow with smaller window | 1100x760 | 112 |
| storyui14 | Replace with actual previous audio tracks and retained Takes/resources | 1620x1080 | 79 |
| storyui15 | Slow Append, ETA, retained failure, Retry, Cancel, Enhance, reopen | 1620x1080 | 120 |
| storyui16 | Continue with Ollama profile, preference persistence, Retry/Cancel/Enhance/reopen | 1100x760 | 116 |
| libraryui15 | Asset CRUD, direct reference Save/Update, project thumbnail and reopen | 1620x1080 | 114 |

Own screenshot inspection confirms retained failure progress and visible ETA.
The smaller character form is usable with scrolling, but its save error remains
in the parent library window and can be partly obscured by the editor; move
that feedback into the editor during final layout closure. Shared long-value,
Hide/privacy, keyboard/focus, tooltip/icon consistency and all-panel visual
inspection remain outstanding. These screenshots are not final visual approval.

### All15 groups: current closure ledger

| Group | Functional state | Exact remaining work/acceptance |
|---|---|---|
| 1 | Contextual picker including no-project library character target works | Wrong type, missing library/file, OS-dialog cancellation and late replies across every target; common final privacy/layout |
| 2 | Existing live Cast/Undo, roles and relationships retained; rich library copy now preserves metadata | Default-image/crop/guide-strength cards and actual H3 conditioning evidence for every role |
| 3 | Existing completion/catalog/snippets/Composer/Enhance retained | Catalog error/reload, all text contexts, BUG002 final and common focus/privacy |
| 4 | Core Story/Enhance functionally complete and focused native accepted, including bound providers/preference, Append/Continue/Replace, ETA and failure lifecycle | Shared final visual/long-value/privacy/keyboard acceptance; new-format/older-reader rollback compatibility; no live-provider quality claim |
| 5 | Core Projects/Characters/Assets and full standalone character authoring/copy functionally complete and focused native accepted | Broad missing-media/storage-error feedback, editor-local save error, shared final visual/keyboard/tooltips; rich-metadata older-reader compatibility |
| 6 | Existing component paths/shifts/types/geometry/FPS implementation retained | Reference pixel/large-reference Python parity, OS chooser/error/late replies, mixed-source/projectFPS montage |
| 7 | Existing seeds/Run/SIGMAS/upscaler/LoRA worker implementation retained | LoRA catalog/filter/info/short names/formats/warnings/errors; remaining samplers, Stage2 Python/high-res Motion/Exp prefix; Run stop/cancel/dependencies/network and legacy-tab limits |
| 8 | Existing Timeline/dialog slices retained | Section/context/group operations, ordered Ctrl/Shift selection/Delete focus, ZoomToFit/range/views/waveform/collapsible panels |
| 9 | Existing LastFrame/normal Motion/reference-video/audio evidence retained | Combo/Influence and transition/Blend default3 closure; Exp missing prefix/unaccepted; older denied diagnostic is still prohibited |
| 10 | Existing readonly optional frozen-plan review retained; ordinary Generate unchanged | Optional Approve and Generate of precisely reviewed inputs, stale refresh and UI/admission tests |
| 11 | Existing status and upscaler messages retained | Sampling preview transport/runtime/UI/cadence/bounded memory and Hide/focus |
| 12 | Existing synchronized explicit A/B retained | Default pair/swap/per-side decisions, Attach as Take/Captures, cleanup/provenance/external-original protection |
| 13 | Existing direct MP4 and frozen CPU mix retained; no new Final implementation this checkpoint | One-action Final/missing generation/fresh seeds/filename; transitions, larger sources, cancel/retry/save-error closure |
| 14 | Existing session cache retained | RAM accounting/budget/eviction/Clear UI; no disk encoding cache or storage redesign |
| 15 | Existing import/inspection retained | CheckMedia/Relink/stable IDs/affected bindings and bounded-interval larger-source preparation |

Before publication, explicitly verify older-reader behavior for rich library
metadata, provider transport/pairs, and retained Takes whose scenes were
removed. The existing compatibility-copy test is not proof for these new forms.
Do not silently strip metadata or modify users' originals. K1/K2 and own-fixture
BUG001/003, full image/video regression, protocol11 final package/start checks
and whole-UI final acceptance remain. Functional/algorithmic Python parity and
audited memory-owner/unload/minimum/budget/dynamicVRAM gaps remain separate open
requirements. No GPU inference or reservation was used in CP13, worker/inference
sources are unchanged from CP12, and all own UI/server/provider processes exited
and were reaped. No runtime console or user data was read. No operation was newly
rejected this checkpoint; CP12's rejected broad edit and all earlier denied
Exp/log operations were not retried. No publication, service restart, storage
migration, commit or push occurred.

Next: complete Timeline/optional reviewed Generate/Final as coherent flows,
closing their finite criteria and the remaining LoRA catalog/error work. Then
finish the other ledger items and common compatibility/regression/visual gate;
publish the completed candidate reversibly before the separate storage stage.

## Checkpoint 14: optional reviewed Generate, Timeline commands, direct Final

Group10's optional reviewed-generation path is functionally implemented. Review
captures one exact u64 seed, the selected scene and semantic input revision,
shows model/settings and compiled conditioning, and starts that reviewed scene.
Changing relevant inputs or seed options requires Refresh/review; a revision
change alone with identical inputs does not. The existing server revision check
still closes the admission race. Draft fields commit before Review, Refresh or
Approve, and invalid sampling raw text is handled through the same field error
path as direct Generate. Direct and reviewed Generate share one seed preparation
function; the widget advances only after confirmed idempotent admission. Normal
Generate is still independent of Review. No model/file/binary gate was added.
The shape label explicitly identifies base sampling; Stage2/upscale settings are
shown separately rather than pretending base resolution is necessarily final.

Own native Review flows pass at1620x1080 and1100x760: prepare, change description,
observe the disabled stale approval, verify no extra Take, Refresh, approve the
new exact seed/prompt, then ordinary Generate again. The CPU backend receives
seeds9876543210123/9876543210124; the widget ends at9876543210125. Three pure
tests also cover full-u64, input identity/model/geometry changes, a stale compile
response and legacy random seed selection exactly once. This proves the UI /
admission contract with a synthetic backend, not new H3/Python numerical parity.

Group8 now has ordered Ctrl-toggle/Shift-range selection for Sections and audio,
with the clicked item primary, Timeline-scoped Delete/Backspace, group Section
deletion with Undo/Redo, and Section context Split/Delete. Split uses the snapped
midpoint and preserves authored image/dialogue metadata; explicitly trimmed video
source intervals split proportionally, following Director. Pending reference edits
are applied to the effective project before split so they cannot be overwritten
by an older creative snapshot. Zoom to Fit, lower zoom for long timelines, bounded
tick density, and explicit shelf/inspector collapse controls are implemented.
Native focused flows pass normal/smaller sizes, including Delete inside a text
field leaving Sections intact. A real focus gap was corrected: clicking a Timeline
object now gives it keyboard focus, and Delete uses that scope rather than the
previous text field or a floating window. The specifically tested Section subset
is not full group8 parity: group duplicate/drag, complete context commands,
conditional Fit Last/Fit All, range/display modes and waveform remain.

Group13 now has a single Render current Timeline action for an already usable
montage. It freezes and admits internally through existing queue/idempotency;
manual Freeze/Render remains optional. The primary action is at the dialog top
for smaller windows. Own native tests verify direct export, a deliberately lost
admission reply retrieved as the same job, and Cancel during encoding followed
by Retry. An injected publication failure retains the encoded result; retry saves
the same bytes (own test SHA3f0f976e2a924532921041439e36ef4e1d01a3c3f37d69dcc00663d4cd075495).
No format/codec or mix algorithm changed. The12-second source montage has282frames
at24fps after its authored6-frame blend: container/video/audio durations11.750s,
H264/yuv420p/bt709 andAAC stereo44100Hz. In-memory decoded PCM has518175samples per
channel, RMS0.0303613094 and peak0.0800705850. The longer Cancel/Retry fixture
exports3594frames/149.750s with6603975samples/channel, RMS0.0284652887 and peak
0.0803478882. This is genuine CPU montage/encode evidence, not model quality.

Automatic generation for an unfinished Timeline is still missing. Director's
`startFinalVideo` in renderer.js explicitly starts **AllFreshSeeds** for the
generation run, not merely MissingOnly, and resumes pending Final afterward.
Use that source behavior when closing UI043; do not silently substitute another
Run mode. Output filename selection, generated transitions and broader source /
mixed-FPS closure also remain. No claim of full group13 completion is made.

All14 check stages pass with222 tests (140UI,13contract,20API,12persistence,
5profiles,1pair,3F05,1lock,1compatibility,1adapter,19sharedH3,6enhancement),
Clippy and normal/QA builds. Latest immutable package:
`comfyui/ui-groups-20261021/packages/checks-c163adc3`, protocol11. Worker/inference
are unchanged. Review normal case reviewui05, Timeline cases sectioncommands04/05
and Final qfinaldirect03/04/05 use this exact package; the earlier smaller Review
case reviewui04 used checks-8483a89d before the later Timeline focus/layout-only
changes. Focused tests required QA fixes: modifier changes are actual egui0.36
ModifiersChanged events, disabled-button input is explicit, and one fixture
already had a baseline Take (the assertion now compares the real baseline).
An initial compile attempt used an obsolete RawInput modifier field and was
corrected against the installed egui source. No automatic-review rejection or
permission workaround occurred. No runtime console or user data was read.

### All15 groups: finite remaining closure

All rows still await common final visual/privacy/keyboard, complete package /
rollback compatibility and regression acceptance. No group is whole-plan final
accepted merely because a focused native slice passes.

| Group | Functional state | Concrete remaining criteria |
|---|---|---|
| 1 | Existing contextual picker including standalone library target | Wrong type/missing file/library, actual OS cancel and late replies across targets; common visual/privacy |
| 2 | Existing Cast/Undo/roles and rich copy transport retained | Default-image/crop/guide-strength cards and H3 conditioning evidence for every role |
| 3 | Existing completion/catalog/snippets/Composer/Enhance retained | Catalog error/reload, all prompt contexts and BUG002 final/focus/privacy |
| 4 | Core Story/provider/Append/Continue/Replace functionally complete fromCP13 | Shared visual/long-value/privacy/keyboard and older-format/rollback compatibility; no live LLM-quality claim |
| 5 | Full library character/asset core functionally complete fromCP13 | Editor-local save error (currently partly hidden in parent), missing-media/storage errors, rich-metadata older readers and final icons/keyboard |
| 6 | Paths/shifts/types/canvas/FPS retained | Reference pixel/large-reference Python parity, actual OS chooser paths and mixed-FPS montage |
| 7 | Seed/Run/SIGMAS/upscaler/worker LoRAs retained | LoRA catalog/filter/info/short-name/format/warning/error closure; other samplers, Run cancellation/dependencies/network; Stage2 Python/high-res Motion/Exp prefix and legacy-tab limits |
| 8 | New ordered Section/audio selection, Delete/group delete/Undo/Redo, midpoint split, Fit, collapse controls | Group duplicate/move and remaining Scene/audio context commands; conditional fit semantics, range/display modes, waveform; smaller600px and keyboard-only scope/Hide final |
| 9 | Existing LastFrame/normalMotion/reference-video/audio evidence retained | Combo/Influence/transitions, Blend standard3; missing Exp high-res prefix and unaccepted Exp; denied diagnostic still prohibited |
| 10 | Optional exact reviewed Generate functionally complete with native stale/Refresh/seed/direct-Generate proof | Broader network/reconnect/late compile reply and shared final visual/privacy/keyboard acceptance; existing admission race guard retained |
| 11 | Existing status/progress retained | Sampling preview transport/runtime/UI/cadence/bounded memory/Hide |
| 12 | Existing synced A/B retained | Default pair/swap/per-side decisions, Attach/Captures, cleanup/provenance/original protection |
| 13 | Ready Timeline direct Final, lost-admission reply, retained-save-failure retry and cancel/retry now native verified | Unfinished Timeline AllFreshSeeds→Final continuation, filename, transitions/larger mixed-FPS sources and final visual closure |
| 14 | Existing session cache | Actual RAM budget/accounting/eviction/Clear UI; no disk cache or memory redesign |
| 15 | Existing import/inspection | CheckMedia/Relink stable IDs/affected bindings and bounded larger-source preparation |

K1/K2 and own-fixture BUG001/003 still require closure; F05 remains green. Keep
CP13's format cases explicit (rich reference metadata, provider transport/pairs,
and retained Takes with removed scenes); never silently strip an original.
Full Python functional/algorithmic and audited memory-owner/unload/minimum /
budget/dynamicVRAM gaps, P08 broader runtime closure and denied Exp/log operations
remain separately open. No memory patch or new GPU model test was introduced.
All own test processes are reaped; no GPU reservation was taken, no production
source/selector or running user service was changed. No publication/storage
migration/commit/push. Next coherent work: finish AllFreshSeeds→Final continuation
and filename, then the remaining Timeline commands/navigation/waveform and full
ledger; shared final acceptance still precedes reversible UI publication and
the separate storage stage.

## Checkpoint 15: unfinished Timeline through AllFreshSeeds to named Final

Group13 now connects the complete session-local workflow: Render current
Timeline detects a missing accepted generated scene, starts the existing
server-owned **AllFreshSeeds** Run, waits for its accepted results and refreshed
project, then freezes and queues Final using the choices captured at the first
click. Imported scenes remain clips rather than generation targets. Existing
Takes are retained. Changing the next filename/options draft does not change the
pending export. Actual generation/mix inputs changing during the Run ends the
continuation with a specific message and keeps completed Takes. The user can
close the Final dialog; polling and continuation are independent of its visibility.
Project/session changes cancel only the UI continuation, never the server Run.
No automatic cross-session continuation is claimed. Ordinary Generate and the
optional reviewed Generate remain unchanged.

Final filename is carried in the frozen plan and saved-delivery metadata through
publication. Unicode, spaces and an existing case-insensitive .mp4 suffix work;
the suffix is added otherwise. Empty means the existing automatic filename.
Only a basename is accepted, since this field cannot select a different output
destination. Atomic no-replace publication preserves existing files and adds a
collision suffix. This is not the separate storage-policy stage. Old default
Options JSON remains unchanged when no filename is selected; old readers of a
new named-delivery field still need the explicit compatibility acceptance.

Actual native UI/CPU tests cover a project with two accepted scenes and one
missing scene: all three generated scenes receive fresh seeds, the original
three Takes remain, Final uses the new Run's three accepted Takes, and reopening
retains the delivery. qchain01 also changes the next draft while generation is
pending. qchain03 cancels the first running synthetic job before a new Run and
Final. qchain05/07/08 inject a failed first Run and successfully retry through
Final. qfilename01 verifies `Min film β.MP4`; qfilename02 injects a publication
failure and saves exactly the retained encoded bytes on Retry with
`Återprov β.mp4`. Normal1620×1080 and smaller1100×760 layouts were exercised.
The smaller Final options screenshot was visually inspected: filename/labels,
primary actions and audio controls fit; scrolling exposes the remaining options.
This focused view is not whole-UI visual acceptance.

There is an unresolved intermittent failure: qchain02 and qchain06 completed
the replacement Run but Final preparation returned the former generic video-API
network error. qchain06's own structured QA result shows no pending continuation,
no busy/admitting state and that error. The cause is unproved; later passing
tests do not establish a repair. qchain04 instead failed because a QA wait
exceeded the fixture's10-second limit; qchain05 used an8-second wait. That extra
wait was removed in qchain06/07/08; qchain07 and latest qchain08 pass without it.
The client now distinguishes timeout, connection and interrupted-response
failures without exposing URL, payload or raw error details. This improves the
next diagnosis and user feedback; it is not claimed to fix the earlier failure.
Group13 reliability remains open until this is isolated or a controlled failure
and recovery test establishes the relevant behavior.

The final own montage remains282frames at24fps,11.750s container/video/audio,
H264/yuv420p/bt709 withAAC stereo44100Hz. The AllFreshSeeds fixture's decoded PCM
has518175samples/channel, RMS0.0303631005 and peak0.0800705850. These tests use a
synthetic CPU generator and the real montage/encoding path, not H3 generation
quality or Python model parity. Worker/inference code and protocol11 are unchanged.

Latest immutable package: `comfyui/ui-groups-20261021/packages/checks-f4a97856`.
All14 check stages pass with255 tests (143UI,13contract,20API,12persistence,
5profiles,8storage,5assembly,17jobs,1pair,3F05,1lock,1compatibility,1adapter,
19sharedH3,6enhancement), Clippy and normal/QA builds. Latest qchain08 uses this
package; earlier slices retain their exact package IDs in the structured report.
The newly included jobs suite exposed an obsolete test expecting rejection of
full-u64 seed / an unused upscaler path; it now tests the already implemented
valid transport. Long encrypted temporary paths exceeded Unix socket length;
only that suite now uses the shorter own encrypted `g15` directory. No system
or socket-security setting was changed. A failed integration-test console tail
earlier displayed own synthetic scheduler status lines automatically; the check
runner no longer displays runtime console tails. No saved runtime console log,
user data or previously denied log was intentionally read; compiler diagnostics
and own structured acceptance results were used.

### Updated whole15 closure ledger

Core groups4/5/10 remain functionally complete for the documented slices, with
their listed acceptance still outstanding. No whole group is finally accepted.
Group8 did not receive further functional changes in this checkpoint.

| Group | Functional state | Concrete remaining acceptance / implementation |
|---|---|---|
| 1 | Contextual project/library/media picker retained | Wrong type/missing media/library; actual OS cancel and late replies across targets; common visual/privacy |
| 2 | Cast/Undo/roles and rich transport retained | Default-image/crop/guide cards and every H3 role's conditioning proof |
| 3 | Completion/catalog/snippets/Composer/Enhance retained | Catalog reload/error; all prompt contexts, BUG002 and final focus/privacy |
| 4 | Story/provider core complete fromCP13 | Final visual/long values/privacy/keyboard; older-format/rollback compatibility |
| 5 | Rich library editor/card core complete fromCP13 | Editor-local save-error presentation; missing-media/storage errors; old rich formats and final icons/keyboard |
| 6 | Paths/shifts/types/canvas/FPS retained | Reference pixel/large-reference Python parity; actual OS chooser and mixed-FPS montage |
| 7 | Seed/Run/SIGMAS/upscaler/LoRA transport retained | LoRA catalog/filter/info/short-name/formats/warnings/errors; other samplers, Run network/dependencies; Stage2 Python/high-res Motion/Exp prefix and legacy limits |
| 8 | CP14 Section/audio selection, Delete/Undo/split/Fit/collapse | Group duplicate/move; remaining Scene/audio context commands; right-hand split selection; conditional Fit, range/display modes, waveform;600px/keyboard/popup/Hide |
| 9 | LastFrame/normalMotion/reference-video/audio evidence retained | Combo/Influence/transitions; Blend default3; missing Exp prefix and unaccepted Exp; denied diagnostic still prohibited |
| 10 | Optional exact reviewed Generate core complete | Broader network/reconnect/late compilation and common final acceptance |
| 11 | Status/progress retained | Sampling preview transport/runtime/UI/cadence/bounded memory/Hide |
| 12 | Synchronized A/B retained | Default pair/swap/per-side decisions; Attach/Captures, cleanup/provenance/original protection |
| 13 | Ready and unfinished AllFreshSeeds→Final, filename, cancel/error/save retry implemented and focused CPU/native tested | Intermittent API failure isolation/recovery; generated transitions; larger/mixed-FPS sources; named-delivery old readers and final visual/runtime acceptance |
| 14 | Session cache retained | Real RAM budget/accounting/eviction/Clear UI; no disk cache or general memory redesign |
| 15 | Existing import/inspection retained | CheckMedia/Relink stable IDs/affected bindings; interval-bounded larger-source preparation |

K1/K2 and BUG001/003 remain explicit closure tasks; F05 stays green. Final
package/protocol11 start/rollback, all affected dialogs/panels at normal/smaller
sizes, long/empty/filled values, consistent icons/tooltips/keyboard/Hide and
existing image-job regression remain mandatory before publication. Full Python
functional/algorithmic parity, audited memory gaps and Exp/P08 gaps remain
separate unaccepted requirements. No denied Exp/log operation was retried.
All own fixture processes are reaped; no GPU reservation/job, production
selector/source/service change, publication, storage migration, commit or push.
Next: isolate the intermittent Final API failure with structured own-fixture
failure/recovery evidence, then finish the listed Timeline commands/range/views/
waveform and remaining closure ledger. Do not add speculative retry delays or
declare the complete UI ready based only on these CPU tests.

### Checkpoint15 storage blocker after the tests

The structured JSON and Markdown reports were saved successfully. The subsequent
full code snapshot failed with ENOSPC while copying
`inference/fixtures/shared-memory-helpers.json`. `source-checkpoint-15` is a new
incomplete code snapshot (263files,480425042bytes); `support-checkpoint-15` is
empty. Neither `source-checkpoint-15.json` nor the14-to15 diff was created;
the reports' expected snapshot/diff references must not be treated as completed.
statvfs reports only12288bytes available to this executor (about1.99GB free is
reserved/unavailable). The sandbox read view presents the volume read-only;
that must not be confused with the approved execution view that reached ENOSPC.
No files were deleted/moved, no space was reclaimed and no mount setting changed.
Original baseline comparison remains empty, as do inference changes sinceCP14.
The built package and completed acceptance results remain intact. New media/test
output cannot proceed safely until legitimate space is available; do not move
it to a different disk. This blocker supersedes the normal next-step scheduling.

## Checkpoint 16: verified runtime lifetime, explicit Final Retry and approved snapshot cleanup

The space blocker is resolved within the explicit cleanup decisions. Only the
inventoried incomplete source-checkpoint-15 was first removed (263files,
480425042logical bytes;482136064bytes became available). The later04:38UTC
approval allowed redundant source snapshots after restoration proof, while
explicitly retaining test videos/raw outputs outside snapshots.

Historical checkpoints13/12/11 were reconstructed strictly in memory from
retained14 using reversed existing patches. Checkpoint10's patch could not be
verified strictly at `crates/video-contract/src/fixtures/director-h3-loras.json`;
10 was therefore retained, checked against its complete manifest and used to
reconstruct9/8/7. Every reconstructed and actual file hash matched the historical
manifest; paths/inodes/devices and absence of symlinks were checked again before
deletion. Only source-checkpoint-07/08/09/11/12/13 were removed. Actual additional
available space increased6017654784bytes to6485188608bytes at that check. Reports:
`source-snapshot-recovery-verification-02.json`, `source-snapshot-cleanup-01.json`.
Memory/compare source copies were also checked against preserved branch sources:
the five examined copies have2–7 unique source-file contents and were retained
under the instruction to keep unproven/unique copies. No media outside source
snapshots, output evidence, baselines, current source, packages or rollback were
deleted. The read-only three-root inventory explained that most bulk is repeated
inference fixtures/build trees, rather than model weights.

The HTTP lifetime defect is now reproduced independently of Imagine services.
Eight OS threads share the same reqwest client and each perform40requests against
an owned std::net HTTP keepalive responder. A/B/A changes only runtime lifetime:
per-request current-thread runtimes produced30/320 and51/320errors; the persistent
variant produced0/320. The saved nested causes include `dispatch task is gone`
and `runtime dropped the dispatch task`, not timeout. A second experiment uses
the actual new candidate runtime helper:29/320 and19/320 errors in the old pattern,
0/320 with the real helper. The750ms test deadline is unchanged across variants;
no product deadline was increased and no automatic retry loop was introduced.
`client-runtime-probe-01.json` and `client-runtime-probe-02.json` preserve both.

The candidate now owns one long-lived multithread Tokio runtime through
OnceLock. Existing API callers use the same block_on entry point, so pooled
connection tasks no longer lose their runtime after each request. Tokio's
rt-multi-thread feature is explicit; no package installation or model/inference
change occurred. This proves the architectural defect and its component remedy.
It does NOT recover the missing underlying error from qchain02/qchain06: the
old qchain02 structured report has only a QA timeout; qchain06 has the old generic
transport message. Their exact historical cause is still not established.

Final preparation now retains the complete captured request (project revision,
options and filename) after a retryable failure. A visible explicit Retry uses
that request, independently of the next draft. Normal Render still uses current
choices. Success clears the retained request; reopening a project clears its
preparation state, and session/epoch guards reject stale replies. The existing server revision conflict remains the actual
failure if the captured project revision no longer exists. Preparation failure
now retains sanitized operation/code/retryable/elapsed_ms instead of discarding
the failure category. JSON body transport failures keep their category; malformed
JSON remains invalid_response. No URLs, bodies, prompts or raw private errors are
added to UI diagnostics. The error label is moved to the top for small windows.

Native CPU acceptance after the runtime fix passes the failed-first-Run path at
1620×1080(qchain09) and1100×760(qchain10), plus Cancel running generation followed
by a new Run and Final(qchain11). Explicit dropped-freeze-reply tests
qfreeze01/02/03 wait for the known error, keep deliveries at0, change the next
filename draft, then use Retry. The resulting named Final still uses the original
choices, contains exactly the new Run's three Takes and reopens with one delivery.
Observed fault category is synthetic_freeze_lost, operation freeze, retryable true;
there is no automatic retry or extra generation Run. The final screenshot is
captured after the ordinary dialog animation, not by delaying generation or
masking a transport race. qfreeze03 checks the latest error-label placement.

All exports remain282frames/24fps/11.750s withAAC stereo44100Hz and518175decoded
samples/channel. RMS0.0303631005, peak0.0800705850 and the same own output SHA
7f537c863ed90068bc2c17f474e9496214e1bf6facc116a67b47faca0bcdf595 are retained.
These are real montage/encoding and native UI tests with a synthetic CPU generator,
not H3 or full Python model parity.

All14 check stages pass with256tests (144UI plus the previous112), Clippy for
normal/QA and normal/QA builds. Package checks-b1fcb00f covers qchain09/10/11 and
qfreeze01/02; latest checks-11adb50a adds the top error placement and the stronger
project-change clearing assertion, and is used by qfreeze03. Protocol11 and
worker/inference are unchanged. Core groups4/5/10 remain documented functional
slices; group13's AllFreshSeeds/named Final/explicit retry slice is now verified.
This is not whole-group or whole-UI acceptance.

Code evidence now uses `snapshot-ui-groups-incremental.py`: only changed source
files are copied, with a full per-file hash/content-location manifest and patches.
Unchanged large fixtures refer to retained14; full historical restoration must
keep that base. No new full source snapshot is created. The previous CP15 report's
missing full snapshot stays documented as historical failure, not overwritten.

### Remaining closure after checkpoint16

The full15-row ledger inCP15 remains the controlling list. No group8 commands,
range/display/waveform work was silently counted as implemented during this fix.
Group13 still needs generated transitions, broader/larger/mixed-FPS sources,
named-delivery older-format compatibility and shared final runtime/visual
acceptance. Group4/5 editor-local save-error presentation and old rich-format /
rollback tests remain explicit. Groups1/2/3/6/7/9/11/12/14/15 retain every listed
role/picker/catalog/sampler/preview/A-B/cache/Relink criterion. K1/K2 and own-fixture
BUG001/002/003 still need their separate closure; no broad claim that the runtime
fix repairs BUG003 is made. Full Python functional/algorithmic and audited memory
parity, Exp/P08 gaps and the specifically denied operations remain separate.

Next: remaining Timeline group commands/right-side split selection, conditional
Fit, range/display and waveform, then remaining finite ledger and whole-UI
visual/icon/privacy/keyboard/start/rollback/image regression. No production
publication until UI/fixes and final acceptance are complete; storage afterward.
No GPU job/reservation, forbidden log read, user data, product activation/service
restart, commit or push occurred in this checkpoint.

## Checkpoint 17: native Timeline range, display and narrow layout

Group8 now has session-local Sections/Media/Prompts display, whole-second View
start/end fields and draggable range endpoints. The eight finite range cases in
`ui/src/video/navigation-director.json` come from executing Director's actual
`geometry.clampTimelineViewRange` with synthetic numbers. The native regression
also covers nonfinite fallback and the absence of a forced scroll offset after
ordinary pan. Zoom to Fit uses the same range state. Mouse movement and repaint
do not reapply an old offset; only explicit range/zoom/Fit or an actual extent /
width change requests a new offset. Reattaching the same project keeps this view;
opening another project resets it. These are viewport choices, not authoring or
generation parameters.

Sections labels use Section text, Media uses the first native image/video binding
and additional count, and Prompts combines the enabled global prefix/suffix with
the Section prompt. Empty text has the Director Section fallback. Long/newline
labels stay clipped to their own bars and start at the left. Hidden/unfocused
viewports show a generic Section label. Native image/video bindings are typed
separately; the original mixed binding order is not represented. Consequently
mixed-media first-label ordering is still a specific contract/projection gap,
not a claim of complete display parity. Timeline extent still derives from native
Scenes/audio rather than adding a new independent project-duration contract.

Split now selects the newly created right Section only after successful edits,
retaining the existing binding-copy and trimmed-video split behavior. Ordered
Ctrl-toggle/Shift-range, text-field Delete safety, group Delete/Undo/Redo and
save/live reattachment remain exercised in the same native flow.

An initial1100x760 probe (navigation01) exposed an incorrect response rectangle
for long multiline fields. The field already had bounded Resize/internal scroll;
its returned TextEdit response described the entire485px content instead of the
visible frame, confusing scroll-to-control and popup placement. The shared helper
now reports the visible frame without changing the text, ID or internal scroll.
A synthetic regression checks short/long field geometry and retained content.
The first version of that test failed because its test Context texture delta was
not cleared; the harness was corrected and is now green. This is not evidence
that the older Story/Enhance mouse-scroll report has been fully isolated.

The initial600x760 run (navigation03) then exposed a real product layout defect:
wrapping toolbar controls displaced the Section row outside the available track
viewport and clipped Zoom's label. The toolbar now allocates Zoom as one bounded
control and gives narrow windows a taller Timeline. The empty preview message
wraps within the center panel instead of drawing outside it. navigation04/05 pass
the92-action flow at600x760/1100x760. Final navigation06/07 add actual horizontal
wheel input and both range-handle drags:110actions pass in each size, with14
navigation assertions including retained7-12s after pointer movement, full-range
handle restoration and new-right split selection. Normal scrolling is native,
not replaced with direct state mutation in QA. Both panels can still collapse.

Actual own screenshots from navigation04/06/07 were visually inspected for long
prompt clipping, toolbar/Zoom, accessible Scene/Section rows, range, wrapped empty
preview, collapsed panels and hidden labels. This closes that bounded layout
slice, not every dialog or final whole-UI visual acceptance. Audio lanes remain
reachable through native vertical scrolling; waveform is not implemented yet.

Latest immutable code-only package is `checks-c84ccdbd`, protocol11 unchanged.
All14 check stages pass with259tests (147UI plus112 others), fmt and Clippy for
normal/QA, and both builds. The preceding `checks-844bf1c2` covered navigation04/05;
the final package additionally exposes horizontal wheel input in QA. No server,
worker, model, GPU or export algorithm changed. CP16 runtime/Final Retry remains
in the candidate and its regression coverage; CP17 does not claim a new full
Final or GPU/Python parity run.

Code proof remains incremental: source-incremental-17.json and the17 incremental /
cumulative patches reference the retained16/14 chain. No full source copy or
additional cleanup is needed. All own UI/server processes exit and are reaped.
No production selector, original working source, services, user data, storage
policy, forbidden logs, GPU reservation, commit or push is touched.

### All15 groups: functional slices and concrete remaining closure

No whole group is finally accepted. Core4/5/10 remain functionally complete for
their documented slices; all shared final visual/runtime acceptance is open.

| Group | Functional slice verified so far | Remaining implementation / acceptance |
|---|---|---|
| 1 | Contextual media/library/project picker | Wrong-type/missing media/library, real OS cancel, late replies across all targets, common visual/privacy |
| 2 | Cast/Undo/roles/rich transport | Default-image/crop/guide cards and every H3 role's conditioning proof |
| 3 | Completion/catalog/snippets/Composer/Enhance; corrected multiline response geometry | Catalog reload/error, all prompt contexts/BUG002 and final popup/focus/privacy |
| 4 | Story/provider core fromCP13 | Older-format/rollback and final long/empty/filled/visual/privacy/keyboard |
| 5 | Rich library editor/card core fromCP13 | Editor-local save error; missing-media/storage error; older rich formats and icons/keyboard |
| 6 | Paths/shifts/types/canvas/FPS | Reference pixel/large-reference Python parity, actual OS chooser and mixed-FPS montage |
| 7 | Seed/Run/SIGMAS/upscaler/actual LoRA patching | LoRA catalog/filter/info/short names/formats/errors, other samplers, Run network/dependencies, Stage2 Python/high-res Motion/Exp prefix/legacy limits |
| 8 | Selection/Delete/Undo/split/right selection/Fit/collapse plus native range/display, pan retention and600/1100px slices | Group duplicate/move, remaining Scene/audio context commands, conditional FitLast/FitAll, waveform, mixed binding label order and broader keyboard/popup/final visual |
| 9 | LastFrame/normalMotion/reference-video/audio evidence retained | Combo/Influence/transitions, Blend default3, missing Exp prefix/unaccepted Exp; specifically denied diagnostic remains prohibited |
| 10 | Optional exact reviewed Generate core | Network/reconnect/late compilation and shared final acceptance; direct Generate remains free |
| 11 | Status/progress | Sampling-preview transport/runtime/UI/cadence/bounded memory/Hide |
| 12 | Synchronized A/B core | Default pair/swap/per-side decisions, Attach/Captures, cleanup/provenance/original protection |
| 13 | AllFreshSeeds to named Final plus explicit retained Retry/cancel/save recovery, CPU/native tested | Generated transitions, larger/mixed-FPS sources, named delivery older readers, final GPU/runtime/visual acceptance; exact historic intermittent errors not retrospectively proven |
| 14 | Existing session cache | Actual RAM budget/accounting/eviction/Clear UI, without general memory redesign or disk cache |
| 15 | Existing media import/inspection | CheckMedia/Relink stable IDs/affected bindings and interval-bounded larger-source preparation |

K1/K2 and own-fixture BUG001/003 need their explicit closure; F05 remains green.
Shared protocol11 start/rollback/image-job regressions and all affected UI panels
at normal/smaller sizes with long/empty/filled values, icons/tooltips/keyboard and
Hide still precede publication. Full Python functional/algorithmic parity,
audited memory gaps and Exp/P08 remain separate unaccepted requirements.

Next: finish the remaining group8 command and waveform flow. Director Sections
are globally placed while native Sections currently belong to Scenes; group
duplicate/move and conditional overflow Fit must explicitly map that difference
rather than silently restricting operations to the current Scene. No permission
blocker is known for ordinary implementation; no denied diagnostic is needed for
this CPU/UI work. Continue the finite whole15 ledger, then final acceptance and
UI publication, then the separate storage stage.

## Checkpoint 18: safe parallel handoff during group8 implementation

The last fully verified package/functional acceptance is still checkpoint17,
`checks-c84ccdbd` (259tests/14stages, native110actions at600x760/1100x760).
The current source contains additional group8 work in progress, not newly
accepted functionality: descriptor-based CPU waveform API and RAM-only native
waveform projection, global Section operation helpers and optional explicit
Timeline duration metadata. Two source checks (server+UI with ui-preview) pass;
the second includes all current edits but reports10 dead-code warnings because
the group operation helpers are not connected to controls yet. No new full
Clippy/build/native/GPU/Python acceptance is claimed. No live fixture process or
GPU reservation was started. No additional cleanup or publication occurred.

The waveform implementation streams two decode passes with fixed PCM buffering
and16-512peak bins instead of retaining the full audio signal. It projects
packed/planar and signed/unsigned sample semantics, source interval and volume,
and conceals the visual under Hide. It still needs actual Director/PyAV numeric
comparison, errors/retry presentation, source-trim/volume component regression
and real UI inspection. Its decode semaphore2 is a concurrent CPU work bound,
not a media/model allowlist. This is implementation awaiting evidence, not parity.

Global Section helpers express Director gap fallback, grouped movement bounds,
new default owner for duplicated Sections, and owner-envelope rebasing. They
still need control/selection/history integration and oracle/native testing.
Explicit Timeline duration is stored separately from the content-derived legacy
extent, is format-validated and omitted in the separate older-format copy.
Its UI, compatibility disclosure/tests and downstream Final duration semantics
still need implementation before it can be accepted or published.

User requested parallel work. The integration owner retains group8, preview /
A-B and shared integration files; it will not begin group14/15 edits before the
split is established. Side workers must use the immutable checkpoint17 manifest,
not the changing current source. Excluding the separate inference tree, the
manifest contains354files/4828863logicalbytes sufficient for this Cargo server/UI
workspace. The inference tree is a separate workspace and is not a dependency in
the root Cargo.toml. A worker can restore those354files by copying each manifest
content_path after verifying its SHA; no large fixture or full source snapshot is
needed. Root Git HEAD alone is insufficient because the baseline includes the
user's earlier dirty changes. The cumulative patch is supplemental evidence,
not a patch to apply blindly to HEAD.

See UI-GROUPS-PARALLEL-HANDOFF-18.md for exact ownership and integration seams.
Proposed implementation worker owns group14/15-specific new modules and focused
tests in its own small source/target/package/loopback/state/media directories.
The cache must reach actual reference RAM ownership/accounting/eviction; merely
labelling a thumbnail cache as a VAE/reference encoding cache is not acceptance.
It must identify that runtime seam first, without memory-optimizer integration,
changing Qwen's frozen inference profile or adding persistent encoding disk cache.
The read-only worker checks the approved K1/K2 and BUG001/002/003 remnants against
the same frozen source and reports proposed minimal changes without source edits.

The15-group functional closure ledger remains unchanged from17. No whole group
became fully accepted in this partial stage. Integration of group8, waveform and
conditional Fit continues next, alongside the separately owned group14/15 work.
Production, user data, denied diagnostics/logs, publication order and later
storage stage remain unchanged.

## Checkpoint 19: integrated side deliveries and native group8 acceptance

K1/K2 uses only worker B's verified-v2 patch (14 files). Its old-file hashes
matched before integration. The truncated earlier patch was not used. Historical
normal-path executable/model/revision locks and the arbitrary image step/CFG
ceilings are removed; actual numeric/format/access requirements and explicitly
selected frozen comparison controls remain. Full inference/GPU acceptance of
this combined candidate is still pending.

Worker A's worker-a-review-01 delivery is now integrated (13 module/test edits,
7 shared hook files). Module old/new hashes were verified before writes. Shared
hooks were merged with exact unique context; Timeline mod/workspace hooks were
merged manually to retain the newer group8/waveform/duration changes. The first
two context preflights stopped before writes when newer Timeline context did not
match. The integrated test wrapper now supplies its own mandatory encrypted A
fixture/evidence directories. Two equivalent test-only Clippy expressions were
updated for the installed toolchain. G15 health/relink, atomic resource/document
commit, revision-bound thumbnails and interval-only reference output are wired.
G14 RAM cache core/API/UI are wired but unavailable until the real inference
encoding owner/control transport is attached. No substitute thumbnail/test cache
is being represented as VAE caching.

The combined package checks-613246c9 passes17 stages,297 CPU/HTTP/UI-unit test
cases, server-library Clippy, normal and QA UI all-target Clippy, and both binary
builds. The narrower checks-482bdfa3 package had277 passing cases before A.
F05 persistence and runtime/Final-Retry regression coverage remain enabled.
No full inference build or new GPU result is claimed by these checks.

Group8 now connects global Section/mixed-audio move, gap-aware duplicate,
selection-after-success, mixed delete/undo and conditional Fit Last/Fit All.
An executed Director source oracle covers9 numeric cases. The new explicit
Timeline length is edited in an Apply/Cancel dialog bound to its project rather
than an inline menu field which the actual native test proved was not committing.
Native testing also caught and fixed sync_draft clearing secondary Sections when
audio was primary. Global Add Section/resize/ripple/remaining Scene commands,
mixed typed-binding ordering and downstream Final treatment of explicit length
and gaps remain open; group8/group13 are not finally accepted.

Own native groupflow06/07 passed91 actions at1100x760/600x760. Navigation08/09
passed110 actions at both sizes, including range/display/scroll retention,
RightSplit selection, delete focus, Undo/Redo and reopen. Actual waveform
rendering reflects50% volume (raw max0.8 -> drawn max0.4) and is concealed by
Hide. Own screenshots of those frames were inspected. The immediate duration
dialog screenshot caught the opening fade; the harness now waits250ms before
that screenshot. The combined-package groupflow08/09 runs pass92 actions at
both sizes, and the stable normal/small duration-dialog frames were inspected.
Final all-surface visual acceptance is still required.

Waveform03 compares7 synthetic formats at3 peak counts against the actual
source-extracted Director/PyAV functions: all21 peak arrays, rates, channels
and durations match exactly (MAE/max0). Installed FFmpeg9 and PyAV/libavformat62
differ in synthesized MOV/AAC discard padding: preserve the decoded AAC final
frame like the reference, while honoring real MP3/Opus padding. This is a
measured codec-version compatibility boundary, not general codec/Python H3
pipeline parity. Existing failed own fixtures/reports are retained.

Whole15 closure ledger from17 remains controlling. Groups4/5 core flows remain
functionally ready with save-error/older-format/visual acceptance outstanding.
G14 actual owner and G15 native picker/large streamed ingress acceptance remain.
BUG001 session reopen, BUG002 normalized reference-name completion/binding and
BUG003 specific runtime error classification remain next implementation work;
their historical runtime root causes are not inferred from code alone. Preview,
A/B/Captures, continuity/Exp/samplers, Final gaps/transitions, full Python and
algorithmic memory parity remain explicit open requirements. No denied log or
Exp diagnostic was retried, no storage migration, cleanup, GPU reservation or
production publication occurred. Continue remaining implementation before the
shared final visual/regression acceptance and reversible publication; storage
is still the later separate stage.

## Checkpoint 20: actual H3 VAE cache owner and native Relink

G14 now reaches the real reference-image/video/audio VAE encode calls. CPU tensor
bytes belong to a server-session RAM LRU owner, shared with the budget/clear API
through Backend/Scheduler. One-job worker lifetime and model unloading are
unchanged. Protocol12 adds correlated private cache lookup/store/fetch exchanges
over the existing stdin/stdout channel, chunked by its existing4MiB frame bound.
The default retained payload budget is256MiB, editable in UI; disabled/oversized
entries still encode normally. Cache ownership counts retained encoding bytes,
not active readers, model weights or thumbnail memory. Clear/eviction does not
invalidate an active reader or cancel an encode; old-generation completions do
not repopulate. Worker exit aborts pending tickets. There is no persistent cache.

Keys cover preprocessed tensor bytes, shape/dtype, encode policy and selected
weights/library identity (canonical model path plus device/inode/length and
mtime/ctime precision). They do not hash multi-GB model files or introduce model
allowlists/startup locks. This assumes ordinary filesystem metadata reflects
model replacement, and is not a new frozen-model provenance check. Actual VAE
component loading still occurs on cache hits; no performance improvement or
general model-owner/memory-parity fix is claimed.

The supervised real pipe CPU regression creates separate workers, stores an
encoding, proves a later hit, then proves Clear and Off. Three owner tests cover
multi-frame transfer, active readers after clear, clear during encode, aborted
workers, LRU/budget/off and invalid offsets. An actual tch CPU tensor test proves
byte-identical FP16/BF16/FP32/FP64 reconstruction and that encode is not called on
hit; changed preprocessing policy misses. The15 existing CPU wire/admission/
lifecycle tests pass on protocol12. Worker build and Clippy pass; worker package
worker-697e019b has SHA256
460ff5630d995200b75d7fcd9c7e11fc2284687f53971d8d0d7e7db1ac1b4acd.
No GPU reservation/run was made. A real H3 cache-on/hit/off comparison is still
required after GPU-slot coordination; CPU transport evidence is not H3/Python
pipeline parity. Publication must package matching protocol12 server/worker and
runtime metadata, never silently reuse the older protocol11 worker.

G15 native health07/08 each pass53 actual renderer/input actions at1100x760 and
600x760: Check Media, existing project replacement choice, impact preview, Back
without mutation, Apply retaining asset ID, preserved bindings/creative data,
Save/reopen retaining revision1->2, Hide and cache budget/Clear/Off. Real file
picker import/cancel, broad media/large ingress and later storage migration are
not included. Cache UI runs use a synthetic CPU backend exposing the real empty
owner/control API; these native cases do not encode real VAE tensors. Separate
tensor/pipe tests supply that component evidence. Normal/small preview/privacy/
cache frames were inspected; final cross-UI icon/tooltip/keyboard consistency is
still open.

Native inspection caught a real health error:50ms of44.1kHz stereo f32 is17640
bytes but validation reused the12288-byte RGB sample limit. The audio-specific
sample budget now matches the requested PCM. A focused valid stereo-float WAV
test passes, and health05/06/07/08 report all own audio/image entries ok. This is
not a timeout increase or an audio-quality change. Earlier harness failures were
an undeclared fixture-only flag and choosing an old audio ID instead of the newly
attached image; their evidence remains, without changes to product selection.

Combined package checks-c8023fd4 passes302 registered cases/17stages, normal/QA
builds and scoped Clippy, including existing F05, Final-Retry and image regressions.
The19 whole15 ledger remains open with these concrete updates: G14 implementation
is connected/CPU+native-control verified, awaiting GPU/component parity and final
visual consistency; G15's existing-project Relink flow is native verified,
awaiting OS file-picker/cancel, broader formats, >2GiB ingress and final acceptance.
No other whole group is newly finally accepted. Timeline/Final semantics,
BUG001/002/003, full Python/algorithmic memory parity and prior Exp/P08 gaps remain.

Next: fix the known session-reopen/completion/error-classification code paths
with own CPU/native fixtures while coordinating a GPU window for the actual
cache-on/hit/off H3 comparison; then remaining Timeline/Final and whole15 closure.
No new permission blocker exists for ordinary CPU/UI work. Denied diagnostics
were not retried. User services, original source and active product remain
unchanged; no storage migration, cleanup, commit or push occurred. Evidence stays
incremental, not full source/media snapshots.

## Checkpoint 21: own native reproduction of project loss and completion failures

BUG001's two SessionChanged paths (Runs and Document) called document=None.
The shared session_ended transition now retains the project snapshot, folder,
selection and raw text while invalidating server handles/old asynchronous replies.
Old active jobs/runs become Interrupted; no job is resumed automatically. The
project remains visible with a Reconnect/Reopen action. Reopening the same saved
project binds fresh runtime resources and keeps the raw text drafts; history from
the previous server is reset on successful reattachment. A failed reopen retains
the snapshot/drafts. Opening a different project still uses normal replacement.
No attempt is made to recreate lost server-only media or claim recovery of every
unpersisted server-side mutation; a session-only project may require opening a
saved project. This addresses the reproduced displayed-project reset, not an
unobserved deletion of any user's file.

The unit regression feeds both real UI event handlers, then failed/successful
reattachment. Native recovery03/04 each pass33 actions at1100x760/600x760. The
harness stops only its own synthetic server, starts another on that same own
loopback port, waits for the real session-change response, observes the retained
project/text, clicks Reopen and saves the retained text to the original own file.
All three own processes are reaped. Normal/small retained-project banners and
draft fields were visually inspected.

BUG002: completion did not search VisualReference.name, and the UI's normalized
token spelling differed from server reference selection/manifest aliases. The
same spelling function now lives in the shared creative contract, without changing
stored names/aliases or adding alias restrictions. Completion searches names,
aliases, descriptions and kinds and displays the name alongside the token. Both
normal and I2V binding selection use that spelling. The existing image+voice
freeze/worker-transport tests now use the nontrivial alias "Keeper A!" with its
@keeper_a token, retaining their actual RGB/PCM assertions. Native namealias01/02
each pass25 actions at both sizes: search by Harbor name, insert @keeper_a,
compile to <Subject 1>, Save/reopen. Existing simple alias fallback remains.
Stored aliases remain unmodified. Screenshots show the wrapped popup inside the
small viewport. This is compile/transport acceptance, not new trained GPU output.

BUG003: H3 component_load_failed/out_of_memory/generation_failed categories had
no specific UI translation and fell into the generic server-error message. They
now retain their category/code and distinguish a load failure, exhausted memory,
and an unspecified H3 inference failure. The shared job-row message projection
is tested and used by actual native rows. Recovery03/04 deliberately fail their
own CPU worker with h3_generation_failed and display the specific message in an
expanded queue at both sizes. The unit test also covers load/OOM spellings.
The historical user's failing GPU stage/root cause is still unknown and is not
claimed fixed by an injected-error test; it was not investigated through private
or denied logs.

Combined checks-17edd4b8 passes304 registered cases/17stages, normal/QA builds and
scoped Clippy, with F05, Final-Retry, image, cache, interval and Relink regressions
retained. No H3 numerical algorithm changed this phase. Worker-697e019b from20
remains the protocol12 cache worker awaiting a coordinated trained GPU comparison.
The standard sandbox NVML query failed to contact the driver (exit9); a normal
read-only elevated aggregate query succeeded:4%GPU,3213/16376MiB at observation.
This is not an exclusive-slot grant or evidence of who uses memory. No process
list, job history, runtime logs, system changes or GPU reservation/run was used.

The whole15 ledger remains with specific closures: G3's name/token completion
and binding mismatch are now own native/transport verified, but catalog-wide
reload/errors and every prompt context still need shared final acceptance. The
reproduced BUG001 reset and BUG003 category projection are verified as above;
historical inference root cause and trained GPU/cache acceptance remain open.
G8 global Add Section/resize/ripple/remaining Scene commands, G13 explicit output
length/gaps/transitions, G14 GPU equivalence, G15 OS picker/large ingress, other
preview/A-B/Captures/sampling/continuity requirements and common final visual /
icon/tooltip/keyboard acceptance are not silently closed. No publication/storage
migration, cleanup, commit/push or production-service restart occurred.

Next: finish the remaining Timeline/Final semantics using the source oracle,
prepare the bounded cache miss/hit/off raw-output comparison and obtain the
ordinary exclusive GPU ticket when the coordinated slot is available. Continue
the whole15 closure/visual acceptance before publication, then the separate
approved storage stage. No additional ordinary CPU/UI permission is required.

## Checkpoint 22: real GPU Generate/cache equivalence and global Section editing

Candidate `checks-f13095e3` passes305 cases/17 stages, normal and QA builds,
server/UI Clippy including ui-preview, and formatting. The first check identified
an obsolete unused Scene-local trim implementation; it was removed and its
multi-Section regression now exercises the actual global implementation. F05,
Final Retry, image, K1/K2, recovery and reference/Relink tests remain enabled.
Native worker is protocol12 `worker-5ca59d38`, SHA256
`d687aa7faafa951f52b44384507c9bad04ec5acb47a1be121a7461ec7f9c1f4e`.

The actual H3 RAM reference-cache owner now retains the dense tensor stride
permutation as well as shape/dtype/bytes. A cache hit reconstructs that layout
before transfer, preserving the operation's input layout. Non-dense outputs are
not retained; generation still runs normally. The focused native CPU test covers
F16/BF16/F32/F64, transposed layout, exact bytes and skipping encode on a hit.
Lifetime hit/miss counters expose actual lookup behavior, without new admission
gates. Existing budget/off/clear generations still invalidate pending stores.

Two real GPU acceptance cases used the standard worker and shared coordinator,
ordinary native UI Generate,20 res_multistep/simple steps, seed42,416x224,
39frames/24fps, CFG1/shifts12/3, no adapters. `cachesmall02` uses an own synthetic
416x224 image plus own stereo tone; `cachevideo01` uses an own160x96 reference
video without source audio. Each executes miss, hit, then disabled cache, with
one fresh worker per job. The test server delegates to unchanged PipeBackend and
scheduler and captures only its own returned RGB8/float32 PCM before export.
It is not a latent observer and introduces no new runtime environment parameter.

All six jobs pass. Image+audio:2misses/2entries, then2hits/no miss, then disabled
2misses/0entries/0bytes; retained size38784B. Video:1miss/1entry, then1hit, then
disabled1miss/0entries/0bytes; retained size40320B. Within each case raw RGB8 and
PCM, and decoded H264/AAC outputs are byte-identical between all three variants:
MAE/RMSE/max0, correlation1 for nonconstant inputs. Each MP4 has39frames,1.625s,
H264/yuv420p/bt709 limited-range and32kHz stereo AAC. Raw PCM RMS/peak are
0.1599620693/0.5316876769(image+audio) and0.01114850692/0.06118071079(video).
These are Rust cache equivalence results, not Python pipeline parity or visual
quality acceptance. Timings include UI submission/polling and result inspection;
they are not a cache-speed benchmark. Model components still load on a cache hit.

Real Generate also exposed a remaining reproducible failure: `cachegpu02`
(large own portrait+video+tone) and `cacheia01`(same portrait+tone) fail with
h3_out_of_memory in textencoding before observed sampling. The latter records
cache hits/misses/entries/used bytes all zero. Thus this failure precedes cache
lookup. Rust's Text budget0 is not itself a demonstrated estimator mismatch:
ComfyUI CLIP calls an optional estimator, but actual MiniMaxH3TEModel and its
SD1ClipModel base define none, so that path also supplies0. No speculative
headroom, quality reduction or memory optimization was installed. A smaller
explicit fixture succeeds; it does not replace the failed larger acceptance.
The historical user's Generate failure remains unproved. BUG003 now has both
specific error UI and a real successful worker/sampling/export path, but the
larger-reference OOM remains an unresolved correctness/acceptance issue.

An earlier small Generate succeeded before its harness confused public job ID
with the worker request ID for raw-file lookup. Only that test mapping was fixed;
the successful media and failed harness report remain preserved. Terminal jobs
are no longer incorrectly cancelled by harness cleanup after an inspection error.

Group8 Add Section now searches the global Director Section gaps with its1s
default, attaches to the selected compatible Scene, extends/rebases that owner
from its actual Section envelope, and selects the new Section. With no compatible
owner it creates a standalone inherited Scene; an Imported owner is not converted.
An owned insertion does not fall back to an earlier gap. The global resize path
now trims the neighboring Section even across Scene ownership, and Ripple shifts
all following Sections up to the project editing limit. Ownership, metadata,
bindings and Undo remain on the normal transactional edit path. The old Scene-local
trim path is no longer the production gesture implementation.

The source-only Director VM oracle now has19 cases, including10 added Add/trim/
ripple cases. A separate owner-envelope regression preserves metadata while
rebasing Section-local times. Native `globaledit01/02` each pass41 actions at
1100x760/600x760: Add/select/Undo, cross-owner end trim, global Ripple/clamp,
Undo/Save/reopen. The normal/small rendered ripple screenshots were inspected;
controls wrap without overlap, the timeline clips/pans within its viewport and
the sidebar retains normal scrolling. This is a focused visual slice, not whole
UI final visual acceptance.

Final timing clarification: the actual Director hard-cut assembly body was run
source-only with own tiny CPU tensors. Scenes at0..1 and5..7 produce72frames/3s
at24fps for project editing lengths3,12 and100s. Director concatenates Scene
content and does not insert black frames for timeline gaps or pad to editing
length. Rust's existing imported assembly test likewise verifies concatenation
without a timeline gap. Therefore the earlier assumption that Final must pad to
an explicit editing length is not a demonstrated Director requirement. No
speculative black-padding change was made. Generated transitions and broader
Final runtime/native acceptance remain open.

### Whole15 closure after22 (functional slices, not final approval)

| Group | New closure / concrete remaining requirements |
| --- | --- |
| 1 | Core picker/attach available; wrong type/missing/OS-cancel/late and library picker acceptance remain |
| 2 | Image roles/crop/guidance wired; all-role actual conditioning and default-image acceptance remain |
| 3 | Name/token completion and binding mismatch verified21; catalog reload/error/all prompt contexts and final acceptance remain |
| 4 | Story/provider core complete; editor save-error, older-format/rollback and common visual acceptance remain |
| 5 | Rich library/character core complete; save/missing-media/storage error, older-format and common visual acceptance remain |
| 6 | Geometry/scenetype/component paths/shifts wired; large Reference/Python, chooser and mixed-FPS acceptance remain |
| 7 | LoRA patching/Stage2/upscaling slices verified; catalog/filter/info/error formats, runtime samplers and full Python handoff/highres acceptance remain |
| 8 | Global Add, trim/ripple, groupmove/duplicate, Fit, range/display/waveform now tested; remaining Scene/context commands, mixed binding ordering and full keyboard/visual acceptance remain |
| 9 | Continuity slices wired; Combo/Influence/transition behavior, Blend default3 and Exp prefix/full runtime gaps remain; denied diagnostic untouched |
| 10 | Optional exact reviewed plan Generate core complete; late compile/network/reconnect/final acceptance remain |
| 11 | Sampling preview transport/runtime/cadence/bounded RAM/Hide acceptance remains |
| 12 | A/B default pair/swap/per-side/Attach/Captures/cleanup/provenance closure remains |
| 13 | Ready/unfinished Final, filename/retry/cancel/save recovery verified earlier; source timing clarified22; real transitions, larger/mixed-FPS, older named readers and final runtime/UI acceptance remain |
| 14 | Actual RAM cache, budget/off/clear and three VAE input paths functionally verified on small CPU/GPU/native fixtures; large-reference OOM and common final keyboard/icon/visual acceptance remain |
| 15 | Native project Relink/health available; OS replacement picker/cancel, broader formats and >2GiB ingress remain |

All own workers, isolated servers and native UI processes from these cases have
exited. Terminal engine reports show no worker PID/job; graceful own server exit0
awaited scheduler shutdown and releases its ordinary coordinator tickets. No
user processes were read, interrupted or restarted. No saved runtime console
logs or denied operations were queried. No production publication, cleanup,
storage migration, commit/push or system change occurred. Full Python functional/
algorithmic parity and audited model-memory lifetime/budget/dynamic-offload gaps
remain separate open requirements, not accepted by byte-identical Rust cache
results.

Next: complete the finite remaining Scene/context and preview/A-B flows, then
continuity/runtime/Final gaps and the explicit15-group error/compatibility
acceptance. Isolate the large-reference text OOM with a bounded source/component
comparison rather than blind full-job retries or arbitrary budget changes.
Perform whole-UI normal/small visual/icon/tooltip/keyboard acceptance before
reversible complete publication; only then start the approved storage stage.

## Checkpoint 23: large-reference text OOM isolated against actual Comfy dynamic path

No product numerical, memory-policy or UI behavior change in this checkpoint.
Only a separate Rust text-component example and an opt-in own-fixture harness
were added. The example reuses H3's actual static component preparation, Manager
placement with Text budget0 and unchanged text/vision forwards. It records shape,
dtype, stride and this process's CUDA allocation metadata; it does not dump GPU
latents or add a worker observer/environment contract. Scoped example fmt,
Clippy -D warnings and build pass. CP22's305-case product suite/package remains
the last product build; no full generation was rerun.

The exact own portrait, Standard geometry, becomes content1315x736 padded29px
on the right to1344x736. Both workers read identical CPU input safetensors:
SHA256 `cd039a99979eee8e770c8aff0063c559ab0bf2ad961bb7f7c57910b50800ba67`.
This isolates text placement/forward on the same pixels; it is not a new claim
that every Director resize pixel matches Rust. One fixed short neutral prompt,
same selected Qwen3VL32B INT8ConvRot checkpoint, BF16 constructor, FP32 vision
and text compute, PyTorch attention, no Triton/compiler, synchronous offload,
and no full DiT/VAE/export are used.

`textoom01`: Rust and static Python both fail on a500MiB allocation after all27
vision blocks. Rust reaches language block0, passes attention and then fails
before that block completes. The vision grid produces3864 patches and the
language input has988 tokens. This narrows the failure to the first language
block's post-attention/MLP path; the allocation size also matches a5120x25600
FP32 projection weight, but that exact failing projection is not separately
traced and is not asserted as proven. Rust's model-loaded PyTorch allocation
is11364250624B. This is before any reference VAE/cache or sampling. The same
failure with the shorter prompt rules out dependence on the prior long prompt
or its audio/video bindings for this component reproduction.

`textoom02`: requesting dynamic without the normal main.py bootstrap leaves
CoreModelPatcher static. actual_dynamic=false and OOM500MiB are recorded. It
is a static control, not dynamic acceptance. `textoom03` initially bootstrapped
AIMDO after host_buffer had already imported its unset library handle; the
result is a test-only hostbuf_allocate AttributeError before model load. The
bootstrap order was corrected to match main.py; no module reload, install,
service start or system setting was used and neither failed case was hidden.

`textoom04`: normal process-local comfy_aimdo.control.init runs before Comfy
host_buffer imports; init_devices then selects ModelPatcherDynamic and enables
aimdo as main.py does. actual_dynamic=true. The same tensor/model/prompt passes
all27 vision and50 language blocks and produces its conditioning. Static02 and
dynamic04 initial driver-free memory differ by only15794176B(about15.1MiB),
with the static path failing and dynamic succeeding. Static01 Rust and Python
had identical initial driver-free memory13214089216B and both failed. This
demonstrates a functional difference between the selected static product path
and the normal dynamic reference path for this input/environment. It does not
prove a universal VRAM threshold or exact internal eviction chronology.

Important measurement distinction: dynamic AIMDO weight allocations are outside
Torch's usual allocator. Dynamic04's maximum observed PyTorch allocated bytes
are142385152B, while driver-free memory drops to282394624B of16769744896B total.
Therefore this is NOT a claim that the32B model consumes136MiB of VRAM. Metadata
hooks are synchronized observation points, not a continuous peak-VRAM benchmark.
Component wall times17.13s Rust-fail/56.79s Python-static-fail/10.51s Python-
dynamic-pass are likewise not matched performance benchmarks.

The verified functional issue is the normal product choosing StaticModel /
StaticMemoryServices, which lacks the dynamic model owner/casting/eviction path.
The installed Rust Aimdo binding currently only exposes model-mmap lifetime,
not ModelPatcherDynamic's virtual-buffer/host-pin/prefetch ownership. The ordinary
Python bootstrap is an essential part of the reference; a direct API import
alone did not enable it. This is consistent with the earlier memory audit, now
with an actual failing/succeeding component reproduction. No arbitrary reserve,
input-size cap, model allowlist, silent CPU/Python fallback or quality reduction
was added. Normal larger-reference Generate remains blocked until the dynamic
Rust path is implemented and verified; the historical user's failure still is
not retrospectively established.

Next code boundary: implement native process-local AIMDO device/virtual-buffer
ownership and the H3 text model's dynamic gather/cast/release lifecycle, preserving
quantization and operations. Use this saved same-input Python conditioning as
the component control, verify Rust conditioning numerically and then rerun the
larger own normal Generate through sampling/export/cache. Preserve static
component controls and the previously passing small full-generation regression.
This is implementation work under the existing parity mandate, not a new
permission request. General lifetime/unload, VAE budget and asynchronous-prefetch
parity still require separate closure; this diagnosis does not accept those gaps.

The first component exception used the upstream CUDA error string, which includes
driver process accounting. That text was not used to inspect any job/process;
subsequent exception projection records only class/allocation failure and omits
that accounting. Original own evidence is preserved. No saved runtime console
log or previously denied operation was read. All own component processes have
exited and their exclusive coordinator sockets are closed. No publication,
storage change, cleanup, commit/push or user-service action occurred.

## Checkpoint 24: dynamic native text owner and larger-reference Generate

The reproduced CP23 static-placement OOM is fixed in the isolated product worker.
Text now uses AIMDO device contexts and a model VBAR, CPU-backed source weights,
operation-scoped fault/signature/gather/cast/unpin, and error-safe ownership.
Quantized payload and tensor parameters follow original field order and 1024-byte
packing, preserving all scalar/layout metadata. Module ranges align to512 bytes;
the model virtual reservation is source model_size*10, not a physical VRAM cap.
Tiny cast modules, noncast parameters and buffers follow source preload rules.
Linear, token/position embedding, patch convolution and vision LayerNorm use
the dynamic lease; existing dtype/kernel math remains in place. Static controls
and all other component paths remain available and unchanged by this fix.

Self-review corrected the initial module allocation order to actual H3 semantics:
SD1ClipModel forwards the inner Llama language units; language layers come first
in registration order, then the remaining dynamic small-first/descending-size
order. This is distinct from the static offload sort. Conditioning-prefill does
not use layer-ahead prefetch because past_key_values is absent. A focused CPU
ordering regression covers that distinction. Four dynamic CPU tests pass; the
existing VAE-cache CPU regression (four native dtypes and dense transposed
strides) also passes. Scoped fmt, full worker Clippy -D warnings and worker build
pass. Last broad server/UI suite remains CP22's305 cases; it was not rerun or
miscounted as new evidence for this inference-only change.

Final component `textdynamic02` consumes the exact CP23 own Standard portrait
tensor [1,736,1344,3], inputSHA cd039a99979eee8e770c8aff0063c559ab0bf2ad961bb7f7c57910b50800ba67.
Conditioning [1,988,5120] and token tags [988] are byte-identical to saved normal
dynamic Python `textoom04`: MAE/RMSE/relativeRMSE/max all0, correlation1 within
floating-point metric rounding, conditioningRMS16.111379390741124. Same chosen
32B INT8ConvRot text checkpoint, BF16 construction, FP32 text/vision forwards,
PyTorch attention and synchronous offload. Component wall29.743s
includes tracing; no performance comparison is asserted. AIMDO weights lie outside
Torch allocation counters; neither those counters nor stage samples are a
continuous VRAM peak or evidence that the model uses only a few hundredMiB.

Final worker `/home/thhel/ChatGPT/imagine/comfyui/ui-groups-20261021/worker-packages/worker-893e4173/imagine-h3-worker`
SHA256 `0d4a99826c64cf89e19cfb890303bdfad13e86ad5e55498151a2eccfcacf2010`, protocol12.
UI/server use unchanged last verified package `/home/thhel/ChatGPT/imagine/comfyui/ui-groups-20261021/packages/checks-f13095e3`.
`cachedynamic02` runs actual own native UI Generate -> shared exclusive scheduler
-> standard worker -> sampling/decode/direct MP4. Large portrait plus synthetic
stereo voice reference,416x224,39frames24fps,20res_multistep/simple,seed42,CFG1,
shifts12/3. Cache miss/hit/off all succeed. Variant wall seconds including UI,
queue handling and output inspection: miss=58.051, hit=77.793, off=78.553.
This is a correctness regression, not a cache-speed benchmark or cold-cache test.
First variant has2misses/2entries; second2hits; off retains0entries/0bytes.
Raw RGB8 and float32 PCM have measured MAE/RMSE/max0 and exact bytes; decoded
MP4 RGB/PCM also have matching bytes. PCM RMS0.163795804701,peak0.886447548866;
39frames,1.625s,H264/yuv420p/bt709 limited,AAC32k stereo. No gain correction,
normalization, size cap, model allowlist, arbitrary reserve or Python fallback.
Earlier `textdynamic01`/`cachedynamic01` remain preserved as the pre-order-fix
variant; final acceptance uses02 and the exact final binary above.

This establishes text-component Python parity and a working larger-reference
normal Generate/cache path. It does not establish a full generated Python/Rust
pipeline comparison, all reference-resize pixels, or the historical user's exact
Generate failure. General model owner lifetime/unload timing, host pin pools,
VAE budgets and broader dynamic-prefetch parity remain explicit open items.
Groups6/14 lose the reproduced large-reference functional blocker; their other
parity/UI acceptance criteria remain. All15 groups' exact remaining work is
recorded in UI-GROUPS-OPEN-24.md, without promoting unfinished features to done.

All own UI/server/component workers exited; all exclusive GPU leases released.
No private user data or runtime-console logs were read; no denied operation was
retried. No production selector/artifact/service, storage policy, historical
fixture, cleanup, commit or push changed. Next: resume finite remaining Timeline,
continuity, preview/A-B/Final requirements and common visual/keyboard acceptance.
Publication still follows complete accepted UI/fixes; storage is a later stage.

## Checkpoint 27 — generated Captures and integrated LoRA catalog

The isolated candidate is `comfyui/ui-groups-20261021/packages/checks-c6b43806`.
All 18 check stages and 365 registered CPU cases pass, including the new
Capture mutation/cleanup, disk-failure and cross-Scene receipt cases and the
integrated worker-B LoRA catalog/API/headless UI tests. No publication or GPU
work occurred. The protocol-12 trained worker remains unchanged from CP24.

G12's generated-Capture flow now supports independent attachment state,
detach/reattach, Accept/Reject/Candidate, cross-Scene reuse without inventing a
new generation recipe, explicit removal, saved-project reopen and preview.
Removal commits the manifest first, retains shared/provenance resources, and
only unlinks released owned project blobs. A failed save cannot remove those
blobs; a separately retained original is unchanged. The native normal/small
flows and real own loopback identity/provenance tests pass in captures08/09.
The A/B failure/explicit Retry regression passes in takereview12.

Worker B's G7 catalog/filter/local Info is integrated, with no catalog or
metadata gate on manual paths or Apply. Native QA found a real Scene LoRA
footer below the viewport; bounded dialogs, scrollable bodies and fixed
Apply/Cancel footers correct it. Local Info, manual paths, project/Scene merge,
Apply/Undo/Redo and CPU Generate pass at 1100x760 and 600x760 in loranative18/17.
These are synthetic CPU backend checks, not trained LoRA or Python inference.
The configured native catalog popup/reload and uncommon ECMAScript regex
dialect semantics remain open; fancy-regex is not claimed identical to JS.

G12 remains incomplete for arbitrary external video with unknown generation
provenance. The coherent next migration is optional generated provenance and
optional accepted model, with actual video-content identity and source asset
revision for imported Takes. No placeholder model/seed/recipe was introduced.
Retained removed receipts still need their final UI presentation after the last
dependent is removed; explicit deletion is already supported by the API.

The verified A G15 CP25 streaming/Relink package is ready, but not integrated
in CP27. Dense first Attach+Save above 2 GiB requires enough encrypted space
for simultaneous copies; its worker explicitly did not claim that acceptance.
All other group requirements, memory/Python parity and specific Exp/log access
limits remain in UI-GROUPS-OPEN-27.md. Native normal/small visual inspection was
performed on own Capture/LoRA/Info images. All own UI/server processes exited;
no GPU lease was acquired, production process changed, private data read or
denied operation retried. Source evidence remains incremental, not full copies.

## CP28 — imported Takes, streaming media and bounded Capture layout

The isolated candidate now attaches external project video as a Take with
unknown generation provenance, not a fabricated seed/model/recipe. Accepted
identity is the actual video content fingerprint. Generated recipe JSON stays
object-shaped. Import identity survives cross-Scene reuse and repeated import;
Save/reopen and explicit cleanup preserve the external original. Pre-rename
save-failure tests cover both import and ordinary Accept/Reject rollback.

Worker A's G15 descriptor/streaming ingress, probes and cancellable replacement
selector are integrated. The fixed 2 GiB AV ingress gate is removed; buffered
image limits and actual size/length/free-space/write errors remain. Native
project-asset Relink/Save/reopen keeps the asset ID and existing bindings. The
actual OS chooser still lacks acceptance: test injection is not counted as an
OS picker test. Dense >2 GiB first Attach+Save is not run with about 3 GiB free;
original plus upload alone needs over 4 GiB, before staging and saved copies.

Actual visual inspection found Capture rows expanding a small modal outside
the viewport. Separate wrapped metadata/action rows fix it; final own tests
importtake03/04 pass at 1100x760 and 600x760. Generated Capture lifecycle and
maintenancecp28b Relink/cache also pass on the same package checks-18dc5e31.
Four native cases cover 184 actions. Supplemental CPU Final export from
unknown-recipe Takes passes: 48 frames, 2.000 s, H264/AAC MP4. No new GPU or
Python-pipeline acceptance is asserted. New Blend Seam defaults now follow
Director's three video frames; persisted 6/11 values survive reopening/editing.

All processes reaped, production and storage migration unchanged. OPEN28
retains exact continuity, sampling-preview, transition Final, regex, native
picker/large-source and common acceptance gaps. Code evidence remains a small
incremental patch chain; no cleanup, full source snapshots or denied reads.

## CP30 — relocated own fixtures, dense first Save and Final warning acceptance

Tomas moved the agent-owned task directories directly under `/home/thhel/tmp`.
The active synthetic root is now
`/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01`.
The source stays at `comfyui/ui-groups-20261021/source`. Ordinary bounded
elevation sees `/home`, `/dev/mapper/cryptroot[/@home]`, btrfs, read-write;
the default sandbox exposes that non-workspace destination read-only. No mount,
security, product-storage or selector setting was changed. All 1,837 prior
incremental content files matched after explicit own-prefix resolution.

Only active own harness/test-fixture guards were rebased. Historical evidence
is immutable. Tests share one fixture module per test crate; no synthetic root
was introduced into production storage policy. The combined UI/server package
`checks-41bff3af` passes 381 CPU/HTTP/UI-unit cases, 20 sequential check stages,
normal/QA builds and strict Clippy. Four tch CPU preview tests also pass again.
The appended dense-file test passes separately; its format and normal/QA
all-target Clippy plus small streaming regressions pass afterwards.

G15's previously capacity-blocked first Attach+Save now passes with a dense
2,147,549,228-byte synthetic PCM16 stereo WAV. All bytes are written, the real
upload/Attach/Save APIs produce the first bundle, a fresh loopback harness
reopens it, and independent streaming hashes preserve source/saved identity,
stable project asset ID and revision. The run took 355.97 seconds in the debug
test build. Peak process high-water RSS was 119,341,056 bytes; both HTTP and
scheduler harnesses close. Per-inode allocated bytes are recorded with no
claim of unique physical CoW extents. No sparse saved-bundle substitute is used.

Final's Director-compatible missing-transition warning and concatenation pass
actual native UI at 600x760 (Render current Timeline) and 1100x760
(Freeze/Render). Both outputs are identical MP4/H264/yuv420p/bt709/AAC stereo,
282 frames at 24fps, 11.750 seconds, with nonzero decoded PCM. QA was corrected
to scroll the inspector and keep the selected export command in view; the
earlier failed attempts did not submit jobs and all own processes closed.
The warning was inspected in actual screenshots. This is CPU export evidence,
not generated transition conditioning or new H3/Python pipeline parity.

Retained Capture receipts and the fast-preview component were already added in
CP29. G11 still needs sampler/transport/RAM/API/UI wiring and codec/VAE-quality
acceptance. Incoming transition/Combo/Influence conditioning and provenance,
Exp transport/runtime (specific denied diagnostics remain denied), uncommon
ECMAScript regex semantics, native OS picker, broader media/parity and common
visual acceptance remain in OPEN30. Final completion-label/snapshot messaging
also needs inspection: the successful export screenshots still show Preparing
and an earlier-snapshot hint. Nothing is published; storage changes wait.

A separate source/metadata-only inventory explains about 304.17 GiB summed
allocated blocks under `comfyui`, mostly frozen test binaries/build targets
and old candidate packages. Active product and runtime dependencies still use
that directory. No inventory target was moved or deleted; reflink sharing
means that sum is not a measurement of uniquely reclaimable physical bytes.

## CP31 — transient H3 sampling-preview transport and actual native UI

G11 now has a timed packed-AV denoised callback, protocol 13 preview events,
one latest RGB animation in server RAM, a session-owned binary API and a
silent native UI player. Settings preserve the requested update interval;
Director's preview size/frame/pixel bounds apply only to effective rendering.
Preview failure is a nonfatal warning. Preview payloads do not advance final
video/audio accounting, become project assets or change the seed/recipe.
No latent tensor or model is retained by the callback between sampling steps.

The 21-stage combined check passes 398 registered test results, normal/QA
builds and strict Clippy. Five tch CPU preview/callback tests and wire checks
pass; the protocol-13 native worker is built separately. This is not a new
GPU or full Python-pipeline parity claim. Fast latent RGB/resize/cadence was
already compared to the source-executed Director/Pillow oracle in CP29/30.

Actual native UI tests pass at 1100x760 and 600x760 using a synthetic CPU
backend. Own screenshots were visually inspected for the settings grid,
tooltip, animation, Hide/reveal, focus loss and cancellation. One additional
normal-size case verifies a second Generate, navigation to another Scene,
cancellation there and returning without stale preview. All 161 actions pass
and own UI/server children are reaped. Animation changes frame 0 to frame 1;
Hide/focus loss drops the texture, terminal/navigation clears the clip.
Earlier failed QA attempts targeted an inactive Generate button for hover or
a Run cancel control/the collapsed queue; correcting test controls fixed them
without adding product restrictions.

The actual GPU preview-off/on equality harness is prepared but not run: the
last authorized aggregate GPU check showed 8% utilization and 3121 MiB used.
No GPU lease or job was started. Raw RGB transport is currently used instead
of Director's WebP delivery. VAE-quality and image-workspace preview production,
actual GPU output equality and full Python parity remain explicitly open.

Incoming transition source tracing confirms a normal target Take, not a third
bridge clip. Boundary intent still lacks authored prompt/tail/reuse/audio
fields; generated TakeVideo resources need a direct frozen-input path, followed
by exact tail/end-image/paired-audio projection and the existing runtime opening
window. A worker-applied source/dependency receipt is needed before Final can
remove its missing-transition warning. No transition implementation or Exp
diagnostic was silently substituted in this checkpoint.

The read-only cleanup proposal selects only three exact old Cargo incremental
caches, about 12.95 GiB exclusive Btrfs extents. It preserves executable files,
frozen controls, active packages, source, evidence and rollback. No cleanup,
publication, storage migration or user-process restart was performed.

## CP32 — incoming boundary conditioning in the target Take

Transition/Combo/Influence now project the incoming source into the next
Scene's ordinary Take: frozen accepted TakeVideo or imported clip, selected
video tail, optional paired audio and opening image. Boundary prompt, tail,
reuse and audio settings are authored in the UI. The resolver and dependency
projection follow the traced Director semantics; no third bridge Take is
created. Old untracked Takes keep their accepted state and Final warning.

Protocol 14 carries a worker-applied boundary receipt. Native Reference
conditioning uses the existing opening attention window for the first
boundary video and its paired audio, retaining other references. The native
worker requires observing the matching attention layout before acknowledging
application. This path is built and CPU-tested, not yet GPU-accepted.

The 22-stage combined regression passes 417 registered test results, normal
and QA builds and strict Clippy. Three real CPU attention-window tests pass.
The boundary decoder has eight focused CPU tests. An actual Torch/Rust index
oracle compares 4,287,842 indices across 38,804 pairs without differences;
six source-executed Director cases cover trim, tail and paired-audio rules.
These component comparisons do not establish full Python-pipeline parity.

Actual native UI tests pass at 1100x760 and 600x760 with synthetic CPU sampling
and real freeze/decode/queue/export. A request for 23 tail frames produces 22
effective frames and 40,426 paired samples under the independently rounded
endpoint rule. The resulting Take retains the receipt with no extra clip.
Own screenshots were inspected; the narrow Take Compare action now wraps
without horizontal clipping and its tooltip remains visible. Own test children
are reaped. No GPU lease was taken: the last aggregate check was 13% and
3257 MiB used. No cleanup or publication occurred.

OPEN32 retains actual GPU/Python parity, explicit nonadjacent-source boundary
selection, explicit continuity readiness text, endpoint-image preflight,
older-reader compatibility, Final completion presentation and the other
previous group criteria. Exp diagnostic denials and memory parity gaps remain
separate and unchanged.

## CP33 — physical boundary identity, compatibility and Final presentation

Director source tracing confirms that the physical previous-to-target seam
chooses boundary settings independently of an explicitly selected media source.
The resolver now retains these two identities separately; a nonadjacent
Combo/Influence source no longer silently selects Hard Cut. A focused pure test
and the real-resource queue test cover both variants and dependency identity.
The UI reports readiness from source video bytes, not legacy endpoint/latent
artifacts. Explicit endpoint overrides now share ordinary image preflight.

An explicit older-format export gives a specific representability error for
new boundary fields or applied conditioning receipts rather than stripping the
recipe. Normal Save retains complete metadata and is unchanged. Final displays
Saved/Saving export correctly. Its snapshot hint compares actual authoring
metadata while excluding only canonical newly published delivery assets and
revision bookkeeping; real concurrent edits still warn.

All 22 combined stages pass: 424 registered test results, normal/QA builds and
strict Clippy. A QA-only supplement observes the existing readiness label.
Actual synthetic-CPU native Combo/Influence flows pass in normal/small windows;
the complete readiness text is visibly reachable in the small inspector. Final
direct Render and separate Freeze/Render pass at 600x760 and 1100x760 with the
same 282-frame, 11.750s MP4/H264/AAC and nonzero audio. Screenshots show Saved
and no false earlier-snapshot warning; the genuine ungenerated-transition
warning remains for old untracked Takes.

The approved real-GPU preview off/on test also passes, using the immutable
CP32 native package and unchanged protocol-14 worker. Each variant generates
39 frames with four sampling steps. Pre-export RGB8/float32 PCM and decoded
MP4 video/audio are byte-identical; on-preview reaches the actual native UI.
Both workers and own server/UI children are reaped through the normal shared
coordinator. This is a Rust preview regression, not Python pipeline parity or
a base-model quality/performance benchmark. Boundary GPU acceptance remains.

After Tomas's exact approval, only the three selected old Cargo incremental
cache directories were removed. Existing Cargo locks were held, every target
inode was checked for active use, symlinks were excluded and sibling metadata
was unchanged. Immediate statvfs available increase was 7,663,374,336 bytes;
the later observed increase was 8,002,519,040 bytes. These measurements can
include concurrent filesystem changes and differ from logical/exclusive extent
estimates. Source, binaries, packages, rollback and test media were preserved.
No product publication, storage migration or user-process restart occurred.

## CP34 — real GPU boundary paths and generated Take to Final

Four ordinary worker cases now pass with the protocol-14 worker: Reference
Combo with new audio, Reference Combo with paired previous audio, Reference
Influence with paired previous audio, and I2V with new audio. Each uses the
same own synthetic trimmed source, 23 requested tail frames, 416x224 and
39 target frames at 24 fps, four res_multistep/simple sampling steps and
seed 42. Reference cases select 22 effective frames; I2V derives only its
opening image. Actual worker acknowledgements are required and persisted;
the Reference cases pass the matching attention-window assertion. All produce
finite nonzero PCM and 1.625s MP4/H264/AAC. No gain/fallback was added.

After accepting the real I2V result, CPU Final freezes two clips, renders
87 frames / 3.625s intended video with stereo AAC44.1k, retains exactly the
existing three target Takes and records Saved. The applied boundary receipt
correctly removes the missing-transition warning. This is real GPU-generated
media through Final, not a synthetic sampler result. It remains a bounded
functional proof, not full Python or twenty-step quality parity.

The harness initially needed sequence-order correction and raw request-UUID
discovery. The first successful Combo output was recovered and measured from
existing files without repeating sampling. Final's first test request supplied
an empty Options object instead of omitting it for contract defaults; only
the test payload was corrected. Own processes are reaped, ordinary shared
GPU reservations released and no user process was stopped.

A separate CPU executable links the unchanged pre-UI baseline project reader.
It opens an explicit compatible two-Scene copy and preserves Blend Seam 3;
new authored tail/prompt metadata saves/reopens in the current reader and
is refused specifically by older-format export without changing the original.
Original legacy reader source/binaries remain hash-verified unchanged. This
does not cover rich library, Story or Take/media compatibility.

Nonadjacent native-UI acceptance is prepared but not run. Automatic approval
review rejected the exact isolated CPU server/UI workflow twice, referring to
the old component-only mandate; on the second review it treated the later
full-UI authorization quoted in justification as untrusted. No process or
fixture started, no alternative execution was used, and no further attempt
is made without direct relevant approval. This new UI-start block is separate
from the unchanged denied Exp diagnostic and log reads.

All new Python harnesses parse, source rustfmt and strict example Clippy pass.
No product numerics changed; the normal server/UI candidate remains CP33.
The three-cache cleanup is complete and was not repeated. The extent estimate
12.95GiB and observed available-space increases 7.14/7.45GiB are different
measurements; existing evidence does not apportion compression, accounting
or concurrent filesystem activity. OPEN34 lists all publication prerequisites.

## CP35 — approved native UI retry, save-error visibility and ECMAScript filtering

The directly approved retry of the nonadjacent boundary UI test passes. Scene3
uses Scene1 media while retaining the physical Scene2→3 boundary. Story/Enhance
and standalone character/library flows pass at 1100×760 and 600×760, including
provider failure, Cancel/Retry, edited review Apply and image/voice save/reopen.
Synthetic pre-mutation save faults retain the exact edited source and media
settings. Visual inspection found and fixed a character-save error below the
viewport: it now appears beside Save/Cancel with library-specific text. Known
write failures are distinguished from unconfirmed network/durability outcomes.
Reload/folder tools use the shared icon buttons and tooltips, and the library
folder field follows available width. Primary Save/Cancel actions retain text.

G7 filtering now uses an owned, optional installed Node helper for the exact
`new RegExp(pattern)` semantics used by Director, without flags or translation.
One request batches the ordered catalog; results are cached across repaints.
Empty filters need no helper. Missing/failed engines keep all choices and manual
paths/Apply usable. Only this editor's helper is cancelled/reaped on closure,
scope change or obsolete in-flight filtering. No installation or new model gate
was added. Actual Director-source oracles cover invalid patterns, UTF-16,
escapes/classes/backreferences and duplicates. Native normal/small catalog
Refresh, selection, Info, Apply/Undo/Redo and synthetic CPU Generate pass.

The combined 22-stage checks pass 430 registered results; final fmt, strict
normal/QA Clippy and builds pass. Candidate packages/cp35-final is not activated.
Initial test-only Clippy and empty-text-event harness failures are retained with
their passing corrections. All own CPU server/UI processes are reaped; no GPU,
runtime-log reads, additional cleanup, publication or storage migration occurred.

OPEN35 retains missing image-workspace preview, permitted Exp transport,
Director preview-quality/delivery and full Python/memory parity, rich older
format compatibility and common whole-UI/native-chooser acceptance. Earlier
specific Exp/log denials remain prohibited. CP35 is not full-plan completion.

## CP36 — shared image/video sampling preview and corrected release triage

Native image jobs now request the existing C++ fast preview and publish only the
latest downscaled RGB frame into the existing RAM hub. Final-image handling is
unchanged. The client gates preview decoding by seconds; the native callback
still renders/encodes each step. This is not a throughput optimization or a GPU
numerical-parity claim. Enable after submission requires the next native job.

One UI owner/poller now serves Timeline and image desks, with explicit workspace
or project/Scene identity. Job ownership, cancellation, navigation and stale
response rejection are retained. Single-image previews show step text rather
than a misleading animation duration. Image settings preserve the video's frame
count. Backends without a preview producer give a nonblocking, readable notice.

Integrated CPU tests, normal/QA Clippy and builds pass after correcting a lint
and module ordering. Actual separate CPU-native UI tests pass at 1100x760 and
600x760 for Generate, Hide/reveal, focus loss, Cancel, repeat, workspace change,
return and completed output. Existing video-preview lifecycle passes again.
The first image UI attempt had an empty synthetic model catalog; the test fixture
was corrected with header-only synthetic catalog files. No product admission
rule was changed to make the test pass. Settings, tooltip and preview layouts
were visually inspected using own test screenshots. No GPU or production start,
publication, storage migration, cleanup or user-process restart occurred.

OPEN36 corrects an overbroad prior label: ordinary Exp already has tail capture,
freeze, byte transport, prefix/mask and sampler wiring. Its masked runtime remains
unaccepted. A concrete implementation gap is Exp with trained Stage2 upscaling:
the geometry contract requires the base size and runtime explicitly rejects that
combination; Director retains the original high-resolution prefix across stages.
The specific previously denied diagnostic remains denied. Implementing this
contract does not itself require repeating that diagnostic.

Remaining image producers (MOSS/Rust-Qwen/Comfy) and agreed preview behavior remain
open. No guessed H3 latent factors are substituted for a different model family.
A side-agent read limited ComfyUI clean source outside the latest express source
exception; that reading was stopped and is not used as acceptance evidence. The
exact incident and narrow reference dependency are recorded in CHECKPOINT-36.

Release triage separates actual missing functionality and focused source-faithful
acceptance from broad speed/VRAM sweeps. Performance benchmarking and optimization
are not new UI release gates. Known functional/algorithmic differences are not
waived by this distinction. OPEN35's remaining per-group criteria are preserved,
with CP36 closures and corrected labels in OPEN36; publication is still pending.

## CP37 — remaining image preview producers and Exp Stage2 prefix contract

Rust MOSS/Qwen and the Python image adapter now emit optional bounded RGB sampler
previews through the shared job-scoped RAM hub. Cadence is checked before render,
bad preview data is nonfatal, and final output accounting remains separate.
Projection tables come from the already allowed Imagine native source, not the
stopped external source reads. CPU scalar/Torch and cross-language RGB/resize
oracles pass. No ComfyUI algorithmic-parity claim follows from those own oracles.

An actual MOSS four-step 256x256 GPU retry passes through the normal shared
Scheduler/Ticket: preview disabled/enabled gives byte-identical final PNGs, and
the enabled job delivers four previews. The old test-only zero-utilization guard
had mistaken aggregate graphics activity for an inference conflict; it was
removed after Tomas requested another attempt. Coordinator admission remains.
Both own workers/server are reaped and the reservation is released. Ordered
cold/warm wall times are not a preview performance comparison.

Exp with trained Stage2 now carries explicit base/factor handoff metadata,
area-pools the Stage1 prefix and restores the retained original high-resolution
prefix/mask after Stage2 upscale/re-noise. Audio is not spatially resized.
Director source establishes the geometry/factor contract. Focused CPU, freeze,
wire, compatibility and builds pass. Matching H3 protocol 15 is required for the
new descriptor; ordinary legacy descriptors retain their representation.
Actual Exp Stage2 GPU output/audio and masked quality remain unaccepted. The
specifically denied masked-step diagnostic was neither run nor retried.

Normal/QA server/UI and both workers build; touched production Clippy and fmt
pass. Immutable cp37-components packages are not activated. Source evidence is
incremental (27 files, 388587 bytes, depending on CP36), with unchanged original
baseline. The exact remaining group criteria are carried into OPEN37 without
turning broad performance benchmarking into a release gate. No publication,
storage migration, cleanup, user process action or runtime-log read occurred.

## CP38 — ordinary Exp Stage2 GPU and actual image-preview acceptance

Ordinary coordinated source and Exp-target jobs now pass with trained factor2
upscaling: base416x224, output832x448,39frames24fps. Each runs four intervals of
the20-step Stage1 schedule plus four Euler Stage2 intervals. Both export
MP4/H264/yuv420p/AAC stereo32kHz with exact1.625s duration. Raw PCM RMS is
0.00526461/0.00183866, peak0.0381013/0.0130889; no gain or normalization.
Source/target job times are81.4115/84.4561s, not a performance comparison.
The target provenance identifies the accepted high-resolution source handoff.
Target-first/source-final raw RGB correlation is0.998590,MAE1.93272/255.
Own source/target frames0/19/38 were visually inspected. No generic flat-green
failure appears; the target follows its explicitly changed colored-pattern
description. This is short functional evidence, not full Python or quality
parity. No intermediate observer/dump or denied masked-step diagnostic ran.

Rust Qwen also passes a separate four-step256x256 explicit-workflow preview
off/on regression with byte-identical final PNGs and four real previews. Its
first own test request incorrectly reused MOSS's normal scheduler and received
HTTP422 before admission; the harness was corrected to the existing Qwen beta
semantics. No product setting/validation changed and the frozen baseline is
untouched. This new preview workload is not a baseline speed comparison.

Real MOSS generation, preview texture, final PNG and terminal texture cleanup
pass through the actual native UI at1100x760 and600x760. The final PNG matches
the earlier API-only off/on case. All eight inspected normal/small screenshots
have readable tooltips, aligned dialog fields/actions and centered preview/final
images without new overlap. Lower edit fields require the existing outer scroll
when Queue is expanded; that scrolling is not accepted merely from screenshots.

The initial native-start request was rejected before execution because automatic
review applied the older service-start prohibition and did not accept forwarded
later approvals. Tomas then directly approved the isolated operation; its normal
retry was allowed. A test-fixture empty catalog and an invalid resize assumption
were subsequently fixed as ordinary recoverable test failures. Only the own
example server gained --own-image-catalog; production UI/backend/worker code was
unchanged. The corrected fixture has fmt/Clippy/build pass in a separate package.
The QA startup flag fixes min/max window size, so the narrow case starts directly
at600x760 rather than changing compositor settings or calling1100px 'small'.
Failed attempts remain preserved with their exact outcomes.

The external ComfyUI source incident remains the exact stopped CP36 read, not
a new denial or current Stage2 implementation blocker. No further source/log
access or denied diagnostic occurred. Own schedulers shut down normally and all
own servers/UI/workers/leases are released. No cleanup, activation, storage
migration or user-process action. Matching cp37 production components remain
unpublished; cp38-fixture-server is test-only. OPEN38 retains exact remaining
focused parity and common functional/visual acceptance, without adding broad
benchmarking or an unspecified WebP/VAE preview feature as a release gate.

## CP39 — reference pixels, manual SIGMAS and executable sampler selection

The allowed Director/Pillow CPU reference-pixel probe passes seven byte-exact
RGB float32 cases, including non-square boundary geometry and replicated edge
padding. Its resize provider is the allowed Director media Pillow Lanczos
function; this is not a claim that the stopped external ComfyUI wrapper or a
full trained reference pipeline was executed.

The source-executed Director Stage2 composition oracle exposed a real manual
SIGMAS mismatch: admission incorrectly capped values at one and validated
unused generated-schedule metadata. Those restrictions are removed for manual
curves. All twelve composition/error cases now match; successful FP32 sigma
bits and scheduler callback arguments are identical.

The H3 worker previously admitted and dispatched only Euler/res_multistep while
the UI exposed a sampler field. It now dispatches the selected name through
the existing shared solver registry, KSAMPLER, seeded default noise and Brownian
owners. DDIM retains its actual random-inpaint factory behavior. CFG++ capture
uses the port's real positive/absent-negative aggregation; a focused test compares
it bit-for-bit with conditioning_batch and guidance, without a negative model
call. Six focused CPU tests pass, including all45 registered names and existing
saved-original numerical/noise/mask oracles. This is not full-model GPU or
complete Python-pipeline acceptance.

Adaptive callback counts must not abort generation: DPM adaptive reports six
callbacks for four planned intervals, while dpm_fast reports two. A shared
progress tracker preserves labels across stages and corrects the display total
without adding model callbacks. H3 wire16 carries this semantic change; old
wire15 runtime packages must not be paired with the new worker. Twenty-five
targeted server/protocol/pipe tests, three tracker cases and the CPU worker-wire
suite pass. Server, normal UI and H3worker build; worker/UI strict Clippy and
changed-file rustfmt checks pass. The new component package is unactivated and
is not yet a complete runtime release bundle.

The actual empty native image UI passes a600x760 expanded-Queue scroll probe:
31 actions, ten geometric checks, outer offset460 unchanged across pointer
movement and seed-popup open/Escape. The bottom control is fully visible and
actually usable. No server/job/model/GPU/screenshots or runtime-log reads occur;
only a separate geometry-only QA result is read. Two Rust geometry regressions
and three plan tests also pass. Production scrolling behavior was not changed
speculatively; the added hooks are ui-preview-only test instrumentation.

All own processes are reaped; no GPU reservation was taken in this checkpoint.
No publication, storage migration, cleanup, user process action or previously
denied diagnostic/source/log access occurred. OPEN39 separates the remaining
focused model/reference acceptance, contextual UI flows, common visual checks
and final complete runtime packaging from broad performance experiments.


## CP40 — focused real-DiT, contextual editor and Final VFR range

- G7 real BF16 DiT on numeric fixtures: res_multistep/Heun manual sigma>1,
  CFG++ finite-domain and dpm_fast actual2/2 progress PASS. No text/job/VAE/
  export or full Python pipeline claim. Own worker/normal lease finished.
- G1/G2/G3 native contextual Project-assets flow PASS normal1100x760 and
  small600x760. Images/voice/Section/audio/Story target IDs, live updates,
  default/enabled/Undo/reopen and Unicode/caret completion verified.
- Visual QA found and fixed duplicate Custom value widget IDs, small Cast
  clipping and detached ID label. Final short layout probes/screenshots PASS
  in both sizes; no speculative scroll policy or extra product control.
- G13 procedural MOV/MP4 and VFR Matroska/MP4 Final oracles found a causal
  soundtrack-range mismatch (container2s versus decoded24frames/24fps=1s).
  Sound uses the same selected-frame range as picture; missing frame-count
  metadata uses a cancellable one-pixel count pass. Both corrected rawRGB
  comparisons are exact, PCM max1.49e-8; MP4 AV clock/color pass.
- Native OS-picker manual fixture prepared, NOT accepted by injected paths.
  Normal product has no QA manual-root flag. Accepted publication order stands.
- Exact older Exp1-to3/20 diagnostic denial documented without retry. Its current
  necessity is unestablished; ordinary functional/parity gaps remain explicit.
- Normal server/UI build, strict UI and lib-unit-test inference Clippy, affected
  CPU tests and focused Final regression pass. No complete runtime bundle or
  publication yet, no storage change/cleanup/user process change.
- Evidence: UI-GROUPS-CHECKPOINT-40.md, UI-GROUPS-OPEN-40.md,
  EXP-APPROVAL-SCOPE-40.md; incremental40 depends39. No full snapshots.


## CP41 — import ownership, library lifecycle and bounded continuity

- Reproduced same-library late-reply overwrite before fix; picker requests now
  own their response token across reopen/refresh. No model/file policy gate.
- Native Story/Section Cancel and actual malformed-PNG recovery pass normal/small.
  The captured-target error is now visible in the picker, not hidden behind it.
- Older/rich own library fixtures, missing-media rollback, Unicode/keyboard and
  reopen pass. Open errors describe Open rather than project Save; F05 retained.
- Library/editor thumbnail starvation fixed with source namespace+asset keys and
  actual project/session lifetime reset. Actual blue fixture is visible in both.
- Four direct async event tests cover stale preview/comparison/project/import and
  captured removed targets; two typed continuity tests cover reuse/style/audio,
  invalidation/freeze/prompt/pending receipt/reopen without any model/job.
- Normal server/UI build and strict QA UI Clippy pass. No GPU in41, all own
  processes ended, no runtime log/user data/denied-operation access or cleanup.
- Publisher can take explicit candidate --source-root (root8 versus candidate16),
  verified from source, not yet staged/activated. Small publisher patch retained.
- Actual OS chooser remains manual and unaccepted; exact click list and CP41 own
  script/fixture paths are included in CHECKPOINT41. Full masked Exp/Python parity
  remains an open porting requirement, not proved by typed UI tests or waived.
- Remaining: joint visual/icon presentation, matching runtime bundle+rollback,
  human OS point/decision, then approved publication and separate storage stage.


## CP42 — published UI candidate, 2026-10-22

Selected for the next manual start: `comfyui/products/20261022T221439Z-a81a1f6f`.
Normal server/UI, H3worker16 and imageworker1 are frozen with matching runtime
configuration, dependency map and preserved rollback. The actual isolated package
smoke passed real H3 Ready16, pre-inference invalid-frame rejection, normal server
manifest loading/H3 availability and absence of QA entry in normal UI. No GPU or
jobs in this smoke. Imageworker CUDA Ready is covered by its reused CP38 artifact,
not falsely claimed by the CPU smoke.

Joint native visual finish passed 127 actions at1100x760 and127 at600x760.
Final Hide leakage and narrow-window overflow were fixed and inspected afterfix;
shared icons/tooltips now cover Cast reload/remove, cache clear/refresh,
media-health check and Review refresh. A duplicated cache tooltip was removed.
Normal build, strict QA UI Clippy, focused cache tests and changed-file fmt pass.
All own processes were reaped; no GPU reservation was taken in42. No user process
was restarted, no user data/logs read, no cleanup or denied experiment repeated.

Tomas explicitly moved the native OS picker/Cancel/Relink manual acceptance to his
first sharp test (Sentinel_94c7c7cc47d0819193f457b61caa580a). It is handed over,
not marked tested. Full Exp/Python and separate memory algorithm parity remain
OPEN porting requirements; UI publication does not waive or accept them.

Evidence: moved own task root `ui-groups-01/evidence/UI-GROUPS-CHECKPOINT-42.md`,
`source-incremental-42.json` (6sourcefiles215869bytes, depends41),
`cp42packagesmoke01/smoke-result.json`, `cp42visualreport01/REPORT.md`.
Storage implementation has not begun; it is the next separately specified stage
after this verified UI publication. No migration/deletion or new access implied.
