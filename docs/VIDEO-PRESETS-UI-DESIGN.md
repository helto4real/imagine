# Video presets in the UI

Status: **Implemented 2026-10-25** (first phase; no preset saving).

Date: 2026-10-25.

Scene presets exist for MCP productions (`config/video-presets.yaml`, see
[MCP-VIDEO-IMPLEMENTATION-PLAN.md](MCP-VIDEO-IMPLEMENTATION-PLAN.md), "Scene
presets"). A preset is a named recipe: an H3 variant (reference or I2V),
settings overrides (model file and other components, sampling, shifts,
reference quality, stage two) and LoRAs with a merge mode. This design makes the
same presets usable in the UI, with identical results to MCP.

## 1. Principles

- **One preset file, one application rule.** The UI and MCP read the same file,
  and the server turns a preset into project edits with the same code for both.
  A scene made by an agent with a preset and a scene the user sets to that preset
  have the same settings.
- **A preset writes ordinary settings.** Applying a preset sets the scene's mode,
  its settings slot and its LoRA override, exactly as editing them by hand would.
  Everything stays editable afterwards; nothing depends on the preset file at
  generation time.
- **Undoable and revision-checked.** Applying is one edit batch, undone as one
  step, rejected on a stale revision like any other edit.
- **The file stays hand-editable.** The UI never rewrites the user's file in the
  first phase.

## 2. What the user can do

1. **Apply a preset to a scene.** The scene inspector gets a **Preset** dropdown
   above "Scene mode": the presets by key, with unavailable ones disabled and
   their reason as a tooltip. Choosing one applies it.
2. **Apply to several scenes.** With several scenes selected, the same dropdown
   applies the preset to all of them in one undoable step.
3. **Set the project default.** The Project generation window gets a **Default
   preset** dropdown: it sets the default scene mode, the project's settings for
   that mode and the project's **global LoRA stack** from the preset's LoRAs.
   New scenes and scenes that "Use project generation settings" follow it.
4. **See where a scene came from.** The dropdown shows the preset a scene was
   set to. When the scene's settings or LoRAs have been changed since, it shows
   **Modified**, with a **Reapply** action that restores the preset's values.
5. **Browse presets.** A **Video presets** window (project menu) lists every
   preset: key, variant, description, model file, sampling values that differ,
   LoRAs, and availability with the reason (invalid setting, missing file, a
   problem with the file itself). It has **Reload** and shows the file's path.

Not in the first phase: creating, editing or deleting presets from the UI
(section 7).

## 3. Server

### 3.1 One application function

Move the preset-to-edits logic out of `engine::apply` into
`src/video/presets.rs`:

```
fn scene_edits(project, scene_id, preset) -> Result<Vec<Edit>, code>
```

It produces `SetSceneModel` (override when the variant differs from the project
default, otherwise project default), `SetModelSettings` with the layered slot
(only when the preset has settings overrides) or `InheritModelSettings`, and the
scene LoRA override. MCP's storyboard apply and the UI both call it.

A project-default counterpart sets `SetDefaultModel`, the project's generation
slot for that model, and the global LoRA stack (`creative.h3_loras`) to the
preset's LoRAs.

### 3.4 LoRA layering (decided)

- A preset applied as the **project default** sets the **global** LoRA stack.
- A preset applied to a **scene** adds its LoRAs **on top of the global stack**
  (`Add To Global`). Scene LoRAs never replace the global ones by default.
- `lora_merge` in the preset file still allows an explicit `Replace Global` or
  `Disable LoRAs` for a preset that needs it; its default changes from
  `Replace Global` to `Add To Global` for both the UI and MCP, and the example
  file and agent guidance say so.

### 3.2 Endpoints

| Route | Purpose |
| --- | --- |
| `GET /v1/video/presets` | The catalog for the UI: key, variant, description, default flag, settings overrides, LoRAs, merge mode, availability and problem per preset, plus the file path and a file problem. Model and LoRA paths are included here (the UI is local); MCP's summary keeps omitting them. |
| `POST /v1/video/projects/{id}/presets/plan` | `{schema_version, expected_project_revision, preset, scenes: [ids]}` or `{…, project: true}`; returns the edit batch built by `scene_edits` / `project_edits` without changing the project. The UI submits it through `/edits`, so it is recorded in the UI's undo history like any edit. Unknown (`preset_unknown`, 404) or invalid preset (its problem code, 422), unknown scene (`scene_not_found`), neither or both targets (`invalid_request`) and stale revision (`revision_conflict`, 409) are errors. |

### 3.3 Remembering the preset

Add `scene_presets: BTreeMap<SceneId, AppliedPreset>` to the project's creative
data (serde default, omitted when empty, so older projects load unchanged), with
`AppliedPreset { key, fingerprint }`. The fingerprint is a hash of the scene
settings, model and LoRA override the preset produced. The UI shows **Modified**
when the scene's current values no longer match the fingerprint, and the key
alone if the preset was later removed from the file. MCP records the same entry.
The older-compatible export omits it, like other newer fields.

## 4. UI

| Place | Change |
| --- | --- |
| Scene inspector (`ui/src/video/inspector.rs`, `generation_ui.rs`) | **Preset** dropdown above "Scene mode"; shows key or "Custom"; **Modified** badge and **Reapply**; applies to all selected scenes |
| Project generation window | **Default preset** dropdown above "Default scene mode" |
| Project menu | **Video presets…** opens the presets window |
| New `ui/src/video/presets.rs` | Catalog state, fetch through `GET /v1/video/presets`, the window, and the plan request |

The catalog is fetched when the inspector or a preset control first needs it and
on **Reload**. Following the Helto design system: concise labels, no helper
paragraphs, gold selected state, error text for unavailable presets. Preset keys
and file names follow the LoRA window's privacy behaviour and are concealed in
hide mode.

## 5. Tests

- Server: `scene_edits` matches MCP storyboard application for the same preset;
  apply to one and several scenes; project default; default presets restore
  plain project settings; errors; undo restores the previous values; fingerprint
  and Modified detection after an edit; catalog endpoint shapes.
- UI: dropdown lists presets and disables unavailable ones; choosing sends one
  plan request with the selected scenes; Modified and Reapply; presets window
  rows; concealment in hide mode.

## 6. Decisions and open questions

1. **Modified tracking (section 3.3): decided, yes.**
2. **LoRA layering (section 3.4): decided.** Project default preset LoRAs are the
   global stack; scene preset LoRAs are added to it.
3. **Creating presets from the UI (section 7):** open; not in the first phase
   unless requested.

## 7. Later: creating and editing presets

"Save scene as preset…" would write a new entry. Rewriting the YAML with a
serializer would drop the user's comments and formatting. Options:

- append a generated entry as text at the end of the `presets` list, keeping the
  rest of the file byte-for-byte; or
- keep UI-created presets in a second file (`config/video-presets.ui.yaml`)
  that the catalog merges, so the hand-written file is never touched.

The second is safer and simpler; keys must still be unique across both files.
