# Imagine generation profiles

Imagine loads `profiles/*.yaml` and `profiles/*.yml` in this folder
(`config/image/` of the configuration root) at startup.
`--profiles /absolute/directory` selects a different catalog. Invalid files,
unknown fields and duplicate IDs/keys fail startup. Restart Imagine after an edit;
increment `version` when changing a published profile. Already queued jobs retain
their resolved settings. The proxy does not load graphs or YAML itself.

`qwen.yaml` is **Qwen Image 2.1** (version 5); `quick-draft.yaml` is
**Qwen Image 2.1 Turbo** (version 3). Both accept zero to ten ordered references:
no references selects text-to-image; references select image-edit. Each recipe
owns its model files and sampling defaults. Standard Qwen and MOSS (`moss.yaml`)
carry a plain default negative prompt, which the server also reports in their
model metadata; Turbo retains its empty negative prompt and
zero-negative-conditioning setting.

A profile without inline `prompt_guidance` takes it from `guidance/<id>.md` beside
the profiles folder when that file exists (`guidance/moss.md` for MOSS prompts,
`guidance/qwen-character-sheet.md` and `guidance/moss-character-sheet.md` for the
character sheet fields); inline guidance wins. `describe_profile` returns it to
agents. `character-sheet/` holds the server's own prompts: `portrait.txt` and
`turnaround.txt` for the two Qwen passes, and `moss.txt` for MOSS.

`qwen-character-sheet.yaml` makes a sheet in two Qwen passes, a portrait and a
turnaround guided by up to eight identity references, which the server composes
and publishes with both passes. `moss-character-sheet.yaml` has MOSS draw the
whole sheet in one text-to-image pass from the description alone, so it takes
no references and publishes only the sheet. `moss.txt` fills
`{character_description}`, `{outfit}` (the clothing description, or an outfit
MOSS chooses), and `{portrait_percent}`/`{views_percent}` from the layout's
`portrait_fraction`. The job's `execution_plan.output_roles` lists the images a
sheet publishes.

Character sheets and upscale retain their dedicated operation profiles.
The overlapping `image_generation`, `qwen_scene` and frozen-v2 wrapper presets
are preserved in [`docs/archive/image-profiles/`](../../docs/archive/image-profiles/),
outside the active catalog. Native frozen binaries,
manifests and evidence are unchanged. Catalog descriptions never decrypt or
expose default prompts.

## API

All routes use the shared local Imagine workspace without Bearer authentication:

- `GET /v1/profiles`: IDs, names, versions, model IDs and YAML SHA-256 hashes.
- `GET /v1/profiles/{id}`: defaults (prompts omitted), partial override schema,
  reference limits, sizing, optional prompt guidance and NSFW availability.
- `POST /v1/jobs/validate`: resolve/validate without queueing or GPU use; returns
  effective parameters with prompts and client metadata omitted. Seed remains
  null until job admission if not supplied.
- `POST /v1/jobs`: resolve, validate, queue and persist the request.
- `POST /v1/jobs/batch`: resolve and validate every item before atomic admission.

A normal JSON request selects a YAML profile and supplies overrides:

```json
{
  "profile": "qwen",
  "profile_version": 4,
  "prompt": "A blue ceramic teapot on a wooden table, studio photograph.",
  "width": 1024,
  "height": 1024,
  "sampling": {"steps": 24},
  "seed": 42
}
```

Omit `profile_version` to use the loaded version. A version mismatch is HTTP 409.
Nested objects merge recursively; lists replace completely, so `loras: []` clears
the profile's base LoRAs. Explicit `null` is a value, not a deletion instruction:
nullable fields such as seed accept it; sampling/steps/etc. reject it.
Unknown or incompatible settings are rejected before admission. To override a
Turbo custom schedule's step count, also supply a matching `custom_sigmas` list
or clear it with `[]` to use the selected scheduler. A nonempty schedule must
have `steps + 1` finite, descending/non-increasing entries and end at zero.

`model`, model component paths, sampling, dimensions, prompt/negative prompt,
inputs, output and client metadata can all be overridden within Imagine's
validated capabilities. Choosing another model requires matching component
settings; selecting its own profile is usually simpler. Model and LoRA paths
are relative to the existing model roots, never shell commands.

The canonical component settings field is
`model_options.qwen_image_2_1.generation`. The old `workflow` spelling is still
accepted in legacy requests without a profile. Requests without `profile` keep
the existing explicit API contract, including the current UI client.

`operation` is inferred as `text_to_image` or `image_edit` from references when
neither the profile nor the client specifies it. Set `inputs.reference_images`
to ordered asset IDs returned by `POST /v1/assets`. Asset ownership is checked.

The maximum longest side is **4096 pixels**. For `image_upscale`, **3200 pixels
is only the default** when `max_side` is omitted, not the maximum. The upscale
profile is now version 2; its generation settings are unchanged.

For aspect-based sizing, send `max_side` (256–4096, multiple of 32) and optionally
`aspect_ratio`, e.g. `"3:4"`. These cannot be combined with explicit width/height.
Without a ratio, the profile's dimensions supply the ratio. An upscale profile
may set `sizing: {max_side: 3200, reference_aspect: true}` to use its first
reference. Explicit width/height overrides bypass that sizing policy. Results
are aligned to multiples of 32; reference aspect sizing rounds down, otherwise
the short side rounds to nearest/even.

`nsfw: true` appends the profile's `nsfw_loras` to the resolved base LoRA list.
It defaults to false and is only an explicit LoRA opt-in, never inferred from
content. A profile without configured NSFW LoRAs rejects true. Clients can also
supply ordinary explicit LoRA lists in `generation.loras`.

Jobs expose `effective.resolved_profile` with ID, version and the loaded YAML's
SHA-256, plus all resolved execution parameters and seed. This is provenance,
not client-writable metadata. Random seeds are resolved once per job; retries
with the same Idempotency-Key reuse the existing job. Prompts in job responses
remain owner-scoped. MCP excludes prompt text and client metadata from results.

## MCP

The current tools are `list_profiles`, `describe_profile`, `register_image`,
`generate`, `generate_batch`, `get_job` and `confirm_saved`. Use `profile`, replacing the old
`workflow_id`. Select `quick-draft` instead of `parameters.mode`. No Comfy graph
is consulted. `--mcp-workflows-dir` is accepted only for CLI compatibility and is
unused by MCP. The reference graphs are retained solely as historical test data.

```json
{
  "name": "generate",
  "arguments": {
    "job_id": "teapot-001",
    "profile": "qwen",
    "parameters": {
      "prompt": "A blue ceramic teapot on a wooden table, studio photograph.",
      "width": 1024,
      "height": 1024,
      "sampling": {"steps": 24},
      "seed": 42
    }
  }
}
```

`parameters` has the same override fields as the API, except profile selection
and registered reference IDs are separate MCP arguments. Each batch item may be
an override object or `{parameters: {...}, references: [image_id]}`. All items
use the selected profile. MCP asks Imagine to validate every request before
starting the batch; Imagine repeats authoritative validation at admission.
MCP keeps its process-owned jobs, retry handling, private reference storage,
FIFO coordination and local result files.

## Character sheets

`qwen-character-sheet` selects the two-pass Qwen character-sheet operation.
It accepts structured identity/clothing descriptions and up to eight identity
references, and returns a composed sheet, portrait and turnaround. See
[the character-sheet API](../docs/CHARACTER-SHEETS.md) for layout, pose-guide,
result and live-validation boundaries.
