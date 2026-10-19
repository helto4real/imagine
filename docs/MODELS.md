# Models and LoRAs

Choose **Settings → Models** to set the server's model folder or refresh its
catalog. The default is the project's `models/` directory. On first setup,
Imagine preserves the existing `QWEN_MODEL_ROOT` / legacy local-settings fallback.
Once saved, the model-library setting takes precedence. The UI can point directly
to an existing `ComfyUI/models` directory; no weights are moved or copied.

## Discovery and supported weights

Discovery runs at server startup, on library setting changes and on Refresh.
It runs off the HTTP/UI event loop, reads filenames and bounded safetensors
headers for diffusion candidates only, and never loads inference weights onto the GPU or reads image previews,
sidecars, prompts or job history. The recognized directories are:

- `diffusion_models/`, with `unet/` as a legacy alias.
- `text_encoders/`, with `clip/` as a legacy alias.
- `vae/`, `loras/` and `checkpoints/`.

Subdirectories and symlinks are supported, with cycle protection and canonical
file deduplication within each category. Paths are scanned in deterministic order.
Group-named directories are optional and are not treated as compatibility proof.
LoRAs at every scanned subdirectory level retain their relative paths. Discovery
lists weight files by extension without reading LoRA headers or classifying
compatibility: `.safetensors`, `.safetensor`, `.gguf`, `.ckpt`, `.pt`, `.pth`,
`.bin` and `.sft` (case-insensitive). Sidecars and previews are excluded.

The Qwen Image 2.1 adapters retain these default filenames:

| Role | Default filename |
| --- | --- |
| INT8 diffusion | `qwen_image_2.1_int8_convrot.safetensors` |
| Turbo diffusion | `qwen_image_2.1_turbo_bf16.safetensors` |
| Text encoder | `qwen3vl_8b_int8_convrot.safetensors` |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` |

Diffusion candidates must have the expected native architecture indicators in
valid safetensors headers. Other diffusion formats/checkpoints remain discoverable
but unsupported. Encoder and VAE weight files are selectable regardless of name,
family, quantization, header contents or compatibility metadata. Their contents
are not inspected during discovery. The selected backend loads the chosen files;
discovery does not establish inference compatibility. Native binary/hash validation
still runs before inference. The frozen engine settings and native source are unchanged.

Existing family-specific filenames remain automatic default hints. If there is
no such hint, a sole discovered file is used; ambiguous choices require selection.
In **Settings → Models**, focus **Text encoder** or **VAE** and type to filter the
full relative paths. Search is case-insensitive and all whitespace-separated
words must match. Arrow up/down navigates, Enter selects, and Escape cancels the
search without changing the saved binding. Clicking a result also selects it.
No matching files leaves the previous binding intact. Selections are saved per
diffusion variant. Missing components block generation.
The selected variant exposes its own operations: Turbo supports image generation
and editing, while character sheets retain their existing INT8 restriction.
Turbo uses the shipped quick-draft defaults (including its fixed eight-step
schedule and CFG 1); incompatible control values produce an actionable error.

## Model-loading errors

Rust and Comfy model-loading failures identify the component, the selected
root-relative file and a bounded cause in the UI. Causes distinguish missing or
unreadable files, unsupported formats/architectures/quantization, invalid checkpoints,
missing weights, mismatched weight dimensions and insufficient memory. A missing
tensor name is included when safely available. Unrecognized failures explicitly
report that the loader did not provide a recognized cause; no cause is guessed.

The job API retains `error.code` and adds `component`, `file`, `reason` and optional
`tensor` for `model_component_load_failed`. Diagnostics refer to the submitted
job's frozen file selection, not later changes in Settings. They remain in the
session's in-memory job state. Raw exceptions, stack traces, prompts and tensor
contents are not added to API errors or logs. Older workers can still return a
generic failure. Rebuild/deploy the Rust worker with `./prepare-rust-backend.sh`
and restart Imagine and its UI to enable the full path.

## LoRA desk

The layers button beside the model selector opens a separate LoRA window in both
generation and editing. Search, add, toggle, reorder, remove and edit strength
there. The numeric strength field accepts finite values outside the slider's
usual -2 to 2 range. One strength is sent to the native engine per LoRA.
Search narrows the available list on each keystroke without submitting or
rescanning. Whitespace-separated words must all occur somewhere in the full
relative path, in any order and without case sensitivity. For example,
`qwen somemoreword` matches `loras/qwen/styles/somemoreword_v2.safetensors`.
Clearing the search restores the list; selected LoRAs are unaffected by search.

Every discovered LoRA is selectable directly, regardless of metadata or tensor
names. The user chooses which files belong in each model group's saved list;
there is no compatibility filter or separate assignment step. The native loader
handles the selected files when generating. Discovery is not an inference check.

UI preferences retain the selected variant, ordered list, enable flags and
strengths per group, including an explicitly empty list. Switching INT8/Turbo
retains that group's list. Missing saved entries remain visible;
an enabled invalid entry blocks submission until disabled, removed or repaired.
The UI's list is not implicitly inherited by API/MCP profile requests. Upscale
continues to use its explicit profile and LoRAs.

## Persistence and job isolation

- Server folder and component bindings:
  `<server data directory>/model-library.json`, atomically saved with mode 0600.
- UI group selections: the existing `.imagine/ui-settings.json` preference file.
- Resolved execution paths and file stamps: only the existing in-memory job store.

A submitted job captures the model root, component paths, LoRA list and strengths.
Later changes affect subsequent jobs. Discovery stamps reject stale catalog
entries; execution checks device/inode/size/mtime before each inference pass.
These stamps are not full weight checksums. Internal absolute paths/stamps are
excluded from public validation and job responses and cannot be supplied by a
client. Existing explicit profiles retain their component choices while using
the configured root. Existing retry/idempotency behavior remains in force.

## API

`/v1/models` retains the engine capability contract. The filesystem catalog is:

- `GET /v1/model-library`: settings, files, model candidates, readiness and warnings.
- `POST /v1/model-library/refresh`: rescan without loading inference.
- `PUT /v1/model-library`: save the complete settings object and rescan atomically.

Catalog file IDs derive from root-relative paths. Model entries provide adapter
`model`, group, defaults, operations and a request schema. Their `id` identifies
the discovered diffusion file, not the engine adapter. Model `available` means
the supported diffusion and selected/default components are present/unambiguous; it is not live inference
acceptance. To submit a catalog selection, use the ordinary job endpoints:

```json
{
  "model": "qwen-image-2.1-int8",
  "operation": "text_to_image",
  "prompt": "A blue ceramic teapot",
  "model_selection": {
    "diffusion": "<diffusion file ID from the catalog>",
    "loras": [{"id": "<LoRA file ID>", "strength": 0.8}]
  }
}
```

`loras: []` explicitly selects no LoRAs. Catalog selection cannot be mixed with
`profile` or explicit generation/component settings; clients choose one source
of model configuration. Server-side resolution and validation apply equally to
single jobs, batches and `/v1/jobs/validate`. External clients may use the same
catalog endpoints; this change does not add new MCP tools.

## Validation

Tests use tiny synthetic weight headers, temporary settings and fake inference.
They cover directory aliases/recursion/symlinks, diffusion headers, ambiguity,
unfiltered encoder/VAE/LoRA discovery and selection, cross-family choices, searchable
component pickers, persistence, missing/replaced files, empty lists,
queued-job isolation, API rejection and real egui add/remove/close interactions.
CPU-rendered desktop, compact, settings and LoRA views are separate from native
window/file-dialog acceptance. No live model generation is part of this change.
