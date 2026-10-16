# Character sheets

`character_sheet` is a job operation. The first supported model is
`qwen-image-2.1-int8`, selected by the `qwen-character-sheet` profile.
Turbo is not enabled for this operation. The matched Qwen reference workload
has passed live acceptance; see the [reference baseline](../native/reference-cache/ACCEPTANCE-REUSE.md).
This does not validate every allowed layout, prompt, reference count or GPU.

## Request

Use the existing authenticated `POST /v1/jobs` endpoint:

```json
{
  "profile": "qwen-character-sheet",
  "operation": "character_sheet",
  "seed": 42,
  "character_sheet": {
    "character_description": "An adult character with short dark hair and freckles.",
    "clothing_description": "A blue jacket, grey trousers and black shoes.",
    "layout": {
      "output_width": 1536,
      "portrait_fraction": 0.45
    }
  },
  "inputs": {
    "reference_images": []
  }
}
```

The profile supplies the operation, so it can be omitted when selecting this
profile. Descriptions are plain text or supported Helto encrypted envelopes.
At least a nonblank character description or one identity reference is required.
Each description is limited to 12,000 UTF-8 bytes. Clothing is sent only to the
body-view pass. Empty clothing asks the model to retain the reference outfit,
or choose a coherent outfit when no reference exists.

References are ordered, owner-scoped asset IDs from `POST /v1/assets`. Up to
eight identity references are accepted, reserving two of the current ten Qwen
slots for the generated portrait and pose guide. A full existing character
sheet is an ordinary identity reference: both panels are regenerated, with no
promise of exact pixel preservation.

Optional `inputs.pose_guide` is another owned asset ID. It replaces the built-in
synthetic guide and reaches only the body-view pass. Its ownership and geometry
are checked before queue admission, just like identity references.

The output is 16:9. `output_width` is 512–4096 in multiples of 512. The default
is 1536x864 with a 0.45 portrait fraction. The fraction must leave at least 32
pixels in each panel. Generation widths round to multiples of 32, independently
of the exact composition boundary. The default passes are 704x864 and 832x864.
Top-level `width`, `height`, `max_side`, `aspect_ratio` and a nonempty `prompt`
are rejected for sheet requests. Layout owns the canvas. Scene colour-matching
options are unsupported; LoRAs and sampling use the normal Qwen profile options.

`POST /v1/jobs/validate` performs validation without queueing or inference.
`GET /v1/profiles/qwen-character-sheet` describes the options. Validation and
profile descriptions omit character/clothing descriptions and prompt text.
Owner-authenticated job details retain the effective request within the session.

The new profile copies the existing explicit Qwen model and sampling defaults.
Each pass with references selects the accepted hybrid/prefetch native profile.
A portrait pass without references uses frozen v2; its generated portrait and
pose guide make the following body pass eligible for the reference profile.
The frozen v2 binary remains unchanged and available as fallback. The original
1600x896 validation is separate from the matched sheet acceptance.

## Execution and results

One job holds one GPU coordination ticket across portrait, turnaround and
composition. Original references enter both passes; the generated portrait and
pose guide follow the originals, in that order, in pass two. Every reference,
including internal images, uses the existing shrink-only /32 preparation. The
full-size portrait is retained for composition. No intermediate public assets
are created during inference.

The portrait uses the resolved job seed. The turnaround uses the next seed,
wrapping within Imagine's existing 53-bit seed range. Omitting the seed chooses
one random seed at admission. An identical idempotent submission reuses the
existing job. `effective.execution_plan` records the strategy, prompt,
composition and guide revisions, dimensions, reference roles and both seeds.
These are server-owned metadata. Profile settings resolve at admission; templates
are compiled into the executable. Queued plans are checked before execution.

Progress retains the existing phase and within-pass sampling fields and adds
`stage`: `portrait`, `turnaround` or `compose`. Step counts apply to the current
pass, not to the entire job. Cancellation or inference failure discards the
whole result. Publication stages files and commits all asset records plus job
success together; a staging/commit failure removes staged files.

A successful job has three entries in `output.assets`, in this order:

| Role | Content |
| --- | --- |
| `character_sheet` | Primary composed 16:9 sheet |
| `portrait` | Original full-resolution portrait pass |
| `turnaround` | Original full-resolution three-view body pass |

`output.count: 1` means one sheet, not one returned artifact. All three assets
share the existing owner isolation and session cleanup. The CPU compositor
covers/crops the portrait panel and contains the body-view image without
stretching. RGBA is composited onto white. No generated prompt metadata is
embedded in PNGs.

The existing MCP `generate` tool can select this profile and supply structured
sheet parameters and its usual identity references. Its existing delivery path
exports the primary sheet only. Custom pose-guide upload and access to all
three result roles are available through the HTTP API and the native
[Character sheets tab](UI.md#character-sheets). There is no separate recompose endpoint.

## Responsibility boundaries

- `generation.rs` resolves a typed image or character-sheet task and orchestrates
  passes under the existing scheduler lifecycle.
- `model_adapter.rs` explicitly selects supported sheet strategies and owns
  Qwen prompts, reference roles and revision metadata. A future model needs an
  explicit strategy and validation; generic image-edit support does not imply
  character-sheet support.
- `character_sheet.rs` owns model-independent inputs, geometry, composition and
  the synthetic pose guide.
- `engine.rs` accepts an `ImagePass` containing resolved settings and in-memory
  references, without access to public asset IDs or the store. The native
  adapter handles reference preparation and native request mapping.
- The existing scheduler owns cancellation and GPU coordination. The store
  publishes the completed result group atomically.

The AIO portrait and turnaround instruction text was retained in versioned
source templates under `src/prompts/qwen-character-sheet-v1/`. Native source
uses the same ordered `<imageN>` markers. Graph imports, ComfyUI execution,
general workflow plugins and inference optimizations are outside this change.

## Verification boundary

Synthetic tests exercise schema/ownership checks, privacy filtering, geometry,
RGBA composition, guide determinism, native reference preprocessing, prompt
roles, seed wrap, idempotency, sequential passes, cancellation, second-pass
failure and publication rollback. Existing API, proxy, coordinator and UI-client
regressions remain applicable. The accepted reference test covers a 1536x864
sheet, 704x864 portrait, 832x864 body pass, CFG 1/2.5 and the documented
1,000,000-pixel reference cap. Result equality, cache behavior and API
cancellation/recovery are recorded in the acceptance report. The capability-wide
`live_validated` flag remains conservative; acceptance is workload-specific.
Further live runs require separate authorization.
