# Video contract foundation (P01)

P01 implements the in-memory domain in `src/video`, exported through
`imagine_server::video`. `ProjectService` provides Create/Read/Edit with owned
snapshots and atomic optimistic revision checks. It performs no disk IO,
HTTP serving, inference, queue admission or runtime discovery.

- `contract.rs`: document identities, exact rational timing, model/profile
  selection, sections, asset bindings, continuity intent and retained Take identity.
- `project.rs`: structural validation and session-owned project collection.
- `editing.rs`: typed, atomic commands. Invalid batches and revision conflicts
  leave both document and revision unchanged. A batch can retain incompatible
  authoring; execution validation reports blockers instead of deleting inputs.
- `planning.rs`: explicit adapter registration, support/availability/evidence
  metadata, input validation and canonical scene input fingerprints. Input
  fingerprints exclude unrelated revisions, inactive settings and the scene's
  own Accepted state; they include effective references and continuity sources.

Model settings remain scoped to the full model/profile identity. Unknown adapter
settings survive document round trips, but execution validation blocks unknown
adapters or schemas. No default adapter is registered. Tests register two explicitly
synthetic adapters with different input/settings/continuity capabilities; neither
is advertised to product clients. The H3 settings DTO is an initial authoring
schema with explicit sampling and Stage 2 handoff/upscaler identity, without image
CFG defaults. Complete H3 scheduling/conditioning constraints and artifact
compatibility are P05 work. Latent continuation currently fails closed with
`continuity_artifact_unverified`, even when a synthetic adapter declares support.

The scene vector defines sequence order. Continuity sources must precede targets,
which prevents cycles. Section times are relative to their scene and must fit;
scene timing and delivery fps have no global H3 alignment constraint. Asset records
are identities/revisions, not file handles or media bytes. Runtime media inspection,
Take publication and audio-track editing are later packages.
The isolated P02 [storage foundation](../video-storage/README.md) now persists
project resources, completed Take records and separate library copies through
`ProjectSession`; runtime/API media integration remains pending.

`schemas.json` is generated from the same Rust DTOs; `create-project.json` and
`edit-project.json` are synthetic authoring examples, not executable H3 requests.
Deserialize requests and use the validated service boundary: JSON Schema describes
wire shape/ranges, while document relationships and adapter constraints are checked
by Rust. Tests deserialize/apply the examples and compare the checked-in schemas
against generated schemas; no external JSON Schema validator is required.

Regenerate schemas:

```sh
cargo run --locked --quiet --example video_schema > design/video-contract/schemas.json
```

Validation:

```sh
cargo test --locked --test video_contract
cargo test --locked --workspace
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Workspace tests use synthetic workers and temporary loopback servers; sandbox
restrictions on socket binding require test-only escalation. The existing live
Qwen test remains ignored. No native UI/manual or trained-model acceptance has
been performed for P01. No HTTP video routes are registered by this package.

Verified 2026-10-19: all 17 P01 synthetic tests passed; the product workspace
passed 207 tests with two existing tests ignored. Clippy with `-D warnings`
and workspace formatting checks passed. Workspace tests needed escalation only
for temporary synthetic loopback listeners. The separate `inference/` workspace
was not built or modified by P01.
