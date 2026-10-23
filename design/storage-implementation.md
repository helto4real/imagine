# Approved storage stage after CP42

CP42 `20261022T221439Z-a81a1f6f` is published for the next manual start and
remains immutable/rollback-capable. Storage work changes only the isolated
`comfyui/ui-groups-20261021/source` candidate until its own acceptance. No user
data migration, deletion, service restart or new GPU/performance work is implied.

## Exact destinations

- Private project: `/run/media/veracrypt1/imagine/projects/<id>/project.json`;
  generated `outputs/images`, `outputs/videos`, `outputs/audio`; private `work`,
  `tmp`, `autosave`. All private disk bytes must start on the mounted volume.
- Library private metadata: `/run/media/veracrypt1/imagine/library/`.
- Selected reference pictures/video/voices: original path only, never copied to
  session, project, library or frozen inference/montage staging. Thumbnails RAM.
- Public preferences only: `$HOME/.config/imagine/ui.json`; model/LoRA selection
  and catalog configuration in `models.json`. No prompt/reference data there.
- Technical cache: `$HOME/.cache/imagine/`, no private media/text/thumbnails.
- Sanitized technical logs: `$HOME/.local/state/imagine/logs/`; no raw errors,
  private text, media paths or dumps. Private error details only RAM/server→UI.
- Sockets and locks: `$XDG_RUNTIME_DIR/imagine/`, no `/tmp` fallback.
- Clipboard normal memory. Swap/hibernation/security settings unchanged.
- External LLM API tokens removed; local session/server authorization preserved.

## Source write-path inventory (not a user filesystem inventory)

| Owner / current code path | Current behavior | Required storage integration |
|---|---|---|
| `src/main.rs`, `config.rs` | `.imagine` data default, data/server.lock, model-library and enhancement JSON | verified private session/project root; public catalog path; runtime-only lock; private provider metadata |
| `src/session.rs`, `storage.rs` | RAM SQLite/uploads, data/media session directory | keep RAM state; disk staging only private per-project tmp/work |
| `video/api.rs::staging`, `streaming_import_api.rs` | selected refs byte-upload to private-temp descriptor | local path registration without disk copy; source descriptor in RAM |
| `video/persistence.rs` | copies all resources to session staging | explicit external source ownership, no copy; owned generated resources stay private |
| `video/bundle.rs` | `.imagine-lock`, media/takes/continuity hash blobs, project manifest | runtime lock; new private project layout; no external reference blobs; legacy read unchanged |
| `video/library.rs`, `library_api.rs` | independent original-media library copies | private metadata and retained original-path links only |
| `reference_inputs.rs`, `assembly/freeze.rs` | private full-source frozen copies | retained read-only source descriptor and change detection; no hidden reference copy |
| `video/output.rs` | pinned export root/date/numbered files with same-volume partials | per-project outputs/{images,videos,audio}; same first-byte encrypted policy |
| `ui/settings.rs`, `app.rs`, UI main | `.imagine/ui-settings.json`, including negative prompt/paths; local output helper | separate harmless UI/model JSON from private state; no unencrypted private export helper |
| `src/model_library.rs` | model-library.json under data | shared models.json namespace; preserve concurrent UI selections |
| `src/enhancement.rs`, shared/enhancement.rs, UI enhancement | saved provider profiles and secrets | no token fields/persistence/header; private profile metadata |
| `comfy_engine.rs`, H3 launcher, native engine | data/rust-cache and inherited temp/cache environment | technical cache separated; child TMPDIR/private work pinned from first byte |
| `scheduler.rs::log_job`, server/UI entrypoints | structured job event stderr plus fallible entry errors | sanitized status only; raw/private details not persisted by launchers |
| `ui/video/thumbnails.rs`, `player.rs`, waveform/media helpers | RAM texture/cache; memfd or decoder pipes | retain RAM-only behavior; no new disk thumbnails |
| `shared/privacy.rs` | reads existing Helto keystore session to decrypt profiles | preserve security/auth behavior; not an authorization to migrate/read the live keystore |
| start/coordinator configuration | current `.imagine/gpu.sock` selector | prepare candidate runtime socket path; do not disturb current user's coordinator |

## Implementation ownership and acceptance

Main owns layout/volume policy, server/API/UI transport integration, new project
output/session/config paths, sanitized logging and final integrated acceptance.
Token worker owns only enhancement token removal and its direct tests. Resource
worker owns source origin across persistence/bundle/library/frozen inputs/Capture.
No parallel edits to the same modules without explicit handoff.

Required vertical checks use only new own procedural fixtures: path link→bind→
Save/reopen→library→worker/Final freeze with zero original byte copies; missing or
changed source→Check Media/Relink; external Capture→Remove preserves original;
legacy manifest unchanged; public settings contain no private values; volume
missing/read-only/replaced creates no private fallback bytes; private first-byte
staging and temp in every child path; local session authorization unchanged.
All tests must distinguish injected source-fixture policies from the real mounted
encrypted destination; `/home/thhel/tmp` synthetic evidence is not product policy.

Initial foundation: `shared/storage_layout.rs` maps exact classes and runtime
paths without fallback; `private_storage::Root` reuses existing volume inspection
and dirfd/no-follow mechanics, checks before creating child directories and never
creates a missing mount. These are not yet all caller integrations.

Rollback was checked read-only: CP42 rollback selector chooses its preserved
copies of previous `20261021T165201Z-6dd93d2e`. All four binary bytes match the
previous package; H3 identity and dependency map match; the actual current CP42
selection was unchanged. Rollback was NOT executed.

Full Exp/Python and separate memory algorithm parity remain open and separate.

## Storage checkpoint 01 — integrated source, not published

The candidate now routes normal server startup through `Root::system`, stable
private image-workspace metadata and `Config::load_private`; it no longer purges
legacy data. RAM session state remains RAM; its disposable disk staging is the
workspace's private tmp. Image/video destination endpoints now return their
distinct output classes; Timeline and Final select their own project videos.
Private publication writes directly in those folders (no extra date directory).

New project Save/Open/Save As and private library APIs use guarded roots. Modern
project manifests mark storage_layout=1; legacy manifests remain unchanged.
Explicit opening of a legacy project outside the private root is read-only; an
explicit Save As creates a separate private project. Older-compatible copies
remain explicit and report nonrepresentable features rather than copying
original references. Normal UI no longer offers an arbitrary private output or
library folder. Server restart recovery can reopen a modern private project by
ID without an OS folder selection.

Original media are registered by absolute path, with readonly source identity
and change detection; no original bytes enter project/library/freeze copies.
Decoded working tensors and preview textures remain RAM. Private bundle locks
use the runtime directory, not project `.imagine-lock` files. Three decided
metadata files may coexist with private library.json without blocking first
library creation. No actual user data was opened to develop these behaviors.

Worker scratch uses a held directory inode and `/proc/<serverpid>/fd/<fd>`;
child environment overrides are applied after static runtime settings. The
optional Python worker pins tempfile.tempdir explicitly before runtime imports,
so failure does not invoke its normal /tmp search. Native compiler/kernel
cache stays technical; worker stderr remains discarded. Normal server/UI
entrypoint failures print static status, not raw exceptions. No new persistent
private diagnostic dump or general-purpose disk logger was added.

Remaining before storage release: actual private-mode UI/HTTP vertical acceptance
with owned fixtures, source writer audit completion (catalog/template editing
versus private runtime text), coordinator/start-path packaging for runtime-only
socket, real mounted private destination proof, worker scratch behavior in an
actual own child, integrated image/video/Final generation regression as needed,
normal/small visual inspection, matching runtime package and reversible publish.
Current tests are explicit synthetic volume/component proofs, not a claim of
full production mount, GPU or Python pipeline acceptance. CP42 remains selected.
No data migration, user service restart, cleanup or rollback was performed.

## Storage checkpoint 02 / CP43 — integrated candidate and staged package

All decided source write paths above are now integrated. MCP links original
references and keeps EXIF-normalized model input/preview in RAM, while returning
the server's already published output without a cache copy. Acknowledgement is
idempotent and non-destructive. Prompt-catalog editing creates a private metadata
copy of the in-memory source templates before its first edit; the external source
catalog remains unchanged. Its private directory coexists with first library
creation. Linked H3 FirstFrame retains the original descriptor without staging
copy. The frozen descriptor's identity is checked across queue/decode so source
changes cannot silently replace the approved input.

An actual `Root::system` proof on `/dev/mapper/veracrypt1` (ext3) created only a
fresh source-owned synthetic project, saved/reopened it, linked a procedural PNG
without copying and published directly in its `outputs/images`. Missing/readonly
mounts, path replacement, no runtime fallback and public/private preferences are
separately checked with injected own volume fixtures. An actual Python child
using the shipped bootstrap wrote only through its pinned private tmp handle.
These are storage proofs; they do not claim a new GPU/model or Python parity run.

Native Home-mode catalog Save/Save As/Delete passed at 1100x760 and 600x760,
49 actions per size. Private project/library/copy UI passed 27 actions per size.
Own screenshots were visually inspected; all own processes were reaped. The
first catalog QA launch was rejected because its handoff omitted the later
explicit CPU-test authorization; one normal retry supplied that human evidence
and was approved. No alternate execution path was used or runtime log read.

Normal server/UI/proxy build, formatting, strict Clippy, 29 image lifecycle and
25 video API tests pass. Focused reference/Final/library/source-change regression
evidence is recorded separately. No raw-error logger was added: normal entrypoint
failures and the proxy diagnostic sink emit only technical records; private
details stay in RAM responses. Operator-requested upstream stack terminal output
is not persisted by Imagine. No system swap/hibernation settings were changed.

`comfyui/products/20261022T234700Z-998934ba` contains matching storage server/UI,
proxy/helper binaries, stack script and optional Comfy Python bootstrap. H3
protocol16 and image protocol1 workers are byte-identical to CP42. The package
smoke passed normal CLIs, missing-runtime fail-closed and CPU H3 Ready/invalid-frame.
It deliberately did not open a normal production private workspace/settings or
submit real model jobs. Common private APIs/config/volume paths were tested
through own injected instances and the real-volume synthetic proof instead.

Start selectors support the bundled proxy only for a selected package declaring
it; CP42's legacy launch behavior is retained. The new package uses
`$XDG_RUNTIME_DIR/imagine/gpu.sock`, runtime-only locks and a matching packaged
Python bootstrap rather than mixing it with an older checkout. Synthetic
publication tests verified first selection, subsequent rollback including proxy
helpers/scripts/storage flag, common socket paths and missing-runtime failure.
The staged candidate's four rollback binary hashes, H3 identity and dependency
map match CP42. The existing CP42 rollback was also verified read-only against
its own previous package. Neither actual rollback nor storage activation was run.

CP42 remains selected. Storage package publication/first manual startup are
separate from these isolated implementation proofs; no user data was migrated,
deleted, inspected or rewritten, and no running user process was restarted.
OS-file-picker acceptance remains explicitly handed to the user from CP42.
Full Exp/Python and memory algorithm parity remain separate, open requirements.

## CP44 — storage selected for next manual start

The packaged proxy → own isolated private API integration passed in own case
`sp44f`. It used a procedural EXIF6 JPEG, synthetic CPU backend and owned mock
coordinator/backend only. MCP authentication was enforced. Registration made no
new disk file, preview/normalization stayed RAM, and generation created exactly
one new file directly in private project outputs/images. The original inode and
hash stayed unchanged. Repeated confirm_saved did not copy/delete anything;
outputs survived proxy close. Missing-source failure created no dump/cache file.
All subprocess stdout/stderr went to /dev/null; no runtime log was read or saved.
Pure logging privacy regressions from CP43 separately prove diagnostic redaction.
The injected own encrypted-home volume is not represented as the production
volume: the actual exact production volume was separately proven in CP43.

The harness needed corrections for the supported environment-token contract,
legacy SSE reply parsing and valid source geometry; product binaries did not
change during these tests. The explicit synthetic empty negative prompt avoided
any source-profile keystore dependency. Every failed own run was reaped before
the next fresh case; no user data or real model/GPU was involved.

`20261022T234700Z-998934ba` is now selected by the atomic active-product selector
for the next manual server/UI/proxy start. Actual selected paths, private-storage
flag, packaged proxy and binary hashes were checked after selection. CP42's
immutable package remains unchanged and all four rollback bytes match it.
No running user instance was restarted, so selection does not claim those
processes are already using this storage version. No migration/cleanup occurred.

Manual start after the user's jobs finish: normal server first, matching proxy
(the server API must be reachable for MCP initialization), then UI. Usual proxy
options/backends are preserved; --proxy-only starts only the proxy if backends
already run. The volume must be mounted/writable and XDG_RUNTIME_DIR available.
Rollback selection (not executed):
`python3 tools/publish-video-product.py --rollback comfyui/products/20261022T234700Z-998934ba`.
This selects preserved CP42 server/UI/workers for the next manual start and does
not roll back, migrate or delete private data. OS picker acceptance stays handed
to the user; full Exp/Python and memory parity remain separate open requirements.
