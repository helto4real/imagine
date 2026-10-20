# P02 storage foundation

This is the isolated project/library part of P02, exported by
`imagine_server::video::{persistence,library}`. It does not register HTTP routes,
change session cleanup, integrate UI, decode media, run models or touch the
scheduler/worker protocol. All IO requires caller-selected existing directories
or caller-supplied file handles. No real project or runtime data is used in tests.

## Ownership and lifecycle

`ProjectSession` owns a `ProjectDocument`, revision, resource handles and separate
persistence status. `update` stages and verifies a complete candidate before
changing session state; `edit` reuses P01 typed edits. Both autosave synchronously
after first Save. A later UI/controller may debounce calls without moving storage
ownership into the UI. Mutations require exclusive access; stale document revisions
fail without changes. A save failure keeps the updated session draft and its files
available for Retry, with `NotSaved` and the previous `saved_revision`.

Before first Save the application supplies a `SessionStaging` bound to its own
session directory. Copies are streamed into mode-0600 files immediately unlinked
from that directory; only owned descriptors keep them alive. No recovery manifest,
hidden history, global index or backup directory is created. Dropping the handles
releases the data. This is ordinary unlinked storage, not encryption/secure erasure.

First `save_to` requires an existing empty dedicated destination. It binds that
root; later saves never recreate missing roots or choose a fallback path.
`open` validates the manifest and every referenced resource. `save_as` writes an
independent copy with a new project ID and changes the active location only after
success. A failed Save as retains the original binding. `copy_resource` streams a
verified resource to a caller-owned writer without exposing its internal path.

## Format and transaction

```text
selected-project/
  .imagine-lock
  project.json
  media/<sha256>
  takes/<sha256>
  continuity/<sha256>
```

`project.json` is a strict version-1 `ProjectDocument`: P01 project/scene/asset IDs,
all attached completed Takes and provenance, versioned resources, and optional
library-origin identity/revision. Resource records distinguish reference versions,
Take video/audio and opaque continuation artifacts. Rendered media and artifact
bytes are copied; model weights and derived previews/caches are not included.
No runtime session IDs, queue/runs, idempotency register or job history is modeled.

Current and historical reference versions must exist; all stored Takes need video
resources. Candidate/Accepted/Rejected/Outdated are retained, including a selected
Take later marked Outdated. Continuation bytes have explicit format/version,
model identity and geometry; storage integrity does not imply H3 compatibility.
Full plan/result publication and continuation interpretation remain P03/P05 work.

The private `bundle` module supplies shared filesystem mechanics for projects and
libraries. It pins root device/inode, rejects symlink roots/components and child
files, uses directory-relative `openat`/`renameat` with no-follow flags, and rechecks
root identity. Destination directories/files use 0700/0600 on the current Unix
platform. The empty persistent lock file coordinates writers; an expected manifest
hash plus revision checks rejects stale writes. Hard-linked input bundle files are
rejected. Blob paths are derived from validated SHA-256 and a closed category enum;
manifest-provided arbitrary relative/absolute paths are not followed.

Each transaction verifies/streams immutable blobs, syncs files, then atomically
replaces and syncs the manifest in the same root. Only new files from a failed
pre-commit attempt are cleaned up; previous manifest/resources are preserved.
There is no background garbage collection or deletion of old unreferenced blobs;
explicit project cleanup remains a later integration concern. No code in this
module can delete another project, a library root, an original or an export.

A directory-sync failure after rename is reported as
`storage_durability_uncertain`: the atomic rename may already have committed a
valid new manifest, but durability is not confirmed. The bound writer retains
its new expected hash so Retry is possible; a first-save failure of this kind
keeps the chosen binding and reports NotSaved. A failed Save as never switches
the active original; an uncertain target can be explicitly opened to recover it.
This distinguishes a pre-commit write failure from a post-rename durability error.

## Separate library

`LibrarySession::create/open` explicitly accesses a chosen library folder, with
`library.json` and its own resource copies/lock. Entries contain either a complete
project snapshot or a `CharacterDocument` (description/tags/visual IDs/voice
profile/reference IDs). Explicit Save to Library creates or revision-checks an
entry; project autosave never calls library saving. Entry listing is in memory
from the opened catalog, with no global/persistent search database.

`copy_project` returns a new unsaved `ProjectSession` with independent staged
media, a fresh project ID, and origin IDs/revision. `copy_character` returns an
independent staged `CharacterCopy` for later application to a project by the
reference editor. Copies remain usable after source/library deletion. Neither
library updates nor project edits propagate back to prior copies.

## Verification and remaining integration

Tests use temporary roots and synthetic byte payloads, not real codecs or model
artifacts. They cover reopen after session deletion, autosave/Save as, all Take
statuses and historical references, hashes/symlinks, a changed/disconnected root,
concurrent reads/writers, pre-commit injected disk-full, Retry, and independent
project/character/voice library copies. The injected write failure is synthetic;
it is not a physical full-disk or power-loss experiment.

```sh
cargo test --locked --lib video::persistence
cargo test --locked --test video_persistence
cargo test --locked --test video_contract
cargo test --locked --workspace
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo build --locked --bin imagine-server
python3 scripts/test_server_privacy.py
```

Still pending: opaque location-ID API binding, streaming HTTP asset upload/ranges,
media format inspection/decoding, fresh runtime asset-ID registration on Open,
UI Save/Open/close handling and debouncing, explicit resource cleanup, and Take
publication from inference. The existing global session cleanup is unchanged;
its later integration must never put saved roots inside disposable session data.

Verified 2026-10-20: 12 integration tests plus two pre-commit failure/locking unit
tests passed. The product workspace passed 221 tests with two existing tests
ignored; the two synthetic binary privacy tests also passed. Clippy with
`-D warnings`, workspace formatting, and the server build passed. Workspace and
binary privacy tests used test-only escalation for temporary loopback listeners.
No manual UI, real codec, H3/model, physical full-disk or power-loss test was run.
The separate inference workspace was not built or changed by this slice.
