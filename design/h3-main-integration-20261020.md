# H3 preparation for local main — 2026-10-20

The narrow H3 product delivery is based on frozen inference commit
`40ff91114a6d71782aa1a79a76df6940546570bb`, directly after main base
`901c847929cc9a68d046910fefdda1dba30e8214`. The product evidence and scope
remain in `h3-product-worker-product-integration-20261020.md`.

## Preserved work and merge boundary

Before source changes, all 357 pending main files and 82 pending H3 files were
copied and checksum-verified in
`/home/thhel/Documents/Codex/2026-10-20/task/h3-merge-backup-20261020T144547Z`.
This includes tracked modifications and untracked work, with patches and refs.
Six overlapping files were three-way merged in a separate preview without
text conflicts. Main's pending later-core/D-05 work is not part of this H3
commit and must remain uncommitted and unchanged in behavior. The manual test
root is excluded from staging. No broad add/reset/clean is permitted.

## New opt-in runtime package; not activated

The old test manifest referenced a shared image-worker output that was rebuilt.
Main's current image manifest passed all six artifact hashes. Its image worker
SHA-256 is `6a73523a3b477baebfb0eb7030f890f12475000e601fa97feefdf1bca946df23`.
This retains the existing artifact identity, not a new numerical acceptance or
an assertion that the current dirty core is fully validated.

Eight hash-verified project artifacts have been copied into content-addressed
paths below `/home/thhel/ChatGPT/imagine/comfyui/h3-main-20261020/artifacts/`.
All 31 external library artifacts remain independently hash-checked. The
existing compiler/frontend directories and system library search paths remain
external prerequisites. Model files are neither copied nor loaded.

New combined manifest:
`/home/thhel/ChatGPT/imagine/comfyui/h3-main-20261020/product-runtime.json`
SHA-256: `366d40e00bd58368ce0decbcecd778cf6be9acb8c96ad08da79aa0b61c2688a0`.
Input manifest hashes, path mapping and output identity are recorded in the
sibling `provenance.json`. The H3 worker bytes remain
`31294b34e1472409fa6a3c7a829c2fb4088fc4028e83799e3c603118c4e122a0`.
The canonical dependency-map digest was recomputed only to identify relocated,
byte-identical dependencies. No expected content hash was replaced to admit drift.

The already verified product server's `--prepare-h3-runtime` command validated
both input deployments and assembled this new manifest. The active
`comfyui/runtime-current.json` was not changed. No worker/service, CUDA or model
was started. Do not execute the generic prepare launcher to mutate this snapshot.

## Completed local integration and CPU validation

Normal configured OpenPGP signing succeeded. Product commit
`cd0ef53be69dd539ef0ffdf0cb03f4c1c5835fc7` was signature-verified and brought
onto local main with `git merge --ff-only`, including the standalone H3 freeze.
No unsigned commit, stash, reset, clean, broad staging, push or product restart
was used. Six overlapping code paths and the additional ignore rule were
combined with the existing local work. All 357 original pending files remain:
350 byte-identical and seven with exact preservation proofs. Their original
28 tracked modifications remain unstaged; unrelated later-core work is not
part of either H3 commit.

The combined product working tree passed 240 CPU tests, zero failed, three
explicit live/GPU tests ignored. Workspace all-target Clippy with warnings
denied, formatting and workspace build passed. These tests include preserved
local image/UI improvements, not only the bare committed tree. The separate
inference workspace and its pending later-core/D-05 tests were not executed.
The newly built product server independently verified and reassembled the
snapshot into identical manifest bytes, without starting a service or worker.
Full command/log hashes are in `h3-main-cpu-validation-20261020.json`.

## Remaining activation gates

The active image-only manifest is unchanged. The new H3 manifest requires
explicit selection at a later authorized launch. Product activation is a separate step:
real image → H3 → image recovery, cancellation/Ticket lifecycle and native GUI/
actual AV synchronization must be qualified with the selected deployment.
No GPU test, product restart, push or unsigned commit is part of this preparation.
