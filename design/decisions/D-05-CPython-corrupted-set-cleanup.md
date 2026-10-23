# D-05-CPYTHON-CORRUPTED-SET-CLEANUP

Date: 2026-10-20. Status: **proposal; awaiting Tomas's explicit approval**.

## Approval requested in plain terms

Approve a single boundary in the locked CPython set implementation: when the
already reproduced reentrant comparison bug makes the set's `used` count larger
than its actual active entries, Rust clear/destruction must stay within the
allocated table. Do not reproduce the C code's resulting out-of-bounds reads,
possible unrelated decrefs, crash or other undefined cleanup outcome.

Keep exact original behavior for valid sets and the defined observable prefix
of this reproduced corrupt state. Keep its original member order, counters,
callback history and overwritten-entry ownership behavior before cleanup.
On native clear, publish the empty set before releasing the actual valid
entries in table order, retain the source pop cursor, and do not touch memory
outside that table. Destruction releases only actual valid entries. Preserve
the already reproduced overwritten-reference leak; this proposal does not
silently repair the reentrant add algorithm or change its returned result.

This would be a specific non-exact acceptance boundary for **clear and
destruction of this demonstrated corrupt state only**. It is not a general
permission to change valid set behavior, other corrupt-state operations,
source algorithms, callback errors, allocation policy, GC, precision, RNG,
samplers, quantization, model memory policy or autotuning. Stop for any further
required deviation. Whole Stage 3 remains unfinished.

## Concrete evidence

The existing own reentrant fixture inserts a key into a dummy slot during an
outer key equality comparison. Original `set_add_entry` then overwrites that
same slot while incrementing `used` again. At the observed prefix:

- `len(set)` / `used` = **5**;
- actual iterator members = **4**, ordered **[5, 1, 2, 3]**;
- capacity = **8**, the inline smalltable.

[The bounded diagnostic](../stage-3-evidence/set-corrupt-counter-diagnostic.json)
reproduces those exact facts in **three fresh CPython 3.12.13 processes**.
Each uses only seven own metadata objects, a 512 MiB address-space cap,
no GPU, no model weights and no application runtime data. Each deliberately
exits with `os._exit` **before clear or Python object finalization**, avoiding
the invalid source cleanup. No crash was executed or claimed.
The reproducible controller is
`inference/tools/diagnose_set_corrupt_counter.py`.

The retained, SHA256-verified CPython source is
`comfyui/stage3-core-20261020/sources/cpython/Objects/setobject.c`:

- `set_clear_internal`, lines 396–442: a small table is copied into a fixed
  `setentry small_copy[PySet_MINSIZE]`; the release loop tests only `used > 0`,
  with no table-end check.
- `set_dealloc`, lines 482–504: the release loop likewise tests `used > 0`
  without a table-end check.

With five expected active entries but only four present, traversing all eight
valid slots leaves the local counter equal to one. The next slot read lies
outside the valid table. This conclusion is from the original source and the
measured table/count mismatch; it does not depend on observing a crash.

## Correction to earlier evidence

The earlier **96-history** general-set fixture generator continued to `clear`
after this one corrupt prefix to observe a leaked helper reference. That tail
can execute undefined C behavior. Its historically observed output and passed
native replay **must not be treated as exact acceptance evidence for that
cleanup**, even though the value happened to match. The other valid histories
and the prefix before unsafe cleanup are separate evidence.

Do not rerun that original corrupt cleanup. After approval, adjust this one
oracle tail to retain the defined prefix separately, exit its reference child
before invalid cleanup, and validate the proposed bounded native cleanup as an
explicitly approved criterion. Preserve the old executable/evidence as
historical observations with this limitation; do not overwrite them or label
undefined output as a deterministic reference.

The native implementation already uses a table-bounded owner release, so the
unsafe C cleanup is not emulated today. This newly identified difference is
now explicit and blocks treating that branch as accepted or continuing the
dependent full registry/loader gate without the requested decision.

## Why approval is required

Tomas required explicit approval for deviations from ComfyUI or the agreed
design. The [design, Section 14.2](../rust-comfy-parity-engine.md#142-acceptance-gates)
states: “All exact checks pass, or every specific non-exact acceptance criterion
has explicit approval.” No previous identity/set-order or raw-padding approval
covers this invalid cleanup. This document records a proposal, not approval.
