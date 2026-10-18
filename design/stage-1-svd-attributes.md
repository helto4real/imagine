# Stage 1: SVDQuant layout tensor attributes

Date: 2026-10-18. Full **95 crate tests and 87 unfiltered Python-free primitives
pass**. Stage 1 remains incomplete.

Layout fields now retain their ordinary tensor objects across executable calls.
Read-only storage snapshots share those objects; actual copies retain only
objects whose tensor identity the original operation preserves. Metadata-only
ATen TensorImpl identity crosses the native ABI. No storage bytes, GPU process
metadata or synchronization is needed for that identity.

This closes a concrete cache gap: the layout previously recreated an ordinary
argument object for `proj_up` on each call, losing the source tensor's
`_ck_natural_lora_up` attribute. The backend itself already implemented the
original cache; the layout now passes the retained object to it.

The frozen source oracle covers FP16/BF16, direct CUDA enabled/disabled, and
separate versus genuinely shared scale/proj_up tensor objects: eight scenarios,
nine state transitions each, with 64 numerical executions. It compares exact
outputs, field object identity, cached tensor bytes and cache identity before
and after calls. Repetition, logical transpose and already-contiguous storage
retain the cache. Clone, dtype copy, detach and CPU transfer follow original
field-copy behavior. Multiple fields referencing one tensor retain one object.

The original cache deliberately checks metadata rather than tensor contents.
An in-place update to tile-packed proj_up retains an already-created natural
copy; its resulting arithmetic is reproduced. This is original behavior, not
an invalidation repair or a numerical relaxation. The unfused LoRA environment
flag is set only inside the scoped test and restored afterward; engine defaults
and all autotuning remain unchanged.

The fixture contains 23,752 physical records: 23,737 ordinary, five approved Sol
and ten approved null-export records. The separate 3,572 prior controlled
unwritten-storage combinations remain unchanged. All 32 original CUDA object
hashes and four Stage 0 lock hashes remain unchanged.

Own logs under `/tmp/imagine-rust-comfy-parity-20261017/runtime`:
`stage1-svd-attributes-reference.log`, `stage1-layout-attributes-build.log`,
`stage1-layout-attributes-native-build.log`, `stage1-svd-attributes-focused.log`
and `stage1-svd-attributes-full.log`. See [checkpoint](stage-1-svd-attributes-checkpoint.json).
No model/product run, ordinary runtime data, process inspection, service change,
commit, push or new approval request was involved.

General Triton, Sage/chunked public wrappers, broader dynamic and FFI semantics,
asynchronous ownership and final restoration remain obligations in the
[completion ledger](stage-1-completion.md).
