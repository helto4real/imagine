# Stage 1: compiler environment and warm-cache ordering

The locked compiler header defines 37 cache-invalidating environment names.
All names are independently recognized by the original installed native helper.
The candidate stores that fixed list as Rust data and reads only those named
configuration variables when a compiler memory miss reaches compilation.

629 synthetic original cases and 17 Python boolean-knob cases pass exact
comparison. The two original conversions remain distinct: cache normalization
recognizes on/true/1 and off/false/0, while Python boolean knobs also recognize
y/yes. Empty invalidating values are omitted. Other strings retain their bytes.

The initial four-step original JIT sequence passes exact candidate comparison:
compile under value A, change to B with a warm JIT hit, clear JIT state and compile
under B, then clear and restore A from disk. Comparison includes same-object
identity, exact compiler-event presence/cache-hit state, digest, assembly,
metadata and metadata JSON file hash. Warm memory reuse precedes invalidating
configuration reads; the lower-level disk digest includes their normalized values.

All eight steps, including four additional Unicode/quoting steps, pass exact candidate comparison. Environment
fields use the existing source-compatible JSON string formatter, preserving
ASCII escaping and original metadata insertion order. Native compiler tools
continue to receive their ordinary process environment; an explicit `_env_vars`
equivalent changes cache/metadata inputs without inventing tool environment
changes, as in the original compiler API.

This is focused evidence, not a new full Stage 1 checkpoint. Broader configuration,
dynamic scalar contracts and final isolation/restoration continue. No external
runtime logs, prompts or model material are read, and no autotuning policy or
numerical tolerance changes.

Owned evidence under `/tmp/imagine-rust-comfy-parity-20261017/runtime`:
`triton-environment-original.json`, `stage1-triton-environment-original.log`,
`stage1-triton-environment-focused.log`, `stage1-triton-environment-unicode-focused.log` and the separately retained initial
capture. The header hash and exact name list are in the Stage 1 toolchain lock.
