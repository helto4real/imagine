# Stage 1: compiler modes and callbacks

Five original compiler scenarios, twice each, now pass focused Rust comparison:
normal cold/hit, binary-only cold/hit, always-compile cold/cold, override cold/hit
and dump cold/hit. Tests compare exact compiler digest, metadata, assembly bytes,
all stored source/stage/binary/metadata/override/dump file hashes, listener cache
state and lowering-stage order. Warm JIT calls retain object identity and return
before the lower-level compiler modes or listener, including always-compile.

The native compilation scope preserves these source behaviors:

- Raw source is stored even when only the binary is requested.
- Override/dump managers are constructed before metadata cache lookup and use
  the AST-source hash; cache storage uses the full compiler hash.
- A lowering stage runs before its result is replaced by an explicit override.
- IR overrides are parsed at the original stage boundary. LLIR/PTX/binary
  overrides retain their respective representation and downstream order.
- Cold dumps include original stages and SASS text. The SASS parser transcribes
  the frozen extract/processSassLines/parseCtrl functions, including branch labels
  and instruction-control fields.
- Existing metadata-group insertion order and surviving prior children remain.

The compiler uses the already installed original `cuobjdump` executable. Its hash
is recorded in the Stage 1 toolchain lock and verified profile. Python-free
isolation now admits ten specifically hashed compiler-only files at reference
paths, including this executable; no Python installation or runtime is required.
The actual NVIDIA disassembler warning occurs in both reference and candidate
and is retained in the evidence. No compiler/kernel flags or arithmetic changed.

These native compilation scopes and callback representations are development
interfaces. General invalidating-environment/debug/instrumentation and other
remaining configuration/error contracts are still tracked work. Custom external
Python cache/plugin classes are not used by the frozen base inference path.
Remote/cloud backends remain outside the explicitly local scope.

Original capture 1 is retained as historical reference-only evidence. Capture 2
uses the dynamic FP32 limit signature of the corresponding existing Rust FP8
component and is the paired input for this focused test. Neither capture runs a
GPU kernel. No new timing equality criterion or autotuning change is adopted.

Owned evidence under `/tmp/imagine-rust-comfy-parity-20261017/runtime`:
`compiler-mode-original-1.json`, `compiler-mode-original-2.json`,
`stage1-compiler-modes-original-2.log`, `stage1-stream-and-modes-focused.log`,
`cuobjdump-tool.json` and `stage1-modes-profile-update.log`.
This focused result does not replace the last complete unfiltered checkpoint or
claim completion of Stage 1.
