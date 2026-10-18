# Stage 1: standalone Triton compiler investigation

Date: 2026-10-17. Status: source/toolchain inputs matched; original pipeline
oracle and all six standalone compiler comparisons passed, including complete
byte-identical cubins. Stage 1 subsequently paused at the separate
[empty-width ConvRot decision](decisions/D-05-INT8-ConvRot-empty-width.md);
Tomas explicitly approved its scoped criterion and Stage 1 resumed.
**This is not a general runtime JIT backend or Stage 1 completion.**

Tomas authorized continued Stage 1 work under the approved design, with explicit
approval required for deviations. No deviation or substitute has been adopted.
The existing D-05 approvals cover only their recorded INT8/Triton autotuning
criteria and have not been broadened to compiler output or other operations.

## Matched inputs

The source and build dependencies remain local in the isolated development root.
The newly fetched inputs have their own [toolchain lock](stage-1-triton-toolchain-lock.json).
The Stage 0 reference files are unchanged.

| Input | Evidence |
| --- | --- |
| Triton 3.7.1 source | Official `v3.7.1`, commit `f797708c0626e5f9840ca5b0a98790e2c7cb09ad`; 1,466 tracked file hashes |
| Installed Python compiler/backend files | All 115 installed files match the corresponding release source byte-for-byte; installed files also checked against Stage 0 |
| Frozen `libtriton.so` | Original SHA256 `e1e17b42f65d04185099e8512cdedf3989355e7e7d3f429022c5302e4833f3d7`; original extension has unresolved Python C API symbols and is not loaded into the standalone process |
| LLVM | `1f126a6dea50d185c0781743a667390037ae88bd`, 23.0.0git; upstream `llvm-hash.txt`, downloaded revision header and original extension's embedded LLVM build path agree |
| LLVM build distribution | `llvm-1f126a6d-almalinux-x64-1`; archive SHA256 `33acf0e06def309bd2728ddaf9b40c71302491e84732d14dfcde70453daffe4c`; 5,271 file hashes and 13 symlink targets locked |
| Native build image | Existing pinned manylinux 2.28 image `ca4cca0a2e1591ee30adffe584ff5144fecebde14c4ad0130f54e260878dc426` |
| Triton assembler | Original bundled `ptxas` 12.8.93, SHA256 `c960a4f238b17d5c5d3c01ad2bbc1ebd2c5aecc459cb4d223bff10b45f9b8fca`; current bytes match the Stage 0 locked wheel RECORD |
| Libdevice | Original bundled bitcode, SHA256 `5c2fae37c86e68c3a38605a95f512d7d12d5f3db986310be47f57304aa72a5ee`; also verified against the locked RECORD |

The CUDA 13 toolkit used for Kitchen's native build is not substituted for
Triton's own CUDA 12.8 assembler. The captured SM89 reference emits PTX 8.7.
Matching version/source inputs does not alone establish native compiler parity.

## Python-independent build boundary

Upstream CMake supports `TRITON_BUILD_PYTHON_MODULE=OFF`. The build retains
the original compiler libraries, passes and LLVM. NVIDIA and AMD dialects are
built because the original standalone `triton-opt` registers both; this does not
change the initial NVIDIA target or add a HIP runtime requirement to inference.
Proton's compiler dialect dependencies are distinct from enabling its profiler.

`build_triton.sh` uses the same immutable manylinux image root in a Bubblewrap
mount/network namespace. Source and dependencies are read-only; only the own
build directory is writable. This avoids Podman's additional writable VFS layer.
No ordinary runtime directory is mounted. Build-time Python utilities validate
sources and generate bindings; no interpreter is part of the compiler candidate.

`prepare_triton_llvm.py` preserves the original helper prefix and lambda bodies
from `python/src/llvm.cc` and `third_party/nvidia/triton_nvidia.cc`. It removes
Python binding includes/types and exposes the same default LLVM optimizer,
data layout, short-pointer/FTZ configuration and assembly generation through a
small development CLI. Original PTX version/target/debug-flag substitutions are
also performed inside that native CLI. The host test tool only invokes tools and
compares outputs; it does not transform the generated assembly. The CLI does not
substitute `opt`/`llc` default pipelines.
The FP8 probe explicitly rejects external library dependencies; full libdevice
linking, options and production compiler ABI remain future work, not an adopted
restriction on the engine.

## Original compiler oracle

`reference_triton_pipeline.py` uses six synthetic FP8 specializations: float32,
float16 and bfloat16 input to E4M3FN/E5M2, on the initial SM89 target. It records
the original pass-manager pipeline strings, optimized TTIR, TTGIR, LLVM-dialect
MLIR, optimized LLVM IR, PTX, cubin and launch metadata.

Before comparing a new compiler, the tool reruns the original backend's TTGIR
and LLVM stages from saved TTIR. All six reproduce the original JIT's TTGIR and
LLVM IR exactly, ignoring only a terminal text newline. Original option values
are checked against the original metadata. This is an oracle self-check, not
standalone compiler equivalence or a numerical tolerance.

The standalone comparison passed for all six cases: TTGIR, LLVM-dialect MLIR,
optimized LLVM IR and PTX match exactly (only terminal text newlines are ignored),
and every byte of each complete cubin matches. No debug section was ignored.
The actual compiler tools ran in an empty mount/network namespace containing
selected ELF dependencies, tools, synthetic IR and one verified Kitchen source
copy as compiler data. No Python interpreter/library, ComfyUI installation or
GPU device was available there. This proves the recorded compiler boundary;
it does not claim a new Rust production kernel-launch path or model execution.

The assembler records the original source file's timestamp and size in DWARF.
An initial run with that file absent produced the same machine-code sections
and identical PTX, but different `.debug_line`/`.rel.debug_line` metadata. A
byte-verified own copy of the frozen Kitchen kernel source, preserving its source
metadata, is mounted at its original debug pathname. All cubin bytes then match.
That plaintext source is compiler data; it is not an interpreter or a ComfyUI
installation. No original environment is mounted as a directory tree.

The optimized-TTIR frontend boundary is explicit: no Rust/Python-independent
frontend, general signature/specialization rules or complete kernel coverage
has yet been demonstrated by this probe. Saved specializations do not define a
new supported-shape limit or replace the remaining general JIT obligation.

## Failed attempts retained

- Initial Podman run failed creating its writable VFS layer with disk quota
  exceeded. Reuse of the same image root through Bubblewrap avoids that copy.
- Initial root-level mountpoints could not be created in the read-only image;
  mountpoints now live under the namespace's writable temporary filesystem.
- Ninja was unavailable in the image; its existing Make generator is used.
- Upstream target dependencies omitted generated CGA and operation-interface
  headers. Generate the 45 original declaration targets before compiler objects;
  no source, pass or generated declaration is patched.
- Linking the LLVM helper initially used an archive path, but upstream builds
  the required pass as an OBJECT library. The bridge links that original object.
- Link attempts exhausted the isolated build's disk quota. Only the own disposable
  pip download cache and Rust incremental build cache were removed; source locks,
  dependencies, test binaries/results and runtime autotuning caches were retained.
  Explicit temporary/object paths and the linker's `--no-keep-memory` mode avoid
  unnecessary transient use; no GPU codegen option changed. The final build passed.
- Repeated CMake configuration recursively rediscovered its generated lit target
  directories, eventually exceeding filename limits. Recreate only those generated
  lit rules before configuration; all test sources, compiler objects and actual
  validation results remain unchanged.
- The first compiler-isolation launch lacked the directory behind a loader-path
  symlink. Creating the empty `/usr/lib64` mountpoint fixed the namespace setup.
- The initial complete-cubin comparison exposed the source timestamp/size issue
  described above. The final comparison retains strict complete-file equality.
- Oracle development initially omitted the original `module.context` ownership
  attribute, then incorrectly reconstructed typed options from JSON lists.
  Restoring the original ownership and typed compile options fixed those tooling
  errors; all six original oracle cases now pass.

Compiler artifacts/logs remain under `/tmp/imagine-rust-comfy-parity-20261017`.
Successful standalone outcomes, hashes and retained failed attempts are recorded in
[stage-1-evidence.json](stage-1-evidence.json), with no claim of unrun validation.
No model weights, ordinary runtime data, service changes, commit or push are
involved in this compiler investigation. No compiler exception has been adopted.
The later empty-width ConvRot decision is separately approved for its scoped
controlled-storage criterion and bounded investigation.
