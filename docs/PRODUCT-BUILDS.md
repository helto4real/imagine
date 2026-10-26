# Product builds from main

Normal entry points are `start-server.sh`, `start-ui.sh` and `start-proxy.sh`.
They all use `tools/product_build.py`. The checkout must be on **local `main`**.
There is no automatic pull, checkout, commit, package selection or service restart.
Uncommitted product changes are included; the build report marks them `dirty`.
A remote `origin/main` update has no effect until deliberately integrated locally.

## What follows main, and what stays pinned

| Layer | Source used by normal startup |
| --- | --- |
| API, UI, proxy, shared product contracts and Comfy worker scripts | Current main checkout, including local edits |
| Rust image/H3 worker adapters | Exact `product_files` allowlist in `native/product/build-lock.json`, overlaid from main |
| Rust inference core, memory/allocator/offload code, native bindings, Cargo dependency lock and vendor code | `source_base` Git archive plus `core_files` restored from `core_revision` and hash-checked `core_patches` |
| Kitchen, Aimdo, Pillow, LibTorch, CUDA libraries and headers, Triton compiler tools | Existing local artifacts with hashes in the product build lock |
| System C++ compiler, GPU driver and C runtime libraries (`build.system`: `clang++` and files under `/usr/`) | The installed versions; a difference from the lock is reported and recorded in `build.json` |
| Optional stable-diffusion.cpp backend and Qwen native profile | Existing `native/upstream-lock.json` and `native/qwen-default/source-lock.json`; not rebuilt by product launchers |

An edit under `inference/` does **not** automatically enter production. Only
allowlisted product files follow main. Core files are restored from the pinned
Git revision even when their working copies have changed. New adapter files
must be explicitly added to the allowlist. Moving generic inference code into
that allowlist to bypass the core lock is not an acceptable update procedure.

The initial lock preserves the previously used frozen base and explicit core
additions. Its `core_revision` is the source restoration revision. H3's
`identity.core_revision` is the separate, existing deployment provenance field;
it must not be mistaken for the complete source tree used to build a worker.

## Startup and build-only commands

```sh
./start-server.sh
./start-ui.sh
./start-proxy.sh --log

# Prepare all three components without starting services, workers or models:
./build-product.sh
# Prepare only the server, both native workers and their combined runtime:
./prepare-rust-backend.sh
# Prepare one component; prints its completed artifact directory:
python3 tools/product_build.py prepare ui
```

Each start runs Cargo with `--locked --offline`, reusing its compilation cache.
All Rust code — the native image and H3 workers, server, UI and proxy — is
built with the **release** profile (`--release`, optimized, without debug
information), as set by `build.profile` in the build lock. CUDA, LibTorch and
the Triton artifacts are prebuilt and only verified. A shared
build lock serializes preparation. Rust/Cargo versions and native headers
are checked against the native lock before worker builds. CUDA libraries are
verified, not rebuilt. Python 3.12+ orchestrates preparation only; it is not an
inference runtime for the Rust backend.

Operating-system inputs follow distribution upgrades instead: the system
`clang++` and the libraries under `/usr/` (C runtime, `libcuda`, NCCL) named in
`build.system`. When one differs from the lock, preparation prints
`system input changed since the build lock`, uses the installed version or hash
for that build's `runtime.json`, and records the locked and installed values
under `system_changes` in `build.json`. The lock keeps the values the accepted
baseline was built with. A system change is not GPU evidence; compare against
the frozen baseline when quality or speed matters.

Prepared dependencies and the Git objects named by the lock must already exist.
There are no downloads. The machine-specific prepared paths are in the lock's
`build`, `image` and `h3` sections. Old dated target directories remain compiler
caches only; they are no longer independent source owners or launch selectors.

LibTorch and its packaged NVIDIA libraries/headers live in the project-local
`.venv-inference`, independent of ComfyUI's `.venv`. Restore it explicitly with
`python3 tools/prepare_inference_environment.py`; see
[the dependency lock and restore procedure](../native/product/README.md).
Normal launchers only verify these dependencies and never install packages.

Successful preparation copies executables into a fresh directory under
`comfyui/main-build/releases/`. Native worker snapshots are reused by binary
hash under `comfyui/main-build/workers/`. The server receives that release's
`runtime.json`, containing **both image and H3** workers. Its existing Rust
manifest validator runs during preparation without starting a model or CUDA.

`build.json` records the commit, dirty state, source hashes, complete resolved
build lock, tool versions, worker snapshot path and copied artifact hashes.
The native source cache also has a complete source hash receipt. These are build
metadata, not application logs. To reproduce local edits later, commit/preserve
the source separately; hashes are evidence, not a backup of uncommitted files.

The helper checks source inputs again after the build. A changed source tree,
failed Cargo command, incompatible protocol, missing artifact, changed pinned
library or modified native source cache stops preparation. **There is no fallback
to an old binary or package.** Failed releases are discarded; existing processes
and executable inodes remain untouched.

Hash verification caches use file identity, size, mtime and ctime. Changes force
rehashing; an expected hash is never silently replaced with the new value.
The cache is local build metadata, not a security boundary against its owner.

## Runtime behavior and overrides

The launchers preserve private storage and default both coordinator variables to
`$XDG_RUNTIME_DIR/imagine/gpu.sock`. `XDG_RUNTIME_DIR` must be absolute for a
normal launch. Explicit coordinator environment variables remain supported;
server and proxy must point to the same coordinator. No alternative GPU queue
or unreserved execution is introduced.

Server/UI arguments and proxy options, including `--log` and `--proxy-only`,
are passed through. The exact command `start-proxy.sh --stop` invokes the source
supervisor without compiling, so a compiler/dependency failure cannot prevent
stopping the stack. It does not require the current branch to be main.

`start-server.sh --help`, an explicit `--backend native|comfy`, and an explicit
`--rust-runtime PATH` skip default native preparation. An explicit runtime is a
caller-managed deployment and does not receive the default lock's guarantees.
The native backend remains tied to its existing executable/profile locks.
Help and build-only preparation do not launch services or models.

## Explicit rollback

Normal startup **never reads or sources `comfyui/active-product.sh`**. Historical
packages remain untouched. To run a particular existing package explicitly:

```sh
./start-server.sh --product-package /absolute/path/to/package
./start-ui.sh --product-package /absolute/path/to/package
./start-proxy.sh --product-package /absolute/path/to/package --log
```

The package must contain `package.json` and the component artifacts with matching
hashes. Server rollback also verifies its declared runtime dependency files.
The package's recorded private-storage mode is retained. Each component must be
selected explicitly; no persistent global selector changes. Use a matching set
of components, and stop/restart them deliberately. A package launch is an explicit
exception to main-source startup and skips compilation.

`tools/publish-video-product.py` is retained for historical package tooling.
Its `--activate`/`--rollback` selectors no longer affect the normal launchers.
Do not use those commands as the current deployment procedure.

## Procedure for future agents

1. Read this document and `native/product/build-lock.json` before changing build
   or launch behavior. Ordinary product changes belong on main; retain the single
   shared preparation path and the existing privacy/coordinator behavior.
2. For adapter-only work, change main's allowlisted files. If a new file is needed,
   review its responsibility before adding its exact path to `product_files`.
   A wire version change also needs an explicit matching identity/profile update.
3. For a core/CUDA/toolchain/dependency update, treat it as a separate native
   change. Preserve the accepted baseline and evidence, follow repository native
   instructions, validate the change, then update the pinned source revision,
   individual source/patch hashes and affected dependency/tool/header hashes.
   System compiler/driver/library upgrades are recorded, not blocked (see above);
   do not widen `build.system` to project-owned inputs, disable verification or
   recalculate the entire lock merely to make startup succeed.
4. Never hand-edit the generated native source cache. Remove only the named
   `comfyui/main-build/native-source-...` directory **and its sibling `.json`
   receipt** when a source-cache rebuild is necessary, with no build in progress.
   Preparation restores it from Git. Do not remove frozen controls, library
   artifacts or worker/release directories potentially used by running services.
5. Run the focused checks below and build-only preparation. GPU comparisons and
   service restarts are separate work; neither happens as part of these checks.
   Keep static/build results separate from live model evidence in the report.

## Restoring native inputs

Pinned inputs live under the project, never in `/tmp`, which is cleared at boot.
`python3 tools/restore_native_inputs.py` restores what can be restored
exactly, and verifies everything it covers:

- the CUDA 13.0 headers in `build.cuda_include`, from the five exact wheels in
  `native/product/native-inputs-lock.json` (cached in `comfyui/inference-wheels/`);
- missing files named by the image worker's Triton compiler profile: source
  copies from the original files it names, the Triton driver from the product
  artifact store, and `triton-opt`/`imagine-triton-llvm` from a rebuild by
  `tools/build_triton_compiler_tools.sh`.

Every restored file and tree must have its pinned SHA-256; a mismatch stops the
restore and nothing is rewritten. The compiler rebuild uses the pinned image,
Triton source and LLVM from `design/stage-1-triton-toolchain-lock.json` in the
original sandbox, and reproduces the pinned binaries byte for byte. A dangling
link under `comfyui/` on the way to a restored file becomes a real directory.

```sh
python3 tools/restore_native_inputs.py
# Only when it reports missing compiler tools (takes a while):
tools/build_triton_compiler_tools.sh && python3 tools/restore_native_inputs.py
```

## Checks

```sh
python3 -m unittest discover -s tools -p test_product_build.py -v
python3 -m unittest discover -s tools -p test_restore_native_inputs.py -v
bash -n start-server.sh start-ui.sh start-proxy.sh prepare-rust-backend.sh build-product.sh
./build-product.sh
```

`tools/prepare-h3-product-source.py` remains a compatibility entry point for
source preparation. It now prints the verified source directory; callers must
use that output instead of hardcoding the historical `comfyui/h3-build/source`.
