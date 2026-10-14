# Imagine

Experiments toward a local image/video generation backend with a Rust API and
no ComfyUI runtime dependency. The current implementation is a patched native
**stable-diffusion.cpp** engine, build tooling and Qwen Image 2.1 benchmarks.
The Rust service and MiniMax H3 support are design work, not implemented features.

Start with [the design](design-document.md), the
[frozen 1600×896 profile](experiments/qwen-image21-sage/BASELINE.md) and the
[experiment decision register](experiments/qwen-image21-sage/TEST-DECISIONS.md).

## Build the native engine

The tested Linux build uses CMake, Ninja, GCC/G++, CUDA at `/opt/cuda`, and an
RTX 4070 Ti SUPER (SM89). CUDA 13.4 was used for the accepted measurements.
Python 3 is used for setup and benchmark tooling, not native inference.

```sh
python3 native/prepare.py
./build-inference.sh
./stable-diffusion.cpp/build-cuda/bin/sd-server --help
```

The setup script downloads pinned upstream revisions into `stable-diffusion.cpp/`
and applies the patches in `native/patches/`, including the modified ggml kernels.
It never overwrites an existing checkout. An interrupted initial setup may leave
a partial checkout; inspect it before removing it and retrying.
The exact CUTLASS header snapshot used in the experiments is included under
`third_party/cutlass/`; it does not depend on a local Python package cache.

```sh
python3 native/prepare.py --check
```

This verifies upstream revisions, the 29 recorded native source hashes, patch
checksums and 822 CUTLASS headers. It does not certify a new toolchain's generated
binary or image output. Adapt the explicit compiler/CUDA paths and architecture
in `build-inference.sh` for other machines; those machines are outside the
measured baseline scope.

The native checkout is ignored by the outer Git repository. **When changing it,
export both the stable-diffusion.cpp and ggml diffs, including new files, into
`native/patches/` and update the lock/checksums before committing.** Committing
only the outer repository will not capture new unexported native edits.

## Frozen local baseline

The independently preserved, hash-checked baseline executable is intentionally
not distributed through Git. The repository contains its identity, settings,
synthetic prompts and compact measurement evidence. A fresh clone can build the
engine, but the strict frozen launcher requires the exact archived executable;
it does not silently accept a rebuilt development binary as baseline v1.

For a machine with that executable and the model files:

```sh
export QWEN_MODEL_ROOT=/path/to/models
./experiments/qwen-image21-sage/run.sh --dry-run
./experiments/qwen-image21-sage/run.sh --serve
# Or, after stopping the server, replay one first request and two warm requests:
./experiments/qwen-image21-sage/run.sh --run
```

Pillow is needed for the replay client. Model files are not supplied; their
relative filenames are listed in the baseline document. An optional ignored
`experiments/qwen-image21-sage/local-settings.json` can contain
`{"model_root": "/path/to/models"}`; the environment variable takes precedence.
Native development builds can use the same flags in `baseline-environment.json`,
but must be validated separately before being called the frozen baseline.

## Measurements and cleanup

The selected profile saves 590.625 MiB of VAE workspace versus the preceding
control, with 12/12 reference images pixel-identical. Its measured warm request
to PNG mean was 39.219 s. This is not a fresh simultaneous comparison with
ComfyUI and not a universal bit-identity guarantee.

All experiment reports and decision explanations are retained. The compact
accepted evidence is in `experiments/qwen-image21-sage/evidence/1600x896-v1/`.
Old generated images, raw logs, traces and redundant build artifacts were
removed. Historical `runs/` paths in reports are provenance identifiers, not
downloadable assets. Archived one-off harnesses may need those removed inputs
and the original ComfyUI/Kitchen environment; they are not standalone test suites.
The current baseline launcher and its offline checker use the preserved files.

No model weights, private ComfyUI data, credentials, generated images or frozen
binaries belong in Git. The only published prompts are the synthetic benchmark
prompts. Personal filesystem roots have been replaced with portable paths or
placeholders in reports and archived harnesses.

## Third-party notices

The engine and ggml patches are based on the MIT-licensed upstream projects;
their notices are retained in `native/licenses/`. CUTLASS has its own license
and per-file notices in `third_party/cutlass/`. Downloaded upstream dependencies
retain their respective licenses. These notices do not license model weights.
