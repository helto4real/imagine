# Isolated inference dependencies

The default Rust backend builds and loads LibTorch from
`<project>/.venv-inference/lib/python3.12/site-packages/torch`. Its NVIDIA
libraries are installed beside Torch. ComfyUI's Python environment is not used
by this product path. Python packages distribute the native libraries; Python
does not run inference.

`inference-environment-lock.json` pins CPython 3.12.13 on Linux x86_64 and 30
exact wheel files, including URLs and SHA-256. Versions come from the installed
Torch dependency closure in ComfyUI `7cebca52227c63b4180ceed448cbd7ad22903b6d`,
after its requirements installation. This excludes ComfyUI's frontend, custom
nodes and unrelated packages/conflicts. Pip is also pinned for restoration.

With an existing matching Python interpreter:

```sh
python3 tools/prepare_inference_environment.py --python python3.12
# Recreate without network when the wheel cache is already available:
python3 tools/prepare_inference_environment.py --python python3.12 --offline
# Verify an existing environment without installing or downloading:
python3 tools/prepare_inference_environment.py --verify
```

The environment and `comfyui/inference-wheels/` cache are ignored by Git.
Keep a backup of that wheel cache for offline restoration. An existing
environment is verified, never upgraded or overwritten. A failed new install
is removed. Missing/incompatible Python, changed wheels, package mismatches or
changed locked native files/headers stop restoration; no hashes are rewritten.

`build-lock.json` retains the original native file/header hashes and frozen
Kitchen, AIMDO, Pillow and Triton artifacts. Only the LibTorch/NVIDIA environment
paths moved, and the existing packaged NCCL library is now explicitly hashed
alongside the retained system-library pins. It matches ComfyUI's installed file.
Some frozen Kitchen objects retain the old build-time ComfyUI
RUNPATH; the explicit product `LD_LIBRARY_PATH` selects the isolated libraries.
Validate loader resolution with ComfyUI's environment hidden before changing
this deployment. System libraries, the GPU driver and model directories remain
separate dependencies. The CUDA headers and the Triton compiler profile's files
are restored by `tools/restore_native_inputs.py` from `native-inputs-lock.json`;
see `docs/PRODUCT-BUILDS.md`. The optional `--backend comfy` and development reference
tools still have their explicit ComfyUI dependencies.

After restoration, use `./build-product.sh` for build-only validation. Neither
restoration nor that command starts services or runs a model.
