#!/usr/bin/env bash
set -euo pipefail
parity_root=${IMAGINE_PARITY_ROOT:?Set the isolated development root}
reference_site=${IMAGINE_REFERENCE_SITE:?Set the locked package source}
export XDG_CACHE_HOME="$parity_root/runtime/cache" TORCH_HOME="$parity_root/runtime/torch"
export TRITON_CACHE_DIR="$parity_root/runtime/triton" CUDA_CACHE_PATH="$parity_root/runtime/cuda"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$reference_site/torch/lib"
exec "$parity_root/venv/bin/python" -S -B "$@"
