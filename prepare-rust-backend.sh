#!/usr/bin/env bash
# Build/deploy the Rust worker using already prepared, locked local dependencies.
# No Python, model loading, service startup or downloads.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
parity_root=${IMAGINE_RUST_DEPENDENCIES:-$project_root/comfyui/moss-20261018}
reference_site=${IMAGINE_RUST_SITE:-/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages}
for artifact in "$parity_root/kitchen-manylinux/libimagine_kitchen.so" "$parity_root/aimdo/aimdo.so" "$parity_root/pillow/libimagine_pillow.so" "$parity_root/runtime/triton-compiler-profile.json"; do
    [[ -f $artifact ]] || { echo "Missing prepared Rust dependency: $artifact" >&2; exit 1; }
done
export CARGO_HOME="$parity_root/cargo" CARGO_TARGET_DIR="$parity_root/target"
export LIBTORCH="$reference_site/torch" LIBTORCH_CXX11_ABI=1 CXX=clang++ CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0
export IMAGINE_CUDA_INCLUDE="$parity_root/cuda13/nvidia/cu13/include"
unset LIBTORCH_USE_PYTORCH LIBTORCH_BYPASS_VERSION_CHECK
cargo build --manifest-path "$project_root/inference/Cargo.toml" --locked --offline --bin imagine-worker
# The server writes the manifest without loading the worker or CUDA.
CARGO_HOME="${IMAGINE_SERVER_CARGO_HOME:-$HOME/.cargo}" CARGO_TARGET_DIR="$project_root/target" cargo build --manifest-path "$project_root/Cargo.toml" --release --locked --offline --bin imagine-server
"$project_root/target/release/imagine-server" --prepare-rust-runtime "$parity_root" "$reference_site" "$project_root/comfyui/runtime-current.json"
