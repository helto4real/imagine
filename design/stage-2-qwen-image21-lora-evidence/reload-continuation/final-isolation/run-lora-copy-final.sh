#!/usr/bin/env bash
set -euo pipefail
cd /home/thhel/ChatGPT/imagine
export CARGO_HOME=/tmp/imagine-rust-comfy-parity-20261017/cargo CARGO_TARGET_DIR=/tmp/imagine-rust-comfy-parity-20261017/target CARGO_INCREMENTAL=0
export LIBTORCH=/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages/torch LIBTORCH_CXX11_ABI=1 CXX=clang++ CARGO_BUILD_JOBS=2 LD_LIBRARY_PATH=/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages/torch/lib
cargo build --locked --offline --manifest-path inference/Cargo.toml --bin qwen21-first-image > /tmp/imagine-rust-comfy-parity-20261017/runtime/lora-linear-build-final.txt 2>&1
bash /tmp/run-qwen-lora-isolation.sh > /tmp/imagine-rust-comfy-parity-20261017/runtime/lora-linear-isolation-final.txt 2>&1
