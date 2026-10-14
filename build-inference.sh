#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$project_dir/stable-diffusion.cpp"
build_dir="$source_dir/build-cuda"

if [[ ! -f "$source_dir/CMakeLists.txt" ]]; then
  printf '%s\n' 'Run python3 native/prepare.py first.' >&2
  exit 1
fi

cmake -S "$source_dir" -B "$build_dir" -G Ninja \
  -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DGGML_CUDA_INT8_GATE_UP_CUTLASS_PATH="$project_dir/third_party/cutlass/include" \
  -DSD_CUDA=ON \
  -DSD_WEBP=ON \
  -DSD_WEBM=ON \
  -DSD_SERVER_BUILD_FRONTEND=OFF

cmake --build "$build_dir" --target sd-cli sd-server \
  --parallel "${SD_BUILD_JOBS:-2}"
