#!/usr/bin/env bash
set -euo pipefail
parity_root=${IMAGINE_PARITY_ROOT:?Set the isolated development root}
reference_site=${IMAGINE_REFERENCE_SITE:?Set the frozen reference package source}
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
image_root=$(python3 -S -B "$project_root/inference/tools/prepare_triton.py" --root "$parity_root" --site "$reference_site")
mkdir -p "$parity_root/triton-standalone/make"
python3 -S -B - "$parity_root" <<'PY'
from pathlib import Path
import shutil,sys
# Upstream's lit registration recursively scans its binary tree, including
# prior generated lit target directories. Recreate only those generated rules;
# keep every test source, compiler object, executable and validation result.
directory=Path(sys.argv[1])/'triton-standalone/make/test/CMakeFiles'
for path in directory.glob('check-triton-lit-tests*'):
    assert path.parent==directory and not path.is_symlink()
    if path.is_dir():shutil.rmtree(path)
PY
python3 -S -B "$project_root/inference/tools/prepare_triton_llvm.py" \
    --source "$parity_root/sources/triton-candidate" --output "$parity_root/triton-standalone/generated"
# Same pinned manylinux image root, without VFS's extra writable-layer copy.
# Only the candidate build directory is writable. No network or ordinary data.
bwrap --unshare-all --die-with-parent --ro-bind "$image_root" / \
    --proc /proc --dev /dev --tmpfs /tmp \
    --ro-bind "$parity_root/sources/triton-candidate" /tmp/triton-src \
    --ro-bind "$parity_root/llvm/llvm-1f126a6d-almalinux-x64-1" /tmp/llvm \
    --bind "$parity_root/triton-standalone/make" /tmp/build --chdir /tmp/build \
    --ro-bind "$parity_root/triton-standalone/generated" /tmp/generated \
    --ro-bind "$project_root/inference/native" /tmp/native \
    --clearenv --setenv PATH /opt/rh/gcc-toolset-14/root/usr/bin:/usr/local/bin:/usr/bin:/bin \
    --setenv LANG C.UTF-8 /bin/bash -e -c '
cmake -G "Unix Makefiles" -S /tmp/triton-src -B /tmp/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/tmp/llvm/bin/clang-23 -DCMAKE_CXX_COMPILER=/tmp/llvm/bin/clang++ \
    -DCMAKE_C_FLAGS=--gcc-toolchain=/opt/rh/gcc-toolset-14/root/usr \
    -DCMAKE_CXX_FLAGS=--gcc-toolchain=/opt/rh/gcc-toolset-14/root/usr \
    -DLLVM_INCLUDE_DIRS=/tmp/llvm/include -DLLVM_LIBRARY_DIR=/tmp/llvm/lib \
    -DMLIR_DIR=/tmp/llvm/lib/cmake/mlir -DLLD_DIR=/tmp/llvm/lib/cmake/lld \
    -DTRITON_BUILD_PYTHON_MODULE=OFF -DTRITON_BUILD_PROTON=OFF -DTRITON_BUILD_UT=OFF \
    -DTRITON_BUILD_WITH_CCACHE=OFF -DTRITON_OFFLINE_BUILD=ON -DTRITON_EXT_ENABLED=ON \
    "-DTRITON_CODEGEN_BACKENDS=nvidia;amd" -DTRITON_VERSION=3.7.1
# Upstream target dependencies omit generated interfaces used by other targets.
# Generate declarations first; do not patch a pass, declaration or build flag.
mapfile -t declaration_targets < <(sed -n -E "s/^([[:alnum:]_-]*(IncGen|TableGen)):.*$/\1/p" /tmp/build/Makefile)
test "${#declaration_targets[@]}" -gt 0
cmake --build /tmp/build --target "${declaration_targets[@]}" --parallel 2
cmake --build /tmp/build --target triton-opt --parallel 2
# The original Python module enables C++ exceptions/RTTI for its bindings;
# compiler pass libraries retain upstreams own no-exceptions/no-RTTI flags.
read -r -a llvm_cxxflags <<< "$(/tmp/llvm/bin/llvm-config --cxxflags)"
read -r -a llvm_libraries <<< "$(/tmp/llvm/bin/llvm-config --libs --link-static all)"
read -r -a llvm_system <<< "$(/tmp/llvm/bin/llvm-config --system-libs --link-static)"
/tmp/llvm/bin/clang++ --gcc-toolchain=/opt/rh/gcc-toolset-14/root/usr \
    "${llvm_cxxflags[@]}" -fexceptions -frtti -O3 -DNDEBUG \
    -I/tmp/triton-src/include -I/tmp/build/include -I/tmp/generated \
    -c /tmp/native/triton_llvm_driver.cpp -o /tmp/build/triton_llvm_driver.o
export TMPDIR=/tmp/build
/tmp/llvm/bin/clang++ --gcc-toolchain=/opt/rh/gcc-toolset-14/root/usr \
    /tmp/build/triton_llvm_driver.o \
    /tmp/build/lib/Target/LLVMIR/CMakeFiles/TritonLLVMIR.dir/LLVMIRBreakPhiStruct.cpp.o \
    -L/tmp/llvm/lib -Wl,--no-keep-memory -Wl,--start-group "${llvm_libraries[@]}" -Wl,--end-group \
    "${llvm_system[@]}" -o /tmp/build/bin/imagine-triton-llvm
'
