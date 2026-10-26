#!/usr/bin/env bash
# Rebuild the image worker's pinned Triton compiler tools (triton-opt and
# imagine-triton-llvm) into comfyui/native-toolchain, from the inputs pinned by
# design/stage-1-triton-toolchain-lock.json. The sandboxed step is
# inference/tools/build_triton.sh's; only the persistent input locations
# differ, and the sandbox sees the same paths. tools/restore_native_inputs.py
# accepts the result only if each binary has the compiler profile's SHA-256.
# Fetches the pinned image and Triton source when missing; no service, model
# or GPU is used.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
lock=$project_root/design/stage-1-triton-toolchain-lock.json
toolchain=$project_root/comfyui/native-toolchain
image_id=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["build_image"]["id"])' "$lock")
image_digest=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["build_image"]["manifest_digest"])' "$lock")
llvm=$project_root/inference/target/parity-data/llvm/$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["llvm"]["distribution"])' "$lock")
source=$toolchain/triton-candidate
image_root=$toolchain/manylinux-${image_id:0:8}/rootfs
jobs=${IMAGINE_TRITON_BUILD_JOBS:-2}
mkdir -p "$toolchain"

if [[ ! -e $image_root/bin/bash ]]; then
    podman pull "quay.io/pypa/manylinux_2_28_x86_64@$image_digest"
    [[ $(podman image inspect --format '{{.Id}}' "quay.io/pypa/manylinux_2_28_x86_64@$image_digest") == "$image_id" ]]
    mkdir -p "$image_root"
    container=$(podman create "$image_id" /bin/true)
    podman export "$container" | tar -x -C "$image_root"
    podman rm "$container" >/dev/null
fi
if [[ ! -d $source/.git ]]; then
    commit=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["triton"]["commit"])' "$lock")
    git clone --quiet --filter=blob:none --no-checkout https://github.com/triton-lang/triton "$source"
    git -C "$source" checkout --quiet "$commit"
fi
# Every source and LLVM file must match the toolchain lock before building.
python3 -S -B - "$lock" "$source" "$llvm" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
lock=json.loads(Path(sys.argv[1]).read_text());source=Path(sys.argv[2]);llvm=Path(sys.argv[3])
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==lock['triton']['commit']
for name,expected in lock['triton']['source_file_sha256'].items():assert sha(source/name)==expected,name
for name,expected in lock['llvm']['file_sha256'].items():assert sha(llvm/name)==expected,name
for name,target in lock['llvm']['symlinks'].items():assert str((llvm/name).readlink())==target,name
assert lock['llvm']['commit'] in (llvm/'include/llvm/Support/VCSRevision.h').read_text()
PY
mkdir -p "$toolchain/triton-standalone/make"
python3 -S -B "$project_root/inference/tools/prepare_triton_llvm.py" \
    --source "$source" --output "$toolchain/triton-standalone/generated"
bwrap --unshare-all --die-with-parent --ro-bind "$image_root" / \
    --proc /proc --dev /dev --tmpfs /tmp \
    --ro-bind "$source" /tmp/triton-src \
    --ro-bind "$llvm" /tmp/llvm \
    --bind "$toolchain/triton-standalone/make" /tmp/build --chdir /tmp/build \
    --ro-bind "$toolchain/triton-standalone/generated" /tmp/generated \
    --ro-bind "$project_root/inference/native" /tmp/native \
    --clearenv --setenv PATH /opt/rh/gcc-toolset-14/root/usr/bin:/usr/local/bin:/usr/bin:/bin \
    --setenv LANG C.UTF-8 --setenv JOBS "$jobs" /bin/bash -e -c '
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
mapfile -t declaration_targets < <(sed -n -E "s/^([[:alnum:]_-]*(IncGen|TableGen)):.*$/\1/p" /tmp/build/Makefile)
test "${#declaration_targets[@]}" -gt 0
cmake --build /tmp/build --target "${declaration_targets[@]}" --parallel "$JOBS"
cmake --build /tmp/build --target triton-opt --parallel "$JOBS"
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
echo "$toolchain/triton-standalone/make/bin"
