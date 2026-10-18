#!/usr/bin/env bash
set -euo pipefail
parity_root=${IMAGINE_PARITY_ROOT:?Set IMAGINE_PARITY_ROOT to the isolated development directory}
reference_site=${IMAGINE_REFERENCE_SITE:?Set IMAGINE_REFERENCE_SITE to the locked package source}
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
image=ca4cca0a2e1591ee30adffe584ff5144fecebde14c4ad0130f54e260878dc426
python3 -S -B "$project_root/inference/tools/verify_scipy_beta.py"
python3 -B "$project_root/inference/tools/prepare_kitchen.py" "$parity_root/sources" "$parity_root/generated"
# NVIDIA's wheels use lib/ whereas the compiler's Unix layout expects lib64/.
# Only the isolated toolchain is adjusted; package/header bytes stay unchanged.
if [[ ! -e "$parity_root/cuda13/nvidia/cu13/lib64" ]]; then
    ln -s lib "$parity_root/cuda13/nvidia/cu13/lib64"
fi
if [[ ! -e "$parity_root/cuda13/nvidia/cu13/lib/libcudart.so" ]]; then
    ln -s libcudart.so.13 "$parity_root/cuda13/nvidia/cu13/lib/libcudart.so"
fi
mkdir -p "$parity_root/kitchen-manylinux"
# Reuse the immutable pinned image root without copying a VFS writable layer.
# The same compiler, build directory, source paths and flags are retained.
image_root=$(python3 -S -B - "$parity_root" "$image" <<'PYIMAGE'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);image_id=sys.argv[2]
images=json.loads((root/'containers/vfs-images/images.json').read_text())
item=next(x for x in images if x['id']==image_id)
assert item['digest']=='sha256:1e7b8460ad8fb3fb0fbe1cd818845159ac896b4fd17452d56e892c291f3b97a8'
layer=item['layer'];assert len(layer)==64 and all(c in '0123456789abcdef' for c in layer)
path=root/'containers/vfs/dir'/layer
assert (path/'bin/bash').is_file()
print(path)
PYIMAGE
)
bwrap --unshare-all --die-with-parent --tmpfs / \
    --ro-bind "$image_root/usr" /usr --ro-bind "$image_root/opt" /opt \
    --ro-bind "$image_root/etc" /etc --ro-bind "$image_root/bin" /bin \
    --ro-bind "$image_root/lib" /lib --ro-bind "$image_root/lib64" /lib64 \
    --proc /proc --dev /dev --tmpfs /tmp \
    --ro-bind "$project_root/inference/native" /src \
    --ro-bind "$parity_root/sources" "$parity_root/sources" \
    --ro-bind "$parity_root/cuda13" "$parity_root/cuda13" \
    --ro-bind "$parity_root/generated" "$parity_root/generated" \
    --bind "$parity_root/kitchen-manylinux" /build \
    --ro-bind "$reference_site" "$reference_site" \
    --clearenv --setenv PATH /usr/local/bin:/usr/bin:/bin \
    --setenv LANG C.UTF-8 --setenv PARITY_ROOT "$parity_root" \
    --setenv REFERENCE_SITE "$reference_site" /bin/bash -e -c '
cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-14/root/usr/bin/g++ \
    -DCMAKE_CUDA_COMPILER="$PARITY_ROOT/cuda13/nvidia/cu13/bin/nvcc" \
    -DCMAKE_CUDA_HOST_COMPILER=/opt/rh/gcc-toolset-14/root/usr/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DGENERATED_DIR="$PARITY_ROOT/generated" -DREFERENCE_SOURCES="$PARITY_ROOT/sources" \
    -DLIBTORCH="$REFERENCE_SITE/torch" -DFROZEN_CUDA_INCLUDE="$REFERENCE_SITE/nvidia/cu13/include"
cmake --build /build --parallel 2
'
