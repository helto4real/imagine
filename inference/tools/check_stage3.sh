#!/usr/bin/env bash
set -euo pipefail
parity_root=${IMAGINE_PARITY_ROOT:?Set the isolated development root}
reference_site=${IMAGINE_REFERENCE_SITE:?Set the locked package source}
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"
export CARGO_HOME="$parity_root/cargo"
export CARGO_TARGET_DIR="${IMAGINE_STAGE3_TARGET_DIR:-$parity_root/target-stage3}"
export CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2
export LIBTORCH="$reference_site/torch" LIBTORCH_CXX11_ABI=1 CXX=clang++
export IMAGINE_CUDA_INCLUDE="$parity_root/cuda13/nvidia/cu13/include"
export LD_LIBRARY_PATH="$reference_site/torch/lib"
unset LIBTORCH_USE_PYTORCH LIBTORCH_BYPASS_VERSION_CHECK
python="$parity_root/venv/bin/python"
output=${1:?Pass a fresh evidence directory}
mkdir -p "$parity_root/stage3-validation"
cargo fmt --manifest-path inference/Cargo.toml --check
cargo clippy --manifest-path inference/Cargo.toml --locked --offline --all-targets -- -D warnings
cargo test --manifest-path inference/Cargo.toml --locked --offline --lib --test primitives \
    --no-run --message-format=json > "$parity_root/stage3-validation/build.jsonl"
primitive_binary=$("$python" -S -B - "$parity_root/stage3-validation/build.jsonl" <<'PY'
import json,sys
from pathlib import Path
items=[json.loads(line)for line in Path(sys.argv[1]).read_text().splitlines()]
binaries=[x['executable']for x in items if x.get('reason')=='compiler-artifact'and x['target']['name']=='primitives'and x.get('executable')]
assert len(binaries)==1
print(binaries[0])
PY
)
"$python" -S -B inference/tools/validate_stage3.py --root "$parity_root" \
    --site "$reference_site" --binary "$primitive_binary" --output "$output"
metadata_binary=$("$python" -S -B - "$parity_root/stage3-validation/build.jsonl" <<'PY'
import json,sys
from pathlib import Path
items=[json.loads(line)for line in Path(sys.argv[1]).read_text().splitlines()]
binaries=[x['executable']for x in items if x.get('reason')=='compiler-artifact'and x['target']['kind']==['lib']and x.get('executable')]
assert len(binaries)==1
print(binaries[0])
PY
)
"$python" -S -B inference/tools/validate_stage3.py --root "$parity_root" \
    --site "$reference_site" --binary "$metadata_binary" --output "$output/size-representation" \
    --test context_windows::shapes::tests::original_size_metadata_representation_matches_torch_tuple_subtype
