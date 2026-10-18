#!/usr/bin/env bash
set -euo pipefail
parity_root=${IMAGINE_PARITY_ROOT:?Set the isolated development root}
reference_site=${IMAGINE_REFERENCE_SITE:?Set the frozen package source}
model_root=${1:?Supply the authorized local model root}
mode=${2:-preflight}
run_name=${3:-qwen21-first-image}
case "$mode" in preflight|load-check|run) ;; *) echo 'Expected preflight, load-check or run' >&2; exit 2 ;; esac
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"
driver_profile=${4:-$project_root/inference/profiles/qwen21-first-int8.json}
export CARGO_HOME="$parity_root/cargo" CARGO_TARGET_DIR="$parity_root/target" CARGO_INCREMENTAL=0
export LIBTORCH="$reference_site/torch" LIBTORCH_CXX11_ABI=1 CXX=clang++ CARGO_BUILD_JOBS=2
unset LIBTORCH_USE_PYTORCH LIBTORCH_BYPASS_VERSION_CHECK
export LD_LIBRARY_PATH="$reference_site/torch/lib"
cargo fmt --manifest-path inference/Cargo.toml --check
cargo build --locked --offline --manifest-path inference/Cargo.toml --bin qwen21-first-image
python3 -B inference/tools/prepare_aimdo.py --site "$reference_site" --output "$parity_root/aimdo"
extra=()
if python3 -B -c 'import json,sys;sys.exit("reference_resolution" not in json.load(open(sys.argv[1])))' "$driver_profile"; then
    python3 -B inference/tools/prepare_pillow.py --output "$parity_root/pillow"
    extra+=(--pillow-library "$parity_root/pillow/libimagine_pillow.so")
fi
exec python3 -B inference/tools/test_isolation.py --root "$parity_root" --site "$reference_site" \
    --binary "$parity_root/target/debug/qwen21-first-image" \
    --aot-library "$parity_root/triton-aot/libimagine_triton_fp8.so" --aot-fixture "$parity_root/triton-aot/probe.json" \
    --int8-library "$parity_root/int8-aot/libimagine_int8_probe.so" --int8-fixture "$parity_root/int8-aot/probe.json" \
    --backend-fixture "$parity_root/runtime/backend-contract.json" --aimdo-library "$parity_root/aimdo/aimdo.so" \
    "${extra[@]}" --qwen-driver-profile "$driver_profile" \
    --qwen-model-root "$model_root" --qwen-mode "$mode" --qwen-run-name "$run_name"
