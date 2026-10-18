#!/usr/bin/env bash
# Explicit development-only trained runs. Never invoked by ordinary tests.
set -euo pipefail
parity_root=${IMAGINE_PARITY_ROOT:?Set the isolated development root}
reference_site=${IMAGINE_REFERENCE_SITE:?Set the frozen package source}
comfy_root=${1:?Frozen ComfyUI source checkout}
model_root=${2:?Authorized model root}
prefix=${3:?Fresh run-name prefix}
[[ "$prefix" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"
driver_profile=${4:-$project_root/inference/profiles/qwen21-first-int8.json}
reports="$parity_root/runtime/$prefix"
mkdir "$reports"
rust_run() {
    bash inference/tools/launch_qwen21.sh "$model_root" run "$prefix-rust-$1" "$driver_profile" > "$reports/rust-$1.log" 2>&1
    cp "$parity_root/runtime/qwen21-run-isolation.json" "$reports/rust-$1-isolation.json"
}
python_run() {
    bash inference/tools/launch_reference.sh inference/tools/reference_qwen_first_image.py \
        --site "$reference_site" --comfy "$comfy_root" --model-root "$model_root" \
        --own-root "$parity_root/runtime/$prefix-python-$1" --profile "$driver_profile" > "$reports/python-$1.log" 2>&1
}
# Alternate order; fresh processes/full component loads every time. No cache
# flushing, clock control, autotuning override or unrelated process inspection.
for iteration in 1 2 3; do
    if [[ "$iteration" == 2 ]]; then
        rust_run "$iteration"
        python_run "$iteration"
    else
        python_run "$iteration"
        rust_run "$iteration"
    fi
    bash inference/tools/launch_reference.sh inference/tools/compare_qwen_first_image.py \
        --site "$reference_site" --rust "$parity_root/runtime/$prefix-rust-$iteration/first-image" \
        --python "$parity_root/runtime/$prefix-python-$iteration" \
        --output "$reports/comparison-$iteration.json" > "$reports/comparison-$iteration.log"
    echo "Matched pair $iteration: exact parity passed"
done
