#!/usr/bin/env bash
# Explicit trained functional checks; never part of ordinary cargo tests.
set -euo pipefail
parity_root=${IMAGINE_PARITY_ROOT:?Set the isolated development root}
reference_site=${IMAGINE_REFERENCE_SITE:?Set the frozen package source}
comfy_root=${1:?Frozen ComfyUI source checkout}
model_root=${2:?Authorized model root}
prefix=${3:?Fresh run-name prefix}
[[ "$prefix" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"
reports="$parity_root/runtime/$prefix"
mkdir "$reports"
# Functional parity only, one pair per selected path. GPU runs are serial.
for variant in int8 bf16 turbo int8-two-references int8-keep-vision; do
    profile="$project_root/inference/profiles/qwen21-editing-$variant.json"
    name="$prefix-$variant"
    bash inference/tools/launch_qwen21.sh "$model_root" run "$name-rust" "$profile" > "$reports/$variant-rust.log" 2>&1
    cp "$parity_root/runtime/qwen21-run-isolation.json" "$reports/$variant-isolation.json"
    bash inference/tools/launch_reference.sh inference/tools/reference_qwen_editing.py \
        --site "$reference_site" --comfy "$comfy_root" --model-root "$model_root" \
        --own-root "$parity_root/runtime/$name-python" --profile "$profile" > "$reports/$variant-python.log" 2>&1
    bash inference/tools/launch_reference.sh inference/tools/compare_qwen_first_image.py \
        --site "$reference_site" --rust "$parity_root/runtime/$name-rust/first-image" \
        --python "$parity_root/runtime/$name-python" \
        --output "$reports/$variant-comparison.json" > "$reports/$variant-comparison.log"
    echo "$variant: exact editing parity passed"
done
