#!/usr/bin/env bash
# Explicit trained 1MP comparison. Reuses the pinned generated portrait fixture.
set -euo pipefail
parity_root=${IMAGINE_PARITY_ROOT:?Set the isolated development root}
reference_site=${IMAGINE_REFERENCE_SITE:?Set the frozen package source}
comfy_root=${1:?Frozen ComfyUI source checkout}
model_root=${2:?Authorized model root}
prefix=${3:?Fresh run-name prefix}
[[ "$prefix" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"
profile=${4:-$project_root/inference/profiles/qwen21-editing-portrait-1mpx-int8.json}
reports="$parity_root/runtime/$prefix"
mkdir "$reports"
bash inference/tools/launch_qwen21.sh "$model_root" run "$prefix-rust" "$profile" > "$reports/rust.log" 2>&1
cp "$parity_root/runtime/qwen21-run-isolation.json" "$reports/isolation.json"
bash inference/tools/launch_reference.sh inference/tools/reference_qwen_editing.py \
    --site "$reference_site" --comfy "$comfy_root" --model-root "$model_root" \
    --own-root "$parity_root/runtime/$prefix-python" --profile "$profile" > "$reports/python.log" 2>&1
bash inference/tools/launch_reference.sh inference/tools/compare_qwen_first_image.py \
    --site "$reference_site" --rust "$parity_root/runtime/$prefix-rust/first-image" \
    --python "$parity_root/runtime/$prefix-python" \
    --output "$reports/comparison.json" > "$reports/comparison.log"
echo '1MP portrait editing: exact parity passed'
