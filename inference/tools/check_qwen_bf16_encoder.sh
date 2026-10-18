#!/usr/bin/env bash
# Explicit trained precision control, never invoked by ordinary regression tests.
set -euo pipefail
parity_root=${IMAGINE_PARITY_ROOT:?Set the isolated development root}
reference_site=${IMAGINE_REFERENCE_SITE:?Set the frozen package source}
comfy_root=${1:?Frozen ComfyUI source checkout}
model_root=${2:?Authorized model root}
prefix=${3:?Fresh run-name prefix}
variant=${4:?int8, bf16 or turbo}
[[ "$prefix" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
case "$variant" in int8|bf16|turbo) ;; *) exit 2 ;; esac
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"
profile="$project_root/inference/profiles/qwen21-lora-$variant-bf16-encoder.json"
# Test-launch guard only. No weights are read until the host has enough room
# for the checkpoint mapping, an independent encoder copy and small components.
python3 -B - "$model_root" "$profile" <<'PY'
import json,sys
from pathlib import Path
p=json.loads(Path(sys.argv[2]).read_text())
size=(Path(sys.argv[1])/p['checkpoints']['text']).stat().st_size
available=int(next(line.split()[1]for line in Path('/proc/meminfo').read_text().splitlines()if line.startswith('MemAvailable:')))*1024
required=2*size+4*1024**3
assert available>=required,f'Isolated test deferred: available host RAM {available}, guard requires {required}'
print(f'Host RAM guard passed: {available} available, {required} required')
PY
reports="$parity_root/runtime/$prefix"
mkdir "$reports"
mkdir "$parity_root/runtime/$prefix-rust"
touch "$parity_root/runtime/$prefix-rust/memory-observation-enabled"
bash inference/tools/launch_qwen21.sh "$model_root" run "$prefix-rust" "$profile" > "$reports/rust.log" 2>&1
cp "$parity_root/runtime/qwen21-run-isolation.json" "$reports/isolation.json"
bash inference/tools/launch_reference.sh inference/tools/reference_qwen_editing.py \
    --site "$reference_site" --comfy "$comfy_root" --model-root "$model_root" \
    --own-root "$parity_root/runtime/$prefix-python" --profile "$profile" \
    --memory-observation > "$reports/python.log" 2>&1
comparison_status=0
bash inference/tools/launch_reference.sh inference/tools/compare_qwen_first_image.py \
    --site "$reference_site" --rust "$parity_root/runtime/$prefix-rust/first-image" \
    --python "$parity_root/runtime/$prefix-python" --output "$reports/comparison.json" \
    > "$reports/comparison.log" 2>&1 || comparison_status=$?
python3 -B - "$reports" "$comparison_status" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads((p/'comparison.json').read_text())
assert all(v.get('exact_bytes',v.get('exact'))and v.get('all_finite',True)for v in r['metrics'].values()),'Numerical parity failed'
assert not r['diagnostic_memory_plan'],'Live placement overrides forbidden in this control'
(p/'strict-comparison-status.json').write_text(json.dumps(dict(exit_status=int(sys.argv[2]),numerical_exact=True,paired_component_states_exact=r['component_memory']['all_observed_fields_exact']),indent=2)+'\n')
PY
for engine in rust python; do
    own="$parity_root/runtime/$prefix-$engine"
    states="$own/component-memory.json"
    if [[ "$engine" == rust ]]; then own="$own/first-image"; states="$own/memory-manager.json"; fi
    python3 -S -B inference/tools/reference_qwen_memory_metadata.py \
        --comfy "$comfy_root" --observations "$own/memory-services.json" \
        --observed-states "$states" --output "$reports/$engine-metadata-policy.json"
done
echo 'Numerical tensors/PNG exact; each observed placement matches frozen policy on its own measured inputs. See strict-comparison-status.json for paired placement equality.'
