#!/usr/bin/env bash
# Development-only namespace for the preserved pre-upgrade reference.
# Its canonical paths keep original source/JIT identities and ELF lookups.
set -euo pipefail
frozen_comfy=${IMAGINE_FROZEN_COMFY_ROOT:?Set the independently preserved ComfyUI checkout including its .venv}
parity_root=${IMAGINE_PARITY_ROOT:?Set the isolated development root}
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mapfile -t reference < <(python3 -S -B - "$project_root/design/reference-local/reference-lock.json" <<'PY'
import json,sys
lock=json.load(open(sys.argv[1]))
print(lock['roots']['comfy'])
print(lock['comfy_revision'])
PY
)
[[ ${#reference[@]} == 2 && $(git -C "$frozen_comfy" rev-parse HEAD) == "${reference[1]}" ]]
[[ -x "$frozen_comfy/.venv/bin/python" && $# -gt 0 ]]
mkdir -p "$parity_root/runtime/frozen-tmp"
exec bwrap --die-with-parent --unshare-net --ro-bind / / \
  --bind "$project_root" "$project_root" --bind "$parity_root" "$parity_root" \
  --dev-bind /dev /dev --proc /proc \
  --ro-bind "$frozen_comfy" "${reference[0]}" \
  --setenv IMAGINE_REFERENCE_DIR "$project_root/design/reference-local" \
  --setenv TMPDIR "$parity_root/runtime/frozen-tmp" \
  --setenv PYTHONDONTWRITEBYTECODE 1 "$@"
