#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 -m venv "$root/.venv-privacy"
"$root/.venv-privacy/bin/python" -m pip install -r "$root/scripts/privacy-requirements.txt"
