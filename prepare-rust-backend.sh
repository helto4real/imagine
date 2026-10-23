#!/usr/bin/env bash
# Prepare the main server and both workers; no service startup or model loading.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
exec python3 "$ROOT/tools/product_build.py" prepare server
