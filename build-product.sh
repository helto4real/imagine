#!/usr/bin/env bash
# Build the same artifacts as normal startup, without starting any service.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
for component in server ui proxy; do
    python3 "$ROOT/tools/product_build.py" prepare "$component"
done
