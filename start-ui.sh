#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
cargo build --release --locked -p imagine-ui
exec "$ROOT/target/release/imagine-ui" --project "$ROOT" "$@"
