#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# Incremental build; Cargo.lock pins the dependencies. No native rebuild or GPU load here.
cargo build --release --locked --bin imagine-server
exec "$ROOT/target/release/imagine-server" --project "$ROOT" "$@"
