#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export IMAGINE_COORDINATOR_SOCKET="${IMAGINE_COORDINATOR_SOCKET:-$ROOT/.imagine/gpu.sock}"
# Incremental build; Cargo.lock pins the dependencies. No native rebuild or GPU load here.
cargo build --release --locked --bin imagine-server
exec "$ROOT/target/release/imagine-server" --project "$ROOT" "$@"
