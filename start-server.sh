#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export IMAGINE_COORDINATOR_SOCKET="${IMAGINE_COORDINATOR_SOCKET:-$ROOT/.imagine/gpu.sock}"
backend=rust
custom_runtime=false
arguments=("$@")
for ((index=0; index<${#arguments[@]}; index++)); do
    case ${arguments[index]} in
        --backend) backend=${arguments[index+1]:-} ;;
        --rust-runtime) custom_runtime=true ;;
        --help) exec env CARGO_HOME="${IMAGINE_SERVER_CARGO_HOME:-$HOME/.cargo}" CARGO_TARGET_DIR="$ROOT/target" cargo run --release --locked --offline --bin imagine-server -- --help ;;
    esac
done
if [[ $backend == rust && $custom_runtime == false ]]; then
    "$ROOT/prepare-rust-backend.sh"
fi
# Incremental build; Cargo.lock pins the dependencies. No native rebuild or GPU load here.
CARGO_HOME="${IMAGINE_SERVER_CARGO_HOME:-$HOME/.cargo}" CARGO_TARGET_DIR="$ROOT/target" cargo build --release --locked --offline --bin imagine-server
exec "$ROOT/target/release/imagine-server" --project "$ROOT" "$@"
