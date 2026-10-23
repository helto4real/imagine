#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# Local prepared compiler/dependency environment; no model or service execution.
source /home/thhel/.codex/worktrees/h3-product-worker/imagine/comfyui/h3-product-20261020/product-env.sh
export CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
export CARGO_TARGET_DIR="$ROOT/comfyui/h3-main-20261020/target-product"
cargo build --locked --offline --workspace
python3 "$ROOT/tools/prepare-h3-product-source.py"
# Native D-05 work is intentionally excluded: only H3 is overlaid on the frozen base.
export CARGO_TARGET_DIR="$IMAGINE_H3_PRODUCT_ROOT/target-product"
unset CARGO_PROFILE_DEV_DEBUG CARGO_PROFILE_TEST_DEBUG
cargo build --locked --offline --manifest-path "$ROOT/comfyui/h3-build/source/inference/Cargo.toml" --bin imagine-h3-worker
mkdir -p "$ROOT/comfyui/h3-build/bin"
# Atomic replacement preserves any already-running worker's executable inode.
install -m 755 "$CARGO_TARGET_DIR/debug/imagine-h3-worker" "$ROOT/comfyui/h3-build/bin/.imagine-h3-worker.new"
mv -f "$ROOT/comfyui/h3-build/bin/.imagine-h3-worker.new" "$ROOT/comfyui/h3-build/bin/imagine-h3-worker"
package="$(python3 "$ROOT/tools/publish-video-product.py" \
    --server "$ROOT/comfyui/h3-main-20261020/target-product/debug/imagine-server" \
    --ui "$ROOT/comfyui/h3-main-20261020/target-product/debug/imagine-ui" \
    --worker "$ROOT/comfyui/h3-build/bin/imagine-h3-worker")"
python3 "$ROOT/tools/publish-video-product.py" --activate "$package"
printf '%s\n' 'Built and selected immutable server/UI/H3 artifacts for the next manual start. No service was started; rollback is saved in the package.'
