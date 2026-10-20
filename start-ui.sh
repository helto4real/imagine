#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ -f "$ROOT/comfyui/active-product.sh" ]]; then
    source "$ROOT/comfyui/active-product.sh"
    sha256sum --check --status <<< "$IMAGINE_PRODUCT_UI_SHA256  $IMAGINE_PRODUCT_UI"
    exec "$IMAGINE_PRODUCT_UI" --project "$ROOT" "$@"
fi
cargo build --release --locked -p imagine-ui
exec "$ROOT/target/release/imagine-ui" --project "$ROOT" "$@"
