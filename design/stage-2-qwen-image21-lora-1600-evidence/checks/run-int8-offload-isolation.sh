#!/usr/bin/env bash
set -euo pipefail
cd /home/thhel/ChatGPT/imagine
parity_root=/tmp/imagine-rust-comfy-parity-20261017
reference_site=/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages
export IMAGINE_PARITY_ROOT="$parity_root" IMAGINE_REFERENCE_SITE="$reference_site"
export CARGO_HOME="$parity_root/cargo" CARGO_TARGET_DIR="$parity_root/target" CARGO_INCREMENTAL=0
export LIBTORCH="$reference_site/torch" LIBTORCH_CXX11_ABI=1 CXX=clang++ CARGO_BUILD_JOBS=2 LD_LIBRARY_PATH="$reference_site/torch/lib"
cargo test --locked --offline --manifest-path inference/Cargo.toml --test primitives --no-run --message-format=json > "$parity_root/runtime/lora-test-build.json"
primitive_binary=$(python3 -B -c 'import json,sys;items=[json.loads(line)for line in open(sys.argv[1])];paths=[x["executable"]for x in items if x.get("reason")=="compiler-artifact"and x["target"]["name"]=="primitives"and x.get("executable")];assert len(paths)==1;print(paths[0])' "$parity_root/runtime/lora-test-build.json")
for test_name in qwen_image21_original_int8_offload_inventory_and_lazy_linear_match; do
    python3 -B inference/tools/test_isolation.py --root "$parity_root" --site "$reference_site" \
        --binary "$primitive_binary" --aot-library "$parity_root/triton-aot/libimagine_triton_fp8.so" \
        --aot-fixture "$parity_root/triton-aot/probe.json" \
        --backend-fixture "$parity_root/runtime/backend-contract.json" \
        --int8-library "$parity_root/int8-aot/libimagine_int8_probe.so" --int8-fixture "$parity_root/int8-aot/probe.json" \
        --aimdo-library "$parity_root/aimdo/aimdo.so" --pillow-library "$parity_root/pillow/libimagine_pillow.so" \
        --test-filter "qwen_lora_tests::$test_name"
    cp "$parity_root/runtime/isolation-filtered-result.json" "$parity_root/runtime/$test_name-isolation.json"
done
