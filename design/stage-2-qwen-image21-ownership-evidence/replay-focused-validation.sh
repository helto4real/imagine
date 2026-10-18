#!/usr/bin/env bash
set -euo pipefail
parity_root=${IMAGINE_PARITY_ROOT:?Set the isolated development root}
reference_site=${IMAGINE_REFERENCE_SITE:?Set the locked package source}
project_root=/home/thhel/ChatGPT/imagine
cd "$project_root"
export CARGO_INCREMENTAL=0
export CARGO_HOME="$parity_root/cargo" CARGO_TARGET_DIR="$parity_root/target"
export LIBTORCH="$reference_site/torch" LIBTORCH_CXX11_ABI=1 CXX=clang++ CARGO_BUILD_JOBS=2
unset LIBTORCH_USE_PYTORCH LIBTORCH_BYPASS_VERSION_CHECK
export LD_LIBRARY_PATH="$reference_site/torch/lib"
export XDG_CACHE_HOME="$parity_root/runtime/cache" TORCH_HOME="$parity_root/runtime/torch"
export TRITON_CACHE_DIR="$parity_root/runtime/triton" CUDA_CACHE_PATH="$parity_root/runtime/cuda"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IMAGINE_KITCHEN_LIBRARY="$parity_root/kitchen-manylinux/libimagine_kitchen.so"
python3 -B inference/tools/prepare_aimdo.py --site "$reference_site" --output "$parity_root/aimdo"
export IMAGINE_AIMDO_LIBRARY="$parity_root/aimdo/aimdo.so"
export IMAGINE_PRIMITIVE_FIXTURES="$parity_root/runtime/primitives.json"
export IMAGINE_TRITON_PROBE_LIBRARY="$parity_root/triton-aot/libimagine_triton_fp8.so"
export IMAGINE_TRITON_PROBE_FIXTURE="$parity_root/triton-aot/probe.json"
export IMAGINE_INT8_PROBE_LIBRARY="$parity_root/int8-aot/libimagine_int8_probe.so"
export IMAGINE_INT8_PROBE_FIXTURE="$parity_root/int8-aot/probe.json"
export IMAGINE_AUTOTUNING_POLICY_FIXTURE="$parity_root/runtime/autotuning-policy.json"
export IMAGINE_BACKEND_CONTRACT_FIXTURE="$parity_root/runtime/backend-contract.json"
export IMAGINE_TRITON_DRIVER_LIBRARY="$parity_root/triton-driver/libimagine_triton_driver.so"
export IMAGINE_TRITON_FRONTEND_CUBINS="$parity_root/triton-frontend/standalone"
export IMAGINE_TRITON_COMPILER_PROFILE="$parity_root/runtime/triton-compiler-profile.json"
python3 -B inference/tools/prepare_pillow.py --output "$parity_root/pillow"
export IMAGINE_PILLOW_LIBRARY="$parity_root/pillow/libimagine_pillow.so"
cargo fmt --manifest-path inference/Cargo.toml --check
cargo clippy --locked --offline --manifest-path inference/Cargo.toml --all-targets -- -D warnings
cargo test --locked --offline --manifest-path inference/Cargo.toml --test primitives -- --test-threads=1 qwen_image21_original
cargo test --locked --offline --manifest-path inference/Cargo.toml --test primitives --no-run --message-format=json > "$parity_root/runtime/qwen-image21-test-build.json"
primitive_binary=$(python3 -B -c '
import json,sys
items=[json.loads(line) for line in open(sys.argv[1])]
paths=[x["executable"] for x in items if x.get("reason")=="compiler-artifact" and x["target"]["name"]=="primitives" and x.get("executable")]
assert len(paths)==1
print(paths[0])
' "$parity_root/runtime/qwen-image21-test-build.json")
for test_name in \
    model_management_tests::qwen_image21_original_shared_model_manager_order_pressure_weak_owners_and_failures_match \
    model_management_tests::qwen_image21_original_shared_static_clones_patch_switches_and_forced_cpu_patches_match \
    model_management_tests::qwen_image21_original_shared_host_registration_cap_alias_and_owner_release_match \
    model_management_tests::qwen_image21_original_shared_static_multiple_models_partial_eviction_and_reloads_match; do
    python3 -B inference/tools/test_isolation.py --root "$parity_root" --site "$reference_site" \
        --binary "$primitive_binary" --aot-library "$IMAGINE_TRITON_PROBE_LIBRARY" \
        --aot-fixture "$IMAGINE_TRITON_PROBE_FIXTURE" \
        --backend-fixture "$IMAGINE_BACKEND_CONTRACT_FIXTURE" \
        --int8-library "$IMAGINE_INT8_PROBE_LIBRARY" --int8-fixture "$IMAGINE_INT8_PROBE_FIXTURE" \
        --aimdo-library "$IMAGINE_AIMDO_LIBRARY" --pillow-library "$IMAGINE_PILLOW_LIBRARY" \
        --test-filter "$test_name"
    cp "$parity_root/runtime/isolation-filtered-result.json" "$parity_root/runtime/${test_name##*::}-isolation.json"
done
