#!/usr/bin/env bash
set -euo pipefail
parity_root=${IMAGINE_PARITY_ROOT:?Set the isolated development root}
reference_site=${IMAGINE_REFERENCE_SITE:?Set the locked package source}
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"
export CARGO_HOME="$parity_root/cargo" CARGO_TARGET_DIR="$parity_root/target"
export LIBTORCH="$reference_site/torch" LIBTORCH_CXX11_ABI=1 CXX=clang++ CARGO_BUILD_JOBS=2
export IMAGINE_CUDA_INCLUDE="$parity_root/cuda13/nvidia/cu13/include"
unset LIBTORCH_USE_PYTORCH LIBTORCH_BYPASS_VERSION_CHECK
export LD_LIBRARY_PATH="$reference_site/torch/lib"
export XDG_CACHE_HOME="$parity_root/runtime/cache" TORCH_HOME="$parity_root/runtime/torch"
export TRITON_CACHE_DIR="$parity_root/runtime/triton" CUDA_CACHE_PATH="$parity_root/runtime/cuda"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IMAGINE_KITCHEN_LIBRARY="$parity_root/kitchen-manylinux/libimagine_kitchen.so"
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
python3 -S -B inference/tools/test_reference_checks.py
"$parity_root/venv/bin/python" -S -B inference/tools/check_generated_sources.py \
    --site "$reference_site" --root "$parity_root" --output "$parity_root/runtime/generated-source-restoration.json"
"$parity_root/venv/bin/python" -S -B inference/tools/generate_triton_scalar_sources.py \
    --site "$reference_site" --output inference/src/triton_scalar_sources.rs --check
python3 -S -B inference/tools/generate_triton_environment_names.py \
    --source "$project_root/inference/native/triton" --output inference/src/triton_environment_names.rs --check
"$parity_root/venv/bin/python" -S -B inference/tools/generate_native_argument_signatures.py \
    --site "$reference_site" --output inference/src/native_argument_signatures.rs --check
bash inference/tools/build_native.sh
# These local tools/inputs must already be prepared from the separate locked
# compiler toolchain; this script never downloads or silently substitutes them.
"$parity_root/venv/bin/python" -S -B inference/tools/reference_triton_frontend.py \
    --site "$reference_site" --output "$parity_root/triton-frontend"
cargo build --locked --offline --manifest-path inference/Cargo.toml --example triton_fp8_frontend
python3 -S -B inference/tools/build_triton_driver.py --root "$parity_root" --site "$reference_site"
python3 -S -B inference/tools/test_triton_frontend.py --root "$parity_root" --compile-selected --raw-frontend
"$parity_root/venv/bin/python" -S -B inference/tools/reference_triton_extern.py \
    --site "$reference_site" --output "$parity_root/triton-extern-full"
python3 -S -B inference/tools/test_triton_extern.py --root "$parity_root" --site "$reference_site"
"$parity_root/venv/bin/python" -S -B inference/tools/reference_triton_rowwise_frontend.py \
    --site "$reference_site" --output "$parity_root/triton-rowwise-frontend"
python3 -S -B inference/tools/test_triton_frontend.py --root "$parity_root" --family rowwise --compile-selected --raw-frontend
"$parity_root/venv/bin/python" -S -B inference/tools/reference_triton_adaln_frontend.py \
    --site "$reference_site" --output "$parity_root/triton-adaln-frontend"
python3 -S -B inference/tools/test_triton_frontend.py --root "$parity_root" --family adaln --compile-selected --raw-frontend
"$parity_root/venv/bin/python" -S -B inference/tools/reference_triton_adaln_frontend.py \
    --site "$reference_site" --output "$parity_root/triton-adaln-scalar-frontend" \
    --epsilon-types i32 i32_div16 i64 one u1 --scalar-slice
python3 -S -B inference/tools/test_triton_frontend.py --root "$parity_root" --family adaln \
    --reference-directory "$parity_root/triton-adaln-scalar-frontend" --compile-selected --raw-frontend
"$parity_root/venv/bin/python" -S -B inference/tools/reference_triton_rope_frontend.py \
    --site "$reference_site" --output "$parity_root/triton-rope-frontend"
python3 -S -B inference/tools/test_triton_frontend.py --root "$parity_root" --family rope --compile-selected --raw-frontend
"$parity_root/venv/bin/python" -S -B inference/tools/reference_triton_rms_rope_frontend.py \
    --site "$reference_site" --output "$parity_root/triton-rms-rope-frontend"
python3 -S -B inference/tools/test_triton_frontend.py --root "$parity_root" --family rms-rope --compile-selected --raw-frontend
"$parity_root/venv/bin/python" -S -B inference/tools/reference_triton_rms_rope_frontend.py \
    --site "$reference_site" --output "$parity_root/triton-rms-rope-scalar-frontend" \
    --epsilon-types i32 i32_div16 i64 one u1 --scalar-slice
python3 -S -B inference/tools/test_triton_frontend.py --root "$parity_root" --family rms-rope \
    --reference-directory "$parity_root/triton-rms-rope-scalar-frontend" --compile-selected --raw-frontend
"$parity_root/venv/bin/python" -S -B inference/tools/reference_triton_block_quant_frontend.py \
    --site "$reference_site" --output "$parity_root/triton-mxfp8-frontend" --family mxfp8
python3 -S -B inference/tools/test_triton_frontend.py --root "$parity_root" --family mxfp8 --compile-selected --raw-frontend
"$parity_root/venv/bin/python" -S -B inference/tools/reference_triton_block_quant_frontend.py \
    --site "$reference_site" --output "$parity_root/triton-nvfp4-frontend" --family nvfp4
python3 -S -B inference/tools/test_triton_frontend.py --root "$parity_root" --family nvfp4 --compile-selected --raw-frontend
"$parity_root/venv/bin/python" -S -B inference/tools/reference_triton_block_quant_frontend.py \
    --site "$reference_site" --output "$parity_root/triton-dequant-nvfp4-frontend" --family dequant_nvfp4
python3 -S -B inference/tools/test_triton_frontend.py --root "$parity_root" --family dequant-nvfp4 --compile-selected --raw-frontend
"$parity_root/venv/bin/python" -S -B inference/tools/reference_triton_int8_frontend.py \
    --site "$reference_site" --output "$parity_root/triton-int8-frontend"
python3 -S -B inference/tools/test_triton_frontend.py --root "$parity_root" --family int8 --compile-selected --raw-frontend
"$parity_root/venv/bin/python" -S -B inference/tools/reference_triton_int8_frontend.py \
    --site "$reference_site" --output "$parity_root/triton-int8-double-frontend" --output-types fp64
python3 -S -B inference/tools/test_triton_frontend.py --root "$parity_root" --family int8 \
    --reference-directory "$parity_root/triton-int8-double-frontend" --compile-selected --raw-frontend
"$parity_root/venv/bin/python" -S -B inference/tools/reference_triton_w4a8_frontend.py \
    --site "$reference_site" --output "$parity_root/triton-w4a8-frontend"
python3 -S -B inference/tools/test_triton_frontend.py --root "$parity_root" --family w4a8 --compile-selected --raw-frontend
"$parity_root/venv/bin/python" -S -B inference/tools/reference_triton_na_frontend.py \
    --site "$reference_site" --output "$parity_root/triton-na-frontend"
python3 -S -B inference/tools/test_triton_frontend.py --root "$parity_root" --family na --compile-selected --raw-frontend
"$parity_root/venv/bin/python" -S -B inference/tools/reference_triton_na_frontend.py \
    --site "$reference_site" --output "$parity_root/triton-na-dynamic-frontend" --dynamic-kernels
python3 -S -B inference/tools/test_triton_frontend.py --root "$parity_root" --family na \
    --reference-directory "$parity_root/triton-na-dynamic-frontend" --compile-selected --raw-frontend
for family in adaln rms-rope; do
    script_family=${family//-/_}
    "$parity_root/venv/bin/python" -S -B "inference/tools/reference_triton_${script_family}_frontend.py" \
        --site "$reference_site" --output "$parity_root/triton-$family-uint-frontend" \
        --epsilon-types u64 u64_div16 --scalar-slice
    python3 -S -B inference/tools/test_triton_frontend.py --root "$parity_root" --family "$family" \
        --reference-directory "$parity_root/triton-$family-uint-frontend" --compile-selected --raw-frontend
done
"$parity_root/venv/bin/python" -S -B inference/tools/reference_triton_na_frontend.py \
    --site "$reference_site" --output "$parity_root/triton-na-uint-frontend" --unsigned-scales
python3 -S -B inference/tools/test_triton_frontend.py --root "$parity_root" --family na \
    --reference-directory "$parity_root/triton-na-uint-frontend" --compile-selected --raw-frontend
"$parity_root/venv/bin/python" -S -B inference/tools/reference_triton_na_frontend.py \
    --site "$reference_site" --output "$parity_root/triton-na-integer-frontend" --integer-boundaries
python3 -S -B inference/tools/test_triton_frontend.py --root "$parity_root" --family na \
    --reference-directory "$parity_root/triton-na-integer-frontend" --raw-frontend
"$parity_root/venv/bin/python" -S -B inference/tools/reference_triton_nvfp4_literals.py \
    --site "$reference_site" --output "$parity_root/triton-nvfp4-literals"
python3 -S -B inference/tools/test_triton_frontend.py --root "$parity_root" --family nvfp4 \
    --reference-directory "$parity_root/triton-nvfp4-literals" --compile-selected --raw-frontend
"$parity_root/venv/bin/python" -S -B inference/tools/reference_triton_specialization.py \
    --site "$reference_site" --frontend-reference "$parity_root/triton-frontend/reference.json" \
    --output "$parity_root/runtime/triton-specialization-reference.json" \
    --unicode-rust inference/src/python_printable.rs
"$parity_root/venv/bin/python" -S -B inference/tools/prepare_triton_compiler_profile.py --root "$parity_root" --site "$reference_site" --output "$parity_root/runtime/triton-compiler-profile.json"
"$parity_root/venv/bin/python" -S -B inference/tools/reference_primitives.py \
    --site "$reference_site" --output "$IMAGINE_PRIMITIVE_FIXTURES" \
    --triton-frontend-reference "$parity_root/triton-frontend/reference.json" \
    --triton-specialization-reference "$parity_root/runtime/triton-specialization-reference.json"
"$parity_root/venv/bin/python" -S -B inference/tools/probe_triton_aot.py \
    --site "$reference_site" --output "$parity_root/triton-aot"
"$parity_root/venv/bin/python" -S -B inference/tools/build_int8_aot.py \
    --site "$reference_site" --output "$parity_root/int8-aot"
"$parity_root/venv/bin/python" -S -B inference/tools/reference_autotuning_policy.py \
    --site "$reference_site" --output "$IMAGINE_AUTOTUNING_POLICY_FIXTURE"
"$parity_root/venv/bin/python" -S -B inference/tools/reference_backend_contract.py \
    --site "$reference_site" --output "$IMAGINE_BACKEND_CONTRACT_FIXTURE" \
    --check-rust inference/src/backend_contract/tables.rs
cargo fmt --manifest-path inference/Cargo.toml --check
cargo test --locked --offline --manifest-path inference/Cargo.toml -- --test-threads=1
cargo test --locked --offline --manifest-path inference/Cargo.toml --no-run --message-format=json > "$parity_root/runtime/test-build.json"
primitive_binary=$(python3 -B -c '
import json,sys
items=[json.loads(line) for line in open(sys.argv[1])]
paths=[x["executable"] for x in items if x.get("reason")=="compiler-artifact" and x["target"]["name"]=="primitives" and x.get("executable")]
assert len(paths)==1
print(paths[0])
' "$parity_root/runtime/test-build.json")
python3 -B inference/tools/test_isolation.py --root "$parity_root" --site "$reference_site" \
    --binary "$primitive_binary" --aot-library "$IMAGINE_TRITON_PROBE_LIBRARY" \
    --aot-fixture "$IMAGINE_TRITON_PROBE_FIXTURE" \
    --backend-fixture "$IMAGINE_BACKEND_CONTRACT_FIXTURE" \
    --int8-library "$IMAGINE_INT8_PROBE_LIBRARY" --int8-fixture "$IMAGINE_INT8_PROBE_FIXTURE"
