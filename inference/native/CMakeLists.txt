cmake_minimum_required(VERSION 3.24)
project(imagine_kitchen LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(CUDAToolkit REQUIRED)
include("${GENERATED_DIR}/kernel_sources.cmake")
set(KITCHEN_DIR "${REFERENCE_SOURCES}/comfy-kitchen/comfy_kitchen/backends/cuda")
add_library(imagine_kitchen SHARED bridge.cpp ${KITCHEN_CUDA_SOURCES})
target_include_directories(imagine_kitchen PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${LIBTORCH}/include;${LIBTORCH}/include/torch/csrc/api/include>
    "${GENERATED_DIR}" "${KITCHEN_DIR}" "${KITCHEN_DIR}/sage_attention"
    "${KITCHEN_DIR}/flash_compat" "${REFERENCE_SOURCES}/cutlass/include"
    "${REFERENCE_SOURCES}/flash-attention/csrc/flash_attn/src"
    "${FROZEN_CUDA_INCLUDE}")
target_compile_definitions(imagine_kitchen PRIVATE COMFY_HAVE_CUTLASS _GLIBCXX_USE_CXX11_ABI=1)
target_compile_options(imagine_kitchen PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_OPERATORS__;-U__CUDA_NO_HALF_CONVERSIONS__;-U__CUDA_NO_BFLOAT16_OPERATORS__;-U__CUDA_NO_BFLOAT16_CONVERSIONS__;-U__CUDA_NO_BFLOAT162_OPERATORS__;-U__CUDA_NO_BFLOAT162_CONVERSIONS__;--expt-relaxed-constexpr;--expt-extended-lambda;--use_fast_math;--compress-mode=size;-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden;-O3>)
# The Rust boundary uses the C ABI. Upstream launch symbols may also be
# exported; no nanobind/Python libraries or symbols are included.
set_target_properties(imagine_kitchen PROPERTIES CXX_VISIBILITY_PRESET default)
target_link_directories(imagine_kitchen PRIVATE "${LIBTORCH}/lib")
target_link_libraries(imagine_kitchen PRIVATE torch torch_cpu torch_cuda c10 c10_cuda CUDA::cudart_static ${CMAKE_DL_LIBS})
