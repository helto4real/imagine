cmake_minimum_required(VERSION 3.18)
project(imagine_comfy_adapters LANGUAGES CXX CUDA)
add_library(imagine-comfy-adapter-loader STATIC
    src/adapter_loader.cpp
    src/model_keys.cpp
    src/weight_policy.cpp
    src/adapter_math.cpp
    src/weight_patcher.cpp
    src/model_patcher.cpp
    src/layer_cast.cpp
    src/native_layer_storage.cpp
    src/native_tensor.cpp
    src/tensor_copy.cpp
    src/tensor_copy.cu
    src/tensor_arithmetic.cpp
    src/tensor_arithmetic.cu
    src/tensor_matmul.cu
    src/tensor_contract.cpp
    src/tensor_layout.cpp
    src/tensor_reduce.cpp
    src/tensor_reduce.cu
    src/tensor_inverse.cu
    src/tensor_int8.cu
    src/tensor_int8_eager.cu
    src/int8_policy.cpp)
target_include_directories(imagine-comfy-adapter-loader PUBLIC src)
target_compile_features(imagine-comfy-adapter-loader PUBLIC cxx_std_17)

find_package(CUDAToolkit REQUIRED)
add_library(imagine-comfy-int8-kernels STATIC kitchen/int8_linear.cu)
target_include_directories(imagine-comfy-int8-kernels PRIVATE kitchen)
target_compile_options(imagine-comfy-int8-kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;--expt-relaxed-constexpr;--expt-extended-lambda;--compress-mode=size;
    -U__CUDA_NO_HALF_OPERATORS__;-U__CUDA_NO_HALF_CONVERSIONS__;
    -U__CUDA_NO_BFLOAT16_OPERATORS__;-U__CUDA_NO_BFLOAT16_CONVERSIONS__;
    -U__CUDA_NO_BFLOAT162_OPERATORS__;-U__CUDA_NO_BFLOAT162_CONVERSIONS__>)
set_target_properties(imagine-comfy-int8-kernels PROPERTIES
    CUDA_STANDARD 17 POSITION_INDEPENDENT_CODE ON)
target_link_libraries(imagine-comfy-int8-kernels PUBLIC CUDA::cudart)
target_link_libraries(imagine-comfy-adapter-loader PUBLIC imagine-comfy-int8-kernels)
target_link_libraries(imagine-comfy-adapter-loader PUBLIC CUDA::cudart CUDA::cublas CUDA::cusolver)
target_compile_options(imagine-comfy-adapter-loader PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
set_source_files_properties(src/tensor_copy.cu src/tensor_arithmetic.cu src/tensor_matmul.cu PROPERTIES COMPILE_OPTIONS "--fmad=false")
