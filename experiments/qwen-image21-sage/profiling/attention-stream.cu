#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"

// Diagnostic bridge compiled with the same context-layout definitions as GGML.
extern "C" cudaStream_t benchmark_stream(ggml_backend_t backend) {
    return static_cast<ggml_backend_cuda_context *>(backend->context)->stream();
}
