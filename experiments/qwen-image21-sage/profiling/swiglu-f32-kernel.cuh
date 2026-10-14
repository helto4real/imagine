template<int block_threads>
static __global__ void quantize_swiglu_i8_convrot_warp_cuda(
        const float * src, int8_t * dst, float * scales, int64_t k, int64_t rows) {
    const int64_t row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    extern __shared__ float values[];
    __shared__ float maxima[32];
    const float * input = src + row * 2 * k;
    float local_max = 0.0f;
    for (int64_t group = warp; group < k / 256; group += block_threads / 32) {
        float v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int64_t index = group * 256 + lane + j * 32;
            v[j] = (ggml_cuda_op_silu_single(input[index]) * input[k + index]) * (1.0f / 16.0f);
        }
        // The first two radix-4 stages stay within each warp's 32-element slice.
#pragma unroll
        for (int stride = 1; stride <= 4; stride *= 4) {
            const int base = lane & ~(3 * stride);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float a = __shfl_sync(0xffffffff, v[j], base);
                const float b = __shfl_sync(0xffffffff, v[j], base + stride);
                const float c = __shfl_sync(0xffffffff, v[j], base + 2 * stride);
                const float d = __shfl_sync(0xffffffff, v[j], base + 3 * stride);
                const int part = (lane / stride) & 3;
                v[j] = part == 0 ? a + b + c - d : part == 1 ? a + b - c + d :
                       part == 2 ? a - b + c + d : -a + b + c + d;
            }
        }
        // Stride 16 crosses two register slices; stride 64 is entirely local.
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
            const float a = __shfl_sync(0xffffffff, v[j], lane & 15);
            const float b = __shfl_sync(0xffffffff, v[j], (lane & 15) + 16);
            const float c = __shfl_sync(0xffffffff, v[j+1], lane & 15);
            const float d = __shfl_sync(0xffffffff, v[j+1], (lane & 15) + 16);
            v[j] = lane < 16 ? a + b + c - d : a + b - c + d;
            v[j+1] = lane < 16 ? a - b + c + d : -a + b + c + d;
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float a = v[j], b = v[j+2], c = v[j+4], d = v[j+6];
            v[j] = a + b + c - d;
            v[j+2] = a + b - c + d;
            v[j+4] = a - b + c + d;
            v[j+6] = -a + b + c + d;
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            values[group * 256 + lane + j * 32] = v[j];
            local_max = fmaxf(local_max, fabsf(v[j]));
        }
    }
    local_max = warp_reduce_max<32>(local_max);
    if (lane == 0) maxima[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        local_max = lane < block_threads / 32 ? maxima[lane] : 0.0f;
        local_max = warp_reduce_max<32>(local_max);
        if (lane == 0) scales[row] = maxima[0] = local_max / 127.0f;
    }
    __syncthreads();
    const float scale = maxima[0], inv = scale == 0.0f ? 0.0f : 1.0f / scale;
    for (int64_t i = tid; i < k; i += block_threads) {
        int value = scale == 0.0f ? 0 : __float2int_rn(values[i] * inv);
        dst[row * k + i] = (int8_t) max(-127, min(127, value));
    }
}

template<int block_threads>
static __global__ void quantize_swiglu_i8_convrot_vec_cuda(
        const float * src, int8_t * dst, float * scales, int64_t k, int64_t rows) {
    const int64_t row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    extern __shared__ float values[];
    __shared__ float maxima[32];
    const float * input = src + row * 2 * k;
    float local_max = 0.0f;
    for (int64_t group = warp; group < k / 256; group += block_threads / 32) {
        float v[8];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int64_t index = group * 256 + lane * 4 + j * 128;
            const float4 gate = *reinterpret_cast<const float4 *>(input + index);
            const float4 up = *reinterpret_cast<const float4 *>(input + k + index);
            v[4*j]   = (ggml_cuda_op_silu_single(gate.x) * up.x) * (1.0f/16.0f);
            v[4*j+1] = (ggml_cuda_op_silu_single(gate.y) * up.y) * (1.0f/16.0f);
            v[4*j+2] = (ggml_cuda_op_silu_single(gate.z) * up.z) * (1.0f/16.0f);
            v[4*j+3] = (ggml_cuda_op_silu_single(gate.w) * up.w) * (1.0f/16.0f);
        }
#pragma unroll
        for (int j = 0; j < 8; j += 4) {
            const float a = v[j], b = v[j+1], c = v[j+2], d = v[j+3];
            v[j] = a + b + c - d; v[j+1] = a + b - c + d;
            v[j+2] = a - b + c + d; v[j+3] = -a + b + c + d;
        }
#pragma unroll
        for (int stride = 1; stride <= 4; stride *= 4) {
            const int base = lane & ~(3 * stride), part = (lane / stride) & 3;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float a = __shfl_sync(0xffffffff, v[j], base);
                const float b = __shfl_sync(0xffffffff, v[j], base + stride);
                const float c = __shfl_sync(0xffffffff, v[j], base + 2*stride);
                const float d = __shfl_sync(0xffffffff, v[j], base + 3*stride);
                v[j] = part == 0 ? a + b + c - d : part == 1 ? a + b - c + d :
                       part == 2 ? a - b + c + d : -a + b + c + d;
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float a = __shfl_sync(0xffffffff, v[j], lane & 15);
            const float b = __shfl_sync(0xffffffff, v[j], (lane & 15) + 16);
            const float c = __shfl_sync(0xffffffff, v[j+4], lane & 15);
            const float d = __shfl_sync(0xffffffff, v[j+4], (lane & 15) + 16);
            v[j] = lane < 16 ? a + b + c - d : a + b - c + d;
            v[j+4] = lane < 16 ? a - b + c + d : -a + b + c + d;
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            *reinterpret_cast<float4 *>(values + group*256 + lane*4 + j*128) = make_float4(v[4*j],v[4*j+1],v[4*j+2],v[4*j+3]);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) local_max = fmaxf(local_max,fabsf(v[j]));
    }
    local_max = warp_reduce_max<32>(local_max);
    if (lane == 0) maxima[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        local_max = lane < block_threads/32 ? maxima[lane] : 0.0f;
        local_max = warp_reduce_max<32>(local_max);
        if (lane == 0) scales[row] = maxima[0] = local_max/127.0f;
    }
    __syncthreads();
    const float scale = maxima[0], inv = scale == 0.0f ? 0.0f : 1.0f/scale;
    for (int64_t i = tid*4; i < k; i += block_threads*4) {
        const float4 val = *reinterpret_cast<const float4 *>(values+i);
        const float vv[4] = {val.x,val.y,val.z,val.w};
        uint32_t packed = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int v = scale == 0.0f ? 0 : __float2int_rn(vv[j]*inv);
            packed |= uint32_t(uint8_t(max(-127,min(127,v)))) << (8*j);
        }
        *reinterpret_cast<uint32_t *>(dst+row*k+i) = packed;
    }
}
