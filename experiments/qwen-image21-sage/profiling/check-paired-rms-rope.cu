// Isolated Q/K kernel experiment. All inputs are generated here, without model data.
#include "ggml-cuda/common.cuh"
#include "../runs/paired-rms-rope-build/baseline-kernel.cuh"
#include "paired-rms-rope-kernel.cuh"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <vector>

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        std::exit(2);
    }
}

template<class T> static T * upload(const std::vector<T> & input) {
    T * ptr;
    check(cudaMalloc(&ptr, input.size() * sizeof(T)));
    check(cudaMemcpy(ptr, input.data(), input.size() * sizeof(T), cudaMemcpyHostToDevice));
    return ptr;
}

static uint32_t random_word(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static size_t run(int tokens, int heads, int batches, int mode, bool timing) {
    const size_t elements = size_t(tokens) * heads * batches * 128;
    std::vector<nv_bfloat16> q(elements), k(elements);
    std::vector<float> qw(128), kw(128), pe(size_t(tokens) * 256);
    uint32_t state = 42;
    for (size_t i = 0; i < elements; ++i) {
        float a = (int(random_word(state) % 20001) - 10000) / 997.0f;
        float b = (int(random_word(state) % 20001) - 10000) / 991.0f;
        if (mode == 1) { a = (i % 2) ? -0.0f : 0.0f; b = -a; }
        if (mode == 2) { a *= 1.e-12f; b *= 1.e-10f; }
        if (mode == 3) { a *= 1.e12f; b *= 1.e10f; }
        q[i] = __float2bfloat16_rn(a);
        k[i] = __float2bfloat16_rn(b);
    }
    for (int i = 0; i < 128; ++i) {
        qw[i] = (int(random_word(state) % 1025) - 512) / 256.0f;
        kw[i] = (int(random_word(state) % 1025) - 512) / 257.0f;
    }
    for (int token = 0; token < tokens; ++token) {
        for (int pair = 0; pair < 64; ++pair) {
            float angle = (token + 1) * (pair + 1) * 0.0037f;
            auto m = pe.data() + size_t(token) * 256 + pair * 4;
            m[0] = std::cos(angle); m[1] = -std::sin(angle);
            m[2] = std::sin(angle); m[3] = std::cos(angle);
        }
    }
    auto dq = upload(q), dk = upload(k);
    auto dqw = upload(qw), dkw = upload(kw), dpe = upload(pe);
    nv_bfloat16 * out[4];
    for (auto & p : out) check(cudaMalloc(&p, elements * sizeof(*p)));
    RopeApplyStrides strides = {{1,128,128*heads,int64_t(128)*heads*tokens},{1,2,4,256}};
    const int rows = tokens * heads * batches;
    auto launch = [&](bool paired) {
        if (paired) {
            paired_rms_norm_rope_apply_bf16<<<rows,128,64*sizeof(float)>>>(
                dq,dk,dqw,dkw,dpe,out[2],out[3],heads,tokens,1.e-6f,3.e-6f,strides);
        } else {
            rms_norm_rope_apply_bf16<<<rows,128,32*sizeof(float)>>>(
                dq,dqw,dpe,out[0],heads,tokens,1.e-6f,strides);
            rms_norm_rope_apply_bf16<<<rows,128,32*sizeof(float)>>>(
                dk,dkw,dpe,out[1],heads,tokens,3.e-6f,strides);
        }
        check(cudaGetLastError());
    };
    launch(false); launch(true); check(cudaDeviceSynchronize());
    std::vector<nv_bfloat16> reference(elements), candidate(elements);
    for (int side = 0; side < 2; ++side) {
        check(cudaMemcpy(reference.data(),out[side],elements*2,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(candidate.data(),out[side+2],elements*2,cudaMemcpyDeviceToHost));
        if (std::memcmp(reference.data(),candidate.data(),elements*2)) {
            std::fprintf(stderr,"MISMATCH tokens=%d heads=%d batch=%d mode=%d side=%d\n",tokens,heads,batches,mode,side);
            std::exit(3);
        }
        for (auto v : candidate) if (!std::isfinite(float(v))) std::exit(4);
    }
    std::printf("{\"tokens\":%d,\"heads\":%d,\"batch\":%d,\"mode\":%d,\"exact_bytes\":%zu,\"timings\":[",tokens,heads,batches,mode,elements*4);
    if (timing) {
        for (int i=0;i<20;++i) { launch(false); launch(true); }
        check(cudaDeviceSynchronize());
        cudaEvent_t begin, end;
        check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
        for (int batch=0;batch<8;++batch) {
            float ms[2];
            for (int order=0;order<2;++order) {
                int variant = (batch + order) % 2;
                check(cudaEventRecord(begin));
                for (int i=0;i<50;++i) launch(variant);
                check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
                check(cudaEventElapsedTime(&ms[variant],begin,end));
            }
            std::printf("%s{\"separate_ms\":%.6f,\"paired_ms\":%.6f}",batch?",":"",ms[0]/50,ms[1]/50);
        }
        check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
    }
    std::printf("]}\n"); std::fflush(stdout);
    for (auto p : out) check(cudaFree(p));
    check(cudaFree(dq)); check(cudaFree(dk)); check(cudaFree(dqw)); check(cudaFree(dkw)); check(cudaFree(dpe));
    return elements * 4;
}

int main(int argc, char **) {
    if (argc > 1) { run(17,32,2,0,false); return 0; }
    for (int tokens : {1,17,57,230,5600,5657,5830}) run(tokens,32,1,0,tokens>=5600);
    run(17,1,2,0,false);
    run(230,32,2,0,false);
    for (int mode=1;mode<=3;++mode) run(57,32,2,mode,false);
    return 0;
}
