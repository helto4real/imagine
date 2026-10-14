// Local regression test for the optional CUDA pool-memory query on SM89/VMM.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Query = bool (*)(ggml_backend_t, ggml_cgraph*, size_t*, size_t*);
static constexpr size_t MiB = 1024 * 1024;

int main() {
    unsetenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    unsetenv("GGML_CUDA_GRAPH_OPT");
    auto backend = ggml_backend_cuda_init(0);
    assert(backend);
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto query = reinterpret_cast<Query>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_graph_pool_memory"));
    assert(query);
    auto ctx = ggml_init({16 * MiB, nullptr, true});
    auto graph_for = [&](ggml_type a_type, ggml_type b_type, int64_t k, int64_t m, int64_t n, int64_t batches = 1) {
        auto a = ggml_new_tensor_2d(ctx, a_type, k, m);
        auto b = ggml_new_tensor_3d(ctx, b_type, k, n, batches);
        auto d = ggml_mul_mat(ctx, a, b);
        auto g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, d);
        return g;
    };
    auto check = [&](const char* label, ggml_cgraph* graph, size_t expected_resident, size_t expected_growth) {
        size_t resident = SIZE_MAX, growth = SIZE_MAX;
        assert(query(backend, graph, &resident, &growth));
        std::printf("%s resident=%zu growth=%zu\n", label, resident, growth);
        assert(resident == expected_resident && growth == expected_growth);
    };
    auto vae = graph_for(GGML_TYPE_F16, GGML_TYPE_F16, 2592, 1433600, 288);
    check("VAE half output; no graph allocation", vae, 0, 788 * MiB);
    auto vae2 = ggml_mul_mat(ctx, ggml_graph_node(vae, 0)->src[0], ggml_graph_node(vae, 0)->src[1]);
    ggml_build_forward_expand(vae, vae2);
    check("serial operations reuse high-water mark", vae, 0, 788 * MiB);
    check("bf16 direct float output", graph_for(GGML_TYPE_BF16, GGML_TYPE_BF16, 2592, 1433600, 288), 0, 0);
    // F32 override converts both F16 inputs, but no longer needs a half output.
    setenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE", "f32", 1);
    const size_t converted = (size_t(2592) * (1433600 + 288) * 4 + 2 * MiB - 1) / (2 * MiB) * (2 * MiB);
    check("f32 override input conversions", vae, 0, converted);
    unsetenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    // 0.25 MiB half output + 3 MiB pointer arrays rounds to 4 MiB.
    check("broadcast pointer arrays", graph_for(GGML_TYPE_F16, GGML_TYPE_F16, 1, 1, 1, 131072), 0, 4 * MiB);
    check("mixed input conversion plus output", graph_for(GGML_TYPE_F16, GGML_TYPE_F32, 1024, 1024, 1024), 0, 4 * MiB);
    auto small = graph_for(GGML_TYPE_F16, GGML_TYPE_F16, 32, 1536, 1536);
    check("before real GEMM", small, 0, 6 * MiB);
    auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    assert(ggml_gallocr_alloc_graph(alloc, small));
    auto dst = ggml_graph_node(small, 0);
    std::vector<ggml_fp16_t> ones(32 * 1536, ggml_fp32_to_fp16(1.0f));
    for (auto src : {dst->src[0], dst->src[1]}) {
        ggml_backend_tensor_set(src, ones.data(), 0, ggml_nbytes(src));
    }
    for (int pass = 0; pass < 2; ++pass) {
        assert(ggml_backend_graph_compute(backend, small) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);
        std::vector<float> actual(1536 * 1536);
        ggml_backend_tensor_get(dst, actual.data(), 0, ggml_nbytes(dst));
        for (float v : actual) assert(v == 32.0f);
        check("after real GEMM; reuse without double count", small, 6 * MiB, 0);
        check("VAE deducts existing physical pool", vae, 6 * MiB, 782 * MiB);
    }
    setenv("GGML_CUDA_GRAPH_OPT", "1", 1);
    size_t resident = 123, growth = 123;
    assert(!query(backend, small, &resident, &growth) && resident == 0 && growth == 0);
    unsetenv("GGML_CUDA_GRAPH_OPT");
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::puts("PASS: all CUDA pool-memory checks");
}
