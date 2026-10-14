// Local CUDA regression experiment: copied halves vs packed gate/up.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    auto backend = ggml_backend_cuda_init(0);
    if (!backend) return 2;
    for (int rows : {1, 57, 230, 5600}) {
        constexpr int channels = 12288;
        auto* ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
        auto* packed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * channels, rows);
        auto* gate = ggml_cont(ctx, ggml_view_2d(ctx, packed, channels, rows, packed->nb[1], 0));
        auto* up = ggml_cont(ctx, ggml_view_2d(ctx, packed, channels, rows, packed->nb[1], channels * sizeof(float)));
        auto* reference = ggml_mul(ctx, up, ggml_silu(ctx, gate));
        auto* direct = ggml_swiglu(ctx, packed);
        auto* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, reference);
        ggml_build_forward_expand(graph, direct);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) return 2;
        std::vector<float> input(size_t(2) * channels * rows);
        std::mt19937 random(42 + rows);
        for (auto& value : input) value = (int(random() % 16001) - 8000) / 1000.f;
        ggml_backend_tensor_set(packed, input.data(), 0, input.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 2;
        std::vector<float> expected(size_t(channels) * rows), actual(expected.size());
        ggml_backend_tensor_get(reference, expected.data(), 0, expected.size() * sizeof(float));
        ggml_backend_tensor_get(direct, actual.data(), 0, actual.size() * sizeof(float));
        size_t mismatches = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (std::memcmp(&expected[i], &actual[i], sizeof(float)) != 0) ++mismatches;
        }
        // Independently check gate order and the row stride against the formula.
        for (size_t i = 0; i < actual.size(); i += 997) {
            size_t offset = (i / channels) * 2 * channels + i % channels;
            float g = input[offset], u = input[offset + channels];
            float formula = (g / (1.f + std::exp(-g))) * u;
            if (!std::isfinite(actual[i]) || std::abs(actual[i] - formula) > 2e-5f * std::max(1.f, std::abs(formula))) return 1;
        }
        std::printf("rows=%d channels=%d compared=%zu bit_mismatches=%zu formula=PASS\n", rows, channels, actual.size(), mismatches);
        if (mismatches) return 1;
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return 0;
}
