// Local CUDA regression test; synthetic values only.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

bool ggml_cuda_can_fuse_rms_norm_rope_apply(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, bool = true);

static void require(bool ok, const char * reason) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", reason); std::exit(1); }
}

int main(int argc, char **) {
    const bool small = argc > 1;
    auto backend = ggml_backend_cuda_init(0);
    auto cpu = ggml_backend_cpu_init();
    ggml_backend_t backends[] = {backend, cpu};
    require(backend != nullptr, "CUDA backend");
    int expected_fusions = 0;
    for (int test = 0; test < 16; ++test) {
        const int dim = test == 6 ? 64 : 128;
        const int heads = test == 0 ? 1 : 32;
        const int tokens = test == 0 ? 1 : test == 1 ? 57 : test == 2 ? 230 : small ? 9 : 5600;
        const int batches = test == 3 ? 2 : 1;
        const bool interleaved = test != 7;
        const bool strided = test == 4;
        const float eps = test == 5 ? 0.5f : 1e-6f;
        auto ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
        auto storage = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, heads + int(strided), tokens, batches);
        auto x = strided ? ggml_view_4d(ctx, storage, dim, heads, tokens, batches,
                    storage->nb[1], storage->nb[2], storage->nb[3], 0) : storage;
        if (test >= 13) x = ggml_scale(ctx, x, 1.f);
        auto weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
        auto pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, dim/2, tokens);
        auto ref_norm = ggml_rms_norm(ctx, x, eps);
        ggml_set_output(ref_norm);  // Prevent reference fusion, including RMSNorm/MUL.
        auto ref_mul = ggml_mul_inplace(ctx, ref_norm, weight);
        auto reference = ggml_rope_apply(ctx, ref_mul, pe, interleaved);
        auto norm = ggml_rms_norm(ctx, x, eps);
        auto mul = ggml_mul_inplace(ctx, norm, weight);
        auto output = ggml_rope_apply(ctx, mul, pe, interleaved);
        auto result = output;
        if (test == 15) {
            result = ggml_view_4d(ctx, output, output->ne[0], output->ne[1], output->ne[2], output->ne[3], output->nb[1], output->nb[2], output->nb[3], 0);
            result->src[1] = x;
            result->src[2] = weight;
        }
        if (test == 8) ggml_set_output(norm);
        if (test == 9 || test == 14) ggml_set_output(mul);
        auto graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, reference);
        ggml_build_forward_expand(graph, result);
        auto extra = test == 10 ? ggml_scale(ctx, norm, 2.f) : nullptr;
        if (extra) ggml_build_forward_expand(graph, extra);
        for (auto tensor : {storage, weight, pe, reference, output}) ggml_set_output(tensor);
        const bool use_scheduler = test == 13 || test == 14;
        auto allocator = !use_scheduler ? ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)) : nullptr;
        auto scheduler = use_scheduler ? ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true) : nullptr;
        require(scheduler ? ggml_backend_sched_alloc_graph(scheduler, graph) : ggml_gallocr_alloc_graph(allocator, graph), "graph allocation");

        std::vector<float> input(ggml_nelements(storage)), weights(dim), table(ggml_nelements(pe));
        std::mt19937 rng(42 + test);
        for (auto & v : input) v = test == 11 ? 0.f : (int(rng() % 20001) - 10000) * (test == 12 ? 1e-8f : .001f);
        for (auto & v : weights) v = (int(rng() % 4001) - 2000) * .001f;
        for (size_t i = 0; i < table.size(); i += 4) {
            float angle = float(i % 731) * .017f;
            table[i] = std::cos(angle); table[i+1] = -std::sin(angle);
            table[i+2] = std::sin(angle); table[i+3] = std::cos(angle);
        }
        ggml_backend_tensor_set(storage, input.data(), 0, input.size()*sizeof(float));
        ggml_backend_tensor_set(weight, weights.data(), 0, weights.size()*sizeof(float));
        ggml_backend_tensor_set(pe, table.data(), 0, table.size()*sizeof(float));

        const bool eligible = ggml_cuda_can_fuse_rms_norm_rope_apply(norm, mul, output, false);
        require(eligible == (dim == 128 && interleaved), "shape/mode eligibility");
        if (eligible && test != 8 && test != 9 && test != 10 && test != 14) ++expected_fusions;
        auto alias = *output; alias.data = x->data;
        require(!ggml_cuda_can_fuse_rms_norm_rope_apply(norm, mul, &alias), "overlapping input must fall back");
        auto invalid_weight = *weight; invalid_weight.type = GGML_TYPE_F16;
        auto invalid_mul = *mul; invalid_mul.src[1] = &invalid_weight;
        auto invalid_rope = *output; invalid_rope.src[0] = &invalid_mul;
        require(!ggml_cuda_can_fuse_rms_norm_rope_apply(norm, &invalid_mul, &invalid_rope), "wrong weight type must fall back");
        require((scheduler ? ggml_backend_sched_graph_compute(scheduler, graph) : ggml_backend_graph_compute(backend, graph)) == GGML_STATUS_SUCCESS, "graph compute");
        std::vector<float> expected(ggml_nelements(reference)), actual(expected.size());
        ggml_backend_tensor_get(reference, expected.data(), 0, expected.size()*sizeof(float));
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size()*sizeof(float));
        size_t mismatches = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            require(std::isfinite(actual[i]), "finite output");
            mismatches += std::memcmp(&actual[i], &expected[i], sizeof(float)) != 0;
        }
        // Independent double-precision check of norm, rotation and output permutation.
        for (size_t index = 0; index < actual.size(); index += 997) {
            size_t row = index / dim, col = index % dim;
            size_t token = row % tokens, head = (row / tokens) % heads, batch = row / (tokens * heads);
            size_t base = batch*storage->nb[3]/4 + token*storage->nb[2]/4 + head*dim;
            double sum = 0;
            for (int j = 0; j < dim; ++j) sum += double(input[base+j])*input[base+j];
            double scale = 1. / std::sqrt(sum/dim + eps);
            size_t pair = interleaved ? col/2 : col % (dim/2);
            size_t c0 = interleaved ? 2*pair : pair, c1 = interleaved ? 2*pair+1 : pair+dim/2;
            double a = scale*input[base+c0]*weights[c0], b = scale*input[base+c1]*weights[c1];
            size_t pos = (token*(dim/2)+pair)*4;
            bool second = interleaved ? col%2 : col >= size_t(dim/2);
            double value = second ? a*table[pos+2]+b*table[pos+3] : a*table[pos]+b*table[pos+1];
            require(std::abs(actual[index]-value) <= 2e-6*std::max(1.,std::abs(value)), "independent formula");
        }
        std::printf("case=%d dims=%d,%d,%d,%d compared=%zu bit_mismatches=%zu formula=PASS\n",
                    test, dim, heads, tokens, batches, actual.size(), mismatches);
        require(mismatches == 0, "bitwise reference comparison");
        if (scheduler) ggml_backend_sched_free(scheduler);
        ggml_gallocr_free(allocator); ggml_free(ctx);
    }
    std::printf("PASS expected_fused_launches_with_flag_on=%d\n", expected_fusions);
    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    return 0;
}
