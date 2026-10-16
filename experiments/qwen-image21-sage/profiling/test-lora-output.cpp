// Synthetic graph equivalence fixtures; no model weights, prompts or images.
#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

struct Case {
    const char * name;
    int rank, width, rows, branches;
    float strength;
    bool attention = false, marked = true, keep_mat = false, force_f32 = false, inplace = true;
    float bias = 0;
};

static void run(ggml_backend_t backend, FILE * file, const Case & c) {
    auto ctx = ggml_init({16 * 1024 * 1024, nullptr, true}); assert(ctx);
    auto base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.width, c.rows);
    auto x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.rank, c.rows);
    auto w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.rank, c.width);
    ggml_set_input(base); ggml_set_input(x); ggml_set_input(w);
    // Retain buffers that this test reads after graph completion; otherwise the
    // allocator may legally reuse them once their last graph consumer finishes.
    ggml_set_output(x);
    if (!c.inplace) ggml_set_output(base);
    auto output = base;
    std::vector<ggml_tensor *> retained;
    for (int branch = 0; branch < c.branches; ++branch) {
        auto input = c.attention ? ggml_scale(ctx, x, 1.f / 32.f) : x;
        auto mat = ggml_mul_mat(ctx, w, input);
        if (c.force_f32) ggml_prec_set_acc(mat, GGML_PREC_F32);
        if (c.keep_mat) { ggml_set_output(mat); retained.push_back(mat); }
        auto delta = c.attention ? ggml_scale(ctx, mat, 32.f) : mat;
        delta = c.bias == 0 ? ggml_scale_inplace(ctx, delta, c.strength) : ggml_scale_bias(ctx, delta, c.strength, c.bias);
        output = c.inplace ? ggml_add_inplace(ctx, output, delta) : ggml_add(ctx, output, delta);
        if (c.marked) ggml_set_name(output, "lora_output_sum");
    }
    ggml_set_output(output);
    auto graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, output);
    auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    assert(ggml_gallocr_alloc_graph(alloc, graph));
    std::vector<float> values(ggml_nelements(base));
    const float edge[] = {0.f, -0.f, 1.f, -1.f, 65504.f, -65504.f, 0x1p-126f, -0x1p-126f, 0x1p-149f};
    for (size_t i = 0; i < values.size(); ++i) values[i] = i % 23 == 0 ? edge[(i / 23) % 9] : float(int((i * 7919) % 4096) - 2048) / 257.f;
    ggml_backend_tensor_set(base, values.data(), 0, ggml_nbytes(base));
    std::vector<float> inputs(ggml_nelements(x));
    for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = float(int((i * 131) % 2048) - 1024) / 4093.f;
    ggml_backend_tensor_set(x, inputs.data(), 0, ggml_nbytes(x));
    std::vector<ggml_fp16_t> weights(ggml_nelements(w));
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = ggml_fp32_to_fp16(float(int((i * 73) % 1024) - 512) / 997.f);
    ggml_backend_tensor_set(w, weights.data(), 0, ggml_nbytes(w));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    std::vector<float> result(ggml_nelements(output));
    ggml_backend_tensor_get(output, result.data(), 0, ggml_nbytes(output));
    assert(fwrite(result.data(), sizeof(float), result.size(), file) == result.size());
    for (auto mat : retained) {
        ggml_backend_tensor_get(mat, result.data(), 0, ggml_nbytes(mat));
        assert(fwrite(result.data(), sizeof(float), ggml_nelements(mat), file) == size_t(ggml_nelements(mat)));
    }
    std::vector<float> unchanged(inputs.size());
    ggml_backend_tensor_get(x, unchanged.data(), 0, ggml_nbytes(x));
    assert(memcmp(unchanged.data(), inputs.data(), ggml_nbytes(x)) == 0);
    if (!c.inplace) {
        ggml_backend_tensor_get(base, result.data(), 0, ggml_nbytes(base));
        assert(memcmp(result.data(), values.data(), ggml_nbytes(base)) == 0);
    }
    printf("PASS %s output_floats=%lld retained=%zu\n", c.name, (long long)ggml_nelements(output), retained.size());
    fflush(stdout);
    ggml_gallocr_free(alloc); ggml_free(ctx);
}

int main(int argc, char ** argv) {
    assert(argc == 2);
    auto backend = ggml_backend_cuda_init(0); assert(backend);
    FILE * file = fopen(argv[1], "wb"); assert(file);
    const Case cases[] = {
        {"positive", 16, 640, 1025, 1, .73f},
        {"negative", 32, 1024, 513, 1, -.37f},
        {"zero", 16, 640, 1025, 1, 0.f},
        {"tiny", 16, 640, 1025, 1, 0x1p-126f},
        {"attention", 16, 4096, 1025, 1, .71f, true},
        {"multiple", 16, 640, 1025, 3, -.23f},
        {"out_of_place_bias", 16, 640, 1025, 1, 1.0003f, false, true, false, false, false, .0123f},
        {"unmarked_fallback", 16, 640, 1025, 1, .73f, false, false},
        {"retained_mat_fallback", 16, 640, 1025, 1, .73f, false, true, true},
        {"f32_compute_fallback", 16, 640, 1025, 1, .73f, false, true, false, true},
        {"short_sequence_fallback", 16, 640, 33, 1, .73f},
        {"large_rank_fallback", 512, 640, 513, 1, .73f},
        {"full_qwen_gate_up", 16, 24576, 5600, 1, .73f},
    };
    for (const auto & c : cases) run(backend, file, c);
    assert(fclose(file) == 0);
    ggml_backend_free(backend);
}
