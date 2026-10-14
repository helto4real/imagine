// Synthetic benchmark of the actual GGML ConvRot + I8 Linear graph.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
template<class T> static std::vector<T> read(const std::string& file, size_t count) {
    std::vector<T> data(count);
    std::ifstream input(file, std::ios::binary);
    if (!input.read(reinterpret_cast<char*>(data.data()), count * sizeof(T))) std::exit(2);
    return data;
}
static void save(ggml_tensor* tensor, const std::string& path, size_t offset, size_t bytes) {
    std::vector<char> data(bytes);
    ggml_backend_tensor_get(tensor, data.data(), offset, bytes);
    std::ofstream file(path, std::ios::binary);
    file.write(data.data(), bytes);
}
int main(int argc, char** argv) {
    if (argc != 8) return 2;
    const std::string input = argv[1], output = argv[2];
    const int m = std::atoi(argv[3]), k = std::atoi(argv[4]), n = std::atoi(argv[5]);
    const int batches = std::atoi(argv[6]), iterations = std::atoi(argv[7]);
    auto backend = ggml_backend_cuda_init(0);
    if (!backend) return 2;
    auto* ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
    auto* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    auto* w = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, k, n);
    auto* scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    auto* q = ggml_quantize_i8_convrot(ctx, x, 256);
    auto* y = ggml_mul_mat_i8_tensorwise(ctx, w, q, scale, nullptr, 256);
    auto* graph = ggml_new_graph(ctx);
    const bool quantize_only = std::getenv("QWEN_LINEAR_QUANT_ONLY") != nullptr;
    ggml_build_forward_expand(graph, quantize_only ? q : y);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 2;
    auto hx = read<float>(input + "/x.f32", size_t(m)*k);
    auto hw = read<int8_t>(input + "/w.i8", size_t(n)*k);
    auto hs = read<float>(input + "/scale.f32", n);
    ggml_backend_tensor_set(x, hx.data(), 0, hx.size()*sizeof(float));
    ggml_backend_tensor_set(w, hw.data(), 0, hw.size());
    ggml_backend_tensor_set(scale, hs.data(), 0, hs.size()*sizeof(float));
    auto launch = [&]() {
        if (ggml_backend_graph_compute_async(backend, graph) != GGML_STATUS_SUCCESS) std::exit(2);
    };
    for (int i = 0; i < 20; ++i) launch();
    ggml_backend_synchronize(backend);
    std::ofstream metadata(output + "/result.json");
    metadata << "{\"engine\":\"ggml\",\"batches\":[";
    for (int batch = 0; batch < batches; ++batch) {
        uint64_t start = now_ns();
        for (int i = 0; i < iterations; ++i) launch();
        ggml_backend_synchronize(backend);
        uint64_t end = now_ns();
        if (batch) metadata << ',';
        metadata << "{\"start_ns\":" << start << ",\"end_ns\":" << end
                 << ",\"iterations\":" << iterations << "}";
    }
    metadata << "]}\n";
    const size_t checked_rows = std::getenv("QWEN_LINEAR_VALIDATE_ALL") ? size_t(m) : size_t(std::min(m, 8));
    if (!quantize_only) save(y, output + "/y.f32", 0, checked_rows*n*sizeof(float));
    save(q, output + "/q.i8", 0, checked_rows*k);
    save(q, output + "/scales.f32", size_t((m+3)/4*4)*k, checked_rows*sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}
