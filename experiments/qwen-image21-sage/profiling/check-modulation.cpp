// Local synthetic CUDA regression tests; no model or user data.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

bool ggml_cuda_can_fuse_norm_mul(const ggml_tensor *, const ggml_tensor *);
bool ggml_cuda_can_fuse_gated_residual(const ggml_tensor *, const ggml_tensor *);
static void require(bool ok, const char * why) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", why); std::exit(1); }
}
static ggml_tensor * dependency(ggml_context * ctx, ggml_tensor * x, ggml_tensor * other) {
    auto v = ggml_view_4d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->ne[3], x->nb[1], x->nb[2], x->nb[3], 0);
    v->src[1] = other;
    return v;
}
int main(int argc, char **) {
    auto backend = ggml_backend_cuda_init(0);
    require(backend, "CUDA backend");
    for (int test = 0; test < 10; ++test) {
        const int width = test == 7 ? 2048 : 4096;
        const int tokens = test == 0 ? 1 : argc > 1 ? 17 : 5600;
        const float eps = test == 3 ? 0.5f : 1e-6f;
        auto ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
        auto storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, tokens);
        auto residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, tokens);
        auto row = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
        auto x = ggml_scale(ctx, storage, 1.f);
        auto scale = ggml_scale_bias(ctx, row, 1.f, 1.f);
        auto gate = ggml_tanh(ctx, row);
        auto ref_norm = ggml_norm(ctx, x, eps);
        ggml_set_output(ref_norm);
        auto ref_mod = ggml_mul(ctx, ref_norm, scale);
        auto ref_mul = ggml_mul(ctx, x, gate);
        ggml_set_output(ref_mul);
        auto ref_add = ggml_add(ctx, residual, ref_mul);
        auto norm_input = dependency(ctx, x, scale);
        auto norm = ggml_norm(ctx, norm_input, eps);
        auto mod = ggml_mul(ctx, norm, scale);
        auto mod_live = dependency(ctx, mod, norm_input);
        mod_live->src[2] = scale;
        auto gate_input = dependency(ctx, x, gate);
        auto mul = ggml_mul(ctx, gate_input, gate);
        auto add = ggml_add(ctx, residual, mul);
        auto add_live = dependency(ctx, add, gate_input);
        add_live->src[2] = residual;
        add_live->src[3] = gate;
        if (test == 5) { ggml_set_output(norm); ggml_set_output(mul); }
        auto graph = ggml_new_graph(ctx);
        for (auto v : {ref_mod, ref_add, mod_live, add_live}) {
            ggml_set_output(v); ggml_build_forward_expand(graph, v);
        }
        if (test == 6) {
            ggml_build_forward_expand(graph, ggml_scale(ctx, norm, 2.f));
            ggml_build_forward_expand(graph, ggml_scale(ctx, mul, 2.f));
        }
        auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(ggml_gallocr_alloc_graph(allocator, graph), "allocation");
        std::mt19937 rng(100 + test);
        std::vector<float> input(width*tokens), res(input.size()), weights(width);
        for (auto & v : input) v = test == 1 ? 0.f : (int(rng()%20001)-10000) * (test == 2 ? 1e-8f : .001f);
        for (auto & v : res) v = (int(rng()%20001)-10000)*.001f;
        for (auto & v : weights) v = (int(rng()%8001)-4000)*.001f;
        if (test == 4) std::fill(input.begin(), input.end(), 1.f);
        if (test == 8) std::fill(weights.begin(), weights.end(), -1.f);
        if (test == 9) std::fill(weights.begin(), weights.end(), 0.f);
        ggml_backend_tensor_set(storage, input.data(), 0, input.size()*4);
        ggml_backend_tensor_set(residual, res.data(), 0, res.size()*4);
        ggml_backend_tensor_set(row, weights.data(), 0, weights.size()*4);
        require(ggml_cuda_can_fuse_norm_mul(norm, mod) == (width==4096), "norm eligibility");
        require(ggml_cuda_can_fuse_gated_residual(mul, add) == (width==4096), "gate eligibility");
        auto alias = *mod; alias.data = norm_input->data;
        require(!ggml_cuda_can_fuse_norm_mul(norm, &alias), "norm alias rejection");
        auto alias_add = *add; alias_add.data = gate_input->data;
        require(!ggml_cuda_can_fuse_gated_residual(mul, &alias_add), "gate alias rejection");
        auto bad_row = *scale; bad_row.type = GGML_TYPE_F16;
        auto bad_mod = *mod; bad_mod.src[1] = &bad_row;
        require(!ggml_cuda_can_fuse_norm_mul(norm, &bad_mod), "dtype rejection");
        require(ggml_backend_graph_compute(backend, graph)==GGML_STATUS_SUCCESS, "compute");
        std::vector<float> gate_values(width);
        ggml_backend_tensor_get(gate, gate_values.data(), 0, gate_values.size()*4);
        for (int kind = 0; kind < 2; ++kind) {
            std::vector<float> expected(input.size()), actual(input.size());
            ggml_backend_tensor_get(kind ? ref_add : ref_mod, expected.data(),0,expected.size()*4);
            ggml_backend_tensor_get(kind ? add : mod, actual.data(),0,actual.size()*4);
            size_t mismatches = 0;
            for (size_t i=0;i<actual.size();++i) {
                require(std::isfinite(actual[i]), "finite result");
                mismatches += std::memcmp(&actual[i],&expected[i],4)!=0;
            }
            for (size_t i=0;i<actual.size();i+=8191) {
                double expected_value;
                if (kind) expected_value=double(res[i])+double(input[i])*double(gate_values[i%width]);
                else {
                    size_t base=i/width*width; double mean=0, square=0;
                    for(int j=0;j<width;++j) {mean+=input[base+j];square+=double(input[base+j])*input[base+j];}
                    mean/=width;
                    expected_value=(input[i]-mean)/std::sqrt(square/width-mean*mean+eps)*(1.+weights[i%width]);
                }
                // The fusion consumes the existing CUDA tanh result; check its multiply/add independently.
                const double bound = kind ? 4e-6*std::max(1.,double(std::abs(res[i])+std::abs(input[i]))) : 4e-6*std::max(1.,std::abs(expected_value));
                if (std::abs(actual[i]-expected_value)>=bound) { std::fprintf(stderr,"formula case=%d kind=%d i=%zu actual=%.9g expected=%.12g reference=%.9g x=%.9g w=%.9g res=%.9g\n",test,kind,i,actual[i],expected_value,expected[i],input[i],weights[i%width],res[i]); std::exit(1); }
            }
            std::printf("case=%d kind=%s values=%zu mismatches=%zu formula=PASS\n",test,kind?"gate":"norm",actual.size(),mismatches);
            require(mismatches==0,"bitwise comparison");
        }
        ggml_gallocr_free(allocator); ggml_free(ctx);
    }
    ggml_backend_free(backend);
    std::puts("PASS expected_fused_calls=7 norm + 7 gate");
}
