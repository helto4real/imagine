// Compare the newly targeted stage-3 convolution, preserving scaling, bias and input.
#include "model/vae/wan_vae.hpp"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include <cassert>
#include <cstring>
#include <vector>

static std::vector<unsigned char> run(bool strips, bool residual, int width, int height,
                                      int channels, int outputs, bool target = true, bool patches_only = false) {
    setenv("SD_VAE_STRIP_CONV", "1", 1);
    setenv("SD_VAE_STRIP_STAGE3", strips ? "1" : "0", 1);
    setenv("SD_VAE_STRIP_SHORTCUT", "1", 1);
    setenv("SD_VAE_STRIP_RESIDUAL", "0", 1);
    setenv("SD_QWEN21_RETAIN_IDLE", "0", 1);
    setenv("SD_VAE_FUSED_UPSCALE_IM2COL", "0", 1);
    setenv("SD_VAE_REUSE_PAD_SCALE", "0", 1);
    auto backend = ggml_backend_cuda_init(0); assert(backend);
    auto ctx = ggml_init({64 * 1024 * 1024, nullptr, true}); assert(ctx);
    const std::string prefix = target ? (residual ?
        "first_stage_model.decoder.upsamples.3.upsamples.0.residual.2" :
        "first_stage_model.decoder.upsamples.3.upsamples.3.resample.1") : "test.other";
    GGMLRunnerContext rc; rc.backend = backend; rc.ggml_ctx = ctx;
    std::map<std::string, ggml_tensor*> weights;
    auto input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, height,
                                   residual ? 1 : channels, residual ? channels : 1);
    ggml_set_input(input); ggml_set_output(input);
    ggml_tensor* output;
    if (residual) {
        WAN::CausalConv3d conv(channels, outputs, {1,3,3}, {1,1,1}, {0,1,1});
        conv.set_scale(1.f / 128.f); conv.init(ctx, {}, prefix); conv.get_param_tensors(weights, prefix);
        output = conv.forward(&rc, input);
    } else {
        Conv2d conv(channels, outputs, {3,3}, {1,1}, {1,1});
        conv.set_scale(1.f / 128.f); conv.init(ctx, {}, prefix); conv.get_param_tensors(weights, prefix);
        output = conv.forward(&rc, ggml_upscale(ctx, input, 2, GGML_SCALE_MODE_NEAREST));
    }
    auto graph = ggml_new_graph_custom(ctx, 4096, false); ggml_build_forward_expand(graph, output);
    const bool gemms_only = std::getenv("VAE_TEST_GEMM_ONLY") != nullptr;
    std::vector<ggml_tensor*> gemms;
    for (int i=0;i<ggml_graph_n_nodes(graph);++i) {
        auto t=ggml_graph_node(graph,i);
        if(t->op==GGML_OP_MUL_MAT) {gemms.push_back(t);if(gemms_only)ggml_set_output(t);}
    }
    int writes = 0;
    for (int i=0; i<ggml_graph_n_nodes(graph); ++i) writes += ggml_graph_node(graph,i)->op == GGML_OP_SET;
    const int expected = (height * (residual ? 1 : 2) + 63) / 64;
    assert(writes == (strips && target ? expected : 0));
    if (patches_only) {
        std::vector<ggml_tensor*> patches;
        for (int i=0; i<ggml_graph_n_nodes(graph); ++i) {
            auto t=ggml_graph_node(graph,i); if(t->op==GGML_OP_IM2COL) patches.push_back(t);
        }
        ggml_tensor* joined=nullptr;
        int total_h=height*(residual?1:2),row=0;
        for(auto t:patches) {
            int rows=strips&&target?std::min(64,total_h-row):total_h;
            int crop=row && t->ne[2]>rows ? 1 : 0;
            auto part=ggml_cont(ctx,ggml_view_4d(ctx,t,t->ne[0],t->ne[1],rows,1,t->nb[1],t->nb[2],t->nb[3],crop*t->nb[2]));
            joined=joined?ggml_concat(ctx,joined,part,2):part;row+=rows;
        }
        assert(row==total_h);output=joined;
        graph=ggml_new_graph_custom(ctx,4096,false);ggml_build_forward_expand(graph,output);
    }
    auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    assert(ggml_gallocr_alloc_graph(alloc, graph));
    std::vector<float> values(ggml_nelements(input));
    const float edge[] = {0.f,-0.f,128.0625f,128.1875f,-128.0625f,0x1p-126f,-0x1p-126f,0x1p-149f};
    for (size_t i=0; i<values.size(); ++i)
        values[i] = i % 19 == 0 ? edge[(i/19)%8] : float(int((i*7919)%8192)-4096)/32.f;
    ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
    if (!patches_only) {
    auto weight = weights.at(prefix + ".weight");
    std::vector<ggml_fp16_t> w(ggml_nelements(weight));
    for (size_t i=0; i<w.size(); ++i) w[i] = ggml_fp32_to_fp16(float(int((i*131)%2003)-1001)/8192.f);
    ggml_backend_tensor_set(weight, w.data(), 0, ggml_nbytes(weight));
    auto bias = weights.at(prefix + ".bias"); std::vector<float> b(outputs);
    for (int i=0; i<outputs; ++i) b[i] = float(i%5)/10.f;
    ggml_backend_tensor_set(bias, b.data(), 0, ggml_nbytes(bias));
    }
    assert(ggml_backend_graph_compute(backend,graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    std::vector<unsigned char> bytes(ggml_nbytes(output));
    ggml_backend_tensor_get(output, bytes.data(), 0, bytes.size());
    if (gemms_only) {
        const int width_out=width*(residual?1:2), height_out=height*(residual?1:2);
        int row=0;
        for(auto t:gemms) {
            const int rows=strips&&target?std::min(64,height_out-row):height_out;
            const int crop=row && t->ne[0]/width_out>rows?1:0;
            std::vector<float> raw(ggml_nelements(t));ggml_backend_tensor_get(t,raw.data(),0,ggml_nbytes(t));
            for(int c=0;c<outputs;++c)std::memcpy(bytes.data()+4*(size_t(c)*width_out*height_out+row*width_out),raw.data()+c*t->ne[0]+crop*width_out,size_t(rows)*width_out*4);
            row+=rows;
        }
        assert(row==height_out);
    }
    std::vector<float> after(values.size()); ggml_backend_tensor_get(input, after.data(), 0, ggml_nbytes(input));
    assert(std::memcmp(values.data(), after.data(), ggml_nbytes(input)) == 0);
    std::printf("EXEC %s strips=%d shape=%dx%dx%d->%d workspace=%.6f MiB bytes=%zu\n",
                residual ? "residual" : "resample", strips, width,height,channels,outputs,
                ggml_gallocr_get_buffer_size(alloc,0)/1048576.,bytes.size()); std::fflush(stdout);
    ggml_gallocr_free(alloc); ggml_free(ctx); ggml_backend_free(backend); return bytes;
}
int main(int argc, char**) {
    int failures = 0;
    bool full = argc == 2;
    bool patches = argc == 3;
    for (bool residual : {true}) {
        std::vector<std::vector<int>> shapes = full ?
            std::vector<std::vector<int>>{{800,448,576,288}} :
            std::vector<std::vector<int>>{{1,1,1,1},{5,33,7,8},{65,129,576,288}};
        for (auto s : shapes) {
            auto a=run(false,residual,s[0],s[1],s[2],s[3],true,patches);
            auto b=run(true,residual,s[0],s[1],s[2],s[3],true,patches);
            if (a!=b) {
                size_t n=0; float maxerr=0;
                for(size_t i=0;i<a.size();i+=4) {float x,y;std::memcpy(&x,a.data()+i,4);std::memcpy(&y,b.data()+i,4);if(std::memcmp(&x,&y,4)){++n;maxerr=std::max(maxerr,std::abs(x-y));}}
                std::printf("FAIL byte identity: differing_floats=%zu max_abs_error=%g\n",n,maxerr);++failures;continue;
            }
            std::printf("PASS complete convolution byte identity\n"); std::fflush(stdout);
        }
        if(!full && !patches) assert(run(false,residual,5,33,7,8,false)==run(true,residual,5,33,7,8,false));
    }
    if(failures) return 1;
    std::puts("PASS all comparisons and retained-input integrity");
}
