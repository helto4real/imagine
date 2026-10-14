// Local synthetic byte-identity tests; no model tensors or private inputs.
#include "core/ggml_extend.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

struct Run {
    Run() = default;
    Run(const Run&) = delete;
    Run(Run&& other) noexcept : ctx(other.ctx), alloc(other.alloc), graph(other.graph), input(other.input), weight(other.weight), output(other.output), fused(other.fused) { other.ctx=nullptr; other.alloc=nullptr; }
    ggml_context* ctx;
    ggml_gallocr_t alloc;
    ggml_cgraph* graph;
    ggml_tensor* input;
    ggml_tensor* weight;
    ggml_tensor* output;
    int fused;
    ~Run() { ggml_gallocr_free(alloc); ggml_free(ctx); }
};
static Run make(ggml_backend_t backend, ggml_backend_t fusion_backend, bool enabled,
                int w, int h, int channels, int batch, bool convolution, bool circular = false) {
    setenv("SD_VAE_FUSED_UPSCALE_IM2COL", enabled ? "1" : "0", 1);
    Run r{};
    r.ctx = ggml_init({32 * 1024 * 1024, nullptr, true}); assert(r.ctx);
    r.input = ggml_new_tensor_4d(r.ctx, GGML_TYPE_F32, w, h, channels, batch);
    r.weight = ggml_new_tensor_4d(r.ctx, GGML_TYPE_F16, 3, 3, channels, 8);
    auto bias = ggml_new_tensor_1d(r.ctx, GGML_TYPE_F32, 8);
    auto up = ggml_upscale(r.ctx, r.input, 2, GGML_SCALE_MODE_NEAREST);
    auto out = ggml_ext_conv_2d(r.ctx, up, r.weight, bias, 1, 1, 1, 1, 1, 1, false, circular, false, 1.f/128.f, fusion_backend);
    r.graph = ggml_new_graph(r.ctx); ggml_build_forward_expand(r.graph, out);
    ggml_tensor* patch = nullptr;
    for (int i=0;i<ggml_graph_n_nodes(r.graph);++i) {
        auto n=ggml_graph_node(r.graph,i);
        if (n->op==GGML_OP_IM2COL) { patch=n; r.fused+=n->op_params[7]==2; }
    }
    assert(patch);
    r.output=convolution ? out : patch;
    r.graph=ggml_new_graph(r.ctx); ggml_build_forward_expand(r.graph,r.output);
    if (r.fused) {
        auto cpu=ggml_backend_cpu_init(); assert(cpu);
        assert(!ggml_backend_supports_op(cpu,patch)); ggml_backend_free(cpu);
        for (int i=0;i<ggml_graph_n_nodes(r.graph);++i) assert(ggml_graph_node(r.graph,i)->op!=GGML_OP_UPSCALE);
    }
    r.alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    assert(ggml_gallocr_alloc_graph(r.alloc,r.graph));
    std::vector<float> chunk(256*1024);
    const float edge[]={0.f,-0.f,128.f,-128.f,1.f,-1.f,0.000001f,-0.000001f,
        0x1p-126f,-0x1p-126f,0x1p-149f,-0x1p-149f,0x1p-17f,-0x1p-17f,
        128.0625f,128.1875f,-128.0625f,8384512.f,-8384512.f,2e7f,-2e7f,
        std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()};
    for (size_t start=0;start<size_t(ggml_nelements(r.input));start+=chunk.size()) {
        size_t count=std::min(chunk.size(),size_t(ggml_nelements(r.input))-start);
        for(size_t i=0;i<count;++i) {
            auto k=start+i;
            chunk[i]=convolution ? float(int((k*7919)%8192)-4096)/32.f :
                ((k%3)==0 ? edge[(k/3)%(sizeof(edge)/sizeof(*edge))] : float(int((k*7919)%8192)-4096)/32.f);
        }
        ggml_backend_tensor_set(r.input,chunk.data(),start*sizeof(float),count*sizeof(float));
    }
    if(convolution) {
        std::vector<ggml_fp16_t> values(ggml_nelements(r.weight));
        for(size_t i=0;i<values.size();++i) values[i]=ggml_fp32_to_fp16(float(int(i%17)-8)/256.f);
        ggml_backend_tensor_set(r.weight,values.data(),0,ggml_nbytes(r.weight));
        float biases[8]={0.f,1.f,-1.f,0.1f,-0.1f,2.f,-2.f,0.5f};
        ggml_backend_tensor_set(bias,biases,0,sizeof(biases));
    }
    assert(ggml_backend_graph_compute(backend,r.graph)==GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    return r;
}
static void compare(ggml_backend_t backend,int w,int h,int c,int batch,bool convolution) {
    std::vector<unsigned char> reference;
    size_t before=0,after=0;
    {
        auto r=make(backend,backend,false,w,h,c,batch,convolution);
        assert(r.fused==0);
        before=ggml_gallocr_get_buffer_size(r.alloc,0);
        reference.resize(ggml_nbytes(r.output));
        ggml_backend_tensor_get(r.output,reference.data(),0,reference.size());
    }
    {
        auto r=make(backend,backend,true,w,h,c,batch,convolution);
        assert(r.fused==1 && ggml_nbytes(r.output)==reference.size());
        after=ggml_gallocr_get_buffer_size(r.alloc,0);
        std::vector<unsigned char> chunk(4*1024*1024);
        for(size_t i=0;i<reference.size();i+=chunk.size()) {
            size_t n=std::min(chunk.size(),reference.size()-i);
            ggml_backend_tensor_get(r.output,chunk.data(),i,n);
            if(std::memcmp(chunk.data(),reference.data()+i,n)) {
                for(size_t j=0;j<n;++j) if(chunk[j]!=reference[i+j]) {
                    std::fprintf(stderr,"FAIL byte %zu: %u != %u\n",i+j,chunk[j],reference[i+j]);std::abort();
                }
            }
        }
    }
    std::printf("PASS %s input=%dx%dx%d batch=%d compared=%zu bytes graph_before=%.3f MiB graph_after=%.3f MiB\n",
        convolution?"convolution":"im2col",w,h,c,batch,reference.size(),before/1048576.,after/1048576.);
    std::fflush(stdout);
}
int main(int argc,char**argv) {
    auto backend=ggml_backend_cuda_init(0);assert(backend);
    compare(backend,1,1,1,1,false);
    compare(backend,5,3,7,2,false);
    compare(backend,33,17,32,1,false);
    compare(backend,5,3,7,2,true);
    compare(backend,33,17,32,1,true);
    {
        auto r=make(backend,nullptr,true,5,3,7,1,false);assert(r.fused==0);
        auto cpu=ggml_backend_cpu_init();assert(cpu);
        auto c=make(backend,cpu,true,5,3,7,1,false);assert(c.fused==0);
        ggml_backend_free(cpu);
        auto p=make(backend,backend,true,5,3,7,1,false,true);assert(p.fused==0);
    }
    if(argc>1) compare(backend,800,448,288,1,false);
    ggml_backend_free(backend);
    std::puts("PASS: all fusion byte-identity and fallback checks");
}
