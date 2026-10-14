// Exercise the actual CausalConv3d graph builder and CUDA kernels with synthetic data.
#include "model/vae/wan_vae.hpp"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include <cassert>
#include <cstring>
#include <vector>
static std::vector<unsigned char> run(bool reuse, int width, int height, int channels, int outputs,
                                     bool patches_only, bool target=true) {
    setenv("SD_VAE_REUSE_PAD_SCALE",reuse?"1":"0",1);
    auto backend=ggml_backend_cuda_init(0);assert(backend);
    auto ctx=ggml_init({64*1024*1024,nullptr,true});assert(ctx);
    const std::string prefix=target?"first_stage_model.decoder.upsamples.4.upsamples.0.residual.2":"test.other";
    WAN::CausalConv3d conv(channels,outputs,{1,3,3},{1,1,1},{0,1,1});
    conv.set_scale(1.f/128.f);conv.init(ctx,{},prefix);
    std::map<std::string,ggml_tensor*> weights;conv.get_param_tensors(weights,prefix);
    auto input=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,width,height,1,channels);ggml_set_input(input);ggml_set_output(input);
    GGMLRunnerContext rc;rc.backend=backend;rc.ggml_ctx=ctx;
    auto output=conv.forward(&rc,input);
    auto graph=ggml_new_graph_custom(ctx,1024,false);ggml_build_forward_expand(graph,output);
    ggml_tensor* patches=nullptr;
    for(int i=0;i<ggml_graph_n_nodes(graph);++i)if(ggml_graph_node(graph,i)->op==GGML_OP_IM2COL)patches=ggml_graph_node(graph,i);
    assert(patches && patches->src[1]->op==GGML_OP_SCALE);
    assert((patches->src[1]->view_src!=nullptr)==(reuse && target));
    if(reuse && target)assert(patches->src[1]->view_src->op==GGML_OP_PAD);
    if(patches_only){output=patches;graph=ggml_new_graph_custom(ctx,1024,false);ggml_build_forward_expand(graph,output);}
    auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));assert(ggml_gallocr_alloc_graph(alloc,graph));
    std::vector<float> values(ggml_nelements(input));
    const float edge[]={0.f,-0.f,128.0625f,128.1875f,-128.0625f,0x1p-126f,-0x1p-126f,0x1p-149f,0x1p-17f};
    for(size_t i=0;i<values.size();++i)values[i]=patches_only && i%3==0 ? edge[(i/3)%9]:float(int((i*7919)%8192)-4096)/32.f;
    ggml_backend_tensor_set(input,values.data(),0,ggml_nbytes(input));
    if(!patches_only){
        auto weight=weights.at(prefix+".weight");std::vector<ggml_fp16_t> w(ggml_nelements(weight));
        for(size_t i=0;i<w.size();++i)w[i]=ggml_fp32_to_fp16(float(int(i%17)-8)/256.f);
        ggml_backend_tensor_set(weight,w.data(),0,ggml_nbytes(weight));
        auto bias=weights.at(prefix+".bias");std::vector<float> b(outputs);for(int i=0;i<outputs;++i)b[i]=float(i%5)/10.f;
        ggml_backend_tensor_set(bias,b.data(),0,ggml_nbytes(bias));
    }
    assert(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);ggml_backend_synchronize(backend);
    std::vector<unsigned char> bytes(ggml_nbytes(output));ggml_backend_tensor_get(output,bytes.data(),0,bytes.size());
    // Reusing PAD must not modify the original input shared with residual paths.
    std::vector<float> after(values.size());ggml_backend_tensor_get(input,after.data(),0,ggml_nbytes(input));assert(std::memcmp(values.data(),after.data(),ggml_nbytes(input))==0);
    ggml_gallocr_free(alloc);ggml_free(ctx);ggml_backend_free(backend);return bytes;
}
int main(){
    for(bool patches:{true,false})for(auto shape:std::vector<std::vector<int>>{{1,1,1,1},{5,3,7,8},{65,33,288,144}}){
        auto a=run(false,shape[0],shape[1],shape[2],shape[3],patches);
        auto b=run(true,shape[0],shape[1],shape[2],shape[3],patches);assert(a==b);
        std::printf("PASS %s %dx%dx%d output_channels=%d bytes=%zu; input unchanged\n",patches?"im2col":"complete convolution",shape[0],shape[1],shape[2],shape[3],a.size());
    }
    assert(run(false,5,3,7,8,false,false)==run(true,5,3,7,8,false,false));
    std::puts("PASS unrelated-convolution fallback; all byte comparisons passed");
}
