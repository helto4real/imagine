// Compare the entire target block, retaining the original input and every output pixel.
#include "model/vae/wan_vae.hpp"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include <cassert>
#include <chrono>
#include <cstring>
#include <vector>

static std::vector<float> run(bool strip, int width, int height, bool target = true) {
    setenv("SD_VAE_STRIP_RESIDUAL", strip ? "1" : "0", 1);
    setenv("SD_VAE_STRIP_CONV", "1", 1);
    setenv("SD_VAE_FUSED_UPSCALE_IM2COL", "0", 1);
    setenv("SD_VAE_REUSE_PAD_SCALE", "0", 1);
    setenv("SD_QWEN21_RETAIN_IDLE", "0", 1);
    auto backend = ggml_backend_cuda_init(0); assert(backend);
    auto ctx = ggml_init({64 * 1024 * 1024, nullptr, true}); assert(ctx);
    GGMLRunnerContext rc; rc.backend = backend; rc.ggml_ctx = ctx;
    const std::string prefix = target ? "first_stage_model.decoder.upsamples.4.upsamples.0" : "test.other";
    String2TensorStorage shapes;
    for (const auto& pair : std::vector<std::pair<std::string, int>>{{"residual.2.weight",288*144},{"residual.6.weight",144*144},{"shortcut.weight",288*144}}) {
        int k = pair.first == "shortcut.weight" ? 1 : 3;
        int64_t ne[4] = {k,k,1,pair.second};
        shapes[prefix+"."+pair.first] = TensorStorage(prefix+"."+pair.first,GGML_TYPE_F16,ne,4,0);
    }
    WAN::ResidualBlock block(288,144);
    block.init(ctx,shapes,prefix);
    std::vector<GGMLBlock*> children; block.get_all_blocks(children);
    for(auto child:children)if(auto conv=dynamic_cast<WAN::CausalConv3d*>(child))conv->set_scale(1.f/128.f);
    std::map<std::string,ggml_tensor*> weights;block.get_param_tensors(weights,prefix);
    auto input=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,width,height,1,288);
    ggml_set_input(input);ggml_set_output(input);
    std::vector<ggml_tensor*> cache(2,nullptr);int idx=0;
    auto output=block.forward(&rc,input,1,cache,idx);assert(idx==2);
    assert(output->type==GGML_TYPE_F32 && output->ne[0]==width && output->ne[1]==height && output->ne[3]==144);
    auto graph=ggml_new_graph_custom(ctx,4096,false);ggml_build_forward_expand(graph,output);
    int matmuls=0;
    for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto t=ggml_graph_node(graph,i);assert(ggml_backend_supports_op(backend,t));matmuls+=t->op==GGML_OP_MUL_MAT;}
    int strips=(height+63)/64;
    assert(matmuls==(target?(strip?3*strips:strips+2):3));
    auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    assert(ggml_gallocr_alloc_graph(alloc,graph));
    std::vector<float> values(ggml_nelements(input));
    const float edge[]={0.f,-0.f,128.0625f,128.1875f,-128.0625f,0x1p-126f,-0x1p-126f,0x1p-149f};
    for(size_t i=0;i<values.size();++i)values[i]=i%19==0?edge[(i/19)%8]:float(int((i*7919)%8192)-4096)/32.f;
    ggml_backend_tensor_set(input,values.data(),0,ggml_nbytes(input));
    size_t ordinal=0;
    for(auto& pair:weights){
        auto t=pair.second;
        if(t->type==GGML_TYPE_F16){
            std::vector<ggml_fp16_t> w(ggml_nelements(t));
            for(size_t i=0;i<w.size();++i)w[i]=ggml_fp32_to_fp16(float(int((i*131+ordinal*17)%2003)-1001)/8192.f);
            ggml_backend_tensor_set(t,w.data(),0,ggml_nbytes(t));
        }else{
            std::vector<float> w(ggml_nelements(t));
            for(size_t i=0;i<w.size();++i)w[i]=pair.first.find("gamma")!=std::string::npos?1.f+float(int(i%7)-3)/32.f:float(int(i%5)-2)/10.f;
            ggml_backend_tensor_set(t,w.data(),0,ggml_nbytes(t));
        }
        ++ordinal;
    }
    auto start=std::chrono::steady_clock::now();
    assert(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);ggml_backend_synchronize(backend);
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    std::vector<float> result(ggml_nelements(output));ggml_backend_tensor_get(output,result.data(),0,ggml_nbytes(output));
    std::vector<float> after(values.size());ggml_backend_tensor_get(input,after.data(),0,ggml_nbytes(input));
    assert(std::memcmp(values.data(),after.data(),ggml_nbytes(input))==0);
    std::printf("EXEC block_strip=%d target=%d shape=%dx%dx288->144 nodes=%d matmuls=%d workspace=%.6f MiB compute=%.3f ms bytes=%zu input_unchanged=1\n",strip,target,width,height,ggml_graph_n_nodes(graph),matmuls,ggml_gallocr_get_buffer_size(alloc,0)/1048576.,ms,result.size()*sizeof(float));std::fflush(stdout);
    ggml_gallocr_free(alloc);ggml_free(ctx);ggml_backend_free(backend);return result;
}
int main(int argc,char**) {
    int failures=0;
    const auto shapes=argc>1?std::vector<std::pair<int,int>>{{1600,896}}:std::vector<std::pair<int,int>>{{1,1},{5,33},{65,129}};
    for(auto [w,h]:shapes){
        auto a=run(false,w,h),b=run(true,w,h);assert(a.size()==b.size());
        size_t differences=0,nonfinite=0;float maxerr=0;
        for(size_t i=0;i<a.size();++i){if(!std::isfinite(a[i])||!std::isfinite(b[i]))++nonfinite;if(std::memcmp(&a[i],&b[i],4)){++differences;maxerr=std::max(maxerr,std::abs(a[i]-b[i]));}}
        std::printf("COMPARE %dx%d differing_floats=%zu max_abs_error=%g nonfinite=%zu bytes=%zu\n",w,h,differences,maxerr,nonfinite,a.size()*sizeof(float));std::fflush(stdout);
        failures+=(differences!=0 || nonfinite!=0);
    }
    if(argc==1){auto a=run(false,5,33,false),b=run(true,5,33,false);assert(std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0);std::puts("PASS unrelated-block fallback");}
    return failures?1:0;
}
