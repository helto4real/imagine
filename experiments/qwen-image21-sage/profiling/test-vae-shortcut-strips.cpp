// Exercise the real DupUp3D mapping and direct addition, including preserved live inputs.
#include "model/vae/wan_vae.hpp"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include <cassert>
#include <cstring>
#include <vector>

static std::vector<float> run(bool strips,int width,int height,bool target=true,bool first=true) {
    setenv("SD_VAE_STRIP_SHORTCUT",strips?"1":"0",1);
    auto backend=ggml_backend_cuda_init(0);assert(backend);
    auto ctx=ggml_init({32*1024*1024,nullptr,true});assert(ctx);
    GGMLRunnerContext rc;rc.backend=backend;rc.ggml_ctx=ctx;
    WAN::DupUp3D op(1152,576,2,2);
    op.init(ctx,{},target?"first_stage_model.decoder.upsamples.2.avg_shortcut":"test.other");
    auto input=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,width,height,1,1152);
    auto source_main=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,width*2,height*2,first?1:2,576);
    ggml_set_input(input);ggml_set_output(input);ggml_set_input(source_main);ggml_set_output(source_main);
    auto main=ggml_scale(ctx,source_main,1.f);
    auto output=op.forward_add(&rc,input,main,first);
    auto graph=ggml_new_graph_custom(ctx,4096,false);ggml_build_forward_expand(graph,output);
    int acc=0;
    for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto t=ggml_graph_node(graph,i);assert(ggml_backend_supports_op(backend,t));acc+=t->op==GGML_OP_ACC;}
    assert(acc==(strips&&target&&first?(height+31)/32:0));
    if(acc)assert(output->view_src!=nullptr);
    auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));assert(ggml_gallocr_alloc_graph(alloc,graph));
    const float edge[]={0.f,-0.f,1.f,-1.f,1.00000011920928955078125f,-1.00000011920928955078125f,0x1p-126f,-0x1p-126f,0x1p-149f,-0x1p-149f,1e30f,-1e30f};
    std::vector<float> x(ggml_nelements(input)),m(ggml_nelements(source_main));
    for(size_t i=0;i<x.size();++i)x[i]=i%17==0?edge[(i/17)%12]:float(int((i*7919)%8192)-4096)/32.f;
    for(size_t i=0;i<m.size();++i)m[i]=i%13==0?edge[(i/13+3)%12]:float(int((i*1327)%8209)-4104)/128.f;
    ggml_backend_tensor_set(input,x.data(),0,ggml_nbytes(input));ggml_backend_tensor_set(source_main,m.data(),0,ggml_nbytes(source_main));
    assert(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);ggml_backend_synchronize(backend);
    std::vector<float> result(ggml_nelements(output));ggml_backend_tensor_get(output,result.data(),0,ggml_nbytes(output));
    std::vector<float> after(x.size());ggml_backend_tensor_get(input,after.data(),0,ggml_nbytes(input));assert(std::memcmp(x.data(),after.data(),ggml_nbytes(input))==0);
    after.resize(m.size());ggml_backend_tensor_get(source_main,after.data(),0,ggml_nbytes(source_main));assert(std::memcmp(m.data(),after.data(),ggml_nbytes(source_main))==0);
    std::printf("EXEC strips=%d target=%d first=%d shape=%dx%dx1152->576 acc=%d nodes=%d workspace=%.6f MiB bytes=%zu live_inputs_unchanged=1\n",strips,target,first,width,height,acc,ggml_graph_n_nodes(graph),ggml_gallocr_get_buffer_size(alloc,0)/1048576.,ggml_nbytes(output));std::fflush(stdout);
    ggml_gallocr_free(alloc);ggml_free(ctx);ggml_backend_free(backend);return result;
}
static bool compare(int w,int h,bool target=true,bool first=true){
    auto a=run(false,w,h,target,first),b=run(true,w,h,target,first);assert(a.size()==b.size());
    size_t diff=0,nonfinite=0;float error=0;
    for(size_t i=0;i<a.size();++i){nonfinite+=!std::isfinite(a[i])||!std::isfinite(b[i]);if(std::memcmp(&a[i],&b[i],4)){++diff;error=std::max(error,std::abs(a[i]-b[i]));}}
    std::printf("COMPARE %dx%d target=%d first=%d differing_floats=%zu max_abs_error=%g nonfinite=%zu bytes=%zu\n",w,h,target,first,diff,error,nonfinite,a.size()*sizeof(float));std::fflush(stdout);return diff==0&&nonfinite==0;
}
int main(int argc,char**){
    bool ok=true;
    if(argc>1)ok=compare(400,224);
    else{for(auto [w,h]:std::vector<std::pair<int,int>>{{1,1},{5,33},{65,129}})ok=compare(w,h)&&ok;ok=compare(5,33,false)&&ok;ok=compare(5,33,true,false)&&ok;}
    return ok?0:1;
}
