// Synthetic replay of Qwen21's prefix-reuse attention graph. No model data.
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

extern "C" cudaStream_t benchmark_stream(ggml_backend_t);
static void require(bool value) { if (!value) { std::fprintf(stderr,"benchmark check failed\n"); std::exit(2); } }
static uint64_t now() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void upload(ggml_tensor * t, const std::string & path) {
    std::vector<char> data(ggml_nbytes(t)); std::ifstream f(path,std::ios::binary);
    require(bool(f.read(data.data(),data.size()))); ggml_backend_tensor_set(t,data.data(),0,data.size());
}
static void save(ggml_tensor * t, const std::string & path) {
    std::vector<char> data(ggml_nbytes(t)); ggml_backend_tensor_get(t,data.data(),0,data.size());
    std::ofstream f(path,std::ios::binary); f.write(data.data(),data.size()); require(bool(f));
}
int main(int argc, char ** argv) {
    require(argc == 6);
    const std::string input=argv[1], output=argv[2];
    const int prefix=std::atoi(argv[3]), variant=std::atoi(argv[4]), iterations=std::atoi(argv[5]);
    const int n=5600,d=128,h=32,width=d*h;
    auto backend=ggml_backend_cuda_init(0); require(backend);
    auto ctx=ggml_init({32*1024*1024,nullptr,true});
    auto graph=ggml_new_graph(ctx);
    auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,n);
    std::vector<ggml_tensor *> weights,scales;
    for(int i=0;i<4;i++) { weights.push_back(ggml_new_tensor_2d(ctx,GGML_TYPE_I8,width,width)); scales.push_back(ggml_new_tensor_1d(ctx,GGML_TYPE_F32,width)); }
    auto qw=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,d), kw=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,d);
    auto pe=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,2,2,d/2,n);
    auto pk=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,d,prefix,h,1);
    auto pv=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,d,h,prefix,1);
    auto quant=ggml_quantize_i8_convrot(ctx,x,256);
    std::vector<ggml_tensor *> qkv;
    for(int i=0;i<3;i++) { auto t=ggml_mul_mat_i8_tensorwise(ctx,weights[i],quant,scales[i],nullptr,256); qkv.push_back(ggml_reshape_4d(ctx,t,d,h,n,1)); ggml_build_forward_expand(graph,qkv.back()); }
    std::vector<int> ends={ggml_graph_n_nodes(graph)};
    auto norm_rope=[&](ggml_tensor * t, ggml_tensor * w) {
        auto norm=ggml_mul_inplace(ctx,ggml_rms_norm(ctx,t,1e-6f),w);
        auto out=ggml_rope_apply(ctx,norm,pe,true);
        auto dep=ggml_view_4d(ctx,out,out->ne[0],out->ne[1],out->ne[2],out->ne[3],out->nb[1],out->nb[2],out->nb[3],0);
        dep->src[1]=t; dep->src[2]=w; return dep;
    };
    auto q=norm_rope(qkv[0],qw), k=norm_rope(qkv[1],kw);
    ggml_build_forward_expand(graph,q); ggml_build_forward_expand(graph,k); ends.push_back(ggml_graph_n_nodes(graph));
    k=ggml_concat(ctx,pk,k,1);
    auto v=ggml_concat(ctx,pv,qkv[2],2);
    ggml_build_forward_expand(graph,k); ggml_build_forward_expand(graph,v); ends.push_back(ggml_graph_n_nodes(graph));
    auto vp=ggml_permute(ctx,v,0,2,1,3);
    auto vh=ggml_cast(ctx,variant ? vp : ggml_cont(ctx,vp),GGML_TYPE_F16);
    ggml_build_forward_expand(graph,vh); ends.push_back(ggml_graph_n_nodes(graph));
    auto attention=ggml_sage_attn(ctx,q,k,vh,1.f/std::sqrt(float(d)),GGML_SAGE_ATTN_AUTO);
    ggml_build_forward_expand(graph,attention); ends.push_back(ggml_graph_n_nodes(graph));
    auto joined=ggml_reshape_3d(ctx,attention,width,n,1);
    auto oq=ggml_quantize_i8_convrot(ctx,joined,256);
    auto y=ggml_mul_mat_i8_tensorwise(ctx,weights[3],oq,scales[3],nullptr,256);
    ggml_build_forward_expand(graph,y); ends.push_back(ggml_graph_n_nodes(graph));
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend); require(buffer);
    upload(x,input+"/x.f32"); upload(qw,input+"/norm.f32"); upload(kw,input+"/norm.f32"); upload(pe,input+"/pe.f32");
    upload(pk,input+"/pk-"+std::to_string(prefix)+".f32"); upload(pv,input+"/pv-"+std::to_string(prefix)+".f32");
    for(int i=0;i<4;i++) { upload(weights[i],input+"/w"+std::to_string(i)+".i8"); upload(scales[i],input+"/scale.f32"); }
    auto launch=[&](ggml_cgraph * g) { require(ggml_backend_graph_compute_async(backend,g)==GGML_STATUS_SUCCESS); };
    for(int i=0;i<20;i++) launch(graph);
    ggml_backend_synchronize(backend);
    save(y,output+"/y.f32"); save(vh,output+"/v.f16");
    std::ofstream meta(output+"/result.json");
    meta<<"{\"engine\":\"native\",\"prefix\":"<<prefix<<",\"direct_v_cast\":"<<variant<<",\"batches\":[";
    const char * labels[]={"qkv","rms_rope","prefix_concat","v_layout_cast","attention","out_projection","whole"};
    auto stream=benchmark_stream(backend);
    cudaEvent_t start,stop; require(cudaEventCreate(&start)==cudaSuccess); require(cudaEventCreate(&stop)==cudaSuccess);
    bool first=true;
    for(int part=0;part<7;part++) {
        auto view=ggml_graph_view(graph,part==6?0:(part==0?0:ends[part-1]),part==6?ends.back():ends[part]);
        for(int warm=0;warm<5;warm++) launch(&view);
        ggml_backend_synchronize(backend);
        for(int batch=0;batch<5;batch++) {
            auto begin=now(); require(cudaEventRecord(start,stream)==cudaSuccess);
            for(int i=0;i<iterations;i++) launch(&view);
            require(cudaEventRecord(stop,stream)==cudaSuccess); require(cudaEventSynchronize(stop)==cudaSuccess);
            auto end=now(); float ms=0; require(cudaEventElapsedTime(&ms,start,stop)==cudaSuccess);
            if(!first) meta<<',';
            first=false;
            meta<<"{\"part\":\""<<labels[part]<<"\",\"start_ns\":"<<begin<<",\"end_ns\":"<<end<<",\"iterations\":"<<iterations<<",\"cuda_ms\":"<<ms/iterations<<"}";
        }
    }
    meta<<"],\"buffer_bytes\":"<<ggml_backend_buffer_get_size(buffer)<<"}\n";
    std::ofstream nodes(output+"/graph.tsv"); nodes<<"index\top\ttype\tbytes\tne\tnb\n";
    for(int i=0;i<ggml_graph_n_nodes(graph);i++) { auto t=ggml_graph_node(graph,i);nodes<<i<<'\t'<<ggml_op_name(t->op)<<'\t'<<ggml_type_name(t->type)<<'\t'<<ggml_nbytes(t)<<'\t';for(int j=0;j<4;j++)nodes<<t->ne[j]<<',';nodes<<'\t';for(int j=0;j<4;j++)nodes<<t->nb[j]<<',';nodes<<'\n'; }
    cudaEventDestroy(start); cudaEventDestroy(stop); ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);
}
