// Paired fusion, graph allocator lifetimes and conservative fallback checks.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

bool ggml_cuda_can_fuse_bf16_qk_rms_rope(ggml_tensor * const *);
static void require(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr,"FAIL %s\n",what); std::exit(1); }
}
static ggml_tensor * live(ggml_context * ctx, ggml_tensor * out, ggml_tensor * input, ggml_tensor * weight) {
    auto v=ggml_view_4d(ctx,out,out->ne[0],out->ne[1],out->ne[2],out->ne[3],out->nb[1],out->nb[2],out->nb[3],0);
    v->src[1]=input; v->src[2]=weight; return v;
}
static size_t compare(ggml_tensor * a, ggml_tensor * b) {
    const auto bytes=ggml_nbytes(a);
    require(bytes==ggml_nbytes(b),"size");
    std::vector<unsigned char> av(bytes),bv(bytes);
    ggml_backend_tensor_get(a,av.data(),0,bytes); ggml_backend_tensor_get(b,bv.data(),0,bytes);
    require(std::memcmp(av.data(),bv.data(),bytes)==0,"exact output");
    return bytes;
}
int main(int argc,char **) {
    auto backend=ggml_backend_cuda_init(0); require(backend,"backend");
    size_t checked=0;
    for(int test=0;test<(argc>1?1:12);++test) {
        int tokens=argc>1?17:test==0?5600:test==1?1:test==3?5657:test==4?5830:57;
        int heads=test==5?1:32,batch=test==2?2:1;
        auto ctx=ggml_init({32*1024*1024,nullptr,true}); auto graph=ggml_new_graph(ctx);
        auto qf=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,128,heads,tokens,batch);
        auto kf=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,128,heads,tokens,batch);
        auto q=ggml_cast(ctx,qf,GGML_TYPE_BF16), k=ggml_cast(ctx,kf,GGML_TYPE_BF16);
        auto qw=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,128), kw=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,128);
        auto pe=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,2,2,64,tokens);
        auto kp=test==6?ggml_dup_tensor(ctx,pe):pe;
        ggml_build_forward_expand(graph,q); ggml_build_forward_expand(graph,k);
        ggml_tensor * nodes[2][10],*ends[2][2],*extra[2]={};
        for(int version=0;version<2;++version) {
            for(int side=0;side<2;++side) {
                auto n=nodes[version]+side*5;
                n[0]=ggml_cast(ctx,side?k:q,GGML_TYPE_F32);
                n[1]=ggml_rms_norm(ctx,n[0],side?3.e-6f:1.e-6f);
                n[2]=ggml_mul_inplace(ctx,n[1],side?kw:qw);
                n[3]=ggml_rope_apply(ctx,n[2],side?kp:pe,true);
                n[4]=ggml_cast(ctx,n[3],GGML_TYPE_BF16);
                ends[version][side]=live(ctx,n[4],side?k:q,side?kw:qw);
                if(test==7) ggml_set_output(n[0]);
                if(test==8) ggml_set_output(n[1]);
                if(test==10) ggml_set_output(n[3]);
                if(test==9 && side==0) extra[version]=ggml_scale(ctx,n[2],2.f);
                // Reference chains are separated by a live view; only candidate chains are adjacent.
                ggml_build_forward_expand(graph,version?n[4]:ends[version][side]);
            }
            for(auto t:ends[version]) {ggml_set_output(t);ggml_build_forward_expand(graph,t);}
        }
        if(test==9) for(auto t:extra){ggml_set_output(t);ggml_build_forward_expand(graph,t);}
        auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(ggml_gallocr_alloc_graph(alloc,graph),"allocate");
        unsigned state=42;auto rnd=[&](){state=1664525u*state+1013904223u;return state;};
        for(auto t:{qf,kf,qw,kw}) {
            std::vector<float> values(ggml_nelements(t));
            for(auto &v:values)v=(int(rnd()%20001)-10000)/997.f;
            ggml_backend_tensor_set(t,values.data(),0,values.size()*4);
        }
        std::vector<float> positions(ggml_nelements(pe));
        for(size_t i=0;i<positions.size();i+=4){float a=(i+1)*.0037f;positions[i]=std::cos(a);positions[i+1]=-std::sin(a);positions[i+2]=std::sin(a);positions[i+3]=std::cos(a);}
        ggml_backend_tensor_set(pe,positions.data(),0,positions.size()*4);
        if(kp!=pe)ggml_backend_tensor_set(kp,positions.data(),0,positions.size()*4);
        if(test==11) {
            require(ggml_cuda_can_fuse_bf16_qk_rms_rope(nodes[1]),"eligible");
            auto out=nodes[1][9];auto original=out->data;
            for(auto alias:{q,k,qw,kw,pe,nodes[1][4]}) {
                out->data=alias->data;
                require(!ggml_cuda_can_fuse_bf16_qk_rms_rope(nodes[1]),"cross-overlap guard");
            }
            out->data=original;
        }
        require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"compute");
        for(int side=0;side<2;++side)checked+=compare(ends[0][side],ends[1][side]);
        if(test==7||test==8||test==10)for(int side=0;side<2;++side)compare(nodes[0][side*5+(test==7?0:test==8?1:3)],nodes[1][side*5+(test==7?0:test==8?1:3)]);
        if(test==9)compare(extra[0],extra[1]);
        std::printf("case=%d tokens=%d heads=%d batch=%d exact=true\n",test,tokens,heads,batch);std::fflush(stdout);
        ggml_gallocr_free(alloc); ggml_free(ctx);
    }
    ggml_backend_free(backend);std::printf("PASS checked_bf16_bytes=%zu expected_paired_calls=%d\n",checked,argc>1?1:7);
}
