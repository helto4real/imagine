// Exact BF16 output and graph-lifetime checks with synthetic inputs only.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
bool ggml_cuda_can_fuse_int8_down_residual(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *);
static void require(bool ok, const char * what) { if (!ok) { std::fprintf(stderr,"FAIL %s\n",what);std::exit(1); } }
static ggml_tensor * view_of(ggml_context * ctx, ggml_tensor * x, ggml_tensor * row) {
    auto v=ggml_view_4d(ctx,x,x->ne[0],x->ne[1],x->ne[2],x->ne[3],x->nb[1],x->nb[2],x->nb[3],0);
    v->src[1]=row;return v;
}
int main(int argc,char **) {
    auto backend=ggml_backend_cuda_init(0);require(backend,"backend");size_t checked=0;
    const int cases=argc>1?1:14;
    for(int test=0;test<cases;++test) {
        const int m=argc>1?4096:test==0?5600:test==2?4097:test==3?5657:test==4?5830:test==5?4095:4096;
        const int k=test==6?4096:12288,n=4096;
        auto ctx=ggml_init({32*1024*1024,nullptr,true});
        auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,m);
        auto w=ggml_new_tensor_2d(ctx,GGML_TYPE_I8,k,n);
        auto ws=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,n);
        auto rs=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,n,m);
        auto gs=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,n);
        auto residual=ggml_cast(ctx,rs,GGML_TYPE_BF16);
        auto gate=ggml_tanh(ctx,gs);
        auto q=ggml_quantize_i8_convrot(ctx,x,256);
        ggml_tensor * mat[2],*view[2],*mul[2],*out[2];
        for(int i=0;i<2;++i) {
            mat[i]=ggml_mul_mat_i8_tensorwise(ctx,w,q,ws,nullptr,256);mat[i]->src[4]=gate;
            view[i]=view_of(ctx,mat[i],gate);mul[i]=ggml_mul(ctx,view[i],gate);
            out[i]=ggml_add(ctx,residual,mul[i]);out[i]->src[2]=w;out[i]->src[3]=q;out[i]->src[4]=ws;
        }
        ggml_set_output(mat[0]); // Preserve the standalone reference GEMM.
        ggml_tensor * kept=nullptr,*ref_kept=nullptr,*extra=nullptr,*ref_extra=nullptr;
        if(test==7||test==8||test==9) {
            kept=test==7?mat[1]:test==8?view[1]:mul[1];ref_kept=test==7?mat[0]:test==8?view[0]:mul[0];
            ggml_set_output(kept);ggml_set_output(ref_kept);
        }
        auto graph=ggml_new_graph(ctx);
        for(auto o:out){auto live=view_of(ctx,o,residual);live->src[2]=gate;ggml_set_output(live);ggml_build_forward_expand(graph,live);}
        if(test==10||test==11) {
            extra=ggml_scale(ctx,test==10?mat[1]:view[1],2.f);ref_extra=ggml_scale(ctx,test==10?mat[0]:view[0],2.f);
            for(auto e:{extra,ref_extra}){ggml_set_output(e);ggml_build_forward_expand(graph,e);}
        }
        auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));require(ggml_gallocr_alloc_graph(alloc,graph),"allocate");
        std::vector<float> input(size_t(m)*k),res(size_t(m)*n),scales(n),g(n);std::vector<int8_t> weights(size_t(n)*k);
        unsigned state=1234+test;auto rnd=[&](){state=1664525u*state+1013904223u;return state;};
        for(auto &v:input)v=(int(rnd()%20001)-10000)*(test==13?1e-8f:.001f);
        for(auto &v:weights)v=int(rnd()%255)-127;
        for(auto &v:res)v=(int(rnd()%20001)-10000)*.001f;
        for(int i=0;i<n;++i){scales[i]=(int(rnd()%2001)-1000)*.000017f;g[i]=test==12?0.f:(int(rnd()%2001)-1000)*.003f;}
        ggml_backend_tensor_set(x,input.data(),0,input.size()*4);ggml_backend_tensor_set(w,weights.data(),0,weights.size());
        ggml_backend_tensor_set(ws,scales.data(),0,scales.size()*4);ggml_backend_tensor_set(rs,res.data(),0,res.size()*4);ggml_backend_tensor_set(gs,g.data(),0,g.size()*4);
        require(ggml_cuda_can_fuse_int8_down_residual(mat[1],view[1],mul[1],out[1])==(m>=4096&&k==12288),"eligibility");
        auto alias=*out[1];alias.data=residual->data;require(!ggml_cuda_can_fuse_int8_down_residual(mat[1],view[1],mul[1],&alias),"residual overlap guard");
        alias.data=q->data;require(!ggml_cuda_can_fuse_int8_down_residual(mat[1],view[1],mul[1],&alias),"quant overlap guard");
        auto bad=*view[1];bad.view_offs=4;require(!ggml_cuda_can_fuse_int8_down_residual(mat[1],&bad,mul[1],out[1]),"view offset guard");
        require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"compute");
        std::vector<unsigned char>a(size_t(m)*n*2),b(a.size());ggml_backend_tensor_get(out[0],a.data(),0,a.size());ggml_backend_tensor_get(out[1],b.data(),0,b.size());
        size_t differences=0;for(size_t i=0;i<a.size();++i)differences+=a[i]!=b[i];
        std::printf("case=%d rows=%d k=%d bytes=%zu differences=%zu\n",test,m,k,a.size(),differences);require(differences==0,"BF16 result bits");checked+=a.size();
        for(int i=0;i<2;++i){auto actual=i?extra:kept,reference=i?ref_extra:ref_kept;if(!actual)continue;
            std::vector<float>av(size_t(m)*n),bv(av.size());ggml_backend_tensor_get(actual,av.data(),0,av.size()*4);ggml_backend_tensor_get(reference,bv.data(),0,bv.size()*4);
            require(std::memcmp(av.data(),bv.data(),av.size()*4)==0,"retained/reused F32 bits");for(float v:av)require(std::isfinite(v),"finite retained output");}
        ggml_gallocr_free(alloc);ggml_free(ctx);
    }
    ggml_backend_free(backend);std::printf("PASS checked_bytes=%zu expected_fused_calls=%d\n",checked,argc>1?1:7);
}
